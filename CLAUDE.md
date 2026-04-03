# qwen-engine 프로젝트 컨텍스트

## 프로젝트
Qwen3.5-27B GGUF 모델용 커스텀 CUDA 추론 엔진. 4x CMP 100-210 (Volta sm70, 16GB HBM2, PCIe 1.0 x1, P2P 불가) 서버에서 실행.

## 현재 상태: 동작하지만 느림
- `<think>` logit=29.25 (llama.cpp 27.12와 유사) ✅
- 의미있는 reasoning 출력 생성 ✅
- 7.5 t/s (llama.cpp 22 t/s 대비 1/3)
- 80+ 토큰 coherent 출력 후 반복 시작 (fp16 누적 오차)

## 모델: Qwen3.5-27B (Gated DeltaNet + Attention 하이브리드)
- 64 layers: [GDN, GDN, GDN, Attn] × 16 반복
- hidden=5120, num_q_heads=24, num_kv_heads=4, head_dim=256
- GDN: num_k=16, num_v=48, k_dim=128, v_dim=128, conv_kernel=4
- intermediate_size=17408, vocab=248320
- RoPE: partial rotary dim=64 (head_dim의 25%)
- GGUF: Q5_K_M mixed (Q5_K + Q6_K), 19GB

## 파일 구조
- src/main.cu: 진입점 + 생성 루프
- src/gguf.h: GGUF 파서 (mmap, v3)
- src/gpu_loader.h: 멀티GPU 병렬 로딩
- src/model.cuh: QwenModel (MLP, GDN, Attn forward)
- src/quant_gemv.cuh: Q5_K/Q6_K/Q8 양자화 GEMV 커널 (dp4a)
- src/ops.cuh: RMSNorm, SiLU, residual, embedding dequant
- src/gdn_kernels.cuh: Conv1d, GDN recurrent, RMSNormGated
- src/attention.cuh: RoPE, head norm, attention score/softmax/value, gate, KV cache
- src/turboquant.cuh: TurboQuant KV cache (WHT + Lloyd-Max 3bit)
- run.py: Python 토크나이저 래퍼

## P0: GEMV 속도 최적화 (핵심 병목!)
현재:
- Q5_K: 283 GB/s (이전 519 달성했었음)
- Q6_K: 100 GB/s (이전 347 달성했었음)
- 하드웨어 memcpy 한계: 708 GB/s, llama.cpp GEMV: 450 GB/s

문제: quant_gemv.cuh의 Q6_K 커널이 switch문 + elem/sg/pos/quarter 매번 계산으로 느림.
핵심: sub가 loop에서 constant이므로 sg, quarter, ql offset, shift가 전부 constant.
switch를 밖으로 호이스트하고 arithmetic indexing:
- sg = sub/4, quarter = sub%4
- ql pointer = ql + sg*64 + (quarter&1)*32
- ql_shift = (quarter/2)*4 (0 or 4)
- qh_shift = quarter*2 (0,2,4,6)
- scale: sc[sub*2], sc[sub*2+1]

Q5_K도 byte 접근 → uint32_t 벡터화 로드 가능 (519 GB/s 달성한 적 있음).
주의: dp4a는 __dp4a(int, int, int) 오버로드 사용. 반드시 (int) 캐스트.

## P1: 출력 안정성 (긴 생성)
- ~80토큰 후 반복 시작
- state clamping ±1e6 추가됨
- softmax power-of-2 threads 수정됨
- gate_buf / attn_scores 버퍼 분리됨

## 빌드/실행
```bash
cd /home/paru/qwen-engine/build && cmake .. && make -j14
./qwen-engine /home/paru/models/gguf/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled.Q5_K_M.gguf 248045 846 198 12675 248046 198 248045 74455 198
```

## llama.cpp 레퍼런스 (22 t/s)
```bash
GGML_CUDA_FORCE_MMQ=1 /home/paru/ik_llama.cpp/build/bin/llama-cli -m /home/paru/models/gguf/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled.Q5_K_M.gguf -ngl 999 -fa on --threads 14 --temp 0 -p "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n" -n 50
```

## GDN 수식
g = ssm_a * softplus(alpha + dt_bias)   // ssm_a 음수
decay = exp(g)                           // 0 < decay < 1
beta = sigmoid(b_proj)
Q, K = L2_norm(conv_out), scale = 1/sqrt(k_dim)
S_new = decay * S + k * beta * (v - decay * S^T @ k)
o = S_new^T @ q * scale
GQA: k_head = v_head % num_k (modulo)

## 해결된 버그들
1. GPU간 P2P 불가 → host pinned memory 경유
2. RoPE multi-GPU + partial rotary dim=64
3. Conv1d weight layout: weight[d*kw+i]
4. dt_bias F32
5. Q6_K dequant scale: is = n/16 + l/16
6. GQA modulo mapping
7. gate_buf 분리
8. state clamping ±1e6
