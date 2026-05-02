# qengine

> 영어 README 가 메인입니다 → [README.md](README.md)
>
> 🚧 **Prefill 은 현재 llama.cpp 보다 2~4배 느림. 최적화 진행 중 — 목표: llama.cpp 의 1.5배.** Generation 은 이미 +30~50% 우세.

NVIDIA 채굴 카드 (CMP 100-210, V100) 처럼 **아무도 안 사가는 GPU** 에서 굴리려고 처음부터 새로 짠 Qwen3 hybrid (GDN + Attention) 추론 엔진.

> 한국 고등학생이 **vibe coding** 으로 만든 거. 코드는 대부분 Claude 가 작성, 방향 / 디버깅 / 아키텍처 결정은 사람이.

vLLM, llama.cpp MMQ, FlashAttention, bitsandbytes — 전부 cuBLAS Tensor Core 경로라 sm_70 + CMP 환경에선 1/64 속도로 굴러가거나 아예 안 돌아감. 그래서 직접 짰음.

## 뭐 하는 거냐

- **Qwen3.5 / Qwen3.6 dense hybrid 27B / 9B** GGUF Q8_0 서빙. MoE (Qwen3-Moe 등) **미지원**
- **Qwen3-VL 멀티모달** (mmproj — ViT + M-RoPE + spatial reshape). llama.cpp 도 mtmd 로 지원함, 우리 unique 강점 X
- **OpenAI 호환 HTTP API** (chat/completions, streaming, tool calls)
- **Continuous batching** N슬롯 동시 + per-slot prefix cache
- **Speculative decoding** — MTP K=1 (동작). DFlash + DDTree 코드 있지만 **현재 동작 안 함** (drafter mismatch)
- **3-bit KV cache (MTP_TQ)** — Walsh-Hadamard 회전 + Lloyd-Max 양자화. 27B 256K 컨텍스트 fp16 17GB → 3.5GB. llama.cpp [#21038](https://github.com/ggml-org/llama.cpp/pull/21038) 가 같은 방향 (rotation + 4-bit RTN) 머지됨
- 멀티 GPU layer-parallel split, P2P 없이 pinned-host activation bridge

## 왜 만들었냐

CMP 100-210, V100 같은 채굴 카드는 중고 시장에 헐값으로 풀려있고 **HBM2 16GB** 인데, NVIDIA 가 SW 로 일부러 죽여놨음:

- **Tensor Core throttle** — HMMA latency 64배 (8 → 512 cycle), cuBLAS WMMA 5 TFLOP 한계
- **PCIe Gen1 x1**, P2P 없음, NVLink 없음
- **CUPTI 차단** — torch.profiler 못 씀
- 다 펌웨어 차원이라 우회 불가

그래서:

- **DP4A (int8) 17 TFLOP**, **HFMA2 (fp16 SIMD) 24 TFLOP** 경로로 GEMM 우회 — 이쪽은 throttle 안 걸림
- **자작 Q8_0 GEMV 커널** — 같은 HW 에서 cuBLAS 4배
- `max_seq > 32K` 일 때 cuBLAS strict path 회피
- GPU 간 hidden state 는 pinned-host 로 bridge (P2P 없음)

결과: **27B 256K 컨텍스트 4× CMP 위에서 안정** — gen ~30 t/s, prefill ~130 t/s @ 10K.

## 성능 (4× CMP 100-210, Q8_0, batch 1)

| 모델 | 컨텍스트 | Prefill | Generation | 비고 |
|---|---|---:|---:|---|
| 27B (Qwen3.6) | 10K | 132 t/s | ~30 t/s | FA on, MTP_TQ=1 |
| 27B (Qwen3.6) | 256K | — | ~28 t/s | MTP_TQ=1 KV 3.5GB |
| 9B (Qwen3.5) | 32K | — | 47–58 t/s (단일 슬롯) | |
| 9B (Qwen3.5) | 32K | — | 32 t/s × 4 슬롯 = 128 t/s 합산 | continuous batching |

같은 HW 에서 vLLM 은 27B Q8_0 아예 못 띄움 (sm_70 + Tensor Core throttle).

## 빌드

```bash
git clone https://github.com/Haru-neo/qengine.git
cd qengine && mkdir build && cd build
cmake .. && make -j$(nproc)
```

## 실행

**27B 256K + 비전:**
```bash
MTP_TQ=1 ./build/qwen-engine \
  /path/to/Qwen3.6-27B-Q8_0.gguf \
  --serve 8000 --max-seq 262144 --slots 1 \
  --vision-mmproj /path/to/Qwen3.6-27B-mmproj.gguf
```

**9B 4-way batching:**
```bash
QWEN_SLOTS=4 ./build/qwen-engine \
  /path/to/Qwen3.5-9B-Q8_0.gguf \
  --serve 8001 --max-seq 32768
```

**특정 GPU 만:**
```bash
CUDA_VISIBLE_DEVICES=0,1 ./build/qwen-engine ... --serve 8001
```

자세한 옵션 / API / 아키텍처 → [README.md](README.md)

## 한계

- **MoE 미지원** — dense Qwen3 hybrid 만. Qwen3-Moe 안 됨
- **DFlash + DDTree spec 동작 X** — drafter (z-lab pretrained) 가 stock Qwen3.5 기준이라 Qwopus distill 와 분포 mismatch. accept ≈ 0%, chain degenerate. 코드는 fine-tune 후 사용 목적으로 넣어둠
- **Prefill 2~4배 느림** (vs llama.cpp). 최대 약점. 최적화 진행 중
- batched MTP / spec 미지원 (`slots > 1` 이면 plain greedy)
- Q8_0 만 정식 지원 (Q5_K_M / Q6_K 는 quality 떨어짐)
- sm_70 타겟 — sm_75 에서 돌긴 하지만 최적화 X. sm_80+ 은 vLLM/SGLang 쓰는 게 나음
- 단일 호스트 only, 멀티 노드 X
- Linux only

## 만든 사람

이 코드베이스는 [vibe coding](https://en.wikipedia.org/wiki/Vibe_coding) 으로 만들어졌음 — 코드는 대부분 Anthropic Claude 가 여러 세션에 걸쳐 작성. 방향 / 디버깅 / 아키텍처 결정 / 커널 검증 / HW 리버스 엔지니어링은 **HARU-Neo** ([@Haru-neo](https://github.com/Haru-neo)) — 한국 고등학생. 버그는 내 탓, 좋은 커널은 Claude.

## 라이선스

Apache 2.0 — [LICENSE](LICENSE)
