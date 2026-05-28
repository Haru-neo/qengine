# qengine

> 영어 README 가 메인입니다 → [README.md](README.md)
>
> **9B 단일 GPU prefill 이 ~4.6K tok 까지 llama.cpp 대비 1.5–3× 빨라졌습니다** (2026-05-02 default flip, 커밋 `ca30368`). **split-K FlashAttention 디폴트 ON** 이후 (커밋 `f2e52b8`) 9B 18K prefill 도 **1.22× llama.cpp**, 27B 3GPU 18K 는 **거의 평행 (0.99×)** 까지 올라왔습니다. Generation 은 여전히 +30–50% 우세합니다. 9B 듀얼 GPU long-ctx 는 아직 llama 의 layer-pipeline 에 밀립니다 (단일 GPU 가 더 빨라서 실사용 영향은 작음).

NVIDIA 채굴 카드 (CMP 100-210, V100) — sm_70, HBM2 16GB, PCIe Gen1 x1, P2P 없음 — 에서 굴리려고 처음부터 새로 짠 Qwen3.5 / Qwen3.6 hybrid (GDN + Attention) 추론 엔진입니다. fork 가 아니고, 모든 커널을 sm_70 제약 기준으로 직접 작성했습니다.

vLLM, llama.cpp MMQ, FlashAttention, bitsandbytes — 전부 cuBLAS Tensor Core 경로라 sm_70 + CMP 환경에서는 1/64 속도로 돌거나 아예 동작하지 않습니다. 그래서 직접 짰습니다.

## 무엇을 하나요

- **Qwen3.5 / Qwen3.6 dense hybrid 27B / 9B** GGUF Q8_0 서빙. MoE (Qwen3-Moe 등) **미지원**입니다.
- **v2-MTP 내장 GGUF 지원 (2026-05-27)** — `blk.N.nextn.*` 으로 패킹된 MTP nextn head 를 자동 감지해서 spec drafter 로 사용합니다. 외부 `mtp_head_*.bin` 불필요. accept rate 가 73 → 83 % 로 올라갔습니다.
- **Qwen3-VL 멀티모달** (mmproj — ViT + M-RoPE + spatial reshape). llama.cpp 도 mtmd 로 지원하므로 우리만의 강점은 아닙니다.
- **OpenAI 호환 HTTP API** (chat/completions, streaming, tool calls)
- **Qwen3 thinking 제어** — `/no_think` (와 `/think`) 디렉티브를 user **또는** system 메시지에 넣거나, vLLM 방식 `extra_body.chat_template_kwargs.enable_thinking` 도 지원합니다.
- **Continuous batching** N 슬롯 동시 + per-slot prefix cache
- **Speculative decoding** — MTP K=1 (동작합니다). DFlash + DDTree 코드도 있지만 **현재 동작하지 않습니다** (drafter mismatch).
- **3-bit KV cache (MTP_TQ)** — Walsh-Hadamard 회전 + Lloyd-Max 양자화. 27B 256K 컨텍스트가 fp16 기준 17GB 인데 3.5GB 로 줄어듭니다. llama.cpp [#21038](https://github.com/ggml-org/llama.cpp/pull/21038) 에 같은 방향 (rotation + 4-bit RTN) 이 머지됐습니다.
- **Fused gate+up GEMM** (`MLP_GATEUP_FUSED_KERNEL=1`, 기본 OFF) — gate/up 두 Q8_0 weight 가 SMEM 에서 같은 Q8_1 input tile 을 공유합니다. v2 모델에서 prefill +5–10 %.
- 멀티 GPU layer-parallel split, P2P 없이 pinned-host activation bridge 로 통신합니다.

## 왜 만들었나요

CMP 100-210, V100 같은 채굴 카드는 중고 시장에 헐값으로 풀려있고 **HBM2 16GB** 인데, NVIDIA 가 소프트웨어로 일부러 막아뒀습니다.

- **Tensor Core throttle** — HMMA latency 64 배 (8 → 512 cycle), cuBLAS WMMA 5 TFLOP 한계
- **PCIe Gen1 x1**, P2P 없음, NVLink 없음
- **CUPTI 차단** — torch.profiler 도 못 씁니다
- 전부 다이에 박힌 e-fuse + PMU bootrom 이중락이라 펌웨어가 아닙니다. 소프트웨어 unlock 경로는 없습니다 (전수 시도 후 확인).

그래서 이렇게 풀었습니다.

- **DP4A (int8) 17 TFLOP**, **HFMA2 (fp16 SIMD) 24 TFLOP** 경로로 GEMM 을 우회합니다 — 이쪽은 throttle 이 걸리지 않습니다.
- **자작 Q8_0 GEMV 커널** — 같은 HW 에서 cuBLAS 대비 4 배 빠릅니다.
- `max_seq > 32K` 일 때 cuBLAS strict path 를 회피합니다.
- GPU 간 hidden state 는 pinned-host 로 bridge 합니다 (P2P 가 없으니).

결과: **27B 256K 컨텍스트가 3× CMP 위에서 안정적으로 동작합니다** — gen 26 t/s @ short, prefill 188 t/s @ 4.6K, 139 t/s @ 18K.

## 성능 (CMP 100-210, Q8_0, batch 1, split-K FA 디폴트 ON)

bench_curl.sh real OpenAI 요청 (qengine), llama-bench `-p N` 동일 길이 (llama.cpp build 8462). FA on, layer split. **굵게** 가 더 빠른 쪽입니다. 9B 단일 GPU 는 2026-05-02, 27B 3GPU 와 9B 18K split-K 는 2026-05-03 측정.

### 9B Q8_0 — 단일 GPU

| Prompt | qengine PP t/s | llama.cpp PP t/s | qengine TG t/s | llama.cpp TG t/s |
|---:|---:|---:|---:|---:|
| 297 | **594** | 199 | 70.4 | — |
| 1.16K | **683** | 316 | — | — |
| 4.62K | **584** | 361 | — | — |
| 18.4K | **393** | 324 | 27.6 | — |
| tg64 | — | — | — | 46.6 |

→ ~4.6K 까지 1.56–2.99×, 18K 도 split-K 로 **1.22×** (이전 0.83× 였음). Generation +51%.

### 9B Q8_0 — dual GPU (layer split, split-K 이전 측정)

| Prompt | qengine PP t/s | llama.cpp PP t/s | qengine TG t/s | llama.cpp TG t/s |
|---:|---:|---:|---:|---:|
| 297 | **495** | 188 | 68.6 | — |
| 1.16K | **611** | 412 | — | — |
| 4.62K | 519 | **574** | — | — |
| 18.4K | 259 | **545** | 27.4 | — |

9B 듀얼 18K 는 llama.cpp 의 layer pipeline + activation transfer overlap 이 우세합니다. 다만 9B 는 단일 GPU 가 이미 더 빠르니 실사용 영향은 작고, 주로 27B 시나리오에서 멀티 GPU 가 쓰입니다. split-K 적용 후 재측정 대기.

### 27B Q8_0 — 3 GPU (2026-05-03)

| Prompt | qengine PP t/s | llama.cpp PP t/s | qengine TG t/s | llama.cpp TG t/s |
|---:|---:|---:|---:|---:|
| 297 | **185** | 74.2 | 26.3 | — |
| 1.16K | **200** | 127.8 | 23.1 | — |
| 4.62K | **188** | 146.0 | 16.1 | — |
| 18.4K | 139 | 140.0 | 7.7 | — |
| tg128 | — | — | — | 17.7 |

→ ≤4.6K 1.29–2.49×, 18K 0.99× (평행). Generation +48% (26.3 vs 17.7 t/s @ short).

같은 HW 에서 vLLM 은 27B Q8_0 을 아예 띄우지 못합니다 (sm_70 + Tensor Core throttle).

## 빌드

```bash
git clone https://github.com/Haru-neo/qengine.git
cd qengine && mkdir build && cd build
cmake .. && make -j$(nproc)
```

## 실행

**27B 256K + 비전 (권장 설정):** **3 GPU** 가 4 GPU 보다 prefill 12 % 빠릅니다 (transfer hop 이 한 번 적음). v2 Q8_0 28 GB 모델은 3×16 GB 에 들어갑니다 (분배 자동 조정).

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
MTP_TQ=1 MLP_GATEUP_FUSED=1 MLP_GATEUP_FUSED_KERNEL=1 \
MINF_SPARSE_ATTN=1 MINF_BUDGET=0.10 \
MINF_PROFILE_PATH=./profiles/27B_block_sparse.bin \
./build/qwen-engine \
  /path/to/Qwopus3.6-27B-v2-MTP-Q8_0.gguf \
  --serve 8000 --max-seq 262144 --slots 1 \
  --vision-mmproj /path/to/Qwopus3.6-27B-v2-mmproj.gguf
```

v1 모델은 `MLP_GATEUP_FUSED_KERNEL` 을 빼고 `mtp_head_<hidden>.bin` 을 `/home/paru/mtp_work/` 에 두세요.

**9B 4-way batching:**
```bash
QWEN_SLOTS=4 ./build/qwen-engine \
  /path/to/Qwopus3.5-9B-Q8_0.gguf \
  --serve 8001 --max-seq 32768
```

**Qwen3 thinking 끄기 (`/no_think`)**: user/system 메시지에 `/no_think` 한 줄 넣거나, 다음 중 하나 — vLLM 호환:

```python
client.chat.completions.create(
    model="qwen",
    messages=[{"role":"user","content":"What is 1+1?"}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
```

자세한 옵션 / API / 아키텍처는 [README.md](README.md) 를 참고해주세요.

## 한계

- **MoE 미지원** — dense Qwen3 hybrid 만 됩니다. Qwen3-Moe 는 동작하지 않습니다.
- **DFlash + DDTree spec 동작 안 함** — drafter (z-lab pretrained) 가 stock Qwen3.5 기준이라 Qwopus distill 와 분포 mismatch 가 발생합니다. accept ≈ 0%, chain 이 degenerate 합니다. 코드는 fine-tune 후 사용 목적으로 남겨뒀습니다.
- **9B 듀얼 GPU long-ctx prefill** 은 아직 llama.cpp 의 layer pipeline 이 빠릅니다 (~0.48×, split-K 적용 전 측정값). 9B 단일 GPU 가 듀얼보다 이미 빠르니 실사용 영향은 작고, 27B 3GPU 18K 는 이제 평행 (0.99×) 입니다. 9B 듀얼 격차를 닫으려면 pinned-host bridge 를 pipeline 으로 바꿔야 합니다 — PR 환영합니다.
- batched MTP / spec 미지원 (`slots > 1` 이면 plain greedy 로 동작합니다)
- Q8_0 만 정식 지원합니다 (Q5_K_M / Q6_K 는 quality 가 떨어집니다).
- sm_70 타겟입니다 — sm_75 에서도 돌지만 최적화는 안 됐습니다. sm_80+ 은 vLLM / SGLang 을 쓰시는 편이 낫습니다.
- 단일 호스트 전용입니다, 멀티 노드는 지원하지 않습니다.
- Linux 전용입니다.

## 라이선스

Apache 2.0 — [LICENSE](LICENSE)
