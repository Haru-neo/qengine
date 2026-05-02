# qengine

> 영어 README 가 메인입니다 → [README.md](README.md)
>
> **9B 단일 GPU prefill 이 ~4.6K tok 까지 llama.cpp 대비 1.5–3× 빨라졌습니다** (2026-05-02 default flip 이후 — 커밋 `ca30368`). 18K 이상 long context 는 9B 단일 GPU 0.83×, 멀티 GPU 는 llama 의 layer-pipeline 이 더 잘 스케일해서 0.5× 수준입니다. Generation 은 여전히 +30–50% 우세합니다.

NVIDIA 채굴 카드 (CMP 100-210, V100) — sm_70, HBM2 16GB, PCIe Gen1 x1, P2P 없음 — 에서 굴리려고 처음부터 새로 짠 Qwen3 hybrid (GDN + Attention) 추론 엔진입니다. fork 가 아니고, 모든 커널을 sm_70 제약 기준으로 직접 작성했습니다.

vLLM, llama.cpp MMQ, FlashAttention, bitsandbytes — 전부 cuBLAS Tensor Core 경로라 sm_70 + CMP 환경에서는 1/64 속도로 돌거나 아예 동작하지 않습니다. 그래서 직접 짰습니다.

## 무엇을 하나요

- **Qwen3.5 / Qwen3.6 dense hybrid 27B / 9B** GGUF Q8_0 서빙. MoE (Qwen3-Moe 등) **미지원**입니다.
- **Qwen3-VL 멀티모달** (mmproj — ViT + M-RoPE + spatial reshape). llama.cpp 도 mtmd 로 지원하므로 우리만의 강점은 아닙니다.
- **OpenAI 호환 HTTP API** (chat/completions, streaming, tool calls)
- **Continuous batching** N 슬롯 동시 + per-slot prefix cache
- **Speculative decoding** — MTP K=1 (동작합니다). DFlash + DDTree 코드도 있지만 **현재 동작하지 않습니다** (drafter mismatch).
- **3-bit KV cache (MTP_TQ)** — Walsh-Hadamard 회전 + Lloyd-Max 양자화. 27B 256K 컨텍스트가 fp16 기준 17GB 인데 3.5GB 로 줄어듭니다. llama.cpp [#21038](https://github.com/ggml-org/llama.cpp/pull/21038) 에 같은 방향 (rotation + 4-bit RTN) 이 머지됐습니다.
- 멀티 GPU layer-parallel split, P2P 없이 pinned-host activation bridge 로 통신합니다.

## 왜 만들었나요

CMP 100-210, V100 같은 채굴 카드는 중고 시장에 헐값으로 풀려있고 **HBM2 16GB** 인데, NVIDIA 가 소프트웨어로 일부러 막아뒀습니다.

- **Tensor Core throttle** — HMMA latency 64 배 (8 → 512 cycle), cuBLAS WMMA 5 TFLOP 한계
- **PCIe Gen1 x1**, P2P 없음, NVLink 없음
- **CUPTI 차단** — torch.profiler 도 못 씁니다
- 전부 펌웨어 차원이라 우회가 불가능합니다.

그래서 이렇게 풀었습니다.

- **DP4A (int8) 17 TFLOP**, **HFMA2 (fp16 SIMD) 24 TFLOP** 경로로 GEMM 을 우회합니다 — 이쪽은 throttle 이 걸리지 않습니다.
- **자작 Q8_0 GEMV 커널** — 같은 HW 에서 cuBLAS 대비 4 배 빠릅니다.
- `max_seq > 32K` 일 때 cuBLAS strict path 를 회피합니다.
- GPU 간 hidden state 는 pinned-host 로 bridge 합니다 (P2P 가 없으니).

결과: **27B 256K 컨텍스트가 4× CMP 위에서 안정적으로 동작합니다** — gen ~30 t/s, prefill ~130 t/s @ 10K.

## 성능 (CMP 100-210, Q8_0, batch 1, 2026-05-02 측정)

bench_curl.sh real OpenAI 요청 (qengine), llama-bench `-p N` 동일 길이 (llama.cpp build 8462). FA on, layer split. **굵게** 가 더 빠른 쪽입니다.

### 9B Q8_0 — 단일 GPU

| Prompt | qengine PP t/s | llama.cpp PP t/s | qengine TG t/s | llama.cpp TG t/s |
|---:|---:|---:|---:|---:|
| 297 | **594** | 199 | 70.4 | — |
| 1.16K | **683** | 316 | — | — |
| 4.62K | **563** | 361 | — | — |
| 18.4K | 270 | **324** | 27.6 | — |
| tg64 | — | — | — | 46.6 |

→ ~4.6K 까지 1.56–2.99× 우세합니다. 18K 만 0.83×, Generation 은 +51% 입니다.

### 9B Q8_0 — dual GPU (layer split)

| Prompt | qengine PP t/s | llama.cpp PP t/s | qengine TG t/s | llama.cpp TG t/s |
|---:|---:|---:|---:|---:|
| 297 | **495** | 188 | 68.6 | — |
| 1.16K | **611** | 412 | — | — |
| 4.62K | 519 | **574** | — | — |
| 18.4K | 259 | **545** | 27.4 | — |

llama.cpp 는 layer pipeline + activation transfer overlap 으로 멀티 GPU 가 잘 스케일됩니다. 우리는 pinned-host bridge 가 sequential 이라 long-ctx 에서 추월당합니다. 다음 작업 항목입니다.

### 27B Q8_0 — 3 GPU

> ⚠️ default flip (2026-05-02) 이전 측정값입니다. 코드 패스는 9B 와 동일하니 비례한 개선이 예상됩니다. 재측정 대기 중입니다.

| Prompt | qengine PP t/s | llama.cpp PP t/s | qengine TG t/s | llama.cpp TG t/s |
|---:|---:|---:|---:|---:|
| 128 | 38 | **83** | 26.6 | 17.8 |
| 2048 | 37 | **137** | 23.5 | — |
| 8192 | 33 | **146** | 17 | — |
| 18K | 22 | — | 7.8 | — |

같은 HW 에서 vLLM 은 27B Q8_0 을 아예 띄우지 못합니다 (sm_70 + Tensor Core throttle).

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

자세한 옵션 / API / 아키텍처는 [README.md](README.md) 를 참고해주세요.

## 한계

- **MoE 미지원** — dense Qwen3 hybrid 만 됩니다. Qwen3-Moe 는 동작하지 않습니다.
- **DFlash + DDTree spec 동작 안 함** — drafter (z-lab pretrained) 가 stock Qwen3.5 기준이라 Qwopus distill 와 분포 mismatch 가 발생합니다. accept ≈ 0%, chain 이 degenerate 합니다. 코드는 fine-tune 후 사용 목적으로 남겨뒀습니다.
- **Long context (≥18K) prefill** 은 아직 llama.cpp 가 빠릅니다 — 9B 1GPU 0.83×, 9B 2GPU 0.48× 입니다. 멀티 GPU pipelining + K/V-sharing FA 가 다음 작업입니다 (`FA_NT=2` 를 시도했지만 long-ctx 에서 14% 더 느려서 infra 만 보존했습니다).
- batched MTP / spec 미지원 (`slots > 1` 이면 plain greedy 로 동작합니다)
- Q8_0 만 정식 지원합니다 (Q5_K_M / Q6_K 는 quality 가 떨어집니다).
- sm_70 타겟입니다 — sm_75 에서도 돌지만 최적화는 안 됐습니다. sm_80+ 은 vLLM / SGLang 을 쓰시는 편이 낫습니다.
- 단일 호스트 전용입니다, 멀티 노드는 지원하지 않습니다.
- Linux 전용입니다.

## 라이선스

Apache 2.0 — [LICENSE](LICENSE)
