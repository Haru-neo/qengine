# qengine

> 영어 README 가 메인입니다 → [README.md](README.md)
>
> **9B 단일 GPU prefill 이 ~4.6K tok 까지 llama.cpp 대비 1.5–3× 빨라졌습니다** (2026-05-02 default flip, 커밋 `ca30368`). **split-K FlashAttention 디폴트 ON** 이후 (커밋 `f2e52b8`) 9B 18K prefill 도 **1.22× llama.cpp** 까지 올라왔습니다. Generation 은 여전히 +30–50% 우세합니다. **멀티 GPU prefill 은 이제 파이프라인** 처리됩니다 (GPU별 compute / D2H / H2D 스트림 + 더블버퍼 hidden chunk): 9B 듀얼 18K 가 **1.32×**, 27B 3GPU 18K 가 **1.45×** 로, long-ctx 멀티 GPU 격차가 해소됐습니다.

NVIDIA 채굴 카드 (CMP 100-210, V100) — sm_70, HBM2 16GB, PCIe Gen1 x1, P2P 없음 — 에서 굴리려고 처음부터 새로 짠 Qwen3.5 / Qwen3.6 hybrid (GDN + Attention) 추론 엔진입니다. fork 가 아니고, 모든 커널을 sm_70 제약 기준으로 직접 작성했습니다.

vLLM, llama.cpp MMQ, FlashAttention, bitsandbytes — 전부 cuBLAS Tensor Core 경로라 sm_70 + CMP 환경에서는 1/64 속도로 돌거나 아예 동작하지 않습니다. 그래서 직접 짰습니다.

## 무엇을 하나요

- **Qwen3.5 / Qwen3.6 dense hybrid 27B / 9B** GGUF Q8_0 서빙. **Qwen3-MoE / Qwen3.x-A3B** GGUF 도 자동 감지(레이어별 `ffn_gate_inp` 라우터)돼서 grouped-expert FFN 으로 돌긴 하지만, 이 MoE 경로는 **실험적이고 실가중치로 검증 안 됐습니다** — 정식 지원 아님, 미검증으로 취급하세요.
- **v2-MTP 내장 GGUF 지원 (2026-05-27)** — `blk.N.nextn.*` 으로 패킹된 MTP nextn head 를 자동 감지해서 spec drafter 로 사용합니다. 외부 `mtp_head_*.bin` 불필요. accept rate 가 73 → 83 % 로 올라갔습니다.
- **Qwen3-VL 멀티모달** (mmproj — ViT + M-RoPE + spatial reshape). llama.cpp 도 mtmd 로 지원하므로 우리만의 강점은 아닙니다.
- **OpenAI 호환 HTTP API** (chat/completions, streaming, tool calls)
- **임베딩 + 리랭킹** 을 같은 binary 로 제공 (`--mode embed` / `--mode rerank`): Qwen3-Embedding-4B (last-token pool + L2 정규화) 와 Qwen3-Reranker-4B (`cls.output` 분류 헤드 기반 cross-encoder, instruction 인식). 채팅 서버가 `/v1/embeddings` · `/v1/rerank` 를 사이드카로 자동 리버스 프록시합니다.
- **Qwen3 thinking 제어** — `/no_think` (와 `/think`) 디렉티브를 user **또는** system 메시지에 넣거나, vLLM 방식 `extra_body.chat_template_kwargs.enable_thinking` 도 지원합니다.
- **Continuous batching** N 슬롯 동시 + per-slot prefix cache
- **Speculative decoding** — MTP K=1 (동작합니다). **DFlash** block-diffusion 도 엔진 쪽은 lossless 로 동작합니다 (block-diffusion draft + chain tree-verify; verify 가 `temp≤0` 면 greedy, `temp>0` 면 타깃 분포에서 샘플 → 출력 분포 일치). 단 stock drafter 는 distill 모델과 분포가 안 맞아 out-of-box accept length ≈ 1–3 (가속 거의 없음) — **분포-매칭 드래퍼**를 학습하면 accept ~4.5–5, 27B 3GPU 기준 mid-30~upper-40 t/s 가 목표치입니다.
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

- **DP4A (int8) 17 TFLOP** 경로로 LLM 의 GEMM 을 돌립니다 — CMP 에서 throttle 이 안 걸리는 경로입니다. (HFMA2 fp16 SIMD 24 TFLOP 도 throttle 이 안 걸리고, 이 엔진에선 비전 인코더가 씁니다.)
- **자작 Q8_0 GEMV 커널** — 같은 HW 에서 throttle 된 cuBLAS 경로보다 훨씬 빠릅니다.
- `max_seq > 32K` 일 때 VRAM 을 많이 먹는 strict per-score 버퍼(자작 커널, cuBLAS 아님)를 회피합니다.
- GPU 간 hidden state 는 pinned-host 로 bridge 합니다 (P2P 가 없으니).

결과: **27B 256K 컨텍스트가 3× CMP 위에서 안정적으로 동작합니다** — gen 27 t/s @ short, prefill 268 t/s @ 4.6K, 203 t/s @ 18K (prefill 파이프라인 ON).

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

→ ~4.6K 까지 1.62–2.99×, 18K 도 split-K 로 **1.22×** (이전 0.83× 였음). Generation +51%.

### 9B Q8_0 — dual GPU (layer split, prefill 파이프라인 ON)

| Prompt | qengine PP t/s | llama.cpp PP t/s | qengine TG t/s | llama.cpp TG t/s |
|---:|---:|---:|---:|---:|
| 297 | **594** | 188 | 68.4 | — |
| 1.16K | **968** | 412 | 66.7 | — |
| 4.62K | **982** | 574 | 50.8 | — |
| 18.4K | **720** | 545 | 26.4 | — |
| tg64 | — | — | — | 44.2 |

크로스 GPU prefill 파이프라이닝으로 9B 듀얼 18K 가 259 → **720 t/s (2.78×)**, 이제 모든 길이서 llama.cpp 우세입니다 (18K 도 1.32×). 샘플 토큰은 sequential 경로와 bit-equivalent (18K 까지 per-token greedy argmax 로 검증).

### 27B Q8_0 — 3 GPU (prefill 파이프라인 ON, 2026-05-03)

| Prompt | qengine PP t/s | llama.cpp PP t/s | qengine TG t/s | llama.cpp TG t/s |
|---:|---:|---:|---:|---:|
| 297 | **212** | 74.2 | 27.1 | — |
| 1.16K | **264** | 127.8 | 25.6 | — |
| 4.62K | **268** | 146.0 | 20.4 | — |
| 18.4K | **203** | 140.0 | 11.4 | — |
| tg128 | — | — | — | 17.7 |

→ **2.86× / 2.07× / 1.84× / 1.45×** (297 / 1.16K / 4.62K / 18K). 파이프라이닝이 18K 를 평행(139)에서 **1.45×** 로 끌어올렸습니다. Generation +53% (27.1 vs 17.7 t/s @ short).

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

**임베딩 + 리랭커 사이드카** (Qwen3 4B dense, 같은 binary, GPU 3 공유):

```bash
# 임베딩 사이드카 (8001 포트, GPU 3)
CUDA_VISIBLE_DEVICES=3 ./build/qwen-engine \
  /path/to/Qwen3-Embedding-4B-Q8_0.gguf --serve 8001 --mode embed

# 리랭커 사이드카 (8002 포트, GPU 3 공유)
# 스태거: 임베딩 사이드카가 로딩을 끝낼 때까지 기다린 뒤 띄우세요. 같은 GPU 에
# 두 CUDA 컨텍스트를 동시에 올리면 CMP 100-210 에서 두 번째가 간헐적으로 깨집니다.
CUDA_VISIBLE_DEVICES=3 ./build/qwen-engine \
  /path/to/Qwen3-Reranker-4B-Q8_0.gguf --serve 8002 --mode rerank

# 채팅 서버에 proxy 연결 (클라이언트는 8000번만 알면 됨)
./build/qwen-engine /path/to/27B.gguf --serve 8000 \
  --proxy-embed  127.0.0.1:8001 \
  --proxy-rerank 127.0.0.1:8002 \
  ...
```

`scripts/launch_all.sh` 한 줄로 3개 다 띄울 수 있어요 (임베딩→리랭커 스태거 포함).

`POST /v1/embeddings` (OpenAI 호환, 2560 dim) 와 `POST /v1/rerank` 둘 다 채팅 서버를 통해 자동 라우팅됩니다.

리랭커는 진짜 **cross-encoder** 입니다. 각 `query`+`document` 쌍을 공식 Qwen3-Reranker 템플릿(`<Instruct>/<Query>/<Document>`)으로 모델에 넣고 **`cls.output` 분류 헤드**를 읽어 `relevance_score = softmax([yes_logit, no_logit])[yes]` 를 냅니다. llama.cpp `--reranking` 과 같은 경로이고, LM head 의 raw yes/no 토큰 logit 을 읽는 것보다 훨씬 변별력이 좋습니다.

선택적 **instruction** 필드도 받습니다 (Qwen3-Reranker 는 instruction 인식 — 명확한 작업 설명이 특히 비영어 문서 판단을 날카롭게 합니다):

```bash
curl http://localhost:8000/v1/rerank -H "Content-Type: application/json" \
  -d '{"instruction":"질문에 답이 되는 문서를 찾아라",
       "query":"자비스 같은 AI 비서 어떻게 만들어?",
       "documents":["LLM 에 음성인식과 TTS 를 결합한다.",
                    "자비스는 아이언맨에 나오는 캐릭터다."]}'
```

> **4B 리랭커 GGUF 만들기:** 커뮤니티 Qwen3-Reranker-4B GGUF 대부분이 변환 과정에서 `cls.output.weight` 분류 헤드를 빠뜨려 이 경로가 깨집니다(모델이 그냥 "Okay" 를 뱉음). 공식 `Qwen/Qwen3-Reranker-4B` safetensors 를 최신 `convert_hf_to_gguf.py` 로 변환(리랭커 자동 감지 + cls head 합성)한 뒤, `--tensor-type "cls.output=f16"` 로 양자화하면 작은 헤드는 fp16 정밀도를 유지하고 나머지는 Q8_0 가 됩니다.

> **비ASCII 주의:** JSON 을 `ensure_ascii=True` 로 직렬화하는 클라이언트(Python `requests` 기본값)는 한글/CJK 를 `\uXXXX` 이스케이프로 보냅니다. 서버는 모든 문자열 추출 경로에서 이걸 UTF-8 로 디코드합니다 — 예전 빌드는 `query` 만 디코드해서 비ASCII **document** 가 쓰레기로 채점됐습니다. 수정 완료, 클라이언트 우회 불필요.

자세한 옵션 / API / 아키텍처는 [README.md](README.md) 를 참고해주세요.

## 한계

- **MoE 는 실험적, 미검증** — Qwen3-MoE / Qwen3.x-A3B GGUF 는 자동 감지(레이어별 `ffn_gate_inp` 라우터)돼서 grouped Q8_0 expert FFN 으로 모든 prefill/decode 경로에서 돌긴 하지만, **실가중치로 검증되지 않았습니다**. 정식 지원 아닌 미검증 경로로 취급하세요. 검증된 경로는 dense Qwen3 hybrid (GDN + Attention) 입니다.
- **DFlash 는 분포-매칭 드래퍼가 필요합니다** — 엔진 쪽 DFlash 경로는 lossless 로 동작하지만, stock drafter (z-lab pretrained) 가 Qwopus distill 와 분포 mismatch 라 out-of-box accept length 가 낮습니다 (가속 거의 없음). 네 모델 출력으로 드래퍼를 학습/파인튜닝하면 accept length ~4.5–5 가 나옵니다.
- **DFlash pipelined-fold 는 quality-equivalent 지만 bit-identical 은 아닙니다** — fold 경로(`DFLASH_FOLD=0` 으로 끔)가 bonus 토큰을 single-token 커널 대신 batched tree-verify 커널로 forward 하므로 near-tie argmax 토큰이 가끔 뒤집힙니다. 정확도는 안 변합니다 (compiled-C++ 통과율 동일). byte-exact 가 필요하면 `DFLASH_FOLD=0`.
- **실험적 speculative-prefill (kvgen 사이드카) 는 프로덕션용 아닙니다** — 0.8B 사이드카(`--mode kvgen`, `SPEC_PREFILL_AUTO`, 기본 OFF)가 긴 프롬프트 앞부분의 KV 를 예측해서 27B 가 뒷부분만 real-prefill 하게 하지만, 예측 KV 가 근사치라 long-ctx (예: 256K) 에서 needle recall 이 사실상 0 (gist-only) 입니다. 기본 런처에서 꺼져 있고 일반 chat 에 안 엮입니다. 긴 프롬프트는 full dense prefill 을 씁니다.
- batched MTP / spec 미지원 (`slots > 1` 이면 plain greedy 로 동작합니다)
- Q8_0 만 정식 지원합니다 (Q5_K_M / Q6_K 는 quality 가 떨어집니다).
- sm_70 타겟입니다 — sm_75 에서도 돌지만 최적화는 안 됐습니다. sm_80+ 은 vLLM / SGLang 을 쓰시는 편이 낫습니다.
- 단일 호스트 전용입니다, 멀티 노드는 지원하지 않습니다.
- Linux 전용입니다.

## 라이선스

Apache 2.0 — [LICENSE](LICENSE)
