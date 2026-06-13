# Multi-GPU pipeline-parallel kvgen — implementation plan

Goal: cut the 0.8B kvgen prefill predictor latency by pipelining its 24-layer
body across GPUs. Current single-GPU: 100K = 53.8s (after 2504b28/660879c/
c614fd1). Target with 3-stage pipeline: ~25-28s (~2x). **This is gated by one
real risk — the cross-GPU TAP-GATHER over PCIe-1.0-x1 (see §4). Measure it
first; if it dominates, abort.**

## Deployment context (from the user, 2026-06-13)
- Production: 27B on GPU0-2, reranker+embedding on GPU3. kvgen currently also
  on GPU3 (coexists, ~4.7GB).
- Spec-prefill flow is **sequential per request**: kvgen generates predicted KV,
  THEN the 27B injects+prefills+decodes. So during the kvgen phase the 27B's
  GPU0-2 COMPUTE is idle (weights resident, not computing). The 0.8B is ~0.85GB
  total Q8 → its layer shards fit in GPU0-2's spare VRAM (~3-5GB free/GPU).
- Therefore multi-GPU kvgen borrows GPU0-2's idle compute during the kvgen
  phase. **Latency win; under concurrent load it contends with a 27B prefill on
  the same GPUs (throughput tradeoff) — acceptable for the single-user flow.**
- Bonus: moving kvgen layers off GPU3 frees room there for rerank/embed.

## What the engine already gives us (verified, file:line)
- **Layer→GPU map**: `gpu_loader.h:36-103`, `PP_LAYER_BOUNDS` env, `layer_gpu[]`.
  `forward_*_chunk` read `int g = gpu->layer_gpu[layer]` and use GPU g's
  buffers/weights — they do NOT call cudaSetDevice (caller's job), they take an
  explicit `stream`. (`model.cuh` forward_gdn_chunk:3631, forward_attn_chunk:4600,
  forward_mlp_chunk:5877.)
- **GDN state + KV cache auto-placed per owning GPU**: `init_gdn_states`
  (`model.cuh:2789-2794`) and `init_kv_cache_caps` (`model.cuh:1910-1914`)
  cudaSetDevice(layer_gpu[layer]) before malloc. So loading the 0.8B multi-GPU
  places state/cache correctly with zero extra code.
- **Pipeline v3 reference** (the 27B prefill): `main.cu:2460-2600`. Per-GPU
  worker threads (each `cudaSetDevice(stage_gpu)` once at start), pinned host
  bridge `v2_host_xfer[NB]` + per-GPU `v2_gpu_hidden[g][NB]`, events
  `v2_sd[stage][buf]` (compute done) / `v2_dh[stage][buf]` (D2H done). **The
  no-P2P host fence is `cudaEventSynchronize(v2_dh[s-1][buf])` at main.cu:2540**
  BEFORE the H2D — cudaStreamWaitEvent alone gives stale reads on this CMP HW.
- **GDN sequential-state is pipeline-safe**: each stage processes chunks IN
  ORDER (chunk 0,1,2…), so a stage's layers' GDN rec_state updates in chunk
  order; the pipeline only overlaps DIFFERENT stages on different chunks. No
  reordering of a layer's state. (This is why layer-pipeline works and
  data-parallel-chunks does NOT — the latter needs the 18.9MB GDN state to hop
  GPUs every chunk = ~29s over PCIe-x1, a dead-end.)

## What's kvgen-specific and must be built (file:line of current single-GPU)
serve_kvgen single-GPU assumptions to lift (`main.cu`):
- `6211-6216` single-GPU enforcement check → remove / generalize.
- `6227,6235` `hbuf` allocated on GPU0, stays GPU0 across all 24 layers
  (`6356-6390` loop) → must hop GPUs at stage boundaries.
- `6230-6234` tap buffers `lvl[]` all on GPU0; `6385-6389` capture via D2D;
  `6391-6401` final-norm tap → taps are produced on whatever GPU the layer ran.
- `6402-6423` heads read `lvl[HD.taps[li*3+k]]` (3 taps/head) + run GEMMs, all
  on GPU0 → need all referenced taps resident on the heads' GPU.

## §4 THE RISK — cross-GPU tap-gather (measure before building the rest)
The 27B prefill pipeline only moves the **hidden state (1MB/chunk)** between
stages. kvgen additionally needs, for each chunk, the **22 captured tap levels**
(union of HD.taps) on the heads' GPU. Raw fp32: 22 × 256tok × 1024 × 4B ≈
22MB/chunk × ~385 chunks ≈ 8.5GB over PCIe-1.0-x1 (~250MB/s) ≈ **34s** — would
erase the pipeline gain.

Mitigations (apply in order, measure each):
1. **Quantize taps to Q8 before the cross-GPU copy** (the heads quantize to Q8
   for the GEMM anyway — `kvq.quantize_chunk`): ~5.5MB/chunk → ~8.5s total,
   overlappable. This is the key enabler.
2. **Overlap** the gather on a dedicated xfer stream under the next chunk's body
   compute (pipelined body ≈ 140ms/chunk; Q8 gather ≈ 22ms/chunk → hides).
3. **Heads on the last stage's GPU** so that stage's taps need no transfer; only
   earlier stages' taps gather forward (incrementally, as each chunk exits a
   stage).
GATE: prototype just the tap-gather (Q8, overlapped) and measure its exposed
cost at 100K. If it adds >~5s exposed, multi-GPU kvgen is not worth it on this
PCIe-x1 HW — STOP and bank the single-GPU levers instead (builder hoist / FA
NT-batch). Do NOT build the full pipeline before this gate passes.

## Build increments (each ends at a validation gate; commit per increment)
- **Inc 0 — tap-gather microbench** (the §4 gate). Standalone: time Q8 tap D2H→
  H2D for 22 levels × 256 tok, overlapped vs exposed, at the real PCIe-x1.
  GO/NO-GO. (~1-2h)
- **Inc 1 — multi-GPU body, SEQUENTIAL (correctness, expect SLOWER)**. Remove
  the single-GPU check; load 0.8B across N GPUs (PP_LAYER_BOUNDS for 24 layers,
  e.g. 8/8/8); in the chunk loop, before each forward_*_chunk do
  `cudaSetDevice(layer_gpu[layer])` and hop hbuf across stage boundaries via the
  pinned-host bridge + `cudaEventSynchronize` host fence (copy the v3 pattern).
  Capture taps on their producing GPU; gather (Q8) to the heads' GPU after the
  body; run heads there. VALIDATE: relerr-vs-python-dense BIT-IDENTICAL to
  single-GPU (cross-GPU is just data movement; fp16 hidden hop noise must stay
  ~0 in argmax terms — check the DFKI planes match the single-GPU file).
  Expect this to be ~the same or slower (no overlap yet) — correctness only.
  (~half day)
- **Inc 2 — chunk pipelining (the speedup)**. Spawn per-stage worker threads
  (v3 pattern), pipeline chunks k, k+1, k+2 across stages; tap-gather + heads on
  a trailing thread overlapped. VALIDATE relerr identical to Inc 1 + MEASURE
  100K wall. Target ~25-28s. Confirm all stages saturate (nvtop) and the
  tap-gather hides. (~half day)
- **Inc 3 — coexistence VRAM check**. Launch kvgen multi-GPU ALONGSIDE the
  resident 27B (GPU0-2) + rerank/embed (GPU3); confirm no OOM at 100K/250K and
  the 27B/rerank/embed still serve. Budget: 27B ~9.3GB + 0.8B shard ~0.5GB +
  0.8B KV(Q8)/activations per GPU0-2; GPU3 rerank+embed+0.8B shard. (~1-2h)

## Validation harness (reuse this session's)
- Cheap correctness: `kvgen_train/check_kvgen_engine.py` (12K relerr vs python
  dense; multi-GPU must stay bit-identical to single-GPU since it's pure data
  movement). Profiling: `KVGEN_PROFILE=1` phase split.
- End-to-end: 27B needle gate (12K code, 4 needles) — but note that needs the
  27B server up on the SAME GPUs; for the multi-GPU kvgen dev, validate against
  the single-GPU kvgen's DFKI output byte/argmax match instead, then one final
  end-to-end gate.
- Infra: kill kvgen via `pgrep -x qwen-engine | grep -v <other_pids>` (never
  pkill -x/-f — matches the bash or other servers). Wait-loops: log-based
  (listening/error/timeout), NOT pgrep (races during load). Clean /dev/shm
  between runs (each 100K DFKI is 3.27GB; the ENOSPC write-safety from 660879c
  now errors loudly instead of emitting zeros).

## Honest expectation
The §4 tap-gather over PCIe-1.0-x1 is the make-or-break. If Q8+overlap hides it,
~2x is real (53.8→~27s). If not, multi-GPU kvgen is a dead-end on this HW and
the single-GPU levers (builder scoring hoist ~3-4s with per-query top_k for
bit-exactness; FA NT-batch ~3s) are the path. Run Inc 0 FIRST.

## MEASURED (2026-06-13, Inc 1 built + run)
Inc 1 (multi-GPU body across 3 GPUs, sequential/blocking transfers) is DONE and
CORRECT: 12K relerr bit-identical to single-GPU (L0K 0.0288 / L7K 0.1145 / L15K
0.0874), 0 sparse errors; single-GPU path unchanged (100K 52.86s). Bug found +
fixed: the 16 head Q8 weights loaded on GPU(N-1) because init_* leaves the
device there — added cudaSetDevice(0) before load_kvgen_heads (heads run on
GPU0; without it quant_gemv faulted cross-GPU → sticky illegal access → all-zero
KV → relerr 1.0).
- **§4 GATE VERDICT: tap-gather is real and large.** 100K Inc 1 = 132.5s vs
  single-GPU 53.8s. Profile: p_cap (hbuf hops + tap-gather) = **62.1s** (fp32,
  blocking, ~18MB/chunk × 385). Body compute unchanged. Naive multi-GPU ~2.5x
  SLOWER.
- **BOTH are required, neither alone suffices:** (a) the body must be PIPELINED
  across GPUs (the ~40s serial body alone + any transfer > 53.8s); (b) the
  tap-gather must shrink. Q8-taps or overlap alone still leave the serial body.
- **Revised tap design (better than gather-all-to-GPU0):** capture each tap on
  its PRODUCING GPU and run each head on the GPU where its 3 taps live — the
  taps_for() 3-windows are narrow (map=[2,3,4,6,8,9,10,12,14,15,16,18,20,21,22,
  24]) so MOST heads' taps are within one GPU's layer range; only boundary heads
  gather. Replicate the small (~12.6MB) head weights per-GPU + per-GPU DFKI
  plane staging. Removes most of the 62s.
- Inc 2 = pipeline the body (reuse 27B pipeline v3 at main.cu:2460-2600) +
  distribute-heads tap design. Multi-hour; pipelining is non-negotiable for any
  speedup. Committed Inc 1 is the validated foundation.
