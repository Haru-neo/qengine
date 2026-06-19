# Long-context prefill quality gate — design

Date: 2026-06-14. Author: paru + Claude. Status: approved (user said proceed autonomously).

## Why
We are optimizing **cold long-context (100K–256K) prefill** for the 27B coding model.
Measured diagnosis (2026-06-14, standalone kvgen profile, GPU3, Q8 KV, MINF on):

256K kvgen-gen = 193.9s profiled (~160s prod). Phase split + scaling (256K/100K, token ratio 2.67×):

| phase | 100K | 256K | ratio | order |
|---|---|---|---|---|
| attn **builder** | 5.4 | 45.0 | **8.35×** | **super-quadratic** (MInference block top-k, budget-proportional top_k) |
| attn fa+merge | 10.9 | 31.0 | 2.86× | linear (MINF working) |
| gdn | 20.0 | 55.2 | 2.77× | linear, HBM2 bandwidth floor (hard) |
| mlp | 4.6 | 12.8 | 2.78× | linear |
| **DFKI D2H** (was mislabeled "embed") | ~15 | ~37 | ~2.5× | PCIe-x1, 8GB int8 planes |

The "embed 40s" was the DFKI int8-plane D2H mis-charged via `streamWaitEvent`+deviceSync; with a
drain-before-embed it drops to 0.0 (confirmed). GEMV kernels are at the hardware floor (gemm_dev:
mode-9 78–93% DP4A peak at N=256; N=1 decode 791/829 GB/s) — do not touch. Multi-GPU kvgen is a
PCIe-x1 wash (Inc 2a/2b/2c: 100K 2-GPU pipeline ≈ 55s ≈ single-GPU 53.8s) — abandoned.

Cold 256K e2e ≈ 215s = kvgen-gen ~160s + inject H2D ~50s + tail ~5s.

## Two lossy levers (both degrade the predicted/injected KV = "memory recall" dimension)
1. **attn builder coarsen** (45s super-quad, pure compute, future-proofs to 512K): bigger pool block /
   top_k cap / strided scoring. Highest ROI.
2. **int4-DFKI** (halves the 8GB DFKI → kvgen D2H ~37s partial + 27B inject H2D ~50s, both ~2×):
   mechanical.

User quality bar: "coding intelligence + memory retention must hold" — needle recall + deterministic
code reasoning, NOT bit-identical. → **Build a quality gate FIRST** (user's call), then levers one at a
time (back-to-back A/B discipline).

## Gate harness (build first)

### Components (under `tools/gate/`)
1. **`build_gate_prompts.py`** — haystack + probe builder.
   - Filler: real code = concatenated qwen-engine repo sources (fixed order), tokenized with the engine
     tokenizer to hit exactly 12K / 100K / 256K.
   - Probes at injected-region depths {10,40,65,90}% + 1 control in the keep-last window:
     - *needle* (recall): unique fact line → recall question. Deterministic.
     - *code-trace* (reasoning): unique-named self-contained snippet with a deterministic computed
       output, e.g. `gate_probe_7741(){a=17;b=4;return a*b-9;}` → "returns?" → `59`. Reading+arith,
       not verbatim.
   - Per length emit TWO requests (needle block, code-trace block) + `answers.json`. Seeded, regenerable.
2. **`run_gate.py`** — sends prompts to chat server :8000, parses answers, scores needle recall% +
   code-trace pass% by depth + control (must be 100%). Also scrapes the server log for prefill e2e
   seconds + the `[SPEC-PREFILL]` line → one run yields **quality + speed**. Emits `scorecard_<cfg>.json`.
3. **`relerr_check.sh`** — wraps `ue_training/kvgen_train/check_kvgen_engine.py` (standalone kvgen DFKI
   vs python-exact predictor, median relerr, seconds) for the fast inner loop before a behavioral run.

### Reference configs (same binary, env-switched)
- `gold` = full prefill (`SPEC_PREFILL_AUTO=0`): exact ceiling. Run at 12K (probe calibration: gold must
  ~100%) and 100K (headroom). Skip 256K full prefill (~732s) — too costly; use spec-current baseline there.
- `spec-current` = current spec-prefill (int8 DFKI + current builder): the no-regression baseline.
- `variant` = after a lossy change.

### Verdict
A lossy variant PASSES iff: control=100% AND needle-recall ≥ spec-current (within noise) AND
code-trace ≥ spec-current, at 100K and 256K. Gold shows headroom. Report quality delta + speed delta together.

### Scale / robustness
~5 probes/request, 2 requests × {100K,256K} per config (~minutes, prefill-dominated). gold once.
Self-test at 12K (gold ~100% before trusting 256K). Deterministic seed. Control needle = server-alive
sanity. Same tokenizer as the engine.

## Then
Lever 1 = builder coarsen (gate-gated), Lever 2 = int4-DFKI (gate-gated). One variable at a time;
trust only same-session back-to-back A/B.
