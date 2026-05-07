#!/usr/bin/env python3
"""Offline sparse-attention profiler for the qwen-engine.

Drives the engine through one dense run + N sparse runs at varying FLOPs
budgets, dumps per-layer attention outputs at each pass via
``MINF_DUMP_ATTN_OUT``, then for each (layer, q_head) picks the smallest
budget whose L2 deviation from dense stays under ``--tol``. Saves the
result as a ``SparseProfile`` binary that the engine loads at startup
through ``MINF_PROFILE_PATH``.

Usage::

    python3 tools/profile_sparse_patterns.py \\
        --model /home/paru/models/gguf/Qwopus3.5-9B-v3.5-Q8_0.gguf \\
        --engine ./build/qwen-engine \\
        --max-seq 16384 \\
        --port 8050 \\
        --calibration tools/calibration_prompt.txt \\
        --budgets 0.05,0.10,0.20,0.40 \\
        --tol 0.05 \\
        --out profiles/9B_block_sparse.bin

The calibration prompt should be ≥ MINF_MIN_SEQ tokens long so the sparse
path actually engages on every chunk. The script picks the best per-head
``top_k`` from the budget sweep and writes the SparseProfile header
expected by ``src/sparse_attn/sparse_config.h``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np  # type: ignore

# Architecture facts hard-coded for the two Qwen3.5/3.6 sizes; if you train a
# new model architecture this list expands.
_ARCH_BY_HIDDEN = {
    5120: dict(num_layers=64, num_q_heads=24, head_dim=256),  # 27B
    4096: dict(num_layers=32, num_q_heads=16, head_dim=256),  # 9B (32 layers, not 64)
}

_MAGIC = 0x464E494D
_VERSION = 1

# Pattern enum mirrors src/sparse_attn/sparse_config.h: SparsePattern
PATTERN_DENSE          = 0
PATTERN_BLOCK_SPARSE   = 1
PATTERN_VERTICAL_SLASH = 2
PATTERN_A_SHAPE        = 3


def write_uniform_profile(path: Path, num_layers: int, num_q_heads: int,
                          pattern: int, *, btk: int = 0, vtk: int = 0,
                          stk: int = 0, win: int = 0, sink: int = 0,
                          flops_budget: float = 0.10) -> None:
    """Synthetic profile: every head gets the same pattern + params. Used by
    the profiler to ask the engine for a full sweep at one configuration."""
    with path.open("wb") as f:
        f.write(struct.pack("<IIIIfIII",
            _MAGIC, _VERSION, num_layers, num_q_heads, flops_budget, 0, 0, 0))
        rec = struct.pack("<B3xIIIIIIII",
            pattern, btk, vtk, stk, win, sink, 0, 0, 0)
        for _ in range(num_layers * num_q_heads):
            f.write(rec)


def wait_port(port: int, timeout: float) -> bool:
    """Block until the engine TCP socket is accepting connections."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                return True
        except OSError:
            time.sleep(2)
    return False


def run_engine(
    *,
    engine: str,
    model: str,
    port: int,
    max_seq: int,
    dump_dir: Path,
    sparse_env: dict[str, str] | None,
    log_path: Path,
) -> subprocess.Popen:
    env = os.environ.copy()
    env["MINF_DUMP_ATTN_OUT"] = str(dump_dir)
    env["QWEN_SLOTS"] = "1"
    if sparse_env:
        env.update(sparse_env)
    cmd = [engine, model, "--serve", str(port), "--max-seq", str(max_seq)]
    log_fh = open(log_path, "wb")
    proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
    return proc


def send_prompt(port: int, prompt: str, timeout: float = 600.0) -> dict:
    body = json.dumps({
        "model": "qwen",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def collect_run(
    *,
    engine: str,
    model: str,
    port: int,
    max_seq: int,
    sparse_env: dict[str, str] | None,
    prompt: str,
    label: str,
    work: Path,
) -> Path:
    """Boot engine with a fresh dump dir, send the calibration prompt once,
    then shut the engine down. Returns the dump directory path."""
    dump_dir = work / f"dump_{label}"
    if dump_dir.exists():
        shutil.rmtree(dump_dir)
    dump_dir.mkdir()
    log_path = work / f"engine_{label}.log"
    print(f"[{label}] launching engine (sparse_env={sparse_env}) → {log_path}", flush=True)
    proc = run_engine(
        engine=engine, model=model, port=port, max_seq=max_seq,
        dump_dir=dump_dir, sparse_env=sparse_env, log_path=log_path,
    )
    try:
        if not wait_port(port, timeout=240):
            raise RuntimeError(f"engine did not bind port {port} in time (see {log_path})")
        # The engine takes another ~10 s after binding to finish spec/MTP setup
        # — give it a head start before the calibration prompt to avoid an early
        # 503/queue-full.
        time.sleep(10)
        t0 = time.time()
        result = send_prompt(port, prompt)
        elapsed = time.time() - t0
        usage = result.get("usage", {})
        print(
            f"[{label}] prompt_tok={usage.get('prompt_tokens')} "
            f"comp={usage.get('completion_tokens')} elapsed={elapsed:.1f}s",
            flush=True,
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    return dump_dir


def discover_layers(dump_dir: Path) -> set[int]:
    """Just return the set of attn layers that have dumps in this dir.
    Loading the actual tensors is deferred to load_layer_dump so we don't
    sit on every candidate's full dataset at once (8 GB+ for 27B)."""
    layers: set[int] = set()
    for f in dump_dir.glob("L*_S*.bin"):
        layer_str = f.stem.split("_")[0]
        layers.add(int(layer_str[1:]))
    return layers


def load_layer_dump(dump_dir: Path, layer: int) -> np.ndarray:
    """Concatenate one layer's chunks into a single [T*num_q*HD] fp16 array.
    Caller is expected to release the array before loading the next layer
    or candidate."""
    pieces: list[tuple[int, np.ndarray]] = []
    for f in dump_dir.glob(f"L{layer}_S*.bin"):
        start = int(f.stem.split("_")[1][1:])
        pieces.append((start, np.fromfile(f, dtype=np.float16)))
    pieces.sort(key=lambda kv: kv[0])
    return np.concatenate([blob for _, blob in pieces], axis=0)


def per_head_l2(
    dense_layer: np.ndarray,
    sparse_layer: np.ndarray,
    num_q: int,
    head_dim: int,
) -> np.ndarray:
    """Per-q_head relative L2 between dense and sparse outputs for one layer.
    The dump tensor is flat fp16 of length T*num_q*head_dim; reshape and take
    Frobenius norms per head."""
    n = dense_layer.size
    if n != sparse_layer.size:
        # Length mismatch usually means one side ran more chunks (e.g. extra
        # warmup). Trim to the shorter one — both starts at offset 0.
        n = min(dense_layer.size, sparse_layer.size)
        dense_layer = dense_layer[:n]
        sparse_layer = sparse_layer[:n]
    t = n // (num_q * head_dim)
    if t * num_q * head_dim != n:
        raise ValueError(
            f"dump size {n} not divisible by num_q*head_dim={num_q*head_dim}"
        )
    d = dense_layer.reshape(t, num_q, head_dim).astype(np.float32)
    s = sparse_layer.reshape(t, num_q, head_dim).astype(np.float32)
    diff = d - s
    num = np.linalg.norm(diff.reshape(t * num_q, head_dim), axis=-1)
    den = np.linalg.norm(d.reshape(t * num_q, head_dim), axis=-1) + 1e-9
    rel = (num / den).reshape(t, num_q).mean(axis=0)  # [num_q]
    return rel


def write_profile(
    out_path: Path,
    num_layers: int,
    num_q_heads: int,
    flops_budget: float,
    head_records: list[dict],
) -> None:
    """SparseProfile binary, format documented in src/sparse_attn/sparse_config.h."""
    with out_path.open("wb") as f:
        # struct Header { uint32 magic, version, num_layers, num_q_heads;
        #                 float flops_budget; uint32 reserved[3]; }
        hdr = struct.pack(
            "<IIIIfIII",
            _MAGIC, _VERSION,
            num_layers, num_q_heads,
            flops_budget,
            0, 0, 0,
        )
        f.write(hdr)
        for r in head_records:
            # struct HeadRecord { uint8 pattern; uint8 pad[3];
            #                     uint32 btk, vtk, stk, win, sink;
            #                     uint32 reserved[3]; }
            f.write(struct.pack(
                "<B3xIIIIIIII",
                int(r["pattern"]),
                int(r.get("block_top_k", 0)),
                int(r.get("vertical_top_k", 0)),
                int(r.get("slash_top_k", 0)),
                int(r.get("window", 0)),
                int(r.get("sink", 0)),
                0, 0, 0,
            ))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--engine", required=True, help="path to qwen-engine binary")
    p.add_argument("--model", required=True, help="GGUF model path")
    p.add_argument("--port", type=int, default=8050)
    p.add_argument("--max-seq", type=int, default=16384)
    p.add_argument("--calibration", required=True,
                   help="path to a text file ≥ MINF_MIN_SEQ tokens")
    p.add_argument("--budgets", default="0.05,0.10,0.20,0.40",
                   help="comma-separated FLOPs budgets to sweep (block-sparse)")
    p.add_argument("--a-shape", default="64x512,64x1024,128x2048",
                   help="comma-separated sink x window pairs to sweep (A-shape)")
    p.add_argument("--tol", type=float, default=0.05,
                   help="max acceptable relative L2 vs dense (else stays DENSE)")
    p.add_argument("--out", required=True, help="output profile .bin path")
    p.add_argument("--hidden", type=int, choices=sorted(_ARCH_BY_HIDDEN),
                   default=4096, help="model hidden_size (4096=9B, 5120=27B)")
    p.add_argument("--keep-dumps", action="store_true",
                   help="leave the per-pass dump dirs on disk after profiling")
    args = p.parse_args()

    arch = _ARCH_BY_HIDDEN[args.hidden]
    num_layers   = arch["num_layers"]
    num_q_heads  = arch["num_q_heads"]
    head_dim     = arch["head_dim"]
    budgets      = [float(b) for b in args.budgets.split(",") if b.strip()]
    a_shape_pairs: list[tuple[int, int]] = []
    for pair in args.a_shape.split(","):
        pair = pair.strip()
        if not pair: continue
        s, w = pair.split("x")
        a_shape_pairs.append((int(s), int(w)))
    prompt       = Path(args.calibration).read_text()

    work = Path(tempfile.mkdtemp(prefix="qengine_profile_"))
    print(f"[profiler] work dir: {work}", flush=True)

    # Pass 0: dense baseline. We only remember the dump dir; tensors are
    # loaded one layer at a time during scoring so total RSS stays bounded.
    dense_dir = collect_run(
        engine=args.engine, model=args.model, port=args.port,
        max_seq=args.max_seq, sparse_env=None,
        prompt=prompt, label="dense", work=work,
    )

    # Block-sparse passes: uniform block-sparse at each budget. No profile
    # needed — the engine falls back to uniform block-sparse when PROFILE_PATH
    # is absent, which is exactly the candidate we want to score.
    sparse_dirs: dict[float, Path] = {}
    for b in budgets:
        env = {
            "MINF_SPARSE_ATTN": "1",
            "MINF_BUDGET": f"{b:.4f}",
            "MINF_MIN_SEQ": "4096",
        }
        sparse_dirs[b] = collect_run(
            engine=args.engine, model=args.model, port=args.port,
            max_seq=args.max_seq, sparse_env=env,
            prompt=prompt, label=f"sp{int(b*100):03d}", work=work,
        )

    # A-shape passes: write a synthetic all-A_SHAPE profile per (sink, window)
    # candidate; engine reads it via MINF_PROFILE_PATH and routes every layer
    # through the deterministic sink+window index. The block-sparse fallback
    # the budget controls doesn't fire for these passes.
    a_shape_dirs: dict[tuple[int, int], Path] = {}
    for sink, window in a_shape_pairs:
        prof_path = work / f"prof_as_{sink}_{window}.bin"
        write_uniform_profile(
            prof_path, num_layers, num_q_heads,
            pattern=PATTERN_A_SHAPE, win=window, sink=sink,
        )
        env = {
            "MINF_SPARSE_ATTN": "1",
            "MINF_BUDGET": "0.10",
            "MINF_MIN_SEQ": "4096",
            "MINF_PROFILE_PATH": str(prof_path),
        }
        a_shape_dirs[(sink, window)] = collect_run(
            engine=args.engine, model=args.model, port=args.port,
            max_seq=args.max_seq, sparse_env=env,
            prompt=prompt, label=f"as_S{sink}_W{window}", work=work,
        )

    # Score per (layer, head). Memory-bounded: we walk one layer at a time
    # and only ever hold dense + one candidate's tensor for that layer. For
    # 27B at 16 K context this caps RSS at ≈ 2 × per-layer dump size = a
    # few hundred MB instead of all-candidates × all-layers.
    head_records: list[dict] = []
    summary_rows: list[str] = ["layer,q_head,chosen,best_l2"]
    dense_layers = discover_layers(dense_dir)

    # (cost, label, pattern_dict, dump_dir) ordered cheapest-first.
    AVG_CTX_BLOCKS = 200  # rough proxy for 16 K context @ block_size=64
    candidates: list[tuple[float, str, dict, Path]] = []
    for b in budgets:
        candidates.append((
            float(b), f"BLOCK_SPARSE@{b:.3f}",
            {"pattern": PATTERN_BLOCK_SPARSE, "block_top_k": max(1, int(b * 100))},
            sparse_dirs[b],
        ))
    for sink, window in a_shape_pairs:
        approx_budget = (sink + window) / max(1, AVG_CTX_BLOCKS * 64)
        candidates.append((
            approx_budget, f"A_SHAPE@s{sink}w{window}",
            {"pattern": PATTERN_A_SHAPE, "window": window, "sink": sink},
            a_shape_dirs[(sink, window)],
        ))
    candidates.sort(key=lambda c: c[0])

    # Pre-compute every candidate's per-(layer, q_head) L2 so we can decide
    # all heads in a layer in one shot, then drop the layer's tensors before
    # moving on.
    for layer in range(num_layers):
        if layer not in dense_layers:
            for q in range(num_q_heads):
                head_records.append({"pattern": PATTERN_DENSE})
                summary_rows.append(f"{layer},{q},DENSE(no_dump),nan")
            continue

        d_layer = load_layer_dump(dense_dir, layer)
        # l2_table[ci, q] = relative L2 of candidate ci at this layer and head.
        l2_table = np.full((len(candidates), num_q_heads), np.inf, dtype=np.float64)
        for ci, (_, _, _, dump_dir) in enumerate(candidates):
            try:
                s_layer = load_layer_dump(dump_dir, layer)
            except ValueError:
                continue
            l2 = per_head_l2(d_layer, s_layer, num_q_heads, head_dim)
            l2_table[ci] = l2
            del s_layer
        del d_layer

        for q in range(num_q_heads):
            chosen_label = None
            chosen_record = None
            chosen_l2 = None
            best_overall_l2 = float(l2_table[:, q].min())
            best_overall_idx = int(np.argmin(l2_table[:, q]))
            best_overall_label = candidates[best_overall_idx][1] if np.isfinite(best_overall_l2) else None
            for ci, (_, label, rec, _) in enumerate(candidates):
                l2 = float(l2_table[ci, q])
                if l2 <= args.tol:
                    chosen_label = label
                    chosen_record = rec
                    chosen_l2 = l2
                    break
            if chosen_label is None:
                head_records.append({"pattern": PATTERN_DENSE})
                summary_rows.append(
                    f"{layer},{q},DENSE(best_was {best_overall_label or 'none'}),"
                    f"{best_overall_l2 if np.isfinite(best_overall_l2) else 'nan'}"
                )
            else:
                head_records.append(chosen_record)
                summary_rows.append(f"{layer},{q},{chosen_label},{chosen_l2:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_profile(
        out_path,
        num_layers=num_layers,
        num_q_heads=num_q_heads,
        flops_budget=min(budgets),
        head_records=head_records,
    )
    summary_path = out_path.with_suffix(".csv")
    summary_path.write_text("\n".join(summary_rows) + "\n")
    print(f"[profiler] wrote {out_path} ({len(head_records)} head records)", flush=True)
    print(f"[profiler] per-head choices: {summary_path}", flush=True)

    if not args.keep_dumps:
        shutil.rmtree(work, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
