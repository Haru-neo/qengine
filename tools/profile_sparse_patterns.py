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
    4096: dict(num_layers=64, num_q_heads=16, head_dim=256),  # 9B
}

_MAGIC = 0x464E494D
_VERSION = 1


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


def load_dumps(dump_dir: Path) -> dict[int, np.ndarray]:
    """Concatenate per-(layer, start_pos) dump files into one [T, num_q, HD]
    fp16 tensor per layer. Files are named L{layer}_S{start_pos}.bin holding
    n_tokens × num_q × HD halves."""
    grouped: dict[int, list[tuple[int, np.ndarray]]] = {}
    for f in sorted(dump_dir.glob("L*_S*.bin")):
        # L{layer}_S{start_pos}.bin
        stem = f.stem  # 'L12_S256'
        layer_str, start_str = stem.split("_")
        layer = int(layer_str[1:])
        start = int(start_str[1:])
        raw = np.fromfile(f, dtype=np.float16)
        grouped.setdefault(layer, []).append((start, raw))
    out: dict[int, np.ndarray] = {}
    for layer, items in grouped.items():
        items.sort(key=lambda kv: kv[0])
        out[layer] = np.concatenate([blob for _, blob in items], axis=0)
    return out


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
        hdr = struct.pack(
            "<5IfI3I",
            _MAGIC, _VERSION,
            num_layers, num_q_heads,
            0,  # padding before float (kept for natural layout in the C struct)
            flops_budget,
            0, 0, 0,
        )
        # Re-pack to match the C layout exactly.
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
                   help="comma-separated FLOPs budgets to sweep")
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
    prompt       = Path(args.calibration).read_text()

    work = Path(tempfile.mkdtemp(prefix="qengine_profile_"))
    print(f"[profiler] work dir: {work}", flush=True)

    # Pass 0: dense baseline.
    dense_dir = collect_run(
        engine=args.engine, model=args.model, port=args.port,
        max_seq=args.max_seq, sparse_env=None,
        prompt=prompt, label="dense", work=work,
    )
    dense = load_dumps(dense_dir)

    # Passes 1..N: sparse at each candidate budget.
    sparse_runs: dict[float, dict[int, np.ndarray]] = {}
    for b in budgets:
        env = {
            "MINF_SPARSE_ATTN": "1",
            "MINF_BUDGET": f"{b:.4f}",
            "MINF_MIN_SEQ": "4096",
        }
        d = collect_run(
            engine=args.engine, model=args.model, port=args.port,
            max_seq=args.max_seq, sparse_env=env,
            prompt=prompt, label=f"sp{int(b*100):03d}", work=work,
        )
        sparse_runs[b] = load_dumps(d)

    # Score per (layer, head). For each head we want the *smallest* budget
    # whose L2 ≤ tol; that maximises sparsity while preserving quality. If no
    # budget meets the bar, leave the head DENSE.
    head_records: list[dict] = []
    summary_rows: list[str] = ["layer,q_head,chosen_budget,best_l2"]
    sparse_layers = set()
    for d in sparse_runs.values():
        sparse_layers.update(d)
    common_layers = set(dense) & sparse_layers
    for layer in range(num_layers):
        for q in range(num_q_heads):
            chosen_budget = None
            best_l2_at_chosen = None
            best_overall_l2 = None
            best_overall_b = None
            if layer in common_layers:
                for b in sorted(budgets):
                    sparse = sparse_runs[b].get(layer)
                    if sparse is None:
                        continue
                    l2 = per_head_l2(dense[layer], sparse, num_q_heads, head_dim)[q]
                    if best_overall_l2 is None or l2 < best_overall_l2:
                        best_overall_l2 = float(l2)
                        best_overall_b = b
                    if l2 <= args.tol and chosen_budget is None:
                        chosen_budget = b
                        best_l2_at_chosen = float(l2)
                        break
            if chosen_budget is None:
                head_records.append({
                    "pattern": 0,  # DENSE
                    "block_top_k": 0,
                })
                summary_rows.append(
                    f"{layer},{q},DENSE,{best_overall_l2 if best_overall_l2 is not None else 'nan'}"
                )
            else:
                # Approximate top_k from budget. The runtime cap of 64 is
                # mirrored here so the engine and profiler stay in sync.
                # n_blocks scales with active context; we encode budget rather
                # than top_k so the engine can resolve it per call.
                top_k_marker = max(1, int(chosen_budget * 100))  # scaled %
                head_records.append({
                    "pattern": 1,  # BLOCK_SPARSE
                    "block_top_k": top_k_marker,
                })
                summary_rows.append(
                    f"{layer},{q},BLOCK_SPARSE@{chosen_budget:.3f},{best_l2_at_chosen:.4f}"
                )

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
