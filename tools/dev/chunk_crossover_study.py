"""Analyse a chunk_crossover_study.sh measurement: per-shape crossover and the cost of candidate dispatch rules.

What
    Reads <prefix>_chunk.txt and <prefix>_gemm.txt (prefill_bench output with the CUDA LittleBit op forced onto the
    chunked decode kernels / the prefill path) and prints, per layer shape: the time of both paths per batch size, the
    first batch size from which the prefill path stays faster, and for each rule in RULES the worst ratio of the rule's
    time to (a) the faster path and (b) the old rule (32 tokens, 48 from 8 Mi signs, before Task 13).
Why
    The chunk limits of engine/ggml/src/ggml-cuda/littlebit-dispatch.h are chosen from these numbers; the rules below are
    the candidates that were compared (docs/engine-changes.md, "Chunk limits").  GPU specific.
Usage
    python tools/dev/chunk_crossover_study.py <prefix> [--rows]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PAT = re.compile(r"LITTLEBIT\s+d_in=\s*(\d+) d_out=\s*(\d+) r=\s*(\d+) r2=\s*(\d+) n=\s*(\d+).*?:\s+([\d.]+) us/op")
Shape = tuple[int, int, int, int]   # d_in, d_out, r, r2 (padded)


def load(path: Path) -> dict[tuple[int, int, int, int, int], float]:
    """(d_in, d_out, r, r2, n) -> us per op."""
    out = {}
    for line in path.read_text(errors="replace").splitlines():
        m = PAT.search(line)
        if m:
            out[tuple(int(v) for v in m.groups()[:5])] = float(m.group(6))
    return out


def signs(s: Shape) -> int:
    return (s[2] + s[3]) * (s[0] + s[1])


def old_rule(s: Shape) -> int:
    """Before Task 13: 32 tokens, 48 for layers with at least 8 Mi signs."""
    return 48 if signs(s) >= 8 << 20 else 32


def bands_rule(s: Shape) -> int:
    """Task 13 first re-tune: sign-count bands only."""
    S = signs(s)
    return 96 if S >= 24 << 20 else 48 if S >= 12 << 20 else 24 if S >= 6 << 20 else 16


def fix2_rule(s: Shape) -> int:
    """Task 13 fix round 2: re-measured on 22 shapes (Qwen3-0.6B / 1.7B / 4B, Llama-3-8B, built-in), chosen so that no
    shape is more than 5 % slower than with the old rule at any measured batch size."""
    S = signs(s)
    return 48 if S >= 8 << 20 else 40 if S >= 6 << 20 else 24 if S >= 1 << 20 else 16


RULES = {"old (32/48)": old_rule, "bands (16/24/48/96)": bands_rule, "fix2 (16/24/40/48)": fix2_rule}


def report(prefix: str, rules: dict = RULES, rows: bool = False) -> None:
    ch, gm = load(Path(prefix + "_chunk.txt")), load(Path(prefix + "_gemm.txt"))
    shapes = sorted({k[:4] for k in ch if k in gm}, key=signs)
    worst = {name: (0.0, None) for name in rules}
    worst_old = {name: (0.0, None) for name in rules}
    for s in shapes:
        ns = sorted(k[4] for k in ch if k[:4] == s and k in gm)
        xo = next((n for i, n in enumerate(ns) if all(gm[s + (m,)] < ch[s + (m,)] for m in ns[i:])), None)
        line = f"{s} {signs(s) / 2**20:5.1f} Mi, d_out/d_in {s[1] / s[0]:4.2f}: crossover {xo}"
        for name, f in rules.items():
            lim = f(s)
            t = {n: (ch if n <= lim else gm)[s + (n,)] for n in ns}
            to = {n: (ch if n <= old_rule(s) else gm)[s + (n,)] for n in ns}
            r_best = max((t[n] / min(ch[s + (n,)], gm[s + (n,)]), n) for n in ns)
            r_old = max((t[n] / to[n], n) for n in ns)
            line += f" | {name}: limit {lim}, vs best {r_best[0]:.2f}@{r_best[1]}, vs old {r_old[0]:.2f}@{r_old[1]}"
            if r_best[0] > worst[name][0]:
                worst[name] = (r_best[0], (s, r_best[1]))
            if r_old[0] > worst_old[name][0]:
                worst_old[name] = (r_old[0], (s, r_old[1]))
        print(line)
        if rows:
            print("    chunk/prefill us: " + " ".join(f"{n}:{ch[s + (n,)]:.0f}/{gm[s + (n,)]:.0f}" for n in ns))
    for name in rules:
        print(f"{name}: worst vs best {worst[name][0]:.3f} at {worst[name][1]}; worst vs old rule "
              f"{worst_old[name][0]:.3f} at {worst_old[name][1]}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    report(sys.argv[1], rows="--rows" in sys.argv)
