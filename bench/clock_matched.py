"""Prompt-processing speed of two GGUF files compared at matched GPU clocks (Task 13b).

What
    Short llama-bench runs (pp512 only, 3 timed repetitions each) of every file of a list, in interleaved rounds, each
    run started like the gate harness (bench/run_bench.py: GPU warm-up load until llama-bench's own warm-up pass).
    For every run the SM clock is sampled every 50 ms; the clock at the first timed repetition and during it are
    stored with the repetition's speed.  The report compares the FIRST timed repetition of each run (the GPU is at its
    boost clock there and has not yet hit its power cap) and lists the clocks, so that ratios are read at equal clocks.
Why
    In the gate harness's 10 repetitions a file that draws more power (the dense F16 model on its tensor cores) is
    throttled by the laptop's ~45 W cap within the run (SM clock down to 705-825 MHz), a lighter one less, so the median
    ratio mixes speed with power behaviour.  This tool does not change any GPU setting (no clock locking).
Usage
    python -m bench.clock_matched --rounds 6 --out bench/results/phase1_13b_clock.json FILE1.gguf FILE2.gguf ...
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from bench.run_bench import ROOT, _bench_cmd, _bench_with_load, _GpuSampler


def run_once(gguf: Path, warmup_s: float) -> dict:
    cmd = _bench_cmd(gguf.resolve(), "cuda", 99, ["-p", "512", "-n", "0", "-r", "3", "-o", "json"])
    with _GpuSampler(period_ms=50) as smp:
        r, info = _bench_with_load(cmd, warmup_s, True)
        t_end = time.perf_counter()
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2000:])
    rows = json.loads(r.stdout[r.stdout.index("["):])
    samples = [float(v) for v in rows[0]["samples_ts"]]
    t1 = info["run1_t"]
    # the first timed repetition takes 512 / speed seconds from its start
    rep1_end = t1 + 512.0 / samples[0] if t1 is not None else None
    clocks = [c for (_, c), t in zip(smp.samples, smp.times) if t1 is not None and t1 <= t <= rep1_end]
    return {"file": str(gguf), "pp512_samples": samples, "clock_at_start_mhz": info["clock_at_start_mhz"],
            "rep1_clock_mhz": [min(clocks), statistics.median(clocks), max(clocks)] if clocks else None,
            "end": t_end}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--warmup", type=float, default=3.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = []
    for rnd in range(a.rounds):
        for f in a.files:
            rec = run_once(ROOT / f, a.warmup)
            rec["round"] = rnd
            runs.append(rec)
            print(f"round {rnd} {Path(f).name:<40} first rep {rec['pp512_samples'][0]:8.1f}  clock at start "
                  f"{rec['clock_at_start_mhz']}  during rep 1 {rec['rep1_clock_mhz']}", flush=True)
            Path(a.out).write_text(json.dumps({"runs": runs}, indent=1))


if __name__ == "__main__":
    main()
