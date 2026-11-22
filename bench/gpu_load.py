"""Keep the GPU busy until stopped (warm-up load for bench/run_bench.py).

What
    Copies a 256 MiB buffer on the first CUDA device in a loop (or, with --mode gemm, runs float16 4096 x 4096 matrix
    products), prints "ready" once it is set up, and runs until it is terminated (or until `--seconds` have passed).
Why
    The laptop GPU of this project idles at 1605 MHz and needs about a second of load to reach its boost clock
    (2040-2340 MHz); it drops back within a fraction of a second when the load stops, e.g. while llama-bench loads a
    model.  bench/run_bench.py therefore starts this load before the measured llama-bench, keeps it running while
    llama-bench loads the model, and stops it only when llama-bench starts its own (untimed) warm-up pass, so the GPU
    reaches the measured repetitions at its boost clock.
Usage
    python -m bench.gpu_load [--seconds S]
"""
from __future__ import annotations

import argparse
import sys
import time


def main(argv: list[str] | None = None) -> int:
    """Command-line entry (see the module docstring)."""
    ap = argparse.ArgumentParser(description="GPU warm-up load (float16 matrix products) until terminated.")
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after this many seconds (0 = run until killed)")
    ap.add_argument("--mode", choices=("copy", "gemm"), default="copy",
                    help="copy: memory-bound copies of 256 MiB (default; close to the power draw of token generation); "
                         "gemm: float16 matrix products (compute-bound, draws more power)")
    a = ap.parse_args(argv)
    import torch
    if not torch.cuda.is_available():
        print("no CUDA device", flush=True)
        return 1
    if a.mode == "gemm":
        x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
        y = x @ x
    else:
        x = torch.empty(64 << 20, device="cuda", dtype=torch.float32)   # 256 MiB, larger than the 24 MB L2
        y = torch.empty_like(x)
    torch.cuda.synchronize()
    print("ready", flush=True)
    t0 = time.perf_counter()
    while not a.seconds or time.perf_counter() - t0 < a.seconds:
        for _ in range(8):
            if a.mode == "gemm":
                y = x @ y
                y = y * 1e-3  # keeps the values finite
            else:
                y.copy_(x)
        torch.cuda.synchronize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
