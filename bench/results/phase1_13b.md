# Task 13b: prompt processing after the tensor-core prefill path

What: the gate harness (`bench/run_bench.py`, the Task 13 protocol: 3 interleaved rounds, each value the median of 10
llama-bench repetitions, tg128 before pp512, 3 s warm-up load kept running until llama-bench's warm-up pass) on the
original Qwen3-0.6B and Qwen3-1.7B, for the LittleBit 0.55 bpw file (F16 head) with the engine's new default (the
exact tensor-core prefill path, `ggml-cuda/littlebit-bp.cu`) and, in the same rounds, the same file with
`GGML_CUDA_LITTLEBIT_F32_PREFILL=1` (the float32 GEMM prefill path, i.e. the engine before this task), plus the dense
F16 and bf16 baselines. Plan: [../plans/phase1_13b.json](../plans/phase1_13b.json); raw data:
[phase1_13b.json](phase1_13b.json), VRAM from a separate pass without the load ([phase1_13b_vram.json](phase1_13b_vram.json)).

Setup: RTX 4050 Laptop GPU (6 GB), CUDA 12.9, engine commit in the JSON's `provenance` (clean tree, llama-bench built
from that commit). Absolute numbers depend on the GPU's clock and power state (this laptop boosts from 1605 to about
2040-2340 MHz and is capped at about 45 W); compare within one run.

Reproduce:

```powershell
.venv\Scripts\python.exe -m bench.run_bench --plan bench/plans/phase1_13b.json --rounds 3 --out bench/results/phase1_13b.json
.venv\Scripts\python.exe -m bench.run_bench --plan bench/plans/phase1_13b.json --rounds 1 --reps 3 --warmup 0 --out bench/results/phase1_13b_vram.json
.venv\Scripts\python.exe -m bench.run_bench --markdown bench/results/phase1_13b.json --vram bench/results/phase1_13b_vram.json
```

## Results

### qwen3-0.6b

| variant | file MiB | tg128 t/s (mean of the round medians; min..max of them) | pp512 t/s (same) | VRAM MiB (separate pass) | SM clock MHz (min..max sampled during the timed repetitions) |
|---|---|---|---|---|---|
| littlebit-0.55 | 642 | 219.6 (216.5..223.4) | 9628 (9301..9934) | 801 | 1740..2130 |
| littlebit-0.55-f32prefill | 642 | 218.7 (217.6..220.5) | 3051 (2958..3202) | 821 | 660..2325 |
| f16 | 1439 | 126.5 (126.0..126.9) | 11643 (11414..12060) | 1609 | 1530..1845 |
| bf16 | 1439 | 126.6 (126.3..126.7) | 9256 (8242..9925) | 1609 | 1545..2325 |

### qwen3-1.7b

| variant | file MiB | tg128 t/s (mean of the round medians; min..max of them) | pp512 t/s (same) | VRAM MiB (separate pass) | SM clock MHz (min..max sampled during the timed repetitions) |
|---|---|---|---|---|---|
| littlebit-0.55 | 1322 | 143.3 (143.0..143.7) | 4118 (3974..4205) | 1191 | 1395..2160 |
| littlebit-0.55-f32prefill | 1322 | 143.3 (142.8..143.6) | 1531 (1528..1534) | 1223 | 990..2325 |
| f16 | 3881 | 50.9 (50.9..51.0) | 3872 (3414..4693) | 3765 | 705..2325 |
| bf16 | 3881 | 51.0 (51.0..51.1) | 3722 (3428..4102) | 3763 | 840..2325 |

- gate littlebit-0.55 > bf16 (tg128, qwen3-0.6b and qwen3-1.7b): **PASS**
- gate littlebit-0.55 > f16 (tg128, qwen3-0.6b and qwen3-1.7b): **PASS**
- gate littlebit-0.55-f32prefill > bf16 (tg128, qwen3-0.6b and qwen3-1.7b): **PASS**
- gate littlebit-0.55-f32prefill > f16 (tg128, qwen3-0.6b and qwen3-1.7b): **PASS**

(The gate lines are printed by the generic renderer for every LittleBit variant; the formal Phase 1 gate stays the one
in [phase1.md](phase1.md).)

## Reading

- **pp512, Qwen3-0.6B:** 9628 t/s with the tensor-core prefill path against 3051 with the float32 GEMM path
  (3.16x; per round 3.14 / 3.01 / 3.32), F16 11643 (LittleBit / F16 per round 0.815 / 0.842 / 0.824, i.e. **0.83x**),
  bf16 9256 (1.04x). Not yet at F16 on this model: its 196 LittleBit linears are small (1024..3072 wide, ranks 128..256
  per path), where the fixed cost per call of the three kernels weighs most (docs/engine-changes.md, "CUDA: tensor-core
  prefill path", op-level table: e.g. k / v 100 us against 67 us for F16).
- **pp512, Qwen3-1.7B:** 4118 against 1531 (2.69x; per round 2.60 / 2.73 / 2.74), F16 3872, bf16 3722. The mean ratio
  1.06x F16 is a **power-cap artifact, not a speed parity**: within each 10-repetition run the F16 model falls from
  5582 / 5156 / 5362 to about 2800-3400 t/s with the SM clock at 705-825 MHz during its timed repetitions (LittleBit's
  minimum 1395 MHz): the dense model draws more power on its tensor cores and hits the laptop's ~45 W cap harder. The
  per-op table (docs/engine-changes.md) has every 1.7B shape slower than F16 (about 91 vs 70 ms of linears per pass).
- **At equal clocks** ([phase1_13b_clock.json](phase1_13b_clock.json), `bench/clock_matched.py`: 6 interleaved rounds
  of short pp512 runs started like the harness, the FIRST timed repetition of each run, SM clock sampled during it; no
  GPU setting changed): Qwen3-1.7B, both files at 2325-2340 MHz in rounds 2-5: LittleBit / F16 = 0.97, 0.83, 0.82,
  0.87 (round 0, both near 2200 MHz: 0.86), i.e. **about 0.8-0.9x F16**; Qwen3-0.6B, the one round with both at
  2340 MHz: 0.98, the other rounds 0.82-0.94 with F16 at a LOWER clock (1665-2115 MHz) than LittleBit (2130-2340), so
  at equal clocks 0.6B is also below F16. First repetitions of the 10-repetition harness runs agree: 1.7B 0.79 /
  0.89 / 0.98, 0.6B 0.82 / 0.82 / 0.90.

- **tg128 unchanged:** decoding (1 token per call) runs the same decode kernels; LittleBit / float32-path ratio per
  round 0.993 .. 1.025 (noise).
- **VRAM:** 801 MiB vs 821 (0.6B) and 1191 vs 1223 (1.7B): the new path's scratch (planes, at most 12 MiB per call) is
  smaller than the float32 path's dequantized tiles (16 MiB each).
- Clock at the first timed repetition: tg128 1830 .. 2340 MHz, pp512 1890 .. 2325 MHz, except two pp512 runs of the
  float32 path that stayed flagged low after their automatic repeat (1380 and 1410 MHz; `clock_low` in the JSON).
