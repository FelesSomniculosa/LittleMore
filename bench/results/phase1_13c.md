# Phase 1, Task 13c: CUDA prefill levers of GGML_OP_LITTLEBIT (lossless)

Engine: Task 13b build `eaf9b01e6` ("13b") against this task's `2d9b9e98b` ("13c"). RTX 4050 Laptop GPU (20 SMs,
power cap about 45 W), CUDA 12.9, no GPU setting changed.

## Op level (prefill_bench, 512 tokens, 3 interleaved rounds)

Cycles = time per op x the median SM clock sampled every 100 ms while the GPU draws more than 25 W (the clock under
this sustained load is power-capped and varies from run to run, so times alone are not comparable). Raw lines:
`phase1_13c_ops_raw.txt` (round, build, shape d_in d_out r r2, us, MHz, kcycles).

| layer (padded ranks) | 13b kcycles | 13c kcycles | 13c / 13b | 13c us (at its clock) |
|---|---|---|---|---|
| 0.6B q 1024 x 2048 (256 + 256) | 287 | 238 | 0.83 | 125 |
| 0.6B k, v 1024 x 1024 (128 + 128) | 185 | 152 | 0.82 | 76 |
| 0.6B o, 1.7B k, v 2048 x 1024 (256 + 256) | 321 | 274 | 0.85 | 140 |
| 0.6B gate, up 1024 x 3072 (256 + 256) | 364 | 295 | 0.81 | 161 |
| 0.6B down 3072 x 1024 (256 + 256) | 440 | 358 | 0.81 | 188 |
| 1.7B q, o 2048 x 2048 (384 + 384) | 616 | 567 | 0.92 | 299 |
| 1.7B gate, up 2048 x 6144 (512 + 512) | 1143 | 1127 | 0.99 | 612 |
| 1.7B down 6144 x 2048 (512 + 512) | 1477 | 1290 | 0.87 | 679 |
| one layer, Qwen3-0.6B (q, k, v, o, gate, up, down) | 2146 | 1764 | 0.82 | |
| one layer, Qwen3-1.7B | 5637 | 5226 | 0.93 | |

## Model level (bench/clock_matched.py, pp512, first timed repetition; `phase1_13c_clock.json`)

Three rounds; in each round the 13b build and the 13c build each ran once with all four files (the ggml-cuda DLL
swapped between them). Only runs whose first repetition ran entirely at 2340 MHz are used for ratios.

| run | Qwen3-0.6B LittleBit | 0.6B F16 | Qwen3-1.7B LittleBit | 1.7B F16 |
|---|---|---|---|---|
| 13b round 0 | 12127 | 14486 | 4522 (clock 1620-2340) | 5656 |
| 13c round 0 | 12349 (no clock sample) | 13001 | 3775 (clock at start 1665) | 6342 |
| 13b round 1 | 12000 | 11836 | 4350 | 5385 |
| 13c round 1 | 13541 | 12708 | 4871 | 6089 (1470 MHz) |
| 13b round 2 | 11028 (no clock sample) | 9050 | 4159 (1800-2340) | 5997 (1575 MHz) |
| 13c round 2 | 12985 | 13936 | 4756 | 5495 |

Reading, at 2340 MHz only:
- **LittleBit itself:** Qwen3-0.6B 12000-12127 (13b) against 12985-13541 (13c), about +9 %; Qwen3-1.7B 4350 against
  4756-4871, about +11 %.
- **Against F16:** the F16 first repetitions at 2340 MHz spread 11836-14486 (0.6B) and 5385-6342 (1.7B) between runs,
  so single-run ratios are noisy (0.6B 13c: 1.07 and 0.93; 1.7B 13c: 0.87). Against the median of all F16 runs at
  2340 MHz (0.6B 13001, 1.7B 5576): 0.6B 0.93 (13b) to 1.02 (13c); 1.7B 0.78 (13b) to 0.86 (13c).
- **Verdict:** Qwen3-0.6B reaches parity with F16 at equal clocks within this noise; Qwen3-1.7B does not (about 0.86).
