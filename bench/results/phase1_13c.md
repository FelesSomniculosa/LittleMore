# Phase 1: CUDA prefill levers of GGML_OP_LITTLEBIT (lossless)

Engine: the first tensor-core prefill build `55474a330` (called "TC build" below) against the speed-lever build `12338df9a` (called "lever build" below; "13b" / "13c" in file names are the same two builds). RTX 4050 Laptop GPU (20 SMs,
power cap about 45 W), CUDA 12.9, no GPU setting changed.

## Op level (prefill_bench, 512 tokens, 3 interleaved rounds)

Cycles = time per op x the median SM clock sampled every 100 ms while the GPU draws more than 25 W (the clock under
this sustained load is power-capped and varies from run to run, so times alone are not comparable). Raw lines:
`phase1_13c_ops_raw.txt` (round, build, shape d_in d_out r r2, us, MHz, kcycles).

| layer (padded ranks) | TC build kcycles | lever build kcycles | lever build / TC build | lever build us (at its clock) |
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

Three rounds; in each round the TC build and the lever build each ran once with all four files (the ggml-cuda DLL
swapped between them).

**Selection rule (one rule for every run of both builds and of F16):** a run counts only if its first timed
repetition has clock samples and all of them are 2340 MHz (min = max = 2340). The clock at the start of the
repetition is not used. Excluded by this rule (5 of 24 runs):
- TC build round 0, 1.7B LittleBit 4522 (rep-1 clock 1620-2340);
- lever build round 0, 0.6B LittleBit 12349 (no clock sample inside rep 1);
- lever build round 1, 1.7B F16 6089 (1470 MHz);
- TC build round 2, 0.6B LittleBit 11028 (no clock sample);
- TC build round 2, 1.7B LittleBit 4159 (1800-2340) and 1.7B F16 5997 (1575 MHz).

| run | Qwen3-0.6B LittleBit | 0.6B F16 | Qwen3-1.7B LittleBit | 1.7B F16 |
|---|---|---|---|---|
| TC build round 0 | 12127 | 14486 | (4522, excluded) | 5656 |
| lever build round 0 | (12349, excluded) | 13001 | 3775 | 6342 |
| TC build round 1 | 12000 | 11836 | 4350 | 5385 |
| lever build round 1 | 13541 | 12708 | 4871 | (6089, excluded) |
| TC build round 2 | (11028, excluded) | 9050 | (4159, excluded) | (5997, excluded) |
| lever build round 2 | 12985 | 13936 | 4756 | 5495 |

Reading (included runs only; ratios against the median of all included F16 runs, which are interchangeable between
builds since the F16 files do not use the changed code):
- **F16 spread at the same 2340 MHz:** Qwen3-0.6B 9050-14486 (median 12854, 6 runs), Qwen3-1.7B 5385-6342 (median
  5576, 4 runs). Single-run ratios are therefore uncertain by tens of percent.
- **Qwen3-0.6B:** TC build 12000-12127 = 0.93-0.94; lever build 12985-13541 = 1.01-1.05. Parity at equal clocks, within this noise.
  (Op-level bound: 0.82x cycles per layer allows up to 0.94 / 0.82 = 1.14; the observed gain is inside it.)
- **Qwen3-1.7B:** TC build one run 4350 = 0.78; lever build runs 3775 / 4871 / 4756 = 0.68 / 0.87 / 0.85 (median 0.85; the 3775 run
  is about 20 % below the other two at the same clock and is not explained). The op-level data (0.93x cycles per
  layer) bounds the model gain to about 7.5 %, i.e. 0.78 / 0.93 = 0.84 from the TC build run; the observed 0.85-0.87 is not
  distinguishable from that bound within the run-to-run noise. Plainly: about 0.84-0.87, below F16.

## VRAM (measured)

Peak GPU memory during `llama-bench -m <LittleBit file> -ngl 99 -p 512 -n 0 -r 3`: the maximum of `nvidia-smi
memory.used` sampled every 50 ms during the run minus the value just before it (0 MiB at idle on this laptop), 2
interleaved rounds per build, ggml-cuda DLL swapped:

| model | TC build | lever build | change |
|---|---|---|---|
| Qwen3-0.6B | 801, 801 MiB | 791, 791 MiB | -10 MiB |
| Qwen3-1.7B | 1191, 1191 MiB | 1181, 1181 MiB | -10 MiB |

(The planes of x no longer go to the CUDA pool; computed for 1.7B down at 512 tokens: 13.1 MB -> 3.7 MB of scratch.)
