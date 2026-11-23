# Phase 1 speed results (CUDA, RTX 4050 Laptop GPU)

**Gate: PASS (formal, original weights).** LittleBit at 0.55 bits per weight, with the default F16 embedding / output
head, generates tokens 1.68x faster than bf16 on Qwen3-0.6B (212.9 vs 126.9 tokens/s) and 2.80x faster on Qwen3-1.7B
(142.9 vs 51.1 tokens/s), both on the original Hugging Face weights (Qwen/Qwen3-0.6B, Qwen/Qwen3-1.7B). With a Q8_0
head: 2.20x and 3.44x; with a Q6_K head: 2.38x and 3.69x. Per round (LittleBit F16 head / bf16): 0.6B 1.614 / 1.733 /
1.687, 1.7B 2.811 / 2.786 / 2.800. `check_gate` on these results: True without proxy results (`gate_status`: "PASS").

Real vs proxy weights (the first 1.7B measurement used weights dequantized from a Q4_K_M file; both measured in
this run, interleaved): LittleBit 0.55 142.9 vs 143.1 tokens/s (-0.17 %), bf16 51.1 vs 51.1 (-0.02 %); pp512 -0.6 % /
-4.9 % (pp512 varies by up to 10 % between rounds). The proxy rows are kept below as a labelled comparison only.

## Setup

- GPU: NVIDIA GeForce RTX 4050 Laptop GPU (6 GB, sm_89), driver 610.78, CUDA 12.9; CPU i7-13620H; Windows 11; laptop
  plugged in. During this run the GPU sat at about 70 °C at its 45 W power cap (nvidia-smi throttle reason "SW power
  cap").
- Build provenance (stored in [phase1.json](phase1.json), `provenance`): engine commit d3a5fd344, working tree clean,
  and llama-bench reports the same commit (`version: 1.0.0-dev (build 15, commit d3a5fd344)`: the engine was rebuilt
  in full after the commit, so its build info is current); LittleMore commit 46b06e3 for the harness.
- Harness: `bench/run_bench.py` (see [../README.md](../README.md)): before each llama-bench call a GPU load
  (`bench/gpu_load.py`, memory-bound copies) runs 3 s and keeps running while llama-bench loads the model, until
  llama-bench starts its own untimed warm-up pass; then `llama-bench -ngl 99 -dev CUDA0 -p 0 -n 128 -r 10` and, with
  the same warm-up, `-p 512 -n 0 -r 10`, default settings otherwise (flash attention auto, F16 KV cache). Each value is
  the median of 10 repetitions; the tables give the mean of 3 rounds (A B C ... A B C ...) and the range of the round
  medians. Raw data: [phase1.json](phase1.json).
- SM clock at the first timed repetition: tg128 1800 .. 2325 MHz (median 2280), pp512 1020 .. 2325 (median 2325). Runs
  below 1800 MHz at either start are flagged and repeated once: 13 of the 48 runs were repeated (the repeat replaces
  the first attempt, whose clocks are not kept); 2 stayed flagged (pp512 started at 1335 and 1020 MHz:
  `qwen3-1.7b-proxy` LittleBit round 0, `qwen3-1.7b` LittleBit Q8_0 head round 2); their tg128, the gate metric,
  started at 2325 MHz.
- First-repetition penalty of tg128 (first repetition against the median of the other nine): LittleBit median 0.4 %
  (-7.6 .. 9.1 %, 27 runs), bf16 0.4 % (0 .. 13.2 %), f16 5.7 % (0.3 .. 14.6 %), Q4_K_M 3.7 % (0.3 .. 16.4 %). Without the
  load (measured before it was added) it was 11-15 %. The reported median does not depend on it.
- VRAM: from a separate pass of the same plan without the warm-up load (`--warmup 0`, 1 round, 3 repetitions;
  [phase1_vram.json](phase1_vram.json)): with the load, its memory stays visible in nvidia-smi for a moment after it
  exits (the speed run showed e.g. 2158 instead of 1609 MiB for bf16 Qwen3-0.6B), so the speed run stores no VRAM.
  (Note: the harness was modified after this run. The speed run's JSON had kept the inflated readings in `vram_mb`;
  they are now nulled in [phase1.json](phase1.json) (see its `vram_note`), so `phase1_vram.json` is the only VRAM
  source. The SM clock columns below were sampled with the harness of this run, from 1 s after the load stopped to the
  end, so their minima include llama-bench's untimed warm-up pass, e.g. 735 MHz; the clock at the first timed
  repetition is the line above. The current harness samples only the timed repetitions.)
  Value = peak of nvidia-smi's `memory.used` during the run minus the value before it (weights, KV cache, compute
  buffers, CUDA context); identical to the first run of this plan for every file measured in both. The bf16 Qwen3-1.7B (3881
  MiB file) fits the 6 GB GPU: 3763 MiB.
- Absolute speeds are about 10 % below the first run of this plan on the same machine (Qwen3-0.6B LittleBit
  236.3 then, 212.9 now; bf16 127.5 / 126.9): this run started after hours of GPU load, at the power cap. LittleBit
  generation (latency and launch bound) loses more than the memory-bound dense models, so the 0.6B ratio is 1.68x here
  against 1.85x then; the 1.7B ratio is 2.80x against 2.82x. Compare files only within one run.

## Files

| variant | what |
|---|---|
| littlebit-0.55 / -0.1 | init-only LittleBit-2 checkpoint (`tools/dev/make_lb2_init.py --bpw 0.55 / 0.1`, residual path, ITQ) converted with `tools/convert/lb_to_gguf.py` (B1 signs, F16 head) |
| littlebit-...-q8_0head / -q6_khead | the same checkpoint converted with `--head-type q8_0 / q6_k` (only the embedding / output head differ; the converter checks the quantized head against the checkpoint) |
| bf16 / f16 | the dense base model through llama.cpp's `convert_hf_to_gguf.py --outtype bf16 / f16` |
| q4_k_m | `llama-quantize <bf16 file> Q4_K_M` (no importance matrix) |

Qwen3-0.6B and Qwen3-1.7B both come from the original Hugging Face repositories; their safetensors store `lm_head`
next to the tied embedding, so llama.cpp writes a separate `output.weight` (the same bytes read per token as a tied
head). Ranks after padding to multiples of 128 (per path; the residual path has the same rank): at 0.55 bpw Qwen3-0.6B
256 (k / v 128), Qwen3-1.7B 384 (q / o), 256 (k / v), 512 (gate / up / down); at 0.1 bpw every linear has the minimum
128. The 0.1-bpw Qwen3-1.7B files of the first run (proxy weights) were not rebuilt on the original weights.

## Results (tokens per second)

### qwen3-0.6b

| variant | file MiB | tg128 t/s (mean of the round medians; min..max of them) | pp512 t/s (same) | VRAM MiB (separate pass) | SM clock MHz (min..max sampled from 1 s after the load stopped, incl. the untimed warm-up pass) |
|---|---|---|---|---|---|
| littlebit-0.55 | 642 | 212.9 (205.6..219.6) | 3113 (3020..3286) | 821 | 735..2340 |
| bf16 | 1439 | 126.9 (126.5..127.4) | 8295 (7638..9489) | 1609 | 1125..2340 |
| littlebit-0.55-q8_0head | 364 | 279.6 (276.0..284.9) | 2959 (2921..3012) | 683 | 630..2340 |
| littlebit-0.55-q6_khead | 292 | 301.5 (298.2..303.9) | 3040 (2974..3118) | 647 | 795..2325 |
| littlebit-0.1 | 626 | 220.9 (216.2..225.1) | 5346 (4648..6219) | 801 | 765..2325 |
| littlebit-0.1-q8_0head | 348 | 289.5 (283.8..294.6) | 5340 (4779..6128) | 663 | 840..2325 |
| f16 | 1439 | 126.2 (126.0..126.6) | 9440 (8926..9715) | 1609 | 1035..2325 |
| q4_k_m | 462 | 251.5 (249.2..254.4) | 11243 (10195..12412) | 829 | 1350..2325 |

### qwen3-1.7b

| variant | file MiB | tg128 t/s (mean of the round medians; min..max of them) | pp512 t/s (same) | VRAM MiB (separate pass) | SM clock MHz (min..max sampled from 1 s after the load stopped, incl. the untimed warm-up pass) |
|---|---|---|---|---|---|
| littlebit-0.55 | 1322 | 142.9 (142.2..143.5) | 1475 (1449..1490) | 1223 | 945..2325 |
| bf16 | 3881 | 51.1 (51.0..51.1) | 3284 (3228..3370) | 3763 | 1125..2325 |
| littlebit-0.55-q8_0head | 766 | 175.8 (170.1..179.1) | 1495 (1449..1523) | 945 | 1020..2325 |
| littlebit-0.55-q6_khead | 622 | 188.3 (185.5..192.4) | 1537 (1528..1556) | 873 | 1110..2325 |
| f16 | 3881 | 51.0 (51.0..51.0) | 4020 (3683..4365) | 3765 | 555..2325 |
| q4_k_m | 1223 | 132.6 (132.4..132.7) | 4731 (4585..4883) | 1517 | 1170..2325 |

### qwen3-1.7b-proxy (PROXY weights: speed only, not the original model)

| variant | file MiB | tg128 t/s (mean of the round medians; min..max of them) | pp512 t/s (same) | VRAM MiB (separate pass) | SM clock MHz (min..max sampled from 1 s after the load stopped, incl. the untimed warm-up pass) |
|---|---|---|---|---|---|
| littlebit-0.55 | 729 | 143.1 (142.9..143.5) | 1484 (1457..1531) | 1223 | 990..2325 |
| bf16 | 3288 | 51.1 (51.0..51.1) | 3452 (3265..3814) | 3763 | 1110..2325 |

- gate littlebit-0.55 > bf16 (tg128, qwen3-0.6b and qwen3-1.7b): **PASS**
- gate littlebit-0.55 > f16 (tg128, qwen3-0.6b and qwen3-1.7b): **PASS**
- gate littlebit-0.55-q6_khead > bf16 (tg128, qwen3-0.6b and qwen3-1.7b): **PASS**
- gate littlebit-0.55-q6_khead > f16 (tg128, qwen3-0.6b and qwen3-1.7b): **PASS**
- gate littlebit-0.55-q8_0head > bf16 (tg128, qwen3-0.6b and qwen3-1.7b): **PASS**
- gate littlebit-0.55-q8_0head > f16 (tg128, qwen3-0.6b and qwen3-1.7b): **PASS**


## Reading the numbers

- **Gate (tg128, LittleBit 0.55 > bf16), original weights:** 0.6B 212.9 vs 126.9 (1.68x), 1.7B 142.9 vs 51.1 (2.80x),
  F16 head; the same holds against F16 dense (126.2, 51.0) and with either quantized head. `check_gate`: True (no proxy
  results needed). No speed lever was needed; the 2A-S variant was not touched.
- **Target "on par with Q4_K_M":** with the F16 head LittleBit is 0.85x Q4_K_M on 0.6B (213 vs 252) and 1.08x on 1.7B
  (143 vs 133); with a Q8_0 head 1.11x / 1.33x, with a Q6_K head 1.20x / 1.42x. On 0.6B the F16 head (297 MiB of the
  642 MiB file, read for every token) is what keeps LittleBit below Q4_K_M, whose head is Q6_K.
- **Head type:** Q8_0 makes generation 1.31x (0.6B) / 1.23x (1.7B) faster than the F16 head, Q6_K 1.42x / 1.32x; VRAM
  drops by 138 / 174 MiB (0.6B) and 278 / 350 MiB (1.7B).
- **0.1 bpw** is barely faster than 0.55 at this size (0.6B: 220.9 vs 212.9): the rank floor of 128 per path leaves
  about half of the sign data (19 vs 35 MiB), and the time is dominated by the head and the per-launch cost of the 196
  small LittleBit ops (docs/engine-changes.md, decode profile).
- **Prompt processing is slower than dense** (0.6B pp512 3113 vs 8295 bf16; 1.7B 1475 vs 3284): the prefill path runs
  float32 GEMMs on the CUDA cores against f16 / bf16 tensor cores (known, docs/engine-changes.md "CUDA: batches above 8
  tokens"; not part of the gate). pp512 also drifts within and between rounds at the power cap (round range up to 29 %
  on the 0.1-bpw files, 22 % on bf16 0.6B), so compare it only coarsely.
- **Spread:** the round medians of tg128 stay within 6.6 % of each other for every file (Qwen3-0.6B LittleBit F16 head
  205.6 .. 219.6 is the widest; 1.7B Q8_0 head 5.2 %); the dense and Q4_K_M files within 2.1 %.

## Perplexity delta of the head type

`llama-perplexity -f wiki.test.raw -ngl 99 -c 512 --chunks 64` (WikiText-2 raw test split, the first 64 chunks of 512
tokens, 32768 tokens; the same chunks for every file):

| file | F16 / bf16 head | Q8_0 head | Q6_K head |
|---|---|---|---|
| Qwen3-0.6B dense bf16 (head quantized with `llama-quantize --token-embedding-type / --output-tensor-type`) | 21.279 +- 0.552 | 21.292 (+0.06 %) | 21.359 (+0.38 %) |
| Qwen3-1.7B dense bf16 (same) | 16.480 +- 0.460 | 16.490 (+0.06 %) | 16.461 (-0.12 %) |
| Qwen3-0.6B LittleBit 0.55, init-only | 2.175e7 | 2.167e7 (-0.4 %) | 2.174e7 (-0.04 %) |
| Qwen3-1.7B LittleBit 0.55, init-only | 6.70e7 | 6.60e7 (-1.5 %) | 7.11e7 (+6.2 %) |

The dense rows are the meaningful ones: a Q8_0 head changes perplexity by +0.06 % on both models, a Q6_K head by +0.38 %
(0.6B) and -0.12 % (1.7B). (The +-0.55 / +-0.46 printed by llama-perplexity is the uncertainty of the absolute value on
this sample; the runs score the same tokens, so their differences are much better determined than that, but their own
uncertainty was not computed.) The init-only LittleBit checkpoints are untrained (perplexity in the tens of millions,
gibberish by design), so their deltas say only that a quantized head moves an extremely flat output distribution by a
few percent; they are not a quality measurement. The LittleBit files' heads are quantized by the converter
(`lb_to_gguf.py --head-type`, which also checks the quantized head against the checkpoint), the dense ones by
llama-quantize; both use ggml's reference quantizers (the converter's are bit-identical to them,
tests/convert/test_head_type.py). The 0.6B rows are from the first run of this plan (same files).

## Reproduce

```powershell
# base models (original weights) and checkpoints
.venv\Scripts\hf.exe download Qwen/Qwen3-1.7B --local-dir .cache/models/Qwen3-1.7B
.venv\Scripts\python.exe tools/dev/make_lb2_init.py .cache/models/Qwen3-1.7B .cache/ckpt/qwen3-1.7b-lb2-0.55 --bpw 0.55
.venv\Scripts\python.exe tools/dev/make_lb2_init.py .cache/models/Qwen3-0.6B .cache/ckpt/qwen3-0.6b-lb2-0.1 --bpw 0.1
# one conversion per file of bench/plans/phase1.json, e.g.
.venv\Scripts\python.exe -m tools.convert.lb_to_gguf .cache/ckpt/qwen3-1.7b-lb2-0.55 -o .cache/gguf/qwen3-1.7b-lb2-0.55-b1-q8_0head.gguf --head-type q8_0
.venv\Scripts\python.exe engine/convert_hf_to_gguf.py .cache/models/Qwen3-1.7B --outtype bf16 --outfile .cache/gguf/qwen3-1.7b-base-bf16.gguf
engine\build\bin\llama-quantize.exe .cache/gguf/qwen3-1.7b-base-bf16.gguf .cache/gguf/qwen3-1.7b-base-q4_k_m.gguf Q4_K_M
# (the proxy rows: tools/dev/dequant_gguf_to_hf.py on a Qwen3-1.7B Q4_K_M GGUF, then the same steps)
# speed, VRAM, tables
.venv\Scripts\python.exe -m bench.run_bench --plan bench/plans/phase1.json --rounds 3 --out bench/results/phase1.json
.venv\Scripts\python.exe -m bench.run_bench --plan bench/plans/phase1.json --rounds 1 --reps 3 --warmup 0 --out bench/results/phase1_vram.json
.venv\Scripts\python.exe -m bench.run_bench --markdown bench/results/phase1.json --vram bench/results/phase1_vram.json
```
