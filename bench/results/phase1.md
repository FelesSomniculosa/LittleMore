# Phase 1 speed results (CUDA, RTX 4050 Laptop GPU)

**Gate: PASS.** LittleBit at 0.55 bits per weight, with the default F16 embedding / output head, generates tokens
1.85x faster than bf16 on Qwen3-0.6B (236.3 vs 127.5 tokens/s) and 2.82x faster on Qwen3-1.7B (143.8 vs 51.0
tokens/s). With a Q8_0 head: 2.34x and 3.64x; with a Q6_K head: 2.49x and 3.91x. The ratio was the same in every
round (0.6B: 1.853 / 1.855 / 1.853; 1.7B: 2.824 / 2.815 / 2.815). **Read the caveat on the Qwen3-1.7B weights below.**

## Setup

- GPU: NVIDIA GeForce RTX 4050 Laptop GPU (6 GB, sm_89), driver 610.78, CUDA 12.9; CPU i7-13620H; Windows 11; laptop
  plugged in. Engine: the LittleMore llama.cpp fork with the Task 13 changes (llama-bench prints the commit of the
  last build's base, 0eb3db1ba; the Task 13 engine changes were built, measured, then committed unchanged).
- Harness: `bench/run_bench.py` (see [../README.md](../README.md)): 3 s GPU warm-up before each file, then
  `llama-bench -ngl 99 -dev CUDA0 -p 0 -n 128 -r 10` and `-p 512 -n 0 -r 10`, default settings otherwise (flash attention
  auto, F16 KV cache). Each value is the median of 10 repetitions; the tables give the mean of 3 rounds (A B C ... A B C
  ...) and the range of the round medians. Raw data: [phase1.json](phase1.json).
- VRAM: the peak of nvidia-smi's `memory.used` during the run minus the value before it (everything the process
  allocated: weights, KV cache, compute buffers, CUDA context). The bf16 Qwen3-1.7B (3288 MiB file) fits the 6 GB GPU:
  3763 MiB used.
- SM clock column: min..max of nvidia-smi samples over the whole run, including the model loading between the two
  llama-bench calls (the low minimums are those pauses); the median during the runs was 1.8-2.3 GHz.

## Files

| variant | what |
|---|---|
| littlebit-0.55 / -0.1 | init-only LittleBit-2 checkpoint (`tools/dev/make_lb2_init.py --bpw 0.55 / 0.1`, residual path, ITQ) converted with `tools/convert/lb_to_gguf.py` (B1 signs, F16 head) |
| littlebit-...-q8_0head / -q6_khead | the same checkpoint converted with `--head-type q8_0 / q6_k` (only the embedding / output head differ) |
| bf16 / f16 | the dense base model through llama.cpp's `convert_hf_to_gguf.py --outtype bf16 / f16` |
| q4_k_m | `llama-quantize <bf16 file> Q4_K_M` (no importance matrix) |

Ranks after padding to multiples of 128 (per path; the residual path has the same rank): at 0.55 bpw Qwen3-0.6B 256
(k / v 128), Qwen3-1.7B 384 (q / o), 256 (k / v), 512 (gate / up / down); at 0.1 bpw every linear has the minimum 128,
so the 0.1 files are barely smaller than the 0.55 ones and decode at about the same speed (the head dominates).

### Caveat: Qwen3-1.7B uses proxy weights (speed only)

The original Qwen3-1.7B bf16 weights are a 4 GB download, which was not approved for this task. The 1.7B files were
made from a Qwen3-1.7B Q4_K_M GGUF already on this machine (Unsloth, with importance matrix), dequantized back to a
Hugging Face directory with `tools/dev/dequant_gguf_to_hf.py` (exact Qwen3-1.7B shapes and tied head; config and
tokenizer from the Qwen/Qwen3-1.7B repository, small files), then made into the init-only LittleBit checkpoints, the
bf16 / f16 files and the Q4_K_M file as above. The speed of every kernel involved depends on tensor shapes and types
only, not on the values. Control on Qwen3-0.6B in the same plan and rounds: the same pipeline through a Q4_K_M file and
back gives tg128 235.6 (LittleBit 0.55) and 127.1 (bf16) against 236.3 and 127.5 with the real weights, i.e. within
0.3 %, inside the round-to-round range. The perplexities of the 1.7B files below are NOT those of the original model
(the proxy carries the Q4_K_M rounding: 17.06 for the proxy bf16 file). The formal 1.7B measurement on the original
weights needs that download and one more run of the plan.

## Results (tokens per second)

### qwen3-0.6b

| variant | file MiB | tg128 t/s (mean of the round medians; min..max of them) | pp512 t/s (same) | VRAM MiB | SM clock MHz (min..max sampled, incl. model loading) |
|---|---|---|---|---|---|
| littlebit-0.55 | 642 | 236.3 (234.9..238.2) | 3525 (3411..3622) | 821 | 825..2340 |
| bf16 | 1439 | 127.5 (126.8..128.6) | 9961 (9898..10048) | 1609 | 1515..2340 |
| littlebit-0.55-q8_0head | 364 | 298.1 (297.1..299.0) | 3680 (3578..3861) | 683 | 1110..2325 |
| littlebit-0.55-q6_khead | 292 | 317.9 (316.9..319.2) | 3515 (3431..3560) | 647 | 1095..2325 |
| littlebit-0.1 | 626 | 241.8 (241.3..242.7) | 6207 (5960..6358) | 801 | 1125..2325 |
| littlebit-0.1-q8_0head | 348 | 304.3 (303.7..305.2) | 5738 (5330..6017) | 663 | 735..2325 |
| f16 | 1439 | 126.9 (126.4..127.4) | 12439 (11901..12806) | 1609 | 1260..2325 |
| q4_k_m | 462 | 277.2 (276.3..277.7) | 14070 (13982..14175) | 829 | 1650..2325 |

### qwen3-1.7b

| variant | file MiB | tg128 t/s (mean of the round medians; min..max of them) | pp512 t/s (same) | VRAM MiB | SM clock MHz (min..max sampled, incl. model loading) |
|---|---|---|---|---|---|
| littlebit-0.55 | 729 | 143.8 (143.6..144.1) | 1716 (1680..1740) | 1223 | 1365..2340 |
| bf16 | 3288 | 51.0 (51.0..51.0) | 3339 (3210..3554) | 3763 | 915..2340 |
| littlebit-0.55-q8_0head | 450 | 185.9 (185.7..186.0) | 1703 (1674..1744) | 945 | 1365..2325 |
| littlebit-0.55-q6_khead | 379 | 199.5 (199.3..199.8) | 1704 (1683..1721) | 873 | 1380..2325 |
| littlebit-0.1 | 646 | 157.3 (155.3..158.8) | 3368 (3278..3437) | 1133 | 795..2325 |
| littlebit-0.1-q8_0head | 368 | 212.9 (212.0..213.7) | 3550 (3516..3574) | 855 | 855..2325 |
| f16 | 3288 | 50.9 (50.9..50.9) | 4557 (4381..4878) | 3765 | 750..2325 |
| q4_k_m | 1056 | 132.7 (132.4..132.9) | 5628 (5533..5736) | 1517 | 1185..2325 |

### qwen3-0.6b-proxy

| variant | file MiB | tg128 t/s (mean of the round medians; min..max of them) | pp512 t/s (same) | VRAM MiB | SM clock MHz (min..max sampled, incl. model loading) |
|---|---|---|---|---|---|
| littlebit-0.55 | 642 | 235.6 (234.5..236.9) | 3551 (3367..3669) | 821 | 1095..2325 |
| bf16 | 1439 | 127.1 (126.9..127.4) | 9598 (9112..10050) | 1609 | 1320..2325 |

- gate littlebit-0.55 > bf16 (tg128, qwen3-0.6b and qwen3-1.7b): **PASS**
- gate littlebit-0.55 > f16 (tg128, qwen3-0.6b and qwen3-1.7b): **PASS**
- gate littlebit-0.55-q6_khead > bf16 (tg128, qwen3-0.6b and qwen3-1.7b): **PASS**
- gate littlebit-0.55-q6_khead > f16 (tg128, qwen3-0.6b and qwen3-1.7b): **PASS**
- gate littlebit-0.55-q8_0head > bf16 (tg128, qwen3-0.6b and qwen3-1.7b): **PASS**
- gate littlebit-0.55-q8_0head > f16 (tg128, qwen3-0.6b and qwen3-1.7b): **PASS**


## Reading the numbers

- **Gate (tg128, LittleBit 0.55 > bf16):** 0.6B 236.3 vs 127.5 (1.85x), 1.7B 143.8 vs 51.0 (2.82x), F16 head; the
  same holds against F16 dense (126.9, 50.9) and with either quantized head. `check_gate` on these results: True for
  every pairing (last lines above). No speed lever was needed; the 2A-S variant was not touched.
- **Target "on par with Q4_K_M":** with the F16 head LittleBit is 0.85x Q4_K_M on 0.6B (236 vs 277) and 1.08x on 1.7B
  (144 vs 133); with a Q8_0 head 1.08x / 1.40x, with a Q6_K head 1.15x / 1.50x. On 0.6B the F16 head (297 MiB of the
  642 MiB file, read for every token) is what keeps LittleBit below Q4_K_M, whose head is Q6_K.
- **Head type:** Q8_0 makes generation 1.26x (0.6B) / 1.29x (1.7B) faster than the F16 head, Q6_K 1.35x / 1.39x; VRAM
  drops by 138 / 174 MiB (0.6B) and 278 / 350 MiB (1.7B).
- **0.1 bpw** is not faster than 0.55 at these sizes: the rank floor of 128 per path leaves about half of the sign data
  (19 vs 35 MiB on 0.6B), and the time is dominated by the head and the per-launch cost of the 196 small LittleBit ops
  (docs/engine-changes.md, decode profile).
- **Prompt processing is slower than dense** (0.6B pp512 3525 vs 9961 bf16; 1.7B 1716 vs 3339): the prefill path runs
  float32 GEMMs on the CUDA cores against f16 / bf16 tensor cores (known, docs/engine-changes.md "CUDA: batches above 8
  tokens"; not part of the gate). At 0.1 bpw pp512 is about 1.8-2x faster than at 0.55 (less GEMM work).
- **Spread:** within a run the 10 repetitions of tg128 have a standard deviation of 0.1-5 tokens/s (largest for the
  fastest files); the round medians of tg128 stay within 2.3 % of each other for every file (1.5 % or less for all but
  the 1.7B 0.1-bpw file). pp512 drifts down within a run
  (power cap: e.g. 4224 -> 3150 tokens/s over 10 repetitions of the 0.6B LittleBit file in a trial run), so its median
  depends more on the run (round range up to 12 %).

## Perplexity delta of the head type

`llama-perplexity -f wiki.test.raw -ngl 99 -c 512 --chunks 64` (WikiText-2 raw test split, the first 64 chunks of 512
tokens, 32768 tokens; the same chunks for every file):

| file | F16 / bf16 head | Q8_0 head | Q6_K head |
|---|---|---|---|
| Qwen3-0.6B dense bf16 (head quantized with `llama-quantize --token-embedding-type / --output-tensor-type`) | 21.279 +- 0.552 | 21.292 (+0.06 %) | 21.359 (+0.38 %) |
| Qwen3-0.6B LittleBit 0.55, init-only | 2.175e7 | 2.167e7 (-0.4 %) | 2.174e7 (-0.04 %) |
| Qwen3-1.7B (proxy) LittleBit 0.55, init-only | 1.120e8 | 1.116e8 (-0.4 %) | 1.123e8 (+0.2 %) |

The dense row is the meaningful one: a Q8_0 head costs 0.06 % perplexity, a Q6_K head 0.38 %. (The +-0.55 printed by
llama-perplexity is the uncertainty of the absolute value on this sample; the three runs score the same tokens, so the
differences between them are much better determined than that, but their own uncertainty was not computed.) The
init-only LittleBit checkpoints are untrained (perplexity in the tens of millions, gibberish by design), so their
deltas only say that the quantized head does not change their output distribution much; they are not a quality
measurement. The LittleBit files' heads are quantized by the converter (`lb_to_gguf.py --head-type`), the dense ones
by llama-quantize; both use ggml's reference quantizers (the converter's are bit-identical to them,
tests/convert/test_head_type.py).

## Reproduce

```powershell
# checkpoints (Qwen3-0.6B from .cache/models/Qwen3-0.6B; 1.7B from the proxy directory, see the caveat)
.venv\Scripts\python.exe tools/dev/make_lb2_init.py .cache/models/Qwen3-0.6B .cache/ckpt/qwen3-0.6b-lb2-0.1 --bpw 0.1
.venv\Scripts\python.exe tools/dev/dequant_gguf_to_hf.py <Qwen3-1.7B-Q4_K_M.gguf> .cache/models/Qwen3-1.7B-proxy --side-files <dir with config.json + tokenizer of Qwen/Qwen3-1.7B>
.venv\Scripts\python.exe tools/dev/make_lb2_init.py .cache/models/Qwen3-1.7B-proxy .cache/ckpt/qwen3-1.7b-proxy-lb2-0.55 --bpw 0.55
# one conversion per file of bench/plans/phase1.json, e.g.
.venv\Scripts\python.exe -m tools.convert.lb_to_gguf .cache/ckpt/qwen3-0.6b-lb2-0.55 -o .cache/gguf/qwen3-0.6b-lb2-0.55-b1-q8_0head.gguf --head-type q8_0
.venv\Scripts\python.exe engine/convert_hf_to_gguf.py .cache/models/Qwen3-0.6B --outtype bf16 --outfile .cache/gguf/qwen3-0.6b-base-bf16.gguf
engine\build\bin\llama-quantize.exe .cache/gguf/qwen3-0.6b-base-bf16.gguf .cache/gguf/qwen3-0.6b-base-q4_k_m.gguf Q4_K_M
# benchmark and tables
.venv\Scripts\python.exe -m bench.run_bench --plan bench/plans/phase1.json --rounds 3 --out bench/results/phase1.json
.venv\Scripts\python.exe -m bench.run_bench --markdown bench/results/phase1.json
```
