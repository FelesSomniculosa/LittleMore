# bench: speed benchmarks and the speed gate

This folder holds the speed measurements of LittleMore and the code that makes them.

## Files

| Path | Content |
|---|---|
| `run_bench.py` | harness around llama.cpp's `llama-bench` (`run_bench`, `run_plan`), the speed gate (`check_gate`) and the Markdown tables (`render_markdown`) |
| `plans/phase1.json` | the list of GGUF files measured for Phase 1 (model, variant name, file) |
| `results/phase1.json` | raw results of the Phase 1 run: every repetition of every round, GPU memory, SM clock |
| `results/phase1.md` | the Phase 1 results, the gate decision and how they were made |
| `../tests/bench/test_check_gate.py` | tests of the gate rule (pure logic, mocked numbers) |

## The gate

From the design spec ("Fast"): on CUDA, LittleBit at 0.55 bits per weight must generate tokens faster than the
16-bit dense model (bf16) of the same base model, for both Qwen3-0.6B and Qwen3-1.7B, on the RTX 4050 Laptop GPU of
this project. `check_gate(results)` returns True only if, for both models, the LittleBit file's `tg128` (tokens per
second generating 128 tokens) is strictly larger than the bf16 file's; a missing or broken measurement fails. The
default LittleBit file has the 16-bit (F16) embedding / output head, which is the configuration correctness is measured
in; the same rule is also reported for the files with a Q8_0 / Q6_K head (converter `--head-type`) and against F16
instead of bf16.

Proxy weights: a result measured on a base model rebuilt from a quantized GGUF (`tools/dev/dequant_gguf_to_hf.py`:
right shapes and types, other weight values, valid for speed only) is stored under `<model>-proxy` with
`"weights": "proxy"` (the plan entry carries both; `run_plan` refuses a mismatch). `check_gate` ignores such results
unless called with `allow_proxy=True`, and then only for a model without an original-weight result; `gate_status` and
the generated tables print such a pass as "PASS (provisional: <model> on proxy weights)".

## How a number is made

- `run_bench(model_gguf, backend, n_gpu_layers)` runs `engine/build/bin/llama-bench -m <file> -ngl <n> -dev <device>
  -o json` twice: generation of 128 tokens (`tg128`, `-p 0 -n 128 -r 10`) and then prompt processing of 512 tokens
  (`pp512`, `-p 512 -n 0 -r 10`), both in tokens per second, default settings otherwise (flash attention auto, F16 KV
  cache, batch 2048 / ubatch 512). Generation comes first because, measured on this GPU, a tg128 right after pp512 in the
  same llama-bench run starts 10-20 % slow while the GPU recovers from the power-capped prompt phase. llama-bench runs
  one untimed pass per test before its 10 timed repetitions; all are kept, and the reported value is their median (the
  first repetition can still be a few percent slow).
- Warm-up: this laptop GPU idles at 1605 MHz and needs about a second of load to reach its boost clock (2040-2340 MHz),
  and drops back while the CPU loads a model. So right before the measured run the harness runs short generations of
  the same model for at least 3 s, and it samples the SM clock with nvidia-smi during the measured run (stored as
  min / median / max) so a slow run can be recognised.
- GPU memory (`vram_mb`): the peak of nvidia-smi's `memory.used` during the run minus the value right before it, i.e.
  what the benchmark process allocated (weights, KV cache, compute buffers, CUDA context).
- Rounds: `run_plan` runs the whole list once per round and repeats the rounds (A B C ... A B C ...), so drifts of the
  machine (clock, temperature, the DRAM bandwidth, which varies between 128 and 188 GB/s from run to run on this GPU)
  hit every file alike. The tables give the mean of the round medians and the range of the round medians.

## Reproduce

From the repository root, with the engine built (docs/development.md) and the GGUF files of the plan made
(bench/results/phase1.md lists the commands):

```powershell
.venv\Scripts\python.exe -m bench.run_bench --plan bench/plans/phase1.json --rounds 3 --out bench/results/phase1.json
.venv\Scripts\python.exe -m bench.run_bench --markdown bench/results/phase1.json
.venv\Scripts\python.exe -m pytest tests/bench -q
```

Every number depends on the GPU, its clocks, power limit and temperature, the driver, and the CPU (llama.cpp's host
side matters at these speeds); compare files only within one run of the plan.
