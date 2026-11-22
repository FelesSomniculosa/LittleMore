"""Pure-logic tests of the speed-gate decision `bench.run_bench.check_gate` (no engine, no GPU: mocked numbers).

The hard gate of the project: LittleBit at 0.55 bits per weight
must generate tokens faster than the 16-bit dense model (bf16) of the same base model, for BOTH Qwen3-0.6B and
Qwen3-1.7B, measured on CUDA.  `check_gate` takes the results table the harness produces and answers that question.
These tests pin the decision rule: strictly faster, both models required, a missing or non-finite measurement fails.
"""
from __future__ import annotations

import math

import pytest

from bench.run_bench import GATE_MODELS, check_gate, gate_status


def _results(lb06: float | None, bf06: float | None, lb17: float | None, bf17: float | None,
             lb_key: str = "littlebit-0.55", dense_key: str = "bf16") -> dict:
    """A results table in the harness's shape: model -> variant -> {"tg128": tokens/s, ...}; None leaves an entry out."""
    res: dict = {}
    for model, lb, bf in (("qwen3-0.6b", lb06, bf06), ("qwen3-1.7b", lb17, bf17)):
        entry = {}
        if lb is not None:
            entry[lb_key] = {"tg128": lb, "pp512": 1000.0, "vram_mb": 800.0}
        if bf is not None:
            entry[dense_key] = {"tg128": bf, "pp512": 9000.0, "vram_mb": 1600.0}
        res[model] = entry
    return res


def test_gate_models_are_the_spec_models():
    """The gate is defined on exactly these two models."""
    assert GATE_MODELS == ("qwen3-0.6b", "qwen3-1.7b")


def test_passes_when_littlebit_faster_on_both():
    assert check_gate(_results(227.0, 120.0, 150.0, 60.0)) is True


@pytest.mark.parametrize("numbers", [
    (227.0, 120.0, 55.0, 60.0),    # 1.7B slower
    (100.0, 120.0, 150.0, 60.0),   # 0.6B slower
    (90.0, 120.0, 50.0, 60.0),     # both slower
], ids=["1.7b-slower", "0.6b-slower", "both-slower"])
def test_fails_when_slower_on_either(numbers: tuple[float, float, float, float]) -> None:
    assert check_gate(_results(*numbers)) is False


def test_equal_speed_is_not_faster():
    """'Faster' is strict: a tie does not pass."""
    assert check_gate(_results(120.0, 120.0, 150.0, 60.0)) is False


@pytest.mark.parametrize("missing", ["lb06", "bf06", "lb17", "bf17", "model17"])
def test_missing_measurement_fails(missing: str) -> None:
    """A gate that was not measured cannot pass."""
    args = dict(lb06=227.0, bf06=120.0, lb17=150.0, bf17=60.0)
    if missing == "model17":
        res = _results(**args)
        del res["qwen3-1.7b"]
    else:
        args[missing] = None
        res = _results(**args)
    assert check_gate(res) is False


@pytest.mark.parametrize("bad", [math.nan, math.inf, -1.0, 0.0])
def test_non_finite_or_non_positive_speed_fails(bad: float) -> None:
    """A broken measurement (NaN, inf, zero or negative tokens/s) never passes, on either side."""
    assert check_gate(_results(bad, 120.0, 150.0, 60.0)) is False
    assert check_gate(_results(227.0, 120.0, 150.0, bad)) is False


def test_other_keys_select_the_variant_and_baseline():
    """The same rule for another LittleBit variant (e.g. the q8_0 head) or another dense baseline (f16)."""
    res = _results(227.0, 120.0, 150.0, 60.0, lb_key="littlebit-0.55-q8_0head", dense_key="f16")
    assert check_gate(res, lb_key="littlebit-0.55-q8_0head", dense_key="f16") is True
    # with the default keys these entries are not found: not measured -> fails
    assert check_gate(res) is False


def test_extra_models_and_variants_are_ignored():
    res = _results(227.0, 120.0, 150.0, 60.0)
    res["qwen3-4b"] = {"littlebit-0.55": {"tg128": 1.0}, "bf16": {"tg128": 99.0}}
    res["qwen3-0.6b"]["q4_k_m"] = {"tg128": 500.0}
    assert check_gate(res) is True


# --- proxy weights (results of a base model rebuilt from a quantized GGUF; speed only, see bench/run_bench.py) -------

def _with_proxy_17b() -> dict:
    """The table of the Phase 1 run: Qwen3-0.6B on its original weights, Qwen3-1.7B only on proxy weights."""
    res = _results(227.0, 120.0, 150.0, 60.0)
    res["qwen3-1.7b-proxy"] = res.pop("qwen3-1.7b")
    return res


def test_proxy_results_do_not_pass_the_gate_by_default():
    """A model measured only on proxy weights counts as not measured for the formal gate."""
    assert check_gate(_with_proxy_17b()) is False


def test_proxy_results_pass_only_when_allowed_and_are_reported_provisional():
    res = _with_proxy_17b()
    assert check_gate(res, allow_proxy=True) is True
    assert gate_status(res) == "PASS (provisional: qwen3-1.7b on proxy weights)"


def test_original_weights_take_precedence_over_proxy():
    """With both, the original-weight result decides (a slower original run is not hidden by a faster proxy run)."""
    res = _results(227.0, 120.0, 55.0, 60.0)          # 1.7B original: LittleBit slower
    res["qwen3-1.7b-proxy"] = _results(227.0, 120.0, 150.0, 60.0)["qwen3-1.7b"]
    assert check_gate(res, allow_proxy=True) is False
    assert gate_status(res) == "FAIL"


def test_gate_status_formal_pass_and_fail():
    assert gate_status(_results(227.0, 120.0, 150.0, 60.0)) == "PASS"
    assert gate_status(_results(227.0, 120.0, 55.0, 60.0)) == "FAIL"
    res = _with_proxy_17b()
    res["qwen3-1.7b-proxy"]["littlebit-0.55"]["tg128"] = 50.0   # slower on the proxy too
    assert gate_status(res) == "FAIL"


# --- the committed Phase 1 results ---------------------------------------------------------------------------------

def test_committed_phase1_results_pass_the_formal_gate():
    """bench/results/phase1.json (the formal run): the gate passes on ORIGINAL weights for both models, without proxy
    results, and the run records a clean engine tree whose commit llama-bench also reports (build provenance)."""
    import json
    from pathlib import Path
    from bench.run_bench import aggregate
    raw = json.loads((Path(__file__).resolve().parents[2] / "bench" / "results" / "phase1.json").read_text(encoding="utf-8"))
    runs = [r for r in raw["runs"] if "error" not in r]
    agg = aggregate(runs)
    for model in GATE_MODELS:
        assert all(v["weights"] == "original" for v in agg[model].values()), model
    assert check_gate(agg) is True
    assert gate_status(agg) == "PASS"
    prov = raw["provenance"]
    assert prov["engine_dirty"] is False
    assert prov["engine_commit"][:9] in prov["llama_bench_version"]


def test_env_of_a_plan_entry_reaches_the_engine_programs():
    """A plan entry's "env" is merged into the environment of llama-bench (and CUDA 12.9 stays on PATH)."""
    from bench import run_bench as rb
    env = rb._env({"GGML_CUDA_LITTLEBIT_F32_PREFILL": "1"})
    assert env["GGML_CUDA_LITTLEBIT_F32_PREFILL"] == "1"
    assert "PATH" in env
    import os
    assert rb._env().get("GGML_CUDA_LITTLEBIT_F32_PREFILL") == os.environ.get("GGML_CUDA_LITTLEBIT_F32_PREFILL")  # no leak
