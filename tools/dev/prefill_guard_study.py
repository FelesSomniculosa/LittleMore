"""Model-level GPU error of LittleBit Qwen3-0.6B on long prompts: spread over correct paths, mutations, attribution.

What
    The measurement behind the coarse prefill guards of tests/engine/test_fused_graph_qwen3_06b.py
    (test_gpu_error_is_within_dense_floor_factor) and the table "These model-level GPU guards are coarse" in
    docs/engine-changes.md.  Every run compares last-position logits from `llama-debug` with the fp32 references
    (tools/dev/reference.py for LittleBit, transformers for the dense model), max-abs over the vocabulary, on 5 prompts of
    22..62 tokens (all above the 16-token chunk limit of the CUDA LittleBit op, so a normal build runs the prefill path).
Why
    The init-only LittleBit model amplifies tiny rounding differences far more than the dense model, so a model-level
    threshold must be judged against the spread of CORRECT runs (other prompts, the other legitimate CUDA path) and
    against deliberately broken builds (mutations), not against one lucky prompt.
Usage (from the repository root, engine built, Qwen3-0.6B assets cached as for tests/engine)
    python tools/dev/prefill_guard_study.py measure <tag> <out.jsonl> [--full]
        LittleBit at -ngl 99, default and exact configuration, per prompt; with --full also the dense model at -ngl 99
        and LittleBit on the pure CPU (-dev none, default configuration).  Run with --full once on the build as
        committed, then without --full on each temporary build (tools/dev/prefill_guard_study.sh makes them).
    python tools/dev/prefill_guard_study.py cpu <out.jsonl>
        pure CPU (-dev none): flash attention on / off x KV cache f16 / f32, LittleBit and dense (the attribution).
    python tools/dev/prefill_guard_study.py table <measure.jsonl>
        the summary table (ranges over the prompts) and how many runs each guard passes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in ("tests/engine", "tests/smoke", "tests/convert", "tools/dev", "."):
    if str(ROOT / _p) not in sys.path:
        sys.path.insert(0, str(ROOT / _p))

EXACT = ("-fa", "off", "-ctk", "f32", "-ctv", "f32")


def prompts() -> list[str]:
    """The 5 prompts (44, 22, 33, 48, 62 tokens with the Qwen3 tokenizer); the first is the test's PROMPT_PREFILL."""
    import _assets as A
    return [
        A.PROMPT_PREFILL,
        "The quick brown fox jumps over the lazy dog while the farmer watches from the old wooden porch of his house.",
        "Water boils at one hundred degrees Celsius at sea level, but at the top of a high mountain it boils at a lower "
        "temperature because the air pressure is lower there.",
        "In the year 1492, Columbus sailed across the Atlantic Ocean with three ships, hoping to find a new route to "
        "Asia, and instead he reached the islands of the Caribbean, which changed the history of the world for centuries.",
        "Machine learning models are trained on large amounts of data. They learn patterns that allow them to make "
        "predictions about new examples. A model that has learned well generalizes to data it has never seen before, "
        "while a model that has only memorized its training set does not. Engineers therefore keep a separate test set.",
    ]


def measure(tag: str, out: Path, full: bool) -> None:
    """Append one JSON line per prompt to `out` (see the module docstring)."""
    import _assets as A
    from test_fused_graph_families import metrics
    b1, dense = A.ensure_b1_gguf(), A.ensure_dense_floor_gguf()
    with out.open("a", encoding="utf-8") as f:
        for pi, prompt in enumerate(prompts()):
            rec: dict = {"tag": tag, "prompt": pi}
            for cfg, extra in (("default", ()), ("exact", EXACT)):
                lb, ids = A.engine_logits(b1, prompt, 99, extra)
                rec["n_tok"] = len(ids)
                rec[f"lb_gpu_{cfg}"] = metrics(lb, A.reference_logits(ids))["max_abs"]
                if full:
                    dn, _ = A.engine_logits(dense, prompt, 99, extra)
                    rec[f"dense_gpu_{cfg}"] = metrics(dn, A.dense_reference_logits(ids))["max_abs"]
            if full:
                cpu, _ = A.engine_logits(b1, prompt, 0, ("-dev", "none"))
                rec["lb_cpu_default"] = metrics(cpu, A.reference_logits(ids))["max_abs"]
            print(json.dumps(rec), flush=True)
            f.write(json.dumps(rec) + "\n")


def cpu(out: Path) -> None:
    """Pure CPU, flash attention x KV type, LittleBit and dense; one JSON line per prompt."""
    import _assets as A
    from test_fused_graph_families import metrics
    cfgs = {"fa_on_f16kv": ("-fa", "on"), "fa_on_f32kv": ("-fa", "on", "-ctk", "f32", "-ctv", "f32"),
            "fa_off_f16kv": ("-fa", "off"), "fa_off_f32kv": EXACT}
    b1, dense = A.ensure_b1_gguf(), A.ensure_dense_floor_gguf()
    with out.open("w", encoding="utf-8") as f:
        for pi, prompt in enumerate(prompts()):
            rec: dict = {"prompt": pi}
            for name, extra in cfgs.items():
                lb, ids = A.engine_logits(b1, prompt, 0, ("-dev", "none", *extra))
                dn, _ = A.engine_logits(dense, prompt, 0, ("-dev", "none", *extra))
                rec["n_tok"] = len(ids)
                rec[f"lb {name}"] = metrics(lb, A.reference_logits(ids))["max_abs"]
                rec[f"dense {name}"] = metrics(dn, A.dense_reference_logits(ids))["max_abs"]
            print(json.dumps(rec), flush=True)
            f.write(json.dumps(rec) + "\n")


def table(path: Path) -> None:
    """Ranges per run tag and the pass counts of the guards (the base tag must have been measured with --full)."""
    recs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    base = {r["prompt"]: r for r in recs if "dense_gpu_default" in r}
    guards = {
        "dense x2, default": lambda r, b: r["lb_gpu_default"] <= 2 * b["dense_gpu_default"],
        "dense x2, exact": lambda r, b: r["lb_gpu_exact"] <= 2 * b["dense_gpu_exact"],
        "pure CPU x2, default (the B1 default-config guard)": lambda r, b: r["lb_gpu_default"] <= 2 * b["lb_cpu_default"],
    }
    rng = lambda xs: f"{min(xs):.3g}..{max(xs):.3g}"  # noqa: E731
    for tag in dict.fromkeys(r["tag"] for r in recs):
        rs = [r for r in recs if r["tag"] == tag]
        b = [base[r["prompt"]] for r in rs]
        print(f"{tag}: LB default {rng([r['lb_gpu_default'] for r in rs])}, exact {rng([r['lb_gpu_exact'] for r in rs])}; "
              f"/dense default {rng([r['lb_gpu_default'] / x['dense_gpu_default'] for r, x in zip(rs, b)])}, "
              f"exact {rng([r['lb_gpu_exact'] / x['dense_gpu_exact'] for r, x in zip(rs, b)])}; "
              f"GPU/CPU default {rng([r['lb_gpu_default'] / x['lb_cpu_default'] for r, x in zip(rs, b)])}")
        for name, g in guards.items():
            print(f"    {name}: {sum(g(r, x) for r, x in zip(rs, b))}/{len(rs)} pass")


def main(argv: list[str]) -> int:
    """Command-line entry (see the module docstring)."""
    if len(argv) >= 3 and argv[0] == "measure":
        measure(argv[1], Path(argv[2]), "--full" in argv)
    elif len(argv) == 2 and argv[0] == "cpu":
        cpu(Path(argv[1]))
    elif len(argv) == 2 and argv[0] == "table":
        table(Path(argv[1]))
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
