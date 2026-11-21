"""The fused-op graph, per model family, against the PyTorch reference.

For each of the four target families (Qwen3 dense, Llama, Gemma 3, Qwen3.5 hybrid DeltaNet) the test builds
a tiny random model, makes an init-only LittleBit-2 checkpoint, converts it with tools/convert/lb_to_gguf.py
(B1 signs), runs it on the CPU (`-ngl 0`) with llama.cpp's `llama-debug` and compares the last-position
logits with `reference.py` (fp32, CPU) on the very same token ids (the ids are read back from the engine, so
tokenization is not part of the comparison).

What is checked (see the constants below for the measured values the thresholds come from):
  * fused logits match the reference: max-abs below FUSED_MAX_ABS, same top-1 token;
  * the fused path is clearly closer to the reference than the old Q1_0 slow path is;
  * the comparison is shown to be able to fail: against a reference whose residual path is zeroed, or whose
    q/k rows are permuted (a RoPE-layout error), the same engine logits are far away;
  * biases of the converted linears are applied (random biases, every family that can carry them);
  * a bias the graph would ignore, and a converted linear nobody asks for, make the load FAIL with a clear
    message; a LoRA adapter aimed at a LittleBit linear is skipped with one warning.

Needs the engine build and, for the Qwen families, the cached Qwen3-0.6B tokenizer; skips otherwise.
Run: `pytest tests/engine/test_fused_graph_families.py -m engine_run -s`.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import _assets as A
import _tiny
import emulate_q8_slow_path as E
import reference as ref
from tools.convert import lb_to_gguf as lb

pytestmark = pytest.mark.engine_run

FAMILIES = _tiny.FAMILIES
# Token text fed to llama-debug.  The Qwen vocabulary is the real Qwen3 one; the tiny SentencePiece models
# (Llama, Gemma 3) know the words of the training corpus in _tiny.py.
PROMPT = {"qwen3": "The capital of France is", "qwen35": "The capital of France is",
          "llama": "the quick brown fox jumps over the lazy dog", "gemma3": "the quick brown fox jumps over the lazy dog"}

# --- thresholds, each tied to a measurement (tiny models, 5-37 tokens, CPU; logit rms is about 0.33) -------
# Fused vs fp32 reference, max-abs over the vocabulary: measured 1.9e-4 (Llama), 2.5e-4 (Qwen3.5), 4.0e-4
# (Qwen3), 7.0e-4 (Gemma 3, the worst). What remains is float32 summation order plus the engine's f16 KV
# cache. The limit is 3e-3: more than 4x the worst case, about 1% of the logit scale, and 3.6x below the
# smallest error of the Q1_0 slow path (1.08e-2, Llama), so it cannot pass the slow path's noise.
FUSED_MAX_ABS = 3e-3
# The slow path's max-abs error was 48x to 73x the fused one; require at least 10x.
SLOW_PATH_MIN_RATIO = 10.0
# Mutated references (no residual path, permuted q/k) were 148x to 6667x farther than the correct one; require 20x.
MUTATION_MIN_RATIO = 20.0


# Default tiny Gemma 3 of these tests: six layers in the real 5:1 pattern. The converter now writes the per-layer
# pattern key from `layer_types` (see test_gemma3_layer_patterns for other patterns); without that key the engine
# assumes 5:1, and the old two-layer default [sliding, full] ran as [sliding, sliding] (4.4e-1 logit error).
FAMILY_OVERRIDES = {"gemma3": dict(num_hidden_layers=6, layer_types=["sliding_attention"] * 5 + ["full_attention"])}


def _skip_reason(family: str) -> str | None:
    """Why this family cannot run here (engine binary or tokenizer missing), or None when it can."""
    if not (A.BIN / ("llama-debug" + A.EXE)).exists():
        return f"engine binary llama-debug not built in {A.BIN} (see docs/engine-changes.md)"
    if family in ("qwen3", "qwen35") and not _tiny.qwen_tokenizer_available():
        return f"Qwen3-0.6B tokenizer not cached at {_tiny.QWEN_TOK_DIR}"
    return None


@dataclass
class Case:
    """One tiny converted model: its HF dir, LittleBit checkpoint, B1 GGUF, prompt and scratch dir."""
    family: str
    hf: Path
    ckpt: Path
    gguf: Path
    prompt: str
    tmp: Path
    _model: nn.Module | None = None

    @property
    def model(self) -> nn.Module:
        """The reference model (fp32, CPU), built on first use."""
        if self._model is None:
            self._model = ref.build_model(str(self.hf), self.ckpt).float().eval()
        return self._model

    @property
    def head_dim(self) -> int:
        """Attention head size of the tiny test models (all families use 64)."""
        return 64


def make_case(family: str, tmp: Path, **over) -> Case:
    """Tiny model -> init-only checkpoint -> B1 GGUF, all in `tmp`."""
    reason = _skip_reason(family)
    if reason:
        pytest.skip(reason)
    over = {**FAMILY_OVERRIDES.get(family, {}), **over}
    ck = _tiny.make_tiny_checkpoint(family, tmp, bpw=2.0, residual=True, **over)
    gguf_path = tmp / f"{family}.gguf"
    lb.convert(ck, gguf_path)
    return Case(family, tmp / "hf", ck, gguf_path, PROMPT[family], tmp)


@pytest.fixture(scope="module", params=FAMILIES)
def case(request, tmp_path_factory) -> Case:
    """A converted tiny model per family, built once per module."""
    return make_case(request.param, tmp_path_factory.mktemp(request.param))


def engine(c: Case, gguf: Path | None = None, ngl: int = 0, extra: tuple = ()) -> tuple[np.ndarray, list[int]]:
    """Last-position logits and token ids of `gguf` (default: the case's B1 GGUF)."""
    g = gguf or c.gguf
    return A.engine_logits(g, c.prompt, ngl, extra, out_dir=c.tmp / "logits" / f"{g.stem}-ngl{ngl}")


def ref_logits(c: Case, ids: list[int], **mutations) -> np.ndarray:
    """fp32 reference logits (no activation quantization, no f16 KV rounding), optionally mutated."""
    E.emulate(c.model, quantize=False, kv_f16=False, head_dim=c.head_dim, **mutations)
    return E.last_logits(c.model, ids)


def metrics(a: np.ndarray, b: np.ndarray) -> dict:
    """Compare logits `a` against reference `b`: max-abs and rms difference, correlation, same top-1 token, logit rms."""
    d = a - b
    return dict(max_abs=float(np.abs(d).max()), rms=float(np.sqrt(np.mean(d * d))),
                corr=float(np.corrcoef(a, b)[0, 1]), top1=bool(a.argmax() == b.argmax()),
                scale=float(np.sqrt(np.mean(b * b))))


def fmt(m: dict) -> str:
    """One-line text of `metrics()` output for assertion messages."""
    return (f"max-abs={m['max_abs']:.3e} rms={m['rms']:.3e} corr={m['corr']:.8f} "
            f"top1={'same' if m['top1'] else 'DIFF'} (logit rms {m['scale']:.3f})")


# ------------------------------------------------------------------------------------------ the main proof

def test_fused_matches_reference(case):
    """Fused-graph logits (CPU) equal the fp32 reference on the same token ids."""
    logits, ids = engine(case)
    want = ref_logits(case, ids)
    m = metrics(logits, want)
    print(f"[{case.family}] {len(ids)} tokens: fused vs reference: {fmt(m)}")
    assert m["max_abs"] <= FUSED_MAX_ABS
    assert m["top1"]


def test_fused_closer_than_slow_path(case):
    """The Q1_0 slow path (Q8_0 activation noise) is clearly farther from the reference than the fused path."""
    fused, ids = engine(case)
    q1 = case.tmp / f"{case.family}-q1_0.gguf"
    A.b1_to_q1_0_gguf(case.gguf, q1)
    slow, ids_slow = engine(case, q1)
    assert ids == ids_slow
    want = ref_logits(case, ids)
    mf, ms = metrics(fused, want), metrics(slow, want)
    print(f"[{case.family}] fused {fmt(mf)}\n[{case.family}] slow  {fmt(ms)}  ratio={ms['max_abs'] / mf['max_abs']:.1f}x")
    assert ms["max_abs"] >= SLOW_PATH_MIN_RATIO * mf["max_abs"]


def test_mutations_are_detected(case):
    """The comparison can fail: a reference without the residual path, or with permuted q/k rows, is far away."""
    logits, ids = engine(case)
    good = metrics(logits, ref_logits(case, ids))["max_abs"]
    for name, kw in (("no-residual", dict(drop_residual=True)), ("permuted-qk", dict(permute_qk=True))):
        bad = metrics(logits, ref_logits(case, ids, **kw))["max_abs"]
        print(f"[{case.family}] {name}: max-abs {bad:.3e} vs correct {good:.3e} ({bad / good:.0f}x)")
        assert bad >= MUTATION_MIN_RATIO * good
    ref_logits(case, ids)  # leave the shared model unmutated


GEMMA_PATTERNS = {
    "2-layer-sliding-full": ["sliding_attention", "full_attention"],
    "4-layer-alternating": ["sliding_attention", "full_attention"] * 2,
    "3-layer-full-first": ["full_attention", "sliding_attention", "sliding_attention"],
    "4-layer-all-sliding": ["sliding_attention"] * 4,
}


@pytest.mark.parametrize("name", list(GEMMA_PATTERNS))
def test_gemma3_layer_patterns(name, tmp_path):
    """Non-default Gemma 3 layer patterns reach the engine (key written by the converter) and match the reference.

    37 tokens against a window of 16, so sliding and global layers differ in what they attend to.
    """
    types = GEMMA_PATTERNS[name]
    c = make_case("gemma3", tmp_path, num_hidden_layers=len(types), layer_types=types)
    import gguf as G
    r = G.GGUFReader(str(c.gguf))
    assert [bool(x) for x in r.get_field("gemma3.attention.sliding_window_pattern").contents()] ==         [t == "sliding_attention" for t in types]
    logits, ids = engine(c)
    m = metrics(logits, ref_logits(c, ids))
    print(f"[gemma3 {name}] {len(ids)} tokens: {fmt(m)}")
    assert m["max_abs"] <= FUSED_MAX_ABS and m["top1"]


# ------------------------------------------------------------------------------------------------- biases

class _Biased(nn.Module):
    """A reference LBLinear plus the bias the base model had (reference.py's LBLinear drops it)."""

    def __init__(self, inner: nn.Module, bias: torch.Tensor):
        super().__init__()
        self.inner, self.bias = inner, bias

    def forward(self, x):
        """Run the wrapped LittleBit linear, then add the bias."""
        return self.inner(x) + self.bias.to(x.dtype)


def _randomise_biases(hf: Path, seed: int = 1) -> list[str]:
    """Give every layer bias of the tiny HF model at `hf` a random value (HF initialises them to zero)."""
    from safetensors.torch import load_file, save_file
    f = hf / "model.safetensors"
    t = load_file(f)
    g = torch.Generator().manual_seed(seed)
    names = [k for k in t if k.endswith(".bias") and ".layers." in k]
    for k in names:
        t[k] = torch.randn(t[k].shape, generator=g) * 0.5
    save_file(t, f, metadata={"format": "pt"})
    return names


BIAS_OVERRIDES = {"qwen3": dict(attention_bias=True), "llama": dict(attention_bias=True, mlp_bias=True),
                  "gemma3": dict(attention_bias=True), "qwen35": dict(attention_bias=True)}


@pytest.mark.parametrize("family", FAMILIES)
def test_biases_are_applied(family, tmp_path):
    """q/k/v/o (and for Llama gate/up/down) biases on converted linears reach the output.

    A reference that adds the same biases is the target; a reference that forgets them is far away.
    """
    reason = _skip_reason(family)
    if reason:
        pytest.skip(reason)
    from make_lb2_init import make_checkpoint
    hf = _tiny.make_tiny_model(family, tmp_path / "hf", **FAMILY_OVERRIDES.get(family, {}), **BIAS_OVERRIDES[family])
    names = _randomise_biases(hf)
    assert names, "the tiny model has no layer biases to test"
    ck = tmp_path / "ckpt"
    make_checkpoint(str(hf), ck, 2.0, use_itq=False, residual=True, seed=0)
    out = tmp_path / "b.gguf"
    lb.convert(ck, out)
    c = Case(family, hf, ck, out, PROMPT[family], tmp_path)
    logits, ids = engine(c)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ref.build_model(str(hf), ck).float().eval()
    from safetensors.torch import load_file
    ckt = load_file(ck / "model.safetensors")
    n = 0
    for name in names:
        mod_name = name[: -len(".bias")]
        parent_name, _, child = mod_name.rpartition(".")
        parent = model.get_submodule(parent_name)
        mod = getattr(parent, child)
        assert isinstance(mod, ref.LBLinear), mod_name
        setattr(parent, child, _Biased(mod, ckt[name].float()))
        n += 1
    c._model = model
    want = ref_logits(c, ids)
    m = metrics(logits, want)
    print(f"[{family}] {n} biases: fused vs reference-with-biases: {fmt(m)}")
    assert m["max_abs"] <= FUSED_MAX_ABS and m["top1"]
    # the same engine output against a reference that ignores the biases must be far away
    for name in names:
        parent_name, _, child = name[: -len(".bias")].rpartition(".")
        parent = model.get_submodule(parent_name)
        setattr(parent, child, getattr(parent, child).inner)
    nobias = metrics(logits, ref_logits(c, ids))
    print(f"[{family}] against a reference without biases: {fmt(nobias)}")
    assert nobias["max_abs"] >= MUTATION_MIN_RATIO * m["max_abs"]


# ------------------------------------------------------------------------- loud failures and the LoRA warning

@pytest.fixture(scope="module")
def llama_case(tmp_path_factory) -> Case:
    """A converted tiny Llama shared by the load-failure and LoRA tests."""
    return make_case("llama", tmp_path_factory.mktemp("llama_fail"))


def _run_debug(gguf: Path, *args: str, ngl: int = 0) -> subprocess.CompletedProcess:
    """Run llama-debug; a load that fails early can lose its log lines (the asynchronous log thread is not
    flushed on that exit path, seen as exit code 1 with empty output), so an empty failing run is retried."""
    for _ in range(8):
        r = A._run([A.BIN / ("llama-debug" + A.EXE), "-m", gguf, "-p", "the quick brown fox", "-ngl", ngl, *args])
        if r.returncode == 0 or r.stdout.strip():
            break
    return r


def test_unconsumed_linear_fails_load(llama_case):
    """A converted linear the architecture never asks for aborts the load with its name, not silently."""
    import gguf as G
    bad = llama_case.tmp / "extra-linear.gguf"
    A.copy_gguf(llama_case.gguf, bad, extra=[("blk.0.bogus_proj.lb_v", np.zeros((128, 16), np.uint8), G.GGMLQuantizationType.B1)])
    r = _run_debug_failing(bad, "blk.0.bogus_proj")
    print(r.stdout[-600:])
    assert r.returncode != 0
    assert "blk.0.bogus_proj" in r.stdout and "LittleBit linear" in r.stdout


def test_ignored_bias_fails_load(tmp_path_factory):
    """A bias next to a converted linear that the architecture would ignore aborts the load with a clear message.

    Qwen3 has no FFN biases; the file below carries one for ffn_up, which the graph would silently drop.
    """
    import gguf as G
    c = make_case("qwen3", tmp_path_factory.mktemp("q3_bias"))
    bad = c.tmp / "ignored-bias.gguf"
    A.copy_gguf(c.gguf, bad, extra=[("blk.0.ffn_up.bias", np.ones(512, np.float32), G.GGMLQuantizationType.F32)])
    r = _run_debug_failing(bad, "bias 'blk.0.ffn_up.bias'")
    print(r.stdout[-600:])
    assert r.returncode != 0
    assert "bias 'blk.0.ffn_up.bias'" in r.stdout and "silently give wrong output" in r.stdout


def test_hadamard_metadata_on_littlebit_linear_fails_load(llama_case):
    """prism.hadamard metadata naming a LittleBit linear is refused at load, not silently ignored.

    The Hadamard fold is keyed by dense weight names; a LittleBit linear has none, so the loader cannot bind it.
    (prec_policy is the same kind of metadata and resolves names the same way; it was only code-read.)
    """
    import gguf as G
    T, A_ = G.GGUFValueType, G.GGUFValueType.ARRAY
    bad = llama_case.tmp / "hadamard.gguf"
    A.copy_gguf(llama_case.gguf, bad, extra_kv=[
        ("prism.hadamard.version", 1, T.UINT32), ("prism.hadamard.tied_output", False, T.BOOL),
        ("prism.hadamard.block_size", 128, T.UINT32),
        ("prism.hadamard.transform", "normalized-sylvester-walsh-hadamard", T.STRING),
        ("prism.hadamard.axis", "input-last-dimension", T.STRING),
        ("prism.hadamard.sign_mode", "identity", T.STRING),
        ("prism.hadamard.weight_names", ["blk.0.attn_q.weight"], A_, T.STRING)])
    ok = _run_debug(llama_case.gguf)
    assert ok.returncode == 0, ok.stdout[-800:]          # control: the same file without the metadata loads
    r = _run_debug(bad)
    print(r.stdout[-500:])
    assert r.returncode != 0
    # llama-debug's asynchronous log thread sometimes loses the error line when the load fails (upstream
    # common/log.cpp: "logs may be lost if not flushed before the program exits"); when the line is there it must be this one.
    if "error loading model" in r.stdout:
        assert "prism.hadamard weight not found: blk.0.attn_q.weight" in r.stdout, r.stdout[-800:]


def _run_debug_failing(gguf: Path, text: str) -> subprocess.CompletedProcess:
    """Run a load that must fail; retry while the error line is missing (async log thread, see test_hadamard...)."""
    for _ in range(12):
        r = _run_debug(gguf)
        if r.returncode == 0 or text in r.stdout:
            break
    return r


def _write_lora(path: Path, arch: str, n_in: int, n_out: int, rank: int = 4) -> None:
    """A rank-4 LoRA adapter GGUF for `blk.0.attn_q.weight` (random, large enough to matter if it were applied)."""
    import gguf as G
    w = G.GGUFWriter(str(path), arch)
    w.add_string("general.type", "adapter")
    w.add_string("adapter.type", "lora")
    w.add_float32("adapter.lora.alpha", 16.0)
    rng = np.random.default_rng(0)
    w.add_tensor("blk.0.attn_q.weight.lora_a", rng.standard_normal((rank, n_in)).astype(np.float32))
    w.add_tensor("blk.0.attn_q.weight.lora_b", rng.standard_normal((n_out, rank)).astype(np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(progress=False); w.close()


def test_lora_on_littlebit_linear_is_skipped_with_one_warning(llama_case):
    """An adapter aimed at a LittleBit linear loads, warns once, and does not change the output."""
    ad = llama_case.tmp / "adapter.gguf"
    _write_lora(ad, "llama", 256, 256)
    base = _run_debug(llama_case.gguf)
    assert base.returncode == 0, base.stdout[-800:]
    withl = _run_debug(llama_case.gguf, "--lora", ad)
    assert withl.returncode == 0, withl.stdout[-800:]
    warns = re.findall(r"LoRA adapters are not supported on LittleBit linears", withl.stdout)
    print(f"{len(warns)} warning(s)")
    assert len(warns) == 1
    la, _ = engine(llama_case)
    lb_, _ = engine(llama_case, extra=("--lora", str(ad)))
    assert np.array_equal(la, lb_), "the adapter must not change the output of a LittleBit linear"
