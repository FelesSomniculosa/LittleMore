"""The reference model of the parity tests must be the PLAIN reference (tools/dev/reference.py's own
`LBLinear.forward`), whatever an earlier test did to the shared, cached object.

Background: `emulate_q8_slow_path.emulate` patches every `LBLinear` of a model in place (an instance-level `forward`,
plus flags such as `_emu_drop_residual`) and replaces transformers' Qwen3 RoPE function process-wide. The 0.6B
reference is built once per test session (`tests/smoke/_assets.py`), so an emulation or mutation one test applies is
still there for the next one. These tests check, on a small synthetic model (fast, no checkpoint needed):

* `unemulate` undoes `emulate` exactly: no instance `forward` is left, the output is bit-identical to the model
  before any emulation, and the RoPE function is the original again;
* `assert_plain` refuses a model with ANY patched module `forward` (not only one `emulate` made) or patched RoPE;
* `_assets.reference_model()` returns the plain model after an earlier caller emulated it with a mutation;
* the no-quantization emulation (`quantize=False, kv_f16=False`, the reset used before) computes bit-identical
  outputs to `LBLinear.forward`, so switching the reference back to the plain path changes no number.

Run with `pytest tests/dev`.
"""
from __future__ import annotations

import types

import pytest
import torch
from torch import nn

import emulate_q8_slow_path as E
import reference as ref


def _lblinear(out_f: int, in_f: int, r: int, gen: torch.Generator, residual: bool) -> ref.LBLinear:
    """A random LBLinear (signs +/-1, scales in 0.5..1.5), with or without the residual path."""
    def path():
        return (torch.where(torch.rand(out_f, r, generator=gen) < 0.5, -1.0, 1.0),
                torch.where(torch.rand(r, in_f, generator=gen) < 0.5, -1.0, 1.0),
                torch.rand(1, out_f, generator=gen) + 0.5, torch.rand(1, r, generator=gen) + 0.5,
                torch.rand(1, r, generator=gen) + 0.5, torch.rand(1, in_f, generator=gen) + 0.5)
    return ref.LBLinear(*path(), residual=path() if residual else None)


def _model() -> tuple[nn.Module, torch.Tensor]:
    """Two LBLinear layers (LittleBit-2, i.e. with residual, then without) and a fixed input."""
    gen = torch.Generator().manual_seed(1234)
    model = nn.Sequential(_lblinear(48, 64, 40, gen, True), nn.SiLU(), _lblinear(32, 48, 24, gen, False))
    return model.eval(), torch.randn(3, 64, generator=gen)


@pytest.fixture(autouse=True)
def _restore_rope():
    """`emulate(kv_f16=True)` replaces a process-wide function: put the original back after every test, also after a
    failure, so no later test in the session sees a patched RoPE."""
    yield
    if E._ORIG_ROPE is not None:
        E._patch_rope(False)


def _rope():
    """transformers' current Qwen3 apply_rotary_pos_emb (the global `emulate(kv_f16=True)` replaces)."""
    from transformers.models.qwen3 import modeling_qwen3 as mq
    return mq.apply_rotary_pos_emb


@torch.no_grad()
def test_unemulate_restores_the_plain_forward_exactly() -> None:
    model, x = _model()
    plain = model(x)
    E.assert_plain(model)                                   # a fresh model is plain
    original_rope = _rope()

    E.emulate(model, drop_residual=True)                    # the default emulation (Q8_0 + f16 RoPE) + a mutation
    emulated = model(x)
    assert not torch.equal(emulated, plain)                 # control: the emulation really changes the output
    assert _rope() is not original_rope                     # ... and replaced the RoPE function
    with pytest.raises(AssertionError, match="patched forward"):
        E.assert_plain(model)

    assert E.unemulate(model) is model
    assert all("forward" not in m.__dict__ for m in model.modules())
    assert not any(k.startswith("_emu_") for m in model.modules() for k in m.__dict__)
    assert _rope() is original_rope
    assert torch.equal(model(x), plain)                     # bit-identical to the model before any emulation
    E.assert_plain(model)
    E.unemulate(model)                                      # idempotent, and safe on a plain model
    assert torch.equal(model(x), plain)


@torch.no_grad()
def test_assert_plain_sees_patches_not_made_by_emulate() -> None:
    model, _ = _model()
    act = model[1]
    act.forward = types.MethodType(lambda self, t: t * 2, act)  # a monkey patch of a non-LBLinear module
    with pytest.raises(AssertionError, match="patched forward"):
        E.assert_plain(model)
    del act.forward
    E.assert_plain(model)

    E.emulate(model, quantize=False, kv_f16=True)           # RoPE patched, then forward undone by hand only
    for m in model.modules():
        m.__dict__.pop("forward", None)
    with pytest.raises(AssertionError, match="apply_rotary_pos_emb"):
        E.assert_plain(model)
    E.unemulate(model)
    E.assert_plain(model)


@torch.no_grad()
def test_no_quantization_emulation_is_bit_identical_to_the_plain_forward() -> None:
    """The reset used before (`emulate(quantize=False, kv_f16=False)`, through `_forward`) and the plain
    `LBLinear.forward` give the same bits, so the perplexity reference numbers do not change with the switch."""
    model, x = _model()
    plain = model(x)
    E.emulate(model, quantize=False, kv_f16=False)
    assert all("forward" in m.__dict__ for m in model.modules() if isinstance(m, ref.LBLinear))  # really _forward
    assert torch.equal(model(x), plain)
    E.unemulate(model)


@torch.no_grad()
def test_reference_model_is_plain_after_an_earlier_emulation(monkeypatch) -> None:
    """`_assets.reference_model()` on the shared object an earlier test left emulated with a mutation (here the small
    model stands in for the cached 0.6B one): plain `LBLinear.forward` again, same output as before the emulation."""
    import _assets as A
    model, x = _model()
    plain = model(x)
    monkeypatch.setattr(A, "_reference_model_built", lambda: model)
    E.emulate(model, drop_residual=True)                    # what tests/engine's residual control leaves behind
    got = A.reference_model()
    assert got is model
    assert all("forward" not in m.__dict__ for m in got.modules())
    assert torch.equal(got(x), plain)
    # and the plain no-kwargs reference_logits path uses it as is (no emulation applied on top)
    calls = []
    monkeypatch.setattr(E, "emulate", lambda *a, **k: calls.append(k))
    monkeypatch.setattr(E, "last_logits", lambda m, ids: m)
    assert A.reference_logits([1, 2]) is model and calls == []
    A.reference_logits([1, 2], quantize=False)
    assert calls == [{"quantize": False}]
