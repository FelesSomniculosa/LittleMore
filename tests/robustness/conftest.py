"""Fixtures and opt-in rule of the robustness campaign (tests/robustness).

These tests are slow and heavy (hundreds of child processes, GPU memory pressure, sanitizers), so they are opt-in:
they are not in pytest.ini's `testpaths`, carry the `robust` marker, and are skipped unless the run selects that
marker (`pytest tests/robustness -m robust`) or sets LITTLEMORE_RUN_ROBUST=1. The ones that need the community
Qwen3-8B model also need LITTLEMORE_RUN_8B=1 (as tests/e2e/test_community_8b.py). Nothing here downloads anything:
a missing model or binary is a skip that says what is missing.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (HERE, ROOT / "python", ROOT / "tests" / "smoke", ROOT / "tests" / "convert", ROOT / "tools" / "dev", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

TINY = ROOT / ".cache" / "gguf" / "tiny-llama-lb2-chatml.gguf"
QWEN06 = ROOT / ".cache" / "gguf" / "qwen3-0.6b-lb2-0.55-b1.gguf"
QWEN06_F32HEAD = ROOT / ".cache" / "gguf" / "qwen3-0.6b-lb2-0.55-b1-f32head.gguf"
QWEN06_HF = ROOT / ".cache" / "models" / "Qwen3-0.6B"
QWEN06_CKPT = ROOT / ".cache" / "ckpt" / "qwen3-0.6b-lb2-0.55"
GGUF_8B = ROOT / ".cache" / "gguf" / "littlebit-qwen3-8b-0.55bpw-b1.gguf"
CKPT_8B = ROOT / ".cache" / "models" / "littlebit-qwen3-8b-0.55bpw"
ENGINE_BIN = ROOT / "engine" / "build" / "bin"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "robust: opt-in robustness campaign (tests/robustness); select with -m robust")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip every robustness test unless the run opted in (see the module docstring)."""
    selected = "robust" in (config.getoption("-m") or "") or os.environ.get("LITTLEMORE_RUN_ROBUST") == "1"
    for item in items:
        if HERE in Path(str(item.fspath)).resolve().parents:
            item.add_marker(pytest.mark.robust)
            if not selected:
                item.add_marker(pytest.mark.skip(reason="robustness campaign: run with -m robust or LITTLEMORE_RUN_ROBUST=1"))


@pytest.fixture(scope="session")
def lib_path() -> Path:
    """The built littlemore shared library, or skip."""
    from littlemore import _lib
    try:
        return _lib.find_library()
    except _lib.LibraryNotFoundError as e:
        pytest.skip(f"littlemore shared library not built: {e}")


@pytest.fixture(scope="session")
def tiny(lib_path: Path) -> Path:
    """The tiny random LittleBit GGUF of the C API test (built once by tests/c/make_tiny_gguf.py, cached)."""
    spec = importlib.util.spec_from_file_location("make_tiny_gguf", ROOT / "tests" / "c" / "make_tiny_gguf.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        return mod.build(TINY)
    except ImportError as e:
        pytest.skip(f"cannot build the tiny test model: {e!r}")


@pytest.fixture(scope="session")
def qwen06(lib_path: Path) -> Path:
    """The init-only Qwen3-0.6B LittleBit 0.55 B1 GGUF (built by the smoke tests; never rebuilt here)."""
    if not QWEN06.exists():
        pytest.skip(f"{QWEN06} missing: run the smoke tests once (docs/development.md section 4) to build it")
    return QWEN06


@pytest.fixture(scope="session")
def gguf_8b(lib_path: Path) -> Path:
    """The converted community Qwen3-8B LittleBit GGUF, only with LITTLEMORE_RUN_8B=1."""
    if os.environ.get("LITTLEMORE_RUN_8B") != "1":
        pytest.skip("8B test: set LITTLEMORE_RUN_8B=1 (see tests/e2e/test_community_8b.py)")
    if not GGUF_8B.exists():
        pytest.skip(f"{GGUF_8B} missing (conversion command in docs/quickstart.md)")
    return GGUF_8B


@pytest.fixture(scope="session")
def has_cuda(lib_path: Path) -> bool:
    """True when the library sees a CUDA device."""
    from littlemore import backends_info
    return any(b["name"].startswith("CUDA") for b in backends_info())
