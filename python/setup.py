"""Build hooks for the LittleMore wheel (the metadata lives in pyproject.toml).

Two things pyproject.toml cannot express:

1. Bundling. When a wheel is built, the prebuilt C library and its backend plugins (CPU variants, CUDA, Vulkan,
   ggml, llama) are copied from a CMake build folder into `littlemore/_bin/` inside the wheel. The SDK finds them
   there at run time, so installing needs no compiler and no CMake. The folder is `LITTLEMORE_BIN_DIR` or, in a
   source checkout, `<repo>/build/bin`. Building without a library is an error (a wheel without it is useless),
   unless `LITTLEMORE_ALLOW_NO_BINARIES=1` is set (used only to build a pure-Python sdist-like wheel for tests).
   The NVIDIA CUDA runtime DLLs (cudart, cublas, cublasLt) are NOT bundled: cublasLt alone is 640 MiB, far above
   PyPI's wheel size limit, and users who have a CUDA toolkit or the `nvidia-*-cu12` wheels (extra: `littlemore[cuda]`)
   already have them; the SDK finds them (see littlemore/_lib.py, `dll_search_dirs`).
2. The wheel tag. The wheel contains native code but no CPython extension module, so it works with any Python 3:
   it is tagged `py3-none-<platform>` (for example `py3-none-win_amd64`) instead of `cp313-cp313-<platform>`.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.dist import Distribution

try:  # setuptools >= 70.1 ships bdist_wheel itself; older ones rely on the `wheel` package
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:  # pragma: no cover
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

HERE = Path(__file__).resolve().parent
# File name patterns of the native files that belong in the wheel, per platform. Executables (test programs)
# are not libraries and are never copied.
if sys.platform == "win32":
    NATIVE_PATTERNS = ["*.dll"]
elif sys.platform == "darwin":
    NATIVE_PATTERNS = ["*.dylib", "*.so"]
else:
    NATIVE_PATTERNS = ["*.so", "*.so.*"]


def native_files() -> list[Path]:
    """The library and plugins to bundle, from LITTLEMORE_BIN_DIR or <repo>/build/bin."""
    folder = Path(os.environ.get("LITTLEMORE_BIN_DIR", HERE.parent / "build" / "bin"))
    files = sorted({p for pat in NATIVE_PATTERNS for p in folder.glob(pat) if p.is_file()})
    has_core = any(p.name.startswith(("littlemore.", "liblittlemore.")) for p in files)
    if not has_core and os.environ.get("LITTLEMORE_ALLOW_NO_BINARIES") != "1":
        raise SystemExit(f"error: no littlemore library found in {folder}. Build the C library first "
                         "(docs/development.md) or point LITTLEMORE_BIN_DIR at the build's bin folder.")
    return files


class build_py(_build_py):
    """build_py that also copies the native files into build/lib/littlemore/_bin/."""

    def run(self) -> None:
        """Build as usual, then copy the native library and plugins into the package (they are not Python files)."""
        super().run()
        dest = Path(self.build_lib) / "littlemore" / "_bin"
        dest.mkdir(parents=True, exist_ok=True)
        for f in native_files():
            shutil.copy2(f, dest / f.name)


class bdist_wheel(_bdist_wheel):
    """Platform-specific but Python-version independent wheel: py3-none-<platform>."""

    def finalize_options(self) -> None:
        """Mark the wheel as non-pure: it bundles native libraries, so it needs a platform tag and goes to platlib."""
        super().finalize_options()
        self.root_is_pure = False  # contains native libraries: goes to platlib and gets a platform tag

    def get_tag(self) -> tuple[str, str, str]:
        """Keep the platform tag but drop the Python-version tag: no CPython extension module, so any Python 3 works."""
        _, _, plat = super().get_tag()
        return "py3", "none", plat


class BinaryDistribution(Distribution):
    """Tells setuptools the distribution has platform-specific content."""

    def has_ext_modules(self) -> bool:
        """Report platform-specific content (the bundled native files) although there is no extension module."""
        return True


setup(cmdclass={"build_py": build_py, "bdist_wheel": bdist_wheel}, distclass=BinaryDistribution)
