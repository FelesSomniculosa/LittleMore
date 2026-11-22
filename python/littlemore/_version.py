"""The one place the package version is written down.

`python/pyproject.toml` reads `__version__` from here (setuptools `attr:`), `littlemore.__version__` re-exports it,
and a test checks that it equals the C library's version (`LM_VERSION_*` in include/littlemore.h), so the Python
package and the shared library it bundles are always released together.
"""
__version__ = "0.1.0"
