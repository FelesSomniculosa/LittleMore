"""LittleMore: run LittleBit (sub-1-bit) language models, and ordinary GGUF models, from Python.

    import littlemore

    with littlemore.Model("model.gguf") as model:
        print(model.chat([{"role": "user", "content": "Hello!"}]))
        for piece in model.chat([{"role": "user", "content": "Tell me a story"}], stream=True):
            print(piece, end="", flush=True)

The package is a thin `ctypes` layer over the LittleMore C library (no compiler needed to install); wheels bundle
the library and its compute-backend plugins (CPU variants, CUDA, Vulkan). See `littlemore.Model` for all options
and `littlemore --help` for the command line (`convert`, `run`, `bench`).
"""
from ._lib import LibraryNotFoundError, LittleMoreError
from ._version import __version__
from .model import SEED_RANDOM, Model, backends_info, version

__all__ = ["Model", "LittleMoreError", "LibraryNotFoundError", "backends_info", "version", "SEED_RANDOM",
           "__version__"]
