"""The `littlemore` command line: `convert`, `run` and `bench`.

    littlemore convert --repo CHECKOUT CHECKPOINT_DIR -o model.gguf   LittleBit checkpoint -> GGUF (needs a source checkout)
    littlemore run model.gguf                           chat in the terminal (or --prompt "..." for one answer)
    littlemore bench model.gguf                         quick speed measurement through the SDK

`convert` wraps the converter script of the source repository (`tools/convert/lb_to_gguf.py`), which needs PyTorch,
transformers and the GGUF writer; a plain `pip install littlemore` does not carry those, so `convert` explains
what is missing instead of failing with an import traceback. `run` and `bench` need only the installed package.

Exit codes: 0 success; 1 a LittleMore or conversion error (printed as `error: ...` on stderr, no traceback);
2 a command-line usage error: a bad option (argparse) or a bad value the library refuses (for example
`--max-tokens 0`), also printed as `error: ...`.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import statistics
import sys
import time
import types
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from . import _lib
from ._lib import LittleMoreError
from ._version import __version__

if TYPE_CHECKING:  # only for the type hints; the real import stays lazy (see _open_model)
    from .model import Model


# --------------------------------------------------------------------------------------------------------------
# Shared option groups
# --------------------------------------------------------------------------------------------------------------
def _add_model_args(p: argparse.ArgumentParser) -> None:
    """Options that choose and configure the model (shared by `run` and `bench`)."""
    p.add_argument("model", type=Path, help="path of the .gguf model")
    p.add_argument("-ngl", "--n-gpu-layers", default="auto", metavar="N|auto",
                   help='layers to put on the GPU: "auto" (default, as many as fit) or a number. Note: with 0 the GPU '
                        "still speeds up big prompt batches; use --cpu-only to avoid the GPU completely")
    p.add_argument("--cpu-only", action="store_true",
                   help='use no GPU at all (devices="none"): the model allocates nothing on the GPU (the loaded backends still '
                        'hold a small driver context, about 73 MiB), generation and prompts run on the CPU')
    p.add_argument("--devices", default=None, metavar="NAMES",
                   help='comma-separated device names as listed by the library, e.g. "CUDA0" (default: every GPU)')
    p.add_argument("--n-ctx", type=int, default=0, help="context size in tokens (0 = library default, 4096)")
    p.add_argument("--n-threads", type=int, default=0, help="CPU threads (0 = library default)")


def _open_model(a: argparse.Namespace, n_ctx: int | None = None) -> "Model":
    """Build a `Model` from the shared options (imported lazily so `--help` stays instant)."""
    from .model import Model

    if a.cpu_only and a.devices:
        raise LittleMoreError("--cpu-only and --devices cannot be combined", None, "USAGE")
    ngl: int | str = a.n_gpu_layers
    if isinstance(ngl, str) and ngl.strip().lower() != "auto":
        try:
            ngl = int(ngl)
        except ValueError:
            raise LittleMoreError(f'--n-gpu-layers must be "auto" or a number, not {ngl!r}', None, "USAGE") from None
    return Model(a.model, ngl, devices="none" if a.cpu_only else a.devices,
                 n_ctx=n_ctx if n_ctx is not None else a.n_ctx, n_threads=a.n_threads)


# --------------------------------------------------------------------------------------------------------------
# convert
# --------------------------------------------------------------------------------------------------------------
def _looks_like_repo(root: Path) -> bool:
    """True when `root` looks like the LittleMore source repository (converter, public header and its CMake project)."""
    cmake = root / "CMakeLists.txt"
    try:
        return ((root / "tools" / "convert" / "lb_to_gguf.py").is_file() and (root / "include" / "littlemore.h").is_file()
                and cmake.is_file() and "project(littlemore" in cmake.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return False


def _split_repo_option(argv: Sequence[str]) -> tuple[str | None, list[str]]:
    """Take `--repo PATH` / `--repo=PATH` out of the converter's arguments. Returns (path or None, remaining arguments)."""
    rest: list[str] = []
    repo: str | None = None
    it = iter(argv)
    for arg in it:
        if arg == "--repo":
            repo = next(it, None)
            if repo is None:
                raise LittleMoreError("--repo needs a folder", None, "USAGE")
        elif arg.startswith("--repo="):
            repo = arg[len("--repo="):]
        else:
            rest.append(arg)
    return repo, rest


def _cmd_convert(argv: Sequence[str]) -> int:
    """Forward `argv` to the converter of a LittleMore source checkout, which must be named explicitly.

    The converter is Python code that gets executed, so it is never searched for in the working directory (a
    folder you merely happen to be in could hold anything): pass `--repo PATH` or set LITTLEMORE_REPO, and the folder
    must look like the LittleMore repository. The path used is printed. The converter is imported from that folder,
    never from whatever `tools` package happens to be first on `sys.path` (a generic name that other projects use too)."""
    repo_arg, argv = _split_repo_option(argv)
    repo_arg = repo_arg or os.environ.get("LITTLEMORE_REPO")
    if not repo_arg:
        print("error: `littlemore convert` runs the converter of the LittleMore source repository, which the installed "
              "package does not include. Name the checkout explicitly: `littlemore convert --repo PATH ...` or set "
              "LITTLEMORE_REPO (it is deliberately not searched for in the current directory, because the converter "
              "is code that gets executed). See docs/quickstart.md.", file=sys.stderr)
        return 1
    root = Path(repo_arg).expanduser().resolve()
    if not _looks_like_repo(root):
        print(f"error: {root} does not look like the LittleMore repository (expected tools/convert/lb_to_gguf.py, "
              "include/littlemore.h and a CMakeLists.txt that defines the project).", file=sys.stderr)
        return 1
    print(f"littlemore convert: using the converter of {root}", file=sys.stderr)
    script = (root / "tools" / "convert" / "lb_to_gguf.py").resolve()
    for name in [n for n in sys.modules if n == "tools" or n.startswith("tools.")]:
        mod_file = getattr(sys.modules[name], "__file__", None)
        if not mod_file or not Path(mod_file).resolve().is_relative_to(root.resolve()):
            del sys.modules[name]  # an unrelated `tools` that was imported earlier would shadow the converter
    # `tools` is a namespace package in the repository (no __init__.py), and Python always prefers a REGULAR package
    # of the same name found anywhere on sys.path, so a path entry is not enough: register the repository's folder
    # as the `tools` package explicitly, then import the converter through it.
    pkg = types.ModuleType("tools")
    pkg.__path__ = [str(root / "tools")]
    sys.modules["tools"] = pkg
    try:
        lb_to_gguf = importlib.import_module("tools.convert.lb_to_gguf")
    except ImportError as e:
        print(f"error: cannot import the converter {script}: {e}. Install the conversion requirements listed in "
              "docs/development.md (PyTorch, transformers, gguf).", file=sys.stderr)
        return 1
    if Path(lb_to_gguf.__file__).resolve() != script:
        print(f"error: imported {lb_to_gguf.__file__} instead of the converter {script}", file=sys.stderr)
        return 1
    return int(lb_to_gguf.main(list(argv)))


# --------------------------------------------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------------------------------------------
def _sampling_kwargs(a: argparse.Namespace) -> dict:
    """The sampling options the user actually gave (the rest keeps the library defaults)."""
    kw = {k: getattr(a, k) for k in ("temperature", "top_k", "top_p", "seed", "max_tokens")
          if getattr(a, k) is not None}
    if isinstance(kw.get("seed"), str) and kw["seed"] != "random":
        try:
            kw["seed"] = int(kw["seed"])
        except ValueError:
            raise LittleMoreError(f'--seed must be a number or "random", not {kw["seed"]!r}', None, "USAGE") from None
    return kw


def _answer(model: "Model", history: list[dict[str, str]], kw: dict) -> str:
    """Stream the answer to stdout and return it. Ctrl+C stops the generation and keeps what was printed."""
    parts: list[str] = []
    gen = model.chat(history, stream=True, **kw)
    try:
        for piece in gen:
            parts.append(piece)
            sys.stdout.write(piece)
            sys.stdout.flush()
    except KeyboardInterrupt:
        sys.stdout.write("\n[stopped]")
    finally:
        gen.close()  # stops the generation and releases the session when we leave early
    sys.stdout.write("\n")
    sys.stdout.flush()
    return "".join(parts)


def _cmd_run(a: argparse.Namespace) -> int:
    """`littlemore run`: one answer for --prompt, otherwise an interactive chat loop. Returns the exit code."""
    from .model import _make_sampling

    kw = _sampling_kwargs(a)
    _make_sampling(**kw)  # refuse a bad value (ValueError) before the model is loaded, not in the middle of a chat
    with _open_model(a) as model:
        history: list[dict[str, str]] = [{"role": "system", "content": a.system}] if a.system else []
        if a.prompt is not None:  # one question, one answer: for scripts and tests
            history.append({"role": "user", "content": a.prompt})
            _answer(model, history, kw)
            return 0
        print(f"LittleMore {__version__}: {model.info()['model']['description']}. "
              "Type /exit to quit, /clear to start a new conversation.", file=sys.stderr)
        while True:
            try:
                line = input("> ")
            except (EOFError, KeyboardInterrupt):
                print(file=sys.stderr)
                return 0
            line = line.strip()
            if not line:
                continue
            if line in ("/exit", "/quit"):
                return 0
            if line == "/clear":
                history = history[:1] if a.system else []
                continue
            # Every call needs the whole conversation: the library keeps nothing between calls.
            history.append({"role": "user", "content": line})
            try:
                history.append({"role": "assistant", "content": _answer(model, history, kw)})
            except LittleMoreError as e:  # e.g. the context is full: report and let the user /clear
                history.pop()
                print(f"error: {e}", file=sys.stderr)


# --------------------------------------------------------------------------------------------------------------
# bench
# --------------------------------------------------------------------------------------------------------------
_FILLER = "The quick brown fox jumps over the lazy dog while the sun sets slowly behind the distant hills. "


def _cmd_bench(a: argparse.Namespace) -> int:
    """Time loading, the first piece (prompt processing plus one token) and generation through the SDK.

    Counts are of *pieces* delivered by the library: one per generated token, except where the library merged
    tokens to complete a UTF-8 character. Numbers include the SDK's small streaming overhead; for rigorous engine
    benchmarks use bench/run_bench.py (llama-bench) from the source repository.
    """
    n_ctx = a.n_ctx or a.n_prompt + a.n_gen + 64
    t0 = time.perf_counter()
    with _open_model(a, n_ctx) as model:
        load_s = time.perf_counter() - t0
        ids = model.tokenize(_FILLER * (a.n_prompt // 10 + 2))[: a.n_prompt]
        prompt = model.detokenize(ids)
        n_prompt = len(model.tokenize(prompt, add_special=True))
        reps: list[dict[str, float]] = []
        for _ in range(a.reps + 1):  # the first repetition is an untimed warm-up (caches, GPU clocks)
            start = time.perf_counter()
            first: float | None = None
            pieces = 0
            for _piece in model.generate(prompt, stream=True, temperature=0, max_tokens=a.n_gen):
                if first is None:
                    first = time.perf_counter() - start
                pieces += 1
            total = time.perf_counter() - start
            if first is None:
                raise LittleMoreError("the model produced no output", None, "BENCH")
            rest = total - first
            reps.append({"first_piece_ms": first * 1000, "pieces": pieces,
                         "generation_pieces_per_s": (pieces - 1) / rest if pieces > 1 and rest > 0 else 0.0})
        reps = reps[1:]
        info = model.info()
    result = {
        "model": str(a.model), "library_version": info["library_version"], "n_ctx": n_ctx,
        "gpu_devices": info["model"]["gpu_devices"], "n_gpu_layers": info["model"]["n_gpu_layers"],
        "load_s": round(load_s, 3), "n_prompt_tokens": n_prompt, "n_gen_max": a.n_gen,
        "first_piece_ms": round(statistics.median(r["first_piece_ms"] for r in reps), 2),
        "pieces": int(statistics.median(r["pieces"] for r in reps)),
        "generation_pieces_per_s": round(statistics.median(r["generation_pieces_per_s"] for r in reps), 2),
        "reps": reps,
    }
    if a.json:
        print(json.dumps(result, indent=2))
    else:
        where = result["gpu_devices"] or "CPU only"
        print(f"model {a.model.name}  ({where}, {result['n_gpu_layers']} GPU layers)\n"
              f"  load                {result['load_s']:.2f} s\n"
              f"  prompt              {n_prompt} tokens, first piece after {result['first_piece_ms']:.0f} ms "
              f"(median of {a.reps})\n"
              f"  generation          {result['generation_pieces_per_s']:.1f} pieces/s "
              f"({result['pieces']} pieces)")
    return 0


# --------------------------------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------------------------------
def _positive_int(text: str) -> int:
    """argparse type: an integer >= 1 (a clean usage error instead of a crash or a meaningless measurement)."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer (>= 1), not {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    """The argument parser (`convert` is listed here for --help but handled before parsing, see `main`)."""
    ap = argparse.ArgumentParser(prog="littlemore", description="Run LittleBit and GGUF models: convert, chat, benchmark.")
    ap.add_argument("--version", action="version", version=f"littlemore {__version__}")
    sub = ap.add_subparsers(dest="command", required=True, metavar="{convert,run,bench}")
    sub.add_parser("convert", help="convert a LittleBit checkpoint to GGUF (source checkout; all options: "
                                   "littlemore convert --help)", add_help=False)

    run = sub.add_parser("run", help="chat with a model in the terminal, or answer one --prompt")
    _add_model_args(run)
    run.add_argument("--prompt", help="answer this single prompt and exit (non-interactive)")
    run.add_argument("--system", help="system message at the start of the conversation")
    run.add_argument("--max-tokens", type=int, help="most tokens per answer (default: library default, 256)")
    run.add_argument("--temperature", type=float, help="sampling temperature; 0 = greedy (default 0.8)")
    run.add_argument("--top-k", type=int, help="top-k sampling (default 40, 0 = off)")
    run.add_argument("--top-p", type=float, help="top-p sampling (default 0.95)")
    run.add_argument("--seed", help='sampling seed or "random" (default 0: the same prompt gives the same answer)')

    bench = sub.add_parser("bench", help="measure load time, first-piece latency and generation speed")
    _add_model_args(bench)
    bench.add_argument("--n-prompt", type=_positive_int, default=128, help="prompt length in tokens (default 128)")
    bench.add_argument("--n-gen", type=_positive_int, default=64, help="most tokens to generate (default 64)")
    bench.add_argument("--reps", type=_positive_int, default=3,
                       help="timed repetitions after one warm-up (default 3, at least 1)")
    bench.add_argument("--json", action="store_true", help="print the result as JSON")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line; returns the exit code."""
    args = list(sys.argv[1:] if argv is None else argv)
    for stream in (sys.stdout, sys.stderr):  # a terminal that cannot show a character must not crash the stream
        try:
            stream.reconfigure(errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass
    if args and args[0] == "convert":  # the converter has its own options: pass everything through untouched
        return _cmd_convert(args[1:])
    a = build_parser().parse_args(args)
    try:
        return _cmd_run(a) if a.command == "run" else _cmd_bench(a)
    except ValueError as e:  # a value the library refuses (bad sampling or size option): a usage error
        print(f"error: {e}", file=sys.stderr)
        return 2
    except LittleMoreError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2 if e.status_name == "USAGE" else 1


if __name__ == "__main__":
    sys.exit(main())
