"""G. The `littlemore` command line and the converter under bad input.

CLI: bad arguments, missing or wrong files, stdin at end-of-file or closed, Ctrl+C (a real console control event,
sent through the Windows console API), no GPU (CUDA hidden) and --cpu-only, non-ASCII paths. Every case must end with
the documented exit code (0 success, 1 error, 2 usage) and an `error: ...` line, never a Python traceback.

Converter: corrupted copies of a tiny LittleBit checkpoint (missing tensor, wrong shape, NaN / Inf scales, rank
mismatch, truncated safetensors, wrong dtype, missing config files) and a read-only output folder. Every case must fail
with a non-zero exit code and a clear `error: ...` message, and must leave NO output file: a silently wrong GGUF is the
worst outcome.
"""
from __future__ import annotations

import ctypes
import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

import _rb
from conftest import ROOT

PY = sys.executable


def cli(*args: str, stdin=subprocess.DEVNULL, env: dict | None = None, timeout: float = 300, input_text: str | None = None,
        cwd: Path | None = None) -> subprocess.CompletedProcess:
    e = _rb.child_env(env)
    e["PYTHONPATH"] = str(ROOT / "python")
    return subprocess.run([PY, "-m", "littlemore", *args], capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout, env=e, input=input_text,
                          stdin=None if input_text is not None else stdin, cwd=cwd)


def no_traceback(p: subprocess.CompletedProcess) -> bool:
    return "Traceback (most recent call last)" not in p.stderr + p.stdout


# --------------------------------------------------------------------------------------------------------------
# CLI arguments and files
# --------------------------------------------------------------------------------------------------------------
def test_cli_bad_arguments(tiny: Path, tmp_path: Path) -> None:
    cases = [
        ((), 2), (("run",), 2), (("bench",), 2), (("nosuchcommand",), 2), (("run", str(tiny), "--max-tokens", "0"), 2),
        (("run", str(tiny), "--max-tokens", "abc"), 2), (("run", str(tiny), "-ngl", "lots"), 2),
        (("run", str(tiny), "-ngl", "-7"), 2), (("run", str(tiny), "--cpu-only", "--devices", "CUDA0"), 2),
        (("run", str(tiny), "--seed", "abc"), 2), (("run", str(tiny), "--seed", "-1"), 2), (("run", str(tiny), "--top-p", "2"), 2),
        (("run", str(tiny), "--top-k", "-1"), 2), (("run", str(tiny), "--temperature", "nan", "--prompt", "hi"), 2),
        (("run", str(tiny), "--n-ctx", "-5"), 2), (("run", str(tiny), "--n-threads", "100000"), 2),
        (("bench", str(tiny), "--n-prompt", "0"), 2), (("bench", str(tiny), "--reps", "-1"), 2),
        (("run", str(tmp_path / "missing.gguf"), "--prompt", "hi"), 1), (("run", str(tmp_path), "--prompt", "hi"), 1),
        (("bench", str(tmp_path / "missing.gguf")), 1), (("run", str(tiny), "--devices", "CUDA9", "--prompt", "hi"), 1),
    ]
    bad = []
    for args, code in cases:
        p = cli(*args)
        msg = (p.stderr.strip().splitlines() or [""])[-1]
        print(ascii(f"{args[:1] + tuple(a if len(a) < 40 else '...' for a in args[1:])} -> {p.returncode}: {msg[:160]}"))
        if p.returncode != code or not no_traceback(p) or "error" not in p.stderr.lower():
            bad.append((args, p.returncode, p.stderr[-600:]))
    assert not bad, bad


def test_cli_run_stdin_eof_closed_and_scripted(tiny: Path) -> None:
    """Interactive mode with stdin at end-of-file exits 0; a scripted session (/clear, a question, /exit) works."""
    p = cli("run", str(tiny), "--cpu-only")  # stdin = DEVNULL: EOF at the first prompt
    assert p.returncode == 0 and no_traceback(p), p.stderr[-800:]
    p = cli("run", str(tiny), "--cpu-only", "--max-tokens", "8", input_text="/clear\nhello\n\n/exit\n")
    assert p.returncode == 0 and no_traceback(p), p.stderr[-800:]
    # stdin closed (a pipe whose writer is gone)
    r, w = os.pipe()
    os.close(w)
    try:
        p = cli("run", str(tiny), "--cpu-only", stdin=r)
    finally:
        os.close(r)
    assert p.returncode == 0 and no_traceback(p), p.stderr[-800:]


def test_cli_no_gpu_and_cpu_only(tiny: Path) -> None:
    for env, extra in (({"CUDA_VISIBLE_DEVICES": "-1"}, ()), ({}, ("--cpu-only",)), ({"CUDA_VISIBLE_DEVICES": "-1"}, ("--cpu-only",))):
        p = cli("run", str(tiny), "--prompt", "Hello", "--max-tokens", "8", "--temperature", "0", *extra, env=env)
        assert p.returncode == 0 and no_traceback(p), (env, extra, p.stderr[-800:])
        b = cli("bench", str(tiny), "--n-prompt", "16", "--n-gen", "8", "--reps", "1", "--json", *extra, env=env)
        assert b.returncode == 0 and '"pieces"' in b.stdout, (env, extra, b.stderr[-800:])


def test_cli_unicode_and_space_paths(tiny: Path) -> None:
    d = _rb.SCRATCH / "cli paths" / "модель 模型 🚀"
    d.mkdir(parents=True, exist_ok=True)
    m = d / "tiny ü.gguf"
    shutil.copyfile(tiny, m)
    p = cli("run", str(m), "--prompt", "Hello", "--max-tokens", "4", "--cpu-only")
    assert p.returncode == 0 and no_traceback(p), p.stderr[-800:]
    p = cli("run", str(d / "missing ü.gguf"), "--prompt", "Hello")
    assert p.returncode == 1 and "missing" in p.stderr and no_traceback(p), p.stderr[-800:]


# --------------------------------------------------------------------------------------------------------------
# Ctrl+C: a real console control event
# --------------------------------------------------------------------------------------------------------------
_SEND_CTRL_C = r"""
import ctypes, sys
k = ctypes.windll.kernel32
pid = int(sys.argv[1])
k.FreeConsole()
if not k.AttachConsole(pid):
    sys.exit(2)
k.SetConsoleCtrlHandler(None, True)       # this helper ignores the event itself
ok = k.GenerateConsoleCtrlEvent(0, 0)     # CTRL_C_EVENT to every process on that console
sys.exit(0 if ok else 3)
"""


def _ctrl_c(pid: int) -> None:
    helper = _rb.SCRATCH / "send_ctrl_c.py"
    helper.write_text(_SEND_CTRL_C, encoding="utf-8")
    r = subprocess.run([PY, str(helper), str(pid)], timeout=30)
    assert r.returncode == 0, f"could not send Ctrl+C (helper exit {r.returncode})"


def test_cli_ctrl_c_at_the_prompt(qwen06: Path) -> None:
    """Ctrl+C at the interactive prompt. A real console event could not be delivered to a process waiting in input() in
    this test environment (control: a plain `python -c "input()"` in a new console did not react to it either, while
    the two cases below did), so the KeyboardInterrupt that Python raises there is injected into input() directly. The
    CLI must end with exit code 0 and no traceback."""
    e = _rb.child_env()
    e["PYTHONPATH"] = str(ROOT / "python")
    boot = ("import builtins, sys, runpy\n"
            "def _ctrl_c(prompt=''):\n    raise KeyboardInterrupt\n"
            "builtins.input = _ctrl_c\n"
            "sys.argv = ['littlemore'] + sys.argv[1:]\nrunpy.run_module('littlemore', run_name='__main__')\n")
    p = subprocess.run([PY, "-c", boot, "run", str(qwen06), "--cpu-only"], capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=300, env=e, stdin=subprocess.DEVNULL)
    assert p.returncode == 0 and "Traceback" not in p.stderr, (p.returncode, p.stderr[-1500:])


@pytest.mark.skipif(os.name != "nt", reason="Windows console control events")
@pytest.mark.parametrize("when", ["during-generation", "during-bench"])
def test_cli_ctrl_c(qwen06: Path, when: str) -> None:
    """The CLI runs in its own (hidden) console; Ctrl+C is delivered through that console (Windows console API). Expected: during a streamed
    answer it prints "[stopped]" and exits 0; at the interactive prompt it exits 0; during bench it exits without a
    traceback, with exit code 130 (the documented code for an interrupted command)."""
    e = _rb.child_env()
    e["PYTHONPATH"] = str(ROOT / "python")
    out = _rb.SCRATCH / f"ctrlc_{when}.txt"
    errf = _rb.SCRATCH / f"ctrlc_{when}.err"
    if when == "during-generation":
        args = ["run", str(qwen06), "--cpu-only", "--prompt", "Tell me a long story", "--max-tokens", "2000", "--n-ctx", "4096"]
    elif when == "at-the-prompt":
        args = ["run", str(qwen06), "--cpu-only"]
    else:
        args = ["bench", str(qwen06), "--cpu-only", "--n-prompt", "512", "--n-gen", "512", "--reps", "5"]
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = 0  # SW_HIDE
    with open(out, "w", encoding="utf-8") as fo, open(errf, "w", encoding="utf-8") as fe:
        # Ctrl+C handling is inherited as "ignored" when the test runner itself has no console: switch it back on in
        # the child (SetConsoleCtrlHandler(NULL, FALSE)) before the CLI starts, as an interactive terminal would have it.
        # At the prompt the CLI must read the console itself, as in a terminal (a pipe's ReadFile is not interrupted by
        # Ctrl+C on Windows): its standard handles are then the new console's, so its output is not captured; a marker
        # file written when input() is entered tells the test that the prompt is waiting.
        mark = _rb.SCRATCH / "ctrlc_prompt.mark"
        mark.unlink(missing_ok=True)
        boot = ("import ctypes, sys, runpy, builtins, pathlib; ctypes.windll.kernel32.SetConsoleCtrlHandler(None, False); "
                + (f"builtins.input = lambda p='', _i=builtins.input: (pathlib.Path({str(mark)!r}).write_text('x'), _i(p))[1]; "
                   if when == "at-the-prompt" else "")
                + "sys.argv = ['littlemore'] + sys.argv[1:]; runpy.run_module('littlemore', run_name='__main__')")
        console = when == "at-the-prompt"
        proc = subprocess.Popen([PY, "-c", boot, *args], stdout=None if console else fo, stderr=None if console else fe,
                                stdin=None if console else subprocess.DEVNULL, env=e,
                                creationflags=subprocess.CREATE_NEW_CONSOLE, startupinfo=si)
        try:
            # wait until the model is loaded and the CLI is busy (generation output, or the prompt banner)
            t0 = time.time()
            while time.time() - t0 < 120:
                text = out.read_text(encoding="utf-8", errors="replace") + errf.read_text(encoding="utf-8", errors="replace")
                if (when == "during-generation" and len(out.read_text(encoding="utf-8", errors="replace")) > 20) or \
                   (when == "at-the-prompt" and mark.exists()) or (when == "during-bench" and time.time() - t0 > 8):
                    break
                time.sleep(0.2)
            time.sleep(0.5)
            _ctrl_c(proc.pid)
            rc = proc.wait(timeout=60)
        finally:
            if proc.poll() is None:
                proc.kill()
    so, se = out.read_text(encoding="utf-8", errors="replace"), errf.read_text(encoding="utf-8", errors="replace")
    print(ascii(f"{when}: exit {rc}; stdout tail {so[-120:]!r}; stderr tail {se[-300:]!r}"))
    assert "Traceback" not in se, se[-2000:]
    if when == "during-generation":
        assert rc == 0 and "[stopped]" in so, (rc, so[-300:], se[-800:])
    elif when == "at-the-prompt":
        assert rc == 0, (rc, se[-800:])
    else:
        assert rc == 130 and "interrupted" in se, (rc, se[-800:])


# --------------------------------------------------------------------------------------------------------------
# Converter on corrupted checkpoints
# --------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_ckpt(lib_path: Path) -> Path:
    """A tiny init-only LittleBit-2 checkpoint (random 2-layer Llama, residual path), built once."""
    out = _rb.SCRATCH / "conv" / "ckpt"
    if (out / "model.safetensors").exists():
        return out
    sys.path[:0] = [str(ROOT), str(ROOT / "tools" / "dev"), str(ROOT / "tests" / "convert")]
    try:
        import _tiny
        from make_lb2_init import make_checkpoint
    except ImportError as e:
        pytest.skip(f"cannot build a tiny checkpoint: {e!r}")
    hf = _tiny.make_tiny_model("llama", _rb.SCRATCH / "conv" / "hf", seed=0)
    make_checkpoint(str(hf), out, 2.0, use_itq=False, residual=True, seed=0)
    return out


def _convert(ckpt: Path, out: Path, *extra: str) -> subprocess.CompletedProcess:
    e = _rb.child_env()
    e["PYTHONPATH"] = str(ROOT / "engine" / "gguf-py")
    return subprocess.run([PY, "-m", "tools.convert.lb_to_gguf", str(ckpt), "-o", str(out), *extra], capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=900, env=e, cwd=ROOT)


LIN = "model.layers.0.mlp.gate_proj."


def _corrupt(ckpt: Path, dst: Path, edit) -> Path:
    """Copy the checkpoint folder and apply `edit(tensors: dict) -> dict` to its safetensors (None = file edit)."""
    import torch
    from safetensors.torch import load_file, save_file
    if dst.is_file():
        dst.unlink()
    elif dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(ckpt, dst)
    if edit is not None:
        t = load_file(str(dst / "model.safetensors"))
        t = edit({k: v.clone() for k, v in t.items()})
        save_file(t, str(dst / "model.safetensors"))
    return dst


def _nan(name: str, value: float):
    def f(t):
        t[name][0, 3] = value
        return t
    return f


CORRUPTIONS = {
    "missing-scale": lambda t: {k: v for k, v in t.items() if k != LIN + "u1"},
    "missing-signs": lambda t: {k: v for k, v in t.items() if k != LIN + "V_packed"},
    "missing-residual-signs": lambda t: {k: v for k, v in t.items() if k != LIN + "U_R_packed"},
    "missing-embedding": lambda t: {k: v for k, v in t.items() if k != "model.embed_tokens.weight"},
    "scale-wrong-shape": lambda t: {**t, LIN + "u1": t[LIN + "u1"][:, :500].contiguous()},
    "scale-nan": _nan(LIN + "u1", float("nan")),
    "scale-inf": _nan(LIN + "v2", float("inf")),
    "mid-scale-nan": _nan(LIN + "v1", float("nan")),
    "residual-scale-inf": _nan(LIN + "u2_R", float("-inf")),
    "rank-mismatch": lambda t: {**t, LIN + "V_packed": t[LIN + "V_packed"][:-1].contiguous()},
    "shape-tensor-lies": lambda t: {**t, LIN + "U_shape": t[LIN + "U_shape"] + 1},
    "scale-wrong-dtype-f16": lambda t: {**t, LIN + "u1": t[LIN + "u1"].half()},
    "scale-wrong-dtype-int": lambda t: {**t, LIN + "u1": t[LIN + "u1"].to(__import__("torch").int32)},
    "signs-wrong-dtype": lambda t: {**t, LIN + "V_packed": t[LIN + "V_packed"].float()},
    "embedding-nan": lambda t: (t["model.embed_tokens.weight"].__setitem__((5, 7), float("nan")), t)[1],
    "norm-inf": lambda t: (t["model.norm.weight"].__setitem__(3, float("inf")), t)[1],
}


LOSSLESS_OK = {"scale-wrong-dtype-f16"}


@pytest.mark.parametrize("case", sorted(CORRUPTIONS) + ["truncated-safetensors", "missing-config", "missing-littlebit-config",
                                                       "missing-tokenizer", "empty-folder", "not-a-folder"])
def test_converter_fails_loudly_and_writes_nothing(tiny_ckpt: Path, case: str) -> None:
    root = _rb.SCRATCH / "conv" / "cases"
    root.mkdir(parents=True, exist_ok=True)
    dst = root / case
    if case in CORRUPTIONS:
        _corrupt(tiny_ckpt, dst, CORRUPTIONS[case])
    else:
        _corrupt(tiny_ckpt, dst, None)
        if case == "truncated-safetensors":
            f = dst / "model.safetensors"
            f.write_bytes(f.read_bytes()[: f.stat().st_size // 2])
        elif case == "missing-config":
            (dst / "config.json").unlink()
        elif case == "missing-littlebit-config":
            (dst / "littlebit_config.json").unlink()
        elif case == "missing-tokenizer":
            for n in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json"):
                if (dst / n).exists():
                    (dst / n).unlink()
        elif case == "empty-folder":
            shutil.rmtree(dst)
            dst.mkdir()
        elif case == "not-a-folder":
            shutil.rmtree(dst)
            dst.write_text("not a checkpoint")
    out = root / f"{case}.gguf"
    for f in root.glob(f"{case}.gguf*"):
        f.unlink()
    p = _convert(dst, out)
    leftovers = sorted(f.name for f in root.glob(f"{case}.gguf*"))
    err_lines = [ln for ln in p.stderr.splitlines() if ln.startswith("error:")]
    print(ascii(f"{case}: exit {p.returncode}, files {leftovers}, error: {(err_lines or [''])[-1][:240]}"))
    if case in LOSSLESS_OK:
        # an F16 scale converts exactly to the F32 the engine reads: a correct conversion, verified by the read-back
        assert p.returncode == 0 and leftovers == [f"{case}.gguf"], p.stderr[-1500:]
        return
    assert p.returncode != 0, f"{case}: the converter accepted a broken checkpoint: {p.stdout[-500:]}"
    assert not leftovers, f"{case}: output left behind: {leftovers}"
    assert "Traceback" not in p.stderr and err_lines, f"{case}: not a clear error message: {p.stderr[-1500:]}"


def test_converter_control_and_read_only_output(tiny_ckpt: Path) -> None:
    """Control: the unmodified copy converts. A read-only output folder is a clean error with no file."""
    root = _rb.SCRATCH / "conv" / "ro"
    if root.exists():
        subprocess.run(["attrib", "-R", str(root), "/S", "/D"], capture_output=True)
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    good = _convert(tiny_ckpt, root / "good.gguf")
    assert good.returncode == 0 and (root / "good.gguf").exists(), good.stderr[-1500:]
    ro = root / "readonly"
    ro.mkdir()
    # Windows: a folder's read-only attribute does not stop file creation; deny write with an ACL instead.
    who = os.environ.get("USERNAME", "")
    deny = subprocess.run(["icacls", str(ro), "/deny", f"{who}:(OI)(CI)(W,AD,WD)"], capture_output=True, text=True)
    try:
        if deny.returncode != 0:
            pytest.skip(f"cannot make a read-only folder: {deny.stderr}")
        try:
            (ro / "probe").write_text("x")
            pytest.skip("the folder is still writable (ACL not honoured)")
        except OSError:
            pass
        p = _convert(tiny_ckpt, ro / "model.gguf")
        last = (p.stderr.strip().splitlines() or [""])[-1]
        print(ascii(f"read-only: exit {p.returncode}: {last[:240]}"))
        assert p.returncode != 0 and "Traceback" not in p.stderr and "error:" in p.stderr, p.stderr[-1500:]
    finally:
        subprocess.run(["icacls", str(ro), "/remove:d", who], capture_output=True)
    assert not list(ro.glob("model.gguf*"))
