#!/usr/bin/env python3
"""Environment check for LittleMore on Windows.

What: verifies that every tool needed to build the llama.cpp fork (CUDA, CPU,
Vulkan backends) and to run the Python tooling is installed, printing each
tool and its version.

Why: builds fail late and cryptically when a tool is missing. This script
fails early with a clear list. It exits 0 only when ALL of these are present:
nvcc >= 12, cmake, ninja, vulkaninfo, Visual Studio Build Tools (via vswhere),
and a CUDA-enabled PyTorch (`torch.cuda.is_available()`).

Usage: python tools/dev/check_env.py
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]

# Why: a freshly installed tool is not on PATH in an already-open terminal
# (Windows only refreshes PATH for new processes), so we also probe the
# standard install directories.
EXTRA_DIRS = {
    "nvcc": [r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.*\bin",
             r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v1[3-9].*\bin"],
    "cmake": [r"C:\Program Files\CMake\bin"],
    "ninja": [r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Ninja-build.Ninja*",
              r"C:\Program Files\Ninja", r"C:\ProgramData\chocolatey\bin"],
    "vulkaninfo": [r"C:\VulkanSDK\*\Bin", r"C:\Windows\System32"],
}
VSWHERE = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"


def find_tool(name: str) -> Optional[str]:
    """Return the full path of executable `name`, or None if not found.

    Looks on PATH first, then in the well-known install directories.
    """
    found = shutil.which(name)
    if found:
        return found
    for pattern in EXTRA_DIRS.get(name, []):
        for d in sorted(glob.glob(os.path.expandvars(pattern)), reverse=True):
            found = shutil.which(name, path=d)
            if found:
                return found
    return None


def run(cmd: list[str]) -> str:
    """Run a command and return combined stdout+stderr (empty on failure)."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return (r.stdout or "") + (r.stderr or "")
    except (OSError, subprocess.SubprocessError):
        return ""


def first_match(pattern: str, text: str) -> Optional[str]:
    """Return the first capture group of `pattern` in `text`, or None."""
    m = re.search(pattern, text)
    return m.group(1) if m else None


def check_nvcc() -> tuple[bool, str]:
    """nvcc must exist and be CUDA major version >= 12."""
    path = find_tool("nvcc")
    if not path:
        return False, "nvcc not found (install CUDA Toolkit 12.x)"
    ver = first_match(r"release (\d+\.\d+)", run([path, "--version"]))
    if not ver:
        return False, f"{path}: could not read version"
    if int(ver.split(".")[0]) < 12:
        return False, f"nvcc {ver} is too old, need >= 12 ({path})"
    return True, f"nvcc {ver} ({path})"


def check_cmake() -> tuple[bool, str]:
    """cmake must be installed."""
    path = find_tool("cmake")
    if not path:
        return False, "cmake not found (winget install Kitware.CMake)"
    return True, f"cmake {first_match(r'version (\S+)', run([path, '--version']))} ({path})"


def check_ninja() -> tuple[bool, str]:
    """ninja must be installed."""
    path = find_tool("ninja")
    if not path:
        return False, "ninja not found (winget install Ninja-build.Ninja)"
    return True, f"ninja {run([path, '--version']).strip()} ({path})"


def check_vulkaninfo() -> tuple[bool, str]:
    """vulkaninfo must be present.

    Why: the stock Windows copy only proves the *runtime*; the SDK is needed
    to compile the Vulkan backend (glslc), so we also report whether the SDK
    is present, and require it.
    """
    path = find_tool("vulkaninfo")
    if not path:
        return False, "vulkaninfo not found (install Vulkan SDK)"
    glslc = find_tool_sdk("glslc")
    if not glslc:
        return False, f"vulkaninfo found ({path}) but Vulkan SDK (glslc) missing"
    ver = first_match(r"Vulkan Instance Version:\s*(\S+)", run([path, "--summary"]))
    return True, f"vulkaninfo, instance {ver or '?'} ({path}); glslc ({glslc})"


def find_tool_sdk(name: str) -> Optional[str]:
    """Locate an executable from the Vulkan SDK (PATH, then C:/VulkanSDK/*/Bin)."""
    found = shutil.which(name)
    if found:
        return found
    for d in sorted(glob.glob(r"C:\VulkanSDK\*\Bin"), reverse=True):
        found = shutil.which(name, path=d)
        if found:
            return found
    return None


def check_vs() -> tuple[bool, str]:
    """Visual Studio Build Tools with the C++ toolchain, found via vswhere."""
    if not Path(VSWHERE).exists():
        return False, "vswhere not found (install VS 2022 Build Tools)"
    out = run([VSWHERE, "-products", "*", "-latest",
               "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
               "-property", "installationPath"]).strip()
    if not out:
        return False, "VS Build Tools with C++ x64 tools not found"
    ver = run([VSWHERE, "-products", "*", "-latest", "-property",
               "installationVersion"]).strip()
    return True, f"VS Build Tools {ver} ({out})"


def check_torch() -> tuple[bool, str]:
    """CUDA-enabled torch must be importable and see a GPU.

    Why: run in a subprocess so a missing/broken torch cannot crash this
    script, and prefer the repo venv's interpreter, which is where torch lives
    even when the script is started with the system Python.
    """
    venv_py = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    py = str(venv_py) if venv_py.exists() else sys.executable
    code = ("import torch;print(torch.__version__);"
            "print(torch.cuda.is_available());"
            "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')")
    try:
        r = subprocess.run([py, "-I", "-c", code], capture_output=True,
                           text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as e:
        return False, f"could not run {py}: {e}"
    lines = r.stdout.strip().splitlines()
    if r.returncode != 0 or len(lines) < 2:
        return False, f"torch not importable in {py} (see docs/setup-windows.md)"
    if lines[1] != "True":
        return False, f"torch {lines[0]} installed but CUDA unavailable (CPU-only wheel?)"
    return True, f"torch {lines[0]} cuda=True ({lines[2]})"


CHECKS: list[tuple[str, Callable[[], tuple[bool, str]]]] = [
    ("nvcc", check_nvcc),
    ("cmake", check_cmake),
    ("ninja", check_ninja),
    ("vulkan", check_vulkaninfo),
    ("vs-build-tools", check_vs),
    ("torch-cuda", check_torch),
]


def main() -> int:
    """Run every check, print a report, and return the process exit code."""
    failed = []
    for name, fn in CHECKS:
        ok, msg = fn()
        print(f"[{'OK' if ok else 'MISSING'}] {name:15s} {msg}")
        if not ok:
            failed.append(name)
    if failed:
        print(f"\nFAIL: missing/invalid: {', '.join(failed)}")
        print("If you just installed something, open a NEW terminal. See docs/setup-windows.md.")
        return 1
    print("\nOK: environment ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
