# Windows setup

LittleMore builds a llama.cpp fork with CUDA, CPU and Vulkan backends and uses
Python tooling around it. This page installs everything on Windows 11.
Tested on: RTX 4050 Laptop (6 GB, compute capability 8.9), VS 2022 Build Tools,
Python 3.13.

Open **PowerShell** for the commands below. After installing any tool, open a
**new terminal**: Windows only refreshes `PATH` for newly started programs.
(`tools/dev/check_env.py` also looks in the standard install folders, so it
works even in an old terminal.)

## 1. Prerequisites you probably already have

- Git, Python 3.13 (`python --version`), `winget` (ships with Windows 11).
- **Visual Studio 2022 Build Tools** with the "Desktop development with C++"
  workload (provides the MSVC compiler that CUDA's `nvcc` requires).

## 2. Build tools

```powershell
winget install -e --id Kitware.CMake
winget install -e --id Ninja-build.Ninja
winget install -e --id Nvidia.CUDA --version 12.9   # pin 12.x: unpinned winget installs CUDA 13.x
winget install -e --id KhronosGroup.VulkanSDK
```

Some installers ask for administrator approval (UAC); accept it.
Windows already ships `vulkaninfo` (runtime only); the SDK adds `glslc`, the
shader compiler needed to build the Vulkan backend.

### CUDA 12.9 vs 13.x

This project targets CUDA 12.x. Unpinned `winget install Nvidia.CUDA`
installs 13.x, and CUDA versions install side by side, so a machine may have
both (check `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA`). The newest
one is usually first on `PATH`. `check_env.py` prefers a `v12.*` nvcc and
prints a WARNING if only a non-12 nvcc exists. When configuring CMake, point
at 12.9 explicitly:

```powershell
cmake -G Ninja -B build -DGGML_CUDA=ON `
  -DCUDAToolkit_ROOT="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
```

To remove 13.x: Settings > Apps > Installed apps, uninstall the
"NVIDIA CUDA ... 13.x" entries (Toolkit, Runtime, Development, etc.).

## 3. Python environment

From the repo root:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -m pip install safetensors numpy pytest transformers gguf
```

`cu128` is the CUDA 12.8 PyTorch wheel index; it supports Python 3.13 and
Ada GPUs (sm_89). Commands use the explicit `.venv\Scripts\python.exe`, so
no activation is needed. Use the venv's python for tests:
`.venv\Scripts\python.exe -m pytest`.

## 4. Verify

```powershell
python tools/dev/check_env.py
```

Expected: every line starts with `[OK]` (nvcc, cmake, ninja, vulkan,
vs-build-tools, torch-cuda), it ends with `OK: environment ready.`, and the
exit code is 0. Any `[MISSING]` line says what to install; the exit code is 1.

## Troubleshooting

- **`nvcc`/`cmake` not found right after install**: open a new terminal.
- **`torch ... CUDA unavailable`**: you installed the CPU wheel. Reinstall with
  the `--index-url` above and check that `nvidia-smi` works.
- **pip cannot fetch the torch wheel** (for example `Failed to resolve
  download.pytorch.org`): a network or DNS problem; try another network or DNS
  server, then rerun the pip command.
