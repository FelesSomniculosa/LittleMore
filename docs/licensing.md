# Licensing

LittleMore mixes pieces under different licences. This page lists each one, what it covers and what it means for you.
It is a plain-language summary written from the licence files and model cards named below, not legal advice.

## The short version

- **The software is MIT.** You may use, change and ship LittleMore, including commercially.
- **The models are a separate matter.** The LittleBit checkpoints that exist today are **non-commercial (CC BY-NC 4.0)**
  and also carry the licence of the base model they were made from. LittleMore does not host or distribute any model.
- **Check the licence of every model you run** before you use it in a product.

## What is under which licence

| Piece | Licence | Notes |
|---|---|---|
| LittleMore's own code: the C API (`include/`, `src/`), the Python SDK and CLI (`python/`), the converter (`tools/convert`), the dev tools (`tools/dev`), tests, benchmarks, docs | MIT, see [LICENSE](../LICENSE) | Copyright holder: "LittleMore contributors". The Python wheel carries the same `LICENSE`. |
| The engine (`engine/`, a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp)) including ggml | MIT (copyright "The ggml authors") | The fork keeps upstream's `LICENSE` file verbatim. The wheel's `THIRD_PARTY_NOTICES` reproduces it for the bundled binaries. |
| The LittleBit linear support the engine fork started from: one commit of [Cybertiron/llama.cpp-littlebit](https://github.com/Cybertiron/llama.cpp-littlebit) | MIT (same as llama.cpp) | Cherry-picked as described in [engine-changes.md](engine-changes.md); that repository's other commits were not taken. The code was then heavily extended (B1 type, fused op, CPU and CUDA kernels). |
| Samsung's LittleBit code ([SamsungLabs/LittleBit](https://github.com/SamsungLabs/LittleBit)) | CC BY-NC 4.0 (non-commercial) | **Nothing from it is copied into this repository.** The reference math in `tools/dev/reference.py` and the converter are written from the papers (below) and checked against their numbers. |
| The LittleBit papers: [arXiv 2506.13771](https://arxiv.org/abs/2506.13771) (LittleBit) and [arXiv 2603.00042](https://arxiv.org/abs/2603.00042) (LittleBit-2) | CC BY-NC-ND 4.0 (both, as shown on their arXiv pages, checked 2026-11-22) | We cite them; we do not copy their text or figures. |
| LittleBit **checkpoints** (model weights) | set by whoever published them | See the next section. |
| Base models the checkpoints derive from (for example Qwen3) | the base model's own licence | Qwen3 is Apache 2.0 (per the Qwen3-8B model card metadata). |

## Model checkpoints: non-commercial today

The checkpoint used in the [quickstart](quickstart.md),
[`LittleBitLLM/littlebit-qwen3-8b-0.55bpw`](https://huggingface.co/LittleBitLLM/littlebit-qwen3-8b-0.55bpw), is
**CC BY-NC 4.0**: its model card says the licence is "inherited from the LittleBit code" and that the base model's own
licence also applies. CC BY-NC means **no commercial use**. It is an independent community release, not an official
Samsung model.

The same reasoning applies to any LittleBit checkpoint made with Samsung's training code: read the licence on its own
model card, and the licence of the base model it was compressed from. The quickstart repeats this warning where you
download the model.

LittleMore itself imposes no restriction on a model's use. The restriction comes from the model's licence.

## What we host and what we bundle

- **We host no models.** You download a checkpoint yourself and convert it on your machine with `littlemore convert`.
  A converted GGUF file is a derivative of the checkpoint and carries the same licence terms.
- **The wheel bundles** the LittleMore library and the llama.cpp/ggml backend plugins built from source (MIT).
- **CUDA runtime DLLs are not bundled.** `ggml-cuda` loads `cudart`, `cublas` and `cublasLt` from the machine
  (a CUDA toolkit, or `pip install "littlemore[cuda]"`, NVIDIA's own runtime wheels, which have NVIDIA's licence). The
  Microsoft Visual C++ runtime is not bundled either. See [quickstart.md](quickstart.md#1-install) and
  [../python/THIRD_PARTY_NOTICES](../python/THIRD_PARTY_NOTICES).
- **Test models in this repository's tests are tiny random models generated on the fly**, so no third-party weights are
  stored in the repository. The slower end-to-end tests use models you download yourself (see
  [development.md](development.md)) and are skipped when those are absent.

## Contributing

By contributing you agree that your contribution is licensed under the MIT licence of this repository. Do not copy code
from Samsung's LittleBit repository (CC BY-NC) or from any other source with licence terms incompatible with MIT.
