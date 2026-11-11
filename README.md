# TensorRT Edge-LLM on Windows with TensorRT for RTX (build notes)

[TensorRT Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM) is NVIDIA's C++ runtime for LLMs and vision-language models. Version 0.11.1 added a [source build for Windows](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/v0.11.1/docs/source/user_guide/getting_started/installation.md#windows-x64-with-tensorrt-rtx), which uses NVIDIA's TensorRT for RTX library, so I tried it. These are the steps and scripts I used to build it and run [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) on my laptop's RTX GPU.

This is independent work, and NVIDIA has not sponsored or endorsed it.

On all 8 test prompts, Edge-LLM's greedy output had the same token ids as Hugging Face transformers in fp16 ([the comparison](results/check.md)).

## Speed

Decode speed from one session on an RTX 5080 Laptop GPU (driver 591.74), with a 19-token prompt and 128 new tokens:

| Runtime | Weights | Tokens per second |
|---|---|---|
| TensorRT Edge-LLM v0.11.1, after one warm-up request | fp16 | 369.0 |
| llama.cpp b11721, CUDA | F16 | 321.8 |
| llama.cpp b11721, Vulkan | F16 | 312.6 |
| llama.cpp b11721, CUDA | Q4_K_M | 488.9 to 502.7 |

I only built Edge-LLM with fp16 weights. Its time comes from CUDA events on the GPU, and llama.cpp's from the host clock. By the wall clock, Edge-LLM decoded at 367.5.

These numbers are published with NVIDIA's written permission under Section 2.13 of the NVIDIA TensorRT for RTX License Agreement. [The speed page](docs/speed.md) has every run and how I measured them.

## What you need

- Windows 10 or 11 x64, and an NVIDIA RTX GPU with a driver that supports CUDA 13. I only tried an RTX 5080 Laptop GPU. The kernels cover Ampere, Ada Lovelace and Blackwell, which leaves out Turing (RTX 20 series).
- Visual Studio 2022 with the C++ build tools. The Build Tools edition is enough.
- Git, Python 3.12 with the `py` launcher, and CMake 3.20 or newer on `PATH`.
- About 13 GB of free disk, plus room for pip's download cache.

You download NVIDIA's parts yourself and build the engine on your machine, so none of them are in this repo.

## Build and run it

Open PowerShell in a new folder and clone this repo into it:

```powershell
git clone https://github.com/john-rocky/tensorrt-edge-llm-windows-notes.git
```

Stay in that folder, outside the clone, and run [the PowerShell blocks](docs/steps.md) in order. Steps 9 to 11 check and time it, so you can stop after step 8 if you only want it running.

1. [Get the CUDA 13.5 parts](docs/steps.md#1-cuda-135-from-nvidias-redistributable-archives) from NVIDIA's archives
2. [Unpack TensorRT for RTX 1.7.1](docs/steps.md#2-tensorrt-for-rtx-171) after downloading it from NVIDIA, which lists a Developer Program membership as a requirement
3. [Clone Edge-LLM](docs/steps.md#3-source)
4. [Generate the CuTe DSL kernels](docs/steps.md#4-cute-dsl-kernels) for your GPU
5. [Build Edge-LLM](docs/steps.md#5-build)
6. [Export Qwen3-0.6B to ONNX](docs/steps.md#6-export-qwen3-06b-to-onnx)
7. [Build the engine](docs/steps.md#7-build-the-engine)
8. [Generate text](docs/steps.md#8-generate)
9. [Compare the tokens with transformers](docs/steps.md#9-check-against-transformers)
10. [Time one short prompt](docs/steps.md#10-time-the-short-chat-task)
11. [Compare that answer with transformers](docs/steps.md#11-check-the-timed-answer)

## Problems I met

The steps already include these fixes.

| Step | What happened | What worked |
|---|---|---|
| 2 | Edge-LLM's support matrix asks for TensorRT RTX 1.7.1.80, but I couldn't find it on NVIDIA's download page or PyPI. | TensorRT for RTX 1.7.1.116, the only 1.7 version on NVIDIA's download page, in its CUDA 13.4 package. It ran with step 1's CUDA 13.5. |
| 1, 5, 7 | The CUDA 13 zips put the runtime DLLs (`cudart64_13.dll`, `nvrtc64_130_0.dll`, `nvrtc-builtins64_135.dll`) in `cuda\bin\x64` instead of `cuda\bin`. | Put both folders on `PATH`. |
| 5 | On Windows set to Japanese (code page 932), MSVC stops in xgrammar's `tokenizer_info.cc` with `error C2001` (newline in string literal). | Set `/utf-8` in the three `*_FLAGS_INIT` options and build in a new folder. |

## Licences

The scripts and notes here are MIT (`LICENSE`). TensorRT Edge-LLM and Qwen3-0.6B are Apache-2.0. TensorRT for RTX and CUDA are under NVIDIA's own license agreements, listed in [the steps](docs/steps.md#licences).
