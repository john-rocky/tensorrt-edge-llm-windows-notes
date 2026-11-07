# TensorRT Edge-LLM on Windows with TensorRT for RTX (build notes)

Steps and scripts to build [NVIDIA TensorRT Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM) v0.11.1 from
source on Windows x64 with TensorRT for RTX, and to run one model through it: export
[Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) to ONNX, build the engine, generate text, and compare the
tokens with Hugging Face transformers on the same prompts. Release 0.11.1 added the Windows source build
([installation guide, "Windows x64 with TensorRT-RTX"](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/v0.11.1/docs/source/user_guide/getting_started/installation.md#windows-x64-with-tensorrt-rtx));
its support matrix lists Windows x64 as a Developer target.

Machine of the run recorded here: Windows 11 Home (26200), Core Ultra 9 386H, GeForce RTX 5080 Laptop GPU (compute
capability 12.0, 16 GB, driver 591.74), Visual Studio 2022 Build Tools (MSVC 19.44).

Result on that machine (2026-11-07): the build passed, and for all 8 prompts of the check (step 9) Edge-LLM's
greedy output had the same token ids as transformers fp16 (`results/check.md`). That check comes from one run of the
nine PowerShell blocks below, as written and in order, in an empty folder on that machine (the blocks of commit
42941fd, unchanged since); every block exited 0. pip's cache on that machine already held most of the Python wheels
from earlier runs.

This repository holds no NVIDIA binaries. TensorRT for RTX comes from NVIDIA's download page under the NVIDIA
TensorRT for RTX License Agreement, and CUDA from NVIDIA's redistributable archives: step 1 downloads the CUDA parts,
and in step 2 you download TensorRT for RTX yourself.
Performance numbers are not part of these notes.

## What you need

- Windows 10 or 11 x64 (this run: Windows 11) and an NVIDIA RTX GPU with a driver that supports CUDA 13 (this run:
  591.74, which `nvidia-smi` reports as CUDA 13.1; the CUDA 13.5 parts below run on it through CUDA minor version
  compatibility). Only compute capability 12.0 was tried. The CuTe DSL kernels of step 4 list compute capabilities
  8.0 to 12.1 (8.6 Ampere, 8.9 Ada Lovelace, 12.0 Blackwell), not 7.5 (Turing).
- Visual Studio 2022 with the C++ build tools (the Build Tools edition is enough).
- Git, Python 3.12 with the `py` launcher, and CMake 3.20 or newer on `PATH`.
- About 13 GB of free disk space for the folder (12.4 GB after step 9), plus room for pip's download cache, which
  pip keeps outside the folder (the PyTorch wheel of step 9 alone is 1.9 GB).

Open PowerShell in an empty folder, clone this repository into it
(`git clone https://github.com/john-rocky/tensorrt-edge-llm-windows-notes.git`; steps 8 and 9 use its `check`
folder), and run the blocks below in order from that folder. Each block can go into a new PowerShell window opened in the same
folder: a block sets everything it needs.

## 1. CUDA 13.5 from NVIDIA's redistributable archives

Instead of the CUDA Toolkit installer, this step takes only the parts the build needs from the
[redistributable archives](https://developer.download.nvidia.com/compute/cuda/redist/) (zip files, checked against the
sha256 in NVIDIA's manifest) into `.\cuda`. It does not touch the display driver.

```powershell
$ErrorActionPreference = "Stop"
$base = "https://developer.download.nvidia.com/compute/cuda/redist/"
$manifest = Invoke-RestMethod "${base}redistrib_13.5.0.json"
New-Item -ItemType Directory -Force cuda, downloads | Out-Null
foreach ($part in "cuda_nvcc", "libnvvm", "cuda_crt", "cuda_cudart", "cuda_nvrtc", "cccl", "libcurand", "cuda_profiler_api", "cuda_nvtx") {
    $entry = $manifest.$part."windows-x86_64"
    $zip = "downloads\" + (Split-Path $entry.relative_path -Leaf)
    curl.exe -sSL --retry 5 -o $zip ($base + $entry.relative_path)
    if ((Get-FileHash $zip -Algorithm SHA256).Hash -ne $entry.sha256.ToUpper()) { throw "sha256 mismatch: $zip" }
    tar -xf $zip -C downloads
    $dir = "downloads\" + [IO.Path]::GetFileNameWithoutExtension($zip)
    robocopy $dir cuda /E /NFL /NDL /NJH /NJS /NP | Out-Null
    Remove-Item -Recurse -Force $dir, $zip
}
.\cuda\bin\nvcc.exe --version
```

CUDA 13 places its runtime DLLs in `cuda\bin\x64` (`cuda\bin` holds `nvcc.exe` and the other tools), so the later
blocks put both folders on `PATH`.

## 2. TensorRT for RTX 1.7.1

Download `tensorrt-rtx-1.7.1.116-windows-amd64-cuda-13.4-release-external.zip` from
<https://developer.nvidia.com/tensorrt-rtx> (TensorRT for RTX 1.7, Windows, CUDA 13.4). The page states that
downloading and using the software means agreeing to the NVIDIA TensorRT for RTX License Agreement. NVIDIA's
prerequisites page says the download needs an NVIDIA Developer Program membership; on 2026-11-07 the link downloaded
without signing in. Save the zip in this folder, then:

```powershell
$ErrorActionPreference = "Stop"
tar -xf tensorrt-rtx-1.7.1.116-windows-amd64-cuda-13.4-release-external.zip
Get-AuthenticodeSignature .\TensorRT-RTX-1.7.1.116\bin\*.dll | Format-Table Status, Path -AutoSize
```

The zip unpacks to `TensorRT-RTX-1.7.1.116` (`bin`: the runtime and ONNX parser DLLs; `lib`: their import libraries;
`include`: the headers). The DLLs carry NVIDIA's signature (`Valid` above). `dumpbin /dependents` lists only Windows
system DLLs for them, so no CUDA DLL has to sit next to them.

The Edge-LLM support matrix names TensorRT RTX 1.7.1.80. NVIDIA's download page and PyPI offer 1.7.1.116 for the
1.7.1 release (checked 2026-11-07); this run used 1.7.1.116.

## 3. Source

```powershell
git clone --branch v0.11.1 --depth 1 https://github.com/NVIDIA/TensorRT-Edge-LLM.git
git -C TensorRT-Edge-LLM submodule update --init --recursive
git -C TensorRT-Edge-LLM rev-parse HEAD
```

`rev-parse` prints `358de11c7c5f30fdaac75f0726a3b2df7e842778` (tag v0.11.1).

## 4. CuTe DSL kernels

The CMake build links kernels that `kernelSrcs\build_cutedsl.py` compiles ahead of time. The repository ships no
Windows artifact, so this step generates it for the GPU (`sm_120` here). For another GPU, put its compute capability
in the four places that say `sm_120` or `120`: twice in this block and twice in the cmake line of step 5 (not tried
here).

```powershell
$ErrorActionPreference = "Stop"
py -3.12 -m venv .venv-cutedsl
.venv-cutedsl\Scripts\python -m pip install --upgrade pip
.venv-cutedsl\Scripts\python -m pip install --no-cache-dir --only-binary=:all: --index-url https://pypi.org/simple "nvidia-cutlass-dsl[cu13]==4.8.0" cupy-cuda13x==13.6.0 ninja==1.13.2
if ($LASTEXITCODE -ne 0) { throw "pip failed" }
$vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath
$env:PATH = "$PWD\.venv-cutedsl\Scripts;$PWD\cuda\bin;$PWD\cuda\bin\x64;$env:PATH"
$env:CUDA_PATH = "$PWD\cuda"
cmd /c "call `"$vs\VC\Auxiliary\Build\vcvars64.bat`" && cd TensorRT-Edge-LLM && `"$PWD\.venv-cutedsl\Scripts\python.exe`" kernelSrcs\build_cutedsl.py --kernels ALL --gpu_arch sm_120 --cuda-version 13.5 --clean"
if ($LASTEXITCODE -ne 0) { throw "build_cutedsl.py failed" }
Get-ChildItem TensorRT-Edge-LLM\cpp\kernels\cuteDSLArtifact\windows\x86_64\sm_120
```

`build_cutedsl.py` needs `lib.exe` (from the Visual Studio x64 environment, `vcvars64.bat`) and `nvcc` on `PATH`. The
folder it writes holds `cutedsl_x86_64.lib`, the headers and `metadata.json`.

Each new environment first updates pip. The pip that comes with Python 3.12.10 (25.0.1) cannot resume a broken
download. pip 26.3 resumes one up to 5 times by default (`--resume-retries`), which helps with the 1.9 GB PyTorch
wheel in step 9.

## 5. Build

```powershell
$ErrorActionPreference = "Stop"
$vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath
$trt = "$PWD\TensorRT-RTX-1.7.1.116".Replace("\", "/")
$env:PATH = "$PWD\.venv-cutedsl\Scripts;$PWD\cuda\bin;$PWD\cuda\bin\x64;$env:PATH"
$env:CUDA_PATH = "$PWD\cuda"
cmd /c "call `"$vs\VC\Auxiliary\Build\vcvars64.bat`" && cd TensorRT-Edge-LLM && cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DTRT_PACKAGE_DIR=$trt -DCUDA_CTK_VERSION=13.5 -DCMAKE_CUDA_ARCHITECTURES=120 -DENABLE_CUTE_DSL=ALL -DCUTE_DSL_ARTIFACT_TAG=sm_120 -DCMAKE_C_FLAGS_INIT=/utf-8 -DCMAKE_CXX_FLAGS_INIT=/utf-8 -DCMAKE_CUDA_FLAGS_INIT=-Xcompiler=/utf-8 && cmake --build build --parallel 8"
if ($LASTEXITCODE -ne 0) { throw "the build failed" }
$env:PATH = "$PWD\TensorRT-RTX-1.7.1.116\bin;$PWD\cuda\bin\x64;$PWD\TensorRT-Edge-LLM\build;$env:PATH"
TensorRT-Edge-LLM\build\examples\llm\llm_build.exe --help
TensorRT-Edge-LLM\build\examples\llm\llm_inference.exe --help
```

The three `*_FLAGS_INIT` options are not in the upstream guide. They make MSVC read the sources as UTF-8. Without
them, on this machine (Windows set to Japanese, ANSI code page 932) MSVC reads a UTF-8 character in xgrammar's
`tokenizer_info.cc` as Shift-JIS and stops with `error C2001` (newline in constant), and it warns C4819 on two Edge-LLM
headers. On a machine with code page 1252 the build may pass without them (not tried here).

The build puts the plugin `NvInfer_edgellm_plugin.dll` in `build` and the tools in `build\examples\llm`. On Windows
the `llm_stream` example is not built.

## 6. Export Qwen3-0.6B to ONNX

```powershell
$ErrorActionPreference = "Stop"
py -3.12 -m venv .venv-export
.venv-export\Scripts\python -m pip install --upgrade pip
.venv-export\Scripts\python -m pip install -e "TensorRT-Edge-LLM[export]"
if ($LASTEXITCODE -ne 0) { throw "pip failed" }
.venv-export\Scripts\hf download Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca --local-dir models\Qwen3-0.6B
if ($LASTEXITCODE -ne 0) { throw "the download failed" }
$env:PYTHONUTF8 = "1"
.venv-export\Scripts\trtedge export models\Qwen3-0.6B work\onnx
if ($LASTEXITCODE -ne 0) { throw "the export failed" }
Get-ChildItem work\onnx\llm
```

The export runs on the CPU and writes fp16 ONNX to `work\onnx\llm`.

## 7. Build the engine

```powershell
$ErrorActionPreference = "Stop"
$env:PATH = "$PWD\TensorRT-RTX-1.7.1.116\bin;$PWD\cuda\bin\x64;$PWD\TensorRT-Edge-LLM\build;$env:PATH"
$env:EDGELLM_PLUGIN_PATH = "$PWD\TensorRT-Edge-LLM\build\NvInfer_edgellm_plugin.dll"
TensorRT-Edge-LLM\build\examples\llm\llm_build.exe --onnx-dir work\onnx\llm --engine-dir work\engines\llm --max-batch-size 1 --max-input-len 1024 --max-kv-cache-capacity 2048
if ($LASTEXITCODE -ne 0) { throw "llm_build failed" }
Get-ChildItem work\engines\llm
```

For each of the model's 28 attention layers, `llm_build` prints two TensorRT warnings that the attributes
`xqa_jit_kernels` and `supports_bounded_kv_cache` are not found in the plugin node and that "the engine may fail to
build". The engine builds: the v0.11.1 attention plugin reads a missing `supports_bounded_kv_cache` as 0 and a missing
`xqa_jit_kernels` as empty (`cpp/plugins/attentionPlugin/attentionPlugin.cpp`, lines 688 and 721).

## 8. Generate

```powershell
$ErrorActionPreference = "Stop"
py -3.12 tensorrt-edge-llm-windows-notes\check\make_input.py tensorrt-edge-llm-windows-notes\check\prompts.json work\input.json
$env:PATH = "$PWD\TensorRT-RTX-1.7.1.116\bin;$PWD\cuda\bin\x64;$PWD\TensorRT-Edge-LLM\build;$env:PATH"
$env:EDGELLM_PLUGIN_PATH = "$PWD\TensorRT-Edge-LLM\build\NvInfer_edgellm_plugin.dll"
TensorRT-Edge-LLM\build\examples\llm\llm_inference.exe --engine-dir work\engines\llm --input-file work\input.json --output-file work\output.json
if ($LASTEXITCODE -ne 0) { throw "llm_inference failed" }
(Get-Content work\output.json -Raw -Encoding UTF8 | ConvertFrom-Json).responses | ForEach-Object { $_.output_text }
```

`make_input.py` writes the 8 prompts of `check\prompts.json` as one request file: chat template with thinking off,
temperature 0 and top-k 1 (greedy), at most 64 new tokens, and the top-2 log-probabilities of every step.

## 9. Check against transformers

`check\` holds 8 prompts and the scripts. The reference is transformers 5.14.1 (the version Edge-LLM pins) in fp16 on
the GPU, greedy, with the same chat template (thinking off) and at most 64 new tokens. A prompt agrees when the token
ids Edge-LLM generated are the same as the reference's.

```powershell
$ErrorActionPreference = "Stop"
py -3.12 -m venv .venv-ref
.venv-ref\Scripts\python -m pip install --upgrade pip
.venv-ref\Scripts\python -m pip install torch==2.13.0+cu130 --index-url https://download.pytorch.org/whl/cu130
if ($LASTEXITCODE -ne 0) { throw "pip failed (torch)" }
.venv-ref\Scripts\python -m pip install transformers==5.14.1 safetensors==0.8.0
if ($LASTEXITCODE -ne 0) { throw "pip failed" }
$env:PYTHONUTF8 = "1"
.venv-ref\Scripts\python tensorrt-edge-llm-windows-notes\check\ref_generate.py --model models\Qwen3-0.6B --prompts tensorrt-edge-llm-windows-notes\check\prompts.json --dtype float16 --device cuda --out work\ref_fp16.json
if ($LASTEXITCODE -ne 0) { throw "the reference run failed" }
.venv-ref\Scripts\python tensorrt-edge-llm-windows-notes\check\compare.py --edge work\output.json --ref work\ref_fp16.json --prompts tensorrt-edge-llm-windows-notes\check\prompts.json --model models\Qwen3-0.6B --out work\check.json --md work\check.md
if ($LASTEXITCODE -ne 0) { throw "compare.py failed" }
```

`compare.py` checks first that both sides fed the model the same prompt token ids (it re-tokenizes the
`formatted_complete_request` that `llm_inference` reports). It then compares the generated ids prompt by prompt, and
prints the table and the verdict (`PASS` = at least 7 of 8 prompts agree, no empty answer, and the prompt ids are
equal on all 8). Where a prompt parts ways,
the table gives the first differing step and the gap between the two most likely tokens there on each side.

## Problems met

| Step | What happened | What worked |
|---|---|---|
| 2 | The Edge-LLM support matrix names TensorRT RTX 1.7.1.80, which is on neither NVIDIA's download page (its 1.7 row offers 1.7.1.116 only) nor PyPI. | TensorRT for RTX 1.7.1.116 for CUDA 13.4. |
| 1, 5, 7 | The CUDA 13 zips put the runtime DLLs (`cudart64_13.dll`, `nvrtc64_130_0.dll`, `nvrtc-builtins64_135.dll`) in `cuda\bin\x64`, not in `cuda\bin`. | Put both folders on `PATH`. |
| 5 | On Windows set to Japanese (code page 932), MSVC stops in xgrammar's `tokenizer_info.cc` with `error C2001` (newline in constant). | `/utf-8` through the three `*_FLAGS_INIT` options in a new build folder. |

## Licences

The scripts and notes here: MIT (`LICENSE`). TensorRT Edge-LLM: Apache-2.0. TensorRT for RTX: NVIDIA TensorRT for RTX
License Agreement. CUDA: the CUDA Toolkit End User License Agreement (the `LICENSE.txt` in each archive; NVIDIA's
manifest marks cccl's as "CCCL EULA"). Qwen3-0.6B: Apache-2.0.
