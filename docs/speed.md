# Speed

This repository is independent work. It is not sponsored or endorsed by NVIDIA.

The numbers in "Speed" and "Compared with other runtimes" are published with NVIDIA's written permission under
Section 2.13 of the NVIDIA TensorRT for RTX License Agreement.

Machine of these runs: Windows 11 Home (26200), Core Ultra 9 386H, GeForce RTX 5080 Laptop GPU (compute
capability 12.0, 16 GB, driver 591.74), Visual Studio 2022 Build Tools (MSVC 19.44).

Step and block numbers are in [the steps](steps.md).

## Speed

After one warm-up request, `llm_inference` took 9.08 ms for the 19-token prompt and decoded at 369.0 tokens per
second. In the first request of a new process, the prompt took 19.59 ms and decode ran at 293.8 tokens per second.

| `llm_inference` run | prefill tok/s | prefill time (19 tokens) | decode tok/s | decode tok/s from the wall clock | spread (prefill / decode) |
|---|---|---|---|---|---|
| after one warm-up request (`--warmup 1`) | 2,092.7 | 9.08 ms | 369.0 | 367.5 | 2.6 % / 0.2 % |
| first request of the process (`--warmup 0`) | 972.9 | 19.59 ms | 293.8 | 292.5 | 11.0 % / 8.9 % |

Setup: the machine above (an MSI Stealth 16 AI+ B3WI laptop) on 2026-11-07, 18:03 to 18:07 JST, on AC power with
Windows' balanced power plan. `nvidia-smi -q -d POWER` read a power limit of 95 W for the GPU (default 80 W, maximum
125 W). TensorRT Edge-LLM v0.11.1, TensorRT for RTX 1.7.1.116 and CUDA 13.5.37 come from steps 1 to 5. The engine is
step 7's: fp16 weights, batch 1, at most 1,024 input tokens, a KV cache of 2,048 tokens.

Each row is three runs of block 10's command line, each a new process, 30 s apart. Run 1 is not counted. The row's
number is the median of runs 2 and 3, and the spread is (largest − smallest) ÷ that median, as in edge-llm-bench,
which marks a row over 5 % as `out`. The first-request row is over 5 %: its decode read 280.7 and 307.0 tok/s.
Before each run the CPU read under 25 % and the GPU had no memory in use. A script ran these runs and the llama.cpp
runs of the next section in one session, from 18:01 to 18:14 JST. Its record of each run (command line, start time,
the readings before the run), the profiles, the answers and llama.cpp's logs are in `results/speed/runs/`. NVIDIA's
permission covers the speed numbers, so the peak memory fields are removed from `llm_inference`'s profiles, and the
end time and duration from its run records. The machine's
readings at the start and the end of the session (`nvidia-smi -q`, the power plan) are in `results/speed/sitting_*.txt`,
and the values behind both tables, run by run, in `results/speed/summary.json`.

Every timed run gave the same answer text as step 11's run, whose 128 token ids are the same as transformers'
(`results/speed/check/check.json`).

How `llm_inference` counts (`examples/utils/profileFormatter.cpp`, `cpp/profiling/timer.cpp`): prefill tok/s is the
prompt tokens divided by the prefill stage's time. Decode tok/s is the 128 generated tokens divided by the decode
steps' time. The first of the 128 tokens comes out of the prefill, so there are 127 decode steps. Both times are CUDA
event times around each stage on the GPU stream, so they leave out the host's work between decode steps. The wall
clock column uses `llm_inference`'s wall-clock time for the request instead: 128 tokens divided by (wall-clock time
minus prefill time).

Run 1 of the first row found no runtime cache and wrote one at its end. The five runs after it loaded that cache.

## Compared with other runtimes

In the same session, with 16-bit weights on both sides, `llm_inference` after one warm-up request decoded at 369.0
tok/s (367.5 from the wall clock). llama.cpp decoded at 321.8 tok/s on CUDA and 312.6 tok/s on Vulkan. For the
19-token prompt, `llm_inference` took 9.08 ms, llama.cpp 59.86 ms on CUDA and 14.52 ms on Vulkan. With the Q4_K_M
file, llama.cpp on CUDA decoded at 502.7 and 488.9 tok/s, at the start and the end of the session.

Same session (2026-11-07, 18:01 to 18:14 JST, the setup above):

| Runtime and backend | Weights (file) | prefill tok/s | prefill time (19 tokens) | decode tok/s | spread (prefill / decode) |
|---|---|---|---|---|---|
| TensorRT Edge-LLM v0.11.1, TensorRT for RTX, after one warm-up request | fp16 (step 7's engine) | 2,092.7 | 9.08 ms | 369.0 | 2.6 % / 0.2 % |
| TensorRT Edge-LLM v0.11.1, TensorRT for RTX, first request of the process | fp16 (step 7's engine) | 972.9 | 19.59 ms | 293.8 | 11.0 % / 8.9 % |
| llama.cpp b11721, CUDA | F16 (`Qwen3-0.6B-F16.gguf`, 1,198 MB) | 317.7 | 59.86 ms | 321.8 | 5.8 % / 0.5 % |
| llama.cpp b11721, Vulkan | F16 (`Qwen3-0.6B-F16.gguf`) | 1,308.9 | 14.52 ms | 312.6 | 2.3 % / 0.8 % |
| llama.cpp b11721, CUDA, at the start | Q4_K_M (`Qwen3-0.6B-Q4_K_M.gguf`, 397 MB) | 879.0 | 21.62 ms | 502.7 | 3.5 % / 2.2 % |
| llama.cpp b11721, CUDA, at the end | Q4_K_M (`Qwen3-0.6B-Q4_K_M.gguf`) | 806.7 | 23.70 ms | 488.9 | 15.9 % / 2.2 % |

edge-llm-bench's session `2026-11-07-windows-qwen3` on the same laptop (2026-11-07, 03:57 to 04:10 JST,
[raw records](https://github.com/john-rocky/edge-llm-bench/tree/main/results/raw/2026-11-07-windows-qwen3)):

| Runtime and backend | Weights (file) | prefill tok/s | prefill time (19 tokens) | decode tok/s | spread (prefill / decode) |
|---|---|---|---|---|---|
| llama.cpp b11721, CUDA | Q4_K_M (`Qwen3-0.6B-Q4_K_M.gguf`) | 854.8 | 22.27 ms | 484.4 | 8.8 % / 0.2 % |
| llama.cpp b11721, Vulkan | Q4_K_M (`Qwen3-0.6B-Q4_K_M.gguf`) | 1,509.2 | 12.59 ms | 486.0 | 2.7 % / 0.7 % |
| llama.cpp b11721, CPU (16 threads), 3 runs counted | Q4_K_M (`Qwen3-0.6B-Q4_K_M.gguf`) | 767.1 | 24.77 ms | 119.7 | 3.9 % / 12.6 % |
| LiteRT-LM v0.19.0, GPU (WebGPU on Direct3D 12), second request of the process | `Qwen3-0.6B_dynamic_wi4b32_afp32.litertlm` (343 MB) | 1,127.0 | 16.86 ms | 457.8 | 3.1 % / 1.0 % |
| LiteRT-LM v0.19.0, GPU (WebGPU on Direct3D 12), first request of the process | `Qwen3-0.6B_dynamic_wi4b32_afp32.litertlm` | 69.4 | 275.44 ms | 217.5 | 14.7 % / 0.6 % |
| LiteRT-LM v0.19.0, CPU (4 threads), second request of the process | `Qwen3-0.6B_dynamic_wi4b32_afp32.litertlm` | 8.6 | 2,214.58 ms | 11.0 | 1.6 % / 8.2 % |
| LiteRT-LM v0.19.0, CPU (4 threads), first request of the process | `Qwen3-0.6B_dynamic_wi4b32_afp32.litertlm` | 18.0 | 1,053.65 ms | 13.0 | 2.6 % / 5.4 % |

All rows run Qwen/Qwen3-0.6B on the short-chat task (19 prompt tokens, greedy, 128 new tokens), and each counts runs
2 and 3 of three unless the row says otherwise. Edge-LLM and the F16 rows have 16-bit weights. The Q4_K_M and
`dynamic_wi4b32_afp32` files are quantized forms of the same model, made by unsloth and litert-community. The F16 file
is unsloth's BF16 GGUF (`unsloth/Qwen3-0.6B-GGUF`, revision 50968a44) converted with b11721's `llama-quantize`.

The runtimes do not count time the same way:

- `llm_inference`: CUDA event times on the GPU, as described in "Speed".
- llama.cpp: `llama-cli -st -lv 4`'s timing lines, on the host clock. The prompt time runs from the start of the prompt
  to the first sampled token. Decode tok/s is the 128 generated tokens over the time from the first token to the
  last, so it includes the host's work between steps. Before the prompt, `llama-cli` runs a warm-up by default (its
  log: "warming up the model with an empty run"), so its rows compare with Edge-LLM's row after one warm-up request.
- LiteRT-LM: its BenchmarkInfo, on the host clock: the prefill turn's tokens over its time, and the decode turn's
  tokens over its time. Each process ran two requests; the table gives the first and the second.

The llama.cpp GPU runs use edge-llm-bench's command line for its GPU rows: `llama-cli -lv 4 -m <file> -c 2048 -f
short-chat.txt -n 128 --temp 0 --top-p 1 -st -ngl 99 -fa on`, with b11721's release assets (the CUDA 13.4 asset and
the Vulkan asset). The Vulkan runs set `GGML_VK_VISIBLE_DEVICES=0`, which is the RTX in their log (`Vulkan0 : NVIDIA
GeForce RTX 5080 Laptop GPU`). The CPU row's log has only the rates, so its prefill time is 19 tokens divided by the
rate.

In edge-llm-bench's session the same Q4_K_M command decoded at 484.4 tok/s on CUDA, and in this session at 502.7 and
488.9 tok/s. A first session the same evening (17:29 to 17:41 JST) is not used: the same command decoded at 338.9
tok/s there, and the cause was not found. The llama.cpp F16 answers were not compared with transformers.
edge-llm-bench's own text check passed on every run of its rows (its records).
