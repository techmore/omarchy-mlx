# 2026-11-01 — M4 decode: fused GEMV and trio default off on G16

Host: M4 Mac mini (Mac16,10, T8132, `Apple M4 (G16G B1)`, 16 GB), Omarchy 4.0,
`linux-gravity-m4` 7.1.13, Vulkan Honeykrisp from techmore/mesa
`m4-honeykrisp-compute` at `a1a094b` (Apple9 compute port with descriptor-table
buffer roots published from the preamble). No cooperative matrix. mlx-lm
0.31.3 (unpatched venv; both arms share it). One wheel built from this branch;
the old default is reproduced with `MLX_OMARCHY_FUSED_GEMV=1
MLX_OMARCHY_FUSED_TRIO=1`.

## Change

`fused_gemv_enabled()` and `fused_trio_enabled()` default to off when the
device name contains `G16`. An explicit environment setting still wins in
both directions. Other devices are unchanged.

## Why

The fused decode GEMV group binds 26 resources and runs through Honeykrisp's
descriptor-table ABI on Apple9. Even with the buffer roots published from the
preamble (which made that kernel's loads overlap and gave +8-14% decode), the
per-node GEMV kernels and standalone trio kernels decode faster on this GPU.

## Decode tok/s (`llm_survey.py`, 32 new tokens, greedy, one process per cell)

| model | fused (old default) | new default | change |
|---|---:|---:|---:|
| granite-4.0-1b-4bit | 36.56 | 45.73 | +25% |
| Qwen2.5-0.5B-Instruct-4bit | 68.78 | 87.82 | +28% |
| Qwen2.5-1.5B-Instruct-4bit | 36.60 | 50.18 | +37% |
| Llama-3.2-1B-Instruct-4bit | 28.28 | 30.22 | +7% |
| Llama-3.2-3B-Instruct-4bit | 11.57 | 13.71 | +18% |
| Qwen3.5-2B-MLX-4bit | 26.11 | 33.03 | +27% |
| Mistral-7B-Instruct-v0.3-4bit | 14.76 | 16.72 | +13% |

Greedy text (96 tokens, chat prompt) is identical between the two defaults on
six models. Qwen2.5-1.5B diverges from word 41 into a different, coherent
sentence (accumulation order of the fused and per-node kernels); the per-node
text matches every earlier per-node run on this host. The kernel logged no GPU
faults.

Not measured: G13 hosts (unchanged by construction), prefill (the fused
routes are decode-only).
