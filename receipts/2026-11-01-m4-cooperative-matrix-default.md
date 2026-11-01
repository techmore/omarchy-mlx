# 2026-11-01 — M4: cooperative matrix on by default; bf16 linear layers keep FMA

Host: M4 Mac mini (Mac16,10, T8132, `Apple M4 (G16G B1)`, 16 GB), Omarchy 4.0,
`linux-gravity-m4` 7.1.13. Vulkan Honeykrisp from techmore/mesa
`m4-honeykrisp-compute`, which now advertises `VK_KHR_cooperative_matrix`
(fp32 8x8x8, subgroup scope) on Apple9 by default (`HK_APPLE9_COOPMAT=0`
turns it off). mlx-lm 0.31.3.

## Change

With the cooperative matrix present, `matmul` sent bf16 linear layers
(row-major lhs, transposed rhs, alpha 1) to `MatmulBF16Coopmat` and kept
`MatmulBf16Fma` for drivers without it. On G16 the FMA kernel is faster, so
when both qualify on a device whose name contains `G16`, the FMA route wins.
Attention-shaped matmuls (alpha != 1, other orientations) keep the
cooperative matrix. Other devices are unchanged.

## Why: bf16 linear-layer matmul, GFLOP/s (`bf16_linear.py`, x @ W.T)

| m | route | 2048x2048 | 512x2048 | 8192x2048 | 2048x8192 |
|---:|---|---:|---:|---:|---:|
| 128 | FMA | 834 | 463 | 1121 | 845 |
| 128 | cooperative matrix | 599 | 420 | 689 | 626 |
| 512 | FMA | 1115 | 824 | 1174 | 1131 |
| 512 | cooperative matrix | 690 | 596 | 720 | 708 |
| 2048 | FMA | 1175 | 1112 | 1197 | 1177 |
| 2048 | cooperative matrix | 720 | 690 | 729 | 733 |

Before this change, turning the cooperative matrix on cost Llama 3.2 1B
bf16 prefill 26% at 128 prompt tokens and 18% at 512, and gained 31% at
2048 (attention).

## After: Llama 3.2 1B bf16 prefill (`prefill_bench.py`, best of three)

| prompt tokens | cooperative matrix off | on (this change) |
|---:|---:|---:|
| 128 | 335.2 | 358.0 |
| 512 | 326.5 | 413.9 |
| 2048 | 193.2 | 379.8 |

## Every model, cooperative matrix off vs on (512-token prefill, 32-token decode)

| model | prefill off | prefill on | decode off | decode on |
|---|---:|---:|---:|---:|
| granite-4.0-1b-4bit | 171.8 | 374.9 | 47.15 | 47.15 |
| Llama-3.2-1B-Instruct-4bit | 317.6 | 475.7 | 32.35 | 31.68 |
| Qwen2.5-1.5B-Instruct-4bit | 255.8 | 385.8 | 53.03 | 61.50 |
| Llama-3.2-1B-Instruct-bf16 | 326.3 | 412.8 | 19.92 | 20.00 |
| Llama-3.2-1B-Instruct-8bit | 29.1 | 481.4 | 28.11 | 28.58 |
| Qwen3.5-2B-MLX-4bit | 64.2 | 73.9 | 34.28 | 34.29 |
| gemma-3-1b-it-4bit | 309.4 | 415.5 | 51.58 | 51.55 |
| Mistral-7B-Instruct-v0.3-4bit | 58.6 | 90.4 | 17.23 | 18.81 |

Greedy text (48 tokens after a ~440-token prompt) is identical with the
cooperative matrix off and on for all 13 cached models that load; no
kernel GPU messages. LFM3 does not load on mlx-lm 0.31.3 (no `lfm3`
model module), independent of this change.
