# 2026-11-01 — M4 decode bandwidth: where the 4-bit GEMV's time goes

Host: M4 Mac mini (Mac16,10, T8132, `Apple M4 (G16G B1)`, 16 GB), Omarchy 4.0,
`linux-gravity-m4` 7.1.13. Vulkan Honeykrisp from techmore/mesa
`m4-honeykrisp-compute` (`a8e7b48`); this branch at `83ae764`; mlx-lm 0.32.0
with the omarchy serve patches. Probe sources: `tools/m4-bw-probe/`.

## Decode profile (granite-4.0-1b-4bit, 44.9 tok/s, 22.3 ms/token)

`HK_APPLE9_DEBUG` launch counts per token and `HK_APPLE9_DUP` cost per class:

| class (preamble/local/resources/tgmem) | launches/token | ms/token | us/launch | share |
|---|---:|---:|---:|---:|
| 4612/128/6/0 (`qmm_vec` Q4 word subgroup bf16) | 281 | 14.60 | 52.0 | 65.5% |
| 41670/256/4/0 | 80 | 2.93 | 36.6 | 13.1% |
| 12240/1024/6/32768 | 40 | 1.58 | 39.5 | 7.1% |
| 4312/256/5/128 | 81 | 0.58 | 7.1 | 2.6% |
| 0/256/5/0 | 83 | 0.43 | 5.1 | 1.9% |

771 launches per token. The GEMV moves ~0.92 GB per token, so it streams at
about 63 GB/s in model.

## Ceilings

Streaming read (`vkbw` + `bw_read.comp`, xor-reduce, 512 MiB-1 GiB):

| load | GB/s |
|---|---:|
| 32-bit | 66-71 |
| 64-bit | 74-77 |
| 128-bit | 78-82 |

Flat across memory types 0 (device-local, host-visible, coherent, cached)
and 1 (not cached), occupancy 10-5120 workgroups of 256, 1-8 loads in flight
per thread, and CPU frequency 0.9-4.1 GHz (cores spun up during the run).
Below 256 MiB the fixed cost shows (64 MiB: 62.8 GB/s).

CPU read (`cpubw.c`, NEON, 1 GiB): 1 thread 33 GB/s, 4 threads 55, 10
threads 71.

Reference: the base M4 reaches about 100 GB/s under macOS (Apple vs.
Oranges, arXiv 2502.05317, of a 120 GB/s rating). That paper dates from
February 2025, so its figures come from an older macOS release than the
current 27.2. Once Linux meets or exceeds them, re-baseline: run the same
probes (Metal ports of `tools/m4-bw-probe`) on an M4 under macOS 27.2. The device tree has no
memory power-management node and nothing on Linux requests a memory
performance state; that is the likely system gap.

Dispatch cost (`vkchain.c`, N dispatches in one command buffer): a 1 MiB
read takes 15.4 us per dispatch against 12.8 us of bandwidth, so ~2.6 us
fixed; pipeline barriers between dispatches add nothing; 8 MiB dispatches
run at 81 GB/s.

## The GEMV against its own access pattern

`gemv_sweep.py` (production `quantized_matmul`, independent copies, one
graph), before and after the in-place nibble decode:

| N x K | before us | after us | after GB/s |
|---|---:|---:|---:|
| 64 x 2048 | 20.2 | 19.3 | 3.8 |
| 512 x 2048 | 27.0 | 26.0 | 22.7 |
| 2048 x 512 | 33.5 | 28.2 | 20.9 |
| 2048 x 2048 | 52.5 | 51.2 | 46.1 |
| 2048 x 8192 | 147.4 | 146.0 | 64.6 |
| 8192 x 2048 | 155.2 | 152.6 | 61.8 |

`bw_gemv.comp` reproduces the GEMV's memory access mix with the arithmetic
replaced by xor: per lane per iteration one 64-bit weight load per row (two
rows), two uvec4 x loads, a 16-bit scale and bias per row, all device
loads. It reads 85 GB/s, and:

| variant | GB/s |
|---|---:|
| as the GEMV (64-bit weights) | 84-85 |
| 128-bit weight loads, same bytes | 81 |
| one-shot workgroups (8 rows each) | 82-84 |
| 24 -> 58 GPRs (live values added) | 83.7 -> 83.3 |
| + 16 / 64 / 128 dependent fmas per row per iteration | 81.7 / 81.6 / 81.3 |

So neither the access pattern, the load width, the register count, the
workgroup lifetime nor dependent latency explains the GEMV's 62-66 GB/s. What
remains is throughput overlap: ~225 instructions per iteration (512 weight
bytes per SIMD group) alternate with the loads instead of hiding them.

## Issue rates (`op_rate.py`, 16 independent chains, G lane-ops/s)

| fma | fadd | fmul | iadd | iand | ior | ushr/ishl | imul | u32->f32 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1231 | 1368 | 1368 | 1288 | ~1550 | 613 | ~800 | 495 | ~660 |

(`iand`, shift and conversion rows are derived from two-op statements.)

## Change: in-place nibble decode (`83ae764`)

The magic-number decode shifted each field down, masked it, and or-ed in
the exponent of 2^23. Masked in place at bit s, the field is the mantissa
of 2^(23 - s) + n, so subtracting 2^(23 - s) gives n exactly, and the
exponent goes in with an add (disjoint bits). The GEMV loop's 32 right
shifts drop to 7 and its 39 ors to none. 90/90 decode `quantized_matmul`
cases (`qmv_bits.py`: 4-bit g64/g32, 8-bit, bf16/f16, 1-4 rows, aligned and
odd shapes) are bit-identical. Decode, 32 tokens, two runs each:

| model | before | after |
|---|---:|---:|
| granite-4.0-1b-4bit | 47.3 | 48.0 |
| Qwen2.5-1.5B-Instruct-4bit | 60.9 | 62.2 |
| Mistral-7B-Instruct-v0.3-4bit | 18.8 | 19.2 |
| Llama-3.2-1B-Instruct-8bit | 28.6 | 28.8 |

## Tried: software prefetch (no gain)

Loading the next iteration's weight words before the current iteration's
math in `Q4_ROWS`, and the block's scale and bias at the top of the loop
(load order only), stayed bit-identical on all 90 cases but moved GEMV time
by -3% to +3% (2048 x 8192: 146.0 -> 151.5 us; 8192 x 2048: 152.6 -> 150.4
us). Not kept: the backend's scheduler already issues these loads early.

## Short kernels pay for their code size

Two synthetic kernels doing the same integer work, one unrolled
(straight-line) and one as a loop, on G16 (`gen_icache.py`, `vkchain`, 4
workgroups, the same shader dispatched 300 times in a row):

| ops per thread | unrolled | code | looped (~340 B) |
|---:|---:|---:|---:|
| 250 | 10.7 us | 9.6 KB | 4.9 us |
| 1000 | 34.7 us | 38 KB | 11.8 us |
| 4000 | 127 us | 152 KB | 39.6 us |

With 160 or more workgroups the unrolled kernel is as fast as the looped
one, so the cost is a cold fetch of the code at the start of each
dispatch, about 0.9 us per KB, even for a 9.6 KB kernel that should fit in
the cache. Mesa's launch records do not request an instruction-cache
flush, so the invalidation (or the slow fetch) comes from below the driver.
Short, large kernels pay it at every launch: decode RoPE compiled to
41.7 KB (one shader carrying every RoPE form behind runtime flags) and cost
~37 us per launch for 256-1024 threads.

## Change: half-split RoPE builds

FastRope{,Freqs}{BF16,F16} carry the half-split, traditional and
passthrough forms behind runtime flags. Builds with
`-DROPE_HALF_SPLIT_ONLY` fix the traditional and passthrough flags false so
those branches fold away (bf16: 41.7 KB / 4402 instructions -> 16.8 KB /
1778), routed when the flags select the half-split form
(`MLX_OMARCHY_NO_ROPE_HALF_SPLIT=1` keeps the general builds). 72/72 rope
cases (`rope_bits.py`: bf16/f16, base/freqs, traditional or not, decode and
17-token shapes, offsets 0/37/4093) are bit-identical. Decode, 32 tokens,
two runs each, general -> half-split:

| model | tok/s |
|---|---:|
| granite-4.0-1b-4bit | 47.9 -> 51.8 |
| Qwen2.5-0.5B-Instruct-4bit | 116.5 -> 121.8 |
| Qwen2.5-1.5B-Instruct-4bit | 62.2 -> 63.8 |
| gemma-3-1b-it-4bit | 53.9 -> 55.0 |
| Llama-3.2-3B-Instruct-4bit | 15.9 -> 16.1 |
| Llama-3.2-1B-Instruct-4bit | within noise |

## Change: bf16 decode attention on 256 lanes

The composition-exact bf16 `sdpa_decode_native` arm ran 1024-lane
workgroups, but only lanes 0..255 do softmax work and lanes below the head
dimension produce outputs; the rest computed a quarter of the scores and
waited at ~22 workgroup barriers. 256 lanes striding the keys by 256 give
the same per-key chains: 30/30 decode SDPA cases (`sdpa_bits.py`: bf16/f16,
head dim 64/128/256, 1-600 keys) are bit-identical. bf16 decode SDPA, 16
heads x 128: 98 -> 90 us at 40 keys, 152 -> 135 us at 400 (f16: 26 / 47).
Decode: granite 1B 51.8 -> 52.4 tok/s, Qwen3.5 4B 21.3 -> 21.6.

Tried and not kept, both bit-identical: rolling the arm's outer loops
(12.3 -> 7.1 KB, but 98 -> 121 us: the unrolled loads overlap), and
finishing both reduction trees in one subgroup with shuffles (neutral
within the +-7% noise at 40 keys).

## Next

Shrink the other large short kernels: bf16 SdpaDecodeNative Hd128/Hd256
(12.3 KB, against 1.5-2.2 KB for f16) and FastRopeNormAppleBF16 (31.5 KB,
Qwen3.5). Keep cutting instructions per weight byte, and look at what the in-model
GEMV does that the mimic does not (the subgroup reduction and store, the
real dependency between each load and its decode). The system ceiling
(82-85 vs ~100 GB/s on macOS) is the other half.
