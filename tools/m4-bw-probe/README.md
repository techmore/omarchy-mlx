# m4-bw-probe: memory, dispatch and ALU probes for Apple G16 (M4)

Measurements behind `receipts/2026-11-01-m4-decode-bandwidth.md`. Build on the
target with the Vulkan headers on the include path:

    gcc -O2 -o vkbw vkbw.c -lvulkan
    gcc -O2 -o vkchain vkchain.c -lvulkan
    gcc -O2 -pthread -o cpubw cpubw.c
    glslangValidator -V --target-env vulkan1.3 -DVEC=4 -DUNROLL=4 bw_read.comp -o read_v4_u4.spv
    glslangValidator -V --target-env vulkan1.3 -DWIDE=0 bw_gemv.comp -o gemv_w0.spv

| file | what it measures |
|---|---|
| `vkbw.c` | read bandwidth of a shader over a large buffer: `vkbw <spv> <vec> <groups> [mem_type\|-1] [MiB] [reps]` |
| `bw_read.comp` | streaming xor-reduce; `VEC` 1/2/4 (32/64/128-bit loads), `UNROLL` loads in flight |
| `bw_gemv.comp` | the 4-bit decode GEMV's access mix without its math; `WIDE`, `EXTRA` (live registers), `ALU` (dependent fmas) |
| `vkchain.c` | N dispatches per command buffer, with or without barriers: per-dispatch fixed cost |
| `cpubw.c` | multi-threaded CPU read bandwidth |
| `op_rate.py` | per-op issue rate through `mx.fast.metal_kernel` |
| `fma_clock.py` | FMA-bound throughput (and implied clock) |
| `gemv_sweep.py`, `gemv_bw.py` | production `quantized_matmul` decode GEMV time and GB/s by shape |
| `qmv_bits.py` | dump / compare decode `quantized_matmul` outputs bit for bit across wheels |
| `bf16_linear.py` | bf16 linear-layer matmul GFLOP/s (FMA vs cooperative-matrix routes) |
| `decode_prof.py` | per-kernel-class decode cost from `HK_APPLE9_DEBUG` / `HK_APPLE9_DUP` |
| `prefill_bench.py` | prefill tok/s at a fixed prompt length |
