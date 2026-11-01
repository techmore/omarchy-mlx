"""Decode GEMV bandwidth on the production MLX path, at granite-1B shapes.

For each (N, K): COPIES independent 4-bit/group-64 bf16 weights (enough that
the set streams from DRAM), one x row; one graph issues every
quantized_matmul and sums the outputs. GB/s = weight+scale+bias bytes / time.
Also a streaming read ceiling (sum of a large bf16 array).
usage: gemv_bw.py [reps]
"""
import sys
import time

import mlx.core as mx

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
SHAPES = [(2048, 2048), (512, 2048), (4096, 2048), (8192, 2048), (2048, 4096),
          (100352, 2048)]
TARGET_BYTES = 768 << 20


def timed(fn):
    fn()
    ts = []
    for _ in range(REPS):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return sorted(ts)[len(ts) // 2]


def main():
    a = mx.random.normal((TARGET_BYTES // 2,)).astype(mx.bfloat16)
    mx.eval(a)
    t = timed(lambda: mx.eval(mx.sum(a)))
    print(f"read ceiling (sum bf16 {TARGET_BYTES >> 20} MB): {TARGET_BYTES / t / 1e9:6.1f} GB/s",
          flush=True)
    del a
    x = mx.random.normal((1, 2048)).astype(mx.bfloat16)
    x4 = mx.random.normal((1, 4096)).astype(mx.bfloat16)
    for n, k in SHAPES:
        wbytes = n * k // 2 + 2 * 2 * n * (k // 64)
        copies = max(1, min(256, TARGET_BYTES // wbytes))
        ws = []
        for _ in range(copies):
            w = mx.random.normal((n, k)).astype(mx.bfloat16)
            q = mx.quantize(w, group_size=64, bits=4)
            mx.eval(q)
            ws.append(q)
        xi = x if k == 2048 else x4

        def run():
            outs = [mx.quantized_matmul(xi, wq, s, b, transpose=True,
                                        group_size=64, bits=4)
                    for wq, s, b in ws]
            mx.eval(sum(o.sum() for o in outs))

        t = timed(run)
        gbs = wbytes * copies / t / 1e9
        print(f"N={n:6d} K={k:5d} copies={copies:4d}  {t / copies * 1e6:7.1f} us/gemv  "
              f"{gbs:6.1f} GB/s", flush=True)
        del ws


if __name__ == "__main__":
    main()
