"""bf16 linear-layer matmul (x @ W.T) GFLOP/s at Llama 3.2 1B shapes.

Run once with HK_APPLE9_COOPMAT=0 (scalar-FMA route) and once with the
default (cooperative-matrix route).
"""
import time

import mlx.core as mx

SHAPES = [(2048, 2048), (512, 2048), (8192, 2048), (2048, 8192)]  # (N out, K in)
for m in (128, 512, 2048):
    cells = []
    for n, k in SHAPES:
        x = mx.random.normal((m, k)).astype(mx.bfloat16)
        ws = [mx.random.normal((n, k)).astype(mx.bfloat16) for _ in range(8)]
        mx.eval(x, ws)

        def run():
            mx.eval([x @ w.T for w in ws])

        run()
        ts = []
        for _ in range(5):
            t = time.perf_counter()
            run()
            ts.append(time.perf_counter() - t)
        t = sorted(ts)[2] / len(ws)
        cells.append(f"{n}x{k}: {2 * m * n * k / t / 1e9:6.0f}")
    print(f"M={m:5d}  " + "  ".join(cells), flush=True)
