"""Dump or compare decode SDPA outputs bit for bit, and time decode SDPA.

usage: sdpa_bits.py dump|compare <file.npz>
Decode shapes: q (1, H, 1, D), k/v (1, KV, L, D), bf16 and f16, D 64/128/256,
L from 1 to 600.
"""
import sys
import time

import mlx.core as mx
import numpy as np

mx.random.seed(11)
cases = []
for dt in (mx.bfloat16, mx.float16):
    for h, kv, d in ((16, 4, 128), (8, 2, 256), (32, 8, 64)):
        for L in (1, 7, 33, 100, 600):
            cases.append((dt, h, kv, d, L))
out = {}
for i, (dt, h, kv, d, L) in enumerate(cases):
    q = mx.random.normal((1, h, 1, d)).astype(dt)
    k = mx.random.normal((1, kv, L, d)).astype(dt)
    v = mx.random.normal((1, kv, L, d)).astype(dt)
    y = mx.fast.scaled_dot_product_attention(q, k, v, scale=d ** -0.5)
    mx.eval(y)
    out[f"c{i}"] = np.array(y.view(mx.uint16))

mode, path = sys.argv[1], sys.argv[2]
if mode == "dump":
    np.savez(path, **out)
    print(f"dumped {len(out)} cases")
else:
    ref = np.load(path)
    bad = [k for k in out if not np.array_equal(out[k], ref[k])]
    for k in bad[:8]:
        print("MISMATCH", cases[int(k[1:])])
    print(f"{len(out) - len(bad)}/{len(out)} cases bit-identical")

for dt in (mx.bfloat16, mx.float16):
    for L in (40, 400):
        q = mx.random.normal((1, 16, 1, 128)).astype(dt)
        k = mx.random.normal((1, 4, L, 128)).astype(dt)
        v = mx.random.normal((1, 4, L, 128)).astype(dt)
        mx.eval(q, k, v)
        run = lambda: mx.eval([mx.fast.scaled_dot_product_attention(q * (1.0 + i * 1e-3), k, v, scale=128 ** -0.5)
                               for i in range(200)])
        run()
        t0 = time.perf_counter(); run(); t = time.perf_counter() - t0
        print(f"{str(dt):14s} L={L:4d} sdpa+scale {t / 200 * 1e6:6.1f} us")
