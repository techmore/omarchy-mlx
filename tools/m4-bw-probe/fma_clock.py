"""Infer the M4 GPU's running clock from an FMA-bound kernel.

16 independent fma chains per thread, ITER iterations, 1M threads.
clock ~= flops / (1280 lanes * 2 flops/fma). Also re-times after a 3 s
warm load, to see whether the clock ramps under sustained work.
"""
import time

import mlx.core as mx

ITER = 2048
CHAINS = 16
body = "uint i = thread_position_in_grid.x;\n float s = inp[i & 1023u];\n"
body += "".join(f" float a{c} = s + {c}.0f;\n" for c in range(CHAINS))
body += f" for (int k = 0; k < {ITER}; ++k) {{\n"
body += "".join(f"  a{c} = fma(a{c}, 0.9999f, 0.0001f);\n" for c in range(CHAINS))
body += " }\n out[i] = " + " + ".join(f"a{c}" for c in range(CHAINS)) + ";\n"
k = mx.fast.metal_kernel(name="fma_clock", input_names=["inp"], output_names=["out"], source=body)
THREADS = 1 << 20
inp = mx.random.normal((1024,))
mx.eval(inp)


def once():
    out = k(inputs=[inp], grid=(THREADS, 1, 1), threadgroup=(256, 1, 1),
            output_shapes=[(THREADS,)], output_dtypes=[mx.float32])[0]
    mx.eval(out)
    return out


def rate():
    once()
    ts = []
    for _ in range(7):
        t = time.perf_counter()
        once()
        ts.append(time.perf_counter() - t)
    t = sorted(ts)[3]
    flops = THREADS * ITER * CHAINS * 2
    return flops / t / 1e9, t


g, t = rate()
print(f"cold:  {g:7.1f} GFLOP/s ({t * 1e3:.2f} ms)  -> ~{g / 2.56:.0f} MHz if 1280 lanes", flush=True)
end = time.perf_counter() + 3
while time.perf_counter() < end:
    once()
g, t = rate()
print(f"warm:  {g:7.1f} GFLOP/s ({t * 1e3:.2f} ms)  -> ~{g / 2.56:.0f} MHz if 1280 lanes", flush=True)
