"""Prefill tok/s at a fixed prompt length: one warmup, best of three.

usage: prefill_bench.py <repo> <prompt_tokens>
"""
import sys
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

repo, n = sys.argv[1], int(sys.argv[2])
model, tok = load(repo)
text = ("The quick brown fox jumps over the lazy dog while the rain falls "
        "on the old stone bridge. ") * 200
ids = mx.array(tok.encode(text)[:n])[None]


def run():
    cache = make_prompt_cache(model)
    t = time.perf_counter()
    y = model(ids, cache=cache)
    mx.eval(y)
    return time.perf_counter() - t


run()
best = min(run() for _ in range(3))
name = repo.split("/")[-1]
print(f"{name} prompt={n}: {n / best:.1f} tok/s ({best * 1e3:.0f} ms)", flush=True)
