"""Per-kernel-class decode cost on the M4, from Honeykrisp's launch hooks.

usage: decode_prof.py <repo> [top_n]

1. Counts Apple9 launches per class key (preamble offset / local x /
   resources / threadgroup bytes) per decoded token: HK_APPLE9_DEBUG launch
   lines from a 49-token run minus a 17-token run, over 32 tokens.
2. For the top classes by count, re-runs decode with HK_APPLE9_DUP=<key>
   (that class launched twice) and prices the class from the slowdown:
   added ms per token, and us per launch.
Generation is greedy; every run is its own process.
"""
import collections
import json
import os
import re
import subprocess
import sys

PY = os.path.expanduser("~/gpu/mlx-venv/bin/python")
GEN = r'''
import sys, time
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
m, tok = load(sys.argv[1])
p = tok.apply_chat_template([{"role": "user", "content": "Explain how a rainbow forms."}],
                            tokenize=False, add_generation_prompt=True)
n = int(sys.argv[2])
for r in stream_generate(m, tok, p, max_tokens=4, sampler=make_sampler(temp=0.0)):
    pass
sys.stderr.write("=== DECODE START\n"); sys.stderr.flush()
for r in stream_generate(m, tok, p, max_tokens=n, sampler=make_sampler(temp=0.0)):
    pass
sys.stderr.write("=== DECODE END\n"); sys.stderr.flush()
print(f"TPS {r.generation_tps:.3f}")
'''
LINE = re.compile(r"hk/apple9: (\S+) abi=(\d+) resources=(\d+) local=(\d+)x\d+x\d+ "
                  r"mode=\d+ threads=(\d+)x(\d+)x(\d+) preamble=(\d+)\+\d+ scratch=\d+ "
                  r"tgmem=(\d+)")


def run(repo, n, extra):
    env = dict(os.environ, **extra)
    r = subprocess.run([PY, "-c", GEN, repo, str(n)], capture_output=True, text=True, env=env)
    tps = float(re.search(r"TPS ([0-9.]+)", r.stdout).group(1))
    return tps, r.stderr


def launches(stderr):
    inside = False
    counts = collections.Counter()
    shape = {}
    for line in stderr.splitlines():
        if line.startswith("=== DECODE START"):
            inside = True
        elif line.startswith("=== DECODE END"):
            inside = False
        elif inside:
            m = LINE.match(line)
            if m:
                what, abi, res, lx, tx, ty, tz, pre, tg = m.groups()
                key = f"{pre}/{lx}/{res}/{tg}"
                counts[key] += 1
                shape.setdefault(key, f"abi={abi} threads={tx}x{ty}x{tz}")
    return counts, shape


def main():
    repo = sys.argv[1]
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    _, short = run(repo, 17, {"HK_APPLE9_DEBUG": "1"})
    _, long_ = run(repo, 49, {"HK_APPLE9_DEBUG": "1"})
    c_short, _ = launches(short)
    c_long, shape = launches(long_)
    per_token = {k: (c_long[k] - c_short.get(k, 0)) / 32.0 for k in c_long}
    total = sum(per_token.values())
    base = sorted(run(repo, 65, {})[0] for _ in range(3))[1]
    ms_token = 1000.0 / base
    print(json.dumps({"model": repo, "decode_tok_s": base, "ms_per_token": round(ms_token, 3),
                      "launches_per_token": round(total, 1)}))
    ranked = sorted(per_token.items(), key=lambda kv: -kv[1])[:top_n]
    rows = []
    for key, n in ranked:
        if n <= 0:
            continue
        # Paired: a fresh baseline next to each duplicated run, so load from
        # other work on the machine cancels; the median of three pairs.
        deltas = []
        for _ in range(3):
            b = run(repo, 65, {})[0]
            d = run(repo, 65, {"HK_APPLE9_DUP": key})[0]
            deltas.append(1000.0 / d - 1000.0 / b)
        added = sorted(deltas)[1]
        rows.append((added, key, n, shape[key]))
    for added, key, n, sh in sorted(rows, reverse=True):
        print(f"{key:>22}  {n:6.1f}/token  +{added:6.2f} ms/token  "
              f"{added * 1000.0 / n:7.1f} us/launch  {100.0 * added / ms_token:5.1f}%  {sh}",
              flush=True)


if __name__ == "__main__":
    main()
