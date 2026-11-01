"""Pair MLX_OMARCHY_TRACE_DISPATCH kernel names with HK_APPLE9_DEBUG launch
classes during one decode step of a model; print counts per (name, class)."""
import collections, os, re, subprocess, sys
repo = sys.argv[1]
GEN = r'''
import sys
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
m, tok = load(sys.argv[1])
p = tok.apply_chat_template([{"role": "user", "content": "Hi"}], tokenize=False, add_generation_prompt=True)
n = 0
for r in stream_generate(m, tok, p, max_tokens=6, sampler=make_sampler(temp=0.0)):
    n += 1
    if n == 4:
        sys.stderr.write("=== STEP START\n"); sys.stderr.flush()
    if n == 5:
        sys.stderr.write("=== STEP END\n"); sys.stderr.flush()
'''
env = dict(os.environ, MLX_OMARCHY_TRACE_DISPATCH="1", HK_APPLE9_DEBUG="1")
r = subprocess.run([os.path.expanduser("~/gpu/mlx-venv/bin/python"), "-c", GEN, repo],
                   capture_output=True, text=True, env=env)
lines = r.stderr.splitlines()
inside = False
LAUNCH = re.compile(r"hk/apple9: (\S+) abi=\d+ resources=(\d+) local=(\d+)x\d+x\d+ .*preamble=(\d+)\+\d+ .*tgmem=(\d+)")
last_name = None
pairs = collections.Counter()
names = collections.Counter()
for ln in lines:
    if ln.startswith("=== STEP START"):
        inside = True; continue
    if ln.startswith("=== STEP END"):
        inside = False; continue
    if not inside:
        continue
    m = LAUNCH.match(ln)
    if m:
        key = f"{m.group(4)}/{m.group(3)}/{m.group(2)}/{m.group(5)}"
        pairs[(last_name, key)] += 1
    elif "omarchy" in ln.lower() or "dispatch" in ln.lower():
        last_name = ln.strip()[:110]
        names[last_name] += 1
print("sample trace lines:")
for ln in [l for l in lines if "TRACE" in l or "dispatch" in l.lower()][:6]:
    print("  ", ln[:160])
print("\n(name, class) counts for one decode step:")
for (n, k), c in pairs.most_common(25):
    print(f"{c:5d}  {k:24s} {n}")
