#!/bin/bash
# Per model: kernels in one decode step with launch count and code size.
cd ~/gpu
I=$HOME/gpu/mesa-install
export VK_ICD_FILENAMES=$I/share/vulkan/icd.d/asahi_icd.aarch64.json LD_LIBRARY_PATH=$I/lib AGX_APPLE9_MAX_SHARED=32768 MLX_OMARCHY_BATCH_WORK=120000 HF_HUB_DISABLE_PROGRESS_BARS=1
NAMES=$(python3 - <<PY
import re
s = open("omarchy-mlx/overlay/mlx/backend/omarchy/compute.h").read()
i = s.index("enum class ComputeKernel")
body = re.sub(r"//[^\n]*", "", s[s.index("{", i)+1 : s.index("};", i)])
print(" ".join(t.strip().split("=")[0].strip() for t in body.split(",") if t.strip()))
PY
)
for m in "$@"; do
  echo "== $m"
  timeout 900 ~/gpu/mlx-venv/bin/python name_classes.py mlx-community/$m 2>/dev/null | grep "DISPATCH kernel=" | python3 -c "
import sys, re, collections
names = '''$NAMES'''.split()
agg = collections.Counter(); size = {}
for ln in sys.stdin:
    m = re.match(r'\s*(\d+)\s+(\d+)/\S+\s+\[rtmod\] DISPATCH kernel=(\d+)', ln)
    if m:
        c, body, k = int(m.group(1)), int(m.group(2)), int(m.group(3))
        n = names[k] if k < len(names) else str(k)
        agg[n] += c; size[n] = body
for n, c in sorted(agg.items(), key=lambda kv: -kv[1] * max(size[kv[0]], 1))[:9]:
    print(f'  {c:4d} launches  {size[n]/1024:6.1f} KB  {n}')
"
done
