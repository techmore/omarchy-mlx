#!/bin/bash
# PMP before/after: GPU and CPU read bandwidth, decode and prefill.
# usage: pmp_ab.sh <label>   (writes ~/gpu/pmp_ab/<label>.txt)
cd ~/gpu
I=$HOME/gpu/mesa-install
export VK_ICD_FILENAMES=$I/share/vulkan/icd.d/asahi_icd.aarch64.json LD_LIBRARY_PATH=$I/lib
export AGX_APPLE9_MAX_SHARED=32768 MLX_OMARCHY_BATCH_WORK=120000 HF_HUB_DISABLE_PROGRESS_BARS=1 MLX_OMARCHY_NO_MLXLM_WARNING=1
mkdir -p pmp_ab; out=pmp_ab/$1.txt
{
  echo "kernel $(uname -r)"; echo "cmdline $(cat /proc/cmdline)"
  echo "pmp driver: $(ls /sys/bus/platform/drivers/apple_pmp | grep -c @) bound, report: $(ls /sys/bus/platform/drivers/apple-pmp-report | grep -c @) bound"
  for r in 1 2 3; do
    echo "gpu read 128-bit: $(bw/vkbw bw/read_v4_u4.spv 4 1280 -1 1024 5 2>&1 | tail -1)"
    echo "gpu read  32-bit: $(bw/vkbw bw/read_v1_u4.spv 1 1280 -1 1024 5 2>&1 | tail -1)"
  done
  bw/cpubw | sed 's/^/cpu /'
  for m in granite-4.0-1b-4bit Qwen2.5-1.5B-Instruct-4bit Llama-3.2-1B-Instruct-4bit Mistral-7B-Instruct-v0.3-4bit; do
    d=$(timeout 1200 ~/gpu/mlx-venv/bin/python llm_survey.py 32 mlx-community/$m 2>/dev/null | grep "^{" | tail -1 | python3 -c "import json,sys; print(json.load(sys.stdin)['decode_tok_s'])" 2>/dev/null)
    echo "decode $m: $d tok/s"
  done
  p=$(timeout 1200 ~/gpu/mlx-venv/bin/python prefill_bench.py mlx-community/granite-4.0-1b-4bit 512 2>/dev/null | grep -oE "[0-9.]+ tok/s" | tail -1)
  echo "prefill granite 512: $p"
} > $out 2>&1
echo PMP_AB_DONE >> $out
