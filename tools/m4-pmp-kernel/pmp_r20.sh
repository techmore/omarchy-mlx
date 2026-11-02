#!/bin/bash
# Release 20 test: with asahi_m4.hw_set* applied at boot, check the GPU with
# PMP off, then start PMP live and run one GPU job, recording the perf state.
# usage: pmp_r20.sh <label>     (writes ~/gpu/pmp_ab/live/<label>-*.txt)
set -u
cd ~/gpu
I=$HOME/gpu/mesa-install
export VK_ICD_FILENAMES=$I/share/vulkan/icd.d/asahi_icd.aarch64.json LD_LIBRARY_PATH=$I/lib
D=~/gpu/pmp_ab/live; mkdir -p $D; L=$D/$1
st() { sudo -n python3 ~/gpu/kernel-pmp/peek.py 0x300e01000 1 | awk '{print $2}'; }
peek() { sudo -n python3 ~/gpu/kernel-pmp/peek.py 0x300e00000 2048 > "$1"; }
gpu() { timeout -s KILL 30 bw/vkbw bw/read_v4_u4.spv 4 1280 -1 1024 "$1" 2>&1 | tail -1; }
{
echo "kernel $(uname -r)"
echo "params: $(for p in hw_set0 hw_set1 hw_set2 hw_set3; do printf '%s=%s ' $p $(cat /sys/module/asahi_m4/parameters/$p 2>/dev/null); done)"
sudo -n dmesg | grep -E "hw_set|failed Event" | head
echo "live +0xf30..+0xf4c: $(cd ~/gpu/fwdump && sudo -n python3 fwpeek.py 0xfffffc20c04d8f30 4 2>&1 | awk '{print $2}' | tr '\n' ' ')"
echo "== PMP off"
echo "perf-state idle: $(st)"
echo "gpu: $(gpu 5)"
echo "gpu: $(gpu 5)"
echo "== starting PMP"
mark=$(sudo -n dmesg | wc -l)
echo 1 | sudo -n tee /sys/module/pmp_report/parameters/pmp_enable > /dev/null
echo 380500000.pmp | sudo -n tee /sys/bus/platform/drivers/apple_pmp/bind > /dev/null
for t in $(seq 1 30); do sudo -n dmesg | tail -n +$mark | grep -q "DISP vote" && break; sleep 1; done
sudo -n dmesg | tail -n +$mark | grep -E "PMP started|vote" | tail -4
sleep 2
echo "perf-state idle with PMP: $(st)"
bw/cpubw | tail -1 | sed 's/^/cpu /'
echo "== GPU job with PMP"
mark2=$(sudo -n dmesg | wc -l)
( gpu 40 > $L-pmp-gpu.txt ) &
for t in 1 2 3 4 5 6; do sleep 0.2; echo "t$t perf-state: $(st)"; peek $L-pmp-job$t.txt; done
wait
echo "gpu: $(cat $L-pmp-gpu.txt)"
echo "gpu again: $(gpu 5)"
echo "perf-state after: $(st)"
sudo -n dmesg | tail -n +$mark2 | grep -iE "asahi|agx|gpu|timeout|event|VCM|PTD" | tail -15
} 2>&1 | tee $L-summary.txt
echo PMP_R20_DONE
