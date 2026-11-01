#!/bin/bash
# Release 19: start PMP after boot and record what the GPU perf-state block
# does. Writes ~/gpu/pmp_ab/live/<label>-*.txt.
# usage: pmp_live.sh <label>
set -u
cd ~/gpu
I=$HOME/gpu/mesa-install
export VK_ICD_FILENAMES=$I/share/vulkan/icd.d/asahi_icd.aarch64.json LD_LIBRARY_PATH=$I/lib
D=~/gpu/pmp_ab/live; mkdir -p $D; L=$D/$1
peek() { sudo -n python3 ~/gpu/kernel-pmp/peek.py 0x300e00000 2048 > "$1"; }
gpu() { timeout -s KILL 30 bw/vkbw bw/read_v4_u4.spv 4 1280 -1 1024 "$1" 2>&1 | tail -1; }
st() { sudo -n python3 ~/gpu/kernel-pmp/peek.py 0x300e01000 1 | awk '{print $2}'; }
mark=$(sudo -n dmesg | wc -l)

{
echo "kernel $(uname -r)"
echo "== PMP off"
echo "perf-state idle: $(st)"
peek $L-off-idle.txt
echo "gpu: $(gpu 5)"
bw/cpubw | tail -2 | sed 's/^/cpu /'

echo "== starting PMP"
echo 1 | sudo -n tee /sys/module/pmp_report/parameters/pmp_enable > /dev/null
echo 380500000.pmp | sudo -n tee /sys/bus/platform/drivers/apple_pmp/bind > /dev/null
echo "bind exit $?"
for t in $(seq 1 30); do
  sudo -n dmesg | tail -n +$mark | grep -q "DISP vote" && break; sleep 1
done
sudo -n dmesg | tail -n +$mark | grep -iE "pmp|pmc|clpc|vote|post-ready" | grep -v "descriptor" | tail -25
echo "pmc bound: $(ls /sys/bus/platform/drivers/ 2>/dev/null | grep -i pmc | while read d; do ls /sys/bus/platform/drivers/$d | grep -c @; done | tr '\n' ' ')"
sleep 2
echo "perf-state idle with PMP: $(st)"
peek $L-pmp-idle.txt
bw/cpubw | tail -2 | sed 's/^/cpu /'

echo "== GPU job with PMP (30 s kill timeout)"
mark2=$(sudo -n dmesg | wc -l)
( gpu 40 > $L-pmp-gpu.txt ) &
for t in 1 2 3 4 5 6; do
  sleep 0.2; echo "t$t perf-state: $(st)"; peek $L-pmp-job$t.txt
done
wait
echo "gpu: $(cat $L-pmp-gpu.txt)"
sudo -n dmesg | tail -n +$mark2 | grep -iE "asahi|agx|gpu|timeout|event" | tail -15
} 2>&1 | tee $L-summary.txt
echo PMP_LIVE_DONE
