#!/bin/bash
# Capture PMP report windows (status/target/actual and the update window) at
# idle and during a GPU job, with PMP running. Writes ~/gpu/pmp_ab/live/rep-*.txt
cd ~/gpu
I=$HOME/gpu/mesa-install
export VK_ICD_FILENAMES=$I/share/vulkan/icd.d/asahi_icd.aarch64.json LD_LIBRARY_PATH=$I/lib
D=~/gpu/pmp_ab/live
cap() { { sudo -n python3 ~/gpu/kernel-pmp/peek.py 0x3803c0000 2048; sudo -n python3 ~/gpu/kernel-pmp/peek.py 0x3803d0000 2048; sudo -n python3 ~/gpu/kernel-pmp/peek.py 0x300e01000 0x100; } > $D/rep-$1.txt; }
cap idle1; sleep 1; cap idle2
mark=$(sudo -n dmesg | wc -l)
( timeout -s KILL 30 bw/vkbw bw/read_v4_u4.spv 4 1280 -1 1024 40 > $D/rep-gpu.txt 2>&1 ) &
for t in 1 2 3 4; do sleep 0.15; cap job$t; done
wait
echo "gpu: $(tail -1 $D/rep-gpu.txt)"
sudo -n dmesg | tail -n +$mark | grep -E "failed Event|stage=" | head -3
echo REP_DONE
