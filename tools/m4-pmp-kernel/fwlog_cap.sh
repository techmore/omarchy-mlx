#!/bin/bash
# Dump the G16 firmware region before and after a GPU job fails under PMP
# and print any firmware log text that appeared. Read-only dumps.
cd ~/gpu
I=$HOME/gpu/mesa-install
export VK_ICD_FILENAMES=$I/share/vulkan/icd.d/asahi_icd.aarch64.json LD_LIBRARY_PATH=$I/lib
D=~/gpu/pmp_ab/fwlog; mkdir -p $D
dumpfw() { sudo -n insmod ~/gpu/fwdump/g16fwdump.ko addr=0x10001150000 size=0x13c000 && sudo -n cat /sys/kernel/debug/g16fw_dump > $1; sudo -n rmmod g16fwdump; }
dumpfw $D/before.bin
echo 1 | sudo -n tee /sys/module/pmp_report/parameters/pmp_enable > /dev/null
echo 380500000.pmp | sudo -n tee /sys/bus/platform/drivers/apple_pmp/bind > /dev/null
sleep 5
dumpfw $D/pmp.bin
timeout -s KILL 30 bw/vkbw bw/read_v4_u4.spv 4 1280 -1 1024 5 2>&1 | tail -1
sleep 1
dumpfw $D/after.bin
sudo -n dmesg | grep -E "failed Event|stage=" | tail -2
python3 - <<'PY'
import re
D="/home/sdolbec/gpu/pmp_ab/fwlog/"
a=open(D+"before.bin","rb").read(); p=open(D+"pmp.bin","rb").read(); b=open(D+"after.bin","rb").read()
def texts(x): return {(m.start(),m.group()) for m in re.finditer(rb"[ -~]{12,}", x)}
for name,old,new in (("PMP start",a,p),("GPU job",p,b)):
    print("== new text after", name)
    for off,t in sorted(texts(new)-texts(old))[:60]: print(hex(off), t[:200].decode())
    diff=sum(1 for i in range(0,len(new),4096) if new[i:i+4096]!=old[i:i+4096]); print("changed 4K pages:", diff)
PY
echo FWLOG_DONE
