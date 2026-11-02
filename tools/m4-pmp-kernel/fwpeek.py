#!/usr/bin/env python3
"""Run as root. Read G16 firmware memory by GPU virtual address, read-only.

Walks the firmware's UAT page tables (iBoot handoff root at PT_BASE; 16K
pages, levels at bits 41:36, 35:25, 24:14) and reads physical pages through
g16fwdump.ko.

usage: fwpeek.py VA [COUNT_QWORDS]          dump qwords at VA
       fwpeek.py --chain VA OFF [OFF ...]   p = [VA]; for each OFF but the
                                            last: p = [p + OFF]; print the
                                            qword and dword at p + last OFF
"""
import struct, subprocess, sys

KO = "/home/sdolbec/gpu/fwdump/g16fwdump.ko"
DBG = "/sys/kernel/debug/g16fw_dump"
PT_BASE = 0x103fff78000
PAGE = 0x4000
cache = {}

def readphys(pa, size):
    subprocess.run(["insmod", KO, f"addr={pa:#x}", f"size={size:#x}"], check=True)
    try:
        return open(DBG, "rb").read()
    finally:
        subprocess.run(["rmmod", "g16fwdump"], check=True)

def page(pa):
    pa &= ~(PAGE - 1)
    if pa not in cache:
        cache[pa] = readphys(pa, PAGE)
    return cache[pa]

def q(pa):
    return struct.unpack_from("<Q", page(pa), pa & (PAGE - 1))[0]

def oa(e):
    return e & 0x0000_ffff_ffff_c000

def translate(va):
    l1 = q(PT_BASE + ((va >> 36) & 0x3f) * 8)
    if l1 & 3 != 3:
        raise ValueError(f"va {va:#x}: no L1 ({l1:#x})")
    l2 = q(oa(l1) + ((va >> 25) & 0x7ff) * 8)
    if l2 & 3 == 1:
        return (oa(l2) & ~((1 << 25) - 1)) | (va & ((1 << 25) - 1))
    if l2 & 3 != 3:
        raise ValueError(f"va {va:#x}: no L2 ({l2:#x})")
    l3 = q(oa(l2) + ((va >> 14) & 0x7ff) * 8)
    if l3 & 3 != 3:
        raise ValueError(f"va {va:#x}: no L3 ({l3:#x})")
    return oa(l3) | (va & (PAGE - 1))

def rq(va):
    return q(translate(va))

def strip(p):
    # Drop pointer-authentication / tag bits: firmware VAs are 0xffff_ffxx_....
    return p | 0xffff_ff00_0000_0000 if (p >> 40) & 0xff in (0xfc, 0xfd, 0xfe, 0xff) else p

if sys.argv[1] == "--chain":
    va = int(sys.argv[2], 0)
    offs = [int(x, 0) for x in sys.argv[3:]]
    p = strip(rq(va)); print(f"[{va:#x}] = {p:#x}")
    for off in offs[:-1]:
        n = strip(rq(p + off)); print(f"[{p:#x}+{off:#x}] = {n:#x}"); p = n
    a = p + offs[-1]; v = rq(a & ~7)
    print(f"{a:#x}: qword {v:#018x} dword {struct.unpack_from('<I', struct.pack('<Q', v), a & 4)[0]:#x}")
else:
    va = int(sys.argv[1], 0); n = int(sys.argv[2], 0) if len(sys.argv) > 2 else 1
    for i in range(n):
        print(f"{va + 8 * i:#018x} {rq(va + 8 * i):#018x}")
