#!/usr/bin/env python3
"""Read-only: dump 32-bit words from physical MMIO. usage: peek.py ADDR [COUNT] (root)."""
import mmap, os, struct, sys
addr = int(sys.argv[1], 0); count = int(sys.argv[2], 0) if len(sys.argv) > 2 else 1
fd = os.open("/dev/mem", os.O_RDONLY | os.O_SYNC)
page = addr & ~0x3fff
span = ((addr + 4 * count - page + 0x3fff) // 0x4000) * 0x4000
m = mmap.mmap(fd, span, mmap.MAP_SHARED, mmap.PROT_READ, offset=page)
for i in range(count):
    a = addr + 4 * i
    v = struct.unpack_from("<I", m, a - page)[0]
    if v or count <= 16:
        print(f"{a:#012x} {v:#010x}")
m.close(); os.close(fd)
