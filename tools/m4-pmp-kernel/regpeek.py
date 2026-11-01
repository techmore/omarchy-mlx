#!/usr/bin/env python3
"""Read-only peek at T8132 power registers through /dev/mem (run as root).
Never writes. Prints each 32-bit word."""
import mmap, os, struct

REGS = [
    ("gfx-tvm gate (bit0 = GFX on)", 0x380760800),
    ("pmgr ps_gfx  (0x380700250)", 0x380700250),
    ("pmgr ps_pmp  (0x380700488)", 0x380700488),
    ("pmgr ps_pms_sram (0x380700490)", 0x380700490),
]
for agent in range(4):
    REGS.append((f"pmc agent {agent} floor (rail0 bits3:0 = DCS)", 0x382800000 + agent * 0x4000))
    REGS.append((f"pmc agent {agent} pending", 0x382800000 + agent * 0x4000 + 4))
for rail in range(2):
    REGS.append((f"pmc rail {rail} enable (bit n = agent n)", 0x3828ac000 + 4 * rail))
REGS.append(("pmc rails +0x20", 0x3828ac020))

fd = os.open("/dev/mem", os.O_RDONLY | os.O_SYNC)
for name, addr in REGS:
    page = addr & ~0x3fff
    m = mmap.mmap(fd, 0x4000, mmap.MAP_SHARED, mmap.PROT_READ, offset=page)
    val = struct.unpack_from("<I", m, addr - page)[0]
    m.close()
    print(f"{addr:#012x} {val:#010x}  {name}")
os.close(fd)
