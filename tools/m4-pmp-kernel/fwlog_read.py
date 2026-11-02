#!/usr/bin/env python3
"""Run as root. Print the G16 firmware's own log, read-only.

The firmware's log function (0x34eb8 in the text at 0xfffffc0000d28000)
writes only when main config +0x250 is nonzero (asahi_m4.hw_set0 =
(0x1dc90 << 32) | 1 sets it). Layout, from that function:
  main+0x1d0 -> ring state, 9 channels x 0x30 (write index at +0x20)
  main+0x1d8 -> headers, 9 x 256 x 0x48
  main+0x200 -> payloads, 9 x 256 x 0xd8: u32 type (3), u32 count,
                u64 timestamp, char text[200] at +0x10
usage: fwlog_read.py [--since TIMESTAMP]
"""
import struct, sys
sys.argv, args = ["x", "0"], sys.argv[1:]
exec(open("/home/sdolbec/gpu/fwdump/fwpeek.py").read().split("if sys.argv[1] ==")[0])

MAIN = 0xffff_fc20_c04f_5a40
since = int(args[1], 0) if args[:1] == ["--since"] else 0

def read(va, n):
    out = b""
    while n:
        pa = translate(va)
        chunk = min(n, PAGE - (va & (PAGE - 1)))
        out += page(pa)[pa & (PAGE - 1):(pa & (PAGE - 1)) + chunk]
        va += chunk; n -= chunk
    return out

enable = struct.unpack("<I", read(MAIN + 0x250, 4))[0]
state = struct.unpack("<Q", read(MAIN + 0x1d0, 8))[0]
payload = struct.unpack("<Q", read(MAIN + 0x200, 8))[0]
print(f"log enable={enable:#x} state={state:#x} payload={payload:#x}")
for ch in range(9):
    s = read(state + ch * 0x30, 0x30)
    print(f"ch{ch}: read={struct.unpack_from('<I', s, 0)[0]} write={struct.unpack_from('<I', s, 0x20)[0]}")
blob = read(payload, 9 * 256 * 0xd8)
msgs = []
for i in range(9 * 256):
    e = blob[i * 0xd8:(i + 1) * 0xd8]
    kind, count, ts = struct.unpack_from("<IIQ", e, 0)
    if kind != 3 or ts <= since:
        continue
    text = e[0x10:0xd8].split(b"\0", 1)[0].decode("latin1").rstrip()
    msgs.append((ts, i // 256, i % 256, text))
for ts, ch, i, text in sorted(msgs):
    print(f"{ts:#x} ch{ch}[{i}] {text}")
print(f"{len(msgs)} messages")
