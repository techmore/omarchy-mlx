#!/bin/sh
# Restore release 18's m1n1 boot.bin (no gfx-tvm node); keep release 17's as boot.bin.r17-tvm.
set -e
diskutil unmount "/Volumes/EFI - OMARC" 2>/dev/null || true
diskutil mount disk0s4
cd "/Volumes/EFI - OMARC/m1n1"
cp boot.bin boot.bin.r17-tvm
cp boot.bin.old boot.bin
sync
shasum -a 256 boot.bin boot.bin.old boot.bin.r17-tvm
cd /
diskutil unmount "/Volumes/EFI - OMARC"
echo ESP_FIX3_OK
