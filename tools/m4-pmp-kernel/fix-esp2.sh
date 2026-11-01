#!/bin/sh
# Copy loader files out for analysis, install the verbose limine.conf, unmount.
set -e
diskutil unmount "/Volumes/EFI - OMARC" 2>/dev/null || true
diskutil mount disk0s4
cd "/Volumes/EFI - OMARC"
mkdir -p /tmp/espcopy
cp m1n1/boot.bin m1n1/boot.bin.old limine.conf /tmp/espcopy/
chmod -R a+rX /tmp/espcopy
cp /tmp/limine.conf.verbose limine.conf
sync
grep -n "^timeout:\|^quiet:\|loglevel" limine.conf | cut -c1-80
cd /
diskutil unmount "/Volumes/EFI - OMARC"
echo ESP_FIX2_OK
