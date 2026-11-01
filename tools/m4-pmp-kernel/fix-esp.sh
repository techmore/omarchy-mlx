#!/bin/sh
# Remount the Linux boot partition read-write, install the edited limine.conf, keep a backup, unmount.
set -e
diskutil unmount "/Volumes/EFI - OMARC" || true
diskutil mount disk0s4
cd "/Volumes/EFI - OMARC"
cp limine.conf limine.conf.before-pmp-off
cp /tmp/limine.conf.pmpoff limine.conf
sync
grep -n "timeout:\|apple_pmp" limine.conf
cd /
diskutil unmount "/Volumes/EFI - OMARC"
echo ESP_FIX_OK
