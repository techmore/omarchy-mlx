#!/bin/sh
# usage: sudo mtr-test.sh <pkgrel> <mtr_alarm> [mtr_fast_die]
# Installs ~/gpu/kernel-pmp release <pkgrel> with the given asahi_m4 MTR
# masks (decimal) and reboots only if the install and m1n1 update succeeded.
set -u
rel=$1; alarm=$2; fast=${3:-0}
pkg=/home/sdolbec/gpu/kernel-pmp/linux-gravity-m4-7.1.13.gravity1-$rel-aarch64.pkg.tar.xz
log=/var/tmp/mtr-test-$rel-$alarm-$fast.log
echo "options asahi_m4 mtr_alarm=$alarm mtr_fast_die=$fast" > /etc/modprobe.d/asahi-m4-mtr-test.conf
pacman -U --noconfirm "$pkg" > "$log" 2>&1
tail -3 "$log"
if test -f /usr/lib/modules/7.1.13-1-$rel-ARCH/vmlinuz && grep -q "m1n1 updated" "$log"; then
  echo "INSTALL_OK, rebooting"; sleep 3; systemctl reboot
else
  echo "INSTALL FAILED, not rebooting; see $log"
fi
