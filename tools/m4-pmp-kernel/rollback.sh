#!/bin/sh
# Return to the normal kernel (release 7, no PMP support) and remove the MTR
# test option. Reboots only if the install and m1n1 update succeeded.
log=/var/tmp/pmp-rollback-$(date +%H%M).log
rm -f /etc/modprobe.d/asahi-m4-mtr-test.conf
pacman -U --noconfirm /home/sdolbec/gpu/kernel-pmp/rollback/linux-gravity-m4-7.1.13.gravity1-7-aarch64.pkg.tar.zst > "$log" 2>&1
tail -3 "$log"
if test -f /usr/lib/modules/7.1.13-1-7-ARCH/vmlinuz && grep -q "m1n1 updated" "$log"; then
  echo "ROLLBACK_OK, rebooting"; sleep 3; systemctl reboot
else
  echo "ROLLBACK FAILED, not rebooting; see $log"
fi
