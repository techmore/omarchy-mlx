#!/bin/sh
# Return to the normal kernel (release 7) and remove the MTR test option.
# Leftover test-kernel module trees are removed and m1n1 rebuilt so the boot
# image carries release 7's own device tree. Reboots only if all succeeded.
log=/var/tmp/pmp-rollback-$(date +%H%M).log
rm -f /etc/modprobe.d/asahi-m4-mtr-test.conf
pacman -U --noconfirm /home/sdolbec/gpu/kernel-pmp/rollback/linux-gravity-m4-7.1.13.gravity1-7-aarch64.pkg.tar.zst > "$log" 2>&1
for d in /usr/lib/modules/7.1.13-1-*-ARCH; do
  case "$d" in */7.1.13-1-7-ARCH) ;; *) pacman -Qo "$d" >/dev/null 2>&1 || rm -rf -- "$d" ;; esac
done
update-m1n1 >> "$log" 2>&1
tail -3 "$log"
if test -f /usr/lib/modules/7.1.13-1-7-ARCH/vmlinuz && [ "$(ls -d /usr/lib/modules/*-ARCH)" = "/usr/lib/modules/7.1.13-1-7-ARCH" ] && tail -1 "$log" | grep -q "m1n1 updated"; then
  echo "ROLLBACK_OK, rebooting"; sleep 3; systemctl reboot
else
  echo "ROLLBACK FAILED, not rebooting; see $log"
fi
