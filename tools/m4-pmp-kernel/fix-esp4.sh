#!/bin/sh
# Check the Linux boot partition, verify kernel image hashes, boot snapshot 1 by default.
diskutil unmount "/Volumes/EFI - OMARC" 2>/dev/null
echo "== fsck (read-only)"; fsck_msdos -n /dev/rdisk0s4
set -e
diskutil mount disk0s4
cd "/Volumes/EFI - OMARC"
echo "== hashes"; /usr/bin/python3 - <<'PY'
import hashlib,re
c=open("limine.conf").read()
for path,h in set(re.findall(r"path: boot\(\):(/\S+?)#([0-9a-f]+)",c)):
    d=hashlib.blake2b(open("."+path,"rb").read()).hexdigest()
    print("OK " if d==h else "BAD", path)
PY
cp limine.conf limine.conf.verbose-bak
cp /tmp/limine.conf.snap limine.conf
sync
grep -n "default_entry\|^/Recovery" limine.conf
cd /
diskutil unmount "/Volumes/EFI - OMARC"
echo ESP_FIX4_OK
