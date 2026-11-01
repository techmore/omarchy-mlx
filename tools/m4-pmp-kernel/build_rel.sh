#!/bin/sh
# usage: build_rel.sh <old_pkgrel> <new_pkgrel>
set -e
cd ~/gpu/kernel-pmp/src/linux-*/
echo "-1-$2" > localversion.10-pkgrel
export PATH=$HOME/.cargo/bin:$PATH
make -s O=$PWD/build/base kernelrelease > build/base/version
make O=$PWD/build/base -j8 all > ~/gpu/kernel-pmp/build$2.log 2>&1 || { grep -E "^error" -A8 ~/gpu/kernel-pmp/build$2.log | head -30; exit 1; }
cd ~/gpu/kernel-pmp
sed -i "s/^pkgrel=$1\$/pkgrel=$2/" PKGBUILD
makepkg -e -d --skipinteg -f > pkg$2.log 2>&1
ls -la linux-gravity-m4-7.1.13.gravity1-$2-aarch64.pkg.tar.xz
