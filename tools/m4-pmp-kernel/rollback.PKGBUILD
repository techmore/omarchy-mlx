# Rollback copy of the installed linux-gravity-m4 7.1.13.gravity1-7, packaged
# from its installed files (no release-7 package file was left on this mini).
pkgname=linux-gravity-m4
pkgver=7.1.13.gravity1
pkgrel=7
pkgdesc="The AArch64 Apple M4 Mac mini (GravityLinux gravity-m4 kernel) and modules"
arch=('aarch64')
url="https://github.com/GravityLinux/linux"
license=('GPL-2.0-only')
depends=(coreutils kmod initramfs "m1n1>=1.6.1")
optdepends=('linux-firmware: firmware images needed for some devices')
provides=(WIREGUARD-MODULE "linux-asahi=${pkgver}")
conflicts=(linux-asahi linux-aurora)
options=('!strip' '!debug')

package() {
  install -d "$pkgdir/usr/lib/modules"
  cp -a /usr/lib/modules/7.1.13-1-7-ARCH "$pkgdir/usr/lib/modules/"
}
