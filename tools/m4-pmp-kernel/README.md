# J773g PMP test kernel

Tested 2026-11-01: PMP starts, then parks DRAM low (CPU reads 69 -> 21
GB/s) and the G16 GPU firmware crashes (MTR alarm). Rolled back. See
receipts/2026-11-01-m4-pmp-test.md.

Goal: start the M4's PMP power-management co-processor and see whether the
GPU's ~82 GB/s read ceiling (macOS ~100 GB/s) moves.

## Why a kernel rebuild

- The running kernel (linux-gravity-m4 7.1.13.gravity1-7, GravityLinux
  gravity-m4 @ 4fc326f6c2) only knows the T6000 PMP protocol and has no
  T8132 PMP report driver. Its device tree has no PMP nodes.
- m1n1-gravity already supports PMP: it finds the node through the `pmp`
  alias, copies board-id, DRAM ids and the ADT tunables, and sets it okay.
- GravityLinux added T8132 PMP on Linux 7.3 (replacement/115-pmp,
  replacement/000-devicetree). Those changes are ported here onto the exact
  7.1.13 recipe (andreinita21/OmarchyM4Mini packaging/j773g/linux-gravity-m4).

## Files

- `PKGBUILD`: the original recipe, release 8, plus the two items below.
- `pmp-t8132.patch`: pmp-report.c and pmp-report.h as in 115-pmp; pmp.rs
  with the T8132 protocol ported to the 7.1 Rust APIs (`port_pmp_rs.py`).
  Adds `apple_pmp.enable=0` (keep PMP off) and `apple_pmp.verbose=1`.
- `t8132-j773g-pmp.dtsi`: Gravity's PMP nodes and steady votes. pmp lists only
  `apple,t8132-pmp-v2`, and the GPU node is not changed, so the current
  kernel binds none of it.

## Checked

- The patches apply to the recipe's source (sha256 matches the PKGBUILD).
- The new t8132-j773g.dtb equals the installed one except for added PMP
  nodes and alias (no removed or changed lines once phandles are normalised).

## Built

`~/gpu/kernel-pmp/linux-gravity-m4-7.1.13.gravity1-8-aarch64.pkg.tar.xz`
(12 minutes on the Mini). Same 1,976 module files as release 7; the kernel
carries `apple,t8132-pmp-v2`, `apple,t8132-pmp-v2-report` and the
`apple_pmp.enable` / `apple_pmp.verbose` parameters; its t8132-j773g.dtb is
byte-identical to the checked one.

## Baseline (release 7, no PMP; `pmp_ab.sh before-r7`)

| | |
|---|---:|
| GPU read, 128-bit loads (1 GiB) | 81.7 GB/s |
| GPU read, 32-bit loads | 70.4 GB/s |
| CPU read, 10 threads | 69.3 GB/s |
| decode granite-4.0-1b-4bit | 52.4 tok/s |
| decode Qwen2.5-1.5B-Instruct-4bit | 64.0 tok/s |
| decode Llama-3.2-1B-Instruct-4bit | 32.4 tok/s |
| decode Mistral-7B-Instruct-v0.3-4bit | 19.3 tok/s |
| prefill granite, 512 tokens | 384 tok/s |

## Next

Install (`sudo pacman -U` the package: the hooks rebuild the initramfs/UKI
and m1n1 stage 2 with the new DTB), reboot, then `dmesg | grep -i pmp` and
`pmp_ab.sh after-r8`.

## Build notes

- linux-api-headers 7.3 on the Mini use `__u128` in `<asm/sigcontext.h>`;
  the 7.1 tools copy of `<linux/types.h>` lacks it, so `resolve_btfids`
  failed. The PKGBUILD adds the same typedefs to
  `tools/include/linux/types.h` (host tools only).
- Built with makepkg on the Mini in ~/gpu/kernel-pmp, Rust 1.93.1 from a
  per-user rustup (`PATH=$HOME/.cargo/bin:$PATH` in the PKGBUILD).

## Recovery

- The current kernel ignores the PMP nodes, so going back to it is enough.
- Rollback package of the installed release 7, built from its installed
  files (same 2,330 paths as `pacman -Ql`):
  `~/gpu/kernel-pmp/rollback/linux-gravity-m4-7.1.13.gravity1-7-aarch64.pkg.tar.zst`
  (`sudo pacman -U` it, then reboot). Omarchy's limine "Snapshots" entry is a
  second way back.
- `apple_pmp.enable=0` on the kernel command line keeps PMP off with the new
  kernel.
- update-m1n1 keeps the previous stage 2 (the boot-loader image) as `m1n1/boot.bin.old` on the ESP.
  If even m1n1 or U-Boot failed, it can be copied back from macOS or 1TR.

## Why PMP might move bandwidth

DRAM controller (DCS) and fabric speed are power states in PMGR. Aurora
Silicon's apple-pmgr-misc work (aurora-silicon/linux, lab/t4157-dcs-
qualification, dcs/whole-word-write) reads the DCS state register as
desired = bits 3:0, actual = bits 7:4, and adds DCS bandwidth voting for
T600x. The Mini's T8132 device tree has no pmgr-misc node (the driver is
built in but bound to nothing), so Linux never asks for a DCS or fabric
state; they stay where iBoot left them. On T8132 PMP owns this (its
handshake includes a dcs-monitor), so starting PMP is the direct test.

## Related work found (2026-11-01)

aurora-silicon (very active): linux (260 branches, incl. ane/h16-* M4 Neural
Engine bring-up, apif-pmp, DCS voting, G16S GPU statistics and frequency
reporting, fix/apple-m4-wait-state which drops the arm64.nowfxt / idle
boot flags), m1n1 fork "aura" (ANE preload handoff), mesa-eryk-m4 (OpenGL ES
3.2 regression runs on G16, September).
