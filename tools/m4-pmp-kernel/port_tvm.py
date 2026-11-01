import shutil, sys, os
shutil.copy(os.environ["TVM"] + "/gfx-tvm.c", "drivers/pmdomain/apple/gfx-tvm.c")
def edit(path, old, new):
    s = open(path).read()
    if new in s: return
    if s.count(old) != 1: sys.exit(f"{path}: no unique {old!r}")
    open(path, "w").write(s.replace(old, new))
edit("drivers/pmdomain/apple/Kconfig", "config APPLE_PMC\n", """config APPLE_GFX_TVM
\tbool "Apple T8132 GFX TVM gate"
\tdepends on OF && PM
\tselect PM_GENERIC_DOMAINS
\tdefault ARCH_APPLE
\thelp
\t  The outer gate of the GPU power domain on T8132. Switched on at
\t  boot and kept on.

config APPLE_PMC
""")
edit("drivers/pmdomain/apple/Makefile", "obj-$(CONFIG_APPLE_PMC)\t\t+= pmc.o\n",
     "obj-$(CONFIG_APPLE_PMC)\t\t+= pmc.o\nobj-$(CONFIG_APPLE_GFX_TVM)\t+= gfx-tvm.o\n")
d = "arch/arm64/boot/dts/apple/t8132-j773g-pmp.dtsi"
edit(d, "\tpmc: pmc@382800000 {", """\t/* GFX TVM gate (aurora-silicon m4/pr2 t8132-j713-gpu.dtsi): the outer
\t * gate of ps_gfx; it reads 0 with PMP running. */
\tgfx_tvm: power-controller@380760800 {
\t\tcompatible = "apple,t8132-gfx-tvm";
\t\treg = <0x3 0x80760800 0x0 0x4>;
\t\t#power-domain-cells = <0>;
\t};

\tpmc: pmc@382800000 {""")
s = open(d).read()
if "&ps_gfx {" not in s:
    open(d, "a").write("""
/* The outer TVM gate is the parent of the GFX PMGR domain. */
&ps_gfx {
\tpower-domains = <&gfx_tvm>;
};
""")
print("gfx-tvm ported")
