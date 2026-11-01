#!/usr/bin/env python3
"""Add aurora-silicon m4/pr1-basic-bringup's PMC voter driver and T8132 CLPC
floors to the 7.1.13 gravity-m4 tree, and enable PMC + CLPC on J773g.
Run from the kernel source root with the aurora files in $PMCPORT."""
import os, shutil, sys
src = os.environ["PMCPORT"]
for f in ["drivers/pmdomain/apple/pmc.c", "drivers/soc/apple/clpc.c",
          "include/linux/soc/apple/pmc.h"]:
    shutil.copy(os.path.join(src, f), f)

def edit(path, old, new):
    s = open(path).read()
    if new in s:
        return
    if s.count(old) != 1:
        sys.exit(f"{path}: expected one match for {old!r}")
    open(path, "w").write(s.replace(old, new))

edit("drivers/pmdomain/apple/Kconfig", "\tselect PM_GENERIC_DOMAINS\n\nendif\n", """\tselect PM_GENERIC_DOMAINS

config APPLE_PMC
\tbool "Apple PMC voter interface"
\tdepends on OF
\tdepends on PM
\tselect PM_GENERIC_DOMAINS
\tdefault ARCH_APPLE
\thelp
\t  The PMC in some Apple SoCs arbitrates the performance states of
\t  shared rails, such as the DRAM controller and the SoC fabric,
\t  between voter agents. This driver enables the static agents and
\t  provides power domains that enable a dynamic agent, such as the
\t  GPU, while its device is powered on. It can also limit the DRAM
\t  controller rail as a thermal cooling device.

endif
""")
edit("drivers/pmdomain/apple/Makefile", "obj-$(CONFIG_APPLE_PMP_REPORT)\t+= pmp-report.o\n",
     "obj-$(CONFIG_APPLE_PMP_REPORT)\t+= pmp-report.o\nobj-$(CONFIG_APPLE_PMC)\t\t+= pmc.o\n")
edit("drivers/soc/apple/Kconfig", "\nendmenu\n\nendif\n", """
config APPLE_CLPC
\ttristate "Apple host performance controller floors"
\tdepends on ARCH_APPLE || COMPILE_TEST
\tdepends on APPLE_PMC && CPU_FREQ
\tdefault ARCH_APPLE
\thelp
\t  Raises the DRAM controller and SoC fabric performance floors of the
\t  host's PMC voter agent while CPU clusters are busy, and releases
\t  them at idle.

endmenu

endif
""")
edit("drivers/soc/apple/Makefile", "obj-$(CONFIG_APPLE_PMP) += pmp.o\n",
     "obj-$(CONFIG_APPLE_PMP) += pmp.o\n\nobj-$(CONFIG_APPLE_CLPC) += clpc.o\n")

dtsi = "arch/arm64/boot/dts/apple/t8132-j773g-pmp.dtsi"
edit(dtsi, "\tpmp_mbox: mailbox@380e08000 {", """\t/*
\t * PMC voter interface (aurora-silicon m4/pr1-basic-bringup,
\t * t8132-memory.dtsi and J713's static voter): PMP sets DRAM (DCS) and
\t * fabric speed from the agents' votes; the AP's agent 3 votes DCS 5 /
\t * fabric 3 through CLPC while a CPU cluster or the GPU is busy.
\t */
\tpmc: pmc@382800000 {
\t\tcompatible = "apple,t8132-pmc";
\t\treg = <0x3 0x82800000 0x0 0x10000>,
\t\t      <0x3 0x828aa000 0x0 0x3000>;
\t\treg-names = "voters", "rails";
\t\t#address-cells = <1>;
\t\t#size-cells = <0>;
\t\tapple,pmp = <&pmp>;
\t\tapple,static-voters = <3>;

\t\tclpc: performance-controller@3 {
\t\t\tcompatible = "apple,t8132-clpc";
\t\t\treg = <3>;
\t\t\tapple,gpu-power-state = <&ps_gfx>;
\t\t};
\t};

\tpmp_mbox: mailbox@380e08000 {""")
print("pmc/clpc ported")
