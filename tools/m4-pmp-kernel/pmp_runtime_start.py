# Release 19: start PMP after boot instead of at boot, and drop the GFX TVM gate.
#
# - pmp-report.c gains pmp_report.pmp_enable (0644). Writing 1 lets a later
#   bind of the apple_pmp driver start PMP and re-runs the post-ready
#   finalize (bit 12/18 handshake, AGX/DISP votes).
# - pmp.rs probe also starts PMP when that switch is set.
# - The gfx_tvm node and the &ps_gfx parent link are removed: with them the
#   Mini hung early in boot after a power-cycle.

def sub(path, old, new, count=1):
    s = open(path).read()
    assert s.count(old) == count, (path, old[:60], s.count(old))
    open(path, "w").write(s.replace(old, new))

rep = "drivers/pmdomain/apple/pmp-report.c"
sub(rep, """#define PMP_REPORT_READY 0x1
#define PMP_READY_WAIT_MS 120000
""", """#define PMP_REPORT_READY 0x1
#define PMP_READY_WAIT_MS 120000

struct apple_pmp_report;
static struct apple_pmp_report *apple_pmp_report_inst;
static void apple_pmp_report_restart(struct apple_pmp_report *rep);

/* Runtime PMP start switch, read by pmp.rs probe.  Writing 1 re-runs the
 * post-ready finalize; then bind apple_pmp to start PMP. */
static int apple_pmp_runtime_enable;

int apple_pmp_runtime_enabled(void)
{
	return READ_ONCE(apple_pmp_runtime_enable);
}

static int apple_pmp_enable_set(const char *val, const struct kernel_param *kp)
{
	int ret = param_set_int(val, kp);

	if (!ret && READ_ONCE(apple_pmp_runtime_enable) && apple_pmp_report_inst)
		apple_pmp_report_restart(apple_pmp_report_inst);
	return ret;
}

static const struct kernel_param_ops apple_pmp_enable_ops = {
	.set = apple_pmp_enable_set,
	.get = param_get_int,
};
module_param_cb(pmp_enable, &apple_pmp_enable_ops, &apple_pmp_runtime_enable, 0644);
MODULE_PARM_DESC(pmp_enable, "1 lets apple_pmp start PMP when bound after boot, and re-runs the post-ready finalize");
""")

sub(rep, """static int apple_pmp_report_probe(struct platform_device *pdev)
{""", """static void apple_pmp_report_restart(struct apple_pmp_report *rep)
{
	if (!rep->offsets->finalize_after_ready)
		return;
	cancel_delayed_work_sync(&rep->finalize_work);
	rep->probed = jiffies;
	dev_info(rep->dev, "pmp_enable=1: waiting for PMP, then finalizing\\n");
	schedule_delayed_work(&rep->finalize_work, 0);
}

static int apple_pmp_report_probe(struct platform_device *pdev)
{""")

sub(rep, """	if (rep->offsets->finalize_after_ready) {
		INIT_DELAYED_WORK(&rep->finalize_work, apple_pmp_report_finalize_work);
		schedule_delayed_work(&rep->finalize_work, 0);
	}

	return 0;
}""", """	if (rep->offsets->finalize_after_ready) {
		INIT_DELAYED_WORK(&rep->finalize_work, apple_pmp_report_finalize_work);
		schedule_delayed_work(&rep->finalize_work, 0);
	}
	apple_pmp_report_inst = rep;

	return 0;
}""")

sub(rep, """	if (rep->offsets->finalize_after_ready)
		cancel_delayed_work_sync(&rep->finalize_work);
	of_platform_depopulate(&pdev->dev);""", """	if (apple_pmp_report_inst == rep)
		apple_pmp_report_inst = NULL;
	if (rep->offsets->finalize_after_ready)
		cancel_delayed_work_sync(&rep->finalize_work);
	of_platform_depopulate(&pdev->dev);""")

rs = "drivers/soc/apple/pmp.rs"
sub(rs, """        // apple_pmp.enable=0 leaves PMP stopped, as on kernels without PMP.
        if *crate::module_parameters::enable.value() == 0 {""", """        // apple_pmp.enable=0 leaves PMP stopped, as on kernels without PMP,
        // unless pmp_report.pmp_enable=1 was set after boot.
        // SAFETY: apple_pmp_runtime_enabled() only reads an int.
        let runtime = unsafe { apple_pmp_runtime_enabled() } != 0;
        if runtime {
            dev_info!(pdev.as_ref(), "PMP started after boot (pmp_report.pmp_enable=1)\\n");
        }
        if *crate::module_parameters::enable.value() == 0 && !runtime {""")

sub(rs, """static PMP_DATA: AtomicPtr<PmpData>""", """unsafe extern "C" {
    fn apple_pmp_runtime_enabled() -> core::ffi::c_int;
}

static PMP_DATA: AtomicPtr<PmpData>""")

dt = "arch/arm64/boot/dts/apple/t8132-j773g-pmp.dtsi"
sub(dt, """	/* GFX TVM gate (aurora-silicon m4/pr2 t8132-j713-gpu.dtsi): the outer
	 * gate of ps_gfx; it reads 0 with PMP running. */
	gfx_tvm: power-controller@380760800 {
		compatible = "apple,t8132-gfx-tvm";
		reg = <0x3 0x80760800 0x0 0x4>;
		#power-domain-cells = <0>;
	};

""", "")
sub(dt, """
/* The outer TVM gate is the parent of the GFX PMGR domain. */
&ps_gfx {
	power-domains = <&gfx_tvm>;
};
""", "")
print("r19 patch applied")
