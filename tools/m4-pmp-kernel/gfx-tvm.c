// SPDX-License-Identifier: GPL-2.0-only OR MIT
/*
 * T8132 GFX TVM gate, the parent of the PMGR GFX power domain.
 *
 * From aurora-silicon/linux m4/pr2-gpu-dcp (drivers/pmdomain/apple/gfx-tvm.c),
 * changed to switch the gate on at probe and keep it on: on J773g with PMP
 * running the gate reads 0 while ps_gfx is on, and G16 work then times out.
 */
#include <linux/bitops.h>
#include <linux/io.h>
#include <linux/iopoll.h>
#include <linux/of.h>
#include <linux/platform_device.h>
#include <linux/pm_domain.h>
#include <linux/slab.h>

#define GFX_TVM_ENABLE BIT(0)

struct apple_gfx_tvm {
	struct generic_pm_domain genpd;
	void __iomem *reg;
};

static int apple_gfx_tvm_set(struct apple_gfx_tvm *tvm, bool enabled)
{
	u32 value = readl(tvm->reg);

	/* Only the GFX enable bit is owned by this domain. */
	value = (value & ~GFX_TVM_ENABLE) | (enabled ? GFX_TVM_ENABLE : 0);
	writel(value, tvm->reg);
	return readl_poll_timeout_atomic(tvm->reg, value,
					 !!(value & GFX_TVM_ENABLE) == enabled,
					 1, 1000);
}

static int apple_gfx_tvm_on(struct generic_pm_domain *genpd)
{
	return apple_gfx_tvm_set(container_of(genpd, struct apple_gfx_tvm, genpd), true);
}

static int apple_gfx_tvm_probe(struct platform_device *pdev)
{
	struct device *dev = &pdev->dev;
	struct apple_gfx_tvm *tvm;
	u32 before;
	int ret;

	tvm = devm_kzalloc(dev, sizeof(*tvm), GFP_KERNEL);
	if (!tvm)
		return -ENOMEM;
	tvm->reg = devm_platform_ioremap_resource(pdev, 0);
	if (IS_ERR(tvm->reg))
		return PTR_ERR(tvm->reg);

	before = readl(tvm->reg);
	ret = apple_gfx_tvm_set(tvm, true);
	dev_info(dev, "GFX TVM gate %#x -> %#x (%d)\n", before, readl(tvm->reg), ret);
	if (ret)
		return ret;

	tvm->genpd.name = "gfx-tvm";
	tvm->genpd.flags = GENPD_FLAG_IRQ_SAFE | GENPD_FLAG_ALWAYS_ON;
	tvm->genpd.power_on = apple_gfx_tvm_on;
	ret = pm_genpd_init(&tvm->genpd, NULL, false);
	if (ret)
		return ret;
	ret = of_genpd_add_provider_simple(dev->of_node, &tvm->genpd);
	if (ret)
		pm_genpd_remove(&tvm->genpd);
	return ret;
}

static const struct of_device_id apple_gfx_tvm_match[] = {
	{ .compatible = "apple,t8132-gfx-tvm" },
	{ }
};

/* The domain is a permanent SoC resource; consumers can outlive probe. */
static struct platform_driver apple_gfx_tvm_driver = {
	.probe = apple_gfx_tvm_probe,
	.driver = {
		.name = "apple-gfx-tvm",
		.of_match_table = apple_gfx_tvm_match,
		.suppress_bind_attrs = true,
	},
};
builtin_platform_driver(apple_gfx_tvm_driver);
