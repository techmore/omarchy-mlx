"""PMC: map the rails window without an exclusive claim. Its rail-enable
words (rails + 0x2000) are the registers pmp-report maps as "event-config"
(0x3828ac000) for its post-ready staging, so both drivers use them."""
p = "drivers/pmdomain/apple/pmc.c"
s = open(p).read()
old = """	pmc->rails = devm_platform_ioremap_resource_byname(pdev, "rails");
	if (IS_ERR(pmc->rails))
		return PTR_ERR(pmc->rails);
"""
new = """	/*
	 * Shared with pmp-report, which maps the rail-enable words as its
	 * "event-config" window: map without an exclusive region claim.
	 */
	{
		struct resource *res = platform_get_resource_byname(pdev, IORESOURCE_MEM, "rails");

		if (!res)
			return -EINVAL;
		pmc->rails = devm_ioremap(dev, res->start, resource_size(res));
		if (!pmc->rails)
			return -ENOMEM;
	}
"""
assert s.count(old) == 1
open(p, "w").write(s.replace(old, new)); print("pmc rails shared")
