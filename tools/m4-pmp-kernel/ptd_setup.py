# Release 21: give the G16G GPU firmware its AON PTD (PMP device table) setup.
#
# The firmware log says "Invalid AON PTD Dev Instance" and "AON PTD Enable
# mask is 0 for mGPUS/Fabric". asahi_m4.ptd_setup (default 0) fills:
#   bit 0: the auxiliary PTD range table at 0xfffffc20c0500000 (power object
#          +0x6318): 12-byte records {u8 id, u8 0, le16 base, le16 count,
#          le16 0, le32 extra}, built by pmp-report from PMP's
#          apple,tunable-ptd-range filtered by apple,tunable-pm-ptd-ranges.
#   bit 1: mGPU PTD device ids "g0TR"/"g1TR" (power object +0x6320/+0x6324)
#          and mask 3 (+0x6360), as GravityLinux's G16S template.
#   bit 2: fabric id "r0TR" (+0x6378) and mask 1 (+0x637c), at the offsets
#          the G16G firmware reads (0x313a4..0x313d0).

def sub(path, old, new):
    s = open(path).read()
    assert s.count(old) == 1, (path, old[:60], s.count(old))
    open(path, "w").write(s.replace(old, new))

rep = "drivers/pmdomain/apple/pmp-report.c"
sub(rep, """static int apple_pmp_report_probe(struct platform_device *pdev)
{""", """/*
 * The G16 GPU firmware's AON PTD range table: one 12-byte record per PMP PTD
 * range the GPU may address (id, base, count, extra), taken from PMP's
 * apple,tunable-ptd-range and filtered by apple,tunable-pm-ptd-ranges.
 * Returns the number of records written, or a negative error.
 */
int apple_pmp_gpu_ptd_ranges(u8 *out, size_t len)
{
	struct device_node *np;
	const __be32 *rng, *pm;
	int rlen, plen, n = 0, i, j;

	np = of_find_compatible_node(NULL, NULL, "apple,t8132-pmp-v2");
	if (!np)
		return -ENODEV;
	rng = of_get_property(np, "apple,tunable-ptd-range", &rlen);
	pm = of_get_property(np, "apple,tunable-pm-ptd-ranges", &plen);
	if (!rng || !pm) {
		of_node_put(np);
		return -ENOENT;
	}
	/* Both properties are little-endian words copied from the ADT. */
	for (i = 0; i < plen / 4; i++) {
		u32 id = get_unaligned_le32(&pm[i]);

		if (!id)
			continue;
		for (j = 0; j + 8 <= rlen / 4; j += 8) {
			u8 *e = out + n * 12;

			if (get_unaligned_le32(&rng[j]) != id)
				continue;
			if ((n + 1) * 12 > len)
				break;
			memset(e, 0, 12);
			e[0] = id;
			put_unaligned_le16(get_unaligned_le32(&rng[j + 1]), e + 2);
			put_unaligned_le16(get_unaligned_le32(&rng[j + 2]), e + 4);
			put_unaligned_le32(get_unaligned_le32(&rng[j + 3]), e + 8);
			n++;
			break;
		}
	}
	of_node_put(np);
	return n;
}
EXPORT_SYMBOL_GPL(apple_pmp_gpu_ptd_ranges);

static int apple_pmp_report_probe(struct platform_device *pdev)
{""")

m4 = "drivers/gpu/drm/asahi/asahi_m4.rs"
sub(m4, """        hw_set0: u64 {""", """        ptd_setup: u32 {
            default: 0,
            description: "Test: bit 0 fills the GPU firmware's AON PTD range table from PMP, bit 1 sets mGPU PTD ids g0TR/g1TR (mask 3), bit 2 sets fabric id r0TR (mask 1)",
        },
        hw_set0: u64 {""")

pw = "drivers/gpu/drm/asahi/g16_power.rs"
sub(pw, """        put32(&mut out, 0xb754, entropy);
""", """        put32(&mut out, 0xb754, entropy);
        let ptd = *crate::module_parameters::ptd_setup.value();
        if ptd & 2 != 0 {
            put32(&mut out, PP + 0x6320, u32::from_be_bytes(*b"g0TR"));
            put32(&mut out, PP + 0x6324, u32::from_be_bytes(*b"g1TR"));
            put32(&mut out, PP + 0x6360, 3);
        }
        if ptd & 4 != 0 {
            put32(&mut out, PP + 0x6378, u32::from_be_bytes(*b"r0TR"));
            put32(&mut out, PP + 0x637c, 1);
        }
""")

g16 = "drivers/gpu/drm/asahi/g16.rs"
sub(g16, """        firmware_space.alloc(
            0xffff_fc20_c050_0000,
            PAGE_SIZE as usize,
            crate::pgtable::prot::PROT_FW_PRIV_RW,
        )?;
""", """        firmware_space.alloc(
            0xffff_fc20_c050_0000,
            PAGE_SIZE as usize,
            crate::pgtable::prot::PROT_FW_PRIV_RW,
        )?;
        if *crate::module_parameters::ptd_setup.value() & 1 != 0 {
            let mut table = [0u8; 19 * 12];
            // SAFETY: `table` is writable for `table.len()` bytes.
            let n = unsafe { apple_pmp_gpu_ptd_ranges(table.as_mut_ptr(), table.len()) };
            if n > 0 {
                firmware_space.write(0xffff_fc20_c050_0000, &table[..n as usize * 12])?;
                pr_info!("asahi_m4: AON PTD range table: {} records from PMP\\n", n);
            } else {
                pr_warn!("asahi_m4: no PMP PTD ranges ({})\\n", n);
            }
        }
""")
sub(g16, """// Keep the opening barrier transport separate from render queues.""", """unsafe extern "C" {
    fn apple_pmp_gpu_ptd_ranges(out: *mut u8, len: usize) -> core::ffi::c_int;
}

// Keep the opening barrier transport separate from render queues.""")
print("r21 ptd_setup applied")
