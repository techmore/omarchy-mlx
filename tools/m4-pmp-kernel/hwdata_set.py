# Release 20: asahi_m4.hw_set0..hw_set3 write one 32-bit word each into the
# G16 hardware-data bundle (BUNDLE_ADDRESS + offset) before it is published.
# Encoding: (offset << 32) | value; 0 leaves the slot unused.
#
# Why: the firmware's perf-state change (0x2e560) reads hardware data +0xf38.
# 0 selects AGFHALWaitForPerfState, which polls forever; nonzero selects an
# event wait that times out ("VCM perf state change timeout") and then writes
# the new state to PMP's device table. Asahi's HwDataB sets the matching
# fields (unk_4f0/unk_4f4 -> +0xf34/+0xf38) to 1 on every chip; this driver
# leaves them 0.

def sub(path, old, new):
    s = open(path).read()
    assert s.count(old) == 1, (path, old[:60], s.count(old))
    open(path, "w").write(s.replace(old, new))

sub("drivers/gpu/drm/asahi/asahi_m4.rs", """        mtr_alarm_flag: u32 {
            default: 0,
            description: "Word at hardware data +0x3b60 beside the alarm mask (T8122 boards use 1; 0 leaves it unset)",
        },
""", """        mtr_alarm_flag: u32 {
            default: 0,
            description: "Word at hardware data +0x3b60 beside the alarm mask (T8122 boards use 1; 0 leaves it unset)",
        },
        hw_set0: u64 {
            default: 0,
            description: "Test override: (offset << 32) | value writes one 32-bit word into the hardware-data bundle; 0 is unused",
        },
        hw_set1: u64 {
            default: 0,
            description: "Second hardware-data word override, as hw_set0",
        },
        hw_set2: u64 {
            default: 0,
            description: "Third hardware-data word override, as hw_set0",
        },
        hw_set3: u64 {
            default: 0,
            description: "Fourth hardware-data word override, as hw_set0",
        },
""")

sub("drivers/gpu/drm/asahi/g16_power.rs", """        put32(&mut out, 0xb754, entropy);
        Ok(out)""", """        put32(&mut out, 0xb754, entropy);
        for set in [
            *crate::module_parameters::hw_set0.value(),
            *crate::module_parameters::hw_set1.value(),
            *crate::module_parameters::hw_set2.value(),
            *crate::module_parameters::hw_set3.value(),
        ] {
            if set == 0 {
                continue;
            }
            let offset = (set >> 32) as usize;
            if offset + 4 > out.len() || offset % 4 != 0 {
                return Err(EINVAL);
            }
            put32(&mut out, offset, set as u32);
            pr_info!("asahi_m4: hardware data +{:#x} = {:#x} (hw_set)\\n", offset, set as u32);
        }
        Ok(out)""")
print("r20 hw_set applied")
