#!/usr/bin/env python3
"""asahi_m4 (G16): optional MTR sensor masks in the hardware data.

The G16 firmware matches MTR (GPU temperature sensor) alarms against the mask
at hardware-data +0x3b58 and handles them through the fast-die masks at
+0x2268 / +0x2f08; at start-up it arms alarms only for sensors in those
masks. The driver left them 0, so an alarm armed by PMP is fatal. These
parameters (default 0: hardware data unchanged) set them for testing.
Run from the kernel source root."""
import sys

def edit(path, old, new):
    s = open(path).read()
    if s.count(old) != 1:
        sys.exit(f"{path}: expected one match:\n{old}")
    open(path, "w").write(s.replace(old, new))

edit("drivers/gpu/drm/asahi/asahi_m4.rs", """        probe_only: u32 {
            default: 0,
            description: "Identify the GPU without starting firmware or registering DRM",
        },
""", """        probe_only: u32 {
            default: 0,
            description: "Identify the GPU without starting firmware or registering DRM",
        },
        mtr_alarm: u64 {
            default: 0,
            description: "MTR sensors whose alarms the firmware matches (hardware data +0x3b58; 0 leaves it unset)",
        },
        mtr_fast_die: u64 {
            default: 0,
            description: "MTR fast-die sensors that handle alarms (hardware data +0x2268 and +0x2f08; 0 leaves them unset)",
        },
        mtr_alarm_flag: u32 {
            default: 0,
            description: "Word at hardware data +0x3b60 beside the alarm mask (T8122 boards use 1; 0 leaves it unset)",
        },
""")
edit("drivers/gpu/drm/asahi/g16_platform.rs", """        for word in 0..12 {
            put32(&mut out, 0x25b4 + word * 4, u32::MAX);
        }
        Ok(out)
    }
""", """        for word in 0..12 {
            put32(&mut out, 0x25b4 + word * 4, u32::MAX);
        }
        // MTR (GPU temperature sensor) masks: the firmware arms, matches and
        // handles sensor alarms only for these sensors. Off unless set.
        let alarm = *crate::module_parameters::mtr_alarm.value();
        let fast_die = *crate::module_parameters::mtr_fast_die.value();
        let flag = *crate::module_parameters::mtr_alarm_flag.value();
        if fast_die != 0 {
            put64(&mut out, 0x2268, fast_die);
            put64(&mut out, 0x2f08, fast_die);
        }
        if alarm != 0 {
            put64(&mut out, 0x3b58, alarm);
        }
        if flag != 0 {
            put32(&mut out, 0x3b60, flag);
        }
        Ok(out)
    }
""")
print("mtr params added")

edit("drivers/gpu/drm/asahi/asahi_m4.rs", """            hwdata.len(), region_c.len(), bundle.len());
""", """            hwdata.len(), region_c.len(), bundle.len());
        dev_info!(dev, "MTR masks: alarm={:#x} fast-die={:#x} flag={}\\n",
            *module_parameters::mtr_alarm.value(),
            *module_parameters::mtr_fast_die.value(),
            *module_parameters::mtr_alarm_flag.value());
""")
print("mtr log added")
