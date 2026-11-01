"""Move the MTR mask parameters from the hardware data (bundle +0) to the
power-parameter object the firmware reads (bundle +0x5380, the main
configuration's +0x469 pointer)."""
import sys
def edit(path, old, new):
    s = open(path).read()
    if s.count(old) != 1:
        sys.exit(f"{path}: expected one match:\n{old}")
    open(path, "w").write(s.replace(old, new))

edit("drivers/gpu/drm/asahi/g16_platform.rs", """        // MTR (GPU temperature sensor) masks: the firmware arms, matches and
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
        Ok(out)""", """        Ok(out)""")
edit("drivers/gpu/drm/asahi/g16_power.rs", """        self.power_data(&mut out)?;
        put32(&mut out, 0xb754, entropy);""", """        self.power_data(&mut out)?;
        // MTR (GPU temperature sensor) masks in the power-parameter object
        // at +0x5380 (the main configuration's +0x469 pointer). The firmware
        // arms alarms for the fast-die sensors (+0x2268, set above as
        // 0x6958, and +0x2f08) and matches every MTR alarm against +0x3b58;
        // an alarm outside +0x3b58 is fatal. Off unless set.
        const PP: usize = 0x5380;
        let alarm = *crate::module_parameters::mtr_alarm.value();
        let fast_die = *crate::module_parameters::mtr_fast_die.value();
        let flag = *crate::module_parameters::mtr_alarm_flag.value();
        if fast_die != 0 {
            put64(&mut out, PP + 0x2268, fast_die);
            put64(&mut out, PP + 0x2f08, fast_die);
        }
        if alarm != 0 {
            put64(&mut out, PP + 0x3b58, alarm);
        }
        if flag != 0 {
            put32(&mut out, PP + 0x3b60, flag);
        }
        put32(&mut out, 0xb754, entropy);""")
print("moved")
