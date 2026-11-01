p = "drivers/soc/apple/pmp.rs"
s = open(p).read()
old = """        let display_monitor = name_str == c"dcs-monitor"
            || name_str == c"disp-monitor"
            || name_str == c"dispext0-monitor"
            || name_str == c"dispext1-monitor";
"""
new = """        // Every "<block>-monitor" descriptor enables the PMP bandwidth
        // monitor for that block; a zero value leaves the monitor (and the
        // DVFS policy fed by it, e.g. DCS from soc-ni*/agx traffic) off.
        let display_monitor = name_str.to_bytes().ends_with(b"-monitor");
"""
assert s.count(old) == 1
open(p, "w").write(s.replace(old, new)); print("all monitors on")
