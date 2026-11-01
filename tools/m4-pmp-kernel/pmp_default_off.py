# Control build: apple_pmp.enable defaults to 0 (PMP stays stopped).
p = "drivers/soc/apple/pmp.rs"
s = open(p).read()
old = """        enable: u32 {
            default: 1,"""
assert s.count(old) == 1
open(p, "w").write(s.replace(old, """        enable: u32 {
            default: 0,"""))
print("pmp default off")
