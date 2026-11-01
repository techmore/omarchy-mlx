"""T8132 PMP device-power votes (AGX, DISP), after aurora-silicon
m4/pr2-gpu-dcp ("expose retained PMP display and GPU ownership",
pmp-display.c): command 0x0f for logical device 5 (AGX) and 8 (DISP), state 1,
as the words 0x0020f50100000000 / 0x0020f80100000000; PMP answers with the
untagged 0x0021000000000000. Aurora's G16 boot holds the AGX vote before the
GPU firmware runs. Here pmp-report sends both once PMP is ready (its
post-ready step), which is before the GPU driver loads; apple_pmp.power_votes=0
turns them off."""
import sys

def edit(path, old, new, count=1):
    s = open(path).read()
    if s.count(old) != count:
        sys.exit(f"{path}: expected {count} of {old[:80]!r}, found {s.count(old)}")
    open(path, "w").write(s.replace(old, new))

rs = "drivers/soc/apple/pmp.rs"
edit(rs, "use core::mem;\n", """use core::mem;
use core::sync::atomic::{AtomicPtr, AtomicU32, Ordering};
""")
edit(rs, "unsafe impl Send for PmpData {}\n", """/// The bound PMP, retained until reboot, for the C device-power vote helper.
static PMP_DATA: AtomicPtr<PmpData> = AtomicPtr::new(core::ptr::null_mut());
/// Device-power replies received (untagged; PMP serializes them).
static PMP_POWER_REPLIES: AtomicU32 = AtomicU32::new(0);

/// Send one PMP device-power request word (marker bit 53, command << 44,
/// device << 40, state << 32). Returns 0 or a negative errno.
#[no_mangle]
pub extern "C" fn apple_pmp_send_power_word(word: u64) -> core::ffi::c_int {
    if *crate::module_parameters::power_votes.value() == 0 {
        return ENODEV.to_errno();
    }
    let ptr = PMP_DATA.load(Ordering::Acquire);
    if ptr.is_null() {
        return ENODEV.to_errno();
    }
    // SAFETY: PMP_DATA holds a leaked Arc reference, retained until reboot.
    let data = unsafe { &*ptr };
    let mut rtk_guard = data.rtkit.lock();
    let Some(rtk) = rtk_guard.as_mut().as_pin_mut() else {
        return ENODEV.to_errno();
    };
    match rtk.send_message(PMP_ENDPOINT, word) {
        Ok(()) => {
            dev_info!(data.dev, "PMP device-power request {:#018x} sent\\n", word);
            0
        }
        Err(e) => e.to_errno(),
    }
}

/// Number of device-power replies PMP has sent.
#[no_mangle]
pub extern "C" fn apple_pmp_power_replies() -> u32 {
    PMP_POWER_REPLIES.load(Ordering::Acquire)
}

unsafe impl Send for PmpData {}
""")
edit(rs, """    fn recv_message(&self, msg: u64) -> Result<()> {
        let opc = (msg >> OPC_SHIFT) & 0xFF;
""", """    fn recv_message(&self, msg: u64) -> Result<()> {
        // Device-power replies: marker bit 53 with (command + 1) << 44 for
        // commands 0x0e/0x0f. No handshake opcode has this pattern.
        if matches!(msg >> 44, 0x20f | 0x210) {
            PMP_POWER_REPLIES.fetch_add(1, Ordering::AcqRel);
            dev_info!(self.dev, "PMP device-power reply {:#018x}\\n", msg);
            return Ok(());
        }
        let opc = (msg >> OPC_SHIFT) & 0xFF;
""")
edit(rs, """        data.start()?;
        Ok(PmpDriver(data))""", """        data.start()?;
        PMP_DATA.store(Arc::into_raw(data.clone()) as *mut PmpData, Ordering::Release);
        Ok(PmpDriver(data))""")
edit(rs, """        verbose: u32 {""", """        power_votes: u32 {
            default: 1,
            description: "1 (default) sends the T8132 AGX and DISP device-power votes once PMP is ready; 0 does not",
        },
        verbose: u32 {""")

rc = "drivers/pmdomain/apple/pmp-report.c"
edit(rc, "#include <linux/soc/apple/pmp-report.h>\n", """#include <linux/soc/apple/pmp-report.h>
#include <linux/delay.h>

/* PMP device-power votes, sent through the Rust PMP owner (pmp.rs). */
int apple_pmp_send_power_word(u64 word);
u32 apple_pmp_power_replies(void);
#define PMP_AGX_REQUEST		0x0020f50100000000ULL
#define PMP_DISP_REQUEST	0x0020f80100000000ULL

static void apple_pmp_report_vote(struct device *dev, const char *name, u64 word)
{
	u32 before = apple_pmp_power_replies();
	int ret = apple_pmp_send_power_word(word);
	int ms;

	for (ms = 0; !ret && ms < 500 && apple_pmp_power_replies() == before; ms++)
		msleep(1);
	dev_info(dev, "PMP %s vote: send %d, %s after %d ms\\n", name, ret,
		 apple_pmp_power_replies() != before ? "acknowledged" : "no reply", ms);
}
""")
edit(rc, """	if (rep->event_config)
		writel(readl(rep->event_config + 0x20) & ~BIT(2),
		       rep->event_config + 0x20);
}
""", """	if (rep->event_config)
		writel(readl(rep->event_config + 0x20) & ~BIT(2),
		       rep->event_config + 0x20);

	/* Hold the AGX and DISP device-power votes, as macOS and Aurora's
	 * J713 do before the GPU firmware runs. */
	apple_pmp_report_vote(rep->dev, "AGX", PMP_AGX_REQUEST);
	apple_pmp_report_vote(rep->dev, "DISP", PMP_DISP_REQUEST);
}
""")
print("power votes added")
