#!/usr/bin/env python3
"""Backport GravityLinux replacement/115-pmp's T8132 PMP support to the
7.1.13 gravity-m4 pmp.rs (commit 4fc326f6c2), keeping that tree's Rust APIs.

Ported (logic only, from the 7.3 driver):
- "apple,t8132-pmp-v2" with a per-chip config; T8132 indexes SetIOReg by the
  wire key at descriptor offset 0x48 and replies with index 0.
- AdvertiseDone (0x36), acknowledged with 8.
- Display-monitor descriptors start at 1 instead of a zeroed buffer.
- 42-bit DMA mask (PMP DART IOVAs sit at 0x100_00000000 on T8132). RTKit's
  shared buffers then come from the C default dma_alloc_coherent path, so the
  7.3 PmpRtkitBuffer is not needed.
- apple_pmp.enable=0 / apple_pmp.verbose=1 parameters.
- 0xc0000 PMP window (T8132 DT; t600x DT maps 0x100000).

usage: port_pmp_rs.py <pmp.rs>
"""
import sys

p = sys.argv[1]
s = open(p).read()


def edit(old, new):
    global s
    if s.count(old) != 1:
        sys.exit(f"expected one match ({s.count(old)}):\n{old[:200]}")
    s = s.replace(old, new)


edit("const PMP_MMIO_SIZE: usize = 0x80000;",
     "// T8132's device tree maps 0xc0000; t600x maps 0x100000.\n"
     "const PMP_MMIO_SIZE: usize = 0xc0000;")
edit("const OPC_SET_IOREG: u64 = 0x34;\n",
     "const OPC_SET_IOREG: u64 = 0x34;\nconst OPC_ADVERTISE_DONE: u64 = 0x36;\n")

edit("""    ioreg_entries: KVec<u32>,
}
""", """    ioreg_entries: KVec<u32>,
    ioreg_advertisements: usize,
}

struct PmpHwConfig {
    advertise_reply_index: bool,
}

const PMP_HW_T6000: PmpHwConfig = PmpHwConfig {
    advertise_reply_index: true,
};

const PMP_HW_T8132: PmpHwConfig = PmpHwConfig {
    advertise_reply_index: false,
};
""")
edit("""            ioreg_entries: KVec::with_capacity(340, GFP_KERNEL)?,
        })""", """            ioreg_entries: KVec::with_capacity(340, GFP_KERNEL)?,
            ioreg_advertisements: 0,
        })""")

edit("""    dev: ARef<device::Device>,
    pmp_mmio:""", """    dev: ARef<device::Device>,
    advertise_reply_index: bool,
    pmp_mmio:""")
edit("fn new(dev: &platform::Device<Core>) -> Result<Arc<PmpData>> {",
     "fn new(dev: &platform::Device<Core>, cfg: &'static PmpHwConfig) -> Result<Arc<PmpData>> {")
edit("""                    dev: dev.as_ref().into(),
                    pmp_mmio,""", """                    dev: dev.as_ref().into(),
                    advertise_reply_index: cfg.advertise_reply_index,
                    pmp_mmio,""")

edit("        let ptr = unsafe { *(ptr_buf.as_ptr() as *const u64) };",
     "        // SAFETY: The buffer holds at least eight bytes, checked above.\n"
     "        let ptr = unsafe { core::ptr::read_unaligned(ptr_buf.as_ptr().cast::<u64>()) };")

# register_ioreg: the 7.3 body (same Rust APIs in both trees).
edit("""    fn register_ioreg(&self, addr: u64) -> Result<u64> {
        let mut state = self.state.lock();
        let msg_buf = if let Some(s) = state.get_buf(addr) {
            s
        } else {
            dev_err!(self.dev, "Unable to find buffer");
            return Err(EIO);
        };
        if msg_buf.size() < 0x44 {
            dev_err!(self.dev, "Buffer too small");
            return Err(EIO);
        }
        // SAFETY: TODO
        let mut size = u32::from_le_bytes(unsafe {
            <&[u8] as TryInto<[u8; 4]>>::try_into(&msg_buf.as_ref()[0x40..0x44]).unwrap()
        });
        if size == 0 {
            let mut name_vec = KVec::with_capacity(0x31, GFP_KERNEL)?;
            name_vec
                .extend_from_slice(unsafe { &msg_buf.as_ref()[0..0x30] }, GFP_KERNEL)
                .unwrap();
            name_vec.push(0, GFP_KERNEL).unwrap();
            let name_str = CStr::from_bytes_until_nul(&name_vec).unwrap();
            let name_str""", """    fn register_ioreg(&self, addr: u64) -> Result<u64> {
        let mut state = self.state.lock();
        let advertisement_index = state.ioreg_advertisements;
        let msg_buf = if let Some(s) = state.get_buf(addr) {
            s
        } else {
            dev_err!(self.dev, "Unable to find buffer");
            return Err(EIO);
        };
        if msg_buf.size() < 0x4c {
            dev_err!(self.dev, "Buffer too small");
            return Err(EIO);
        }
        let mut name_vec = KVec::with_capacity(0x31, GFP_KERNEL)?;
        name_vec
            .extend_from_slice(unsafe { &msg_buf.as_ref()[0..0x30] }, GFP_KERNEL)
            .unwrap();
        name_vec.push(0, GFP_KERNEL).unwrap();
        let name_str = CStr::from_bytes_until_nul(&name_vec).unwrap();

        let display_monitor = name_str == c"dcs-monitor"
            || name_str == c"disp-monitor"
            || name_str == c"dispext0-monitor"
            || name_str == c"dispext1-monitor";

        // SAFETY: TODO
        let mut size = u32::from_le_bytes(unsafe {
            <&[u8] as TryInto<[u8; 4]>>::try_into(&msg_buf.as_ref()[0x40..0x44]).unwrap()
        });
        let wire_index = u32::from_le_bytes(unsafe {
            <&[u8] as TryInto<[u8; 4]>>::try_into(&msg_buf.as_ref()[0x48..0x4c]).unwrap()
        });
        if verbose() {
            dev_info!(
                self.dev,
                "PMP descriptor {} wire_index={:#x} {:?}: size={:#x}",
                advertisement_index,
                wire_index,
                name_str,
                size
            );
        }
        if display_monitor && size == mem::size_of::<u32>() as u32 {
            /* The monitor descriptors carry their initial value through the
             * shared value buffer; a zero-filled buffer disables that PMP
             * policy even though the descriptor is acknowledged. */
            let val_buf_addr = state.value_buf.ok_or(EIO)?;
            let val_buf = state.get_buf(val_buf_addr).ok_or(EIO)?;
            if val_buf.size() < mem::size_of::<u32>() {
                return Err(EINVAL);
            }
            unsafe {
                val_buf.as_mut()[..mem::size_of::<u32>()]
                    .copy_from_slice(&1u32.to_le_bytes());
            }
            if verbose() {
                dev_info!(self.dev, "initialized PMP monitor {:?} to 1", name_str);
            }
        } else if size == 0 {
            let name_str""")
edit("""            if node.property_present(&name_str) {
                let len = node.property_count_elem::<u8>(&name_str)?;
                let data""", """            if node.property_present(&name_str) {
                let len = node.property_count_elem::<u8>(&name_str)?;
                if len > val_buf.size() {
                    dev_err!(self.dev, "tunable {:?} does not fit the value buffer", name_str);
                    return Err(EINVAL);
                }
                let data""")
edit("""            } else {
                dev_info!(self.dev, "unknown property {:?}", name_str);
            }
        }
        state.ioreg_entries.push(size, GFP_KERNEL)?;
        let index = state.ioreg_entries.len() as u64;
        let msg = (OPC_REGISTER_IOREG | OPC_ACK_MASK) << OPC_SHIFT | (index << 32) | size as u64;
        Ok(msg)
    }""", """            } else if verbose() {
                dev_info!(self.dev, "unknown property {:?}", name_str);
            }
        }
        if self.advertise_reply_index {
            state.ioreg_entries.push(size, GFP_KERNEL)?;
        } else {
            /* T8132 carries the SetIOReg lookup key at descriptor offset
             * 0x48; aggregate descriptors reuse keys, so keep the latest
             * size per key rather than an arrival index. */
            let index = wire_index as usize;
            if state.ioreg_entries.len() <= index {
                state.ioreg_entries.resize(index + 1, 0, GFP_KERNEL)?;
            }
            state.ioreg_entries[index] = size;
        }
        state.ioreg_advertisements += 1;
        /* Older firmware wants the advertised count in bits 47:32; T8132
         * wants zero there (the descriptor carries its own index). */
        let reply_index = if self.advertise_reply_index {
            state.ioreg_entries.len() as u64
        } else {
            0
        };
        let msg =
            (OPC_REGISTER_IOREG | OPC_ACK_MASK) << OPC_SHIFT | (reply_index << 32) | size as u64;
        Ok(msg)
    }
    fn advertise_done(&self, count: u64) -> u64 {
        if verbose() {
            let state = self.state.lock();
            dev_info!(
                self.dev,
                "PMP AdvertiseDone count={:#x} advertisements={} wire_entries={}",
                count,
                state.ioreg_advertisements,
                state.ioreg_entries.len()
            );
        }
        // Current firmware expects a fixed 8, not the count echoed back.
        ((OPC_ADVERTISE_DONE | OPC_ACK_MASK) << OPC_SHIFT) | 8
    }""")
edit("""            OPC_SET_IOREG => self.set_ioreg(msg & SET_IOREG_INDEX_MASK)?,
""", """            OPC_SET_IOREG => self.set_ioreg(msg & SET_IOREG_INDEX_MASK)?,
            OPC_ADVERTISE_DONE => self.advertise_done(msg & MSG_IOVA_MASK),
""")

edit("""kernel::of_device_table!(
    OF_TABLE,
    MODULE_OF_TABLE,
    (),
    [(of::DeviceId::new(c"apple,t6000-pmp-v2"), ())]
);
""", """kernel::of_device_table!(
    OF_TABLE,
    MODULE_OF_TABLE,
    <PmpDriver as platform::Driver>::IdInfo,
    [
        (of::DeviceId::new(c"apple,t8132-pmp-v2"), &PMP_HW_T8132),
        (of::DeviceId::new(c"apple,t6000-pmp-v2"), &PMP_HW_T6000),
    ]
);

/// Whether to log PMP's start-up handshake (apple_pmp.verbose=1).
fn verbose() -> bool {
    *crate::module_parameters::verbose.value() != 0
}
""")
edit("""    type IdInfo = ();
""", """    type IdInfo = &'static PmpHwConfig;
""")
edit("""    fn probe(pdev: &platform::Device<Core>, _info: Option<&()>) -> impl PinInit<Self, Error> {
        let dev: ARef<device::Device> = pdev.as_ref().into();
        let data = PmpData::new(pdev)?;""", """    fn probe(
        pdev: &platform::Device<Core>,
        info: Option<&Self::IdInfo>,
    ) -> impl PinInit<Self, Error> {
        let cfg: &'static PmpHwConfig = info.copied().ok_or(ENODEV)?;
        // apple_pmp.enable=0 leaves PMP stopped, as on kernels without PMP.
        if *crate::module_parameters::enable.value() == 0 {
            dev_info!(pdev.as_ref(), "PMP disabled (apple_pmp.enable=0)\\n");
            return Err(ENODEV);
        }
        // PMP DART IOVAs live in the 0x100_00000000 aperture on T8132, so
        // the default 32-bit coherent DMA mask cannot satisfy RTKit's shared
        // memory requests.
        // SAFETY: No DMA allocation or mapping has happened on this device yet.
        unsafe { pdev.dma_set_mask_and_coherent(DmaMask::new::<42>())? };
        let dev: ARef<device::Device> = pdev.as_ref().into();
        let data = PmpData::new(pdev, cfg)?;""")
edit("""    license: "Dual MIT/GPL",
}""", """    license: "Dual MIT/GPL",
    params: {
        enable: u32 {
            default: 1,
            description: "1 (default) starts PMP; 0 leaves it stopped, as on kernels without PMP",
        },
        verbose: u32 {
            default: 0,
            description: "1 logs every descriptor of PMP's start-up handshake",
        },
    },
}""")
edit("""    dma::{
        Coherent,
        CoherentBox, //
    },""", """    dma::{
        Coherent,
        CoherentBox,
        Device as _,
        DmaMask, //
    },""")
open(p, "w").write(s)
print("pmp.rs: T8132 support ported")
