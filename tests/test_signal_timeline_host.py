"""Host-only behavioral check for CPU event signaling. Requires Vulkan headers."""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class SignalTimelineHostTest(unittest.TestCase):
    def test_cpu_event_does_not_retire_pending_gpu_work(self):
        compiler = shutil.which("c++")
        if compiler is None or not Path("/usr/include/vulkan/vulkan.h").exists():
            self.skipTest("C++ compiler and Vulkan headers required")
        source = (ROOT / "overlay/mlx/backend/omarchy/device.cpp").read_text()
        start = source.index("uint64_t Device::signal_timeline(")
        end = source.index("\nDevice::RecoveryResult Device::recover_stalled_submissions(", start)
        code = PREFIX + source[start:end] + SUFFIX
        with tempfile.TemporaryDirectory(prefix="mlx-signal-test-") as tmp:
            cpp = Path(tmp) / "test.cpp"
            exe = Path(tmp) / "test"
            cpp.write_text(code)
            subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                            "-Wno-missing-field-initializers", str(cpp), "-o", str(exe)],
                           check=True)
            subprocess.run([str(exe)], check=True)


PREFIX = r'''
#include <vulkan/vulkan.h>
#include <cassert>
#include <cstdint>
#include <mutex>
#include <stdexcept>
VkSemaphore evsem = reinterpret_cast<VkSemaphore>(0x11);
VkSemaphore gpucompletion = reinterpret_cast<VkSemaphore>(0x22);
uint64_t evvalue = 0, gpuvalue = 4;
int reserve_calls = 0, enqueue_calls = 0;
VkResult VKAPI_CALL fake_signal(VkDevice, const VkSemaphoreSignalInfo* info) {
  if (info->semaphore == evsem) evvalue = info->value;
  else if (info->semaphore == gpucompletion) gpuvalue = info->value;
  else return VK_ERROR_UNKNOWN;
  return VK_SUCCESS;
}
namespace vk { struct Table {
  PFN_vkVoidFunction GetDeviceProcAddr(VkDevice, const char*) {
    return reinterpret_cast<PFN_vkVoidFunction>(fake_signal);
  }
}; Table& device_table() { static Table t; return t; } }
#define VKX_CHECK(x) do { if ((x) != VK_SUCCESS) throw std::runtime_error("vk"); } while (false)
struct Dispatcher { uint64_t reserve(){ ++reserve_calls; return 6; }
  VkSemaphore semaphore(){ return gpucompletion; }
  void enqueue(uint64_t, std::initializer_list<int>, std::initializer_list<int>){++enqueue_calls;}
};
struct Device { std::mutex queue_mutex_; Dispatcher d;
  VkDevice handle(){ return reinterpret_cast<VkDevice>(0x33); }
  Dispatcher& completions(){ return d; }
  uint64_t signal_timeline(VkSemaphore,uint64_t);
};
'''
SUFFIX = r'''
int main() {
  Device device;
  // Generation 5 represents outstanding GPU work; the published counter is 4.
  auto generation = device.signal_timeline(evsem, 7);
  assert(evvalue == 7);
  assert(gpuvalue == 4);
  assert(generation == 0);
  assert(reserve_calls == 0 && enqueue_calls == 0);
}
'''

if __name__ == "__main__":
    unittest.main()
