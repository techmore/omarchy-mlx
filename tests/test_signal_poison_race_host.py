"""Exercise the real host-signal method racing a poisoned queue."""
import pathlib
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class SignalPoisonRaceTest(unittest.TestCase):
    def test_signal_checks_poison_after_acquiring_queue_lock(self):
        source = (ROOT / 'overlay/mlx/backend/omarchy/device.cpp').read_text()
        start = source.index('uint64_t Device::signal_timeline(')
        end = source.index('\nDevice::RecoveryResult', start)
        method = source[start:end]
        header = (ROOT / 'overlay/mlx/backend/omarchy/device.h').read_text()
        start = header.index('  void mark_submit_uncertain() noexcept {')
        end = header.index('\n\n  // Dropped-submission', start)
        guards = header[start:end]
        harness = r'''#include <atomic>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <thread>
using VkSemaphore = int;
struct VkSemaphoreSignalInfo { int type; int semaphore; uint64_t value; };
constexpr int VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO = 1;
constexpr int VK_SUCCESS = 0;
std::atomic<int> signal_calls{0};
int mock_signal(int, const VkSemaphoreSignalInfo*) { ++signal_calls; return VK_SUCCESS; }
using PFN_vkSignalSemaphore = int (*)(int, const VkSemaphoreSignalInfo*);
namespace vk {
struct Table { void* GetDeviceProcAddr(int, const char*) {
  return reinterpret_cast<void*>(&mock_signal);
}};
Table& device_table() { static Table table; return table; }
}
#define VKX_CHECK(call) do { if ((call) != VK_SUCCESS) throw std::runtime_error("error"); } while (0)
class Device {
 public:
  std::mutex queue_mutex_;
  std::atomic<bool> submit_uncertain_{false};
  int handle() { return 1; }
  uint64_t signal_timeline(VkSemaphore, uint64_t);
''' + guards + r'''
};
''' + method + r'''
int main() {
  Device d;
  if (d.signal_timeline(1, 1) != 0 || signal_calls != 1) return 1;
  std::unique_lock<std::mutex> lock(d.queue_mutex_);
  std::atomic<bool> attempted{false}, refused{false};
  std::thread signaler([&] {
    attempted = true;
    try { d.signal_timeline(1, 2); } catch (const std::runtime_error&) { refused = true; }
  });
  while (!attempted) std::this_thread::yield();
  d.mark_submit_uncertain();
  lock.unlock(); signaler.join();
  return !refused || signal_calls != 1;
}
'''
        with tempfile.TemporaryDirectory() as directory:
            cpp = pathlib.Path(directory) / 'check.cpp'
            exe = pathlib.Path(directory) / 'check'
            cpp.write_text(harness)
            subprocess.run(['c++', '-std=c++17', '-pthread', str(cpp), '-o', str(exe)],
                           check=True, timeout=30)
            subprocess.run([str(exe)], check=True, timeout=3)


if __name__ == '__main__':
    unittest.main()
