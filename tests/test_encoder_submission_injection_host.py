"""Run the real encoder submit method against a mocked Vulkan queue."""
import pathlib
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class EncoderSubmissionInjectionTest(unittest.TestCase):
    def test_submit_success_queue_error_and_publication_exception(self):
        source = (ROOT / 'overlay/mlx/backend/omarchy/encoder.cpp').read_text()
        start = source.index('void CommandEncoder::submit()')
        end = source.index('\nCommandEncoder& get_command_encoder(', start)
        submit = source[start:end]
        harness = r'''#include <vulkan/vulkan.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>
#include <sys/syscall.h>
#include <unistd.h>
#define VKX_CHECK(call) do { if ((call) != VK_SUCCESS) throw std::runtime_error("vk error"); } while (0)
int submit_calls = 0; bool fail_queue = false;
namespace vk {
struct Table {
  void CmdPipelineBarrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags,
    VkDependencyFlags, uint32_t, const VkMemoryBarrier*, uint32_t, const VkBufferMemoryBarrier*,
    uint32_t, const VkImageMemoryBarrier*) {}
  VkResult EndCommandBuffer(VkCommandBuffer) { return VK_SUCCESS; }
  VkResult QueueSubmit(VkQueue, uint32_t, const VkSubmitInfo*, VkFence) {
    ++submit_calls; return fail_queue ? VK_ERROR_DEVICE_LOST : VK_SUCCESS;
  }
};
Table& device_table() { static Table t; return t; }
}
namespace trace { struct Counts { int barriers_emitted = 0, vk_submissions = 0; };
Counts& counters() { static Counts c; return c; } }
namespace prof { struct Profile {
  bool profiling() { return false; }
  void on_barrier(bool) {}
  template<class... A> void on_submit_boundary(A...) {}
  template<class... A> void on_submit_end(A...) {}
}; Profile& get() { static Profile p; return p; } uint64_t host_ns() { return 0; } }
namespace omarchy { struct Allocator {
  void flush_noncoherent(VkDevice) {}
  template<class T> void stamp_batch(T&, uint64_t) {}
}; Allocator& allocator() { static Allocator a; return a; } }
bool gated_barriers() { return false; }
class MockCompletion {
 public:
  bool fail_enqueue = false, fail_publish = false, published = false, attached = false;
  uint64_t value = 0;
  std::vector<std::shared_ptr<void>> keep;
  std::vector<std::function<void()>> handlers;
  uint64_t reserve() { return 1; }
  VkSemaphore semaphore() { return reinterpret_cast<VkSemaphore>(1); }
  void enqueue(uint64_t v, std::vector<std::shared_ptr<void>> k,
      std::vector<std::function<void()>> h, VkEvent = VK_NULL_HANDLE, bool pub = true) {
    if (fail_enqueue) throw std::bad_alloc();
    value = v; keep = std::move(k); handlers = std::move(h); published = pub;
  }
  void publish(uint64_t v) {
    if (fail_publish) throw std::bad_alloc();
    if (v != value) throw std::runtime_error("wrong value");
    published = true;
  }
  VkFence acquire_execution_fence() { return reinterpret_cast<VkFence>(2); }
  void attach_execution_fence(uint64_t, VkFence) { attached = true; }
  void detach_execution_fence(uint64_t, VkFence) { attached = false; }
};
class Device {
 public:
  bool poisoned = false; std::mutex lock; MockCompletion completion;
  void ensure_submit_safe() { if (poisoned) throw std::runtime_error("poisoned"); }
  void mark_submit_uncertain() { poisoned = true; }
  std::mutex& queue_mutex() { return lock; }
  MockCompletion& completions() { return completion; }
  VkQueue queue() { return reinterpret_cast<VkQueue>(1); }
  VkDevice handle() { return reinterpret_cast<VkDevice>(1); }
};
void cpu_pd_hold_note_submit(Device&, uint64_t) {}
struct PendingSemaphore { VkSemaphore semaphore; uint64_t value; std::shared_ptr<void> keepalive; };
struct Slot { VkEvent started = reinterpret_cast<VkEvent>(1); uint64_t in_flight = 0; };
class CommandEncoder {
 public:
  Device& device_; bool recording_ = true; VkCommandBuffer cmd_ = reinterpret_cast<VkCommandBuffer>(1);
  uint64_t last_completion_ = 0; int current_slot_ = 0, node_count_ = 0, batch_work_ = 0;
  Slot slots_[1]; std::vector<int> pending_, batch_buffers_;
  std::vector<PendingSemaphore> wait_semaphores_, signal_semaphores_;
  std::vector<std::shared_ptr<void>> retired_pools_;
  std::vector<std::function<void()>> completed_handlers_;
  explicit CommandEncoder(Device& d) : device_(d) {}
  void emit_pending() {}
  void reset_dependency_tracking() {}
  void submit();
};
''' + submit + r'''
int run_case(int mode) {
  Device device; CommandEncoder encoder(device); submit_calls = 0;
  fail_queue = mode == 1; device.completion.fail_publish = mode == 2;
  auto owner = std::make_shared<int>(7); std::weak_ptr<int> weak = owner;
  encoder.retired_pools_.push_back(owner); owner.reset();
  int handler_calls = 0; encoder.completed_handlers_.push_back([&] { ++handler_calls; });
  bool threw = false;
  try { encoder.submit(); } catch (const std::exception&) { threw = true; }
  if (submit_calls != 1 || weak.expired() || !device.completion.attached ||
      encoder.slots_[0].in_flight != 1 || handler_calls != 0) return 1;
  if (mode == 0) {
    if (threw || device.poisoned || !device.completion.published) return 2;
  } else {
    if (!threw || !device.poisoned || device.completion.published) return 3;
    try { encoder.submit(); return 4; } catch (const std::runtime_error&) {}
    if (submit_calls != 1) return 5;
  }
  return 0;
}
int main() { for (int mode = 0; mode < 3; ++mode) if (run_case(mode)) return mode + 1; return 0; }
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
