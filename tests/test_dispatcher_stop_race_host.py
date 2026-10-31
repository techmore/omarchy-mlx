"""Exercise the actual dispatcher worker at a blocked timeline query."""
import pathlib
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class DispatcherStopRaceTest(unittest.TestCase):
    def test_stop_during_counter_query_keeps_pending_handler(self):
        source = (ROOT / 'overlay/mlx/backend/omarchy/device.cpp').read_text()
        def extract(start, end):
            return source[source.index(start):source.index(end, source.index(start))]
        stop = extract('void CompletionDispatcher::stop_without_draining()',
                       '\nvoid CompletionDispatcher::shutdown()')
        drain = extract('void CompletionDispatcher::drain_through(',
                        '\nuint64_t CompletionDispatcher::drained_value()')
        run = extract('void CompletionDispatcher::run()',
                      '\n} // namespace mlx::core::omarchy')
        harness = r'''#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
std::mutex gate_mutex;
std::condition_variable gate_cv;
bool query_entered = false, query_released = false;
constexpr uint64_t kCompletionPollNs = 1000000;
constexpr int VK_SUCCESS = 0;
namespace vk {
struct Table { int GetSemaphoreCounterValue(int, int, uint64_t* value) {
  std::unique_lock<std::mutex> lock(gate_mutex);
  query_entered = true; gate_cv.notify_all();
  gate_cv.wait(lock, [] { return query_released; });
  *value = 1; return VK_SUCCESS;
}};
Table& device_table() { static Table t; return t; }
}
namespace omarchy {
struct Allocator {
  void invalidate_noncoherent(int) {}
  void release_quarantine(uint64_t) {}
  bool has_quarantined() { return false; }
};
Allocator& allocator() { static Allocator a; return a; }
}
class CompletionDispatcher {
 public:
  struct Completion { uint64_t value; std::vector<std::shared_ptr<void>> temporaries;
                      std::vector<std::function<void()>> handlers; bool published; };
  int device_ = 0, semaphore_ = 0;
  std::deque<Completion> pending_;
  std::map<uint64_t, int> resubmitable_;
  std::vector<std::shared_ptr<void>> retired_temporaries_;
  uint64_t drained_value_ = 0;
  std::chrono::steady_clock::time_point last_drain_at_{};
  std::mutex mutex_, drain_mutex_;
  std::condition_variable cv_;
  std::thread thread_;
  bool stop_ = false;
  bool tick_execution_fences() { return false; }
  void run(); void drain_through(uint64_t); void stop_without_draining();
};
''' + stop + drain + run + r'''
int main() {
  CompletionDispatcher d;
  std::atomic<int> calls{0};
  d.pending_.push_back({1, {}, {[&] { ++calls; }}, true});
  d.thread_ = std::thread([&] { d.run(); });
  {
    std::unique_lock<std::mutex> lock(gate_mutex);
    gate_cv.wait(lock, [] { return query_entered; });
  }
  std::thread stopper([&] { d.stop_without_draining(); });
  for (;;) {
    { std::lock_guard<std::mutex> lock(d.mutex_); if (d.stop_) break; }
    std::this_thread::yield();
  }
  { std::lock_guard<std::mutex> lock(gate_mutex); query_released = true; }
  gate_cv.notify_all(); stopper.join();
  return calls != 0 || d.pending_.size() != 1 || d.drained_value_ != 0;
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
