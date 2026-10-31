"""Fault injection into the real completion ownership/publication methods."""
import pathlib
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class CompletionPublicationTest(unittest.TestCase):
    def test_unpublished_entry_survives_failure_and_cannot_run_handler(self):
        source = (ROOT / 'overlay/mlx/backend/omarchy/device.cpp').read_text()
        def extract(start, end):
            return source[source.index(start):source.index(end, source.index(start))]
        enqueue_publish = extract('void CompletionDispatcher::enqueue(',
                                  '\nvoid CompletionDispatcher::reset_progress_event(')
        drain = extract('void CompletionDispatcher::drain_through(',
                        '\nuint64_t CompletionDispatcher::drained_value()')
        harness = r'''#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>
using VkEvent = int;
namespace omarchy {
struct Allocator { void invalidate_noncoherent(int) {} void release_quarantine(uint64_t) {} };
Allocator& allocator() { static Allocator a; return a; }
}
class CompletionDispatcher {
 public:
  struct Completion { uint64_t value; std::vector<std::shared_ptr<void>> temporaries;
    std::vector<std::function<void()>> handlers; VkEvent started; bool published; };
  int device_ = 0; bool stop_ = false;
  std::deque<Completion> pending_;
  std::map<uint64_t, int> resubmitable_;
  std::vector<std::shared_ptr<void>> retired_temporaries_;
  uint64_t drained_value_ = 0;
  std::chrono::steady_clock::time_point last_drain_at_{};
  std::mutex mutex_, drain_mutex_; std::condition_variable cv_;
  void enqueue(uint64_t, std::vector<std::shared_ptr<void>>,
               std::vector<std::function<void()>>, VkEvent, bool);
  void publish(uint64_t); void drain_through(uint64_t);
};
''' + enqueue_publish + drain + r'''
int main() {
  CompletionDispatcher d;
  int handlers = 0;
  auto owned = std::make_shared<int>(42);
  std::weak_ptr<int> weak = owned;
  d.enqueue(1, {owned}, {[&] { ++handlers; }}, 0, false);
  owned.reset();
  try { throw std::bad_alloc(); } catch (const std::bad_alloc&) {}
  d.drain_through(1);
  if (handlers != 0 || weak.expired() || d.pending_.size() != 1 || d.drained_value_) return 1;
  d.publish(1); d.drain_through(1);
  if (handlers != 1 || d.drained_value_ != 1 || !d.pending_.empty()) return 2;
  return 0;
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
