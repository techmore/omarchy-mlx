"""Exercise the real Omarchy fence with host-only mock Event and Stream types."""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FENCE_HEADER = r'''
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
namespace mlx::core {
struct Stream { int index; };
struct array {};
class Fence {
 public:
  explicit Fence(Stream);
  uint32_t update(Stream, const array&, bool);
  void wait(Stream, const array&, uint32_t);
  template <typename T> auto& cast() const { return *static_cast<T*>(fence_.get()); }
 private:
  std::shared_ptr<void> fence_;
};
}
'''
EVENT_HEADER = r'''
#pragma once
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>
#include "mlx/fence.h"
namespace mlx::core {
struct Observations { std::mutex mutex; std::vector<uint64_t> waits; };
inline std::shared_ptr<Observations> observations = std::make_shared<Observations>();
class Event {
 public:
  Event() = default;
  explicit Event(Stream) : state_(observations) {}
  void set_value(uint64_t value) { value_ = value; }
  void signal(Stream) {}
  void wait(Stream) {
    if (!state_) std::abort();
    std::lock_guard<std::mutex> guard(state_->mutex);
    state_->waits.push_back(value_);
  }
 private:
  std::shared_ptr<Observations> state_;
  uint64_t value_ = 0;
};
}
'''
HARNESS = r'''
#include <cassert>
#include <thread>
#include "FENCE_SOURCE"
int main() {
  using namespace mlx::core;
  Stream producer{1}, consumer{2};
  array a;
  Fence fence(producer);
  fence.wait(consumer, a, 0);
  assert(observations->waits.empty());
  assert(fence.update(producer, a, false) == 1);
  assert(fence.update(producer, a, false) == 2);
  fence.wait(consumer, a, 1);
  fence.wait(consumer, a, 2);
  assert((observations->waits == std::vector<uint64_t>{1, 2}));
  // wait() is allowed from any thread while a producer updates the fence.
  std::thread updater([&] { for (int i = 0; i < 1000; ++i) fence.update(producer, a, false); });
  std::thread waiter([&] { for (int i = 0; i < 1000; ++i) fence.wait(consumer, a, 1); });
  updater.join();
  waiter.join();
  for (size_t i = 2; i < observations->waits.size(); ++i) assert(observations->waits[i] == 1);
}
'''


class OmarchyFenceHostTest(unittest.TestCase):
    def test_requested_generation_and_concurrent_wait(self):
        compiler = shutil.which("c++")
        if not compiler:
            self.skipTest("C++ compiler required")
        with tempfile.TemporaryDirectory(prefix="mlx-fence-test-") as tmp:
            root = Path(tmp)
            include = root / "mlx"
            include.mkdir()
            (include / "fence.h").write_text(FENCE_HEADER)
            (include / "event.h").write_text(EVENT_HEADER)
            source = root / "test.cpp"
            source.write_text(HARNESS.replace(
                '"FENCE_SOURCE"',
                '"' + str(ROOT / "overlay/mlx/backend/omarchy/fence.cpp") + '"',
            ))
            exe = root / "test"
            subprocess.run([compiler, "-std=c++17", "-pthread", "-Wall", "-Wextra",
                            "-Werror", "-I", str(root), str(source), "-o", str(exe)],
                           check=True)
            subprocess.run([str(exe)], check=True)


if __name__ == "__main__":
    unittest.main()
