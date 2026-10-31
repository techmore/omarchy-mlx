"""Host-only regression for the real command submission test-hook branch."""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

HARNESS = r'''
#include <cassert>
#include <cstdio>
#include <sys/syscall.h>
#include <unistd.h>
struct SubmitInfo { unsigned signalSemaphoreCount; void* pSignalSemaphores; unsigned commandBufferCount; };
struct TimelineInfo { unsigned signalSemaphoreValueCount; void* pSignalSemaphoreValues; };
struct Device { int queue() { return 7; } } device_;
struct Dispatch {
  int calls = 0;
  SubmitInfo observed{};
  int QueueSubmit(int queue, int count, SubmitInfo* submit, int fence) {
    assert(queue == 7 && count == 1 && fence == 8);
    ++calls;
    observed = *submit;
    return 0;
  }
};
#define VKX_CHECK(call) assert((call) == 0)
struct Outcome { int calls; SubmitInfo submit; TimelineInfo timeline; };
Outcome run(bool simulate_drop, bool simulate_strip) {
  int pointer_target = 0;
  SubmitInfo si{2, &pointer_target, 1};
  TimelineInfo timeline{2, &pointer_target};
  Dispatch dt;
  int submit_fence = 8;
  unsigned long completion_value = 3;
  BRANCH
  return {dt.calls, dt.observed, timeline};
}
int main() {
  auto normal = run(false, false);
  assert(normal.calls == 1 && normal.submit.signalSemaphoreCount == 2);
  assert(normal.submit.pSignalSemaphores != nullptr && normal.submit.commandBufferCount == 1);
  assert(normal.timeline.signalSemaphoreValueCount == 2 && normal.timeline.pSignalSemaphoreValues != nullptr);
  auto drop = run(true, false);
  assert(drop.calls == 0);
  auto strip = run(false, true);
  assert(strip.calls == 1 && strip.submit.commandBufferCount == 1);
  assert(strip.submit.signalSemaphoreCount == 0 && strip.submit.pSignalSemaphores == nullptr);
  assert(strip.timeline.signalSemaphoreValueCount == 0 && strip.timeline.pSignalSemaphoreValues == nullptr);
}
'''


class DropSignalHostTest(unittest.TestCase):
    def test_submit_drop_and_strip_paths(self):
        compiler = shutil.which("c++")
        if not compiler:
            self.skipTest("C++ compiler required")
        source = (ROOT / "overlay/mlx/backend/omarchy/encoder.cpp").read_text()
        start = source.index("      if (simulate_strip) {", source.index("bool simulate_drop ="))
        end = source.index("      queue_t1 =", start)
        code = HARNESS.replace("BRANCH", source[start:end])
        with tempfile.TemporaryDirectory(prefix="mlx-drop-signal-test-") as tmp:
            cpp = Path(tmp) / "test.cpp"
            exe = Path(tmp) / "test"
            cpp.write_text(code)
            subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                            str(cpp), "-o", str(exe)], check=True)
            subprocess.run([str(exe)], check=True)


if __name__ == "__main__":
    unittest.main()
