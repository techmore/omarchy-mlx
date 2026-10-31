"""Host-only checks for command-buffer reuse after GPU completion."""
import pathlib
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class CommandBufferRetirementTest(unittest.TestCase):
    def test_fence_wait_and_reuse_guards(self):
        device = (ROOT / 'overlay/mlx/backend/omarchy/device.cpp').read_text()
        start = device.index('bool CompletionDispatcher::wait_execution_retired(')
        end = device.index('\nbool CompletionDispatcher::wait_settled(', start)
        method = device[start:end]
        encoder = (ROOT / 'overlay/mlx/backend/omarchy/encoder.cpp').read_text()
        start = encoder.index('void CommandEncoder::ensure_recording()')
        end = encoder.index('\n}', start)
        acquire = encoder[start:end]
        self.assertIn('completions.execution_complete(value)', acquire)
        self.assertIn('completions.wait_execution_retired(', acquire)
        self.assertLess(acquire.index('completions.wait_execution_retired('),
                        acquire.index('slots_[oldest].in_flight = 0'))
        harness = r'''#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>
class CompletionDispatcher {
 public:
  std::atomic<bool> complete{false};
  bool execution_complete(uint64_t) { return complete.load(); }
  bool wait_execution_retired(uint64_t, std::chrono::steady_clock::duration);
};
''' + method + r'''
int main() {
  using namespace std::chrono_literals;
  CompletionDispatcher d;
  if (d.wait_execution_retired(1, 0ms)) return 1;
  std::thread signal([&] { std::this_thread::sleep_for(10ms); d.complete = true; });
  bool retired = d.wait_execution_retired(1, 1s);
  signal.join();
  if (!retired) return 2;
  d.complete = false;
  if (d.wait_execution_retired(1, 15ms)) return 3;
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
