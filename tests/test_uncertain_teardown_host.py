"""Host-only coverage for poisoned-device teardown containment."""
import pathlib
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class UncertainTeardownTest(unittest.TestCase):
    def test_stops_worker_without_draining_and_retains_device(self):
        source = (ROOT / 'overlay/mlx/backend/omarchy/device.cpp').read_text()
        begin = source.index('void CompletionDispatcher::stop_without_draining()')
        end = source.index('\nvoid CompletionDispatcher::shutdown()', begin)
        method = source[begin:end]
        self.assertNotIn('handler()', method)
        self.assertNotIn('pending_.', method)
        runtime = source[source.index('  ~Runtime() {'):source.index('\n};\n\nRuntime& runtime()', source.index('  ~Runtime() {'))]
        self.assertIn('dev->submit_uncertain()', runtime)
        self.assertIn('dev->abandon_uncertain()', runtime)
        self.assertIn('dev.release()', runtime)
        self.assertIn('!retained_instance && instance != VK_NULL_HANDLE', runtime)
        harness = '''#include <condition_variable>
#include <mutex>
#include <thread>
class CompletionDispatcher {
 public:
  std::mutex mutex_;
  std::condition_variable cv_;
  bool stop_ = false;
  std::thread thread_;
  void stop_without_draining();
};
''' + method + '''
int main() {
  CompletionDispatcher d;
  d.thread_ = std::thread([&] {
    std::unique_lock<std::mutex> lock(d.mutex_);
    d.cv_.wait(lock, [&] { return d.stop_; });
  });
  d.stop_without_draining();
  if (!d.stop_ || d.thread_.joinable()) return 1;
  d.stop_without_draining();
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
