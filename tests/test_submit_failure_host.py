"""Host-only fail-closed checks for an uncertain Vulkan queue submission."""
import pathlib
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class SubmitFailureTest(unittest.TestCase):
    def test_uncertain_submit_blocks_reuse_and_retains_payload(self):
        header = (ROOT / 'overlay/mlx/backend/omarchy/device.h').read_text()
        start = header.index('  void mark_submit_uncertain() noexcept {')
        end = header.index('\n\n  // A stalled completion', start)
        methods = header[start:end]
        encoder = (ROOT / 'overlay/mlx/backend/omarchy/encoder.cpp').read_text()
        start = encoder.index('void CommandEncoder::submit()')
        submit = encoder[start:]
        catch_start = submit.index('if (submit_attempted) {')
        catch_end = submit.index('device_.completions().detach_execution_fence(', catch_start)
        uncertain = submit[catch_start:catch_end]
        self.assertIn('device_.mark_submit_uncertain();', uncertain)
        self.assertIn('unpublished dispatcher entry owns the payload', uncertain)
        self.assertLess(submit.index('completion_prepared = true;'),
                        submit.index('submit_attempted = true;'))
        self.assertLess(submit.index('slots_[current_slot_].in_flight = completion_value;'),
                        submit.index('submit_attempted = true;'))
        self.assertNotIn('detach_execution_fence', uncertain)
        self.assertLess(submit.index('device_.ensure_submit_safe();',
                                     submit.index('lk(device_.queue_mutex())')),
                        submit.index('submit_attempted = true;'))
        acquire = encoder[encoder.index('void CommandEncoder::ensure_recording()'):]
        self.assertLess(acquire.index('device_.ensure_submit_safe();'),
                        acquire.index('BeginCommandBuffer'))
        harness = '''#include <atomic>
#include <stdexcept>
class Device {
 public:
''' + methods + '''
 private:
  std::atomic<bool> submit_uncertain_{false};
};
int main() {
  Device device;
  device.ensure_submit_safe();
  device.mark_submit_uncertain();
  try { device.ensure_submit_safe(); } catch (const std::runtime_error&) { return 0; }
  return 1;
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
