"""Exercise the real timeout callback without a Vulkan device."""
import pathlib
import subprocess
import tempfile
import unittest


class StalledReplayTest(unittest.TestCase):
    def test_pending_submissions_are_not_replayed(self):
        source = (pathlib.Path(__file__).resolve().parents[1] /
                  'overlay/mlx/backend/omarchy/device.cpp').read_text()
        start = source.index('Device::RecoveryResult Device::recover_stalled_submissions(')
        end = source.index('\nvoid Device::join_completed_handlers()', start)
        callback = source[start:end]
        harness = '''#include <cstdint>
class Device {
 public:
  enum class RecoveryResult { kRecovered, kNotRecoverable, kExhausted };
  RecoveryResult recover_stalled_submissions(uint64_t, uint32_t);
};
''' + callback + '''
int main() {
  Device d;
  for (uint64_t target : {uint64_t(0), uint64_t(1), UINT64_MAX})
    for (uint32_t round : {uint32_t(0), uint32_t(1), uint32_t(2)})
      if (d.recover_stalled_submissions(target, round) != Device::RecoveryResult::kExhausted)
        return 1;
}
'''
        harness = '#include <initializer_list>\n' + harness
        with tempfile.TemporaryDirectory() as directory:
            cpp = pathlib.Path(directory) / 'check.cpp'
            exe = pathlib.Path(directory) / 'check'
            cpp.write_text(harness)
            subprocess.run(['c++', '-std=c++17', str(cpp), '-o', str(exe)], check=True)
            subprocess.run([str(exe)], check=True)


if __name__ == '__main__':
    unittest.main()
