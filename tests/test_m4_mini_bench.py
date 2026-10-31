"""scripts/m4_mini_bench.py without a GPU: the fixed-work checks and the result comparison."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "m4_mini_bench.py"


def result(system, prompt_tps, gen_tps, prompt_sha="p", ids_sha="i" * 16):
    return {
        "schema": "m4-mini-bench/1",
        "exit_code": 0,
        "fixed": {"model_file_sha256": {"model.safetensors": "m"}, "prompt_tokens": 512, "generation_tokens": 128},
        "host": {"system": system},
        "child": {"prompt_ids_sha256": prompt_sha, "device_info": {"device_name": system}},
        "summary": {"prompt_tps": {"median": prompt_tps}, "generation_tps": {"median": gen_tps},
                    "generated_ids_sha256": ids_sha},
    }


class M4MiniBenchTest(unittest.TestCase):
    def run_script(self, *args):
        return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)

    def test_self_test(self):
        proc = self.run_script("--self-test")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("self-test ok", proc.stdout)

    def compare(self, a, b):
        with tempfile.TemporaryDirectory() as tmp:
            pa, pb = Path(tmp, "a.json"), Path(tmp, "b.json")
            pa.write_text(json.dumps(a))
            pb.write_text(json.dumps(b))
            return self.run_script("--compare", str(pa), str(pb))

    def test_compare_reports_ratio(self):
        proc = self.compare(result("Linux", 300.0, 40.0), result("Darwin", 600.0, 80.0))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("50.0%", proc.stdout)
        self.assertIn("identical", proc.stdout)

    def test_compare_refuses_different_prompt(self):
        proc = self.compare(result("Linux", 1.0, 1.0, prompt_sha="a"), result("Darwin", 1.0, 1.0, prompt_sha="b"))
        self.assertEqual(proc.returncode, 1)

    def test_compare_refuses_failed_run(self):
        bad = result("Linux", 1.0, 1.0)
        bad["exit_code"] = 4
        proc = self.compare(bad, result("Darwin", 1.0, 1.0))
        self.assertEqual(proc.returncode, 2)


if __name__ == "__main__":
    unittest.main()
