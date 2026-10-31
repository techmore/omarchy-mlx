"""scripts/m4_mini_bench.py without a GPU: the fixed-work checks, the TensorFold
leg against a fake server, the pinned-binary checks and the result comparison."""

import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "m4_mini_bench.py"


def load_bench():
    spec = importlib.util.spec_from_file_location("m4_mini_bench", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def result(system, prompt_tps, gen_tps, prompt_sha="p", ids_sha="i" * 16, schema="m4-mini-bench/1",
           engine=None, model_sha="m"):
    r = {
        "schema": schema,
        "exit_code": 0,
        "fixed": {"model_file_sha256": {"model.safetensors": model_sha}, "prompt_tokens": 512, "generation_tokens": 128},
        "host": {"system": system},
        "child": {"prompt_ids_sha256": prompt_sha, "device_info": {"device_name": system}},
        "summary": {"prompt_tps": {"median": prompt_tps}, "generation_tps": {"median": gen_tps},
                    "generated_ids_sha256": ids_sha},
    }
    if engine:
        r["fixed"].update(engine=engine, model="qwen3.5-2b")
    return r


# A stand-in for tensorfold-native: the same serve command line, and /health,
# /v1/tokenize and /v1/completions shaped as TensorFold answers them. MODE
# (from the environment) plants one fault.
FAKE_SERVER = textwrap.dedent("""\
    import hashlib, json, os, sys, time
    from http.server import BaseHTTPRequestHandler, HTTPServer

    args = sys.argv[1:]
    assert args[0] == "serve" and "--no-drafts" in args and "--parallel" in args, args
    port = int(args[args.index("--port") + 1])
    mode = os.environ.get("FAKE_TF_MODE", "ok")
    if mode == "exit":
        sys.exit(7)
    if mode == "hang":
        time.sleep(600)

    def sha(ids):
        return hashlib.sha256(",".join(str(i) for i in ids).encode()).hexdigest()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self.send(200, {"status": "ok", "model": "m4-mini-bench", "version": "1.1.9",
                                "backend": "vulkan", "memory": {"peak": 2500000000}})
            else:
                self.send(404, {})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/v1/tokenize":
                assert body["add_special_tokens"] is False
                self.send(200, {"count": 2000, "tokens": [len(w) for w in body["prompt"].split()] * 3})
            elif self.path == "/v1/completions":
                assert body["ignore_eos"] is True and body["temperature"] == 0 and body["stream"] is False
                assert len(body["prompt"]) == 512
                n = body["max_tokens"] - (1 if mode == "short" else 0)
                ids = list(range(100, 100 + n))
                self.send(200, {
                    "choices": [{"text": "x", "finish_reason": "length"}],
                    "usage": {"prompt_tokens": len(body["prompt"]), "completion_tokens": n,
                              "prompt_tokens_details": {"cached_tokens": 256 if mode == "cached" else 0}},
                    "tensorfold": {"tokens_per_second": 3.0, "seconds": 42.3, "prefill_seconds": 4.0,
                                   "time_to_first_token": 4.1, "drafts": False, "sampling": "greedy",
                                   "token_ids": ids, "token_sha256": sha(ids)}})
            else:
                self.send(404, {})

    HTTPServer(("127.0.0.1", port), H).serve_forever()
    """)


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

    def compare_many(self, *results):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i, r in enumerate(results):
                p = Path(tmp, "%d.json" % i)
                p.write_text(json.dumps(r))
                paths.append(str(p))
            return self.run_script("--compare", *paths)

    def test_compare_three_engines_against_the_last(self):
        proc = self.compare_many(
            result("Linux", 300.0, 40.0, schema="m4-mini-bench/2", engine="mlx"),
            result("Linux", 100.0, 3.0, schema="m4-mini-bench/2", engine="tensorfold-vulkan", ids_sha="j" * 16),
            result("Darwin", 600.0, 80.0, schema="m4-mini-bench/2", engine="tensorfold-metal"))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Linux mlx", proc.stdout)
        self.assertIn("Linux tensorfold-vulkan", proc.stdout)
        self.assertIn("reference: Darwin tensorfold-metal", proc.stdout)
        self.assertIn("50.0%", proc.stdout)   # MLX prompt and generation, half the reference
        self.assertIn("3.8%", proc.stdout)    # Vulkan generation, 3 of 80
        self.assertIn("DIFFER", proc.stdout)

    def test_compare_refuses_another_model(self):
        proc = self.compare_many(result("Linux", 1.0, 1.0, model_sha="a"), result("Darwin", 1.0, 1.0, model_sha="b"))
        self.assertEqual(proc.returncode, 1)

    def test_compare_needs_two(self):
        proc = self.run_script("--compare", "only.json")
        self.assertEqual(proc.returncode, 2)

    def test_tensorfold_refuses_the_other_model(self):
        proc = self.run_script("--engine", "tensorfold", "--model", "qwen3-1.7b")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("does not serve", proc.stderr)


class TensorFoldLegTest(unittest.TestCase):
    def setUp(self):
        self.bench = load_bench()
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.fake = self.dir / "tensorfold-fake"
        self.fake.write_text("#!%s\n%s" % (sys.executable, FAKE_SERVER))
        self.fake.chmod(0o755)
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(os.environ.pop, "FAKE_TF_MODE", None)

    def measure(self, mode="ok", timeout=60):
        os.environ["FAKE_TF_MODE"] = mode
        return self.bench.tensorfold_measure(self.fake, self.dir, runs=2, warmup=1, timeout=timeout,
                                             system="Linux", log_dir=self.dir)

    def test_measure_does_the_fixed_work(self):
        rec = self.measure()
        b = self.bench
        self.assertEqual(rec["engine"], "tensorfold-vulkan")
        self.assertEqual(len(rec["runs"]), 2)
        self.assertEqual(len(rec["warmup"]), 1)
        self.assertEqual(b.check_runs(rec), [])
        self.assertEqual(b.tf_check_runs(rec), [])
        words = [len(w) for w in b.prompt_source().split()] * 3
        self.assertEqual(rec["prompt_ids_sha256"], b.ids_sha256(words[:512]))
        run = rec["runs"][0]
        self.assertEqual(run["prompt_tps"], 128.0)  # 512 tokens over prefill_seconds 4.0
        self.assertEqual(run["generation_tps"], 3.0)
        self.assertIsNotNone(rec["server_returncode"])  # the server was stopped
        summary = b.summarize({"child": dict(rec, health_end=rec["health"])})
        self.assertEqual(summary["peak_memory_gb"], 2.5)
        self.assertEqual(summary["generation_tps"]["median"], 3.0)

    def test_planted_faults_are_refused(self):
        rec = self.measure("cached")
        self.assertTrue(any("cached" in p for p in self.bench.tf_check_runs(rec)))
        rec = self.measure("short")
        self.assertTrue(any("generated 127" in p for p in self.bench.check_runs(rec)))

    def test_server_exit_and_timeout(self):
        with self.assertRaises(self.bench.TFError) as cm:
            self.measure("exit")
        self.assertEqual(cm.exception.code, 3)
        self.assertIn("server_log_tail", cm.exception.record)
        with self.assertRaises(self.bench.TFError) as cm:
            self.measure("hang", timeout=3)
        self.assertEqual(cm.exception.code, 4)

    def test_linux_identity_follows_the_package_record(self):
        b = self.bench
        server = self.dir / "tensorfold-native-vulkan"
        server.write_bytes(b"server")
        wrapper = self.dir / "tensorfold-vulkan"
        wrapper.write_text("#!/bin/bash\n")
        build = {"commit": b.TF_VULKAN["commit"], "pull_request": 1544, "server": str(server),
                 "server_sha256": hashlib.sha256(b"server").hexdigest(), "vulkan_icd": str(self.dir / "none.json")}
        bj = self.dir / "build.json"
        bj.write_text(json.dumps(build))
        b.TF_VULKAN = dict(b.TF_VULKAN, build_json=str(bj), wrapper=str(wrapper))
        binary, ident = b.tensorfold_identity("Linux", None)
        self.assertEqual(binary, wrapper)
        self.assertTrue(ident["pinned"])
        self.assertEqual(ident["commit"], b.TF_VULKAN["commit"])
        server.write_bytes(b"changed")
        with self.assertRaises(b.TFError) as cm:
            b.tensorfold_identity("Linux", None)
        self.assertEqual(cm.exception.code, 2)
        bj.write_text(json.dumps(dict(build, commit="0" * 40)))
        with self.assertRaises(b.TFError):
            b.tensorfold_identity("Linux", None)
        _, ident = b.tensorfold_identity("Linux", str(server))
        self.assertFalse(ident["pinned"])

    def test_metal_release_is_hash_checked(self):
        b = self.bench
        payload = b"#!/bin/sh\n"
        archive = self.dir / "release.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            info = tarfile.TarInfo(b.TF_METAL["member"])
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        b.TF_CACHE = self.dir / "cache"
        b.TF_METAL = dict(b.TF_METAL, url=archive.as_uri(),
                          archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                          binary_sha256=hashlib.sha256(payload).hexdigest())
        binary, ident = b.tensorfold_identity("Darwin", None)
        self.assertTrue(ident["pinned"])
        self.assertEqual(binary.read_bytes(), payload)
        self.assertTrue(os.access(binary, os.X_OK))
        binary.unlink()
        b.TF_METAL = dict(b.TF_METAL, archive_sha256="0" * 64)
        with self.assertRaises(b.TFError) as cm:
            b.tensorfold_metal_binary()
        self.assertEqual(cm.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
