#!/usr/bin/env python3
"""m4_mini_bench: one fixed MLX language-model benchmark, Linux and macOS.

The release gate for the Mac mini M4 (J773g, T8132, G16G GPU) compares the
Linux stack (Aurora kernel, Aurora Honeykrisp Vulkan, mlx-omarchy) against
macOS MLX on the same machine. Every number is only useful if both sides ran
exactly the same work and the run says what produced it, so this script fixes
everything that can be fixed:

  - Model: mlx-community/Qwen3-1.7B-4bit at a pinned Hugging Face revision,
    with the weight and tokenizer files checked by SHA-256 (about 1 GB, fits a
    16 GB machine with room to spare).
  - Prompt: exactly 512 token IDs, cut from a fixed English text. The SHA-256
    of the IDs is recorded, so a tokenizer difference between the two systems
    shows up as a hash difference instead of a silent change of work.
  - Generation: exactly 128 tokens, greedy, with every EOS token masked so the
    model cannot stop early. A run that produces another count is refused.
  - Repeats: 1 warmup run (pipeline and shader compilation), then 5 measured
    runs. Medians, minimum, maximum and spread are reported.
  - Metric: mlx-lm's own prompt_tps and generation_tps from the final
    GenerationResponse (the numbers `mlx_lm.generate --verbose` prints), plus
    the time to first token measured here.

Recorded with every result: OS and kernel, the Aurora package versions and the
kernel/m1n1/Mesa commits they pin (from the omarchy-mlx-m4 stack manifest when
installed), the GPU as MLX sees it (mx.device_info(): device name, driver,
driverInfo, ICD path), the Vulkan and GL renderer strings when vulkaninfo or
eglinfo/glxinfo are installed, the macOS GPU name and core count, mlx /
mlx-omarchy / mlx-lm / transformers versions, the git commit of this script,
the load average before the run and the CPU governor.

The measurement runs in a child process under a wall-clock timeout, holding
the per-host GPU lock (/tmp/m1-gpu.lock, the same lock the other omarchy-mlx
hardware tools take). On the G16 kernel a GPU fault can leave the GPU stopped
until the next boot; a timeout ends the run with exit code 4 and a partial
record instead of hanging.

Usage:
  python3 scripts/m4_mini_bench.py --out result.json
  python3 scripts/m4_mini_bench.py --compare linux.json macos.json
  python3 scripts/m4_mini_bench.py --self-test        (no GPU, no mlx)

Linux (packaged venv):  /usr/lib/omarchy-mlx/venv/bin/python scripts/m4_mini_bench.py
macOS: a venv with mlx==0.32.3 mlx-lm==0.31.3 transformers==5.16.1, see
docs/m4-mini-bench.md.

Exit codes: 0 measured; 1 the run did not do the fixed work (token counts,
prompt hash, non-GPU device); 2 bad invocation or unusable environment
(model download or hash failure, lock busy); 3 the child failed; 4 timeout.
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "m4-mini-bench/1"

MODEL_REPO = "mlx-community/Qwen3-1.7B-4bit"
MODEL_REVISION = "3b1b1768f8f8cf8351c712464f906e86c2b8269e"
# LFS object IDs on the pinned revision: the SHA-256 of the file contents.
MODEL_FILE_SHA256 = {
    "model.safetensors": "0e86d9677e519323849eac1bc272caae88567a481ff188c431f70be543d9995f",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
}
MODEL_ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.jinja"]

PROMPT_TOKENS = 512
GEN_TOKENS = 128
RUNS = 5
WARMUP = 1
CHILD_TIMEOUT_S = 900
GPU_LOCK = "/tmp/m1-gpu.lock"
QUIET_LOAD1 = 0.5

# Fixed prompt source. Repeated as needed and cut to PROMPT_TOKENS IDs; never
# edit it, or every earlier result stops being comparable (the prompt hash in
# the result changes, which --compare refuses).
PROMPT_TEXT = (
    "The lighthouse keeper kept a careful log. Each evening she wrote the wind "
    "direction, the height of the swell, the number of ships that passed the "
    "point and the time the lamp was lit. In winter the lamp burned for sixteen "
    "hours; in summer for barely seven. She trimmed the wick at midnight, wound "
    "the clockwork that turned the lens every four hours, and polished the brass "
    "until it reflected the stairs below. Supply boats came twice a month with "
    "oil, flour, coal and letters. When the fog rolled in she sounded the horn "
    "every thirty seconds and counted the echoes off the cliffs to judge how "
    "thick it was. Over twenty years the log grew to forty volumes, a record of "
    "weather, shipping and small events: a seal on the landing, a broken pane "
    "replaced, a storm that moved the boulders on the north beach. Summarise "
    "what the log tells a reader about the work of keeping a light, and explain "
    "which entries would matter most to a sailor approaching the coast at night. "
)


# ---------------------------------------------------------------- helpers

def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ids_sha256(ids):
    return hashlib.sha256(",".join(str(int(i)) for i in ids).encode("ascii")).hexdigest()


def run_cmd(argv, timeout=20):
    """Run a short read-only command; never raises. Returns stdout or None."""
    if shutil.which(argv[0]) is None:
        return None
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout


def read_text(path):
    try:
        return Path(path).read_text(errors="replace").replace("\x00", " ").strip()
    except OSError:
        return None


def stats(values):
    if not values:
        return {}
    med = statistics.median(values)
    return {
        "median": round(med, 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "spread": round((max(values) - min(values)) / med, 4) if med else None,
        "n": len(values),
        "values": [round(v, 3) for v in values],
    }


def script_identity():
    here = Path(__file__).resolve()
    ident = {"path": str(here), "sha256": sha256_file(here)}
    commit = run_cmd(["git", "-C", str(here.parent), "rev-parse", "HEAD"])
    if commit:
        ident["git_commit"] = commit.strip()
        dirty = run_cmd(["git", "-C", str(here.parent), "status", "--porcelain", "--", str(here)])
        ident["git_dirty"] = bool(dirty and dirty.strip())
    stamp = here.parent / "m4_mini_bench.source"
    if stamp.is_file():  # written by the omarchy-mlx-m4-bench package
        ident["packaged_source"] = read_text(stamp)
    return ident


# ------------------------------------------------------------- provenance

PACMAN_PACKAGES = [
    "linux-aurora", "m1n1-aurora", "uboot-aurora", "mesa-aurora", "vulkan-aurora",
    "omarchy-mlx", "omarchy-mlx-vulkan", "omarchy-mlx-vulkan-aurora",
    "omarchy-mlx-m4-bench", "omarchy-aurora-firmware", "vulkan-icd-loader",
]
STACK_MANIFEST = "/usr/share/omarchy-mlx-m4/stack.json"


def linux_provenance():
    p = {
        "kernel_release": platform.release(),
        "proc_version": read_text("/proc/version"),
        "dt_model": read_text("/proc/device-tree/model"),
        "dt_compatible": read_text("/proc/device-tree/compatible"),
    }
    mem = read_text("/proc/meminfo") or ""
    for line in mem.splitlines():
        if line.startswith("MemTotal:"):
            p["mem_total_kib"] = int(line.split()[1])
    gov = read_text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if gov:
        p["cpu_governor_cpu0"] = gov
    params = {}
    pdir = Path("/sys/module/asahi/parameters")
    if pdir.is_dir():
        for f in sorted(pdir.iterdir()):
            v = read_text(f)
            if v is not None:
                params[f.name] = v
    if params:
        p["asahi_module_parameters"] = params

    # Installed package versions, and the commits they pin.
    q = run_cmd(["pacman", "-Q"] + PACMAN_PACKAGES)
    if q is None:  # pacman exits 1 when any name is missing; ask one by one
        q = "".join(run_cmd(["pacman", "-Q", n]) or "" for n in PACMAN_PACKAGES)
    installed = {}
    for line in (q or "").splitlines():
        parts = line.split()
        if len(parts) == 2:
            installed[parts[0]] = parts[1]
    p["packages"] = installed
    p["stack"] = stack_commits(installed)

    # Renderer strings from the system tools, when installed.
    vk = run_cmd(["vulkaninfo", "--summary"], timeout=30)
    if vk:
        keep = ("deviceName", "driverName", "driverInfo", "apiVersion",
                "driverVersion", "conformanceVersion", "deviceID", "vendorID")
        p["vulkaninfo"] = [l.strip() for l in vk.splitlines()
                           if l.strip().split(" ")[0] in keep]
    gl = run_cmd(["eglinfo", "-B"], timeout=30) or run_cmd(["glxinfo", "-B"], timeout=30)
    if gl:
        p["gl_renderer"] = [l.strip() for l in gl.splitlines()
                            if "renderer" in l.lower() or "version string" in l.lower()][:8]
    temps = {}
    for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        t = read_text(zone / "temp")
        if t and t.lstrip("-").isdigit():
            temps[(read_text(zone / "type") or zone.name)] = int(t) / 1000.0
    if temps:
        p["thermal_c"] = temps
    return p


def stack_commits(installed):
    """Commits behind the installed Aurora packages.

    The stack manifest (installed by omarchy-mlx-m4-bench) maps each package
    version to the source commit its PKGBUILD pins. A commit is reported only
    when the installed version is the manifest's version; otherwise the field
    says so, and M4_BENCH_<NAME>_COMMIT can supply it by hand.
    """
    out = {}
    manifest = {}
    if Path(STACK_MANIFEST).is_file():
        try:
            manifest = json.loads(Path(STACK_MANIFEST).read_text()).get("packages", {})
        except (OSError, ValueError):
            manifest = {}
    for name in ("linux-aurora", "m1n1-aurora", "vulkan-aurora", "omarchy-mlx"):
        entry = {"installed": installed.get(name)}
        m = manifest.get(name)
        if m:
            entry["manifest_version"] = m.get("version")
            if installed.get(name) == m.get("version"):
                entry["commit"] = m.get("commit")
                entry["source"] = m.get("source")
            else:
                entry["commit"] = None
                entry["note"] = "installed version differs from the stack manifest"
        env = os.environ.get("M4_BENCH_%s_COMMIT" % name.upper().replace("-", "_"))
        if env:
            entry["commit"] = env
            entry["commit_from"] = "environment"
        out[name] = entry
    return out


def macos_provenance():
    p = {}
    sw = run_cmd(["sw_vers"])
    if sw:
        p["sw_vers"] = dict(l.split(":", 1) for l in sw.splitlines() if ":" in l)
        p["sw_vers"] = {k.strip(): v.strip() for k, v in p["sw_vers"].items()}
    for key in ("hw.model", "machdep.cpu.brand_string", "hw.memsize", "kern.osversion"):
        v = run_cmd(["sysctl", "-n", key])
        if v:
            p[key] = v.strip()
    sp = run_cmd(["system_profiler", "-json", "SPDisplaysDataType"], timeout=60)
    if sp:
        try:
            gpus = json.loads(sp).get("SPDisplaysDataType", [])
            p["gpu"] = [{k: g.get(k) for k in ("sppci_model", "sppci_cores", "spdisplays_mtlgpufamilysupport")}
                        for g in gpus]
        except ValueError:
            pass
    lp = run_cmd(["pmset", "-g"])
    if lp:
        p["pmset"] = [l.strip() for l in lp.splitlines()
                      if l.strip().split(" ")[0] in ("lowpowermode", "powermode", "highpowermode")]
    return p


def host_provenance():
    p = {
        "system": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "utc_start": utc_now(),
        "load_avg_start": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        "env": {k: v for k, v in os.environ.items()
                if k.startswith(("MLX_", "VK_", "HK_", "AGX_", "ASAHI_", "MESA_"))},
    }
    if p["system"] == "Linux":
        p["linux"] = linux_provenance()
    elif p["system"] == "Darwin":
        p["macos"] = macos_provenance()
    la = p["load_avg_start"]
    p["quiet"] = bool(la is not None and la[0] <= QUIET_LOAD1)
    return p


# ------------------------------------------------------------------ model

def resolve_model(model_dir):
    """Download (or reuse) the pinned snapshot and check its file hashes."""
    if model_dir is None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise SystemExit("error: huggingface_hub is not installed in this interpreter (exit 2)")
        model_dir = snapshot_download(MODEL_REPO, revision=MODEL_REVISION,
                                      allow_patterns=MODEL_ALLOW_PATTERNS)
    model_dir = Path(model_dir)
    checked = {}
    for name, want in MODEL_FILE_SHA256.items():
        f = model_dir / name
        if not f.is_file():
            raise SystemExit("error: %s is missing from %s (exit 2)" % (name, model_dir))
        got = sha256_file(f)
        if got != want:
            raise SystemExit("error: %s sha256 %s, expected %s (exit 2)" % (name, got, want))
        checked[name] = got
    return model_dir, checked


# ------------------------------------------------------------ child (GPU)

def build_prompt_ids(tokenizer):
    # Eight copies are about 1,800 tokens on the pinned tokenizer; the cut keeps
    # the first PROMPT_TOKENS, which do not depend on where the text ends.
    ids = tokenizer.encode(PROMPT_TEXT * 8, add_special_tokens=False)
    if len(ids) < PROMPT_TOKENS:
        raise SystemExit("error: the fixed prompt encodes to %d tokens, fewer than %d" % (len(ids), PROMPT_TOKENS))
    return [int(i) for i in ids[:PROMPT_TOKENS]]


def child_main(args):
    import importlib.metadata as md

    import mlx.core as mx
    from mlx_lm import load, stream_generate

    versions = {}
    for dist in ("mlx", "mlx-omarchy", "mlx-metal", "mlx-lm", "transformers", "huggingface-hub"):
        try:
            versions[dist] = md.version(dist)
        except md.PackageNotFoundError:
            pass
    rec = {"versions": versions, "mx_version": getattr(mx, "__version__", None),
           "default_device": str(mx.default_device())}
    info = None
    for getter in (getattr(mx, "device_info", None), getattr(getattr(mx, "metal", None), "device_info", None)):
        if getter is None:
            continue
        try:
            info = getter()
            break
        except Exception as exc:  # noqa: BLE001 - recorded, not fatal here
            rec["device_info_error"] = repr(exc)
    if info is not None:
        rec["device_info"] = {k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in dict(info).items()}

    if "gpu" not in rec["default_device"].lower():
        rec["refusal"] = "MLX default device is %s, not the GPU" % rec["default_device"]
        print(json.dumps({"child": rec}))
        return 1

    t0 = time.perf_counter()
    model, tokenizer = load(args.model_dir)
    rec["load_s"] = round(time.perf_counter() - t0, 3)

    prompt_ids = build_prompt_ids(tokenizer)
    rec["prompt_ids_sha256"] = ids_sha256(prompt_ids)
    eos = sorted(int(e) for e in (tokenizer.eos_token_ids or []))
    rec["masked_eos_ids"] = eos
    bias_cache = {}

    def mask_eos(_tokens, logits):
        vocab = logits.shape[-1]
        bias = bias_cache.get(vocab)
        if bias is None:
            row = [0.0] * vocab
            for e in eos:
                if e < vocab:
                    row[e] = float("-inf")
            bias = bias_cache[vocab] = mx.array(row, dtype=mx.float32)
        return logits + bias

    def one_run():
        mx.clear_cache()
        if hasattr(mx, "reset_peak_memory"):
            mx.reset_peak_memory()
        ids = []
        last = None
        start = time.perf_counter()
        ttft = None
        for resp in stream_generate(model, tokenizer, prompt_ids, max_tokens=GEN_TOKENS,
                                    logits_processors=[mask_eos]):
            if ttft is None:
                ttft = time.perf_counter() - start
            ids.append(int(resp.token))
            last = resp
        wall = time.perf_counter() - start
        return {
            "prompt_tokens": int(last.prompt_tokens),
            "generation_tokens": int(last.generation_tokens),
            "prompt_tps": float(last.prompt_tps),
            "generation_tps": float(last.generation_tps),
            "ttft_s": round(ttft, 4),
            "wall_s": round(wall, 4),
            "peak_memory_gb": round(float(last.peak_memory), 3),
            "finish_reason": last.finish_reason,
            "generated_ids_sha256": ids_sha256(ids),
        }

    rec["warmup"] = [one_run() for _ in range(args.warmup)]
    rec["runs"] = [one_run() for _ in range(args.runs)]
    print(json.dumps({"child": rec}))
    return 0


# ----------------------------------------------------------------- parent

def acquire_lock(path, wait_s):
    import fcntl

    fh = open(path, "a+")
    deadline = time.monotonic() + wait_s
    while True:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fh
        except BlockingIOError:
            if time.monotonic() >= deadline:
                fh.close()
                return None
            time.sleep(2)


def check_runs(rec):
    """Refusals for runs that did not do the fixed work."""
    problems = []
    for i, r in enumerate(rec.get("runs", []) + rec.get("warmup", [])):
        if r["prompt_tokens"] != PROMPT_TOKENS:
            problems.append("run %d processed %d prompt tokens, not %d" % (i, r["prompt_tokens"], PROMPT_TOKENS))
        if r["generation_tokens"] != GEN_TOKENS:
            problems.append("run %d generated %d tokens, not %d" % (i, r["generation_tokens"], GEN_TOKENS))
    digests = {r["generated_ids_sha256"] for r in rec.get("runs", [])}
    if len(digests) > 1:
        problems.append("greedy runs disagree on the generated IDs: %s" % sorted(digests))
    return problems


def summarize(result):
    runs = result.get("child", {}).get("runs", [])
    return {
        "prompt_tps": stats([r["prompt_tps"] for r in runs]),
        "generation_tps": stats([r["generation_tps"] for r in runs]),
        "ttft_s": stats([r["ttft_s"] for r in runs]),
        "peak_memory_gb": max((r["peak_memory_gb"] for r in runs), default=None),
        "generated_ids_sha256": runs[0]["generated_ids_sha256"] if runs else None,
    }


def renderer_line(result):
    di = result.get("child", {}).get("device_info", {})
    parts = [di.get("device_name"), di.get("driver"), di.get("driver_info")]
    if not any(parts):
        parts = [di.get("device_name") or di.get("architecture")]
    return " | ".join(str(p) for p in parts if p)


def parent_main(args):
    result = {
        "schema": SCHEMA,
        "fixed": {"model_repo": MODEL_REPO, "model_revision": MODEL_REVISION,
                  "prompt_tokens": PROMPT_TOKENS, "generation_tokens": GEN_TOKENS,
                  "runs": args.runs, "warmup": args.warmup, "sampler": "greedy, EOS masked"},
        "script": script_identity(),
        "command": [sys.executable] + sys.argv,
        "host": host_provenance(),
    }
    model_dir, hashes = resolve_model(args.model_dir)
    result["fixed"]["model_file_sha256"] = hashes

    lock = acquire_lock(args.lock, args.lock_wait)
    if lock is None:
        print("error: GPU lock %s is held by another job (exit 2)" % args.lock, file=sys.stderr)
        return 2
    child = [sys.executable, str(Path(__file__).resolve()), "--child", "--model-dir", str(model_dir),
             "--runs", str(args.runs), "--warmup", str(args.warmup)]
    code = 0
    try:
        proc = subprocess.run(child, capture_output=True, text=True, timeout=args.timeout)
        result["child_returncode"] = proc.returncode
        result["child_stderr_tail"] = proc.stderr.splitlines()[-40:]
        for line in proc.stdout.splitlines():
            if line.startswith('{"child"'):
                result.update(json.loads(line))
        if "child" not in result:
            code = 3
        elif proc.returncode == 1 or result["child"].get("refusal"):
            code = 1
        elif proc.returncode != 0:
            code = 3
    except subprocess.TimeoutExpired as exc:
        result["timeout_s"] = args.timeout
        tail = exc.stderr or ""
        if isinstance(tail, bytes):
            tail = tail.decode(errors="replace")
        result["child_stderr_tail"] = tail.splitlines()[-40:]
        code = 4
    finally:
        lock.close()

    result["host"]["utc_end"] = utc_now()
    result["host"]["load_avg_end"] = list(os.getloadavg()) if hasattr(os, "getloadavg") else None
    if code == 0:
        problems = check_runs(result["child"])
        if problems:
            result["refusal"] = problems
            code = 1
        result["summary"] = summarize(result)
    result["exit_code"] = code

    out = Path(args.out or "m4-mini-bench-%s-%s.json" % (platform.system().lower(),
                                                        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")))
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")

    print("m4-mini-bench %s on %s %s" % (MODEL_REPO, platform.system(), platform.release()))
    print("renderer: %s" % (renderer_line(result) or "unknown"))
    if "summary" in result:
        s = result["summary"]
        print("prompt     %8.2f tok/s median over %d runs of %d tokens (spread %.1f%%)" % (
            s["prompt_tps"]["median"], s["prompt_tps"]["n"], PROMPT_TOKENS, 100 * (s["prompt_tps"]["spread"] or 0)))
        print("generation %8.2f tok/s median over %d runs of %d tokens (spread %.1f%%)" % (
            s["generation_tps"]["median"], s["generation_tps"]["n"], GEN_TOKENS, 100 * (s["generation_tps"]["spread"] or 0)))
        print("peak memory %.2f GB, generated ids %s" % (s["peak_memory_gb"], s["generated_ids_sha256"][:16]))
    if not result["host"]["quiet"]:
        print("note: load average %.2f at start is above %.1f; the numbers may be low" % (
            result["host"]["load_avg_start"][0], QUIET_LOAD1))
    if result.get("refusal"):
        for p in result["refusal"]:
            print("refused: %s" % p, file=sys.stderr)
    print("wrote %s (exit %d)" % (out, code))
    return code


# ---------------------------------------------------------------- compare

def compare_main(paths):
    a, b = (json.loads(Path(p).read_text()) for p in paths)
    for r, p in ((a, paths[0]), (b, paths[1])):
        if r.get("schema") != SCHEMA or r.get("exit_code") != 0:
            print("error: %s is not a successful %s result" % (p, SCHEMA), file=sys.stderr)
            return 2
    same = {
        "model": a["fixed"]["model_file_sha256"] == b["fixed"]["model_file_sha256"],
        "prompt_ids": a["child"]["prompt_ids_sha256"] == b["child"]["prompt_ids_sha256"],
        "shape": (a["fixed"]["prompt_tokens"], a["fixed"]["generation_tokens"])
        == (b["fixed"]["prompt_tokens"], b["fixed"]["generation_tokens"]),
    }
    if not all(same.values()):
        print("error: the two results did not run the same work: %s" % same, file=sys.stderr)
        return 1
    print("%-12s %14s %14s %8s" % ("metric", a["host"]["system"], b["host"]["system"], "ratio"))
    for key in ("prompt_tps", "generation_tps"):
        x, y = a["summary"][key]["median"], b["summary"][key]["median"]
        print("%-12s %14.2f %14.2f %7.1f%%" % (key, x, y, 100.0 * x / y if y else float("nan")))
    ida, idb = a["summary"]["generated_ids_sha256"], b["summary"]["generated_ids_sha256"]
    print("generated ids %s: %s / %s" % ("identical" if ida == idb else "DIFFER", ida[:16], idb[:16]))
    print("renderers: %s || %s" % (renderer_line(a), renderer_line(b)))
    return 0


# -------------------------------------------------------------- self-test

def self_test():
    class Tok:
        def encode(self, text, add_special_tokens=False):
            return [len(w) for w in text.split()]

    ids = build_prompt_ids(Tok())
    assert len(ids) == PROMPT_TOKENS, len(ids)
    assert ids_sha256(ids) == ids_sha256(list(ids))
    good = {"prompt_tokens": PROMPT_TOKENS, "generation_tokens": GEN_TOKENS, "prompt_tps": 100.0,
            "generation_tps": 10.0, "ttft_s": 1.0, "wall_s": 2.0, "peak_memory_gb": 1.0,
            "finish_reason": "length", "generated_ids_sha256": "ab"}
    assert check_runs({"runs": [good, good], "warmup": [good]}) == []
    short = dict(good, generation_tokens=GEN_TOKENS - 3)
    assert any("generated" in p for p in check_runs({"runs": [short]}))
    other = dict(good, generated_ids_sha256="cd")
    assert any("disagree" in p for p in check_runs({"runs": [good, other]}))
    s = stats([10.0, 12.0, 11.0])
    assert s["median"] == 11.0 and abs(s["spread"] - 2 / 11) < 1e-3
    print("self-test ok")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", help="result JSON path (default m4-mini-bench-<os>-<utc>.json)")
    ap.add_argument("--model-dir", help="an already-downloaded snapshot of the pinned revision")
    ap.add_argument("--runs", type=int, default=RUNS)
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--timeout", type=int, default=CHILD_TIMEOUT_S, help="child wall-clock cap, seconds")
    ap.add_argument("--lock", default=GPU_LOCK)
    ap.add_argument("--lock-wait", type=int, default=600, help="seconds to wait for the GPU lock")
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"))
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    if args.compare:
        return compare_main(args.compare)
    if args.runs < 1 or args.warmup < 0:
        ap.error("--runs must be at least 1 and --warmup at least 0")
    if args.child:
        return child_main(args)
    return parent_main(args)


if __name__ == "__main__":
    sys.exit(main())
