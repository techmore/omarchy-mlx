#!/usr/bin/env python3
"""m4_mini_bench: one fixed language-model benchmark, Linux and macOS, MLX and TensorFold.

The release gate for the Mac mini M4 (J773g, T8132, G16G GPU) compares the
Linux stack (Aurora kernel, Aurora Honeykrisp Vulkan) against macOS on the same
machine. Two engines run the same fixed work:

  - mlx: MLX through mlx-lm, in process (mlx-omarchy on Linux, MLX on macOS).
  - tensorfold: ashhart/TensorFold's native server over loopback HTTP. On
    Linux that is the opt-in Vulkan backend (upstream PR 1544, packaged as
    tensorfold-vulkan); on macOS the release's Metal backend (v1.1.9, the last
    release the PR head contains), downloaded and hash-checked here.

Every number is only useful if both sides ran exactly the same work and the
run says what produced it, so this script fixes everything that can be fixed:

  - Model: one of two pinned Hugging Face snapshots, with every listed file
    checked by SHA-256. qwen3-1.7b (mlx-community/Qwen3-1.7B-4bit, MLX only,
    the original gate) or qwen3.5-2b (mlx-community/Qwen3.5-2B-MLX-4bit, the
    one checkpoint TensorFold's Vulkan engine serves; MLX runs it too).
  - Prompt: exactly 512 token IDs, cut from a fixed English text. The SHA-256
    of the IDs is recorded, so a tokenizer difference between the two systems
    or engines shows up as a hash difference instead of a silent change of work.
    MLX tokenizes with the checkpoint's tokenizer, TensorFold with its own
    (/v1/tokenize); the IDs are then sent to the server as IDs.
  - Generation: exactly 128 tokens, greedy, with every EOS token masked (MLX)
    or ignored (TensorFold's ignore_eos) so the model cannot stop early. A run
    that produces another count is refused.
  - Repeats: 1 warmup run (pipeline and shader compilation), then 5 measured
    runs. Medians, minimum, maximum and spread are reported.
  - Metric: prompt and generation tokens per second as each engine reports
    them (mlx-lm's prompt_tps and generation_tps; TensorFold's prefill_seconds
    and tokens_per_second), plus the time to first token.

Recorded with every result: OS and kernel, the Aurora package versions and the
kernel/m1n1/Mesa commits they pin (from the omarchy-mlx-m4 stack manifest when
installed), the GPU as the engine sees it (MLX: mx.device_info(); TensorFold:
its /health device and, on Linux, vulkaninfo through the same driver manifest),
the Vulkan and GL renderer strings when vulkaninfo or eglinfo/glxinfo are
installed, the macOS GPU name and core count, the engine's versions and source
pins, the git commit of this script, the load average before the run and the
CPU governor.

The measurement runs in a child process (MLX) or a server process
(TensorFold) under a wall-clock timeout, holding the per-host GPU lock
(/tmp/m1-gpu.lock, the same lock the other omarchy-mlx hardware tools take).
On the G16 kernel a GPU fault can leave the GPU stopped until the next boot; a
timeout ends the run with exit code 4 and a partial record instead of hanging.

Usage:
  python3 scripts/m4_mini_bench.py --out result.json
  python3 scripts/m4_mini_bench.py --model qwen3.5-2b --out mlx.json
  python3 scripts/m4_mini_bench.py --engine tensorfold --out tensorfold.json
  python3 scripts/m4_mini_bench.py --compare a.json b.json [c.json ...]
  python3 scripts/m4_mini_bench.py --self-test        (no GPU, no mlx)

Linux (packaged venv):  /usr/lib/omarchy-mlx/venv/bin/python scripts/m4_mini_bench.py
macOS: a venv with mlx==0.32.3 mlx-lm==0.31.3 transformers==5.16.1, see
docs/m4-mini-bench.md. The TensorFold engine needs no mlx.

Exit codes: 0 measured; 1 the run did not do the fixed work (token counts,
prompt hash, non-GPU device, drafts or a prompt cache); 2 bad invocation or
unusable environment (model or engine download or hash failure, lock busy);
3 the child or server failed; 4 timeout.
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

SCHEMA = "m4-mini-bench/2"
# Results of the first schema are MLX runs of qwen3-1.7b; --compare reads them as such.
SCHEMAS = ("m4-mini-bench/1", SCHEMA)

MODELS = {
    "qwen3-1.7b": {
        "repo": "mlx-community/Qwen3-1.7B-4bit",
        "revision": "3b1b1768f8f8cf8351c712464f906e86c2b8269e",
        # LFS object IDs on the pinned revision: the SHA-256 of the file contents.
        "sha256": {
            "model.safetensors": "0e86d9677e519323849eac1bc272caae88567a481ff188c431f70be543d9995f",
            "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        },
        "engines": ("mlx",),
    },
    # The checkpoint TensorFold's Vulkan engine (PR 1544) and its Metal Qwen3.5-2B recipe pin.
    "qwen3.5-2b": {
        "repo": "mlx-community/Qwen3.5-2B-MLX-4bit",
        "revision": "93760be4f1f69842a46bc13dbdc0f19e291392a3",
        # LFS object IDs for the weights and tokenizer; the SHA-256 of the small
        # git files at the revision (their git blob IDs were checked too).
        "sha256": {
            "model.safetensors": "713fe7e5d3c3965f7106b0d0ee17615f7869c23c8d327996df8c1196fbcf07d5",
            "tokenizer.json": "87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4",
            "config.json": "beb7fc5a6e0405fe332821cf1a8ef7b69bb390a8c8933171647de5579debf949",
            "tokenizer_config.json": "e98f1901ac6f0adff67b1d540bfa0c36ac1a0cf59eb72ed78146ef89aafa1182",
            "chat_template.jinja": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
        },
        "engines": ("mlx", "tensorfold"),
    },
}
DEFAULT_MODEL = {"mlx": "qwen3-1.7b", "tensorfold": "qwen3.5-2b"}
MODEL_ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.jinja"]

PROMPT_TOKENS = 512
GEN_TOKENS = 128
RUNS = 5
WARMUP = 1
# The unbatched Vulkan engine decodes a few tokens a second (2.8 tok/s on an
# M1 Pro): six 128-token runs need far longer than MLX does.
CHILD_TIMEOUT_S = {"mlx": 900, "tensorfold": 2400}
SERVER_LOAD_TIMEOUT_S = 600
GPU_LOCK = "/tmp/m1-gpu.lock"
QUIET_LOAD1 = 0.5

# TensorFold. Linux: the tensorfold-vulkan package (techmore/omarchy-pkgs
# elementaryos-mseries-mlx), built from PR 1544's head; its build.json names
# the commit and the server's SHA-256. macOS: the v1.1.9 release archive.
TF_SERVED_NAME = "m4-mini-bench"
TF_CONTEXT = 2048  # 512 prompt + 128 generated, well inside the Vulkan engine's 7,168
TF_VULKAN = {
    "commit": "b3ab9ba217a80c63fcc48f7643ddfe339a8f4bdc",
    "pull_request": 1544,
    "wrapper": "/usr/bin/tensorfold-vulkan",
    "build_json": "/usr/share/tensorfold-vulkan/build.json",
    "package": "tensorfold-vulkan",
}
TF_METAL = {
    "version": "1.1.9",
    "url": "https://github.com/ashhart/TensorFold/releases/download/v1.1.9/tensorfold-1.1.9-macos-arm64.tar.gz",
    "archive_sha256": "f8330102962550beea7d312eb78553f01539452fd442b7b195d63bc0f3ffa9d7",
    "member": "tensorfold-1.1.9-macos-arm64/bin/tensorfold-native",
    "binary_sha256": "4b4b5a5ac152eb2dbed7e9d8004805ed9b74f77c64c3de18eb4f35263c7b9eb8",
}
TF_CACHE = Path(os.environ.get("M4_BENCH_CACHE", str(Path.home() / ".cache" / "m4-mini-bench")))

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
    "tensorfold-vulkan",
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

def resolve_model(model_key, model_dir):
    """Download (or reuse) the pinned snapshot and check its file hashes."""
    spec = MODELS[model_key]
    if model_dir is None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise SystemExit("error: huggingface_hub is not installed in this interpreter (exit 2)")
        model_dir = snapshot_download(spec["repo"], revision=spec["revision"],
                                      allow_patterns=MODEL_ALLOW_PATTERNS)
    model_dir = Path(model_dir)
    checked = {}
    for name, want in spec["sha256"].items():
        f = model_dir / name
        if not f.is_file():
            raise SystemExit("error: %s is missing from %s (exit 2)" % (name, model_dir))
        got = sha256_file(f)
        if got != want:
            raise SystemExit("error: %s sha256 %s, expected %s (exit 2)" % (name, got, want))
        checked[name] = got
    return model_dir, checked


# ------------------------------------------------------------ child (GPU)

def prompt_source():
    # Eight copies are about 1,800 tokens on the pinned tokenizers; the cut keeps
    # the first PROMPT_TOKENS, which do not depend on where the text ends.
    return PROMPT_TEXT * 8


def cut_prompt_ids(ids):
    if len(ids) < PROMPT_TOKENS:
        raise SystemExit("error: the fixed prompt encodes to %d tokens, fewer than %d" % (len(ids), PROMPT_TOKENS))
    return [int(i) for i in ids[:PROMPT_TOKENS]]


def build_prompt_ids(tokenizer):
    return cut_prompt_ids(tokenizer.encode(prompt_source(), add_special_tokens=False))


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


# ----------------------------------------------------- TensorFold (server)

class TFError(Exception):
    """A TensorFold leg that could not run: exit code and reason."""

    def __init__(self, code, message):
        Exception.__init__(self, message)
        self.code = code


def tensorfold_engine_name(system):
    return "tensorfold-metal" if system == "Darwin" else "tensorfold-vulkan"


def tensorfold_command(system, binary, model_dir, port):
    """The serve command: one stream, no drafts, no prompt cache, no thinking."""
    return [str(binary), "serve", str(model_dir),
            "--backend", "mlx" if system == "Darwin" else "vulkan",
            "--name", TF_SERVED_NAME, "--host", "127.0.0.1", "--port", str(port),
            "--context", str(TF_CONTEXT), "--parallel", "1", "--no-drafts",
            "--no-thinking", "--no-update-check"]


def completion_body(prompt_ids):
    return {"model": TF_SERVED_NAME, "prompt": list(prompt_ids), "max_tokens": GEN_TOKENS,
            "temperature": 0, "ignore_eos": True, "return_token_ids": True, "stream": False}


def tf_run_record(reply, wall_s):
    """One measured request, from TensorFold's /v1/completions reply.

    usage gives the token counts; the reply's "tensorfold" block gives the
    server's own timings: tokens_per_second over the decode (first token to
    last), prefill_seconds when the engine reports it, time_to_first_token from
    the request's arrival, and the generated IDs with their SHA-256.
    """
    usage = reply.get("usage") or {}
    tf = reply.get("tensorfold") or {}
    choice = (reply.get("choices") or [{}])[0]
    ids = [int(t) for t in tf.get("token_ids") or []]
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    prefill_s = tf.get("prefill_seconds")
    ttft = tf.get("time_to_first_token")
    if prefill_s:
        prompt_tps, prompt_tps_from = prompt_tokens / float(prefill_s), "prefill_seconds"
    elif ttft:
        prompt_tps, prompt_tps_from = prompt_tokens / float(ttft), "time_to_first_token"
    else:
        prompt_tps, prompt_tps_from = 0.0, None
    return {
        "prompt_tokens": prompt_tokens,
        "generation_tokens": int(usage.get("completion_tokens") or 0),
        "cached_tokens": int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0),
        "prompt_tps": float(prompt_tps),
        "prompt_tps_from": prompt_tps_from,
        "generation_tps": float(tf.get("tokens_per_second") or 0.0),
        "decode_s": tf.get("seconds"),
        "prefill_s": prefill_s,
        "ttft_s": round(float(ttft), 4) if ttft is not None else None,
        "wall_s": round(wall_s, 4),
        "finish_reason": choice.get("finish_reason"),
        "drafts": tf.get("drafts"),
        "sampling": tf.get("sampling"),
        "server_token_sha256": tf.get("token_sha256"),
        "generated_ids_sha256": ids_sha256(ids) if ids else None,
        "generated_ids_count": len(ids),
    }


def tf_check_runs(rec):
    """Refusals specific to the TensorFold leg, on top of check_runs."""
    problems = []
    for i, r in enumerate(rec.get("runs", []) + rec.get("warmup", [])):
        if r.get("generated_ids_sha256") is None:
            problems.append("run %d returned no token ids" % i)
        elif r.get("server_token_sha256") not in (None, r["generated_ids_sha256"]):
            problems.append("run %d: the server's token_sha256 does not match its token ids" % i)
        if r.get("generated_ids_count") != r.get("generation_tokens"):
            problems.append("run %d returned %s ids for %s tokens" % (i, r.get("generated_ids_count"), r.get("generation_tokens")))
        if r.get("cached_tokens"):
            problems.append("run %d reused %d cached prompt tokens" % (i, r["cached_tokens"]))
        if r.get("drafts"):
            problems.append("run %d drafted tokens; the fixed work is plain greedy decoding" % i)
        if not r.get("prompt_tps_from"):
            problems.append("run %d reported neither prefill_seconds nor time_to_first_token" % i)
    return problems


def http_json(url, body=None, timeout=60):
    import urllib.request

    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=max(1, timeout)) as resp:
        return json.loads(resp.read().decode())


def free_port():
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def tensorfold_metal_binary():
    """The pinned macOS release binary, downloaded once into the cache and hash-checked."""
    import tarfile
    import urllib.request

    dest = TF_CACHE / ("tensorfold-%s-macos-arm64" % TF_METAL["version"]) / "tensorfold-native"
    if dest.is_file() and sha256_file(dest) == TF_METAL["binary_sha256"]:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    archive = dest.parent / "archive.tar.gz"
    try:
        with urllib.request.urlopen(TF_METAL["url"], timeout=300) as resp, open(archive, "wb") as fh:
            shutil.copyfileobj(resp, fh)
    except OSError as exc:
        raise TFError(2, "cannot download %s: %s" % (TF_METAL["url"], exc))
    got = sha256_file(archive)
    if got != TF_METAL["archive_sha256"]:
        raise TFError(2, "TensorFold archive sha256 %s, expected %s" % (got, TF_METAL["archive_sha256"]))
    with tarfile.open(archive) as tar:
        member = tar.getmember(TF_METAL["member"])
        src = tar.extractfile(member)
        if not member.isfile() or src is None:
            raise TFError(2, "%s is not a file in the TensorFold archive" % TF_METAL["member"])
        tmp = dest.with_suffix(".part")
        with open(tmp, "wb") as fh:
            shutil.copyfileobj(src, fh)
    got = sha256_file(tmp)
    if got != TF_METAL["binary_sha256"]:
        tmp.unlink()
        raise TFError(2, "tensorfold-native sha256 %s, expected %s" % (got, TF_METAL["binary_sha256"]))
    tmp.chmod(0o755)
    tmp.rename(dest)
    archive.unlink()
    return dest


def tensorfold_identity(system, override):
    """The binary to run and what it is: the pinned source, its hash, and whether it matched the pin."""
    ident = {"engine": tensorfold_engine_name(system)}
    if system == "Darwin":
        binary = Path(override) if override else tensorfold_metal_binary()
        if not binary.is_file():
            raise TFError(2, "TensorFold binary not found: %s" % binary)
        ident.update(binary=str(binary), binary_sha256=sha256_file(binary),
                     release=TF_METAL["version"], release_url=TF_METAL["url"])
        ident["pinned"] = ident["binary_sha256"] == TF_METAL["binary_sha256"]
        return binary, ident
    if system != "Linux":
        raise TFError(2, "the TensorFold leg runs on Linux (Vulkan) or macOS (Metal), not %s" % system)
    build = None
    if Path(TF_VULKAN["build_json"]).is_file():
        try:
            build = json.loads(Path(TF_VULKAN["build_json"]).read_text())
        except ValueError:
            build = None
    ident["build"] = build
    if override:
        binary = Path(override)
        if not binary.is_file():
            raise TFError(2, "TensorFold binary not found: %s" % binary)
        ident.update(binary=str(binary), binary_sha256=sha256_file(binary), pinned=False)
        return binary, ident
    if build is None:
        raise TFError(2, "%s is missing or unreadable: install the %s package, or name a server with --tensorfold-bin"
                      % (TF_VULKAN["build_json"], TF_VULKAN["package"]))
    if build.get("commit") != TF_VULKAN["commit"]:
        raise TFError(2, "the installed %s is built from %s, not the pinned %s"
                      % (TF_VULKAN["package"], build.get("commit"), TF_VULKAN["commit"]))
    server = Path(build.get("server") or "")
    if not server.is_file() or sha256_file(server) != build.get("server_sha256"):
        raise TFError(2, "the installed server %s does not match the SHA-256 its package recorded" % server)
    binary = Path(TF_VULKAN["wrapper"])
    if not binary.is_file():
        raise TFError(2, "%s is missing: reinstall %s" % (binary, TF_VULKAN["package"]))
    ident.update(binary=str(binary), server=str(server), binary_sha256=build["server_sha256"], pinned=True,
                 commit=build["commit"], pull_request=build.get("pull_request"))
    icd = build.get("vulkan_icd")
    if icd:
        vk = run_vulkaninfo(icd)
        if vk:
            ident["vulkaninfo"] = vk
    return binary, ident


def run_vulkaninfo(icd):
    """vulkaninfo --summary through exactly the driver manifest the server is given."""
    if shutil.which("vulkaninfo") is None:
        return None
    env = dict(os.environ, VK_DRIVER_FILES=icd)
    env.pop("VK_ICD_FILENAMES", None)
    env.pop("VK_ADD_DRIVER_FILES", None)
    try:
        out = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True, timeout=30, env=env)
    except (OSError, subprocess.TimeoutExpired):
        return None
    keep = ("deviceName", "driverName", "driverID", "driverInfo", "apiVersion", "driverVersion",
            "conformanceVersion", "deviceID", "vendorID")
    return [l.strip() for l in out.stdout.splitlines() if l.strip().split(" ")[0] in keep] or None


def stop_server(proc):
    import signal

    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except OSError:
        proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            proc.kill()
        proc.wait(timeout=20)


def tensorfold_measure(binary, model_dir, runs, warmup, timeout, system=None, log_dir=None):
    """Serve the checkpoint, tokenize the fixed text, then warmup + measured requests.

    Returns the child-shaped record (the same keys the MLX child prints, plus
    the server's health and log tail). Raises TFError on failure; code 4 when
    the wall-clock deadline passes.
    """
    import tempfile
    import urllib.error

    system = system or platform.system()
    deadline = time.monotonic() + timeout
    remaining = lambda: deadline - time.monotonic()  # noqa: E731
    port = free_port()
    base = "http://127.0.0.1:%d" % port
    log_dir = Path(log_dir or tempfile.mkdtemp(prefix="m4-mini-bench-tf-"))
    log_path = log_dir / "server.log"
    rec = {"engine": tensorfold_engine_name(system),
           "command": tensorfold_command(system, binary, model_dir, port), "server_log": str(log_path)}
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(rec["command"], stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
    try:
        t0 = time.perf_counter()
        health = None
        load_deadline = min(deadline, time.monotonic() + SERVER_LOAD_TIMEOUT_S)
        while health is None:
            if proc.poll() is not None:
                raise TFError(3, "the TensorFold server exited during startup (%d)" % proc.returncode)
            if time.monotonic() >= load_deadline:
                raise TFError(4, "the TensorFold server did not become healthy in time")
            try:
                h = http_json(base + "/health", timeout=5)
                if h.get("status") == "ok":
                    health = h
            except (OSError, ValueError, urllib.error.URLError):
                pass
            if health is None:
                time.sleep(1)
        rec["load_s"] = round(time.perf_counter() - t0, 3)
        rec["health"] = health
        tok = http_json(base + "/v1/tokenize", {"prompt": prompt_source(), "add_special_tokens": False},
                        timeout=remaining())
        prompt_ids = cut_prompt_ids(tok.get("tokens") or [])
        rec["prompt_ids_sha256"] = ids_sha256(prompt_ids)
        rec["ignore_eos"] = True

        def one_run():
            if remaining() <= 0:
                raise TFError(4, "the TensorFold measurement passed its deadline")
            start = time.perf_counter()
            try:
                reply = http_json(base + "/v1/completions", completion_body(prompt_ids), timeout=remaining())
            except OSError as exc:  # urllib.error.URLError and socket timeouts are OSErrors
                if remaining() <= 0:
                    raise TFError(4, "the TensorFold request passed the deadline")
                raise TFError(3, "the TensorFold request failed: %s" % exc)
            return tf_run_record(reply, time.perf_counter() - start)

        rec["warmup"] = [one_run() for _ in range(warmup)]
        rec["runs"] = [one_run() for _ in range(runs)]
        try:
            rec["health_end"] = http_json(base + "/health", timeout=10)
        except (OSError, ValueError):
            pass
        return rec
    except TFError as exc:
        exc.record = rec  # the partial record, completed below with the server's log tail
        raise
    except (OSError, ValueError) as exc:
        err = TFError(3, "the TensorFold leg failed: %s" % exc)
        err.record = rec
        raise err
    finally:
        stop_server(proc)
        rec["server_returncode"] = proc.returncode
        try:
            rec["server_log_tail"] = log_path.read_text(errors="replace").splitlines()[-40:]
        except OSError:
            pass


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
    child = result.get("child", {})
    runs = child.get("runs", [])
    peaks = [r["peak_memory_gb"] for r in runs if r.get("peak_memory_gb") is not None]
    peak = max(peaks) if peaks else None
    if peak is None:  # TensorFold: the server's own peak, from /health after the runs
        mem = (child.get("health_end") or {}).get("memory") or {}
        if mem.get("peak"):
            peak = round(mem["peak"] / 1e9, 3)
    return {
        "prompt_tps": stats([r["prompt_tps"] for r in runs]),
        "generation_tps": stats([r["generation_tps"] for r in runs]),
        "ttft_s": stats([r["ttft_s"] for r in runs if r.get("ttft_s") is not None]),
        "peak_memory_gb": peak,
        "generated_ids_sha256": runs[0]["generated_ids_sha256"] if runs else None,
    }


def renderer_line(result):
    child = result.get("child", {})
    if "health" in child:  # TensorFold: the server's device name, else Vulkan's
        vk = (result.get("engine_identity") or {}).get("vulkaninfo") or []
        names = [l.split("=", 1)[1].strip() for l in vk if "=" in l and l.split()[0] in ("deviceName", "driverName", "driverInfo")]
        parts = [child["health"].get("device")] + names if child["health"].get("device") else names
        parts = parts or [child["health"].get("backend")]
        return " | ".join(str(p) for p in parts if p)
    di = child.get("device_info", {})
    parts = [di.get("device_name"), di.get("driver"), di.get("driver_info")]
    if not any(parts):
        parts = [di.get("device_name") or di.get("architecture")]
    return " | ".join(str(p) for p in parts if p)


def result_engine(result):
    """mlx, tensorfold-vulkan or tensorfold-metal; a first-schema result is MLX."""
    return result.get("fixed", {}).get("engine", "mlx")


def result_model(result):
    return result.get("fixed", {}).get("model", "qwen3-1.7b")


def run_mlx(args, model_dir, result):
    child = [sys.executable, str(Path(__file__).resolve()), "--child", "--model-dir", str(model_dir),
             "--runs", str(args.runs), "--warmup", str(args.warmup)]
    try:
        proc = subprocess.run(child, capture_output=True, text=True, timeout=args.timeout)
    except subprocess.TimeoutExpired as exc:
        result["timeout_s"] = args.timeout
        tail = exc.stderr or ""
        if isinstance(tail, bytes):
            tail = tail.decode(errors="replace")
        result["child_stderr_tail"] = tail.splitlines()[-40:]
        return 4
    result["child_returncode"] = proc.returncode
    result["child_stderr_tail"] = proc.stderr.splitlines()[-40:]
    for line in proc.stdout.splitlines():
        if line.startswith('{"child"'):
            result.update(json.loads(line))
    if "child" not in result:
        return 3
    if proc.returncode == 1 or result["child"].get("refusal"):
        return 1
    if proc.returncode != 0:
        return 3
    return 0


def run_tensorfold(args, model_dir, binary, result):
    try:
        result["child"] = tensorfold_measure(binary, model_dir, args.runs, args.warmup, args.timeout)
    except TFError as exc:
        result["child"] = getattr(exc, "record", {})
        result["error"] = str(exc)
        if exc.code == 4:
            result["timeout_s"] = args.timeout
        return exc.code
    return 0


def parent_main(args):
    system = platform.system()
    engine = args.engine if args.engine == "mlx" else tensorfold_engine_name(system)
    result = {
        "schema": SCHEMA,
        "fixed": {"engine": engine, "model": args.model, "model_repo": MODELS[args.model]["repo"],
                  "model_revision": MODELS[args.model]["revision"],
                  "prompt_tokens": PROMPT_TOKENS, "generation_tokens": GEN_TOKENS,
                  "runs": args.runs, "warmup": args.warmup,
                  "sampler": "greedy, EOS masked" if args.engine == "mlx" else "greedy (temperature 0), ignore_eos"},
        "script": script_identity(),
        "command": [sys.executable] + sys.argv,
        "host": host_provenance(),
    }
    binary = None
    if args.engine == "tensorfold":
        try:
            binary, result["engine_identity"] = tensorfold_identity(system, args.tensorfold_bin)
        except TFError as exc:
            print("error: %s (exit %d)" % (exc, exc.code), file=sys.stderr)
            return exc.code
    model_dir, hashes = resolve_model(args.model, args.model_dir)
    result["fixed"]["model_file_sha256"] = hashes

    lock = acquire_lock(args.lock, args.lock_wait)
    if lock is None:
        print("error: GPU lock %s is held by another job (exit 2)" % args.lock, file=sys.stderr)
        return 2
    try:
        if args.engine == "mlx":
            code = run_mlx(args, model_dir, result)
        else:
            code = run_tensorfold(args, model_dir, binary, result)
    finally:
        lock.close()

    result["host"]["utc_end"] = utc_now()
    result["host"]["load_avg_end"] = list(os.getloadavg()) if hasattr(os, "getloadavg") else None
    if code == 0:
        problems = check_runs(result["child"])
        if args.engine == "tensorfold":
            problems += tf_check_runs(result["child"])
        if problems:
            result["refusal"] = problems
            code = 1
        result["summary"] = summarize(result)
    result["exit_code"] = code

    out = Path(args.out or "m4-mini-bench-%s-%s-%s.json" % (
        system.lower(), engine, datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")))
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")

    print("m4-mini-bench %s with %s on %s %s" % (MODELS[args.model]["repo"], engine, system, platform.release()))
    print("renderer: %s" % (renderer_line(result) or "unknown"))
    if "summary" in result:
        s = result["summary"]
        print("prompt     %8.2f tok/s median over %d runs of %d tokens (spread %.1f%%)" % (
            s["prompt_tps"]["median"], s["prompt_tps"]["n"], PROMPT_TOKENS, 100 * (s["prompt_tps"]["spread"] or 0)))
        print("generation %8.2f tok/s median over %d runs of %d tokens (spread %.1f%%)" % (
            s["generation_tps"]["median"], s["generation_tps"]["n"], GEN_TOKENS, 100 * (s["generation_tps"]["spread"] or 0)))
        peak = "%.2f GB" % s["peak_memory_gb"] if s["peak_memory_gb"] is not None else "not reported"
        print("peak memory %s, generated ids %s" % (peak, (s["generated_ids_sha256"] or "?")[:16]))
    if result.get("error"):
        print("error: %s" % result["error"], file=sys.stderr)
    if not result["host"]["quiet"]:
        print("note: load average %.2f at start is above %.1f; the numbers may be low" % (
            result["host"]["load_avg_start"][0], QUIET_LOAD1))
    if result.get("refusal"):
        for p in result["refusal"]:
            print("refused: %s" % p, file=sys.stderr)
    print("wrote %s (exit %d)" % (out, code))
    return code


# ---------------------------------------------------------------- compare

def result_label(result):
    return "%s %s" % (result.get("host", {}).get("system", "?"), result_engine(result))


def compare_main(paths):
    """Two or more results of the same work; the last one is the reference."""
    results = [json.loads(Path(p).read_text()) for p in paths]
    for r, p in zip(results, paths):
        if r.get("schema") not in SCHEMAS or r.get("exit_code") != 0:
            print("error: %s is not a successful m4-mini-bench result" % p, file=sys.stderr)
            return 2
    ref = results[-1]

    def work(r):
        return (r["fixed"]["model_file_sha256"], r["child"]["prompt_ids_sha256"],
                r["fixed"]["prompt_tokens"], r["fixed"]["generation_tokens"])

    for r, p in zip(results, paths):
        same = {
            "model": r["fixed"]["model_file_sha256"] == ref["fixed"]["model_file_sha256"],
            "prompt_ids": r["child"]["prompt_ids_sha256"] == ref["child"]["prompt_ids_sha256"],
            "shape": work(r)[2:] == work(ref)[2:],
        }
        if not all(same.values()):
            print("error: %s did not run the same work as %s: %s" % (p, paths[-1], same), file=sys.stderr)
            return 1
    print("model %s, %d prompt + %d generated tokens; reference: %s" % (
        result_model(ref), ref["fixed"]["prompt_tokens"], ref["fixed"]["generation_tokens"], result_label(ref)))
    print("%-24s %12s %12s %9s %9s %9s  %s" % ("result", "prompt tok/s", "gen tok/s", "ttft s",
                                               "prompt", "gen", "generated ids"))
    ref_ids = ref["summary"]["generated_ids_sha256"]
    for r in results:
        s = r["summary"]
        x, y = s["prompt_tps"]["median"], s["generation_tps"]["median"]
        rx, ry = ref["summary"]["prompt_tps"]["median"], ref["summary"]["generation_tps"]["median"]
        ttft = (s.get("ttft_s") or {}).get("median")
        ids = s["generated_ids_sha256"] or "?"
        same = "reference" if r is ref else ("identical" if ids == ref_ids else "DIFFER")
        print("%-24s %12.2f %12.2f %9s %8.1f%% %8.1f%%  %s %s" % (
            result_label(r), x, y, "%.3f" % ttft if ttft is not None else "-",
            100.0 * x / rx if rx else float("nan"), 100.0 * y / ry if ry else float("nan"), ids[:16], same))
    for r in results:
        print("renderer %s: %s" % (result_label(r), renderer_line(r) or "unknown"))
    loud = [result_label(r) for r in results if not r.get("host", {}).get("quiet", True)]
    if loud:
        print("note: not quiet at start (load average above %.1f): %s" % (QUIET_LOAD1, ", ".join(loud)))
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

    # TensorFold: a reply as /v1/completions returns it, and the refusals.
    gen = list(range(GEN_TOKENS))
    reply = {"choices": [{"finish_reason": "length"}],
             "usage": {"prompt_tokens": PROMPT_TOKENS, "completion_tokens": GEN_TOKENS,
                       "prompt_tokens_details": {"cached_tokens": 0}},
             "tensorfold": {"tokens_per_second": 20.0, "seconds": 6.35, "prefill_seconds": 2.0,
                            "time_to_first_token": 2.1, "drafts": False, "token_ids": gen,
                            "token_sha256": ids_sha256(gen)}}
    r = tf_run_record(reply, 8.5)
    assert r["prompt_tps"] == PROMPT_TOKENS / 2.0 and r["prompt_tps_from"] == "prefill_seconds", r
    assert check_runs({"runs": [r]}) == [] and tf_check_runs({"runs": [r]}) == []
    cached = tf_run_record(dict(reply, usage=dict(reply["usage"], prompt_tokens_details={"cached_tokens": 64})), 1.0)
    assert any("cached" in p for p in tf_check_runs({"runs": [cached]}))
    for model, spec in MODELS.items():
        assert all(len(v) == 64 for v in spec["sha256"].values()), model
    assert set(DEFAULT_MODEL.values()) <= set(MODELS)
    cmd = tensorfold_command("Linux", "/usr/bin/tensorfold-vulkan", "/m", 8000)
    assert cmd[cmd.index("--backend") + 1] == "vulkan" and "--no-drafts" in cmd
    print("self-test ok")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", help="result JSON path (default m4-mini-bench-<os>-<engine>-<utc>.json)")
    ap.add_argument("--engine", choices=("mlx", "tensorfold"), default="mlx",
                    help="mlx (in process) or tensorfold (Vulkan on Linux, Metal on macOS)")
    ap.add_argument("--model", choices=sorted(MODELS),
                    help="pinned checkpoint (default qwen3-1.7b for mlx, qwen3.5-2b for tensorfold)")
    ap.add_argument("--model-dir", help="an already-downloaded snapshot of the pinned revision")
    ap.add_argument("--tensorfold-bin", help="TensorFold server to run instead of the pinned one (recorded as unpinned)")
    ap.add_argument("--runs", type=int, default=RUNS)
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--timeout", type=int, help="wall-clock cap, seconds (default 900 for mlx, 2400 for tensorfold)")
    ap.add_argument("--lock", default=GPU_LOCK)
    ap.add_argument("--lock-wait", type=int, default=600, help="seconds to wait for the GPU lock")
    ap.add_argument("--compare", nargs="+", metavar="RESULT.json",
                    help="two or more results of the same work; the last is the reference")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    if args.compare:
        if len(args.compare) < 2:
            ap.error("--compare needs at least two results")
        return compare_main(args.compare)
    if args.runs < 1 or args.warmup < 0:
        ap.error("--runs must be at least 1 and --warmup at least 0")
    if args.child:
        return child_main(args)
    args.model = args.model or DEFAULT_MODEL[args.engine]
    if args.engine not in MODELS[args.model]["engines"]:
        ap.error("%s does not serve %s; use --model %s" % (args.engine, args.model, DEFAULT_MODEL[args.engine]))
    if args.timeout is None:
        args.timeout = CHILD_TIMEOUT_S[args.engine]
    return parent_main(args)


if __name__ == "__main__":
    sys.exit(main())
