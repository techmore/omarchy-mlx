# Mac mini M4 benchmark (release gate)

`scripts/m4_mini_bench.py` measures one fixed piece of language-model work on the Mac mini M4 (J773g, T8132, 10-core G16G GPU), under Linux (Aurora kernel, Aurora Honeykrisp Vulkan) and under macOS on the same machine, and compares the results. It is the speed measurement for the M4 mini release gate. It does not certify M4 GPU compute; correctness stays with the existing gates.

Two engines run the same work:

| Engine | Linux | macOS |
|---|---|---|
| `mlx` (default) | mlx-omarchy on Honeykrisp Vulkan, in process | MLX on Metal, in process |
| `tensorfold` | TensorFold's opt-in native Vulkan server ([ashhart/TensorFold PR 1544](https://github.com/ashhart/TensorFold/pull/1544), head `b3ab9ba2`), from the `tensorfold-vulkan` package | TensorFold v1.1.9's native Metal server (`tensorfold-1.1.9-macos-arm64.tar.gz`), downloaded and hash-checked by the script |

TensorFold's Vulkan engine serves exactly one checkpoint, Qwen3.5-2B MLX 4-bit, so the TensorFold legs, and an MLX leg that is to be compared with them, use `--model qwen3.5-2b`. The default MLX run keeps the original `qwen3-1.7b` work, so earlier results stay comparable.

## What is fixed

| Item | Value |
|---|---|
| Model `qwen3-1.7b` (MLX default) | `mlx-community/Qwen3-1.7B-4bit`, revision `3b1b1768f8f8cf8351c712464f906e86c2b8269e` (about 1 GB of weights); `model.safetensors` sha256 `0e86d967…9995f`, `tokenizer.json` sha256 `aeb13307…2dae4` |
| Model `qwen3.5-2b` (TensorFold default) | `mlx-community/Qwen3.5-2B-MLX-4bit`, revision `93760be4f1f69842a46bc13dbdc0f19e291392a3` (1.7 GB of weights, the checkpoint PR 1544 and TensorFold's Qwen3.5-2B recipe pin); sha256 of `model.safetensors` `713fe7e5…07d5`, `tokenizer.json` `87a7830d…2de4`, `config.json`, `tokenizer_config.json` and `chat_template.jinja` |
| Prompt | exactly 512 token IDs cut from a fixed English text in the script; the SHA-256 of the IDs is recorded. MLX tokenizes with the checkpoint's tokenizer, TensorFold with its own (`/v1/tokenize`); the IDs then go to the server as IDs (`/v1/completions`) |
| Generation | exactly 128 tokens, greedy; MLX masks every EOS token, TensorFold is asked to `ignore_eos`, so neither can stop early |
| Repeats | 1 warmup run, then 5 measured runs; median, minimum, maximum and spread |
| Metric, MLX | mlx-lm's `prompt_tps` and `generation_tps` (what `mlx_lm.generate --verbose` prints), plus time to first token |
| Metric, TensorFold | the reply's own timings: 512 / `prefill_seconds` (else / `time_to_first_token`), `tokens_per_second` over the decode, and `time_to_first_token` from the request's arrival |
| TensorFold server | `serve MODEL --backend vulkan` (Linux) or `--backend mlx` (macOS, which selects Metal), `--context 2048 --parallel 1 --no-drafts --no-thinking`, loopback only; no prompt cache |
| Timeout | 900 s (MLX) or 2,400 s (TensorFold: the unbatched Vulkan engine decoded 2.8 tok/s on an M1 Pro) for the whole measurement |

A run that processes another prompt length, generates another token count, or whose five greedy runs disagree on the generated IDs is refused (exit 1) instead of printing a rate. A TensorFold run is also refused if it reused cached prompt tokens, drafted tokens, or returned token IDs that do not match its count or its own `token_sha256`. `--compare` refuses results whose model hashes or prompt-ID hashes differ.

## What is recorded

Every result JSON carries, besides the numbers:

- the script's git commit and SHA-256, and the command line;
- Linux: kernel release and `/proc/version`, device-tree model and compatible, the installed `linux-aurora`, `m1n1-aurora`, `mesa-aurora`, `vulkan-aurora`, `omarchy-mlx` and `omarchy-mlx-vulkan*` versions, the source commits those versions pin (from `/usr/share/omarchy-mlx-m4/stack.json`, installed by the `omarchy-mlx-m4-bench` package, only when the installed version matches), the `asahi` module parameters, the CPU governor and thermal-zone temperatures;
- macOS: `sw_vers`, model, chip, memory, the GPU name and core count from `system_profiler`, the power mode;
- the GPU as MLX sees it (`mx.device_info()`: device name, driver, driverInfo, ICD path on Linux), and the `vulkaninfo --summary` and `eglinfo -B`/`glxinfo -B` renderer lines when those tools are installed;
- mlx, mlx-omarchy, mlx-lm and transformers versions;
- TensorFold: the server command, its `/health` (release, backend, device name on macOS, peak memory), the last 40 lines of its log, and the binary's identity. On Linux that is the `tensorfold-vulkan` package's `/usr/share/tensorfold-vulkan/build.json` (commit, Zig, options, server SHA-256, which the script re-checks) and `vulkaninfo --summary` through the same single driver manifest the server is given. On macOS it is the release version and binary SHA-256. A binary named with `--tensorfold-bin` is recorded as `pinned: false` unless it is the pinned release binary;
- load average at start and end, and the MLX/Vulkan/Mesa environment variables.

If a commit is not known from the stack manifest, set it by hand, for example `M4_BENCH_LINUX_AURORA_COMMIT=fa26b0f92ef6…`.

## Run on Linux

With the packages installed (`omarchy-mlx`, `omarchy-mlx-vulkan-aurora`, `omarchy-mlx-m4-bench`):

```bash
omarchy-mlx-m4-bench --out m4-linux.json
```

From a checkout instead, with the packaged venv:

```bash
/usr/lib/omarchy-mlx/venv/bin/python scripts/m4_mini_bench.py --out m4-linux.json
```

The first run downloads the pinned snapshot through `huggingface_hub`; pass `--model-dir DIR` to use a snapshot already on disk (it is hash-checked either way). The run holds `/tmp/m1-gpu.lock`, the per-host GPU lock the other hardware tools use, and waits up to 600 s for it.

On the G16 kernel a GPU fault can stop the GPU until the next boot. Run nothing else on the GPU during the measurement, and expect exit code 4 with a partial record if the child does not finish in time.

### TensorFold on Linux

Install `tensorfold-vulkan` (techmore/omarchy-pkgs `elementaryos-mseries-mlx`; it depends on `vulkan-aurora`). Before the first model run on a new machine, run upstream's operator gates and kernel contract on the GPU, then the benchmark. `omarchy-mlx-m4-bench` has the TensorFold leg once its package pins this script at or after the commit that added it; until then run `/usr/lib/omarchy-mlx/venv/bin/python scripts/m4_mini_bench.py` from a checkout with the same arguments.

```bash
tensorfold-vulkan-gates
omarchy-mlx-m4-bench --engine tensorfold --out m4-linux-tensorfold.json
omarchy-mlx-m4-bench --model qwen3.5-2b --out m4-linux-mlx-2b.json
```

`tensorfold-vulkan` runs the server with `VK_DRIVER_FILES` naming only the vulkan-aurora manifest, for that process alone. The engine checks no Vulkan driver ID, so Aurora's Borealis ID needs nothing more. The script refuses an installed package that is not built from the pinned commit `b3ab9ba2…`, or whose server no longer matches the SHA-256 the package recorded.

## Run on macOS

Use the same mlx-lm and transformers as the Linux venv, and the MLX release that mlx-omarchy is built from (`mlx.lock`):

```bash
python3.14 -m venv ~/.venvs/m4-mini-bench
~/.venvs/m4-mini-bench/bin/python -m pip install mlx==0.32.3 mlx-lm==0.31.3 transformers==5.16.1 huggingface_hub
~/.venvs/m4-mini-bench/bin/python scripts/m4_mini_bench.py --out m4-macos.json
```

TensorFold's Metal leg needs no mlx. The first run downloads `tensorfold-1.1.9-macos-arm64.tar.gz` (sha256 `f8330102…a9d7`) into `~/.cache/m4-mini-bench` and checks the extracted `tensorfold-native` (sha256 `4b4b5a5a…b8e4`); `--tensorfold-bin` runs another build, recorded as unpinned. v1.1.9 is the last release PR 1544's head contains. TensorFold's recipe lists the M1–M4 Metal path for this model as untested upstream, so check that the generated text is sensible before quoting the number.

```bash
~/.venvs/m4-mini-bench/bin/python scripts/m4_mini_bench.py --engine tensorfold --out m4-macos-tensorfold.json
~/.venvs/m4-mini-bench/bin/python scripts/m4_mini_bench.py --model qwen3.5-2b --out m4-macos-mlx-2b.json
```

## Compare

```bash
python3 scripts/m4_mini_bench.py --compare m4-linux.json m4-macos.json
python3 scripts/m4_mini_bench.py --compare m4-linux-mlx-2b.json m4-linux-tensorfold.json m4-macos-mlx-2b.json m4-macos-tensorfold.json
```

It takes two or more results of the same work and uses the last as the reference. For each result it prints the prompt and generation medians, the time to first token, both rates as a share of the reference, and whether its generated token IDs are identical to the reference's; then every renderer. Different engines are expected to differ in token IDs (TensorFold does not claim token parity with MLX); the prompt IDs must match, or the comparison is refused. Quote a ratio only when the runs started quiet (load average at or below 0.5, recorded as `host.quiet`).

## Exit codes

| Code | Meaning |
|---|---|
| 0 | measured |
| 1 | the run did not do the fixed work, or MLX did not run on the GPU |
| 2 | bad invocation, model or TensorFold download or hash failure, an unpinned `tensorfold-vulkan` package, or the GPU lock stayed busy |
| 3 | the measurement child or the TensorFold server failed (its stderr or log tail is in the result) |
| 4 | the measurement timed out |

`python3 scripts/m4_mini_bench.py --self-test` checks the fixed-work rules without a GPU or mlx. `python3 -m unittest tests.test_m4_mini_bench` also runs the TensorFold leg against a fake server and checks the pinned-binary rules.
