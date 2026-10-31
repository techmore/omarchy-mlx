# Mac mini M4 benchmark (release gate)

`scripts/m4_mini_bench.py` measures one fixed piece of language-model work with MLX on the Mac mini M4 (J773g, T8132, 10-core G16G GPU), once under Linux (Aurora kernel, Aurora Honeykrisp Vulkan, mlx-omarchy) and once under macOS (Metal MLX) on the same machine, and compares the two. It is the speed measurement for the M4 mini release gate. It does not certify M4 GPU compute; correctness stays with the existing gates.

## What is fixed

| Item | Value |
|---|---|
| Model | `mlx-community/Qwen3-1.7B-4bit`, revision `3b1b1768f8f8cf8351c712464f906e86c2b8269e` (about 1 GB of weights) |
| Checked files | `model.safetensors` sha256 `0e86d967…9995f`, `tokenizer.json` sha256 `aeb13307…2dae4` |
| Prompt | exactly 512 token IDs cut from a fixed English text in the script; the SHA-256 of the IDs is recorded |
| Generation | exactly 128 tokens, greedy, every EOS token masked so the model cannot stop early |
| Repeats | 1 warmup run, then 5 measured runs; median, minimum, maximum and spread |
| Metric | mlx-lm's `prompt_tps` and `generation_tps` (what `mlx_lm.generate --verbose` prints), plus time to first token |
| Timeout | 900 s for the whole measurement, in a child process |

A run that processes another prompt length, generates another token count, or whose five greedy runs disagree on the generated IDs is refused (exit 1) instead of printing a rate. `--compare` refuses two results whose model hashes or prompt-ID hashes differ.

## What is recorded

Every result JSON carries, besides the numbers:

- the script's git commit and SHA-256, and the command line;
- Linux: kernel release and `/proc/version`, device-tree model and compatible, the installed `linux-aurora`, `m1n1-aurora`, `mesa-aurora`, `vulkan-aurora`, `omarchy-mlx` and `omarchy-mlx-vulkan*` versions, the source commits those versions pin (from `/usr/share/omarchy-mlx-m4/stack.json`, installed by the `omarchy-mlx-m4-bench` package, only when the installed version matches), the `asahi` module parameters, the CPU governor and thermal-zone temperatures;
- macOS: `sw_vers`, model, chip, memory, the GPU name and core count from `system_profiler`, the power mode;
- the GPU as MLX sees it (`mx.device_info()`: device name, driver, driverInfo, ICD path on Linux), and the `vulkaninfo --summary` and `eglinfo -B`/`glxinfo -B` renderer lines when those tools are installed;
- mlx, mlx-omarchy, mlx-lm and transformers versions;
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

## Run on macOS

Use the same mlx-lm and transformers as the Linux venv, and the MLX release that mlx-omarchy is built from (`mlx.lock`):

```bash
python3.14 -m venv ~/.venvs/m4-mini-bench
~/.venvs/m4-mini-bench/bin/python -m pip install mlx==0.32.3 mlx-lm==0.31.3 transformers==5.16.1 huggingface_hub
~/.venvs/m4-mini-bench/bin/python scripts/m4_mini_bench.py --out m4-macos.json
```

## Compare

```bash
python3 scripts/m4_mini_bench.py --compare m4-linux.json m4-macos.json
```

It prints both medians and Linux as a share of macOS, whether the two sides generated identical token IDs, and both renderers. Quote a ratio only when both runs started quiet (load average at or below 0.5, recorded as `host.quiet`).

## Exit codes

| Code | Meaning |
|---|---|
| 0 | measured |
| 1 | the run did not do the fixed work, or MLX did not run on the GPU |
| 2 | bad invocation, model download or hash failure, or the GPU lock stayed busy |
| 3 | the measurement child failed (its stderr tail is in the result) |
| 4 | the measurement child timed out |

`python3 scripts/m4_mini_bench.py --self-test` checks the fixed-work rules without a GPU or mlx.
