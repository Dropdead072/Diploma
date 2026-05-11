# DL Benchmarks for Custom Optimizers

End-to-end deep-learning benchmarks for the 18 custom PyTorch optimizers in
[`utils/optimizers/torch_optimizers.py`](../utils/optimizers/torch_optimizers.py).

Each script:
- Pins the GPU (`CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=2`) **before** importing torch.
- Loads data directly from HuggingFace [`datasets`](https://huggingface.co/docs/datasets) — no `torchvision.datasets` downloads.
- Builds the model with a deterministic seed (every optimizer starts from the same init).
- Trains for a configurable number of optimization steps.
- Logs per-step loss, gradient norm, wall-clock time, and peak memory (CUDA / CPU).
- Saves a JSON file with the full history and a summary comparison table.

> The default `CUDA_VISIBLE_DEVICES=2` is set via `os.environ.setdefault(...)`, so you can override it on the command line:
> `CUDA_VISIBLE_DEVICES=0 python -m dl_benchmarks.bench_resnet18_cifar10`.

All scripts share the same runner in [`runner.py`](runner.py).

## 1. MLP / Fashion-MNIST — [`bench_mlp_fashion_mnist.py`](bench_mlp_fashion_mnist.py)

A tiny MLP (28*28 → 64 → 32 → 10, ~55 k params) — small enough for the
**Hessian-based** methods (Newton, DiagonalNewton, BlockNewton, GaussNewton-CG,
HessianFree) to remain tractable. Runs all 18 optimizers by default.

```bash
cd Code
python -m dl_benchmarks.bench_mlp_fashion_mnist --max-steps 500 --batch-size 128
# Drop the expensive 2nd-order methods:
python -m dl_benchmarks.bench_mlp_fashion_mnist --no-hessian --max-steps 1000
```

Output: `dl_benchmarks/results_mlp_fashion_mnist.json`

## 2. ResNet-18 / CIFAR-10 — [`bench_resnet18_cifar10.py`](bench_resnet18_cifar10.py)

ResNet-18 (~11 M params, first-conv adapted for 32×32 input). Hessian-based
methods are skipped (memory infeasible). Data: `uoft-cs/cifar10` (HF). The suite includes:

`GD, SGD, MiniBatchSGD, MomentumSGD, Nesterov, AdaGrad, RMSProp, Adam,
LBFGS, StochasticLBFGS, MiniBatchLBFGS`.

```bash
cd Code
python -m dl_benchmarks.bench_resnet18_cifar10 --max-steps 2000 --batch-size 128
# Single optimizer only:
python -m dl_benchmarks.bench_resnet18_cifar10 --only Adam --max-steps 5000
```

Output: `dl_benchmarks/results_resnet18_cifar10.json`

## 3. Qwen/Qwen2.5-0.5B / Russian WikiText — [`bench_qwen_wikitext_ru.py`](bench_qwen_wikitext_ru.py)

**SFT-only** (next-token cross-entropy) fine-tuning of `Qwen/Qwen2.5-0.5B`
on a Russian Wikipedia slice loaded via HuggingFace `datasets`
(default: `wikimedia/wikipedia`, config `20231101.ru`). Any HF text dataset
can be substituted with `--hf-dataset / --hf-config / --text-column`, or a
local text file via `--data-file`.

**Prints a generation sample from the *untrained* model before training** so
you can compare it against post-training generations.

```bash
cd Code
# Default: stream 500 Russian Wikipedia paragraphs
python -m dl_benchmarks.bench_qwen_wikitext_ru --max-steps 200 --batch-size 2

# Quick smoke test with one optimizer only:
python -m dl_benchmarks.bench_qwen_wikitext_ru --only Adam --max-steps 50

# Pick a different HF dataset (e.g., wikitext-2 raw English baseline):
python -m dl_benchmarks.bench_qwen_wikitext_ru \
    --hf-dataset Salesforce/wikitext --hf-config wikitext-2-raw-v1

# Use your own corpus (one paragraph per line):
python -m dl_benchmarks.bench_qwen_wikitext_ru --data-file path/to/ru_wiki.txt

# Skip the untrained generation sample:
python -m dl_benchmarks.bench_qwen_wikitext_ru --skip-untrained-demo
```

### Requirements

```bash
pip install torch torchvision transformers datasets
```

### Suite (Qwen)

Hessian-based methods are excluded (Qwen2.5-0.5B has ~500 M parameters).
DFP/BFGS are also excluded (dense `n×n` inverse Hessian storage).
L-BFGS-style methods are included for reference but expect them to be slow.

```
SGD, MomentumSGD, Nesterov, AdaGrad, RMSProp, Adam, StochasticLBFGS
```

Output: `dl_benchmarks/results_qwen_wikitext_ru.json`

---

## Optimizer scalability summary

| Method                    | MLP (55k) | ResNet-18 (11M) | Qwen2.5-0.5B (500M) |
|---------------------------|:---------:|:---------------:|:-------------------:|
| GD, SGD, MiniBatchSGD     | ✔︎         | ✔︎               | ✔︎ (SGD)            |
| MomentumSGD, Nesterov     | ✔︎         | ✔︎               | ✔︎                  |
| AdaGrad, RMSProp, Adam    | ✔︎         | ✔︎               | ✔︎                  |
| Newton, DiagonalNewton    | ✔︎         | ✗ (n² mem)      | ✗                   |
| BlockNewton, GN-CG, HF    | ✔︎         | ✗               | ✗                   |
| DFP, BFGS                 | ✔︎         | ✗ (n² mem)      | ✗                   |
| L-BFGS / Stoch / Mini     | ✔︎         | ✔︎ (slow)        | ✔︎ (slow)           |

## JSON output schema

```json
{
  "device": "cuda",
  "max_steps": 500,
  "batch_size": 128,
  "runs": {
    "Adam": {
      "summary": {
        "optimizer": "Adam",
        "steps": 50,
        "best_loss": 0.31,
        "final_loss": 0.31,
        "total_time_s": 12.4,
        "heap_peak_kb": 8123.4
      },
      "history": [
        {"step": 0, "elapsed": 0.0, "loss": 2.30, "grad_norm": 6.1, ...},
        ...
      ]
    },
    "SGD": { ... }
  }
}
```
