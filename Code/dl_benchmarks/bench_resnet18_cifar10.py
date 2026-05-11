"""
Benchmark the first-order custom PyTorch optimizers on ResNet-18 / CIFAR-10.

ResNet-18 has ~11 M parameters, so Hessian-based methods (Newton family,
DFP, BFGS, GaussNewtonCG with explicit Hessian, dense L-BFGS history with
full inverse Hessian storage) are not tractable.  We benchmark only methods
that scale to deep networks:

    GD, SGD, MiniBatchSGD, MomentumSGD, Nesterov, AdaGrad, RMSProp, Adam,
    L-BFGS, StochasticLBFGS, MiniBatchLBFGS, HessianFree (HVP via FD)

Run:
    cd Code
    python -m dl_benchmarks.bench_resnet18_cifar10
"""

from __future__ import annotations

import os

# GPU configuration -- must be set before torch is imported.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

import argparse
import json
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from dl_benchmarks.runner import benchmark_suite  # noqa: E402


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(seed: int = 0):
    """Standard torchvision ResNet-18 adapted to 32x32 CIFAR inputs."""
    from torchvision.models import resnet18

    torch.manual_seed(seed)
    model = resnet18(num_classes=10, weights=None)
    # Adapt the first conv to CIFAR (3 -> 64, 3x3, stride 1) and drop maxpool
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_loaders(batch_size: int, cache_dir: str | None = None, num_workers: int = 4):
    """Load CIFAR-10 from HuggingFace ``datasets``."""
    from datasets import load_dataset
    from torchvision import transforms

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_tfm = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    raw = load_dataset("uoft-cs/cifar10", cache_dir=cache_dir)

    # HF cifar10 uses keys "img" / "label"
    img_key = "img" if "img" in raw["train"].column_names else "image"
    label_key = "label" if "label" in raw["train"].column_names else "fine_label"

    def make_collate(tfm):
        def collate(batch):
            xs = torch.stack([tfm(item[img_key].convert("RGB")) for item in batch])
            ys = torch.tensor([item[label_key] for item in batch], dtype=torch.long)
            return xs, ys
        return collate

    train_loader = DataLoader(
        raw["train"], batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        collate_fn=make_collate(train_tfm),
    )
    test_loader = DataLoader(
        raw["test"], batch_size=512, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=make_collate(test_tfm),
    )
    return train_loader, test_loader


def make_eval_fn(test_loader, device):
    def eval_fn(model):
        correct, total, loss_sum = 0, 0, 0.0
        for x, y in test_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out = model(x)
            loss_sum += float(F.cross_entropy(out, y, reduction="sum").item())
            pred = out.argmax(1)
            correct += int((pred == y).sum().item())
            total += y.numel()
        return {"test_loss": loss_sum / total, "test_acc": correct / total}
    return eval_fn


# ---------------------------------------------------------------------------
# Suite -- only scalable methods
# ---------------------------------------------------------------------------

def build_suite(n_samples: int, batch_size: int):
    return [
        ("GD",              dict(lr=1e-2)),
        ("SGD",             dict(lr=1e-2)),
        ("MiniBatchSGD",    dict(lr=1e-2, batch_size=batch_size, n_samples=n_samples)),
        ("MomentumSGD",     dict(lr=1e-2, momentum=0.9)),
        ("Nesterov",        dict(lr=1e-2, momentum=0.9)),
        ("AdaGrad",         dict(lr=1e-2)),
        ("RMSProp",         dict(lr=1e-3)),
        ("Adam",            dict(lr=1e-3)),
        # L-BFGS family scales (only m * n memory) but is slow per step.
        ("LBFGS",           dict(lr=1e-2, history_size=10)),
        ("StochasticLBFGS", dict(lr=1e-2, history_size=10,
                                  batch_size=batch_size, n_samples=n_samples)),
        ("MiniBatchLBFGS",  dict(lr=1e-2, history_size=10,
                                  batch_size=batch_size, curvature_batch_size=4 * batch_size,
                                  n_samples=n_samples)),
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ResNet-18 / CIFAR-10 optimizer benchmark")
    parser.add_argument("--cache-dir", default=None, type=str,
                        help="HuggingFace datasets cache directory")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--max-steps", default=2000, type=int,
                        help="Number of optimization steps per optimizer")
    parser.add_argument("--log-every", default=50, type=int)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save", default="dl_benchmarks/results_resnet18_cifar10.json")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Run only this subset of optimizer short names")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")
    if device.type != "cuda":
        print("WARNING: ResNet-18 training on CPU will be very slow.")

    train_loader, test_loader = get_loaders(args.batch_size, args.cache_dir)
    eval_fn = make_eval_fn(test_loader, device)
    loss_fn = nn.CrossEntropyLoss()
    n_samples = len(train_loader.dataset)

    suite = build_suite(n_samples=n_samples, batch_size=args.batch_size)
    if args.only:
        suite = [(n, kw) for n, kw in suite if n in args.only]
        if not suite:
            raise SystemExit(f"--only filtered out all suites. Valid names: "
                             f"{[s[0] for s in build_suite(n_samples, args.batch_size)]}")

    multi = benchmark_suite(
        suite,
        build_model=build_model,
        train_loader=train_loader,
        loss_fn=loss_fn,
        max_steps=args.max_steps,
        device=device,
        eval_fn=eval_fn,
        log_every=args.log_every,
        grad_clip=5.0,
    )

    print("\n=== Final comparison ===")
    multi.print_comparison()

    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    payload = {
        "device": str(device),
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "runs": {
            name: {
                "summary": multi.get_run(name).summary(),
                "history": multi.get_run(name).history,
            }
            for name in multi.run_names
        },
    }
    with open(args.save, "w") as f:
        json.dump(payload, f, default=float, indent=2)
    print(f"Saved {args.save}")


if __name__ == "__main__":
    main()
