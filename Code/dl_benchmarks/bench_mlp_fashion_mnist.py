"""
Benchmark all 18 custom PyTorch optimizers on a small MLP for Fashion-MNIST.

The MLP is intentionally small (28*28 -> 64 -> 32 -> 10, ~55k parameters)
so that Hessian-based methods (Newton, DiagonalNewton, BlockNewton,
GaussNewton-CG, HessianFree) remain tractable.

Run:
    cd Code
    python -m dl_benchmarks.bench_mlp_fashion_mnist
"""

from __future__ import annotations

import argparse
import os
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

class TinyMLP(nn.Module):
    """28*28 -> 64 -> 32 -> 10 fully-connected classifier (~55 k params)."""

    def __init__(self, hidden1: int = 64, hidden2: int = 32, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x):  # noqa: D401
        return self.net(x)


def build_model(seed: int = 0):
    torch.manual_seed(seed)
    return TinyMLP()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_loaders(batch_size: int, data_root: str, num_workers: int = 2):
    from torchvision import datasets, transforms

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    train_ds = datasets.FashionMNIST(data_root, train=True, download=True, transform=tfm)
    test_ds = datasets.FashionMNIST(data_root, train=False, download=True, transform=tfm)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=512, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader


def make_eval_fn(test_loader, device):
    def eval_fn(model):
        correct, total, loss_sum = 0, 0, 0.0
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += float(F.cross_entropy(out, y, reduction="sum").item())
            pred = out.argmax(1)
            correct += int((pred == y).sum().item())
            total += y.numel()
        return {"test_loss": loss_sum / total, "test_acc": correct / total}
    return eval_fn


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------

def build_suite(n_samples: int, include_hessian: bool):
    """Full suite of all 18 optimizers."""
    suite = [
        ("GD",              dict(lr=1e-2)),
        ("SGD",             dict(lr=1e-2)),
        ("MiniBatchSGD",    dict(lr=1e-2, batch_size=64, n_samples=n_samples)),
        ("MomentumSGD",     dict(lr=1e-2, momentum=0.9)),
        ("Nesterov",        dict(lr=1e-2, momentum=0.9)),
        ("AdaGrad",         dict(lr=1e-2)),
        ("RMSProp",         dict(lr=1e-3)),
        ("Adam",            dict(lr=1e-3)),
        ("DFP",             dict(lr=1e-2)),
        ("BFGS",            dict(lr=1e-2)),
        ("LBFGS",           dict(lr=1e-2, history_size=10)),
        ("StochasticLBFGS", dict(lr=1e-2, history_size=10,
                                  batch_size=64, n_samples=n_samples)),
        ("MiniBatchLBFGS",  dict(lr=1e-2, history_size=10,
                                  batch_size=64, curvature_batch_size=256,
                                  n_samples=n_samples)),
    ]
    if include_hessian:
        suite += [
            ("Newton",         dict(lr=0.5, damping=1e-3)),
            ("DiagonalNewton", dict(lr=0.1, eps=1e-3)),
            ("BlockNewton",    dict(lr=0.5, block_size=64, eps=1e-3)),
            ("GaussNewtonCG",  dict(lr=0.5, cg_iters=20, damping=1e-3)),
            ("HessianFree",    dict(lr=0.5, cg_iters=20, damping=1e-3, fd_eps=1e-4)),
        ]
    return suite


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MLP / Fashion-MNIST optimizer benchmark")
    parser.add_argument("--data-root", default="./data", type=str)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--max-steps", default=500, type=int)
    parser.add_argument("--log-every", default=10, type=int)
    parser.add_argument("--no-hessian", action="store_true",
                        help="Skip the second-order methods (much faster)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--save", default="dl_benchmarks/results_mlp_fashion_mnist.json")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    train_loader, test_loader = get_loaders(args.batch_size, args.data_root)
    eval_fn = make_eval_fn(test_loader, device)
    loss_fn = nn.CrossEntropyLoss()
    n_samples = len(train_loader.dataset)

    suite = build_suite(n_samples=n_samples, include_hessian=not args.no_hessian)

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

    # Persist
    import json
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
