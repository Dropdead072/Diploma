"""
Shared training/benchmarking utility for evaluating the custom PyTorch
optimizers from ``utils.optimizers.torch_optimizers`` on real deep-learning
models.

The runner deliberately works in two modes:

1. ``train_first_order``  -- standard PyTorch loop (closure-based)
   suitable for GD, SGD, Mini-batch SGD, MomentumSGD, Nesterov, AdaGrad,
   RMSProp, Adam, Stochastic L-BFGS and Mini-Batch L-BFGS.

2. ``train_hessian_based`` -- functorch / torch.func loop that materialises
   the full Hessian (or Jacobian) and passes it to ``set_hessian`` before
   each step.  Suitable only for *small* models (e.g. the MLP) because the
   Hessian is dense ``n_params x n_params``.

Per-step metrics (loss, grad norm, wall-clock time, peak GPU/CPU memory)
are recorded via :class:`utils.logging.optimizer_logger.OptimizationLogger`
so the existing reporting infrastructure works unchanged.
"""

from __future__ import annotations

import gc
import math
import os
import sys
import time
import tracemalloc
from contextlib import contextmanager
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

# Make `utils.*` and `core.*` importable when running from Code/ directly.
_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from utils.logging.optimizer_logger import OptimizationLogger, MultiRunLogger  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def grad_norm(params: Iterable[nn.Parameter]) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum().item())
    return math.sqrt(total)


@contextmanager
def peak_memory_tracker(device: torch.device):
    """Yield a getter that returns peak memory in KB for either CUDA or CPU."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        yield lambda: torch.cuda.max_memory_allocated(device) / 1024.0
    else:
        tracemalloc.start()
        try:
            yield lambda: tracemalloc.get_traced_memory()[1] / 1024.0
        finally:
            tracemalloc.stop()


def cycle(loader: DataLoader):
    """Infinite generator over a DataLoader."""
    while True:
        for batch in loader:
            yield batch


# ---------------------------------------------------------------------------
# First-order training loop
# ---------------------------------------------------------------------------

def train_first_order(
    model: nn.Module,
    train_loader: DataLoader,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    optimizer: torch.optim.Optimizer,
    *,
    max_steps: int,
    device: torch.device,
    logger: OptimizationLogger,
    log_every: int = 1,
    eval_fn: Optional[Callable[[nn.Module], Dict[str, float]]] = None,
    grad_clip: Optional[float] = None,
    needs_closure: bool = False,
) -> None:
    """
    Generic mini-batch training loop.

    ``needs_closure=True`` is required by L-BFGS-style optimizers that
    re-evaluate the loss at the lookahead point.  When True the loss is
    computed inside a closure passed to ``optimizer.step(closure)``.
    """
    model.train()
    data_iter = cycle(train_loader)

    with peak_memory_tracker(device) as get_peak_kb:
        for step in range(max_steps):
            batch = next(data_iter)
            inputs, targets = _move_batch(batch, device)

            if needs_closure:
                last_loss: Dict[str, Tensor] = {}

                def closure():
                    optimizer.zero_grad(set_to_none=True)
                    out = model(inputs)
                    loss = loss_fn(out, targets)
                    loss.backward()
                    last_loss["v"] = loss
                    return loss

                optimizer.step(closure)
                loss = last_loss["v"]
            else:
                optimizer.zero_grad(set_to_none=True)
                out = model(inputs)
                loss = loss_fn(out, targets)
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            if step % log_every == 0 or step == max_steps - 1:
                metrics = {
                    "loss": float(loss.detach().item()),
                    "grad_norm": grad_norm(model.parameters()),
                }
                if eval_fn is not None and (step % (log_every * 10) == 0 or step == max_steps - 1):
                    model.eval()
                    with torch.no_grad():
                        metrics.update(eval_fn(model))
                    model.train()
                logger.log(step, metrics)

        logger.heap_peak_kb = float(get_peak_kb())


def _move_batch(batch, device: torch.device) -> Tuple[Tensor, Tensor]:
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        x, y = batch
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    if isinstance(batch, dict):
        # Already moved language-modelling batches will be handled separately.
        return batch, None  # type: ignore[return-value]
    raise ValueError(f"Unsupported batch type: {type(batch)}")


# ---------------------------------------------------------------------------
# Second-order training loop (full Hessian)
# ---------------------------------------------------------------------------

def train_hessian_based(
    model: nn.Module,
    train_loader: DataLoader,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    optimizer,
    *,
    max_steps: int,
    device: torch.device,
    logger: OptimizationLogger,
    log_every: int = 1,
    eval_fn: Optional[Callable[[nn.Module], Dict[str, float]]] = None,
) -> None:
    """
    Training loop for optimizers that need the full Hessian
    (NewtonsMethodTorch, DiagonalNewtonTorch, BlockNewtonTorch,
    GaussNewtonCGTorch, HessianFreeTorch).

    The Hessian is computed via ``torch.autograd.functional.hessian`` of the
    loss w.r.t. the flattened parameter vector.  Cost is O(n_params^2) memory
    so this is only feasible for the small MLP benchmark.
    """
    from torch.nn.utils import parameters_to_vector, vector_to_parameters

    params = [p for p in model.parameters() if p.requires_grad]
    model.train()
    data_iter = cycle(train_loader)

    def flat_params() -> Tensor:
        return parameters_to_vector(params).detach()

    def set_flat(p_flat: Tensor) -> None:
        vector_to_parameters(p_flat, params)

    with peak_memory_tracker(device) as get_peak_kb:
        for step in range(max_steps):
            inputs, targets = _move_batch(next(data_iter), device)
            x0 = flat_params()

            def loss_of_params(p_flat: Tensor) -> Tensor:
                vector_to_parameters(p_flat, params)
                out = model(inputs)
                return loss_fn(out, targets)

            # Gradient w.r.t. the flat parameter vector.
            x0_req = x0.detach().clone().requires_grad_(True)
            loss = loss_of_params(x0_req)
            (grad_flat,) = torch.autograd.grad(loss, x0_req, create_graph=False)

            # Push gradient back into the parameter tensors so optimizer.step()
            # sees it via p.grad.
            set_flat(x0)
            offset = 0
            for p in params:
                n = p.numel()
                g_p = grad_flat[offset : offset + n].reshape(p.shape).detach().clone()
                p.grad = g_p
                offset += n

            # Compute full Hessian
            H = torch.autograd.functional.hessian(loss_of_params, x0_req.detach())
            # Restore parameters (hessian() may have perturbed them)
            set_flat(x0)
            if hasattr(optimizer, "set_hessian"):
                optimizer.set_hessian(H.detach())

            # For HessianFree / GN-CG we can also pass a grad_fn for HVP fallback
            if hasattr(optimizer, "set_grad_fn"):
                def grad_fn(p_flat: Tensor) -> Tensor:
                    p_req = p_flat.detach().clone().requires_grad_(True)
                    L = loss_of_params(p_req)
                    (g,) = torch.autograd.grad(L, p_req)
                    set_flat(x0)
                    return g.detach()
                optimizer.set_grad_fn(grad_fn)

            optimizer.step()

            with torch.no_grad():
                cur_loss = float(loss.detach().item())
                metrics = {
                    "loss": cur_loss,
                    "grad_norm": float(grad_flat.norm().item()),
                }
                if eval_fn is not None and (step % (log_every * 10) == 0 or step == max_steps - 1):
                    model.eval()
                    metrics.update(eval_fn(model))
                    model.train()
                if step % log_every == 0 or step == max_steps - 1:
                    logger.log(step, metrics)

        logger.heap_peak_kb = float(get_peak_kb())


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------

def build_optimizer(name: str, params: Iterable[nn.Parameter], **kwargs):
    """
    Instantiate a custom optimizer by short name.
    See utils.optimizers.torch_optimizers.ALL_TORCH_OPTIMIZERS for the mapping.
    """
    from utils.optimizers.torch_optimizers import ALL_TORCH_OPTIMIZERS

    if name not in ALL_TORCH_OPTIMIZERS:
        raise KeyError(
            f"Unknown optimizer '{name}'. Available: {list(ALL_TORCH_OPTIMIZERS)}"
        )
    return ALL_TORCH_OPTIMIZERS[name](list(params), **kwargs)


HESSIAN_BASED = {"Newton", "DiagonalNewton", "BlockNewton", "GaussNewtonCG", "HessianFree"}
LBFGS_FAMILY = {"LBFGS", "StochasticLBFGS", "MiniBatchLBFGS"}


def is_hessian_based(name: str) -> bool:
    return name in HESSIAN_BASED


def needs_closure_flag(name: str) -> bool:
    """Whether the optimizer benefits from a re-evaluation closure."""
    return False  # our L-BFGS variants use the already-computed gradient


# ---------------------------------------------------------------------------
# High-level benchmark driver
# ---------------------------------------------------------------------------

def benchmark_optimizer(
    *,
    optimizer_name: str,
    optimizer_kwargs: Dict,
    build_model: Callable[[], nn.Module],
    train_loader: DataLoader,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    max_steps: int,
    device: torch.device,
    eval_fn: Optional[Callable[[nn.Module], Dict[str, float]]] = None,
    log_every: int = 1,
    grad_clip: Optional[float] = None,
    force_first_order: bool = False,
) -> OptimizationLogger:
    """
    Run a single optimizer end-to-end and return its filled-in logger.

    Fresh model is built per run so each optimizer starts from the same
    initialisation (as long as ``build_model`` seeds itself deterministically).
    """
    model = build_model().to(device)
    optimizer = build_optimizer(optimizer_name, model.parameters(), **optimizer_kwargs)
    logger = OptimizationLogger(optimizer_name=optimizer_name)

    if is_hessian_based(optimizer_name) and not force_first_order:
        train_hessian_based(
            model, train_loader, loss_fn, optimizer,
            max_steps=max_steps, device=device, logger=logger,
            log_every=log_every, eval_fn=eval_fn,
        )
    else:
        train_first_order(
            model, train_loader, loss_fn, optimizer,
            max_steps=max_steps, device=device, logger=logger,
            log_every=log_every, eval_fn=eval_fn, grad_clip=grad_clip,
            needs_closure=needs_closure_flag(optimizer_name),
        )

    # Free GPU memory between runs
    del model, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return logger


def benchmark_suite(
    suite: List[Tuple[str, Dict]],
    *,
    build_model: Callable[[], nn.Module],
    train_loader: DataLoader,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    max_steps: int,
    device: torch.device,
    eval_fn: Optional[Callable[[nn.Module], Dict[str, float]]] = None,
    log_every: int = 1,
    grad_clip: Optional[float] = None,
) -> MultiRunLogger:
    """Run a list of (optimizer_name, kwargs) configurations and aggregate logs."""
    multi = MultiRunLogger()
    for name, kwargs in suite:
        print(f"\n=== Training with {name} (kwargs={kwargs}) ===")
        t0 = time.time()
        logger = benchmark_optimizer(
            optimizer_name=name,
            optimizer_kwargs=kwargs,
            build_model=build_model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            max_steps=max_steps,
            device=device,
            eval_fn=eval_fn,
            log_every=log_every,
            grad_clip=grad_clip,
        )
        multi._runs[name] = logger  # type: ignore[attr-defined]
        print(f"   -> done in {time.time() - t0:.1f}s | {logger}")
    return multi
