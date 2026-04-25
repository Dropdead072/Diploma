"""
memory_profiler.py
==================
RAM (and optional MPS/CUDA VRAM) usage profiler for both NumPy and PyTorch
optimizers.

Design
------
* Uses ``tracemalloc`` for Python-heap tracking (works for both NumPy and
  PyTorch CPU tensors).
* Uses ``torch.mps.current_allocated_memory()`` / ``torch.cuda.memory_allocated()``
  for device-side VRAM when a GPU backend is active.
* A background thread samples memory every ``sample_interval_s`` seconds so
  that *peak* usage is captured even inside tight inner loops.
* Provides two high-level entry points:

    profile_numpy_optimizer(optimizer, x0, grad_fn, hess_fn, max_iter)
        -> MemoryProfile

    profile_torch_optimizer(optimizer, param_list, closure, max_iter)
        -> MemoryProfile

* ``MemoryProfile`` is a plain dataclass that integrates with the existing
  ``OptimizationLogger`` / ``MultiRunLogger`` infrastructure: call
  ``profile.to_summary_dict()`` and merge it into the logger's summary.

Usage example (NumPy)
---------------------
    from utils.profiling.memory_profiler import profile_numpy_optimizer
    from utils.optimizers.numpy_optimizers_full import Adam

    opt = Adam(lr=1e-3)
    x0  = np.zeros(10)
    prof = profile_numpy_optimizer(opt, x0, grad_fn=my_grad, max_iter=300)
    print(prof)

Usage example (PyTorch)
-----------------------
    from utils.profiling.memory_profiler import profile_torch_optimizer
    from utils.optimizers.torch_optimizers import AdamTorch

    p   = torch.nn.Parameter(torch.zeros(10))
    opt = AdamTorch([p], lr=1e-3)

    def closure():
        if p.grad is not None: p.grad.zero_()
        loss = (p ** 2).sum()
        loss.backward()
        return loss

    prof = profile_torch_optimizer(opt, [p], closure, max_iter=300)
    print(prof)
"""

from __future__ import annotations

import gc
import sys
import threading
import time
import tracemalloc
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Optional torch import
# ---------------------------------------------------------------------------
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Optimizer state size -- the FAIR metric
# ---------------------------------------------------------------------------

def optimizer_state_size_kb(optimizer) -> float:
    """
    Measure the **algorithmic memory footprint** of an optimizer by summing
    the sizes of all numpy arrays and deques-of-arrays stored as instance
    attributes.  This excludes dataset copies, loss temporaries, and logging
    overhead -- it captures only the persistent state that scales with the
    algorithm (momentum buffers, inverse-Hessian approximations, L-BFGS
    curvature pairs, etc.).

    Works for any ``core.optimizer.Optimizer`` subclass (NumPy) and any
    ``torch.optim.Optimizer`` subclass (PyTorch).

    Parameters
    ----------
    optimizer :
        A fitted optimizer instance.

    Returns
    -------
    float
        Total size of optimizer-owned arrays in KB.

    Examples
    --------
    Typical values on a 10-feature problem (n=10 parameters):

    +-----------------------+------------------+---------------------------+
    | Optimizer             | State arrays     | Approx. size              |
    +=======================+==================+===========================+
    | GD / SGD              | none             | ~0 KB                     |
    | Momentum / Nesterov   | h (n,)           | ~0.1 KB                   |
    | AdaGrad / RMSProp     | G or v (n,)      | ~0.1 KB                   |
    | Adam                  | m, v (n,)        | ~0.2 KB                   |
    | DFP / BFGS            | B or H (nxn)     | ~0.8 KB  (n=10)           |
    | L-BFGS (m=10)         | 10x(s,y) (n,)   | ~1.6 KB  (n=10)           |
    | Newton / DiagNewton   | none (stateless) | ~0 KB                     |
    +-----------------------+------------------+---------------------------+
    """
    total_bytes = 0

    def _array_bytes(obj) -> int:
        """Return byte size of a numpy array or torch tensor."""
        if isinstance(obj, np.ndarray):
            return obj.nbytes
        if _TORCH_AVAILABLE and isinstance(obj, torch.Tensor):
            return obj.element_size() * obj.nelement()
        return 0

    def _walk(obj, visited: set) -> int:
        obj_id = id(obj)
        if obj_id in visited:
            return 0
        visited.add(obj_id)

        # Direct array
        b = _array_bytes(obj)
        if b:
            return b

        # Deque / list of arrays (L-BFGS curvature pairs)
        if isinstance(obj, (deque, list)):
            return sum(_walk(item, visited) for item in obj)

        # Dict (torch optimizer state dict)
        if isinstance(obj, dict):
            return sum(_walk(v, visited) for v in obj.values())

        return 0

    visited: set = set()

    # -- NumPy optimizers --------------------------------------------------
    # Walk all instance attributes, skip primitives and callables
    _skip_types = (int, float, str, bool, type(None), type)
    for attr_name, attr_val in vars(optimizer).items():
        if attr_name.startswith('__'):
            continue
        if isinstance(attr_val, _skip_types):
            continue
        total_bytes += _walk(attr_val, visited)

    # -- PyTorch optimizers ------------------------------------------------
    # Also walk self.state (per-parameter state dicts)
    if _TORCH_AVAILABLE and isinstance(optimizer, torch.optim.Optimizer):
        for param_state in optimizer.state.values():
            total_bytes += _walk(param_state, visited)

    return total_bytes / 1024.0


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class MemoryProfile:
    """
    Memory usage profile for a single optimizer run.

    Attributes
    ----------
    optimizer_name : str
        Human-readable label.
    backend : str
        ``"numpy"`` or ``"torch"``.
    n_steps : int
        Number of optimizer steps executed.
    total_time_s : float
        Wall-clock duration of the profiled run.

    -- Python heap (tracemalloc) --
    heap_start_kb : float
        Python heap allocated at the start of the run (KB).
    heap_peak_kb : float
        Peak Python heap allocated during the run (KB).
    heap_end_kb : float
        Python heap allocated at the end of the run (KB).
    heap_delta_kb : float
        Net change in heap allocation (end - start, KB).

    -- Device memory (torch only) --
    device_start_kb : float
        Device memory allocated at the start (KB).  0 for NumPy.
    device_peak_kb : float
        Peak device memory allocated during the run (KB).  0 for NumPy.
    device_end_kb : float
        Device memory allocated at the end (KB).  0 for NumPy.

    -- Per-step samples --
    heap_samples_kb : List[float]
        Heap usage sampled every ``sample_interval_s`` seconds.
    device_samples_kb : List[float]
        Device memory sampled every ``sample_interval_s`` seconds.
    """

    optimizer_name: str = ""
    backend: str = "numpy"
    n_steps: int = 0
    total_time_s: float = 0.0

    # Python heap
    heap_start_kb: float = 0.0
    heap_peak_kb: float = 0.0
    heap_end_kb: float = 0.0
    heap_delta_kb: float = 0.0

    # Device (GPU/MPS)
    device_start_kb: float = 0.0
    device_peak_kb: float = 0.0
    device_end_kb: float = 0.0

    # Time-series samples
    heap_samples_kb: List[float] = field(default_factory=list)
    device_samples_kb: List[float] = field(default_factory=list)

    # ------------------------------------------------------------------
    def to_summary_dict(self) -> Dict[str, Any]:
        """
        Return a flat dict suitable for merging into an
        ``OptimizationLogger.summary()`` dict.
        """
        return {
            "optimizer":        self.optimizer_name,
            "backend":          self.backend,
            "steps":            self.n_steps,
            "total_time_s":     self.total_time_s,
            "heap_start_kb":    round(self.heap_start_kb, 2),
            "heap_peak_kb":     round(self.heap_peak_kb, 2),
            "heap_end_kb":      round(self.heap_end_kb, 2),
            "heap_delta_kb":    round(self.heap_delta_kb, 2),
            "device_start_kb":  round(self.device_start_kb, 2),
            "device_peak_kb":   round(self.device_peak_kb, 2),
            "device_end_kb":    round(self.device_end_kb, 2),
        }

    def __repr__(self) -> str:
        d = self.to_summary_dict()
        lines = [
            f"MemoryProfile -- {d['optimizer']} ({d['backend']})",
            f"  Steps        : {d['steps']}",
            f"  Time         : {d['total_time_s']:.3f} s",
            f"  Heap start   : {d['heap_start_kb']:.1f} KB",
            f"  Heap peak    : {d['heap_peak_kb']:.1f} KB",
            f"  Heap end     : {d['heap_end_kb']:.1f} KB",
            f"  Heap delta   : {d['heap_delta_kb']:+.1f} KB",
        ]
        if self.backend == "torch":
            lines += [
                f"  Device start : {d['device_start_kb']:.1f} KB",
                f"  Device peak  : {d['device_peak_kb']:.1f} KB",
                f"  Device end   : {d['device_end_kb']:.1f} KB",
            ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bytes_to_kb(b: int) -> float:
    return b / 1024.0


def _get_device_bytes() -> int:
    """Return currently allocated device memory in bytes (0 if unavailable)."""
    if not _TORCH_AVAILABLE:
        return 0
    try:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            return torch.mps.current_allocated_memory()
    except Exception:
        pass
    return 0


class _MemorySampler:
    """
    Background thread that samples heap and device memory at a fixed interval.

    Start with ``start()``, stop with ``stop()``.  Results are in
    ``heap_samples_kb`` and ``device_samples_kb``.
    """

    def __init__(self, interval_s: float = 0.01):
        self.interval_s = interval_s
        self.heap_samples_kb: List[float] = []
        self.device_samples_kb: List[float] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            # tracemalloc snapshot
            snapshot = tracemalloc.take_snapshot()
            stats = snapshot.statistics("lineno")
            heap_bytes = sum(s.size for s in stats)
            self.heap_samples_kb.append(_bytes_to_kb(heap_bytes))
            self.device_samples_kb.append(_bytes_to_kb(_get_device_bytes()))
            time.sleep(self.interval_s)


# ---------------------------------------------------------------------------
# Public API -- NumPy
# ---------------------------------------------------------------------------

def profile_numpy_optimizer(
    optimizer,
    x0: np.ndarray,
    grad_fn: Callable[[np.ndarray], np.ndarray],
    hess_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    max_iter: int = 300,
    eps: float = 1e-8,
    sample_interval_s: float = 0.01,
) -> MemoryProfile:
    """
    Profile RAM usage of a single NumPy optimizer run.

    Parameters
    ----------
    optimizer :
        Any object with a ``.step(x, grad_fn, hess_fn)`` method and a
        ``.name`` property (i.e. a subclass of ``core.optimizer.Optimizer``).
    x0 : np.ndarray
        Starting point.
    grad_fn : Callable
        Gradient function ``grad_fn(x) -> np.ndarray``.
    hess_fn : Callable, optional
        Hessian function ``hess_fn(x) -> np.ndarray``.  Pass ``None`` for
        first-order methods.
    max_iter : int
        Maximum number of optimizer steps.
    eps : float
        Gradient-norm convergence tolerance.
    sample_interval_s : float
        Background sampling interval in seconds.

    Returns
    -------
    MemoryProfile
    """
    gc.collect()
    optimizer.reset()

    # Start tracemalloc
    if not tracemalloc.is_tracing():
        tracemalloc.start()
    else:
        tracemalloc.clear_traces()

    heap_start = _bytes_to_kb(tracemalloc.get_traced_memory()[0])
    device_start = _bytes_to_kb(_get_device_bytes())

    sampler = _MemorySampler(interval_s=sample_interval_s)
    sampler.start()

    x = x0.copy()
    t_start = time.perf_counter()
    steps = 0

    try:
        for _ in range(max_iter):
            grad = grad_fn(x)
            if np.linalg.norm(grad) < eps:
                break
            x = optimizer.step(x, grad_fn, hess_fn)
            steps += 1
    except Exception as exc:
        print(f"[MemoryProfiler] NumPy optimizer '{optimizer.name}' raised: {exc}")
    finally:
        sampler.stop()

    t_end = time.perf_counter()

    _, heap_peak_bytes = tracemalloc.get_traced_memory()
    heap_end = _bytes_to_kb(tracemalloc.get_traced_memory()[0])
    heap_peak = _bytes_to_kb(heap_peak_bytes)
    tracemalloc.clear_traces()

    device_end = _bytes_to_kb(_get_device_bytes())
    device_peak = (
        max(sampler.device_samples_kb) if sampler.device_samples_kb else device_start
    )

    return MemoryProfile(
        optimizer_name=optimizer.name,
        backend="numpy",
        n_steps=steps,
        total_time_s=t_end - t_start,
        heap_start_kb=heap_start,
        heap_peak_kb=heap_peak,
        heap_end_kb=heap_end,
        heap_delta_kb=heap_end - heap_start,
        device_start_kb=device_start,
        device_peak_kb=device_peak,
        device_end_kb=device_end,
        heap_samples_kb=sampler.heap_samples_kb,
        device_samples_kb=sampler.device_samples_kb,
    )


# ---------------------------------------------------------------------------
# Public API -- PyTorch
# ---------------------------------------------------------------------------

def profile_torch_optimizer(
    optimizer,
    params: List,
    closure: Callable[[], Any],
    max_iter: int = 300,
    eps: float = 1e-8,
    sample_interval_s: float = 0.01,
) -> MemoryProfile:
    """
    Profile RAM (and optional VRAM) usage of a single PyTorch optimizer run.

    Parameters
    ----------
    optimizer :
        Any ``torch.optim.Optimizer`` subclass (e.g. ``AdamTorch``).
    params : list of torch.nn.Parameter
        The parameter tensors being optimized.
    closure : Callable
        A zero-argument callable that zeros gradients, computes the loss,
        calls ``loss.backward()``, and returns the loss scalar.
    max_iter : int
        Maximum number of optimizer steps.
    eps : float
        Gradient-norm convergence tolerance (computed over all params).
    sample_interval_s : float
        Background sampling interval in seconds.

    Returns
    -------
    MemoryProfile
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is not installed.")

    gc.collect()

    # Derive a human-readable name from the optimizer class
    opt_name = type(optimizer).__name__

    # Start tracemalloc
    if not tracemalloc.is_tracing():
        tracemalloc.start()
    else:
        tracemalloc.clear_traces()

    heap_start = _bytes_to_kb(tracemalloc.get_traced_memory()[0])
    device_start = _bytes_to_kb(_get_device_bytes())

    sampler = _MemorySampler(interval_s=sample_interval_s)
    sampler.start()

    t_start = time.perf_counter()
    steps = 0

    try:
        for _ in range(max_iter):
            optimizer.step(closure)
            steps += 1

            # Convergence check: gradient norm over all params
            grad_norm = 0.0
            for p in params:
                if p.grad is not None:
                    grad_norm += float(p.grad.data.norm() ** 2)
            grad_norm = grad_norm ** 0.5
            if grad_norm < eps:
                break
    except Exception as exc:
        print(f"[MemoryProfiler] Torch optimizer '{opt_name}' raised: {exc}")
    finally:
        sampler.stop()

    t_end = time.perf_counter()

    _, heap_peak_bytes = tracemalloc.get_traced_memory()
    heap_end = _bytes_to_kb(tracemalloc.get_traced_memory()[0])
    heap_peak = _bytes_to_kb(heap_peak_bytes)
    tracemalloc.clear_traces()

    device_end = _bytes_to_kb(_get_device_bytes())
    device_peak = (
        max(sampler.device_samples_kb) if sampler.device_samples_kb else device_start
    )

    return MemoryProfile(
        optimizer_name=opt_name,
        backend="torch",
        n_steps=steps,
        total_time_s=t_end - t_start,
        heap_start_kb=heap_start,
        heap_peak_kb=heap_peak,
        heap_end_kb=heap_end,
        heap_delta_kb=heap_end - heap_start,
        device_start_kb=device_start,
        device_peak_kb=device_peak,
        device_end_kb=device_end,
        heap_samples_kb=sampler.heap_samples_kb,
        device_samples_kb=sampler.device_samples_kb,
    )


# ---------------------------------------------------------------------------
# Batch profiling helpers
# ---------------------------------------------------------------------------

def profile_all_numpy(
    optimizers: List,
    x0: np.ndarray,
    grad_fn: Callable,
    hess_fn: Optional[Callable] = None,
    max_iter: int = 300,
    eps: float = 1e-8,
    sample_interval_s: float = 0.01,
    verbose: bool = True,
) -> List[MemoryProfile]:
    """
    Profile a list of NumPy optimizers in sequence and return their profiles.

    Parameters
    ----------
    optimizers : list
        List of ``core.optimizer.Optimizer`` instances.
    x0, grad_fn, hess_fn, max_iter, eps, sample_interval_s :
        Forwarded to ``profile_numpy_optimizer``.
    verbose : bool
        Print a one-line summary after each optimizer.

    Returns
    -------
    List[MemoryProfile]
    """
    profiles = []
    for opt in optimizers:
        prof = profile_numpy_optimizer(
            opt, x0, grad_fn, hess_fn,
            max_iter=max_iter, eps=eps,
            sample_interval_s=sample_interval_s,
        )
        profiles.append(prof)
        if verbose:
            print(
                f"  {prof.optimizer_name:<30}  "
                f"steps={prof.n_steps:>4}  "
                f"heap_peak={prof.heap_peak_kb:>8.1f} KB  "
                f"time={prof.total_time_s:.3f}s"
            )
    return profiles


def profile_all_torch(
    optimizer_factory_list: List[Callable],
    param_factory: Callable,
    closure_factory: Callable,
    max_iter: int = 300,
    eps: float = 1e-8,
    sample_interval_s: float = 0.01,
    verbose: bool = True,
) -> List[MemoryProfile]:
    """
    Profile a list of PyTorch optimizers in sequence.

    Each optimizer is constructed fresh via a factory to avoid state leakage.

    Parameters
    ----------
    optimizer_factory_list : list of callables
        Each callable takes a parameter list and returns an optimizer instance,
        e.g. ``lambda p: AdamTorch(p, lr=1e-3)``.
    param_factory : Callable
        Zero-argument callable that returns a fresh list of
        ``torch.nn.Parameter`` tensors.
    closure_factory : Callable
        Callable that takes the parameter list and returns a closure suitable
        for ``profile_torch_optimizer``.
    max_iter, eps, sample_interval_s :
        Forwarded to ``profile_torch_optimizer``.
    verbose : bool
        Print a one-line summary after each optimizer.

    Returns
    -------
    List[MemoryProfile]

    Example
    -------
    ::

        from utils.optimizers.torch_optimizers import AdamTorch, RMSPropTorch

        def make_params():
            return [torch.nn.Parameter(torch.zeros(10))]

        def make_closure(params):
            p = params[0]
            def closure():
                if p.grad is not None: p.grad.zero_()
                loss = (p ** 2).sum()
                loss.backward()
                return loss
            return closure

        profiles = profile_all_torch(
            optimizer_factory_list=[
                lambda p: AdamTorch(p, lr=1e-3),
                lambda p: RMSPropTorch(p, lr=1e-3),
            ],
            param_factory=make_params,
            closure_factory=make_closure,
        )
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is not installed.")

    profiles = []
    for factory in optimizer_factory_list:
        params = param_factory()
        opt = factory(params)
        closure = closure_factory(params)
        prof = profile_torch_optimizer(
            opt, params, closure,
            max_iter=max_iter, eps=eps,
            sample_interval_s=sample_interval_s,
        )
        profiles.append(prof)
        if verbose:
            print(
                f"  {prof.optimizer_name:<30}  "
                f"steps={prof.n_steps:>4}  "
                f"heap_peak={prof.heap_peak_kb:>8.1f} KB  "
                f"device_peak={prof.device_peak_kb:>8.1f} KB  "
                f"time={prof.total_time_s:.3f}s"
            )
    return profiles


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def comparison_table(profiles: List[MemoryProfile], sort_by: str = "heap_peak_kb"):
    """
    Print a formatted comparison table for a list of ``MemoryProfile`` objects.

    Parameters
    ----------
    profiles : List[MemoryProfile]
    sort_by : str
        Column to sort by.  One of: ``"heap_peak_kb"``, ``"device_peak_kb"``,
        ``"heap_delta_kb"``, ``"total_time_s"``, ``"n_steps"``.
    """
    rows = [p.to_summary_dict() for p in profiles]
    rows.sort(key=lambda r: r.get(sort_by, 0))

    has_device = any(r["device_peak_kb"] > 0 for r in rows)

    header = (
        f"{'Optimizer':<30} {'Steps':>6} {'HeapStart':>10} {'HeapPeak':>10} "
        f"{'HeapDelta':>10} {'Time(s)':>8}"
    )
    if has_device:
        header += f" {'DevPeak':>10}"
    print(header)
    print("-" * len(header))

    for r in rows:
        line = (
            f"{r['optimizer']:<30} "
            f"{r['steps']:>6} "
            f"{r['heap_start_kb']:>9.1f}K "
            f"{r['heap_peak_kb']:>9.1f}K "
            f"{r['heap_delta_kb']:>+9.1f}K "
            f"{r['total_time_s']:>8.3f}"
        )
        if has_device:
            line += f" {r['device_peak_kb']:>9.1f}K"
        print(line)


# ---------------------------------------------------------------------------
# Optional: pandas summary DataFrame
# ---------------------------------------------------------------------------

def summary_dataframe(profiles: List[MemoryProfile]):
    """
    Return a ``pandas.DataFrame`` with one row per profile.

    Requires ``pandas`` to be installed.  Raises ``ImportError`` otherwise.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for summary_dataframe().")

    rows = [p.to_summary_dict() for p in profiles]
    df = pd.DataFrame(rows)
    # Reorder columns for readability
    cols = [
        "optimizer", "backend", "steps", "total_time_s",
        "heap_start_kb", "heap_peak_kb", "heap_end_kb", "heap_delta_kb",
        "device_start_kb", "device_peak_kb", "device_end_kb",
    ]
    existing = [c for c in cols if c in df.columns]
    return df[existing].sort_values("heap_peak_kb").reset_index(drop=True)
