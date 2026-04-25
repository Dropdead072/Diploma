import time
import numpy as np
from typing import Any, Dict, List, Optional
from core.logger import Logger


class OptimizationLogger(Logger):
    """
    Concrete in-memory logger for optimization runs.

    Records per-iteration metrics produced by the ``optimize()`` loop or any
    custom training loop.  Provides convenience methods for summarising and
    comparing multiple optimizer runs.

    Typical usage::

        logger = OptimizationLogger(optimizer_name="Adam")
        for t in range(max_iter):
            ...
            logger.log(t, {"loss": loss_val, "grad_norm": grad_norm, "x": x.copy()})

        logger.summary()
    """

    def __init__(self, optimizer_name: str = ""):
        """
        :param optimizer_name: Human-readable label for the optimizer being tracked.
        :type optimizer_name: str
        """
        self.optimizer_name = optimizer_name
        self._history: List[Dict[str, Any]] = []
        self._start_time: Optional[float] = None
        self.heap_peak_kb: float = float("nan")  # set externally after profiling

    # ------------------------------------------------------------------
    # Logger interface
    # ------------------------------------------------------------------

    def log(self, step: int, metrics: Dict[str, Any]) -> None:
        """
        Record metrics at the given step.  Automatically attaches a wall-clock
        ``elapsed`` field (seconds since the first ``log`` call).

        :param step: Current iteration index.
        :type step: int
        :param metrics: Dict of metric name -> value.
        :type metrics: Dict[str, Any]
        """
        if self._start_time is None:
            self._start_time = time.perf_counter()

        entry = {"step": step, "elapsed": time.perf_counter() - self._start_time}
        entry.update(metrics)
        self._history.append(entry)

    def reset(self) -> None:
        """Clear all recorded history and reset the timer."""
        self._history = []
        self._start_time = None

    @property
    def history(self) -> List[Dict[str, Any]]:
        """Full list of logged metric dicts, one per step."""
        return self._history

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def losses(self) -> List[float]:
        """Return the ``loss`` series across all steps."""
        return self.get_metric("loss")

    def grad_norms(self) -> List[float]:
        """Return the ``grad_norm`` series across all steps."""
        return self.get_metric("grad_norm")

    def elapsed_times(self) -> List[float]:
        """Return wall-clock elapsed time at each logged step."""
        return self.get_metric("elapsed")

    def best_loss(self) -> float:
        """Return the minimum recorded loss value."""
        losses = self.losses()
        return float(np.min(losses)) if losses else float("inf")

    def final_loss(self) -> float:
        """Return the loss at the last recorded step."""
        losses = self.losses()
        return float(losses[-1]) if losses else float("nan")

    def n_steps(self) -> int:
        """Return the total number of logged steps."""
        return len(self._history)

    def total_time(self) -> float:
        """Return total wall-clock time of the run in seconds."""
        times = self.elapsed_times()
        return float(times[-1]) if times else 0.0

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """
        Return a summary dict with key statistics for this run.

        :returns: Dict with keys: optimizer, steps, best_loss, final_loss,
                  total_time_s, heap_peak_kb.
        :rtype: Dict[str, Any]
        """
        return {
            "optimizer":    self.optimizer_name,
            "steps":        self.n_steps(),
            "best_loss":    self.best_loss(),
            "final_loss":   self.final_loss(),
            "total_time_s": self.total_time(),
            "heap_peak_kb": self.heap_peak_kb,
        }

    def __repr__(self) -> str:
        s = self.summary()
        mem_str = f"{s['heap_peak_kb']:.1f} KB" if s['heap_peak_kb'] == s['heap_peak_kb'] else "n/a"
        return (
            f"OptimizationLogger(optimizer='{s['optimizer']}', "
            f"steps={s['steps']}, "
            f"final_loss={s['final_loss']:.6g}, "
            f"best_loss={s['best_loss']:.6g}, "
            f"time={s['total_time_s']:.3f}s, "
            f"heap_peak={mem_str})"
        )


class MultiRunLogger:
    """
    Aggregates multiple ``OptimizationLogger`` instances -- one per optimizer --
    to enable side-by-side comparison.

    Usage::

        multi = MultiRunLogger()
        for opt in optimizers:
            logger = multi.new_run(opt.name)
            # ... run optimization, calling logger.log(...) each step ...

        multi.comparison_table()
    """

    def __init__(self):
        self._runs: Dict[str, OptimizationLogger] = {}

    def new_run(self, optimizer_name: str) -> OptimizationLogger:
        """
        Create and register a fresh ``OptimizationLogger`` for the given optimizer.

        :param optimizer_name: Label for this run.
        :type optimizer_name: str
        :returns: A fresh logger ready to receive ``log()`` calls.
        :rtype: OptimizationLogger
        """
        logger = OptimizationLogger(optimizer_name=optimizer_name)
        self._runs[optimizer_name] = logger
        return logger

    def get_run(self, optimizer_name: str) -> OptimizationLogger:
        """Retrieve the logger for a previously registered optimizer."""
        return self._runs[optimizer_name]

    @property
    def run_names(self) -> List[str]:
        """Names of all registered runs."""
        return list(self._runs.keys())

    def comparison_table(self) -> List[Dict[str, Any]]:
        """
        Return a list of summary dicts, one per run, sorted by final_loss.

        :returns: List of summary dicts.
        :rtype: List[Dict[str, Any]]
        """
        summaries = [logger.summary() for logger in self._runs.values()]
        return sorted(summaries, key=lambda s: s["final_loss"])

    def print_comparison(self) -> None:
        """Print a formatted comparison table to stdout, including heap peak if available."""
        table = self.comparison_table()
        import math
        has_mem = any(not math.isnan(row.get("heap_peak_kb", float("nan"))) for row in table)
        if has_mem:
            header = (
                f"{'Optimizer':<30} {'Steps':>6} {'Final Loss':>14} "
                f"{'Best Loss':>14} {'Time (s)':>10} {'Heap Peak':>12}"
            )
        else:
            header = (
                f"{'Optimizer':<30} {'Steps':>6} {'Final Loss':>14} "
                f"{'Best Loss':>14} {'Time (s)':>10}"
            )
        print(header)
        print("-" * len(header))
        for row in table:
            line = (
                f"{row['optimizer']:<30} "
                f"{row['steps']:>6} "
                f"{row['final_loss']:>14.6g} "
                f"{row['best_loss']:>14.6g} "
                f"{row['total_time_s']:>10.3f}"
            )
            if has_mem:
                kb = row.get("heap_peak_kb", float("nan"))
                mem_str = f"{kb:>9.1f} KB" if not math.isnan(kb) else f"{'n/a':>12}"
                line += f" {mem_str}"
            print(line)
