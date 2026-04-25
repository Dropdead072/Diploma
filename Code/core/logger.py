from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np


class Logger(ABC):
    """
    Abstract base class for optimization loggers.

    A Logger is attached to a training / optimization loop and records
    per-iteration metrics.  Concrete subclasses decide *how* to store and
    expose those metrics (in-memory, to disk, to a dashboard, etc.).
    """

    @abstractmethod
    def log(self, step: int, metrics: Dict[str, Any]) -> None:
        """
        Record a dictionary of metrics at the given iteration step.

        :param step: Current iteration index (0-based).
        :type step: int
        :param metrics: Mapping of metric name -> scalar value.
        :type metrics: Dict[str, Any]
        """

    @abstractmethod
    def reset(self) -> None:
        """Clear all recorded history."""

    @property
    @abstractmethod
    def history(self) -> List[Dict[str, Any]]:
        """Return the full list of logged metric dicts, one per step."""

    def get_metric(self, key: str) -> List[Any]:
        """
        Extract a single metric series by name across all logged steps.

        :param key: Metric name to extract.
        :type key: str
        :returns: List of values, one per step where the key was present.
        :rtype: List[Any]
        """
        return [entry[key] for entry in self.history if key in entry]
