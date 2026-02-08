from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional


class ObjectiveFunction(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def bounds(self) -> Tuple[Tuple[float, float], ...]:
        """
        Ограничения по каждой координате
        """
        pass

    @abstractmethod
    def value(self, x: np.ndarray) -> float:
        pass

    @abstractmethod
    def grad(self, x: np.ndarray) -> np.ndarray:
        pass

    def hess(self, x: np.ndarray) -> Optional[np.ndarray]:
        """
        Hessian опционален — по умолчанию не реализован
        """
        return None
