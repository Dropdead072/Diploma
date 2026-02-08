from abc import ABC, abstractmethod
from core.objective_function import ObjectiveFunction
from typing import Optional
import time
import numpy as np

class Optimizer(ABC):
    def __init__(self):
        self.iterations = 0

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def reset(self) -> None:
        self.iterations = 0

    @abstractmethod
    def step(
        self,
        x: np.ndarray,
        grad: np.ndarray,
        hess: Optional[np.ndarray] = None) -> np.ndarray:
        pass


def optimize(
    func: ObjectiveFunction,
    optimizer: Optimizer,
    x0: np.ndarray,
    max_iter: int = 1000,
    eps: float = 1e-6
):
    x = x0.copy()
    optimizer.reset()

    history = []
    start_time = time.time()

    for t in range(max_iter):
        f_val = func.value(x)
        grad = func.grad(x)
        grad_norm = np.linalg.norm(grad)

        if not np.isfinite(f_val) or not np.isfinite(grad_norm):
            print("Numerical explosion detected")
            break

        history.append({
            "iter": t,
            "x": x.copy(),
            "f": f_val,
            "grad_norm": grad_norm,
            "time": time.time() - start_time
        })

        if grad_norm < eps:
            break

        try:
            hess = func.hess(x)
        except NotImplementedError:
            hess = None            

        x = optimizer.step(x, grad, hess)

    return x, history
