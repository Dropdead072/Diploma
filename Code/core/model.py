from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Tuple
import numpy as np

from core.optimizer import Optimizer
from core.logger import Logger


class Model(ABC):
    """
    Abstract base class for parametric models trained via gradient-based
    optimisation.

    A Model owns:
      - a parameter vector ``weights`` (np.ndarray)
      - a ``loss_fn`` callable  (weights, X, y) -> scalar
      - a ``grad_fn`` callable  (weights, X, y) -> gradient w.r.t. weights
      - an optional ``hess_fn`` callable (weights, X, y) -> Hessian
      - an ``Optimizer`` instance
      - an optional ``Logger`` instance

    Subclasses implement ``predict`` and ``score``.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        loss_fn: Callable,
        grad_fn: Callable,
        hess_fn: Optional[Callable] = None,
        logger: Optional[Logger] = None,
    ):
        """
        :param optimizer: Optimizer instance used to update weights.
        :type optimizer: Optimizer
        :param loss_fn: Callable (w, X, y) -> float.
        :type loss_fn: Callable
        :param grad_fn: Callable (w, X, y) -> np.ndarray of same shape as w.
        :type grad_fn: Callable
        :param hess_fn: Optional callable (w, X, y) -> (n, n) np.ndarray.
        :type hess_fn: Optional[Callable]
        :param logger: Optional Logger to record per-step metrics.
        :type logger: Optional[Logger]
        """
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.grad_fn = grad_fn
        self.hess_fn = hess_fn
        self.logger = logger
        self.weights: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Compute model predictions for input matrix X.

        :param X: Input array of shape (n_samples, n_features).
        :type X: np.ndarray
        :returns: Predictions of shape (n_samples,) or (n_samples, n_outputs).
        :rtype: np.ndarray
        """

    @abstractmethod
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute a scalar performance metric (e.g. accuracy, R^2).

        :param X: Input array of shape (n_samples, n_features).
        :type X: np.ndarray
        :param y: Target array of shape (n_samples,).
        :type y: np.ndarray
        :returns: Scalar performance metric.
        :rtype: float
        """

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        max_iter: int = 1000,
        eps: float = 1e-6,
        w0: Optional[np.ndarray] = None,
    ) -> "Model":
        """
        Train the model by running the optimizer for up to ``max_iter`` steps.

        :param X: Training inputs, shape (n_samples, n_features).
        :type X: np.ndarray
        :param y: Training targets, shape (n_samples,).
        :type y: np.ndarray
        :param max_iter: Maximum number of optimizer steps.
        :type max_iter: int
        :param eps: Gradient-norm convergence threshold.
        :type eps: float
        :param w0: Optional initial weight vector.  If None, weights are
                   initialised to zeros.
        :type w0: Optional[np.ndarray]
        :returns: self (for method chaining).
        :rtype: Model
        """
        n_features = X.shape[1]
        self.weights = w0.copy() if w0 is not None else np.random.normal(0, 1, n_features)
        self.optimizer.reset()
        if self.logger is not None:
            self.logger.reset()

        # Bind X, y into the loss / grad / hess callables
        def _loss(w: np.ndarray) -> float:
            return self.loss_fn(w, X, y)

        def _grad(w: np.ndarray) -> np.ndarray:
            return self.grad_fn(w, X, y)

        def _hess(w: np.ndarray) -> np.ndarray:
            if self.hess_fn is None:
                raise NotImplementedError
            return self.hess_fn(w, X, y)

        for t in range(max_iter):
            loss_val = _loss(self.weights)
            grad_val = _grad(self.weights)
            grad_norm = float(np.linalg.norm(grad_val))

            if self.logger is not None:
                self.logger.log(
                    t,
                    {
                        "loss": float(loss_val),
                        "grad_norm": grad_norm,
                        "weights": self.weights.copy(),
                    },
                )

            if not np.isfinite(loss_val) or not np.isfinite(grad_norm):
                break

            if grad_norm < eps:
                break

            self.weights = self.optimizer.step(
                self.weights,
                _grad,
                _hess if self.hess_fn is not None else None,
            )

        return self
