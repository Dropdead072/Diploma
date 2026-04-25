"""
Numpy Linear Regression with pluggable loss and optimizer.
"""

import numpy as np
from typing import Optional, Callable

from core.model import Model
from core.optimizer import Optimizer
from core.logger import Logger
from utils.losses.numpy_losses import LossFunction, MSELoss


class LinearRegression(Model):
    """
    Linear regression model:  y_hat = X w

    Trained by minimising a pluggable loss (default: MSE) with a pluggable
    optimizer.  Supports an optional bias term (intercept) appended as the
    last element of the weight vector when ``fit_intercept=True``.

    Parameters
    ----------
    optimizer : Optimizer
        Any optimizer from ``numpy_optimizers_full.py``.
    loss : LossFunction, optional
        Loss function bundle (value + grad + optional hess).
        Defaults to ``MSELoss``.
    fit_intercept : bool
        If True, a column of ones is prepended to X before fitting.
    logger : Logger, optional
        Logger instance for recording per-step metrics.

    Examples
    --------
    >>> from Diploma.Code.utils.optimizers.numpy_optimizers_full import Adam
    >>> from Diploma.Code.utils.losses.numpy_losses import MSELoss
    >>> model = LinearRegression(optimizer=Adam(lr=1e-2), loss=MSELoss)
    >>> model.fit(X_train, y_train, max_iter=500)
    >>> y_pred = model.predict(X_test)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        loss: LossFunction = MSELoss,
        fit_intercept: bool = True,
        logger: Optional[Logger] = None,
    ):
        super().__init__(
            optimizer=optimizer,
            loss_fn=loss.value,
            grad_fn=loss.grad,
            hess_fn=loss.hess,
            logger=logger,
        )
        self.loss = loss
        self.fit_intercept = fit_intercept

    @property
    def name(self) -> str:
        return f"LinearRegression[{self.loss.name}, {self.optimizer.name}]"

    # ------------------------------------------------------------------
    # Data preprocessing
    # ------------------------------------------------------------------

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        if self.fit_intercept:
            return np.hstack([X, np.ones((X.shape[0], 1))])
        return X

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        max_iter: int = 1000,
        eps: float = 1e-6,
        w0: Optional[np.ndarray] = None,
    ) -> "LinearRegression":
        """
        Fit the model to training data.

        :param X: Training inputs, shape (n_samples, n_features).
        :param y: Training targets, shape (n_samples,).
        :param max_iter: Maximum optimizer steps.
        :param eps: Gradient-norm convergence threshold.
        :param w0: Optional initial weight vector.
        :returns: self
        """
        X_aug = self._add_intercept(X)
        return super().fit(X_aug, y, max_iter=max_iter, eps=eps, w0=w0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict targets for input X.

        :param X: Input array, shape (n_samples, n_features).
        :returns: Predicted values, shape (n_samples,).
        """
        if self.weights is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        X_aug = self._add_intercept(X)
        return X_aug @ self.weights

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute R2 (coefficient of determination).

        :param X: Input array, shape (n_samples, n_features).
        :param y: True targets, shape (n_samples,).
        :returns: R^2 score in (-inf, 1].
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return float(1.0 - ss_res / (ss_tot + 1e-12))

    def mse(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return mean squared error on (X, y)."""
        y_pred = self.predict(X)
        return float(np.mean((y - y_pred) ** 2))

    def mae(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return mean absolute error on (X, y)."""
        y_pred = self.predict(X)
        return float(np.mean(np.abs(y - y_pred)))

    @property
    def coef_(self) -> np.ndarray:
        """Feature coefficients (excludes intercept if fit_intercept=True)."""
        if self.weights is None:
            raise RuntimeError("Model has not been fitted yet.")
        return self.weights[:-1] if self.fit_intercept else self.weights

    @property
    def intercept_(self) -> float:
        """Intercept (bias) term.  Returns 0.0 if fit_intercept=False."""
        if self.weights is None:
            raise RuntimeError("Model has not been fitted yet.")
        return float(self.weights[-1]) if self.fit_intercept else 0.0
