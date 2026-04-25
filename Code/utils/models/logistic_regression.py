"""
Numpy Logistic Regression with pluggable loss and optimizer.

Supports both binary (BinaryCrossEntropyLoss) and multi-class
(CrossEntropyLoss) classification via the same interface.
"""

import numpy as np
from typing import Optional

from core.model import Model
from core.optimizer import Optimizer
from core.logger import Logger
from utils.losses.numpy_losses import (
    LossFunction,
    BinaryCrossEntropyLoss,
    _sigmoid,
)


class LogisticRegression(Model):
    """
    Binary logistic regression model:  P(y=1|x) = sigma(x^T w)

    Trained by minimising Binary Cross-Entropy (default) with a pluggable
    optimizer.  Supports an optional bias term when ``fit_intercept=True``.

    For multi-class classification use ``SoftmaxRegression`` below.

    Parameters
    ----------
    optimizer : Optimizer
        Any optimizer from ``numpy_optimizers_full.py``.
    loss : LossFunction, optional
        Loss function bundle.  Defaults to ``BinaryCrossEntropyLoss``.
    fit_intercept : bool
        If True, a column of ones is appended to X before fitting.
    threshold : float
        Decision threshold for ``predict_labels``.  Default 0.5.
    logger : Logger, optional
        Logger instance for recording per-step metrics.

    Examples
    --------
    >>> from Diploma.Code.utils.optimizers.numpy_optimizers_full import Adam
    >>> model = LogisticRegression(optimizer=Adam(lr=1e-2))
    >>> model.fit(X_train, y_train, max_iter=500)
    >>> labels = model.predict_labels(X_test)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        loss: LossFunction = BinaryCrossEntropyLoss,
        fit_intercept: bool = True,
        threshold: float = 0.5,
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
        self.threshold = threshold

    @property
    def name(self) -> str:
        return f"LogisticRegression[{self.loss.name}, {self.optimizer.name}]"

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
    ) -> "LogisticRegression":
        """
        Fit the model to training data.

        :param X: Training inputs, shape (n_samples, n_features).
        :param y: Binary targets in {0, 1}, shape (n_samples,).
        :param max_iter: Maximum optimizer steps.
        :param eps: Gradient-norm convergence threshold.
        :param w0: Optional initial weight vector.
        :returns: self
        """
        X_aug = self._add_intercept(X)
        return super().fit(X_aug, y, max_iter=max_iter, eps=eps, w0=w0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities P(y=1|x).

        :param X: Input array, shape (n_samples, n_features).
        :returns: Probability array, shape (n_samples,), values in [0, 1].
        """
        if self.weights is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        X_aug = self._add_intercept(X)
        return _sigmoid(X_aug @ self.weights)

    def predict_labels(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary class labels using ``self.threshold``.

        :param X: Input array, shape (n_samples, n_features).
        :returns: Integer label array in {0, 1}, shape (n_samples,).
        """
        return (self.predict(X) >= self.threshold).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute classification accuracy.

        :param X: Input array, shape (n_samples, n_features).
        :param y: True binary labels in {0, 1}, shape (n_samples,).
        :returns: Accuracy in [0, 1].
        """
        return float(np.mean(self.predict_labels(X) == y))

    def log_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return binary cross-entropy loss on (X, y)."""
        X_aug = self._add_intercept(X)
        return float(self.loss_fn(self.weights, X_aug, y))

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


class SoftmaxRegression(Model):
    """
    Multi-class logistic regression (Softmax / Multinomial).

    The weight matrix W has shape (n_features, n_classes) and is stored
    flattened as a 1-D vector for compatibility with the optimizer interface.

    Parameters
    ----------
    optimizer : Optimizer
        Any optimizer from ``numpy_optimizers_full.py``.
    n_classes : int
        Number of target classes K.
    loss : LossFunction, optional
        Defaults to ``CrossEntropyLoss``.
    fit_intercept : bool
        If True, a column of ones is appended to X before fitting.
    logger : Logger, optional
        Logger instance for recording per-step metrics.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        n_classes: int,
        loss: Optional[LossFunction] = None,
        fit_intercept: bool = True,
        logger: Optional[Logger] = None,
    ):
        # Import here to avoid circular import at module level
        from Diploma.Code.utils.losses.numpy_losses import CrossEntropyLoss
        _loss = loss if loss is not None else CrossEntropyLoss

        super().__init__(
            optimizer=optimizer,
            loss_fn=_loss.value,
            grad_fn=_loss.grad,
            hess_fn=_loss.hess,
            logger=logger,
        )
        self.loss = _loss
        self.n_classes = n_classes
        self.fit_intercept = fit_intercept
        self._n_features: Optional[int] = None

    @property
    def name(self) -> str:
        return f"SoftmaxRegression[{self.loss.name}, {self.optimizer.name}, K={self.n_classes}]"

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        if self.fit_intercept:
            return np.hstack([X, np.ones((X.shape[0], 1))])
        return X

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        max_iter: int = 1000,
        eps: float = 1e-6,
        w0: Optional[np.ndarray] = None,
    ) -> "SoftmaxRegression":
        """
        Fit the model to training data.

        :param X: Training inputs, shape (n_samples, n_features).
        :param y: Integer class labels in {0,...,K-1}, shape (n_samples,).
        :param max_iter: Maximum optimizer steps.
        :param eps: Gradient-norm convergence threshold.
        :param w0: Optional initial weight vector of shape (n_features * K,).
        :returns: self
        """
        X_aug = self._add_intercept(X)
        self._n_features = X_aug.shape[1]
        if w0 is None:
            w0 = np.zeros(self._n_features * self.n_classes)
        return super().fit(X_aug, y, max_iter=max_iter, eps=eps, w0=w0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        :param X: Input array, shape (n_samples, n_features).
        :returns: Probability matrix, shape (n_samples, n_classes).
        """
        if self.weights is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        from Diploma.Code.utils.losses.numpy_losses import _softmax
        X_aug = self._add_intercept(X)
        W_mat = self.weights.reshape(self._n_features, self.n_classes)
        return _softmax(X_aug @ W_mat)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels (argmax of probabilities).

        :param X: Input array, shape (n_samples, n_features).
        :returns: Integer label array, shape (n_samples,).
        """
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute classification accuracy.

        :param X: Input array, shape (n_samples, n_features).
        :param y: True integer labels, shape (n_samples,).
        :returns: Accuracy in [0, 1].
        """
        return float(np.mean(self.predict(X) == y.astype(int)))
