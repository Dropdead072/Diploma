"""
Numpy implementations of common loss functions.

Each loss exposes three callables with the signature:
    value(w, X, y)  -> float
    grad(w, X, y)   -> np.ndarray  (gradient w.r.t. w)
    hess(w, X, y)   -> np.ndarray  (Hessian w.r.t. w, where available)

They are collected into dataclass-like ``LossFunction`` objects so that a
model can receive a single ``loss`` argument and unpack value / grad / hess.
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class LossFunction:
    """
    Container bundling a loss value, its gradient, and (optionally) its Hessian.

    :param name: Human-readable name.
    :param value: Callable (w, X, y) -> float.
    :param grad: Callable (w, X, y) -> np.ndarray.
    :param hess: Optional callable (w, X, y) -> np.ndarray (nxn).
    """
    name: str
    value: Callable
    grad: Callable
    hess: Optional[Callable] = None


# =============================================================================
# Mean Squared Error  (for regression)
# =============================================================================

def _mse_value(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """
    MSE = (1/n) ||X w - y||^2
    """
    residuals = X @ w - y
    return float(np.mean(residuals ** 2))


def _mse_grad(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    grad_w MSE = (2/n) X^T (X w - y)
    """
    n = X.shape[0]
    residuals = X @ w - y
    return (2.0 / n) * X.T @ residuals


def _mse_hess(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    grad^2_w MSE = (2/n) X^T X
    """
    n = X.shape[0]
    return (2.0 / n) * X.T @ X


MSELoss = LossFunction(
    name="MSE",
    value=_mse_value,
    grad=_mse_grad,
    hess=_mse_hess,
)


# =============================================================================
# Mean Absolute Error  (for regression, no Hessian -- non-smooth)
# =============================================================================

def _mae_value(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """
    MAE = (1/n) ||X w - y||_1
    """
    return float(np.mean(np.abs(X @ w - y)))


def _mae_grad(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    grad_w MAE = (1/n) X^T sign(X w - y)
    """
    n = X.shape[0]
    residuals = X @ w - y
    return (1.0 / n) * X.T @ np.sign(residuals)


MAELoss = LossFunction(
    name="MAE",
    value=_mae_value,
    grad=_mae_grad,
    hess=None,   # non-differentiable at residual = 0
)


# =============================================================================
# Binary Cross-Entropy  (for binary logistic regression)
# =============================================================================

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def _bce_value(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """
    BCE = -(1/n) sum [ y_i log sigma(x_i^T w) + (1-y_i) log(1 - sigma(x_i^T w)) ]
    """
    n = X.shape[0]
    logits = X @ w
    probs = _sigmoid(logits)
    eps = 1e-12
    return float(-np.mean(y * np.log(probs + eps) + (1.0 - y) * np.log(1.0 - probs + eps)))


def _bce_grad(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    grad_w BCE = (1/n) X^T (sigma(Xw) - y)
    """
    n = X.shape[0]
    probs = _sigmoid(X @ w)
    return (1.0 / n) * X.T @ (probs - y)


def _bce_hess(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    grad^2_w BCE = (1/n) X^T diag(sigma(1-sigma)) X
    """
    n = X.shape[0]
    probs = _sigmoid(X @ w)
    S = probs * (1.0 - probs)          # shape (n,)
    return (1.0 / n) * (X.T * S) @ X  # (d, n) * (n,) -> (d, n) then @ (n, d)


BinaryCrossEntropyLoss = LossFunction(
    name="BinaryCrossEntropy",
    value=_bce_value,
    grad=_bce_grad,
    hess=_bce_hess,
)


# =============================================================================
# Softmax Cross-Entropy  (for multi-class logistic regression)
# =============================================================================

def _softmax(Z: np.ndarray) -> np.ndarray:
    """Row-wise numerically stable softmax.  Z shape: (n, K)."""
    Z_shifted = Z - Z.max(axis=1, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    return exp_Z / exp_Z.sum(axis=1, keepdims=True)


def _ce_value(W: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """
    Softmax cross-entropy for multi-class classification.

    W is flattened: shape (n_features * n_classes,).
    y contains integer class labels in {0, ..., K-1}.

    CE = -(1/n) sum_i log P(y_i | x_i)
    """
    n, d = X.shape
    K = W.shape[0] // d
    W_mat = W.reshape(d, K)
    logits = X @ W_mat                  # (n, K)
    probs = _softmax(logits)            # (n, K)
    eps = 1e-12
    log_probs = np.log(probs[np.arange(n), y.astype(int)] + eps)
    return float(-np.mean(log_probs))


def _ce_grad(W: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    grad_W CE = (1/n) X^T (P - Y_onehot),  returned flattened.
    """
    n, d = X.shape
    K = W.shape[0] // d
    W_mat = W.reshape(d, K)
    probs = _softmax(X @ W_mat)         # (n, K)
    Y_oh = np.zeros_like(probs)
    Y_oh[np.arange(n), y.astype(int)] = 1.0
    grad_mat = (1.0 / n) * X.T @ (probs - Y_oh)   # (d, K)
    return grad_mat.ravel()


CrossEntropyLoss = LossFunction(
    name="CrossEntropy",
    value=_ce_value,
    grad=_ce_grad,
    hess=None,   # full Hessian is O(d^2K^2) -- not practical; use diagonal approx
)
