import numpy as np
from collections import deque
from core.optimizer import Optimizer
from typing import Callable, Optional


# =============================================================================
# Group 0 -- Vanilla / deterministic first-order methods
# (GD, Momentum, Nesterov, Newton already exist in numpy_optimizers.py;
#  they are reproduced here so this file is self-contained)
# =============================================================================

class GradientDescent(Optimizer):
    """
    Vanilla (full-batch) Gradient Descent.

    Update rule:
        x_{k+1} = x_k - lr * gradf(x_k)
    """

    def __init__(self, lr: float = 1e-3):
        """
        :param lr: Learning rate.
        :type lr: float
        """
        super().__init__()
        self.lr = lr

    @property
    def name(self) -> str:
        return "Gradient Descent"

    def step(
        self,
        x: np.ndarray,
        grad_fn: Callable[[np.ndarray], np.ndarray],
        hess: Optional[Callable] = None,
    ) -> np.ndarray:
        grad = grad_fn(x)
        self.iterations += 1
        return x - self.lr * grad


class MomentumSGD(Optimizer):
    """
    SGD with Heavy-Ball Momentum (Polyak 1964).

    Accumulates a velocity vector that damps oscillations and accelerates
    convergence along low-curvature directions.

    Update rule (from theory/momentum.md):
        h_0 = 0
        h_k = alpha * h_{k-1} + gradf(x_{k-1})
        x_k = x_{k-1} - lr * h_k
    """

    def __init__(self, lr: float = 1e-3, momentum_coef: float = 0.9):
        """
        :param lr: Learning rate.
        :type lr: float
        :param momentum_coef: Momentum decay coefficient alpha in (0, 1).
        :type momentum_coef: float
        """
        super().__init__()
        self.lr = lr
        self.momentum_coef = momentum_coef
        self.h: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        return "Momentum SGD"

    def reset(self) -> None:
        super().reset()
        self.h = None

    def step(
        self,
        x: np.ndarray,
        grad_fn: Callable[[np.ndarray], np.ndarray],
        hess: Optional[Callable] = None,
    ) -> np.ndarray:
        if self.h is None:
            self.h = np.zeros_like(x)

        grad = grad_fn(x)
        self.h = self.momentum_coef * self.h + grad
        self.iterations += 1
        return x - self.lr * self.h


class NesterovMomentum(Optimizer):
    """
    Nesterov Accelerated Gradient (NAG).

    Evaluates the gradient at the *lookahead* point x - alpha*h rather than
    at the current x, giving optimal O(1/k^2) convergence on smooth convex
    functions.

    Update rule (from theory/momentum.md):
        h_0 = 0
        h_k = alpha * h_{k-1} + gradf(x_{k-1} - alpha * h_{k-1})
        x_k = x_{k-1} - lr * h_k
    """

    def __init__(self, lr: float = 1e-3, momentum_coef: float = 0.9):
        """
        :param lr: Learning rate.
        :type lr: float
        :param momentum_coef: Momentum decay coefficient alpha in (0, 1).
        :type momentum_coef: float
        """
        super().__init__()
        self.lr = lr
        self.momentum_coef = momentum_coef
        self.h: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        return "Nesterov Momentum"

    def reset(self) -> None:
        super().reset()
        self.h = None

    def step(
        self,
        x: np.ndarray,
        grad_fn: Callable[[np.ndarray], np.ndarray],
        hess: Optional[Callable] = None,
    ) -> np.ndarray:
        if self.h is None:
            self.h = np.zeros_like(x)

        x_lookahead = x - self.momentum_coef * self.h
        grad = grad_fn(x_lookahead)
        self.h = self.momentum_coef * self.h + self.lr * grad
        self.iterations += 1
        return x - self.h


class NewtonsMethod(Optimizer):
    """
    Pure Newton's Method (second-order).

    Minimises the local quadratic Taylor model of f at each step:
        x_{k+1} = x_k - [grad^2f(x_k)]^{-1} gradf(x_k)

    Requires ``hess`` to be callable; raises ``NotImplementedError`` otherwise.
    The ``lr`` parameter scales the Newton step (lr=1 is the pure Newton step).
    """

    def __init__(self, lr: float = 1.0, damping: float = 1e-6):
        """
        :param lr: Step-size scaling factor (1.0 = pure Newton step).
        :type lr: float
        :param damping: Tikhonov regularisation added to the Hessian diagonal
                        to prevent singular-matrix errors when the Hessian is
                        rank-deficient (e.g. saturated BCE probabilities).
        :type damping: float
        """
        super().__init__()
        self.lr = lr
        self.damping = damping

    @property
    def name(self) -> str:
        return "Newton's Method"

    def step(
        self,
        x: np.ndarray,
        grad_fn: Callable[[np.ndarray], np.ndarray],
        hess: Optional[Callable] = None,
    ) -> np.ndarray:
        if hess is None:
            raise NotImplementedError(
                "Newton's Method requires a Hessian callable via the `hess` argument."
            )
        grad = grad_fn(x)
        H = hess(x)
        n = H.shape[0]
        H_reg = H + self.damping * np.eye(n)
        self.iterations += 1
        return x - self.lr * np.linalg.solve(H_reg, grad)


# =============================================================================
# Group 1 -- First-order stochastic methods
# =============================================================================

class SGD(Optimizer):
    """
    Stochastic Gradient Descent.

    Identical update rule to vanilla GD, but the caller is expected to supply
    a ``grad_fn`` that returns a *stochastic* gradient estimate (e.g. computed
    on a single randomly-drawn sample).  When used inside the standard
    ``optimize()`` loop the full gradient is used; wrap ``func.grad`` with a
    sampler to get true SGD behaviour.

    Update rule:
        x_{k+1} = x_k - lr * g_k,   g_k ~ stochastic estimate of gradf(x_k)
    """

    def __init__(self, lr: float = 1e-3):
        """
        :param lr: Learning rate (step size).
        :type lr: float
        """
        super().__init__()
        self.lr = lr

    @property
    def name(self) -> str:
        return "SGD"

    def step(
        self,
        x: np.ndarray,
        grad_fn: Callable[[np.ndarray], np.ndarray],
        hess: Optional[Callable] = None,
    ) -> np.ndarray:
        grad = grad_fn(x)
        self.iterations += 1
        return x - self.lr * grad


class MiniBatchSGD(Optimizer):
    """
    Mini-Batch Stochastic Gradient Descent.

    Like SGD but the gradient estimate is averaged over a random mini-batch of
    indices drawn uniformly from ``[0, n_samples)``.  The caller must supply a
    ``grad_fn`` with signature ``grad_fn(x, indices) -> np.ndarray`` that
    computes the gradient on the given subset.

    When ``grad_fn`` does not accept an ``indices`` argument (e.g. the plain
    ``func.grad`` from the standard loop) the class falls back to calling
    ``grad_fn(x)`` directly -- identical to plain SGD.

    Update rule:
        B  ~ Uniform subset of {1,...,n}, |B| = batch_size
        g_k = (1/|B|) sum_{iinB} gradf_i(x_k)
        x_{k+1} = x_k - lr * g_k
    """

    def __init__(self, lr: float = 1e-3, batch_size: int = 32, n_samples: int = 1000):
        """
        :param lr: Learning rate.
        :type lr: float
        :param batch_size: Number of samples per mini-batch.
        :type batch_size: int
        :param n_samples: Total dataset size (used to draw random indices).
        :type n_samples: int
        """
        super().__init__()
        self.lr = lr
        self.batch_size = batch_size
        self.n_samples = n_samples

    @property
    def name(self) -> str:
        return "Mini-Batch SGD"

    def step(
        self,
        x: np.ndarray,
        grad_fn: Callable,
        hess: Optional[Callable] = None,
    ) -> np.ndarray:
        indices = np.random.choice(self.n_samples, size=self.batch_size, replace=False)
        try:
            grad = grad_fn(x, indices)
        except TypeError:
            # grad_fn does not accept indices -- fall back to full gradient
            grad = grad_fn(x)
        self.iterations += 1
        return x - self.lr * grad


# =============================================================================
# Group 2 -- Adaptive first-order methods
# =============================================================================

class AdaGrad(Optimizer):
    """
    AdaGrad -- Adaptive Gradient algorithm.

    Accumulates the sum of squared gradients and uses it as a per-coordinate
    preconditioner.  Coordinates with large historical gradients receive a
    smaller effective learning rate.

    Update rule (from theory/adaptive_first_order.md):
        g_k  = gradf(x_k)
        G_k  = G_{k-1} + diag(g_k)^2          (accumulated sq. gradients)
        x_{k+1} = x_k - lr * G_k^{-1/2} * g_k

    Implemented element-wise as:
        x_{k+1} = x_k - lr * g_k / (sqrt(G_k) + eps)
    """

    def __init__(self, lr: float = 1e-2, eps: float = 1e-8):
        """
        :param lr: Global learning rate.
        :type lr: float
        :param eps: Small constant for numerical stability.
        :type eps: float
        """
        super().__init__()
        self.lr = lr
        self.eps = eps
        self.G: Optional[np.ndarray] = None  # accumulated squared gradients

    @property
    def name(self) -> str:
        return "AdaGrad"

    def reset(self) -> None:
        super().reset()
        self.G = None

    def step(
        self,
        x: np.ndarray,
        grad_fn: Callable[[np.ndarray], np.ndarray],
        hess: Optional[Callable] = None,
    ) -> np.ndarray:
        if self.G is None:
            self.G = np.zeros_like(x)

        grad = grad_fn(x)
        self.G += grad ** 2
        self.iterations += 1
        return x - self.lr * grad / (np.sqrt(self.G) + self.eps)


class RMSProp(Optimizer):
    """
    RMSProp -- Root Mean Square Propagation.

    Fixes AdaGrad's monotonically shrinking learning rate by replacing the
    cumulative sum of squared gradients with an exponential moving average.

    Update rule (from theory/adaptive_first_order.md):
        g_k    = gradf(x_k)
        v_{k+1} = beta * v_k + (1 - beta) * diag(g_k)^2
        x_{k+1} = x_k - lr * v_{k+1}^{-1/2} * g_k

    Implemented element-wise as:
        x_{k+1} = x_k - lr * g_k / (sqrt(v_{k+1}) + eps)
    """

    def __init__(self, lr: float = 1e-3, beta: float = 0.9, eps: float = 1e-8):
        """
        :param lr: Global learning rate.
        :type lr: float
        :param beta: Decay rate for the EMA of squared gradients.
        :type beta: float
        :param eps: Small constant for numerical stability.
        :type eps: float
        """
        super().__init__()
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.v: Optional[np.ndarray] = None  # EMA of squared gradients

    @property
    def name(self) -> str:
        return "RMSProp"

    def reset(self) -> None:
        super().reset()
        self.v = None

    def step(
        self,
        x: np.ndarray,
        grad_fn: Callable[[np.ndarray], np.ndarray],
        hess: Optional[Callable] = None,
    ) -> np.ndarray:
        if self.v is None:
            self.v = np.zeros_like(x)

        grad = grad_fn(x)
        self.v = self.beta * self.v + (1.0 - self.beta) * grad ** 2
        self.iterations += 1
        return x - self.lr * grad / (np.sqrt(self.v) + self.eps)


class Adam(Optimizer):
    """
    Adam -- Adaptive Moment Estimation.

    Maintains exponential moving averages of both the gradient (1st moment)
    and the squared gradient (2nd moment), with bias-correction terms to
    account for the zero-initialisation of the moment buffers.

    Update rule:
        g_k    = gradf(x_k)
        m_k    = beta1 * m_{k-1} + (1 - beta1) * g_k          (1st moment)
        v_k    = beta2 * v_{k-1} + (1 - beta2) * g_k^2         (2nd moment)
        m?_k   = m_k / (1 - beta1^k)                           (bias-corrected)
        v?_k   = v_k / (1 - beta2^k)                           (bias-corrected)
        x_{k+1} = x_k - lr * m?_k / (sqrt(v?_k) + eps)
    """

    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        """
        :param lr: Global learning rate.
        :type lr: float
        :param beta1: Decay rate for the 1st moment estimate.
        :type beta1: float
        :param beta2: Decay rate for the 2nd moment estimate.
        :type beta2: float
        :param eps: Small constant for numerical stability.
        :type eps: float
        """
        super().__init__()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m: Optional[np.ndarray] = None  # 1st moment
        self.v: Optional[np.ndarray] = None  # 2nd moment

    @property
    def name(self) -> str:
        return "Adam"

    def reset(self) -> None:
        super().reset()
        self.m = None
        self.v = None

    def step(
        self,
        x: np.ndarray,
        grad_fn: Callable[[np.ndarray], np.ndarray],
        hess: Optional[Callable] = None,
    ) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(x)
            self.v = np.zeros_like(x)

        self.iterations += 1
        t = self.iterations

        grad = grad_fn(x)
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * grad ** 2

        m_hat = self.m / (1.0 - self.beta1 ** t)
        v_hat = self.v / (1.0 - self.beta2 ** t)

        return x - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# =============================================================================
# Group 3 -- Diagonal and block-diagonal Hessian approximations
# =============================================================================

class DiagonalNewton(Optimizer):
    """
    Diagonal Newton method.

    Uses only the diagonal of the Hessian as a cheap preconditioner.
    This is a diagonal approximation to the full Newton step:

        x_{k+1} = x_k - lr * g_k / (|diag(H(x_k))| + eps)

    The absolute value prevents sign issues when the Hessian is indefinite.
    Requires ``hess`` to be callable; raises ``NotImplementedError`` otherwise.
    """

    def __init__(self, lr: float = 1.0, eps: float = 1e-8):
        """
        :param lr: Step-size scaling factor (1.0 = pure Newton diagonal step).
        :type lr: float
        :param eps: Damping constant added to diagonal for numerical stability.
        :type eps: float
        """
        super().__init__()
        self.lr = lr
        self.eps = eps

    @property
    def name(self) -> str:
        return "Diagonal Newton"

    def step(
        self,
        x: np.ndarray,
        grad_fn: Callable[[np.ndarray], np.ndarray],
        hess: Optional[Callable] = None,
    ) -> np.ndarray:
        if hess is None:
            raise NotImplementedError(
                "DiagonalNewton requires a Hessian callable via the `hess` argument."
            )
        grad = grad_fn(x)
        H = hess(x)
        diag_H = np.diag(H) if H.ndim == 2 else H  # accept full matrix or pre-extracted diag
        self.iterations += 1
        return x - self.lr * grad / (np.abs(diag_H) + self.eps)


class BlockNewton(Optimizer):
    """
    Block-diagonal Newton method.

    Partitions the parameter vector into non-overlapping blocks of size
    ``block_size`` and applies an independent Newton step within each block
    using the corresponding sub-matrix of the Hessian.

    For block b with indices I_b:
        H_b  = H[I_b, :][:, I_b]
        g_b  = g[I_b]
        p_b  = (H_b + eps * I)^{-1} g_b
        x_{k+1}[I_b] = x_k[I_b] - lr * p_b

    Requires ``hess`` to be callable.
    """

    def __init__(self, lr: float = 1.0, block_size: int = 10, eps: float = 1e-6):
        """
        :param lr: Step-size scaling factor.
        :type lr: float
        :param block_size: Number of coordinates per block.
        :type block_size: int
        :param eps: Damping added to each block's diagonal for stability.
        :type eps: float
        """
        super().__init__()
        self.lr = lr
        self.block_size = block_size
        self.eps = eps

    @property
    def name(self) -> str:
        return "Block Newton"

    def step(
        self,
        x: np.ndarray,
        grad_fn: Callable[[np.ndarray], np.ndarray],
        hess: Optional[Callable] = None,
    ) -> np.ndarray:
        if hess is None:
            raise NotImplementedError(
                "BlockNewton requires a Hessian callable via the `hess` argument."
            )
        n = x.shape[0]
        grad = grad_fn(x)
        H = hess(x)
        direction = np.zeros_like(x)

        for start in range(0, n, self.block_size):
            end = min(start + self.block_size, n)
            idx = slice(start, end)
            H_b = H[idx, idx] + self.eps * np.eye(end - start)
            g_b = grad[idx]
            direction[idx] = np.linalg.solve(H_b, g_b)

        self.iterations += 1
        return x - self.lr * direction


# =============================================================================
# Group 4 -- Quasi-Newton methods
# =============================================================================

class DFP(Optimizer):
    """
    DFP -- Davidon-Fletcher-Powell quasi-Newton method.

    Maintains a dense nxn approximation ``B`` to the *inverse* Hessian and
    updates it with a rank-2 correction after each step.

    Inverse Hessian update (from theory/second_order.md):
        s_k = x_{k+1} - x_k
        y_k = gradf(x_{k+1}) - gradf(x_k)

        B_{k+1} = B_k
                  - (B_k y_k y_k^T B_k) / (y_k^T B_k y_k)
                  + (s_k s_k^T) / (y_k^T s_k)

    The search direction is p_k = -B_k gradf(x_k).
    Line search is replaced by a fixed step size ``lr``.

    Memory: O(n^2) -- not suitable for large n.
    """

    def __init__(self, lr: float = 1.0):
        """
        :param lr: Step-size scaling factor applied to the quasi-Newton direction.
        :type lr: float
        """
        super().__init__()
        self.lr = lr
        self.B: Optional[np.ndarray] = None  # inverse Hessian approximation
        self._x_prev: Optional[np.ndarray] = None
        self._g_prev: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        return "DFP"

    def reset(self) -> None:
        super().reset()
        self.B = None
        self._x_prev = None
        self._g_prev = None

    def step(
        self,
        x: np.ndarray,
        grad_fn: Callable[[np.ndarray], np.ndarray],
        hess: Optional[Callable] = None,
    ) -> np.ndarray:
        n = x.shape[0]
        if self.B is None:
            self.B = np.eye(n)

        grad = grad_fn(x)

        if self._x_prev is not None:
            s = x - self._x_prev                    # s_k
            y = grad - self._g_prev                  # y_k
            ys = y @ s                               # y_k^T s_k

            if np.abs(ys) > 1e-10:
                By = self.B @ y
                yBy = y @ By
                # DFP inverse Hessian update
                self.B = (
                    self.B
                    - np.outer(By, By) / yBy
                    + np.outer(s, s) / ys
                )

        direction = self.B @ grad                    # p_k = B_k * g_k
        x_new = x - self.lr * direction

        self._x_prev = x.copy()
        self._g_prev = grad.copy()
        self.iterations += 1
        return x_new


class BFGS(Optimizer):
    """
    BFGS -- Broyden-Fletcher-Goldfarb-Shanno quasi-Newton method.

    Maintains a dense nxn approximation ``H`` to the *inverse* Hessian using
    the symmetric rank-2 update (Sherman-Morrison-Woodbury):

        s_k = x_{k+1} - x_k
        y_k = gradf(x_{k+1}) - gradf(x_k)
        rho_k = 1 / (y_k^T s_k)

        H_{k+1} = (I - rho_k s_k y_k^T) H_k (I - rho_k y_k s_k^T)
                  + rho_k s_k s_k^T

    The search direction is p_k = -H_k gradf(x_k).
    Line search is replaced by a fixed step size ``lr``.

    Memory: O(n^2) -- not suitable for large n.
    """

    def __init__(self, lr: float = 1.0):
        """
        :param lr: Step-size scaling factor applied to the quasi-Newton direction.
        :type lr: float
        """
        super().__init__()
        self.lr = lr
        self.H: Optional[np.ndarray] = None  # inverse Hessian approximation
        self._x_prev: Optional[np.ndarray] = None
        self._g_prev: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        return "BFGS"

    def reset(self) -> None:
        super().reset()
        self.H = None
        self._x_prev = None
        self._g_prev = None

    def step(
        self,
        x: np.ndarray,
        grad_fn: Callable[[np.ndarray], np.ndarray],
        hess: Optional[Callable] = None,
    ) -> np.ndarray:
        n = x.shape[0]
        if self.H is None:
            self.H = np.eye(n)

        grad = grad_fn(x)

        if self._x_prev is not None:
            s = x - self._x_prev                    # s_k
            y = grad - self._g_prev                  # y_k
            ys = y @ s                               # y_k^T s_k

            if np.abs(ys) > 1e-10:
                rho = 1.0 / ys
                I = np.eye(n)
                A = I - rho * np.outer(s, y)
                B = I - rho * np.outer(y, s)
                # BFGS inverse Hessian update
                self.H = A @ self.H @ B + rho * np.outer(s, s)

        direction = self.H @ grad                    # p_k = H_k * g_k
        x_new = x - self.lr * direction

        self._x_prev = x.copy()
        self._g_prev = grad.copy()
        self.iterations += 1
        return x_new


class LBFGS(Optimizer):
    """
    L-BFGS -- Limited-memory BFGS.

    Avoids storing the full nxn inverse Hessian approximation by keeping only
    the last ``history_size`` (s, y) curvature pairs and computing the
    matrix-vector product H_k * g via the two-loop recursion.

    Two-loop recursion (Nocedal & Wright, Algorithm 7.4):
        q <- g_k
        for i = k-1, ..., k-m:
            alpha_i = rho_i * s_i^T q
            q <- q - alpha_i * y_i
        r <- H_0 * q          (H_0 = gamma * I, scaled identity)
        for i = k-m, ..., k-1:
            beta = rho_i * y_i^T r
            r <- r + s_i * (alpha_i - beta)
        p_k = -r

    Memory: O(m * n) where m = history_size (typically 5-20).
    """

    def __init__(self, lr: float = 1.0, history_size: int = 10):
        """
        :param lr: Step-size scaling factor.
        :type lr: float
        :param history_size: Number of (s, y) pairs to retain (m).
        :type history_size: int
        """
        super().__init__()
        self.lr = lr
        self.history_size = history_size
        self._s_list: deque = deque(maxlen=history_size)  # position differences
        self._y_list: deque = deque(maxlen=history_size)  # gradient differences
        self._x_prev: Optional[np.ndarray] = None
        self._g_prev: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        return "L-BFGS"

    def reset(self) -> None:
        super().reset()
        self._s_list.clear()
        self._y_list.clear()
        self._x_prev = None
        self._g_prev = None

    def _two_loop_recursion(self, grad: np.ndarray) -> np.ndarray:
        """Compute H_k * grad via the L-BFGS two-loop recursion."""
        q = grad.copy()
        alphas = []

        # Backward pass
        for s, y in reversed(list(zip(self._s_list, self._y_list))):
            ys = y @ s
            if np.abs(ys) < 1e-10:
                alphas.append(0.0)
                continue
            rho = 1.0 / ys
            alpha = rho * (s @ q)
            q = q - alpha * y
            alphas.append(alpha)

        alphas.reverse()

        # Initial Hessian scaling: H_0 = gamma * I
        if self._s_list:
            s_last = self._s_list[-1]
            y_last = self._y_list[-1]
            gamma = (s_last @ y_last) / (y_last @ y_last + 1e-10)
        else:
            gamma = 1.0
        r = gamma * q

        # Forward pass
        for (s, y), alpha in zip(zip(self._s_list, self._y_list), alphas):
            ys = y @ s
            if np.abs(ys) < 1e-10:
                continue
            rho = 1.0 / ys
            beta = rho * (y @ r)
            r = r + s * (alpha - beta)

        return r

    def step(
        self,
        x: np.ndarray,
        grad_fn: Callable[[np.ndarray], np.ndarray],
        hess: Optional[Callable] = None,
    ) -> np.ndarray:
        grad = grad_fn(x)

        if self._x_prev is not None:
            s = x - self._x_prev
            y = grad - self._g_prev
            if y @ s > 1e-10:          # curvature condition
                self._s_list.append(s.copy())
                self._y_list.append(y.copy())

        direction = self._two_loop_recursion(grad)
        x_new = x - self.lr * direction

        self._x_prev = x.copy()
        self._g_prev = grad.copy()
        self.iterations += 1
        return x_new


class StochasticLBFGS(Optimizer):
    """
    Stochastic / Mini-batch L-BFGS.

    Extends L-BFGS to the stochastic setting by computing gradient differences
    from mini-batch gradient estimates.  To reduce noise in the curvature
    pairs, the overlap correction of Byrd et al. (2016) is approximated by
    using a *larger* batch for the curvature estimation step than for the
    gradient step.

    In practice the caller supplies a ``grad_fn`` that already returns a
    mini-batch gradient.  The two-loop recursion is identical to L-BFGS; the
    only difference is that (s, y) pairs are noisier.

    Memory: O(m * n).
    """

    def __init__(
        self,
        lr: float = 1e-2,
        history_size: int = 10,
        batch_size: int = 32,
        n_samples: int = 1000,
    ):
        """
        :param lr: Step-size scaling factor.
        :type lr: float
        :param history_size: Number of (s, y) curvature pairs to retain.
        :type history_size: int
        :param batch_size: Mini-batch size for gradient estimation.
        :type batch_size: int
        :param n_samples: Total dataset size (used to draw random indices).
        :type n_samples: int
        """
        super().__init__()
        self.lr = lr
        self.history_size = history_size
        self.batch_size = batch_size
        self.n_samples = n_samples
        self._s_list: deque = deque(maxlen=history_size)
        self._y_list: deque = deque(maxlen=history_size)
        self._x_prev: Optional[np.ndarray] = None
        self._g_prev: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        return "Stochastic L-BFGS"

    def reset(self) -> None:
        super().reset()
        self._s_list.clear()
        self._y_list.clear()
        self._x_prev = None
        self._g_prev = None

    def _two_loop_recursion(self, grad: np.ndarray) -> np.ndarray:
        """Identical two-loop recursion as in L-BFGS."""
        q = grad.copy()
        alphas = []

        for s, y in reversed(list(zip(self._s_list, self._y_list))):
            ys = y @ s
            if np.abs(ys) < 1e-10:
                alphas.append(0.0)
                continue
            rho = 1.0 / ys
            alpha = rho * (s @ q)
            q = q - alpha * y
            alphas.append(alpha)

        alphas.reverse()

        if self._s_list:
            s_last = self._s_list[-1]
            y_last = self._y_list[-1]
            gamma = (s_last @ y_last) / (y_last @ y_last + 1e-10)
        else:
            gamma = 1.0
        r = gamma * q

        for (s, y), alpha in zip(zip(self._s_list, self._y_list), alphas):
            ys = y @ s
            if np.abs(ys) < 1e-10:
                continue
            rho = 1.0 / ys
            beta = rho * (y @ r)
            r = r + s * (alpha - beta)

        return r

    def _mini_batch_grad(
        self, x: np.ndarray, grad_fn: Callable
    ) -> np.ndarray:
        """Draw a random mini-batch and compute the gradient estimate."""
        indices = np.random.choice(self.n_samples, size=self.batch_size, replace=False)
        try:
            return grad_fn(x, indices)
        except TypeError:
            return grad_fn(x)

    def step(
        self,
        x: np.ndarray,
        grad_fn: Callable,
        hess: Optional[Callable] = None,
    ) -> np.ndarray:
        grad = self._mini_batch_grad(x, grad_fn)

        if self._x_prev is not None:
            s = x - self._x_prev
            y = grad - self._g_prev
            if y @ s > 1e-10:
                self._s_list.append(s.copy())
                self._y_list.append(y.copy())

        direction = self._two_loop_recursion(grad)
        x_new = x - self.lr * direction

        self._x_prev = x.copy()
        self._g_prev = grad.copy()
        self.iterations += 1
        return x_new


# =============================================================================
# Group 5 -- Gauss-Newton with CG and Hessian-Free
# =============================================================================

def _conjugate_gradient(
    matvec: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    max_iters: int = 50,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Solve the linear system A x = b using the Conjugate Gradient method,
    where A is given implicitly via a matrix-vector product callable.

    :param matvec: Callable that computes A @ v for a given vector v.
    :param b: Right-hand side vector.
    :param max_iters: Maximum number of CG iterations.
    :param tol: Convergence tolerance on the residual norm.
    :returns: Approximate solution x to A x = b.
    """
    x = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    r_dot = r @ r

    for _ in range(max_iters):
        if np.sqrt(r_dot) < tol:
            break
        Ap = matvec(p)
        alpha = r_dot / (p @ Ap + 1e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        r_dot_new = r @ r
        beta = r_dot_new / (r_dot + 1e-30)
        p = r + beta * p
        r_dot = r_dot_new

    return x


class GaussNewtonCG(Optimizer):
    """
    Gauss-Newton method with Conjugate Gradient inner solver.

    Approximates the Hessian as J^T J (valid for least-squares objectives)
    and solves the resulting linear system with CG to avoid forming J^T J
    explicitly.

    The caller must supply a ``hess`` argument that is actually a *Jacobian*
    callable ``jac_fn(x) -> np.ndarray`` of shape (m, n), where m is the
    number of residuals and n is the number of parameters.  The Gauss-Newton
    matrix-vector product is then computed as:

        (J^T J + damping * I) v  =  J^T (J v)

    which requires only two matrix-vector products and avoids forming J^T J.

    If ``hess`` is None, the method falls back to using the full gradient as
    a steepest-descent step (i.e. CG with identity preconditioner).

    Update rule:
        Solve  (J^T J + damping * I) p = -g   via CG
        x_{k+1} = x_k + lr * p
    """

    def __init__(self, lr: float = 1.0, cg_iters: int = 50, damping: float = 1e-4):
        """
        :param lr: Step-size scaling factor applied to the CG solution.
        :type lr: float
        :param cg_iters: Maximum number of CG iterations for the inner solve.
        :type cg_iters: int
        :param damping: Tikhonov damping added to J^T J for numerical stability.
        :type damping: float
        """
        super().__init__()
        self.lr = lr
        self.cg_iters = cg_iters
        self.damping = damping

    @property
    def name(self) -> str:
        return "Gauss-Newton CG"

    def step(
        self,
        x: np.ndarray,
        grad_fn: Callable[[np.ndarray], np.ndarray],
        hess: Optional[Callable] = None,
    ) -> np.ndarray:
        grad = grad_fn(x)

        if hess is not None:
            H_or_J = hess(x)
            if H_or_J.ndim == 2 and H_or_J.shape[0] == H_or_J.shape[1]:
                # Square matrix: treat as Hessian (not Jacobian).
                # Use it directly as the system matrix for the Newton-CG step.
                H = H_or_J
                def gn_matvec(v: np.ndarray) -> np.ndarray:
                    return H @ v + self.damping * v
            else:
                # Non-square: treat as Jacobian J of shape (m, n).
                J = H_or_J
                if J.ndim == 1:
                    J = J.reshape(1, -1)
                def gn_matvec(v: np.ndarray) -> np.ndarray:
                    return J.T @ (J @ v) + self.damping * v

            direction = _conjugate_gradient(gn_matvec, grad, max_iters=self.cg_iters)
        else:
            # Fallback: steepest descent (GN with identity Hessian approx)
            direction = grad

        self.iterations += 1
        return x - self.lr * direction


class HessianFree(Optimizer):
    """
    Hessian-Free optimisation (Truncated Newton with CG).

    Avoids forming or storing the Hessian by computing Hessian-vector products
    (HVPs) via a finite-difference approximation of the gradient:

        H(x) v  ~=  [gradf(x + eps v) - gradf(x)] / eps

    This requires only two gradient evaluations per CG iteration and O(n)
    memory.  The linear system

        (H(x_k) + damping * I) p = -gradf(x_k)

    is solved approximately with ``cg_iters`` steps of CG, giving the
    truncated Newton direction p_k.

    Update rule:
        Solve  (H + damping * I) p = -g   via CG with HVP oracle
        x_{k+1} = x_k + lr * p
    """

    def __init__(
        self,
        lr: float = 1.0,
        cg_iters: int = 50,
        damping: float = 1e-4,
        fd_eps: float = 1e-5,
    ):
        """
        :param lr: Step-size scaling factor applied to the CG solution.
        :type lr: float
        :param cg_iters: Maximum number of CG iterations for the inner solve.
        :type cg_iters: int
        :param damping: Tikhonov damping added to the Hessian for stability.
        :type damping: float
        :param fd_eps: Finite-difference step size for HVP approximation.
        :type fd_eps: float
        """
        super().__init__()
        self.lr = lr
        self.cg_iters = cg_iters
        self.damping = damping
        self.fd_eps = fd_eps

    @property
    def name(self) -> str:
        return "Hessian-Free"

    def _hvp(
        self,
        x: np.ndarray,
        v: np.ndarray,
        grad_fn: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """
        Finite-difference Hessian-vector product:
            H(x) v  ~=  [gradf(x + eps v) - gradf(x)] / eps
        """
        return (grad_fn(x + self.fd_eps * v) - grad_fn(x)) / self.fd_eps

    def step(
        self,
        x: np.ndarray,
        grad_fn: Callable[[np.ndarray], np.ndarray],
        hess: Optional[Callable] = None,
    ) -> np.ndarray:
        grad = grad_fn(x)

        def matvec(v: np.ndarray) -> np.ndarray:
            if hess is not None:
                # Use exact Hessian if available
                H = hess(x)
                return H @ v + self.damping * v
            else:
                # Finite-difference HVP
                return self._hvp(x, v, grad_fn) + self.damping * v

        direction = _conjugate_gradient(matvec, grad, max_iters=self.cg_iters)
        self.iterations += 1
        return x - self.lr * direction


# =============================================================================
# Group 6 -- Mini-batch L-BFGS
# (distinct from StochasticLBFGS: uses a *separate*, larger overlap batch
#  for curvature estimation to reduce noise in (s, y) pairs)
# =============================================================================

class MiniBatchLBFGS(Optimizer):
    """
    Mini-Batch L-BFGS with overlap-corrected curvature pairs.

    Uses two independent mini-batches per step:
      - ``batch_size`` samples for the gradient step direction.
      - ``curvature_batch_size`` samples (>= batch_size) for estimating the
        gradient difference y_k = g(x_{k+1}) - g(x_k), reducing curvature
        noise compared to plain StochasticLBFGS.

    This mirrors the approach of Bollapragada et al. (2018) and is the
    standard "mini-batch L-BFGS" variant referenced in the diploma spec.

    Memory: O(m * n).
    """

    def __init__(
        self,
        lr: float = 1e-2,
        history_size: int = 10,
        batch_size: int = 32,
        curvature_batch_size: int = 128,
        n_samples: int = 1000,
    ):
        """
        :param lr: Step-size scaling factor.
        :type lr: float
        :param history_size: Number of (s, y) curvature pairs to retain (m).
        :type history_size: int
        :param batch_size: Mini-batch size for the gradient / search direction.
        :type batch_size: int
        :param curvature_batch_size: Larger batch used to estimate y_k.
        :type curvature_batch_size: int
        :param n_samples: Total dataset size (used to draw random indices).
        :type n_samples: int
        """
        super().__init__()
        self.lr = lr
        self.history_size = history_size
        self.batch_size = batch_size
        self.curvature_batch_size = curvature_batch_size
        self.n_samples = n_samples
        self._s_list: deque = deque(maxlen=history_size)
        self._y_list: deque = deque(maxlen=history_size)
        self._x_prev: Optional[np.ndarray] = None
        self._g_curv_prev: Optional[np.ndarray] = None  # curvature-batch grad at x_prev

    @property
    def name(self) -> str:
        return "Mini-Batch L-BFGS"

    def reset(self) -> None:
        super().reset()
        self._s_list.clear()
        self._y_list.clear()
        self._x_prev = None
        self._g_curv_prev = None

    def _sample_grad(self, x: np.ndarray, grad_fn: Callable, size: int) -> np.ndarray:
        indices = np.random.choice(self.n_samples, size=size, replace=False)
        try:
            return grad_fn(x, indices)
        except TypeError:
            return grad_fn(x)

    def _two_loop_recursion(self, grad: np.ndarray) -> np.ndarray:
        q = grad.copy()
        alphas = []
        for s, y in reversed(list(zip(self._s_list, self._y_list))):
            ys = y @ s
            if np.abs(ys) < 1e-10:
                alphas.append(0.0)
                continue
            alpha = (s @ q) / ys
            q = q - alpha * y
            alphas.append(alpha)
        alphas.reverse()

        if self._s_list:
            s_last, y_last = self._s_list[-1], self._y_list[-1]
            gamma = (s_last @ y_last) / (y_last @ y_last + 1e-10)
        else:
            gamma = 1.0
        r = gamma * q

        for (s, y), alpha in zip(zip(self._s_list, self._y_list), alphas):
            ys = y @ s
            if np.abs(ys) < 1e-10:
                continue
            beta = (y @ r) / ys
            r = r + s * (alpha - beta)
        return r

    def step(
        self,
        x: np.ndarray,
        grad_fn: Callable,
        hess: Optional[Callable] = None,
    ) -> np.ndarray:
        # Gradient for search direction (small batch)
        g_step = self._sample_grad(x, grad_fn, self.batch_size)
        # Gradient for curvature estimation (large batch)
        g_curv = self._sample_grad(x, grad_fn, self.curvature_batch_size)

        if self._x_prev is not None and self._g_curv_prev is not None:
            s = x - self._x_prev
            y = g_curv - self._g_curv_prev
            if y @ s > 1e-10:
                self._s_list.append(s.copy())
                self._y_list.append(y.copy())

        direction = self._two_loop_recursion(g_step)
        x_new = x - self.lr * direction

        self._x_prev = x.copy()
        self._g_curv_prev = g_curv.copy()
        self.iterations += 1
        return x_new
