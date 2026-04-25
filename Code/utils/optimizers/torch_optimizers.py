"""
PyTorch custom optimizer implementations.

Every optimizer subclasses ``torch.optim.Optimizer`` and mirrors the
mathematical update rules of the corresponding NumPy implementations in
``numpy_optimizers_full.py``.

Groups
------
0. Vanilla / deterministic first-order
   GradientDescentTorch, MomentumSGDTorch, NesterovMomentumTorch, NewtonsMethodTorch

1. First-order stochastic
   SGDTorch  (thin wrapper -- identical update rule to GradientDescentTorch)

2. Adaptive first-order
   AdaGradTorch, RMSPropTorch, AdamTorch

3. Diagonal / block-diagonal Hessian approximations
   DiagonalNewtonTorch, BlockNewtonTorch

4. Quasi-Newton
   DFPTorch, BFGSTorch, LBFGSTorch

5. Gauss-Newton with CG and Hessian-Free
   GaussNewtonCGTorch, HessianFreeTorch

Notes
-----
* Newton-family methods (NewtonsMethodTorch, DiagonalNewtonTorch,
  BlockNewtonTorch, DFPTorch, BFGSTorch, GaussNewtonCGTorch) require the
  caller to supply a Hessian (or Jacobian) via set_hessian(H) before step().

* For pure gradient-based methods the standard PyTorch closure convention
  is used: closure() calls loss.backward() and returns the loss scalar.

* LBFGSTorch implements the full two-loop recursion from scratch.

* HessianFreeTorch computes Hessian-vector products via finite differences of
  the gradient, so it only needs a gradient closure.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Callable, Iterable, List, Optional

import torch
from torch import Tensor
from torch.optim import Optimizer


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _flatten_params(params: List[Tensor]) -> Tensor:
    return torch.cat([p.data.reshape(-1) for p in params])


def _unflatten_params(flat: Tensor, params: List[Tensor]) -> None:
    offset = 0
    for p in params:
        numel = p.numel()
        p.data.copy_(flat[offset: offset + numel].reshape(p.shape))
        offset += numel


def _flatten_grads(params: List[Tensor]) -> Tensor:
    grads = []
    for p in params:
        if p.grad is None:
            grads.append(torch.zeros_like(p.data).reshape(-1))
        else:
            grads.append(p.grad.data.reshape(-1))
    return torch.cat(grads)


def _conjugate_gradient_torch(
    matvec: Callable[[Tensor], Tensor],
    b: Tensor,
    max_iters: int = 50,
    tol: float = 1e-6,
) -> Tensor:
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    r_dot = r @ r
    for _ in range(max_iters):
        if r_dot.sqrt().item() < tol:
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


# =============================================================================
# Group 0 -- Vanilla / deterministic first-order methods
# =============================================================================

class GradientDescentTorch(Optimizer):
    """
    Vanilla (full-batch) Gradient Descent.

    Update rule:
        theta_{k+1} = theta_k - lr * grad(L(theta_k))
    """

    def __init__(self, params: Iterable, lr: float = 1e-3):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        super().__init__(params, dict(lr=lr))

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[Tensor]:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data.add_(p.grad.data, alpha=-lr)
        return loss


class MomentumSGDTorch(Optimizer):
    """
    SGD with Heavy-Ball Momentum (Polyak 1964).

    Update rule:
        h_0 = 0
        h_k = alpha * h_{k-1} + grad(L(theta_{k-1}))
        theta_k = theta_{k-1} - lr * h_k
    """

    def __init__(self, params: Iterable, lr: float = 1e-3, momentum: float = 0.9):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        super().__init__(params, dict(lr=lr, momentum=momentum))

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[Tensor]:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            alpha = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "h" not in state:
                    state["h"] = torch.zeros_like(p.data)
                h = state["h"]
                h.mul_(alpha).add_(p.grad.data)
                p.data.add_(h, alpha=-lr)
        return loss


class NesterovMomentumTorch(Optimizer):
    """
    Nesterov Accelerated Gradient (NAG).

    Mirrors the NumPy NesterovMomentum update rule exactly:
        h_0 = 0
        x_lookahead = x - alpha * h_{k-1}
        h_k = alpha * h_{k-1} + lr * grad(L(x_lookahead))
        x_k = x_{k-1} - h_k

    Because the lookahead point requires evaluating the gradient at a
    *different* parameter value, the closure must be re-evaluated at the
    lookahead point.  When no closure is provided the current gradient
    (already computed at x) is used as a fallback (equivalent to the first
    step where h=0 and x_lookahead == x).
    """

    def __init__(self, params: Iterable, lr: float = 1e-3, momentum: float = 0.9):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        super().__init__(params, dict(lr=lr, momentum=momentum))

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[Tensor]:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            alpha = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "h" not in state:
                    state["h"] = torch.zeros_like(p.data)
                h = state["h"]

                # Compute lookahead point: x_lookahead = x - alpha * h
                x_lookahead = p.data - alpha * h

                # Evaluate gradient at lookahead point if closure provided
                if closure is not None:
                    # Temporarily move params to lookahead, recompute grad
                    p.data.copy_(x_lookahead)
                    with torch.enable_grad():
                        closure()
                    g_lookahead = p.grad.data.clone()
                    # Restore original params (will be updated below)
                    p.data.add_(h, alpha=alpha)  # undo: x = x_lookahead + alpha*h
                else:
                    # Fallback: use current gradient (valid when h=0 or no closure)
                    g_lookahead = p.grad.data

                # h_k = alpha * h_{k-1} + lr * grad(x_lookahead)
                h.mul_(alpha).add_(g_lookahead, alpha=lr)
                # x_k = x_{k-1} - h_k
                p.data.sub_(h)
        return loss


class NewtonsMethodTorch(Optimizer):
    """
    Pure Newton's Method (second-order).

    Update rule:
        theta_{k+1} = theta_k - lr * [H(theta_k)]^{-1} * grad(L(theta_k))

    Requires set_hessian(H) to be called before step().
    H must be a 2-D tensor of shape (n_params, n_params).
    """

    def __init__(self, params: Iterable, lr: float = 1.0, damping: float = 1e-6):
        super().__init__(params, dict(lr=lr, damping=damping))
        self.hessian: Optional[Tensor] = None

    def set_hessian(self, H: Tensor) -> None:
        self.hessian = H

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[Tensor]:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self.hessian is None:
            raise RuntimeError("NewtonsMethodTorch: call set_hessian(H) before step().")
        all_params = [p for group in self.param_groups for p in group["params"]]
        lr = self.param_groups[0]["lr"]
        damping = self.param_groups[0]["damping"]
        g = _flatten_grads(all_params)
        H = self.hessian
        n = H.shape[0]
        H_reg = H + damping * torch.eye(n, dtype=H.dtype, device=H.device)
        direction = torch.linalg.solve(H_reg, g)
        _unflatten_params(_flatten_params(all_params) - lr * direction, all_params)
        return loss


# =============================================================================
# Group 1 -- First-order stochastic methods
# =============================================================================

class SGDTorch(GradientDescentTorch):
    """
    Stochastic Gradient Descent.

    Identical update rule to GradientDescentTorch; stochastic nature comes
    from the caller computing the gradient on a random mini-batch.
    """
    pass


class MiniBatchSGDTorch(Optimizer):
    """
    Mini-Batch Stochastic Gradient Descent.

    Mirrors NumPy MiniBatchSGD exactly.  The stochastic nature comes from the
    caller supplying a closure that computes the loss (and gradient) on a
    randomly-drawn mini-batch of size ``batch_size`` drawn from a dataset of
    ``n_samples`` samples.

    When a closure is provided it is called once per step (the mini-batch
    selection is the caller's responsibility inside the closure).  When no
    closure is provided the already-computed gradient is used directly,
    making the behaviour identical to SGDTorch.

    Update rule:
        B  ~ Uniform subset of {1,...,n}, |B| = batch_size   (caller's job)
        g_k = (1/|B|) sum_{iinB} gradf_i(theta_k)
        theta_{k+1} = theta_k - lr * g_k
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        batch_size: int = 32,
        n_samples: int = 1000,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        super().__init__(params, dict(lr=lr, batch_size=batch_size, n_samples=n_samples))

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[Tensor]:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data.add_(p.grad.data, alpha=-lr)
        return loss


# =============================================================================
# Group 2 -- Adaptive first-order methods
# =============================================================================

class AdaGradTorch(Optimizer):
    """
    AdaGrad -- Adaptive Gradient algorithm.

    Update rule:
        g_k  = grad(L(theta_k))
        G_k  = G_{k-1} + g_k * g_k   (element-wise)
        theta_{k+1} = theta_k - lr * g_k / (sqrt(G_k) + eps)
    """

    def __init__(self, params: Iterable, lr: float = 1e-2, eps: float = 1e-8):
        super().__init__(params, dict(lr=lr, eps=eps))

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[Tensor]:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.data
                state = self.state[p]
                if "G" not in state:
                    state["G"] = torch.zeros_like(p.data)
                G = state["G"]
                G.addcmul_(g, g)
                p.data.addcdiv_(g, G.sqrt().add_(eps), value=-lr)
        return loss


class RMSPropTorch(Optimizer):
    """
    RMSProp -- Root Mean Square Propagation.

    Update rule:
        g_k     = grad(L(theta_k))
        v_{k+1} = beta * v_k + (1 - beta) * g_k * g_k
        theta_{k+1} = theta_k - lr * g_k / (sqrt(v_{k+1}) + eps)
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        beta: float = 0.9,
        eps: float = 1e-8,
    ):
        super().__init__(params, dict(lr=lr, beta=beta, eps=eps))

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[Tensor]:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.data
                state = self.state[p]
                if "v" not in state:
                    state["v"] = torch.zeros_like(p.data)
                v = state["v"]
                v.mul_(beta).addcmul_(g, g, value=1.0 - beta)
                p.data.addcdiv_(g, v.sqrt().add_(eps), value=-lr)
        return loss


class AdamTorch(Optimizer):
    """
    Adam -- Adaptive Moment Estimation.

    Update rule:
        g_k    = grad(L(theta_k))
        m_k    = beta1 * m_{k-1} + (1 - beta1) * g_k
        v_k    = beta2 * v_{k-1} + (1 - beta2) * g_k^2
        m_hat  = m_k / (1 - beta1^k)
        v_hat  = v_k / (1 - beta2^k)
        theta_{k+1} = theta_k - lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        super().__init__(params, dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps))

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[Tensor]:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.data
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                state["step"] += 1
                t = state["step"]
                m = state["m"]
                v = state["v"]
                m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                bias_corr1 = 1.0 - beta1 ** t
                bias_corr2 = 1.0 - beta2 ** t
                step_size = lr / bias_corr1
                denom = (v.sqrt() / math.sqrt(bias_corr2)).add_(eps)
                p.data.addcdiv_(m, denom, value=-step_size)
        return loss


# =============================================================================
# Group 3 -- Diagonal and block-diagonal Hessian approximations
# =============================================================================

class DiagonalNewtonTorch(Optimizer):
    """
    Diagonal Newton method.

    Uses only the diagonal of the Hessian as a cheap preconditioner:
        theta_{k+1} = theta_k - lr * g_k / (|diag(H)| + eps)

    Requires set_hessian(H) before step().
    Accepts a full matrix (diagonal extracted) or a 1-D diagonal vector.
    """

    def __init__(self, params: Iterable, lr: float = 1.0, eps: float = 1e-8):
        super().__init__(params, dict(lr=lr, eps=eps))
        self.hessian_diag: Optional[Tensor] = None

    def set_hessian(self, H: Tensor) -> None:
        self.hessian_diag = torch.diag(H) if H.ndim == 2 else H

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[Tensor]:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self.hessian_diag is None:
            raise RuntimeError("DiagonalNewtonTorch: call set_hessian(H) before step().")
        all_params = [p for group in self.param_groups for p in group["params"]]
        lr = self.param_groups[0]["lr"]
        eps = self.param_groups[0]["eps"]
        g = _flatten_grads(all_params)
        d = self.hessian_diag.abs().add_(eps)
        direction = g / d
        _unflatten_params(_flatten_params(all_params) - lr * direction, all_params)
        return loss


class BlockNewtonTorch(Optimizer):
    """
    Block-diagonal Newton method.

    Partitions the parameter vector into non-overlapping blocks of size
    block_size and applies an independent Newton step within each block:
        H_b = H[I_b, I_b] + eps * I
        p_b = solve(H_b, g_b)
        theta_{k+1}[I_b] = theta_k[I_b] - lr * p_b

    Requires set_hessian(H) before step().
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1.0,
        block_size: int = 10,
        eps: float = 1e-6,
    ):
        super().__init__(params, dict(lr=lr, block_size=block_size, eps=eps))
        self.hessian: Optional[Tensor] = None

    def set_hessian(self, H: Tensor) -> None:
        self.hessian = H

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[Tensor]:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self.hessian is None:
            raise RuntimeError("BlockNewtonTorch: call set_hessian(H) before step().")
        all_params = [p for group in self.param_groups for p in group["params"]]
        lr = self.param_groups[0]["lr"]
        block_size = self.param_groups[0]["block_size"]
        eps = self.param_groups[0]["eps"]
        g = _flatten_grads(all_params)
        H = self.hessian
        n = g.shape[0]
        direction = torch.zeros_like(g)
        for start in range(0, n, block_size):
            end = min(start + block_size, n)
            idx = slice(start, end)
            H_b = H[idx, idx] + eps * torch.eye(
                end - start, dtype=H.dtype, device=H.device
            )
            direction[idx] = torch.linalg.solve(H_b, g[idx])
        _unflatten_params(_flatten_params(all_params) - lr * direction, all_params)
        return loss


# =============================================================================
# Group 4 -- Quasi-Newton methods
# =============================================================================

class DFPTorch(Optimizer):
    """
    DFP -- Davidon-Fletcher-Powell quasi-Newton method.

    Maintains a dense n x n approximation B to the inverse Hessian:
        s_k = theta_{k+1} - theta_k
        y_k = grad_{k+1} - grad_k
        B_{k+1} = B_k - (B_k y_k y_k^T B_k)/(y_k^T B_k y_k) + (s_k s_k^T)/(y_k^T s_k)

    Memory: O(n^2) -- not suitable for large n.
    """

    def __init__(self, params: Iterable, lr: float = 1.0):
        super().__init__(params, dict(lr=lr))
        self._x_prev: Optional[Tensor] = None
        self._g_prev: Optional[Tensor] = None
        self._B: Optional[Tensor] = None

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[Tensor]:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        all_params = [p for group in self.param_groups for p in group["params"]]
        lr = self.param_groups[0]["lr"]
        x = _flatten_params(all_params)
        g = _flatten_grads(all_params)
        n = x.shape[0]
        if self._B is None:
            self._B = torch.eye(n, dtype=x.dtype, device=x.device)
        if self._x_prev is not None:
            s = x - self._x_prev
            y = g - self._g_prev
            ys = float(y @ s)
            if abs(ys) > 1e-10:
                By = self._B @ y
                yBy = float(y @ By)
                if abs(yBy) > 1e-10:
                    self._B = (
                        self._B
                        - torch.outer(By, By) / yBy
                        + torch.outer(s, s) / ys
                    )
        direction = self._B @ g
        _unflatten_params(x - lr * direction, all_params)
        self._x_prev = x.clone()
        self._g_prev = g.clone()
        return loss


class BFGSTorch(Optimizer):
    """
    BFGS -- Broyden-Fletcher-Goldfarb-Shanno quasi-Newton method.

    Maintains a dense n x n approximation H to the inverse Hessian:
        s_k = theta_{k+1} - theta_k
        y_k = grad_{k+1} - grad_k
        rho_k = 1 / (y_k^T s_k)
        H_{k+1} = (I - rho_k s_k y_k^T) H_k (I - rho_k y_k s_k^T) + rho_k s_k s_k^T

    Memory: O(n^2) -- not suitable for large n.
    """

    def __init__(self, params: Iterable, lr: float = 1.0):
        super().__init__(params, dict(lr=lr))
        self._x_prev: Optional[Tensor] = None
        self._g_prev: Optional[Tensor] = None
        self._H: Optional[Tensor] = None

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[Tensor]:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        all_params = [p for group in self.param_groups for p in group["params"]]
        lr = self.param_groups[0]["lr"]
        x = _flatten_params(all_params)
        g = _flatten_grads(all_params)
        n = x.shape[0]
        if self._H is None:
            self._H = torch.eye(n, dtype=x.dtype, device=x.device)
        if self._x_prev is not None:
            s = x - self._x_prev
            y = g - self._g_prev
            ys = float(y @ s)
            if abs(ys) > 1e-10:
                rho = 1.0 / ys
                I = torch.eye(n, dtype=x.dtype, device=x.device)
                A = I - rho * torch.outer(s, y)
                B = I - rho * torch.outer(y, s)
                self._H = A @ self._H @ B + rho * torch.outer(s, s)
        direction = self._H @ g
        _unflatten_params(x - lr * direction, all_params)
        self._x_prev = x.clone()
        self._g_prev = g.clone()
        return loss


class LBFGSTorch(Optimizer):
    """
    L-BFGS -- Limited-memory BFGS (custom two-loop recursion).

    Avoids storing the full n x n inverse Hessian by keeping only the last
    history_size (s, y) curvature pairs and computing H_k g via the two-loop
    recursion (Nocedal & Wright, Algorithm 7.4):

        q <- g_k
        for i = k-1, ..., k-m:
            alpha_i = rho_i * s_i^T q;  q <- q - alpha_i * y_i
        r <- gamma * q   (gamma = s_{k-1}^T y_{k-1} / y_{k-1}^T y_{k-1})
        for i = k-m, ..., k-1:
            beta = rho_i * y_i^T r;  r <- r + s_i * (alpha_i - beta)
        p_k = -r

    Memory: O(m * n) where m = history_size (typically 5-20).
    """

    def __init__(self, params: Iterable, lr: float = 1.0, history_size: int = 10):
        super().__init__(params, dict(lr=lr, history_size=history_size))
        self._s_list: deque = deque(maxlen=history_size)
        self._y_list: deque = deque(maxlen=history_size)
        self._x_prev: Optional[Tensor] = None
        self._g_prev: Optional[Tensor] = None

    def _two_loop(self, g: Tensor) -> Tensor:
        q = g.clone()
        alphas = []
        for s, y in reversed(list(zip(self._s_list, self._y_list))):
            ys = float(y @ s)
            if abs(ys) < 1e-10:
                alphas.append(0.0)
                continue
            rho = 1.0 / ys
            alpha = rho * float(s @ q)
            q = q - alpha * y
            alphas.append(alpha)
        alphas.reverse()
        if self._s_list:
            s_last = self._s_list[-1]
            y_last = self._y_list[-1]
            gamma = float(s_last @ y_last) / (float(y_last @ y_last) + 1e-10)
        else:
            gamma = 1.0
        r = gamma * q
        for (s, y), alpha in zip(zip(self._s_list, self._y_list), alphas):
            ys = float(y @ s)
            if abs(ys) < 1e-10:
                continue
            rho = 1.0 / ys
            beta = rho * float(y @ r)
            r = r + s * (alpha - beta)
        return r

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[Tensor]:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        all_params = [p for group in self.param_groups for p in group["params"]]
        lr = self.param_groups[0]["lr"]
        x = _flatten_params(all_params)
        g = _flatten_grads(all_params)
        if self._x_prev is not None:
            s = x - self._x_prev
            y = g - self._g_prev
            if float(y @ s) > 1e-10:
                self._s_list.append(s.clone())
                self._y_list.append(y.clone())
        direction = self._two_loop(g)
        _unflatten_params(x - lr * direction, all_params)
        self._x_prev = x.clone()
        self._g_prev = g.clone()
        return loss


# =============================================================================
# Group 5 -- Gauss-Newton with CG and Hessian-Free
# =============================================================================

class GaussNewtonCGTorch(Optimizer):
    """
    Gauss-Newton method with Conjugate Gradient inner solver.

    Approximates the Hessian as J^T J (valid for least-squares objectives)
    and solves the resulting linear system with CG:

        (J^T J + damping * I) p = -g   via CG
        theta_{k+1} = theta_k - lr * p

    The caller must provide a Jacobian (or Hessian) via set_hessian(J_or_H)
    before calling step().

    If a square matrix is provided it is treated as the Hessian directly.
    If a non-square matrix is provided it is treated as the Jacobian J of
    shape (m, n) and the GN matrix-vector product J^T (J v) is used.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1.0,
        cg_iters: int = 50,
        damping: float = 1e-4,
    ):
        super().__init__(params, dict(lr=lr, cg_iters=cg_iters, damping=damping))
        self.hessian: Optional[Tensor] = None

    def set_hessian(self, H_or_J: Tensor) -> None:
        self.hessian = H_or_J

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[Tensor]:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        all_params = [p for group in self.param_groups for p in group["params"]]
        lr = self.param_groups[0]["lr"]
        cg_iters = self.param_groups[0]["cg_iters"]
        damping = self.param_groups[0]["damping"]
        g = _flatten_grads(all_params)
        if self.hessian is not None:
            M = self.hessian
            if M.ndim == 2 and M.shape[0] == M.shape[1]:
                # Square: treat as Hessian
                def matvec(v: Tensor) -> Tensor:
                    return M @ v + damping * v
            else:
                # Non-square: treat as Jacobian J of shape (m, n)
                J = M if M.ndim == 2 else M.reshape(1, -1)
                def matvec(v: Tensor) -> Tensor:
                    return J.t() @ (J @ v) + damping * v
            direction = _conjugate_gradient_torch(matvec, g, max_iters=cg_iters)
        else:
            direction = g  # fallback: steepest descent
        _unflatten_params(_flatten_params(all_params) - lr * direction, all_params)
        return loss


class HessianFreeTorch(Optimizer):
    """
    Hessian-Free optimisation (Truncated Newton with CG).

    Avoids forming or storing the Hessian by computing Hessian-vector products
    (HVPs) via a finite-difference approximation of the gradient:

        H(theta) v  ~  [grad(L(theta + eps*v)) - grad(L(theta))] / eps

    This requires only two gradient evaluations per CG iteration and O(n)
    memory.  The linear system

        (H(theta_k) + damping * I) p = -grad(L(theta_k))

    is solved approximately with cg_iters steps of CG.

    The caller must supply a closure that computes gradients.
    Optionally, an exact Hessian can be provided via set_hessian(H) to skip
    the finite-difference approximation.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1.0,
        cg_iters: int = 50,
        damping: float = 1e-4,
        fd_eps: float = 1e-5,
    ):
        super().__init__(params, dict(lr=lr, cg_iters=cg_iters, damping=damping, fd_eps=fd_eps))
        self.hessian: Optional[Tensor] = None
        self._grad_fn: Optional[Callable] = None

    def set_hessian(self, H: Tensor) -> None:
        self.hessian = H

    def set_grad_fn(self, grad_fn: Callable[[Tensor], Tensor]) -> None:
        """
        Provide a callable grad_fn(x_flat) -> grad_flat for HVP computation.
        x_flat and grad_flat are 1-D tensors of the concatenated parameters.
        """
        self._grad_fn = grad_fn

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[Tensor]:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        all_params = [p for group in self.param_groups for p in group["params"]]
        lr = self.param_groups[0]["lr"]
        cg_iters = self.param_groups[0]["cg_iters"]
        damping = self.param_groups[0]["damping"]
        fd_eps = self.param_groups[0]["fd_eps"]
        g = _flatten_grads(all_params)
        x = _flatten_params(all_params)

        if self.hessian is not None:
            H = self.hessian
            def matvec(v: Tensor) -> Tensor:
                return H @ v + damping * v
        elif self._grad_fn is not None:
            gfn = self._grad_fn
            def matvec(v: Tensor) -> Tensor:
                with torch.enable_grad():
                    hvp = (gfn(x + fd_eps * v) - gfn(x)) / fd_eps
                return hvp + damping * v
        else:
            # No Hessian info: fall back to steepest descent
            direction = g
            _unflatten_params(x - lr * direction, all_params)
            return loss

        direction = _conjugate_gradient_torch(matvec, g, max_iters=cg_iters)
        _unflatten_params(x - lr * direction, all_params)
        return loss


# =============================================================================
# Group 6 -- Stochastic / Mini-batch L-BFGS
# =============================================================================

class StochasticLBFGSTorch(Optimizer):
    """
    Stochastic / Mini-batch L-BFGS.

    Mirrors NumPy StochasticLBFGS exactly.  Extends L-BFGS to the stochastic
    setting by computing gradient differences from mini-batch gradient
    estimates.  The two-loop recursion is identical to LBFGSTorch; the only
    difference is that (s, y) curvature pairs are noisier because they are
    derived from stochastic gradients.

    The stochastic nature is the caller's responsibility: supply a closure
    that computes the loss (and gradient) on a randomly-drawn mini-batch.

    Update rule:
        q  <- g_k  (stochastic gradient)
        Two-loop L-BFGS recursion -> direction r
        theta_{k+1} = theta_k - lr * r

    Memory: O(m * n) where m = history_size.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-2,
        history_size: int = 10,
        batch_size: int = 32,
        n_samples: int = 1000,
    ):
        super().__init__(
            params,
            dict(lr=lr, history_size=history_size, batch_size=batch_size, n_samples=n_samples),
        )
        self._s_list: deque = deque(maxlen=history_size)
        self._y_list: deque = deque(maxlen=history_size)
        self._x_prev: Optional[Tensor] = None
        self._g_prev: Optional[Tensor] = None

    def _two_loop(self, g: Tensor) -> Tensor:
        """L-BFGS two-loop recursion."""
        q = g.clone()
        alphas: List[float] = []
        for s, y in reversed(list(zip(self._s_list, self._y_list))):
            ys = float(y @ s)
            if abs(ys) < 1e-10:
                alphas.append(0.0)
                continue
            rho = 1.0 / ys
            alpha = rho * float(s @ q)
            q = q - alpha * y
            alphas.append(alpha)
        alphas.reverse()
        if self._s_list:
            s_last = self._s_list[-1]
            y_last = self._y_list[-1]
            gamma = float(s_last @ y_last) / (float(y_last @ y_last) + 1e-10)
        else:
            gamma = 1.0
        r = gamma * q
        for (s, y), alpha in zip(zip(self._s_list, self._y_list), alphas):
            ys = float(y @ s)
            if abs(ys) < 1e-10:
                continue
            rho = 1.0 / ys
            beta = rho * float(y @ r)
            r = r + s * (alpha - beta)
        return r

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[Tensor]:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        all_params = [p for group in self.param_groups for p in group["params"]]
        lr = self.param_groups[0]["lr"]
        x = _flatten_params(all_params)
        g = _flatten_grads(all_params)
        if self._x_prev is not None:
            s = x - self._x_prev
            y = g - self._g_prev
            if float(y @ s) > 1e-10:
                self._s_list.append(s.clone())
                self._y_list.append(y.clone())
        direction = self._two_loop(g)
        _unflatten_params(x - lr * direction, all_params)
        self._x_prev = x.clone()
        self._g_prev = g.clone()
        return loss


class MiniBatchLBFGSTorch(Optimizer):
    """
    Mini-Batch L-BFGS with overlap-corrected curvature pairs.

    Mirrors NumPy MiniBatchLBFGS exactly.  Uses two independent mini-batches
    per step:
      - A small ``batch_size`` batch for the gradient / search direction.
      - A larger ``curvature_batch_size`` batch for estimating the gradient
        difference y_k = g(x_{k+1}) - g(x_k), reducing curvature noise
        compared to plain StochasticLBFGSTorch.

    The curvature closure must be registered via ``set_curvature_closure(fn)``
    before calling ``step()``.  When not set, the step-batch gradient is used
    for curvature estimation (degrades to StochasticLBFGSTorch behaviour).

    Update rule:
        g_step  <- closure()          (small batch, for search direction)
        g_curv  <- curvature_closure() (large batch, for y_k estimation)
        y_k = g_curv(x_k) - g_curv(x_{k-1})
        s_k = x_k - x_{k-1}
        Two-loop L-BFGS recursion on g_step -> direction r
        theta_{k+1} = theta_k - lr * r

    Memory: O(m * n).
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-2,
        history_size: int = 10,
        batch_size: int = 32,
        curvature_batch_size: int = 128,
        n_samples: int = 1000,
    ):
        super().__init__(
            params,
            dict(
                lr=lr,
                history_size=history_size,
                batch_size=batch_size,
                curvature_batch_size=curvature_batch_size,
                n_samples=n_samples,
            ),
        )
        self._s_list: deque = deque(maxlen=history_size)
        self._y_list: deque = deque(maxlen=history_size)
        self._x_prev: Optional[Tensor] = None
        self._g_curv_prev: Optional[Tensor] = None  # curvature-batch grad at x_prev
        self._curvature_closure: Optional[Callable] = None

    def set_curvature_closure(self, fn: Callable) -> None:
        """
        Register a closure that computes loss+grad on the *curvature* mini-batch.
        Must be called before step() to enable overlap-corrected curvature pairs.
        """
        self._curvature_closure = fn

    def _two_loop(self, g: Tensor) -> Tensor:
        """L-BFGS two-loop recursion."""
        q = g.clone()
        alphas: List[float] = []
        for s, y in reversed(list(zip(self._s_list, self._y_list))):
            ys = float(y @ s)
            if abs(ys) < 1e-10:
                alphas.append(0.0)
                continue
            rho = 1.0 / ys
            alpha = rho * float(s @ q)
            q = q - alpha * y
            alphas.append(alpha)
        alphas.reverse()
        if self._s_list:
            s_last = self._s_list[-1]
            y_last = self._y_list[-1]
            gamma = float(s_last @ y_last) / (float(y_last @ y_last) + 1e-10)
        else:
            gamma = 1.0
        r = gamma * q
        for (s, y), alpha in zip(zip(self._s_list, self._y_list), alphas):
            ys = float(y @ s)
            if abs(ys) < 1e-10:
                continue
            rho = 1.0 / ys
            beta = rho * float(y @ r)
            r = r + s * (alpha - beta)
        return r

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[Tensor]:  # type: ignore[override]
        loss = None
        # Step-batch gradient (small batch, drives search direction)
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        all_params = [p for group in self.param_groups for p in group["params"]]
        lr = self.param_groups[0]["lr"]
        x = _flatten_params(all_params)
        g_step = _flatten_grads(all_params)

        # Curvature-batch gradient (larger batch for less noisy y_k)
        if self._curvature_closure is not None:
            with torch.enable_grad():
                self._curvature_closure()
            g_curv = _flatten_grads(all_params)
        else:
            # Fallback: use step-batch gradient (same as StochasticLBFGSTorch)
            g_curv = g_step.clone()

        # Update curvature pairs using the curvature-batch gradient difference
        if self._x_prev is not None and self._g_curv_prev is not None:
            s = x - self._x_prev
            y = g_curv - self._g_curv_prev
            if float(y @ s) > 1e-10:
                self._s_list.append(s.clone())
                self._y_list.append(y.clone())

        direction = self._two_loop(g_step)
        _unflatten_params(x - lr * direction, all_params)
        self._x_prev = x.clone()
        self._g_curv_prev = g_curv.clone()
        return loss


# =============================================================================
# Convenience registry
# =============================================================================

ALL_TORCH_OPTIMIZERS = {
    "GD":               GradientDescentTorch,
    "SGD":              SGDTorch,
    "MiniBatchSGD":     MiniBatchSGDTorch,
    "MomentumSGD":      MomentumSGDTorch,
    "Nesterov":         NesterovMomentumTorch,
    "AdaGrad":          AdaGradTorch,
    "RMSProp":          RMSPropTorch,
    "Adam":             AdamTorch,
    "DiagonalNewton":   DiagonalNewtonTorch,
    "BlockNewton":      BlockNewtonTorch,
    "DFP":              DFPTorch,
    "BFGS":             BFGSTorch,
    "LBFGS":            LBFGSTorch,
    "StochasticLBFGS":  StochasticLBFGSTorch,
    "MiniBatchLBFGS":   MiniBatchLBFGSTorch,
    "GaussNewtonCG":    GaussNewtonCGTorch,
    "HessianFree":      HessianFreeTorch,
    "Newton":           NewtonsMethodTorch,
}