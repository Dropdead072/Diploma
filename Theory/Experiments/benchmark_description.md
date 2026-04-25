# Benchmark Experiments — Description

This document describes the three benchmark suites implemented in the diploma project:

1. **Mathematical Test Functions** (`optimizers_test_functions.ipynb`)
2. **Classical ML Models** (`optimizers_ClassicalML_benchmark.ipynb` and `optimizers_ClassicalML_benchmark_multi.ipynb`)
3. **Deep Learning** (planned — MLP, ResNet-18, MobileNetV2, Tiny LLM; see [`setting.md`](setting.md))

---

## 0. Computing Platforms

Experiments are split across two platforms to reflect the resource-constrained focus of the diploma.

### Platform A — Limited-resource (laptop / personal workstation)

| Component | Specification |
|---|---|
| CPU | Apple M2 (8-core, 3.49 GHz) |
| RAM | 16 GB unified memory |
| GPU / VRAM | Apple GPU (10-core), shares unified memory — effectively ≤ 8 GB VRAM |
| OS | macOS 14 Sonoma |
| Python | 3.11 (conda env `diploma_env`) |
| Key libraries | PyTorch 2.x (MPS backend), NumPy 1.26, scikit-learn 1.4 |

**Used for:** test-function benchmarks, classical ML benchmarks, MLP experiments, and any model that fits in ≤ 8 GB unified memory.

### Platform B — High-resource / industrial (cloud / HPC node)

| Component | Specification |
|---|---|
| CPU | Intel Xeon (16+ cores) |
| RAM | 64 GB |
| GPU | NVIDIA A100 40 GB (or equivalent) |
| OS | Ubuntu 22.04 |
| Python | 3.11 |
| Key libraries | PyTorch 2.x (CUDA backend), NumPy 1.26 |

**Used for:** ResNet-18, MobileNetV2, and Tiny LLM experiments where VRAM > 8 GB is required.

> **Note:** All results reported in sections 1–2 below were obtained on **Platform A**.

---

## 1. Mathematical Test Functions Benchmark

### Purpose

Evaluates optimizer behaviour on six classical 2-D mathematical test functions widely used in the optimisation literature. The functions cover a range of difficulty levels — from simple convex bowls to highly multimodal landscapes — and allow direct visual inspection of convergence trajectories.

### Test Functions

| Function | Global minimum x\* | f\* | Characteristics |
|---|---|---|---|
| **Sphere** | (0, 0) | 0 | Convex, unimodal, smooth — simplest baseline |
| **Rosenbrock** | (1, 1) | 0 | Non-convex, narrow curved valley; gradient nearly vanishes along the valley |
| **Rastrigin** | (0, 0) | 0 | Highly multimodal (many local minima); tests ability to escape local traps |
| **Ackley** | (0, 0) | 0 | Multimodal with a nearly flat outer region; sensitive to step size |
| **Himmelblau** | nearest of (3,2), (−2.805,3.131), (−3.779,−3.283), (3.584,−1.848) | 0 | Four symmetric global minima; distance measured to the **nearest** global minimum |
| **Levy N.13** | (1, 1) | 0 | Oscillatory landscape; gradient computed numerically |

All functions are implemented in [`test_functions.py`](../Code/utils/objective_functions/test_functions.py) as subclasses of `ObjectiveFunction`, exposing `.value(x)`, `.grad(x)`, and (where applicable) `.hess(x)`.

> **Himmelblau note:** Because the function has four equally valid global minima, the distance metric `‖x_k − x*‖₂` is computed as `min over all four minima`, so an optimizer that converges to any basin is credited correctly.

### Optimizer Suite

The following 14 deterministic optimizers are tested (stochastic variants requiring a dataset index are excluded):

| Group | Optimizers |
|---|---|
| First-order deterministic | GD, Momentum SGD, Nesterov |
| Adaptive first-order | AdaGrad, RMSProp, Adam |
| Diagonal / block Hessian | Diagonal Newton, Block Newton |
| Quasi-Newton | DFP, BFGS, L-BFGS |
| CG-based | Gauss-Newton CG, Hessian-Free |
| Pure second-order | Newton's Method |

### Experimental Protocol

1. **Initialisation** — each optimizer starts from a random point drawn uniformly from the function's search bounds.  
   **Multiple runs:** 5 independent runs with seeds `{42, 43, 44, 45, 46}` are executed per (optimizer, function) pair. Summary statistics (mean ± std of final loss and distance to minimum) are reported alongside single-seed trajectories.
2. **Optimisation loop** — the generic `optimize()` function from [`core/optimizer.py`](../Code/core/optimizer.py) is called with `max_iter=500` and gradient-norm tolerance `eps=1e-8`. At each step the optimizer's `.step(x, grad_fn, hess_fn)` method is called.
3. **Metrics recorded per (optimizer, function, seed) triple**:
   - Objective value at every iteration → convergence curve
   - Distance from the **nearest** global minimum `min_j ‖x_k − x*_j‖₂`
   - Final objective value `f(x_final)`
   - Number of iterations until convergence (or `max_iter` if not converged)
   - Wall-clock runtime (seconds)
   - Whether the run diverged (`f > 1e6`) or hit a numerical failure (NaN/Inf)
4. **Visualisations produced**:
   - **Contour + trajectory plot** — 2-D contour with optimizer path overlaid (seed 42)
   - **Convergence curve** — `f(x_k)` vs. iteration (linear and log scale), mean ± std band across seeds
   - **Summary heatmap** — mean final objective value for every (optimizer, function) pair
   - **Ranking table** — optimizers sorted by mean final value per function

### Hyperparameter Policy

Hyperparameters are **fixed** from a brief preliminary sweep on the Sphere function and held constant across all functions. This intentionally tests robustness rather than per-function tuning. The fixed values are:

| Optimizer | lr | Other |
|---|---|---|
| GD, Momentum SGD, Nesterov | 1e-3 | α = 0.9 |
| AdaGrad | 1e-2 | eps = 1e-8 |
| RMSProp | 1e-3 | β = 0.9 |
| Adam | 1e-3 | β₁=0.9, β₂=0.999 |
| Newton, Diagonal Newton, Block Newton | 1.0 | damping = 1e-6 |
| DFP, BFGS, L-BFGS | 1.0 | m = 10 (L-BFGS) |
| Gauss-Newton CG, Hessian-Free | 1.0 | cg_iters = 50 |

### Training Stability Criteria

Stability is evaluated explicitly as a first-class metric:

| Criterion | Definition |
|---|---|
| **Divergence** | `f(x_k) > 1e6` at any iteration |
| **Numerical failure** | NaN or Inf in parameters or gradient |
| **Hyperparameter sensitivity** | Std of final loss across 5 seeds > 10× median |
| **Stagnation** | `‖x_final − x_init‖ < 1e-4` (optimizer did not move) |

Divergence and failure counts are reported in the summary table alongside loss metrics.

### Key Observations from Results

- **Newton's Method** and **L-BFGS** converge in very few iterations on smooth functions (Sphere, Rosenbrock, Himmelblau) but require a Hessian or curvature information.
- **Adam** and **RMSProp** are robust across functions but converge more slowly than quasi-Newton methods.
- **Rastrigin** and **Ackley** expose the multimodal trap problem: gradient-based methods converge to whichever local minimum is closest to the starting point; variance across seeds is high.
- **Nesterov** can diverge with a fixed step size on non-convex functions (observed on Rosenbrock with `lr=1e-3`).
- **Hessian-Free** uses finite-difference Hessian-vector products and is competitive with full Newton on smooth problems while requiring only gradient evaluations.
- **Newton / BFGS / DFP** become impractical for large parameter spaces (O(n²) memory); on 2-D test functions this is not a concern, but it is noted for the ML benchmarks.

### Applicability Limits

| Method | Practical limit |
|---|---|
| Newton's Method | n ≲ 1 000 (O(n³) solve per step) |
| BFGS / DFP | n ≲ 10 000 (O(n²) matrix storage) |
| L-BFGS | n up to ~10⁶ (O(m·n) memory, m ≪ n) |
| Hessian-Free | Scales to large n; cost = 2 × gradient per CG iter |
| First-order (GD, Adam, …) | Unlimited n; slowest convergence rate |

---

## 2. Classical ML Benchmark

### Purpose

Evaluates all 18 implemented optimizers on supervised learning tasks — **linear regression** (MSE loss) and **binary / multi-class logistic regression** (BCE / CE loss) — using synthetic datasets. Measures both optimisation quality (final loss) and generalisation (R² / accuracy on a held-out test set).

A multi-class extension (`optimizers_ClassicalML_benchmark_multi.ipynb`) adds **softmax regression** (cross-entropy loss) on a synthetic 4-class dataset.

### Datasets

#### Regression
- **N_train = 300**, **N_test = 100**, **n_features = 10**
- `X ~ N(0, I)`, `y = X w_true + 0.3 ε`, `ε ~ N(0, 1)`
- `w_true ~ N(0, I)` (fixed seed `42`)

#### Binary Classification
- Same dimensions as regression
- `logits = X w_cls_true`, `y = (logits + 0.5 ε > 0)` → balanced classes (~50 % positive)

#### Multi-class Classification (multi notebook)
- **N_train = 400**, **N_test = 100**, **n_features = 10**, **K = 4 classes**
- `X ~ N(0, I)`, class labels assigned via softmax of `X W_true`

### Models

| Task | Model | Loss | Hessian available |
|---|---|---|---|
| Regression | `LinearRegression` | MSE = `(1/n) ‖Xw − y‖²` | Yes: `(2/n) XᵀX` |
| Binary classification | `LogisticRegression` | BCE = `−(1/n) Σ [y log σ(xᵀw) + (1−y) log(1−σ(xᵀw))]` | Yes: `(1/n) Xᵀ diag(σ(1−σ)) X` |
| Multi-class | `SoftmaxRegression` | CE = `−(1/n) Σ log P(y_i | x_i)` | Not used (diagonal approx) |

All models are implemented in [`utils/models/`](../Code/utils/models/) and inherit from `core.model.Model`.

### Optimizer Suite (18 optimizers)

| Group | Optimizers |
|---|---|
| First-order deterministic | GD (`lr=1e-2`), SGD (`lr=1e-2`), Mini-Batch SGD (`lr=1e-2`, `batch=32`) |
| Momentum | Momentum SGD (`lr=1e-2`, `α=0.9`), Nesterov (`lr=1e-2`, `α=0.9`) |
| Adaptive first-order | AdaGrad (`lr=1e-1`), RMSProp (`lr=1e-2`), Adam (`lr=1e-2`) |
| Diagonal / block Hessian | Diagonal Newton (`lr=1.0`), Block Newton (`lr=1.0`, `block=5`) |
| Quasi-Newton | DFP (`lr=1.0`), BFGS (`lr=1.0`), L-BFGS (`lr=1.0`, `m=10`) |
| Stochastic quasi-Newton | Stoch. L-BFGS (`lr=1e-2`, `batch=32`), MB L-BFGS (`lr=1e-2`, `batch=32`, `curv_batch=128`) |
| CG-based | Gauss-Newton CG (`lr=1.0`, `cg=20`), Hessian-Free (`lr=1.0`, `cg=20`) |
| Pure second-order | Newton's Method (`lr=1.0`) |

### Hyperparameter Policy

Hyperparameters are **fixed** from a brief preliminary sweep and held constant across all tasks. No per-task tuning is performed. This is intentional: the goal is to compare optimizers under a fair, uniform budget rather than to find the best possible result for each method.

### Experimental Protocol

1. **Data generation** — synthetic datasets are created with `np.random.seed(42)` for reproducibility.
2. **Multiple runs** — 5 independent runs with seeds `{42, 43, 44, 45, 46}` are executed per optimizer. Mean ± std of all metrics are reported.
3. **Helper functions** `run_linreg` / `run_logreg` instantiate a model with the given optimizer and call `model.fit(X_train, y_train, max_iter=300, eps=1e-8)`.
4. **Logging** — an `OptimizationLogger` is attached to each run and records:
   - Loss value at every step
   - Gradient norm at every step
   - Wall-clock time per step
5. **Evaluation** — after fitting, `model.score(X_test, y_test)` returns R² (regression) or accuracy (classification).
6. **Aggregation** — a `MultiRunLogger` collects all loggers and prints a sorted comparison table.

### Metrics

| Metric | Description |
|---|---|
| `Steps` | Number of optimizer iterations executed |
| `Final Loss` | Loss value at the last iteration |
| `Best Loss` | Minimum loss achieved across all iterations |
| `Time (s)` | Total wall-clock time for the run |
| `Peak Memory (MB)` | Estimated peak RAM usage during the run (measured via `tracemalloc`) |
| `R²` | Coefficient of determination on the test set (regression only) |
| `Accuracy` | Fraction of correctly classified test samples (classification only) |
| `Diverged` | Whether the run produced NaN/Inf or loss > 1e6 |

### Resource-Related Evaluation

Memory usage is relevant even for classical ML when the parameter space is large or when second-order methods store dense Hessian approximations:

| Method class | Memory complexity | Feasible on Platform A (16 GB RAM)? |
|---|---|---|
| First-order (GD, SGD, Adam, …) | O(n) | Yes, for any n |
| Diagonal Newton | O(n) | Yes |
| Block Newton (block=5) | O(n) | Yes |
| DFP / BFGS | O(n²) | Yes for n ≲ 10 000; impractical for n > 100 000 |
| L-BFGS (m=10) | O(m·n) | Yes for n up to ~10⁶ |
| Newton's Method | O(n²) storage + O(n³) solve | Yes for n ≲ 1 000 |

For the synthetic datasets used here (n = 10 features), all methods are feasible on Platform A. The table above is provided as a reference for scaling to larger problems.

### Training Stability Criteria

| Criterion | Definition |
|---|---|
| **Divergence** | Loss > 1e6 or NaN/Inf at any step |
| **Numerical failure** | Singular Hessian (LinAlgError) or CG non-convergence |
| **Slow convergence** | Final loss > 2× best achievable loss at `max_iter` |
| **Hyperparameter sensitivity** | Std of final loss across 5 seeds > 10× median |

### Visualisations

1. **Loss curves** — `f(w_k)` vs. iteration on linear and log scale for all optimizers simultaneously, with mean ± std band across seeds.
2. **Gradient norm curves** — `‖∇f(w_k)‖` vs. iteration on log scale.
3. **Final loss bar chart** — horizontal bars sorted by final loss (lower is better).
4. **Time vs. final loss scatter** — each optimizer is a point; reveals the speed–accuracy trade-off.
5. **Memory vs. final loss scatter** — each optimizer is a point; reveals the memory–accuracy trade-off.
6. **Summary DataFrame** — pandas table sorted by final loss, including all metrics and divergence flags.

### Key Observations from Results

#### Linear Regression (MSE)

- **Newton's Method** converges in **2 steps** (exact solution for quadratic MSE).
- **L-BFGS**, **Block Newton**, **Diagonal Newton** converge in 15–20 steps.
- **BFGS** converges in ~50 steps; **DFP** in ~115 steps.
- **Momentum SGD**, **GD**, **SGD**, **Mini-Batch SGD** all reach near-optimal loss within 300 steps.
- **Adam**, **AdaGrad**, **RMSProp** converge more slowly on this quadratic problem; Adam in particular stalls far from the optimum with default `lr=1e-2`.
- **Nesterov** diverges with `lr=1e-2` on this problem (step size too large for the curvature).

#### Binary Logistic Regression (BCE)

- **L-BFGS** converges in **28 steps** to the best achievable loss (0.2262).
- **BFGS**, **Block Newton**, **Diagonal Newton** all reach the same optimum in 100–200 steps.
- **RMSProp** and **Momentum SGD** are the best first-order methods (loss ≈ 0.246–0.261).
- **GD** and **SGD** converge slowly (loss ≈ 0.73–0.81 at 300 steps).
- **Newton's Method** fails with a singular Hessian error (saturated sigmoid probabilities).
- **Gauss-Newton CG** and **Hessian-Free** achieve good accuracy (89 %) but their loss values are inflated due to numerical issues with the CG inner solver on the BCE Hessian.

### Applicability Limits

| Method | Practical limit in ML context |
|---|---|
| Newton's Method | Fails when Hessian is singular (e.g. saturated BCE); impractical for n > 1 000 |
| BFGS / DFP | O(n²) memory; impractical for n > 10 000 parameters |
| L-BFGS | Practical up to ~10⁶ parameters; preferred quasi-Newton for medium-scale ML |
| Stochastic L-BFGS / MB L-BFGS | Noisy curvature pairs; useful when full-batch gradient is too expensive |
| First-order (GD, SGD, Adam) | Unlimited scale; slowest convergence but lowest memory and per-step cost |
| Hessian-Free | Good compromise: O(n) memory, super-linear convergence via CG; sensitive to damping |

### Practical Recommendations

| Scenario | Recommended optimizer(s) |
|---|---|
| Small smooth problem (n ≤ 100, convex) | Newton's Method or L-BFGS |
| Medium problem (100 < n ≤ 10 000) | L-BFGS, BFGS, or Adam |
| Large problem (n > 10 000) | Adam, RMSProp, or Stochastic L-BFGS |
| Memory-constrained (RAM < 4 GB) | SGD, Mini-Batch SGD, Adam, L-BFGS (m ≤ 5) |
| Non-convex / multimodal | Adam or RMSProp (robust to local minima); avoid Newton |
| Noisy / stochastic gradients | Adam, RMSProp, Mini-Batch SGD, Stochastic L-BFGS |

---

## 3. Deep Learning Benchmark (Planned)

As specified in [`setting.md`](setting.md), a third benchmark suite will evaluate optimizers on deep learning models:

| Model | Task | Platform |
|---|---|---|
| MLP Classifier / Regressor | Synthetic tabular data | Platform A |
| ResNet-18 | Image classification (CIFAR-10) | Platform B |
| MobileNetV2 | Image classification (CIFAR-10) | Platform B |
| Tiny LLM (Qwen3-0.6B or similar) | Language modelling | Platform B |

**Evaluation criteria** (from `setting.md`):
- Accuracy / Top-1 Accuracy (CV tasks)
- Cross-Entropy Loss
- Perplexity / Log-Likelihood (LLM)
- Training time and VRAM usage
- Convergence stability across random seeds

**Resource constraints** are a primary concern: methods that exceed the VRAM budget of Platform A (≈ 8 GB) will be flagged as infeasible for limited-resource deployment.

---

## 4. Implementation Notes

### Optimizer Interface

All numpy optimizers implement the `Optimizer` abstract base class from [`core/optimizer.py`](../Code/core/optimizer.py):

```python
class Optimizer(ABC):
    def step(self, x, grad_fn, hess=None) -> np.ndarray: ...
    def reset(self) -> None: ...
    @property
    def name(self) -> str: ...
```

The `optimize()` loop calls `step()` repeatedly until `‖grad‖ < eps` or `max_iter` is reached.

PyTorch optimizers in [`utils/optimizers/torch_optimizers.py`](../Code/utils/optimizers/torch_optimizers.py) subclass `torch.optim.Optimizer` and follow the standard closure convention.

### Loss Functions

Loss functions are bundled as `LossFunction` dataclass instances in [`utils/losses/numpy_losses.py`](../Code/utils/losses/numpy_losses.py), each exposing `.value`, `.grad`, and `.hess` callables with signature `(w, X, y)`.

### Logging

`OptimizationLogger` records per-step `(loss, grad_norm, time)` tuples. `MultiRunLogger` aggregates multiple loggers and provides `print_comparison()` and `summary_df()` methods.

---

## 5. Reproducibility

All experiments use `np.random.seed(42)` (and seeds 43–46 for multi-run statistics) at the top of each notebook. Optimizer hyperparameters are fixed as listed above. To reproduce:

```bash
cd Diploma/Code
jupyter nbconvert --to notebook --execute optimizers_test_functions.ipynb
jupyter nbconvert --to notebook --execute optimizers_ClassicalML_benchmark.ipynb
jupyter nbconvert --to notebook --execute optimizers_ClassicalML_benchmark_multi.ipynb
```

The `diploma_env` conda environment must be active. All dependencies are pinned in `environment.yml` (if present) or can be installed via:

```bash
conda activate diploma_env
pip install torch numpy scipy scikit-learn matplotlib pandas jupyter
```
