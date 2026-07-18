Based on the trace analysis, I can see that the runner is stuck in a local optimum around val_bpb=7.9-8.0, with only iteration 4 achieving a breakthrough (0.1153). The core problem is that the runner lacks mechanisms to systematically escape this plateau.

Here are 4 concrete mechanism improvements:

## Hypothesis 1: **Bayesian Optimization Surrogate**

1. **Domain**: Probabilistic machine learning / AutoML
2. **Core idea**: Replace blind random perturbations with a Gaussian Process surrogate model that learns the response surface of `val_bpb` across hyperparameter space and proposes candidates with highest Expected Improvement.
3. **Implementation target**: `GpuBenchRunner.inner_loop()` — add a GP model (using `scikit-learn`'s `GaussianProcessRegressor`) that updates after each iteration and uses `expected_improvement()` acquisition function to generate the next `changes` dict.
4. **Why it helps**: The current perturbations are random and symmetric; iteration 19 (WD=5e-05) gave 5.7553 vs best 0.1153, but the runner cannot exploit that signal. A GP would learn that small weight decay changes produce wide variance, guiding exploration toward promising (LR, WD) regions.
5. **Implementation complexity**: 4 (requires scikit-learn dependency, kernel selection, fitting overhead per iteration, but GP regressor is small for <30 points)
6. **Risk**: medium (GP may oversmooth the highly non-convex loss landscape; requires careful kernel tuning; may add 5-10ms overhead per iteration which is negligible vs GPU run)

## Hypothesis 2: **Multi-armed Bandit with UCB1 Exploration**

1. **Domain**: Reinforcement learning / online learning
2. **Core idea**: Treat each hyperparameter dimension as an independent bandit arm; use Upper Confidence Bound (UCB1) to select which dimension to perturb, with reward = -Δval_bpb from previous iteration.
3. **Implementation target**: `GpuBenchRunner.__init__()` — maintain a `BanditState` dictionary mapping each hyperparameter key to (count_trials, sum_rewards); in `inner_loop()`, sample UCB1 scores for each arm, select the highest-scoring dimension, then apply a small Gaussian perturbation centered on the best-known value for that dimension.
4. **Why it helps**: Current proposals perturb all dimensions simultaneously (e.g., iter 11 changes 4 params), making attribution impossible. UCB1 isolates one dimension per iteration, learning quickly which hyperparameters actually drive improvements. The trace shows LR changes produce consistent ~8.0, while WD changes produce swings (0.1153→7.9→5.7553)—this should guide the bandit to explore WD more aggressively.
5. **Implementation complexity**: 2 (no new dependencies; simple arithmetic; track rolling history of per-dimension rewards)
6. **Risk**: low (bandit is well-understood; suboptimal only if interactions between hyperparameters dominate, but initial gains from disentangling are substantial)

## Hypothesis 3: **Adaptive Search Radius with Success-Replay Buffer**

1. **Domain**: Evolutionary optimization / CMA-ES
2. **Core idea**: Maintain a success-replay buffer of the last 3 `val_bpb` improvements; if no improvement within 5 iterations, linearly increase the perturbation step size (σ) up to 3×; upon improvement, shrink σ back to baseline.
3. **Implementation target**: `GpuBenchRunner._perturb_params()` (if such a method exists) or inline in `inner_loop()` — store `self.perturb_std = {k: 0.05}` initially; multiply all entries by 1.2 after each consecutive discard; reset to 0.05 upon any keep.
4. **Why it helps**: The trace shows iterations 5-21 all discarded with near-identical val_bpb (~8.0), meaning the runner is stuck in a narrow valley where small steps (LR=0.003±0.0005, WD=1e-4±0.5e-4) produce no signal. Increasing step size would jump to distant regions where the loss function curvature changes, potentially escaping the plateau. The success-replay prevents over-dilation if improvements occur.
5. **Implementation complexity**: 1 (10 lines of code; no external dependencies)
6. **Risk**: low (step size bounding at 3× prevents catastrophic jumps; fast recovery on success)

## Hypothesis 4: **Covariance Matrix Adaptation with Elite Selection**

1. **Domain**: Continuous optimization / CMA-ES
2. **Core idea**: Instead of random perturbations, maintain a full covariance matrix over hyperparameters; after each iteration, update the covariance using the top 20% of recent samples (elite set), adapting the search distribution toward the geometry of the loss landscape.
3. **Implementation target**: `GpuBenchRunner` class — add `self._cma = {'mean': best_config, 'cov': 0.1*I, 'samples': []}`; in `inner_loop()`, sample from multivariate normal, record results, and after every 3 iterations, update mean and covariance via CMA rank-μ update.
4. **Why it helps**: The trace shows heavy correlation between LR and WD (both produce near-identical val_bpb around 8.0 when changed together). CMA-ES learns this correlation structure — if LR and WD are anti-correlated (high LR needs high WD), CMA-ES rotates the search ellipsoid to move along the optimal ridge, avoiding wasted exploration in orthogonal directions.
5. **Implementation complexity**: 5 (CMA-ES requires eigendecomposition of covariance matrix each update, numerical stability for near-singular matrices, hyperparameters for population size, and history management across multiple iterations)
6. **Risk**: medium (overhead of matrix operations is small for 4D space, but initial covariance may collapse prematurely; requires careful damping of update rate to avoid oscillation)

**Recommended priority**: Start with Hypothesis 3 (Adaptive Search Radius) as it directly addresses the "too many discards" bottleneck with minimal complexity, then add Hypothesis 2 (Bandit) to improve attribution, and finally Hypothesis 1 (Bayesian Optimization) for global exploration if needed.