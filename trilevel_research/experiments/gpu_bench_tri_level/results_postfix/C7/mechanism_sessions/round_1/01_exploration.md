Based on the trace analysis, the primary bottleneck is that the search is stuck in a local optimum around LR=0.003 and cannot escape to find better configurations. The discard rate is ~95%, indicating the exploration strategy is too conservative. Here are 4 concrete mechanism changes:

---

## Mechanism 1: Momentum-based Exploration Injection

**Domain**: Stochastic optimization / simulated annealing

**Core idea**: Inject controlled parameter noise that decays over iterations to escape local optima while maintaining convergence.

**Implementation target**: `GpuBenchRunner.__init__()` — add exploration noise buffer; `GpuBenchRunner._propose_changes()` — apply noise to selected parameter.

**Why it helps**: Current proposals are too conservative (tiny LR changes ±0.0002–0.0005). Adding momentum-scaled noise proportional to `1/(1 + iter)` would force exploration of distant points (e.g., LR=0.01, 0.05) early, while letting the search refine later. This directly addresses the "too many discards" bottleneck by generating bolder proposals.

**Implementation complexity**: 2 (add 10–15 lines)
**Risk**: Low — can be toggled off; noise decays to zero, so final behavior is unchanged.

---

## Mechanism 2: Adaptive Search Radius Based on Consecutive Discards

**Domain**: Bayesian optimization / hyperband

**Core idea**: Dynamically expand the search radius (parameter perturbation magnitude) when ≥3 consecutive proposals are discarded.

**Implementation target**: `GpuBenchRunner._propose_changes()` — add `self.discard_streak` counter; multiply parameter deltas by `1.5 ** discard_streak` when `discard_streak >= 3`.

**Why it helps**: The trace shows 15 consecutive discards after iteration 5. Each discard narrows the search (tiny LR adjustments), but the valley is flat. Expanding search radius after discards forces exploration of distant configurations (e.g., BATCH_SIZE=256, HIDDEN_DIM=1024, different optimizers). This directly breaks the local optimum trap.

**Implementation complexity**: 2 (add 8–12 lines)
**Risk**: Medium — could oscillate wildly, but resets to normal radius on first keep.

---

## Mechanism 3: Multi-armed Bandit Parameter Selection with UCB

**Domain**: Reinforcement learning / multi-armed bandits

**Core idea**: Replace uniform random parameter selection with Upper Confidence Bound (UCB) sampling that prioritizes parameters with high uncertainty (few trials) and high potential.

**Implementation target**: `GpuBenchRunner._select_parameter_to_tweak()` — new method using UCB scores: `score = mean_reward + sqrt(2 * log(total_trials) / n_trials_of_param)`.

**Why it helps**: The trace shows 11 out of 22 iterations only change LR, while HIDDEN_DIM, BATCH_SIZE, and WEIGHT_DECAY are rarely tried. UCB would force exploration of under-sampled parameters (e.g., weight decay 1e-4, batch size 64/256) which may yield better models. This reduces the "all eggs in LR basket" problem and increases diversity of proposals.

**Implementation complexity**: 4 (need reward tracking per parameter, confidence bounds calculation)
**Risk**: Medium — requires maintaining per-parameter stats, may initially sample poor configurations.

---

## Mechanism 4: Probabilistic Parameter Inheritance with Crossover

**Domain**: Genetic algorithms / evolutionary strategies

**Core idea**: When proposing new configurations, randomly combine parameters from the current best configuration with one randomly selected "parent" from the top-k historical configurations.

**Implementation target**: `GpuBenchRunner._propose_changes()` — after selecting parameter to change, with 50% probability copy that parameter's value from a random top-3 historical config instead of random perturbation.

**Why it helps**: Current proposals are all single-parameter perturbations from the current best. Crossover recombines successful parameters from different iterations (e.g., LR=0.003 from iter 4 + HIDDEN_DIM=256 from iter 16 + BATCH_SIZE=128 from iter 7). This creates novel but plausible configurations that are more likely to succeed than random perturbations.

**Implementation complexity**: 3 (add 15–20 lines, maintain history buffer)
**Risk**: Low — crossover is additive; worst case behaves like random restart.