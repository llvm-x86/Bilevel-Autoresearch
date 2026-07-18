Based on the trace analysis, the primary bottleneck is that the search is getting stuck in a narrow region around the initial best configuration (LR=0.003, WD=1e-5) and failing to explore promising alternative regimes. The runner's current mechanism appears to be making only small perturbations to the best-known config, leading to all proposals being discarded.

Here are 4 concrete mechanism improvements:

---

## Proposal 1: Replay-Based Interleaved Exploration

**Domain**: Reinforcement Learning (experience replay), Bayesian optimization

**Core idea**: Occasionally re-execute promising *discarded* configurations multiple times with different random seeds to distinguish between "lucky" convergence and genuinely good hyperparameter regimes.

**Implementation target**: `GpuBenchRunner._select_next_config()` — Add a config replay buffer that re-evaluates top-K discarded configs with new seeds before generating new proposals.

**Why it helps**: Trace shows iteration 4 achieved bpb=0.1286, but subsequent nearby configs all failed. This suggests either a "lucky" random seed or a narrow optimal region. Replaying with different seeds would: (1) confirm whether iteration 4 was an outlier, and (2) potentially discover that a previously discarded config (e.g., iter 16 with bpb=1.1665) can achieve <0.1286 with the right seed. Three seeds per config would require only 3 extra iterations, dramatically reducing false negatives.

**Implementation complexity**: 3 (add replay buffer, seed logic, evaluation threshold)

**Risk**: Low — replay is computationally cheap (3 extra runs per cycle) and directly addresses the "too many discards" symptom.

---

## Proposal 2: Adaptive Perturbation Budget with Crossover

**Domain**: Evolutionary algorithms (CMA-ES), Neuroevolution

**Core idea**: When multiple consecutive proposals are discarded, increase the mutation step size and introduce crossover between *discarded* configs to escape local optima, rather than continuing small perturbations around the best-known config.

**Implementation target**: `GpuBenchRunner._mutate_config()` — Track discard streak counter; when streak > 2, randomly swap hyperparameters between the last 3 discarded configs and apply 2x larger perturbation noise.

**Why it helps**: The trace shows 17 consecutive discards after iteration 4. Each proposal only changes 1-2 parameters by tiny amounts (e.g., LR=0.0025 vs 0.0027 vs 0.0028). This is effectively local search around a potential false local minimum. Crossover from discarded configs (e.g., iter 16's LR=0.0027 with iter 3's WD=1e-5) could create configurations that neither parent alone achieved. The non-greedy baseline (value 6.4116) vs best (0.1286) suggests the landscape is highly irregular — crossover helps escape narrow basins.

**Implementation complexity**: 4 (streak counter, crossover logic, adaptive noise scaling, guard against infinite loops)

**Risk**: Medium — crossover could generate illegal parameter combinations (e.g., conflicting batch size constraints), requiring validation.

---

## Proposal 3: Heterogeneous Parameter-Space Resampling

**Domain**: Multi-fidelity optimization, Random Forests (Hyperband)

**Core idea**: Periodically sample hyperparameters from their full distribution (uniform/AIST) rather than from perturbations of the current best, with the sampling frequency controlled by the number of consecutive discards.

**Implementation target**: `GpuBenchRunner._is_fresh_exploration_needed()` — Add a Bernoulli trial: every N consecutive discards (N=3-5), force resample each hyperparameter independently from its original prior distribution with probability p=0.5.

**Why it helps**: The trace shows the search never re-explored HIDDEN_DIM=128/384 after iter 5-6 (both discarded), nor tried BATCH_SIZE other than 64. The runner appears "blind" to entire axes of variation. Resampling from full priors would re-introduce diversity: e.g., try LR=0.01 with WD=0, or HIDDEN_DIM=512 with BATCH_SIZE=128. The best val_bpb is 0.1286, but the baseline without tuning is 6.4116 — suggesting the optimal region might be in a completely different part of parameter space that small perturbations cannot reach.

**Implementation complexity**: 2 (add counter logic, call sampler with prior distributions)

**Risk**: Low — worst case: generates an immediate discard (same as current behavior).

---

## Proposal 4: Surrogate-Gated Early Termination

**Domain**: Bayesian optimization with cheap surrogate models, Hyperband early stopping

**Core idea**: Before launching an expensive GPU training run, use a fast surrogate model (e.g., Random Forest trained on all previous config-outcome pairs) to predict whether the proposed config is likely to beat the current best val_bpb — if not, skip the run and immediately propose a new config.

**Implementation target**: `GpuBenchRunner._should_skip_config()` — After generating a candidate config, query a lightweight surrogate model (fit on all previous results). If predicted bpb > current_best * 1.5, skip the GPU run and re-generate.

**Why it helps**: The trace shows 17 consecutive expensive GPU runs producing non-improving results. Each run requires wall-clock time for subprocess launch, model init, and training. A surrogate model can identify unpromising proposals in microseconds. Since many proposals are tiny perturbations (<5% change) that almost certainly won't beat the best, the surrogate can reject them with high confidence. This would free GPU cycles for more diverse exploration. Even a simple k-NN or linear model trained on 20+ data points would have decent predictive power for this problem.

**Implementation complexity**: 5 (requires sklearn integration, model persistence, prediction threshold tuning, handling cold-start before enough data)

**Risk**: High — false positives (skipping a config that would beat the best) could permanently lock the search out of optimal regions. Mitigation: initially only use surrogate for >10 data points, and maintain a "skip veto" mechanism where 1 in 5 skipped configs is forced to run as a control.