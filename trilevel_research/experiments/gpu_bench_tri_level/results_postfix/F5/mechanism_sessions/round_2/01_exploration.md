Based on the trace analysis, the core problem is that the runner fails to generalize from promising configurations (like iter 20's 0.5247 val_bpb) and keeps returning to the sub-optimal baseline (3.0 val_bpb). The search is trapped in a local optimum and lacks mechanisms to escape it.

Here are 4 concrete mechanism improvements:

---

## Mechanism 1: Adaptive Perturbation Anchoring

**Domain**: Evolutionary optimization / Reinforcement learning

**Core idea**: Dynamically scale perturbation magnitude based on recent improvement correlation, not just distance from current best.

**Implementation target**: `GpuBenchRunner._propose_next_config()`

**How it helps**: Currently, when iter 20 achieves 0.5247 via LR=0.0035, the runner immediately reverts to LR=0.003 in iter 21, losing the signal. By anchoring larger perturbations near recent successes and scaling them down near failures, the runner will amplify valuable deviations. Specifically, track the last N configuration deltas and their resulting val_bpb changes, then double the perturbation magnitude for any hyperparameter that appeared in a recent success (within last 5 iterations with val_bpb < 1.0).

**Implementation complexity**: 3/5

**Risk**: Medium – May over-explore initially, but the anchoring decay (halving every 5 iterations without success) prevents runaway.

---

## Mechanism 2: Gradient-Guided Configuration Smoothing

**Domain**: Stochastic gradient descent / Exploration-exploitation

**Core idea**: Instead of discarding "failed" configurations entirely, use their partial parameter gradients to inform next proposals via exponential moving average.

**Implementation target**: `GpuBenchRunner._update_best()` and `runner_state` dictionary

**How it helps**: The trace shows 19 consecutive discards after iter 2. Each discard wastes gradient signal. By maintaining an EMA of configuration changes weighted by val_bpb delta, the runner can propose "interpolated" configurations (e.g., average LR of 0.0035 and 0.0025 after both individually failed but bracketed the best). This allows the search to "walk along" the gradient toward better values rather than binary keep/discard.

**Implementation complexity**: 4/5

**Risk**: Low – EMA naturally decays stale signals, and only applies smoothing within ±30% of current best to avoid speculative jumps.

---

## Mechanism 3: Memory-Backed Restart with Perturbation Decay

**Domain**: Genetic algorithms / Simulated annealing

**Core idea**: When val_bpb hasn't improved for 10 iterations, restart from a memory of top K configurations (not just the single best) with exponentially decaying perturbation thresholds.

**Implementation target**: `GpuBenchRunner._check_early_stop()` or new `_maybe_restart_from_history()`

**How it helps**: The trace shows the runner keeps returning to LR=0.003 (val_bpb ~7.9) even after discovering 0.0035 (val_bpb 0.5). By storing top 3-5 configs with their val_bpb and re-seeding after stagnation, the runner can break free. Perturbation decay ensures later restarts explore more locally, preventing oscillation back to the poor baseline.

**Implementation complexity**: 2/5

**Risk**: Medium – Could re-explore already poor regions, but decay mechanism (starting perturbation at 10% of original, halving each restart) limits damage.

---

## Mechanism 4: Heuristic-Based Configuration Pruning

**Domain**: Bayesian optimization / Ensemble pruning

**Core idea**: Pre-compute a "similarity score" between proposed and previously attempted configurations, and skip proposals whose expected improvement (based on history) is below a threshold.

**Implementation target**: `GpuBenchRunner._evaluate_hypothesis()` or new `_prune_redundant_proposals()`

**How it helps**: The trace shows 5/22 iterations are literally identical configs (LR=0.003 only), all yielding ~7.9 val_bpb. This wastes 25% of search budget. By detecting when a new proposal is within 5% Euclidean distance of any past configuration (in normalized hyperparameter space), and the past configuration had val_bpb > 5.0, the runner can either skip the expensive GPU run or inject random noise to differentiate it. This frees evaluation calls for genuinely novel configurations.

**Implementation complexity**: 3/5

**Risk**: Low – Conservative pruning (only skip if similarity >95% AND past val_bpb > 2x current best) ensures false positives are rare.