## Proposed Mechanism Improvements for GpuBenchRunner

### Hypothesis 1: Adaptive Search Space Constriction
1. **Domain**: Bayesian Optimization / Bandit Algorithms
2. **Core idea**: Dynamically shrink the search range around the best-performing hyperparameters after each successful iteration, while widening after failures.
3. **Implementation target**: Modify `propose_changes()` method to maintain a probabilistic sampling distribution (e.g., truncated normal) whose variance adapts based on recent validation outcomes.
4. **Why it helps**: The trace shows most proposals are far from the known good region (LR=0.003, batch=64, hidden=256). Adaptive constriction forces exploration near proven configurations while still allowing occasional wide jumps. This prevents radical changes that waste iterations (e.g., jumping from LR=0.003 to 0.01 or 0.001).
5. **Implementation complexity**: 3
6. **Risk**: low — naturally balances exploration/exploitation, and can be tuned conservatively

### Hypothesis 2: Multi-Objective Acceptance Criterion
1. **Domain**: Multi-Objective Optimization / Pareto Frontier Search
2. **Core idea**: Accept proposals that improve val_bpb OR training stability (measured by epoch-to-epoch variance), not just absolute val_bpb minima.
3. **Implementation target**: Modify the `_evaluate_and_accept()` method to compute a composite score = val_bpb - λ * training_var, where λ is small (0.01-0.05).
4. **Why it helps**: The trace shows many good proposals discarded (e.g., iter 15: val_bpb=6.74 vs best 0.68) despite being much better than baseline (6.41). These could represent useful plateaus. Rewarding stability would accept configurations that are "good enough" and stable, providing richer signal for future proposals.
5. **Implementation complexity**: 2
6. Risk: medium — may accept worse val_bpb; needs careful λ tuning

### Hypothesis 3: Gradient-Guided Perturbation
1. **Domain**: Stochastic Gradient Descent / Perturbation Theory
2. **Core idea**: Estimate the local gradient of val_bpb w.r.t. each hyperparameter by running paired trials (e.g., LR +δ and LR -δ), then propose changes in the direction of steepest descent.
3. **Implementation target**: Add a `_estimate_gradient()` method that runs 2*n trials (n hyperparameters) every K iterations, then uses the gradient to inform `propose_changes()`.
4. **Why it helps**: The trace reveals random perturbation wastes iterations on neutral or harmful changes (e.g., 8 tries with different LRs that all fail). Gradient estimation would directly suggest "increase LR slightly" or "decrease hidden dim slightly," converging faster than brute-force random search.
5. **Implementation complexity**: 4
6. **Risk**: medium — gradient estimation uses extra iterations (2 per hyperparameter), but pays off long-term

### Hypothesis 4: Warm-Start Transfer Learning
1. **Domain**: Meta-Learning / Fine-Tuning
2. **Core idea**: Cache the final model weights from the best iteration and reinitialize future runs from those weights instead of random initialization, with a small learning rate for fine-tuning.
3. **Implementation target**: Add a `_weight_cache` dict in `run_iteration()`, saving/loading model state via file I/O before each gpu_bench subprocess call.
4. **Why it helps**: Each iteration currently starts from scratch, wasting GPU cycles on re-learning the same representations. Warm-starting transforms hyperparameter search into a fine-tuning problem, allowing the runner to quickly test subtle changes (e.g., small LR adjustments) that would otherwise be hidden by training noise.
5. **Implementation complexity**: 3
6. **Risk**: medium — risk of overfitting to cached weights; need careful LR reduction for fine-tuning

**Recommended immediate priority**: Hypothesis 1 (adaptive constriction) for fastest impact on discard rate, followed by Hypothesis 4 (warm-start) to improve each trial's efficiency.