Based on the inner loop trace analysis, the primary bottleneck is that the search is making poor proposals that are being discarded, indicating a failure to effectively leverage the information gained from previous iterations. Here are 4 concrete mechanism improvements:

## Mechanism 1: Adaptive Perturbation Scheduling

1. **Domain**: Bayesian Optimization / Evolutionary Strategies
2. **Core idea**: Dynamically scale proposal perturbations based on recent success rate, shrinking step sizes after repeated failures to focus search near known good configurations.
3. **Implementation target**: `GpuBenchRunner._propose_next_hyperparams()` - add perturbation scaling factor that decays with consecutive discard count
4. **Why it helps**: Currently proposals like `LR: 0.0035` or `HIDDEN_DIM: 384` make large jumps from the best config (0.1286 bpb) and fail. After 3+ consecutive discards, reducing perturbation magnitude prevents wild leaps and encourages finer-grained exploration around the best configuration. This directly addresses the trace showing the search repeatedly overshooting away from the best known valley.
5. **Implementation complexity**: 2 (add a counter and scaling factor)
6. **Risk**: Low - only reduces step sizes, cannot make things worse

## Mechanism 2: Configuration Memory with Temporal Decay

1. **Domain**: Reinforcement Learning / Experience Replay
2. **Core idea**: Maintain a weighted history of all tested configurations and their val_bpb, using Gaussian similarity to bias proposals toward neighborhoods containing previously successful configurations.
3. **Implementation target**: `GpuBenchRunner` - add config_history database with `val_bpb` results, modify proposal generation to sample perturbations from regions of known success
4. **Why it helps**: The current mechanism seems to forget that `WEIGHT_DECAY=1e-05` at iteration 4 produced 0.1286 bpb, while iteration 16 with `WEIGHT_DECAY=5e-06` only got 0.8251. By remembering that small weight decay near 1e-05 works well, future proposals can stay closer to that value rather than randomly exploring far from the known optimum. The trace shows many proposals re-testing the same failed configurations (iter 9, 10, 13 repeatedly return to best config without exploration).
5. **Implementation complexity**: 4 (requires storing tuples, computing Gaussian similarity, and modifying proposal logic)
6. **Risk**: Medium - might overfit to early successes and miss better configurations far away

## Mechanism 3: Gradient-Aware Proposal Direction

1. **Domain**: Numerical Optimization / Gradient Descent
2. **Core idea**: Estimate the local gradient of the hyperparameter landscape by evaluating small epsilon perturbations in both directions (±) to determine which direction is promising before committing to larger steps.
3. **Implementation target**: `GpuBenchRunner.run_search()` - add an initial 2-iteration exploration phase per proposal point that tests small positive/negative deltas before proposing a larger change
4. **Why it helps**: The trace shows many unidirectional changes (e.g., iter 5 increases HIDDEN_DIM, iter 6 increases BATCH_SIZE) that fail because there's no feedback on whether the opposite direction would be better. By exploring both δ and -δ for each hyperparameter change, the search can identify valleys versus hills before committing to larger perturbations. This would prevent proposing `HIDDEN_DIM: 384` when maybe decreasing to 256 worked better in iter 4.
5. **Implementation complexity**: 5 (requires doubling evaluations per proposal step)
6. **Risk**: High - doubles computational cost per iteration, may not pay off if the landscape is noisy

## Mechanism 4: Confidence-Weighted Proposal Selection

1. **Domain**: Thompson Sampling / Bandit Algorithms
2. **Core idea**: Track the variance of outcomes for each hyperparameter setting and only propose changes to hyperparameters with high uncertainty (variance), while staying conservative on well-known parameters.
3. **Implementation target**: `GpuBenchRunner._select_hyperparameters_to_change()` - compute per-parameter success variance and threshold proposals to parameters with variance above some mean
4. **Why it helps**: The trace shows `LR: 0.003` is tested repeatedly (iter 2, 7, 8, 12, 17, 18) with wildly varying outcomes (3.02 to 8.03 bpb), yet the engine keeps proposing LR changes. Meanwhile, `HIDDEN_DIM: 256` is consistently better than 384 or 512. By only proposing changes to hyperparameters where outcomes are uncertain (high variance), and leaving stable ones alone, the search would avoid wasting iterations on LR permutations and instead explore more promising dimensions.
5. **Implementation complexity**: 3 (need to compute variance per parameter from history)
6. **Risk**: Low - naturally converges to exploration where it's needed most