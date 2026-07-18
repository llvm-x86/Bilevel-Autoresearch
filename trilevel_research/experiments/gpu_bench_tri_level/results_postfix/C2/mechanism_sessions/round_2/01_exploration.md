Based on the inner loop trace analysis, I'll propose concrete mechanism changes to help GpuBenchRunner find lower val_bpb on the AMD RX 580 HIP MLP benchmark.

## Hypothesis 1: Adaptive Perturbation Scaling

**Domain**: Bayesian Optimization / Evolutionary Strategies

**Core idea**: Dynamically scale the magnitude of hyperparameter perturbations based on recent improvement rate, making larger moves when stuck and smaller moves near optima.

**Implementation target**: `GpuBenchRunner._propose_changes` method — modify perturbation logic to adjust step sizes proportionally to the running improvement rate over the last N iterations.

**Why it helps**: The trace shows the runner is stuck oscillating around the best configuration (iter 4) with small perturbations that rarely improve val_bpb (only 2 keeps out of 22 iterations). By detecting when improvement stalls (e.g., no improvement in 5+ iterations), the mechanism can temporarily increase perturbation magnitudes to escape local optima, then reduce them when approaching better regions. This balances exploration vs. exploitation dynamically.

**Implementation complexity**: 3 (modify existing method, add running statistics, implement adaptive scaling logic)

**Risk**: Low — additive change that doesn't break existing functionality; can be tuned with min/max perturbation bounds and a decay factor.

## Hypothesis 2: Momentum-Based Proposal Direction

**Domain**: Gradient-based Optimization (SGD with momentum)

**Core idea**: Accumulate a "momentum" vector of recently successful perturbation directions to bias proposals toward promising regions of hyperparameter space.

**Implementation target**: `GpuBenchRunner._propose_changes` — add a momentum buffer that tracks the vector difference between kept configurations and uses an exponentially weighted average to bias future perturbations.

**Why it helps**: The trace shows 4 successful configurations (iters 0, 2, 4, 16) with different HIDDEN_DIM values but same LR=0.003 and BATCH_SIZE=128. The runner repeatedly discards configurations that deviate from this pattern. Momentum would accumulate a bias toward LR=0.003 and BATCH_SIZE=128 while still allowing stochastic exploration around the best-proven region, increasing the probability of finding nearby improvements.

**Implementation complexity**: 4 (need to track config state, compute vector differences, maintain momentum buffer, apply bias with configurable momentum coefficient)

**Risk**: Medium — could overly constrain search if momentum is too high; need careful reset logic when improvement plateaus to avoid getting stuck.

## Hypothesis 3: Controlled Random Restart with Success Memory

**Domain**: Simulated Annealing / Multistart Optimization

**Core idea**: Introduce periodic random restarts with config disturbance proportional to iteration count since last improvement, while maintaining a short-term memory of recent successful hyperparameter combinations.

**Implementation target**: `GpuBenchRunner.run` loop — add restart logic: after N discards without improvement, randomly select a new starting point from a distribution centered on the best-performing configuration but with increased variance, and reset perturbation history.

**Why it helps**: The trace shows 17 consecutive discards after iter 4, indicating the runner is trapped in a local minimum. Random restarts would periodically inject fresh diversity, potentially discovering entirely new parameter regions. The memory component ensures the runner can quickly return to the best-known region if the random restart fails, preventing catastrophic forgetting.

**Implementation complexity**: 3 (add iteration counter, implement restart condition, modify config initialization with variance)

**Risk**: Medium — too frequent restarts could waste iterations; need to balance restart frequency vs. local exploitation, and ensure restarts don't discard valuable exploration history.

## Hypothesis 4: Contextual Bandit for Parameter Selection

**Domain**: Multi-Armed Bandits / Reinforcement Learning

**Core idea**: Replace random perturbation with a lightweight contextual bandit that learns which parameter dimensions are most promising to perturb based on the current iteration's context (e.g., recent improvement rate, current config distance from best).

**Implementation target**: `GpuBenchRunner._propose_changes` — add a simple epsilon-greedy bandit with separate arms for each hyperparameter dimension, updating arm rewards based on whether perturbations to that dimension led to val_bpb improvement.

**Why it helps**: The trace shows many proposals that change only LR or BATCH_SIZE individually, but these rarely improve. The bandit would learn that simultaneous changes to specific parameter pairs (LR+BATCH_SIZE) or certain value ranges are more promising. Over iterations, it would allocate more proposal attempts to the most promising perturbation strategies, naturally converging toward the optimal configuration.

**Implementation complexity**: 5 (need to define parameter "arms" — could be individual dimensions or predefined perturbations, implement reward tracking, epsilon-greedy selection, and periodic updates)

**Risk**: High — bandit learning may be slow with only 22 iterations; could require careful reward shaping and may not outperform random search on short horizons. However, it provides a principled framework for learning from past proposals' success rates.