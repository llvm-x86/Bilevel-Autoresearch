## Proposed Mechanism Improvements for GpuBenchRunner

### Hypothesis 1: Bayesian Optimization with Gaussian Processes

1. **Domain**: Automated Machine Learning (AutoML) / Bayesian Optimization
2. **Core idea**: Replace the current heuristic exploration with a Gaussian Process surrogate model that predicts promising hyperparameter configurations based on all historical trials, not just the last best.
3. **Implementation target**: `propose_next_config()` method (or equivalent config suggestion logic) in runner.py
4. **Why it helps**: Current trace shows the runner gets stuck proposing LR=0.003 repeatedly (iterations 5-11) despite constant discard signals. A Bayesian model would learn from these failures and shift exploration to other hyperparameter spaces (e.g., varying HIDDEN_DIM or BATCH_SIZE more aggressively). It maintains an uncertainty estimate, balancing exploitation near known good regions with exploration of untested configurations that have high potential.
5. **Implementation complexity**: 4 (requires integrating a lightweight GP library or implementing a simple GP from scratch)
6. **Risk**: medium (surrogate model may initially propose poor values until enough training data accumulates; cold start problem)

### Hypothesis 2: Adaptive Temperature Sampling for Hyperparameter Exploration

1. **Domain**: Reinforcement Learning / Simulated Annealing
2. **Core idea**: Implement a decaying "temperature" parameter that controls how far new proposals deviate from the current best configuration, starting with wide exploration and gradually refining.
3. **Implementation target: `calculate_exploration_radius()` or within the proposal generation logic in runner.py
4. **Why it helps**: The trace reveals premature narrowing of search—after iter 4's success, the runner repeatedly tries LR=0.003 with tiny perturbations. An adaptive temperature would force larger jumps in early iterations (e.g., testing HIDDEN_DIM in [64, 1024] or BATCH_SIZE in [16, 512]), then gradually decrease perturbation magnitude as the search converges. This mimics the successful simulated annealing pattern and prevents getting trapped in local optima after one good hit.
5. **Implementation complexity**: 2 (simple mathematical decay function applied to random perturbation magnitude)
6. **Risk**: low (easily tunable decay rate; can always anneal slowly)

### Hypothesis 3: Multi-Armed Bandit with Upper Confidence Bound (UCB)

1. **Domain**: Online Learning / Bandit Algorithms
2. **Core idea**: Treat each hyperparameter dimension as an independent bandit arm using UCB selection, allowing the runner to systematically allocate trials to the most promising parameter values while still exploring under-tested regions.
3. **Implementation target**: `select_hyperparameter_dimension()` and `choose_value()` methods in runner.py
4. **Why it helps**: Current failure mode: repeated same LR value despite strong negative signals. UCB would track the win rate of each LR candidate (e.g., 0.003, 0.006, 0.012) and automatically rebalance exploration toward rarely-tested values like HIDDEN_DIM=512 or BATCH_SIZE=32, which only appeared once each in the trace. It provides a principled mathematical framework for the explore-exploit tradeoff, with confidence intervals automatically shrinking for frequently tested values and expanding for under-sampled ones.
5. **Implementation complexity**: 3 (requires maintaining per-dimension statistics and UCB formula)
6. **Risk**: medium (may need careful tuning of exploration hyperparameter c; cold start with uniform initialization)

### Hypothesis 4: Gradient-Free Local Optimization (Nelder-Mead / Pattern Search) Around Best Config

1. **Domain**: Numerical Optimization / Direct Search
2. **Core idea**: After identifying a promising configuration, launch a Nelder-Mead simplex or pattern search that systematically probes neighboring hyperparameter combinations using a geometric simplex of points, rather than random perturbations.
3. **Implementation target**: `local_refinement_phase()` called after a new best val_bpb is found, replacing random single-parameter changes
4. **Why it helps**: The trace shows the runner found a decent config at iter 4 (LR=0.012, likely with default HIDDEN_DIM and BATCH_SIZE) but then failed to improve via isolated single-parameter changes. Nelder-Mead evaluates a simplex of 5-6 nearby configurations simultaneously, catching interactions between hyperparameters (e.g., LR=0.008 combined with HIDDEN_DIM=384 might yield better bpb than any single-variable change). It automatically contracts around promising regions and expands away from poor ones.
5. **Implementation complexity**: 3 (straightforward implementation of simplex operations; need to maintain state across iterations)
6. **Risk**: low (Nelder-Mead is robust and has been successfully applied to ML hyperparameter tuning for decades)