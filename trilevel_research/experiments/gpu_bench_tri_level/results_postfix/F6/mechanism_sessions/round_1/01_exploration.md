## Proposed Mechanism Improvements for GpuBenchRunner

### Hypothesis 1: Bayesian Optimization with Gaussian Processes

**Domain**: Automated Machine Learning (AutoML)

**Core idea**: Replace random grid search with Bayesian optimization using Gaussian Process regression to model the response surface of val_bpb over hyperparameter space, enabling smarter candidate proposals.

**Implementation target**: `_propose_next_config()` method in runner.py

**Why it helps**: The current iteration trace shows massive discard rates (16/22 = 73% rejection). The system repeatedly tests similar LR values around 0.003 even though the optimal LR (0.006) was found early. Bayesian optimization builds a probabilistic model of the objective function, exploiting promising regions (LR ~0.006 with weight decay ~0.0003) while exploring uncertain areas. This directly addresses the "too many discards" bottleneck by producing higher-quality proposals from the start, using prior observations to avoid revisiting known-bad configurations.

**Implementation complexity**: 4 (requires integrating a GP library like scikit-learn or GPyTorch, implementing acquisition functions like Expected Improvement)

**Risk**: medium (GP approximations can be noisy with small sample sizes; requires careful kernel selection for categorical + continuous hyperparameters)

---

### Hypothesis 2: Hyperparameter Population Warm-Starting

**Domain**: Evolutionary Optimization

**Core idea**: Maintain a population of top-k hyperparameter configurations with mutation strengths proportional to their val_bpb improvement, rather than single-point extrapolation.

**Implementation target**: `_main_loop()` and `_select_next_proposal()` in runner.py

**Why it helps**: The current mechanism discards nearly everything because it greedily branches from the single best "keep" configuration (iter 17: LR=0.006, WD=0.0003). This creates a narrow search path. With a population of say 3-5 elite configurations, each with its own mutation variance derived from their individual improvement history, the search naturally explores multiple promising basins simultaneously. For example, if iter 5 (LR=0.006, no WD) also shows promise, both branches get explored with different decay schedules.

**Implementation complexity**: 3 (adds configuration history management and mutation schedule per elite)

**Risk**: low (simple additive change; maintains backward compatibility with single-config search)

---

### Hypothesis 3: Adaptive Learning Rate Rewarming with Validation Feedback

**Domain**: Learning Rate Scheduling Theory

**Core idea**: Implement a mechanism that, after every N discards without improvement, automatically introduces LR adjustments via cosine annealing with restarts centered on the best-known LR.

**Implementation target**: `_adjust_learning_rate_strategy()` method in runner.py

**Why it helps**: The trace shows the system gets stuck in a local valley around LR=0.003/0.006, trying multiple variations that all fail. This resembles neural network optimization getting trapped in sharp minima. By cyclically rewarming the LR (e.g., ±20% from best LR every 3-4 failed proposals), the search can escape local plateaus. The validation feedback directly triggers the restart, linking exploration dynamism to performance stagnation.

**Implementation complexity**: 2 (requires tracking consecutive discard count and applying cosine schedule modifications)

**Risk**: medium (could overshoot good configurations if rewarming amplitude is too large; needs tuning of restart frequency)

---

### Hypothesis 4: Hyperparameter Sensitivity Clock with Diminishing Returns

**Domain**: Experimental Design in Science

**Core idea**: Implement a "sensitivity clock" that tracks how much each hyperparameter has been explored relative to its observed variance in val_bpb, and automatically reduces search granularity for parameters showing low sensitivity.

**Implementation target**: `_estimate_parameter_importance()` and `_prune_search_space()` in runner.py

**Why it helps**: 13 of the 21 proposals changed LR while only 4 meaningfully varied weight decay, and BATCH_SIZE/HIDDEN_DIM were barely touched after iter 4/3. The system wastes computation testing LR repeatedly. A sensitivity clock computes local variance of val_bpb for each hyperparameter change. Parameters where >90% of changes produce <5% val_bpb variation get locked to their best value, redirecting resources to parameters with higher leverage. This directly prevents the observed pathological repetition of LR variations that all yield val_bpb ~7.9-8.0.

**Implementation complexity**: 3 (needs running variance computation and pruning logic, but no external libraries)

**Risk**: low (conservative by default; can be tuned to lock parameters only after high confidence)