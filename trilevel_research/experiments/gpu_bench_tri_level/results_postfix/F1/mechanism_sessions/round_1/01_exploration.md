Based on the trace analysis, the GpuBenchRunner is stuck in a local optimum (LR=0.003, BATCH_SIZE=128, HIDDEN_DIM=256) despite exploring many variations. The key failure mode is that it keeps returning to the same frozen point. Here are 4 concrete mechanism improvements:

---

## Mechanism 1: Adaptive Perturbation Scaling with Momentum

**Domain**: Optimization / Evolutionary Strategies (CMA-ES lineage)

**Core idea**: Dynamically scale perturbation magnitude based on performance history and maintain a momentum term to escape local valleys.

**Implementation target**: `_generate_proposal()` method in runner.py

**Why it helps**: Currently all proposals modify LR perturbatively around 0.003, but none are large enough to escape the basin. Adaptive scaling maintains larger perturbations when stuck (many consecutive discards), while momentum prevents the runner from immediately snapping back to the known good point. This creates a "search inertia" that can push through plateaus.

**Implementation complexity**: 3/5

**Risk**: Medium — if momentum is too large, it could overshoot and never return to good regions. Requires careful tuning of adaptation rate and momentum coefficient.

---

## Mechanism 2: Multi-Axis Random Subspace Sampling

**Domain**: Bayesian Optimization / Randomized Search

**Core idea**: Instead of changing only LR (or LR+one other param), randomly select 2-3 axes to jointly perturb in a coordinated manner.

**Implementation target**: `_choose_parameters_to_modify()` method in runner.py

**Why it helps**: Looking at the trace, iterations 7-8 (which tried LR+BATCH_SIZE+HIDDEN_DIM together) still failed because they used decreasing LR. The real failure is that the search is 1-dimensional (mostly just LR). Joint perturbations can discover interactions—e.g., maybe higher HIDDEN_DIM *requires* higher LR, or lower BATCH_SIZE *requires* weight decay. By sampling random subspaces, we test combinations that isolated perturbations cannot reach.

**Implementation complexity**: 2/5

**Risk**: Low — adds exploration diversity without sacrificing stability, as coordinated changes can discover new regimes.

---

## Mechanism 3: Exploration Bonus Based on Local Entropy

**Domain**: Reinforcement Learning / Intrinsic Motivation

**Core idea**: Add a bonus to proposal selection for regions of hyperparameter space that have been explored less, computed as a kernel density estimate of visited points.

**Implementation target**: `_select_best_proposal()` or new scoring method in runner.py

**Why it helps**: The runner is stuck in a very narrow region (LR ∈ [0.0025, 0.003], all other params near baseline). It keeps proposing small variations of the same point. An entropy bonus would force exploration of genuinely different hyperparameter regimes—e.g., trying LR=0.01 with HIDDEN_DIM=512, or weight decay of 0.01, which the current process never attempts.

**Implementation complexity**: 4/5 (requires maintaining a visited-point buffer and density estimation)

**Risk**: Low — exploration bonuses are well-understood and can be annealed as search progresses. Won't degrade baseline performance because good points are always kept.

---

## Mechanism 4: Adaptive Proposal Rejection with Temperature Annealing

**Domain**: Simulated Annealing / Markov Chain Monte Carlo

**Core idea**: Instead of always keeping the best point, occasionally accept "worse" proposals with a probability that decreases over iterations (simulated annealing schedule).

**Implementation target**: `_apply_proposal()` or the decision loop in runner.py

**Why it helps**: The current mechanism is purely greedy—if val_bpb increases even slightly, the proposal is discarded. This creates a local trap where no escape is possible. By probabilistically accepting worse results (especially early in search), the runner can climb out of basins and discover the true global optimum. The decreasing acceptance probability ensures convergence.

**Implementation complexity**: 3/5

**Risk**: Medium — if temperature schedule is too aggressive, it may accept too many bad proposals early and waste compute. If too conservative, it's equivalent to greedy search. However, annealing theory provides good defaults (geometric schedule, temperature = initial_bpb_range * 0.95^iter).