Looking at the trace, the core bottleneck is that the runner keeps proposing LR variations that fail dramatically (val_bpb ~7-8 vs best 0.12). The problem isn't exploration breadth but proposal quality - the search is trapped proposing minor LR tweaks that all fail, suggesting the cost function landscape has a narrow basin of attraction.

## Mechanism 1: Adaptive Perturbation-Scaling Based on Recent Gradient Sign

**Domain**: Bayesian optimization / trust-region methods

**Core idea**: Scale proposal perturbations proportionally to the consistency of recent gradient directions in validation loss, expanding search when loss decreases consistently and contracting when it oscillates.

**Implementation target**: `GpuBenchRunner` class, `propose_changes()` method - add a `_compute_loss_trend()` helper that tracks sign of (val_bpb_{t-1} - val_bpb_t) over last 5 iterations and scales the perturbation magnitude inversely to sign-flip frequency.

**Why it helps**: The trace shows LR proposals jumping between 0.002 and 0.004 but always landing at val_bpb ~8. This suggests the optimal basin is extremely narrow. By detecting that sign flips every iteration (oscillation), the mechanism would shrink perturbations to sub-millimeter scales, preventing the runner from jumping out of the narrow basin that produced iter4's 0.126.

**Implementation complexity**: 2 (add 15 lines, track deque of last 5 losses)

**Risk**: Low - only affects proposal magnitude, never prevents any region from being explored

## Mechanism 2: Loss-Landscape Memorization via Exponential Decay Cache

**Domain**: Bandit algorithms / Thompson sampling with memory

**Core idea**: Maintain a exponentially-weighted cache of (hyperparameter → val_bpb) mappings, and before proposing any change, query the cache to reject proposals that map near historically-bad regions (val_bpb > 3 standard deviations above current best).

**Implementation target**: `GpuBenchRunner.__init__()` - add `self.proposal_memory = {}` dictionary; in `run_iteration()` before calling `_evaluate_proposal()`, compute L2 distance from proposed config to all cached configs and interpolate expected loss using inverse-distance weighting.

**Why it helps**: The trace shows 15 consecutive proposals all producing val_bpb ~7-8. This is a waste of compute - the runner keeps exploring a region proven to be catastrophic. A memory cache would flag "LR=0.003 → 8.0" and treat any proposal within epsilon of that LR-space as high-risk, forcing exploration of entirely different hyperparameters (e.g., HIDDEN_DIM, BATCH_SIZE).

**Implementation complexity**: 3 (need distance metric between heterogeneous configs, 30 lines)

**Risk**: Medium - could prematurely prune promising regions if cache is too aggressive; need temperature parameter

## Mechanism 3: Autoregressive Proposal Targeting (APT) - Multi-Step Lookahead

**Domain**: Meta-learning / MAML-inspired optimization

**Core idea**: Instead of proposing single-step changes, propose pairs of (change1, change2) where change1 is expected to degrade val_bpb transiently but change2 (which depends on change1) recovers to a lower val_bpb than current best, using a simple linear predictor trained online on the observed gradient in hyperparameter space.

**Implementation target**: `GpuBenchRunner.run_iteration()` - after evaluating each proposal's val_bpb, train a lightweight linear regression model mapping (previous_config, proposed_change) → val_bpb_change. Use this model to simulate 2-step proposals and select those predicted to yield net improvement.

**Why it helps**: Current runner tries isolated LR changes which all fail because the gradient direction is misleading (all LR changes near 0.003 → worse). A 2-step lookahead could discover that "first increase LR to escape plateau, then decrease dramatically" yields net improvement. The trace shows the best config is isolated - this helps find paths to it.

**Implementation complexity**: 4 (need sklearn or custom linear regression, feature encoding for categorical HPs, 60+ lines)

**Risk**: High - linear model on sparse data may overfit; could generate unreasonable proposals

## Mechanism 4: Staged Exploration with Progressive Constriction

**Domain**: Evolutionary algorithms / simulated annealing

**Core idea**: Implement explicit exploration/exploitation phases: first 5 iterations explore broadly across ALL hyperparameter dimensions (not just LR), then constrict to a shrinking hypercube around the best found config, with the constriction rate inversely proportional to the variance of recent losses.

**Implementation target**: `GpuBenchRunner.__init__()` - add `self.exploration_phase = True`, `self.phase_counter = 0`; in `propose_changes()`, during exploration phase randomly select hyperparameter dimension to change with uniform random magnitude; switch to exploitation after 5 iters.

**Why it helps**: The trace shows ALL 22 proposals only modify LR (except iter5 which also changes BATCH_SIZE). This is a pathological search space narrowing. By forcing exploration of HIDDEN_DIM, WEIGHT_DECAY, BATCH_SIZE, and architectural parameters, the mechanism can discover whether the 0.126 valley is robust or if improvement requires non-LR changes.

**Implementation complexity**: 2 (add phase flag and dimension selection logic, 20 lines)

**Risk**: Medium - first 5 iterations waste compute on random search, but trace already wastes 15 iterations; net benefit expected