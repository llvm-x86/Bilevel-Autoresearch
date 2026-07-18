Based on the trace analysis, the runner is stuck in a local optimum around batch_size=160 and failing to find improvements in other directions. The key issue is that the search strategy is too conservative and lacks systematic exploration of the hyperparameter space.

Here are four concrete mechanism improvements:

---

## Proposal 1: Adaptive Perturbation Based on Validation Improvement

**Domain**: Bayesian optimization / bandit algorithms

**Core idea**: Dynamically scale the magnitude of hyperparameter perturbations based on the gap between current and best val_bpb, using larger steps when far from optimum and finer steps when close.

**Implementation target**: `_generate_proposal()` method in `runner.py`

**Why it helps**: Currently all proposals use fixed step sizes regardless of performance proximity. When val_bpb is 0.027 (near-optimal), small steps around batch_size=160 fail to explore genuinely different regions. By scaling perturbation magnitude inversely with (current_val_bpb - best_val_bpb), the runner automatically expands search radius when performance plateaus, escaping the batch_size=160 local trap.

**Implementation complexity**: 3 (requires tracking val_bpb history and computing dynamic scaling)

**Risk**: medium (may overshoot if scaling too aggressive, but base magnitude can be tuned)

---

## Proposal 2: Gradient-Based Directional Search with Momentum

**Domain**: Optimization in machine learning (Adam/SGD momentum analogy)

**Core idea**: Track which hyperparameter directions (increase/decrease) have historically produced val_bpb improvements and bias proposals toward those directions with an exponential moving average of past successful deltas.

**Implementation target**: `_record_trial()` method and `_generate_proposal()` method

**Why it helps**: The trace shows batch_size increases consistently produce better results (64→128→160→0.027), yet the runner keeps trying decreases. A momentum term that remembers "batch_size ↑ → improvement" would bias proposals toward increasing batch_size and also explore other parameters (LR, HIDDEN_DIM) in their historically successful directions.

**Implementation complexity**: 4 (requires maintaining per-parameter momentum buffers and direction history)

**Risk**: low (momentum is naturally bounded and can be reset if performance degrades)

---

## Proposal 3: Multi-Resolution Batch Size Exploration

**Domain**: Multi-fidelity optimization / empirical convergence analysis

**Core idea**: After finding a promising batch_size region, systematically explore multiplicative factors (e.g., ×0.5, ×0.75, ×1.5, ×2.0) rather than fixed additive steps to cover the geometric space of batch sizes.

**Implementation target**: `_propose_param_change()` helper function

**Why it helps**: The current additive steps (64→96→128→160→200) are too dense near small values and miss larger gaps. Batch_size=160 is good, but perhaps ×2=320 or ×0.5=80 might yield even better noise/stability tradeoffs. Multiplicative steps better match the scale-invariant nature of batch-size effects on gradient variance.

**Implementation complexity**: 2 (simple transformation from linear to geometric proposal space)

**Risk**: low (multiplicative factors are standard in batch-size tuning literature)

---

## Proposal 4: Meta-Optimizer with Simulated Annealing Acceptance

**Domain**: Simulated annealing / evolutionary strategies

**Core idea**: Occasionally accept proposals that degrade val_bpb by a small amount (within a decaying temperature schedule) to escape local optima, rather than the strict "discard all non-improvements" policy.

**Implementation target**: `_evaluate_proposal()` method (acceptance logic)

**Why it helps**: The strict "keep only if val_bpb < best" policy creates a greedy hill-climber that gets stuck. Accepting proposals within, say, 5% of current best would allow exploration of neighboring configurations that might lead to better regions (e.g., trying LR=0.0015 with batch_size=128 instead of always discarding it). Temperature decays over iterations to gradually lock onto the best region.

**Implementation complexity**: 5 (requires temperature schedule, acceptance probability computation, and careful tuning of initial acceptance rate)

**Risk**: medium-high (may accept too many bad proposals if temperature not properly tuned, but can be offset by requiring periodic re-evaluation of best)