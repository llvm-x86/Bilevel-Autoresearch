Based on the iteration trace, the core problem is that the search gets stuck in a high-loss plateau (~7-8 val_bpb) after the initial lucky drop to 0.176, and cannot escape because proposals only make tiny perturbations around the failed region. The mechanism needs to detect when the search is failing and force exploration of radically different hyperparameter regimes.

## Proposal 1: Catastrophic Divergence Detector & Recovery

**Domain**: Reinforcement learning exploration-exploitation balance

**Core idea**: When more than N consecutive proposals are discarded with val_bpb > 5× the best known, force a drastic random restart from the best config with heavily modified optimizer state.

**Implementation target**: `GpuBenchRunner._evaluate_config()` or new method `_detect_divergence()`

**Why it helps**: The trace shows 17 consecutive discards after iteration 4, with values never dropping below ~7.8 (except iteration 16 which was a fluke). The search is stuck in a high-loss attractor basin. By detecting this pattern and forcing a reset to the best config with fresh optimizer momentum, we avoid wasting iterations on dead ends.

**Implementation complexity**: 3 (add a counter in the inner loop, check after each discard, trigger a random mutation of the best config)

**Risk**: Low. Best-case: escapes the plateau. Worst-case: wastes a few iterations on random restarts that also fail (but not worse than current ~7.8 loss).

---

## Proposal 2: Adaptive Perturbation Magnitude Scaling

**Domain**: Evolutionary strategies / population-based training

**Core idea**: Scale hyperparameter perturbation magnitude inversely proportional to the gap between current loss and best loss — when far from best, make much larger jumps.

**Implementation target**: `GpuBenchRunner._propose_changes()` or `_mutate_config()`

**Why it helps**: Currently, all proposals perturb LR by ±0.001-0.002 regardless of the massive loss gap. When val_bpb is 7.9 vs best 0.176, moving LR from 0.006 to 0.005 is pointless. Instead, if loss gap > 1.0, allow perturbations of ±50% of the base value (e.g., try LR=0.0001 or LR=0.05). Once near the best regime, shrink perturbations back to ±10%.

**Implementation complexity**: 2 (add a scaling factor based on `current_val_bpb / best_val_bpb` ratio)

**Risk**: Medium. Large perturbations could cause NaN loss or training instability, but the discard mechanism handles that. Risk of overshooting the best region, but that's better than never finding it.

---

## Proposal 3: Hyperparameter Space Expansion via Log-Uniform Sampling

**Domain**: Bayesian optimization / hyperband

**Core idea**: Replace the current linear perturbation of LR with log-uniform sampling from a wider range (1e-5 to 1.0) when the search has failed to improve for >5 iterations.

**Implementation target**: `GpuBenchRunner._propose_changes()` or new `_sample_hyperparams()`

**Why it helps**: The trace shows the search only tries LR values between 0.003 and 0.008. The optimal LR might be orders of magnitude different (e.g., 0.0001 for a stable training regime). Log-uniform exploration would discover this. The current linear sampling cannot escape the local LR band.

**Implementation complexity**: 1 (add a flag to switch between linear/log sampling after consecutive failures)

**Risk**: Low. Log-uniform sampling is standard practice in HPO. Worst case: we sample a terrible LR that gets discarded.

---

## Proposal 4: Multi-Objective Cascade (LR First, Then Other Params)

**Domain**: Curriculum learning for HPO

**Core idea**: Separate the search into phases: first optimize only LR until val_bpb stabilizes below a threshold (e.g., < 0.5), then add weight decay and batch size perturbations.

**Implementation target**: `GpuBenchRunner.__init__()` with phase tracking, `_propose_changes()` to restrict params per phase

**Why it helps**: The trace shows weight decay is being introduced (iter 2, 5, 6, 7, 9, 10) while LR is still clearly terrible (val_bpb ~8). Mixing multiple hyperparameters when the primary one (LR) is far from optimal only adds noise and makes attribution harder. By focusing on LR exclusively until loss drops dramatically, we find the correct learning rate regime faster.

**Implementation complexity**: 2 (add a phase counter, phase transition condition based on val_bpb, filter param changes by phase)

**Risk**: Medium. If LR optimization never finds the 0.176 valley (it was a lucky outlier), we might never explore weight decay. But the adaptive scaling from Proposal 2 mitigates this by making LR search more aggressive when stuck.