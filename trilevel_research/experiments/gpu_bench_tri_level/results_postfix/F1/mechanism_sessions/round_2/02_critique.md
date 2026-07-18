## Critique of Mechanism Proposals

### Mechanism 1: Adaptive Perturbation Budget

**Most likely failure mode**: This mechanism will *actively harm* the search by keeping bad configurations alive longer than necessary. The trace shows the runner is *already* stuck in a local optimum—the problem is not premature discarding, but lack of exploration. Adding patience means the runner will waste even more iterations confirming that val_bpb ~0.0296 is indeed the best, while proposals that are *clearly* divergent (val_bpb > 7.0) get extra run time before being discarded. This makes the convergence problem *worse*, not better.

**Implementation trap**: The "patience counter" logic is deceptively simple. The hard part is defining "consecutive poor performances" relative to what baseline? Against current best? Against a running average? Against the configuration's own history? Wrong choice leads to either never discarding (if threshold too high) or immediate discarding (if too low), effectively reproducing the original behavior.

**Evidence from trace**: The trace shows proposals at iterations 4, 6, 7, 9 producing val_bpb values 10–1000x worse than best (0.0296). These are not "slow to converge"—they are fundamentally broken configurations. Keeping them alive for 3 iterations would waste ~15 evaluations on clearly dead ends.

**Score**: 1 × 3 ÷ 2 = **1.5**

---

### Mechanism 2: Elastic Search Radius with Decay

**Most likely failure mode**: Wide perturbations in early iterations (changing 2-3 params simultaneously) will cause *catastrophic divergence*. The trace shows that even changing 2 params (e.g., LR AND BATCH_SIZE vs just LR) often produces val_bpb > 3.0. Changing 3 params simultaneously (e.g., LR + BATCH_SIZE + DROPOUT) is combinatorially more likely to land in a dead zone. The runner needs *smarter* exploration, not *wider* random exploration.

**Implementation trap**: The hardest part is defining "which 2-3 params to change simultaneously" without creating a random brute-force search. If you pick any 2-3 params uniformly, you'll repeatedly hit the same dead combinations (e.g., high LR + small BATCH_SIZE). The mechanism needs a *correlation model* to avoid re-exploring known bad combos, but that's not specified and would dramatically increase complexity.

**Evidence from trace**: Iteration 5 changed HIDDEN_DIM from 64→128 and LAYERS from 3→2 and got val_bpb=4.37. Iteration 7 changed LR + BATCH_SIZE + DROPOUT and got val_bpb=22.3. These suggest that multi-parameter changes are *already* failing—not because they're too narrow, but because the search lacks gradient information to guide *which* combos to try.

**Score**: 2 × 3 ÷ 4 = **1.5**

---

### Mechanism 3: Gradient-Based Hyperparameter Sensitivity Analysis

**Most likely failure mode**: Finite difference estimates from noisy evaluations (single runs with stochastic training) will produce *misleading gradients* that actively steer the search away from good regions. The val_bpb values in the trace show high variance (e.g., two identical configs at different seeds give 3.45 vs 7.54). Finite differences require smooth, deterministic functions—training loss landscapes are neither. The mechanism will converge on spurious correlations.

**Implementation trap**: The hardest part is maintaining the *history alignment*—finite differences require paired comparisons where exactly one hyperparameter changes. But the trace shows proposals often change 2-3 params simultaneously, making it impossible to attribute val_bpb changes to specific parameters. The `SensitivityTracker` would need to either (a) force single-parameter changes (limiting exploration) or (b) solve a high-dimensional regression with N << D (underdetermined system). Neither is addressed.

**Evidence from trace**: The claim that "BATCH_SIZE and HIDDEN_DIM are frequently explored but never improve" is *not supported*. The trace never explores BATCH_SIZE=256 with LR=0.003 (the best LR). It explores BATCH_SIZE=256 only at LR=0.001 (too low) or LR=0.01 (too high). The failure is in *coordinated* exploration of (LR, BATCH_SIZE) pairs, not sensitivity of individual parameters.

**Score**: 2 × 2 ÷ 5 = **0.8**

---

### Mechanism 4: Multi-Fidelity Pruning with Warmup Phase

**Most likely failure mode**: The warmup phase will systematically *discard the actual best configuration*. The trace shows the best config (LR=0.003, BATCH_SIZE=128, etc.) was found at iteration 2. If a warmup had been applied, it would have needed to declare this config "promising" after only 10% of training—but early training often has high variance and may not show improvement until later epochs. The mechanism trades exploration speed for accuracy, and the wrong choice of threshold or warmup length will kill the one good config the runner has.

**Implementation trap**: The hardest part is designing the *warmup scoring function* that correlates well with final val_bpb. The proposal suggests `val_bpb < current_best * 1.5`. But the trace shows val_bpb values spanning 0.0296 to 22.3—a 750x range. Any fixed threshold will either (a) let everything through (threshold high) or (b) kill everything (threshold low). The mechanism needs *adaptive thresholding* based on warmup duration, which is non-trivial to tune.

**Evidence from trace**: The trace shows 7 proposals total over 10 iterations. The compute budget is already very small (10 iterations). Adding a warmup phase that uses 20% of budget means only ~8 full evaluations instead of 10—a 20% reduction in search iterations. Given that the runner is stuck after iteration 2, reducing the search budget makes it *less* likely to escape.

**Score**: 2 × 2 ÷ 5 = **0.8**

---

## Summary Scores

| Mechanism | Impact | Feasibility | Complexity | Final Score |
|-----------|--------|-------------|------------|-------------|
| 1: Patience Budget | 1 | 3 | 2 | 1.5 |
| 2: Elastic Radius | 2 | 3 | 4 | 1.5 |
| 3: Sensitivity Analysis | 2 | 2 | 5 | 0.8 |
| 4: Warmup Pruning | 2 | 2 | 5 | 0.8 |

---

**Selected**: None.

**Reason**: All 4 proposals either fail to address the root cause (local optimum entrapment due to restricted (LR, BATCH_SIZE) pairing) or introduce mechanisms that would actively worsen the problem (wasting budget, discarding the best config, following spurious gradients). The runner needs *directed exploration* that maintains the discovered good LR=0.003 while systematically perturbing other params—none of these proposals achieve that.