## Critique of Proposed Mechanism Improvements for GpuBenchRunner

---

### Hypothesis 1: Adaptive Mutation Rate Based on Improvement Gradient

1. **Most likely failure mode**: **Gradient estimation noise destroys search coherence.** The "improvement gradient" across discrete iterations is a noisy, sparse signal—especially when >50% of proposals are discarded with zero improvement. False plateaus will trigger premature step-size collapse, while random single-step gains will cause over-zealous expansions, turning the search into chaotic oscillation. The mechanism implicitly assumes a smooth loss landscape, but the trace shows discrete jumps (e.g., iter 9 → iter 10: val_bpb 0.0443 → 0.0438) that look更像是 lucky leaps than gradient-driven progress.

2. **Implementation trap**: **State tracking across failed proposals.** The current code generates a proposal, runs it, then either accepts or discards. If a proposal is discarded (no improvement), does the gradient use the discarded config's result or skip it? Storing the last N *successful* improvement deltas requires preserving config vectors alongside their val_bpb deltas, and handling the case where 3-5 consecutive proposals fail (gradient = 0, triggering unbounded step-size expansion). The boundary condition where step_size exceeds the valid range of a hyperparameter (e.g., LR > 1.0) is not trivial.

3. **Evidence from trace**: Partially supports. Iterations 12-16 show LR stuck at 0.003 with tiny ±0.0001 perturbations and zero improvement—a textbook plateau. However, the *cause* may not be mutation step size but rather that the optimal LR *for this config* simply isn't 0.003. The trace also shows that larger jumps (iter 4: LR=0.005→0.003) *do* produce gains, suggesting the current fixed step size is already near-optimal for exploration. The proposed mechanism could shrink step size *further* (making things worse) or expand it *at the wrong time*.

4. **Score**:  
   Impact: 3 (moderate — could break the one thing working)  
   Feasibility: 4 (well-studied, easy to code)  
   Complexity: 3  
   **Final: 3 × 4 ÷ 3 = 4.0**

---

### Hypothesis 2: Crossover Between Top-K Configurations

1. **Most likely failure mode**: **Crossover produces hyperparameter vectors that violate domain constraints and degenerate search.** Weighted interpolation of categorical or bounded parameters (e.g., HIDDEN=512 and HIDDEN=768 → 640) creates a value *never tested*, which is useful in theory. In practice, for neural network hyperparameters, intermediate values often correspond to *unusual* model widths or learning rates that don't align with standard design patterns, causing training instability. More critically, crossover between configs with *different* batch sizes, LR schedules, or optimizer types can produce semantically meaningless hybrids. The trace shows only 2 truly good configs (iter 4 and iter 9)—interpolating between them creates a config that may combine the *worst* aspects of both.

2. **Implementation trap**: **Hall-of-fame archive management and diversity preservation.** The naive approach stores top-5 configs by val_bpb and generates weighted averages. This fails when the archive converges—all 5 configs become nearly identical, crossover produces clones, and diversity collapses. The hardest part: deciding whether to crossover hyperparameters *independently* (each param averaged) or *as a tuple* (swap entire configs). The former destroys covariance between hyperparameters (e.g., LR and WD that work together). The latter is just random subset selection. Neither is trivial to get right without explicit diversity maintenance.

3. **Evidence from trace**: Weakly supports. There are exactly 2 distinct good clusters (LR=0.003/WD=1e-6/HIDDEN=768 and LR=0.003/WD=1e-5/HIDDEN=512). Crossover *could* produce a hybrid, but the trace also shows that small LR perturbations around 0.003 *never* improved after iter 9, suggesting that the issue is not configuration composition but a fundamental limit of the architecture (val_bpb ~0.0438). Crossover won't escape this basin.

4. **Score**:  
   Impact: 2 (high risk of degenerate configs)  
   Feasibility: 3 (archive logic is tricky)  
   Complexity: 4  
   **Final: 2 × 3 ÷ 4 = 1.5**

---

### Hypothesis 3: Automatic Learning Rate Schedule Injection

1. **Most likely failure mode**: **The cosine schedule prematurely decays the LR too fast, underfitting good configs.** The searched LR (e.g., 0.003) becomes the *maximum* LR, decaying to 0. This means the effective LR during training spends most of its time *below* the searched value. For a deep transformer model, reducing LR below 0.001 may cause convergence to sharp minima or stop training progress entirely. The trace shows val_bpb improving steadily from 0.0443 → 0.0438 with fixed LR=0.003—injecting a decay could *slow* or *halt* this improvement. Furthermore, the runner's outer search assumes LR is fixed during training; changing to a schedule invalidates all previous search results, requiring complete retraining.

2. **Implementation trap**: **Inconsistent optimizer state between schedule-aware and non-schedule configurations.** The runner compares val_bpb across proposals trained with different LR schedules (old: fixed, new: cosine). This confounds the effect of the schedule with the LR value itself. To implement correctly, the LR schedule must be a *property of the search space*, not a post-hoc injection—meaning every proposal after injection must use cosine, and previous results become incomparable. The hardest part: the runner's `TrainingLoop.create_optimizer()` currently takes a flat config dict; adding a schedule requires a structural change to how LR is passed to the training subprocess.

3. **Evidence from trace**: **Does not support.** The trace shows the runner performing *better* with fixed LR=0.003 (val_bpb 0.0438) than with lower LRs like 0.002 (iter 8: 0.0450) or 0.001 (iter 3: 0.0458). This suggests the model benefits from sustained high LR, not decay. A cosine schedule would spend most time at low LR, likely worsening performance. The improvement plateau at iter 9-16 is not due to LR being too static but rather the architecture reaching its representational capacity limit.

4. **Score**:  
   Impact: 1 (likely makes things worse)  
   Feasibility: 5 (trivial to code)  
   Complexity: 2  
   **Final: 1 × 5 ÷ 2 = 2.5**

---

### Hypothesis 4: Validation-Based Early Termination with Warm Restarts

1. **Most likely failure mode**: **Premature termination of configurations that would improve later, destroying search diversity.** The trace shows that configs reaching val_bpb ~7-8 early *never* improve—this is true, but the reason is their high LR (0.005-0.01) causes divergence. Early termination at 20% training would catch these and kill them, which is good. However, for borderline configs (e.g., LR=0.003, WD=1e-6), val_bpb starts at 0.0450 and slowly improves—20% may not be enough to distinguish eventual winners from losers. More critically, warm restarts (resetting optimizer) would repeatedly *destroy* the optimizer's momentum, potentially turning a slow-improving config into a permanently stuck one.

2. **Implementation trap**: **Modifying the external gpu_bench binary or adding validation monitoring.** The runner launches gpu_bench as a subprocess. To support early termination, the runner must either (a) parse stdout live to detect val_bpb trajectories, or (b) modify gpu_bench to accept a `--early-stop` flag. Option (a) is fragile to logging changes; option (b) requires a separate codebase change and recompilation. The hardest part: synchronizing termination signals across subprocesses without leaving orphaned GPU processes. The complexity of 5 is accurate—this is an infrastructure change masquerading as an algorithm change.

3. **Evidence from trace**: Partially supports. Iterations 3, 8, 13, 18 converge to val_bpb ~7-8 quickly and stay there—these are strong candidates for early termination. However, the iteration cadence (~24 iterations in ~24 hours) suggests compute is not the bottleneck; the runner is already exploring 1 config/hour. Early termination would only help if the *number of iterations* is fixed (it isn't—the runner stops after B attempts). The trace shows the runner is limited by *quality*, not quantity—adding more bad configs faster doesn't help.

4. **Score**:  
   Impact: 2 (premature termination risk)  
   Feasibility: 2 (requires binary mod)  
   Complexity: 5  
   **Final: 2 × 2 ÷ 5 = 0.8**

---

### Summary Table

| Hypothesis | Impact | Feasibility | Complexity | Score |
|------------|--------|-------------|------------|-------|
| H1: Adaptive Mutation | 3 | 4 | 3 | **4.0** |
| H2: Crossover | 2 | 3 | 4 | **1.5** |
| H3: LR Schedule | 1 | 5 | 2 | **2.5** |
| H4: Early Termination | 2 | 2 | 5 | **0.8** |

---

### Selected: **Reject all hypotheses** — the trace shows a fundamental architecture/space limitation (val_bpb ~0.0438 is a plateau that no search mechanism change can escape), and all four proposals either misdiagnose the problem or introduce risks that outweigh their marginal benefit; instead, focus on expanding the search space to include larger model sizes or different architectures.