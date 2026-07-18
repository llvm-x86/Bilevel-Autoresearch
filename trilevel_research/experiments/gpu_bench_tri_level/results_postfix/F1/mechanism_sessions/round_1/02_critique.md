## Critique of Proposed Mechanisms

---

### Mechanism 1: Adaptive Perturbation Scaling with Momentum

**Most likely failure mode**: The momentum term will cause the search to oscillate or diverge because the loss landscape is non-stationary—the validation BPB may spike not because of hyperparameters but because of training noise. Momentum accumulates across unrelated perturbations, amplifying random fluctuations into large, meaningless jumps. The runner could end up proposing LR=1e-6 or LR=10.

**Implementation trap**: The adaptation rate must be tuned relative to the acceptance rate, but acceptance decisions are noisy. If you count 5 consecutive discards and double perturbation scale, you're implicitly assuming all 5 were "wasted." But the 5th discard might have been close to improving—doubling perturbation now could skip over the actual optimum. Worse, you need to separate "discard due to bad hyperparams" from "discard due to training noise," which is impossible without multiple trials per point.

**Evidence from trace**: The trace shows the runner already tries LR values in [0.001, 0.005], which is a 5x range. The issue isn't perturbation magnitude—it's that *none of the variations produce lower BPB*. Adaptive scaling won't help if the basin itself is shallow and any departure increases loss. The runner needs *different combinations*, not *larger steps in the same dimension*.

**Score**: impact 2 × feasibility 3 ÷ complexity 3 = **2.0**

---

### Mechanism 2: Multi-Axis Random Subspace Sampling

**Most likely failure mode**: By randomly selecting 2-3 axes, you dramatically increase the proposal variance. The runner might never converge because proposals are too "wild"—LR=0.1 with BATCH_SIZE=8 with HIDDEN_DIM=1024 simultaneously is a guaranteed failure, and these bad experiences could pollute the acceptance buffer, making the runner reluctant to explore even good combinations later.

**Implementation trap**: The random subspace selection must be stratified—you can't pick axes uniformly because some parameters have different scales and sensitivity. For example, weight decay typically matters at 1e-4 scale while LR matters at 1e-3 scale; randomly coupling them with equal perturbation magnitude is a recipe for collapse. You'd need per-parameter perturbation scales calibrated to observed sensitivity, which is a meta-optimization problem.

**Evidence from trace**: Trace iterations 7-8 precisely tried a multi-axis change (LR+BATCH_SIZE+HIDDEN_DIM) and it *failed*—resulting in the highest BPB in the trace (2.23). This suggests that unprincipled joint perturbations are worse than single-axis changes. The hypothesis that "combinations can discover interactions" is correct in theory, but without knowing *which* interactions are productive, random subspace sampling is as likely to hit catastrophic combinations as beneficial ones.

**Score**: impact 3 × feasibility 2 ÷ complexity 2 = **3.0**

---

### Mechanism 3: Exploration Bonus Based on Local Entropy

**Most likely failure mode**: The entropy bonus will fight against the greedy acceptance, creating a "thrashing" behavior—the runner visits a new region, gets a worse BPB due to expected variance, discards it, but the bonus makes it try *another* new region, and another, never settling anywhere. The buffer fills with sparse, bad points that don't provide density information for meaningful exploration guidance.

**Implementation trap**: Kernel density estimation in high-dimensional discrete-continuous mixed spaces (BATCH_SIZE is integer, LR is continuous) requires choosing a kernel bandwidth per dimension and a distance metric that makes sense across dimensions. A change of BATCH_SIZE=128→64 is "closer" than LR=0.003→0.001? By what metric? Poor bandwidth choice will either make all points seem equally distant (no bonus effect) or make only exact duplicates get penalized (trivial bonus). Getting this right requires extensive tuning that the traced runner doesn't have.

**Evidence from trace**: The runner *has* tried genuinely different configurations: iteration 9 tried LR=0.003 with BATCH_SIZE=48 (weird choice), iteration 4 tried LR=0.008 (unusual). These failed not because they were "too similar" to the baseline, but because they produced worse BPB. The entropy hypothesis assumes the runner hasn't explored—it has explored, but exploration produced nothing better. An entropy bonus would only force exploration of *even more* bad configurations.

**Score**: impact 1 × feasibility 2 ÷ complexity 4 = **0.5**

---

### Mechanism 4: Adaptive Proposal Rejection with Temperature Annealing

**Most likely failure mode**: The runner will accept a sequence of bad proposals, each worse than the last, and drift into a truly terrible region from which the annealing schedule (which only decreases acceptance probability) cannot return. Since proposals are greedy in the *current* best point, accepting a worse point means the next proposals are based on a bad center—compounding the error.

**Implementation trap**: The acceptance criterion needs to compare against the *current* best point or the *current* proposed point? Standard simulated annealing accepts worse moves relative to current position, not absolute best. But the runner's loop stores the single best point globally. If you accept a worse proposal, the stored best point becomes stale—proposals should be relative to the *current* point (to enable random walk), but the runner architecture expects proposals relative to the *best* point (to ensure monotonic improvement). These two paradigms are fundamentally incompatible: you'd need to rewrite the proposal generation to track a "wandering" point separate from the best-so-far point.

**Evidence from trace**: There is *no evidence* that accepting worse proposals would help. The runner has discarded configurations that increased BPB, but it has also tried many variations that *decreased* BPB and then bounced back. The trace shows iteration 4-5-6 as a sequence where BPB went 2.048→2.047→2.046—a clear downward trend interrupted when iteration 7 (multi-axis) spiked. The runner *can* find better points; it just can't sustain improvement. Annealing won't fix the root cause: that perturbations are fundamentally 1D and never discover the right combination.

**Score**: impact 4 × feasibility 1 ÷ complexity 3 = **1.33**

---

## Summary Table

| Mechanism | Impact | Feasibility | Complexity | Score |
|-----------|--------|-------------|------------|-------|
| 1. Adaptive Momentum | 2 | 3 | 3 | 2.0 |
| 2. Multi-Axis Sampling | 3 | 2 | 2 | 3.0 |
| 3. Entropy Bonus | 1 | 2 | 4 | 0.5 |
| 4. Simulated Annealing | 4 | 1 | 3 | 1.33 |

**Selected**: Mechanism 2 — Multi-Axis Random Subspace Sampling.

**Why**: It has the highest score (3.0) and directly addresses the trace's root cause—the runner is stuck because it only perturbs LR while keeping other parameters fixed, so it never discovers that the local optimum might require coordinated changes (e.g., higher LR *with* lower BATCH_SIZE *with* different HIDDEN_DIM). The low feasibility score (2) acknowledges the implementation trap of unprincipled joint perturbations, but this can be mitigated by using observed sensitivity to scale per-axis perturbation magnitudes, keeping the implementation simple while adding the needed exploration diversity.