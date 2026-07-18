## Critique of Proposed Mechanism Changes

### Hypothesis 1: Multi-fidelity Early Rejection with Warmup Checkpoint

**Most likely failure mode**:  
False early rejection of promising configurations. If val_bpb at 50% is not monotonically correlated with final val_bpb (e.g., some configurations converge slowly then jump), you could discard the eventual best configuration. The threshold of 4.0 is trivial—no discarded run had val_bpb > 0.7653, so a threshold of 4.0 catches nothing. You need a dynamic threshold like `2.0 * best_val_bpb_at_50%`, but that risks discarding the 0.0416 run if it was worse at 50%.

**Implementation trap**:  
The timing of the 50% checkpoint must be in training iterations, not wall time. If you checkpoint at `max_iter/2`, you must ensure the model state is evaluated correctly without affecting training. The callback must also handle the case where training terminates early—do you save partial results? This interacts with GpuBenchRunner's logging in non-obvious ways.

**Evidence from trace**:  
Weak. The trace shows no learning curves—we only see final val_bpb. The hypothesis that "hopeless" runs could be identified early is plausible but unsupported. The discarded runs span a range of 0.7653 to inf—some might have been climbing slowly. Without observing intermediate values, we cannot confirm monotonic improvement.

**Score**:  
Impact: 2 (threshold is trivially safe, so saves no budget)  
Feasibility: 4 (easy to implement callbacks in PyTorch)  
Complexity: 2 (simple early-stop logic)  
Final: 2 × 4 ÷ 2 = **4.0**

---

### Hypothesis 2: Local Perturbation Surrogate (Adaptive Search Radius)

**Most likely failure mode**:  
Shrinking the search radius exponentially (`exp(-0.5 * discard_rate_ma)`) will rapidly collapse to epsilon. With 21 discards out of 22, the discard rate is 0.9545. After 1 iteration, radius = `exp(-0.5 * 0.9545) ≈ 0.62`. After 5 iterations, `≈ 0.09`. After 10, `≈ 0.006`. The optimizer would be probing sub-percent changes in hidden_dim within 10 iterations—far too small to escape local minima. This guarantees missing any distant better region (e.g., hidden_dim=1024).

**Implementation trap**:  
The perturbation mechanism must preserve integer constraints (hidden_dim is a multiple of 64). Shrinking step size below 64 will produce zero-dimensional changes unless you round stochastically. The code must sample from a discrete distribution centered on the current value with variance proportional to the scaled radius—this is subtle and error-prone. Also, `discard_rate_ma` needs initialization: start with 0.5? 1.0? This choice dominates early behavior.

**Evidence from trace**:  
Moderate. The trace shows 21/22 failures, suggesting the search space is indeed narrow. However, the one success (hidden_dim=384) could be a local optimum, but there could be another better region at hidden_dim=128 or 1024 that was never sampled due to random chance. The hypothesis that shrinking helps assumes we are *already* in the global basin—which is unsupported.

**Score**:  
Impact: 3 (could find finer structure, but risks collapse)  
Feasibility: 2 (discrete perturbations are tricky to implement correctly)  
Complexity: 3 (needs state tracking, non-trivial math)  
Final: 3 × 2 ÷ 3 = **2.0**

---

### Hypothesis 3: Variance-Aware Acceptance Threshold (Annealing)

**Most likely failure mode**:  
Accepting configurations within 2 standard deviations of recent keeps when variance < 0.1 is dangerous. The observed val_bpb values are on a log-scale (bpb = bits per byte). A difference of 0.0485 (from 0.0416 to 0.0901) is a *doubling* of the loss—enormous in log-space. Accepting such configurations contaminates the "elite" population. Over 22 iterations, variance of *kept* values is essentially undefined (only 1 keep). With n=1, standard deviation is 0, so the condition `variance < 0.1` is trivially true, and you'd accept everything—defeating the purpose of optimization.

**Implementation trap**:  
Maintaining running variance requires storing at least 2 kept values. If you keep only 1, variance=0. If you keep 0, variance is NaN. The code must handle these edge cases gracefully (e.g., default to rejecting until n>=3). The annealing schedule (reduce acceptance width over time) requires a counter and schedule function—likely forgotten. Also, the threshold "2 standard deviations" assumes normal distribution of val_bpb, which is false for a log-loss.

**Evidence from trace**:  
Strong in spirit, weak in detail. The trace shows that iteration 17 (hidden_dim=640, val_bpb=0.0901) was discarded despite being only ~2x worse than best. This *feels* like a missed opportunity. But the hypothesis that minor perturbations cause minor drops is contradicted by the data: hidden_dim=512 (val_bpb=0.7653) is 18x worse, not minor. The "flat region" assumption is unproven.

**Score**:  
Impact: 2 (accepts bad configs, pollutes population)  
Feasibility: 1 (variance estimation with single keep is impossible)  
Complexity: 4 (non-trivial threshold logic, edge cases)  
Final: 2 × 1 ÷ 4 = **0.5**

---

### Hypothesis 4: Warm-Start Restart with Best-Seed Ensemble

**Most likely failure mode**:  
Re-running the best configuration with different seeds wastes budget if the original seed was not anomalous. The trace shows hidden_dim=384 achieved val_bpb=0.0416—likely near-optimal. Running it 3 more times with seeds 42, 73, 101 might give 0.0420, 0.0418, 0.0409—confirming robustness. But this uses 3 trials that could have explored hidden_dim=256 or 320. If seed variance is low (<1%), this is a net loss—you learn nothing new.

**Implementation trap**:  
Resetting the optimizer state after restarts is complex. If GpuBenchRunner maintains a proposal history (e.g., for Bayesian optimization), inserting 3 cold-start trials must not contaminate the GP model. The seed override must be passed correctly to the training script without breaking reproducibility. Also, "3 consecutive discards" is ambiguous: after iteration 12's success, iter 13-15 are discards—do you restart at iter 16? Or after iter 16's discard? This logic must be precise.

**Evidence from trace**:  
Weak. The trace shows deterministic hyperparameter exploration—no seed variation tested. There is zero evidence that seed effects cause >1% variance. The hypothesis that 0.0416 was a "fluke" is speculative. Given the RX 580's non-deterministic kernels, seed variance is plausible but not quantified.

**Score**:  
Impact: 2 (wastes budget on confirmation)  
Feasibility: 3 (seed logic is straightforward, but state reset is hard)  
Complexity: 2 (simple counter and seed buffer)  
Final: 2 × 3 ÷ 2 = **3.0**

---

## Selected: Hypothesis 2 — Local Perturbation Surrogate

**Why**: Despite its risks of collapse, it's the only hypothesis that directly addresses the root cause (overly aggressive search steps) with a mechanism that can recover fine-grained structure near the one known good point. The failure mode (collapse) can be mitigated by a minimum radius floor (e.g., hidden_dim step size ≥ 64). The implementation trap (discrete rounding) is solvable with stochastic rounding: `floor((current + gaussian_noise * radius) / 64) * 64`. The score suffers from complexity, but impact potential is highest because it changes how proposals are generated at the source, not how they are filtered or re-run. Given a single promising point (384, 0.0416) and no evidence of distant better regions, local focus is justified.