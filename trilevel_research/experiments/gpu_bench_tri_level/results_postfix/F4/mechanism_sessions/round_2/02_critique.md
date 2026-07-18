## Critique of Proposed Mechanism Changes

### Hypothesis 1: Bayesian Optimization Surrogate

**Most likely failure mode**: GP overconfidence in extrapolation. With only ~25 data points in a 4D space, the GP will produce high uncertainty estimates everywhere except near the single good point (iteration 4). Expected Improvement will aggressively sample near that point, leading to more iterations in the 7.9-8.0 plateau rather than escape. The plateau is flat, so GP will struggle to find gradient signal.

**Implementation trap**: Kernel hyperparameter fitting on-the-fly. The `GaussianProcessRegressor` will re-fit its length scales each iteration. With fewer than 5*dim points, the optimizer will converge to degenerate solutions (near-zero length scales on one dimension, infinite on others). You must freeze kernel params after initial fit or use strong priors.

**Evidence from trace**: The trace shows NO correlation structure—iteration 4 (LR=0.001, WD=0.0001) gives 0.1153, but iteration 17 (LR=0.001, WD=0.0002) gives 7.9885. GP assumes smoothness; the actual landscape is discontinuous. GP will fail catastrophically.

**Score**: (impact: 2 × feasibility: 4) ÷ complexity: 5 = **1.6**

### Hypothesis 2: Multi-armed Bandit with UCB1

**Most likely failure mode**: Non-stationarity kills the bandit. The optimal hyperparameter region shifts as training progresses—a dimension that helped at iteration 5 may hurt at iteration 15. UCB1 cannot unlearn previously high-reward arms; once WD gets high UCB scores from early iterations, it will be over-selected even after the landscape changes, causing performance degradation.

**Implementation trap**: Reward normalization across iterations. The raw `-Δval_bpb` can be positive or negative, and magnitudes vary wildly (iteration 4: -7.87 improvement vs iteration 5: +0.02 degradation). Normalizing rewards per iteration is essential but introduces latency bias. Using unnormalized rewards will cause the bandit to chase outliers.

**Evidence from trace**: The trace does NOT support single-dimension attribution. Look at iteration 4: simultaneous changes to LR (+0.001→0.002) AND WD (+0.0001→0.0002) produced the only breakthrough. Disentangling would miss this interaction. The bandit tests one dimension at a time, which guarantees it will never reproduce the best result.

**Score**: (impact: 3 × feasibility: 4) ÷ complexity: 2 = **6.0**

### Hypothesis 3: Adaptive Search Radius with Success-Replay Buffer

**Most likely failure mode**: Step size oscillation trap. When the runner escapes the plateau by large steps, it immediately shrinks step size upon success. But the new region may require different step sizes; immediate shrinkage re-creates the same plateau condition. The mechanism oscillates: 5 large steps, 1 success, shrink, 5 medium steps, no success, 5 large steps—cycling forever.

**Implementation trap**: The multiplicative factor 1.2 grows exponentially—after 10 consecutive discards, step size is 0.05 × 1.2¹⁰ = 0.31 (6× original). At this scale, perturbations will overshoot the global optimum region entirely, jumping from val_bpb=8.0 to val_bpb=20+ (training divergence). The linear increase is too aggressive; logarithmic growth (e.g., 1.05) is safer but slower.

**Evidence from trace**: STRONG evidence supports this. Iterations 5-21 show 17 consecutive discards with val_bpb stuck at 7.9-8.0. The trace explicitly says "too many discards, 'discard': True for all remaining iterations." This is the exact failure mode that adaptive radius addresses.

**Score**: (impact: 4 × feasibility: 5) ÷ complexity: 1 = **20.0**

### Hypothesis 4: Covariance Matrix Adaptation with Elite Selection

**Most likely failure mode**: Covariance collapse to zero. With only 1 elite sample per iteration (20% of ~5 recent samples), the rank-μ update will shrink covariance along all dimensions to near-zero after 3-4 iterations. The runner will stop exploring entirely, making even smaller perturbations than the current fixed σ. The plateau escape requires increasing variance, but CMA-ES inherently decreases variance over time.

**Implementation trap**: Elite set selection in non-stationary environment. The "top 20% of recent samples" assumes samples are i.i.d. from a stationary distribution. But the runner's sequential sampling means later samples are conditioned on earlier ones. Using recent 3 iterations' data violates CMA-ES assumptions; the covariance estimate will be biased toward the current plateau region.

**Evidence from trace**: The trace shows NO evidence of correlation structure that CMA-ES would exploit. LR and WD both produce flat val_bpb=8.0 when changed independently or together—there's no ridge to follow. CMA-ES would waste computing eigendecompositions on a diagonal matrix, providing no benefit over random perturbations.

**Score**: (impact: 4 × feasibility: 3) ÷ complexity: 5 = **2.4**

---

## Final Recommendation

### Selected: Hypothesis 3 — It directly addresses the trace's "all iterations discarded" pathology with minimal complexity, and the failure mode (step size oscillation) can be mitigated by capping maximum step size at 2× and using logarithmic growth.

**However, I must note a critical flaw in all hypotheses**: None address the root cause—iteration 4's success with LR=0.002 and WD=0.0002. The trace shows that exact combination, yet subsequent iterations never try it again. The simplest fix (not proposed): cache the best config and try it again every 5 iterations to verify it still works. This would immediately identify if the landscape shifted or if iteration 4 was a fluke.