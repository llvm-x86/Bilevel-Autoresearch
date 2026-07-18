## Rigorous Critique of Proposed Mechanism Improvements

### 1. Bayesian Optimization with Gaussian Process Surrogate

**Most likely failure mode**: **Model misspecification collapse** — The GP assumes smoothness of the response surface, but the trace shows *discontinuous* val_bpb jumps (0.12 → 6.8+). With only 3-4 distinct successful configurations observed, the GP will overfit to those points, nearly always predicting Expected Improvement near them, and never exploring the large empty regions. This actually *strengthens* the local optimum trap rather than escaping it.

**Implementation trap**: **History poisoning from failed evaluations** — The binary crashes (iterations 2, 9, 14) return no val_bpb. If you impute these with a penalty, the GP learns a false landscape. If you skip them, the GP sees "holes" in the domain and becomes overconfident about uncertainty in those regions. Getting imputation right requires understanding *why* the binary failed (memory OOM? parameter validation error? stochastic crash?), which the trace doesn't reveal.

**Evidence from trace**: Weak support. The landscape is *not* smooth — HIDDEN_DIM=512 (iter 5) gets 0.35, HIDDEN_DIM=640 (iter 7) gets 17.64. That's an order-of-magnitude difference from 10% parameter increase. GPs with RBF kernels will smooth this into a false gradient suggesting 576 is optimal, wasting iterations.

**Score**: Impact=2 × Feasibility=2 ÷ Complexity=4 = **1.0**

### 2. Adaptive Perturbation Magnitude Scaling

**Most likely failure mode**: **Oscillation trap** — The EMA of discard rate lags behind the true state. If the search finds a slightly better config (val_bpb 0.30 instead of 0.28), discard rate drops, noise scale increases, and the system jumps back to high-val_bpb regions. The EMA ensures you're *always* adjusting to the past, never the present, causing limit cycles between too-large and too-small mutations.

**Implementation trap**: **Boundary effects with bounded parameters** — LEARNING_RATE and WEIGHT_DECAY are log-scale bounded (e.g., 1e-4 to 1e-2). When noise scale is large, you frequently sample outside bounds and clip. Clipping collapses variance, making the mutation effectively deterministic (always returning the bound). The adaptive scale then shrinks, but the damage is done — you're stuck at boundary with no escape.

**Evidence from trace**: Moderate support. Iterations 5-8 show exactly this pattern: small mutations (single parameter changes) are rejected, suggesting scaling *down* (the proposed response) would make them even smaller, further reducing exploration — exactly the wrong direction.

**Score**: Impact=3 × Feasibility=4 ÷ Complexity=2 = **6.0**

### 3. Cross-Validation Informed Rejection Sampling

**Most likely failure mode**: **False confidence from correlated splits** — On a dataset this small, 3 splits have high overlap (especially if k-fold with shuffle). The variance *between* splits is actually smaller than the variance *within* a single evaluation due to stochasticity in the MLP initialization. You get three nearly identical val_bpb values, compute a tight confidence interval, and confidently reject configurations that are actually promising but unlucky with their random seed.

**Implementation trap**: **Multiple hypothesis testing corruption** — After N iterations, you've implicitly tested N×3 configurations. The significance threshold must be Bonferroni-corrected (α/N), which rapidly becomes so strict that *nothing* passes. Without correction, you'll accept a "significant" improvement after ~20 iterations purely by chance (p-value shopping). The trace shows 21 iterations — this is exactly the regime where multiplicity matters.

**Evidence from trace**: Strong support. Iteration 16 gives HIDDEN_DIM=384, LR=0.003, val_bpb=0.35 — close to the 0.28 optimum. This is plausibly a "real" good config that got rejected by noise. But iteration 5 (HIDDEN_DIM=512, LR=0.003) gets 0.35 too — which one is truly better? Cross-validation would help distinguish, but the standard errors overlap, so it wouldn't change the decision.

**Score**: Impact=4 × Feasibility=3 ÷ Complexity=3 = **4.0**

### 4. Population-Based Training (PBT) Style Warm Restart

**Most likely failure mode**: **Premature convergence of the entire population** — PBT's "exploit" step copies the best performer's hyperparameters to all underperformers. If one config gets lucky (e.g., iteration 4's HIDDEN_DIM=256, LR=0.001, val_bpb=0.28), after 1-2 exploit rounds, *all* population members will be clones of it. The "explore" step then perturbs them, but they're all starting from identical weights → identical next perturbations → loss of diversity. You end up with a single-config search that's 5x more expensive.

**Implementation trap**: **Checkpoint management in crashing binaries** — The trace shows crashes at iterations 2, 9, 14 (21% failure rate). PBT requires saving model weights at every exploit/explore boundary. If the binary crashes during save, you lose the entire population state. If it crashes during load, you need fallback logic to reinitialize from scratch. The complexity (rating 5) is correct — you essentially need a fault-tolerant distributed system for one search.

**Evidence from trace**: Strong support. The sequential search clearly gets stuck; parallel exploration *would* help. But the trace also shows that *most* configurations are terrible (val_bpb > 6.0). A population of 6 would likely have 5 bad members and 1 good one, meaning 83% of compute is wasted on bad configs. PBT's exploit step would then collapse to the good one anyway.

**Score**: Impact=4 × Feasibility=2 ÷ Complexity=5 = **1.6**

---

## Synthesis

| Hypothesis | Impact | Feasibility | Complexity | Score |
|------------|--------|-------------|------------|-------|
| 1. Bayesian Optimization | 2 | 2 | 4 | 1.0 |
| 2. Adaptive Scaling | 3 | 4 | 2 | 6.0 |
| 3. Cross-Validation Rejection | 4 | 3 | 3 | 4.0 |
| 4. PBT Warm Restart | 4 | 2 | 5 | 1.6 |

**Selected**: Hypothesis 2 — Adaptive Perturbation Magnitude Scaling, because it's the only proposal where the impact-to-complexity ratio favors deployment: it directly addresses the trace pattern of stuckness, requires minimal code changes, and its failure modes (oscillation, boundary clipping) are detectable and correctable at runtime.