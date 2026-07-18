## Critique of Hypotheses

### Hypothesis 1: Adaptive Perturbation Decay

**Most likely failure mode**: Premature convergence to suboptimal local minimum. The trace shows the current best (HIDDEN_DIM=1024, LR=0.003) results in val_bpb=0.0052, but there's no evidence this is globally optimal. Narrowing search radius too aggressively (k=0.15) will prevent discovering better configurations that require larger jumps (e.g., HIDDEN_DIM=2048 with a different LR).

**Implementation trap**: The decay function `exp(-k * (iter - best_iter))` uses absolute iteration count, but the "best_iter" variable updates dynamically when a better config is found. This creates a sawtooth reset problem: every time a better config is found, the search radius resets to maximum, then decays again. This is actually *cyclic expansion* when it replaces a better config, not continuous decay.

**Evidence from trace**: Weak. The trace shows iter 15-21 all use ~1024 HIDDEN_DIM with LR~0.003, but only 3 distinct configurations were actually tested beyond iter 15. The failure isn't that perturbations are too large—they're already small (HIDDEN_DIM changes of ±128, LR changes of ±0.0005). The issue is *all* nearby points are worse, suggesting a sharp local optimum, not that the radius is too large.

**Score**: (impact 2 × feasibility 4) ÷ complexity 2 = **4.0**

### Hypothesis 2: Directional Gradient-Guided Perturbation

**Most likely failure mode**: Over-fitting to spurious correlations. The claim "HIDDEN_DIM increases consistently improved val_bpb" is misleading—it improved *once* (768→1024). There's no repeated directional signal. Momentum-based biasing on a single positive gradient will aggressively push HIDDEN_DIM toward extreme values (2048, 4096) while ignoring that the optimal could be at 1152 or 896. This destroys the ability to fine-tune around the true optimum.

**Implementation trap**: The EMA of "successful delta vectors" has a critical dimensional normalization issue. HIDDEN_DIM operates on scale O(1000), while LR operates on scale O(0.001). Without per-parameter normalization, the momentum vector will be dominated by HIDDEN_DIM changes and effectively ignore LR entirely. But the trace shows LR perturbations were equally likely to be harmful—this is a *scale artifact*, not a meaningful signal.

**Evidence from trace**: Weak to moderate. There is ONE example where HIDDEN_DIM increase helped, and ONE where LR increase hurt. Two data points do not constitute a directional gradient. The system has 21 iterations total, and only ~5-6 actually tested different configs (many were "discarded" due to validation). Statistical significance is nonexistent.

**Score**: (impact 1 × feasibility 2) ÷ complexity 4 = **0.5**

### Hypothesis 3: Exploration Budget with Forced Random Restarts

**Most likely failure mode**: Throwing away hard-won progress. The current best config (val_bpb=0.0052) is excellent—near-zero validation loss. Random restarts to "alternative basins" with distinct LR (0.001) and HIDDEN_DIM (512) would almost certainly produce val_bpb > 0.5, wasting compute. The system isn't stuck in a bad local minimum; it's *already at near-optimal performance*. The correct behavior is local refinement, not random restart.

**Implementation trap**: The "consecutive discard streak > 5" trigger fires at iter 21 (discards at 16-21 inclusive = 6). But the *reason* for discards matters: are perturbations rejected because they're genuinely worse, or because of a bug? The trace shows *every* perturbation away from the best config is worse—this suggests the optimum is genuinely sharp, not that exploration is needed. A restart threshold should check for *plateau improvement rate*, not discard count.

**Evidence from trace**: Strong for the *observed behavior* (stuck), but the *cause attribution* is wrong. The trace doesn't show there are alternative better basins—it shows the current basin is optimal. Forced restarts would destroy the 0.0052 val_bpb progress with high probability.

**Score**: (impact 3 × feasibility 3) ÷ complexity 3 = **3.0**

---

## Synthesis

The critical flaw across all hypotheses: **they assume the system is in a bad local minimum that needs escape, but the trace evidence shows it's in a very good global optimum that needs careful local refinement.** The val_bpb=0.0052 is suspiciously good—nearly zero—which raises another possibility: the model is overfitting to a small validation set, or there's a data leak.

**Selected**: Hypothesis 1 — it's the only one that preserves the current best while reducing wasteful large perturbations, though the decay-on-best-reset trap must be fixed to decay from global iteration count instead.