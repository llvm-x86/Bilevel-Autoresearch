## Critique of Hypotheses

### Hypothesis 1: Adaptive Perturbation Scaling

**Most likely failure mode**: The improvement rate metric will be noisy and lagging, causing the mechanism to *increase* perturbation size precisely when the runner is *about* to find a good configuration (i.e., after a long plateau), overshooting the optimum. This creates cyclic instability where perturbations grow, overshoot, shrink, stagnate, grow again—a feedback loop that wastes iterations.

**Implementation trap**: The running improvement rate must be computed over a *sliding window* of kept configurations only, but the ratio of "kept" to "total" is extremely sparse (4 keeps out of 22 total). With such sparse positive feedback, the improvement rate is either 0 or undefined most of the time, making the adaptive logic degenerate to "always increase" or "always decrease" based on a single data point.

**Evidence from trace**: The trace shows 22 iterations but only 4 keeps, and 17 consecutive discards after iter 4. The improvement rate over any reasonable window (e.g., 5 or 10 iterations) will be 0% for the vast majority of the run. The mechanism would perpetually increase perturbation size with no stabilization signal, eventually jumping randomly in parameter space.

**Score**: impact 2 × feasibility 3 ÷ complexity 3 = **2.0**

### Hypothesis 2: Momentum-Based Proposal Direction

**Most likely failure mode**: Momentum will lock the runner into a single direction (LR=0.003, BATCH_SIZE=128) while *blindly* varying HIDDEN_DIM, which is exactly what the trace shows is *already failing*. The momentum buffer will amplify the existing bias in the successful configurations, making the search even more narrow and less likely to discover the actual optimum (which may require different LR/BATCH_SIZE).

**Implementation trap**: Computing "vector differences" between configurations requires a metric space over hyperparameters with different scales (LR=0.001–0.01, BATCH_SIZE=64–256, HIDDEN_DIM=256–1024). Normalizing these is nontrivial: a 0.001 change in LR is proportionally large, while a 1-unit change in BATCH_SIZE is negligible. Without careful scaling, momentum will be dominated by the dimension with the largest numeric range (HIDDEN_DIM), not the most important one.

**Evidence from trace**: The 4 successful configs are (256, 0.003, 128), (384, 0.003, 128), (384, 0.003, 128), (512, 0.003, 128). The "successful direction" is essentially "keep LR and BATCH_SIZE constant, vary HIDDEN_DIM." Momentum would reinforce this exact pattern, but the trace already shows this pattern *stopped* yielding improvements after iter 4 (17 consecutive discards). The mechanism would cement a failing strategy.

**Score**: impact 3 × feasibility 3 ÷ complexity 4 = **2.25**

### Hypothesis 3: Controlled Random Restart with Success Memory

**Most likely failure mode**: The runner has only 22 total iterations. A random restart "wastes" at least 1–2 iterations to diverge from the current region, and another 3–5 iterations to re-converge if the restart is worse. With 17 consecutive discards, there is *room* for maybe 1–2 restarts, but these are more likely to jump to *worse* regions (given that the best config is stable across 4 instances) than to discover a better one. The runner could end the 22 iterations farther from the optimum than if it had continued local search.

**Implementation trap**: The "short-term memory" of recent successful configs must be carefully defined—if it includes the *current* best config, the restart distribution is centered there and just adds noise (equivalent to current perturbation). If it excludes it, the runner might forget the best config entirely. The variance parameter needs to grow with iteration count, but finding the right growth rate is fragile: too fast and restarts are useless, too slow and they never escape.

**Evidence from trace**: The runner already found 4 configs with val_bpb around 4.55–4.56, suggesting a genuine local basin. Random restarts are unlikely to find a deeper basin in 22 total iterations. The trace shows *no evidence* that the global optimum is far from the current region—only that local perturbation is failing to improve further.

**Score**: impact 3 × feasibility 4 ÷ complexity 3 = **4.0**

### Hypothesis 4: Contextual Bandit for Parameter Selection

**Most likely failure mode**: With 3 hyperparameter dimensions and 22 iterations, the bandit has at most 7–8 samples per "arm" (if arms = dimensions). With 22 iterations total and only 4 positive outcomes, the reward signal is *catastrophically sparse*. The bandit will converge to a suboptimal arm based on noise, and epsilon-greedy exploration will be indistinguishable from random selection. The result is *worse* than random perturbation because the bandit wastes iterations "exploiting" a dimension that happened to work once.

**Implementation trap**: Defining the "arms" is deceptively hard—should they be individual dimensions (3 arms), value ranges per dimension (e.g., LR < 0.003 vs > 0.003, 6 arms), or full configurations? With only 22 iterations, arms cannot share statistical strength, so the bandit degenerates to counting. The epsilon-greedy parameter needs tuning, but with sparse rewards, any epsilon below 1.0 causes premature convergence. A real contextual bandit would require features that are not available (iteration number is not a useful context).

**Evidence from trace**: The trace shows that *only concurrent changes* to all three parameters could potentially find improvement (since the current region is exhausted). The bandit, by selecting *which dimension* to perturb, actually prevents the simultaneous multi-dimension changes that might be necessary. The mechanism's fundamental assumption (that improvement is driven by selective dimension perturbation) is contradicted by the trace data.

**Score**: impact 2 × feasibility 2 ÷ complexity 5 = **0.8**

## Final Selection

**Selected**: Hypothesis 3 — Despite the risk of wasting iterations, it is the *only* hypothesis that addresses the trace's core failure mode (getting stuck in a local basin after iter 4) without reinforcing the already-failing pattern, and it offers the best impact-feasibility tradeoff given the short horizon.