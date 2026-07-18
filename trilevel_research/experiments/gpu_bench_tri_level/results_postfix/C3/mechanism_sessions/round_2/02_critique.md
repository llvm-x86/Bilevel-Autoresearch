## Rigorous Critique

### Hypothesis 1: Adaptive Perturbation-Scaling Based on Recent Gradient Sign

**Most likely failure mode**: Shrinking perturbations in response to oscillation can trap the search in a local minimum. The trace shows val_bpb oscillating between ~7-8, but this is uniform failure, not oscillation *around* an optimum. The mechanism would detect "sign flips every iteration" and shrink LR steps to near-zero—then the runner would propose LR=0.0025, 0.0026, 0.0024 forever, all still yielding val_bpb ~8. The assumption that oscillation implies *nearness* to optimum is false here; the landscape is simply flat-bad across the explored LR range.

**Implementation trap**: The sign-flip frequency metric is fragile. With 5 iterations, a single outlier (val_bpb=0.126) would zero out the trend and cause expansion into the same bad region. The `deque` needs careful handling of NaN/inf losses and the scaling factor needs bounds to prevent 1e-10 perturbations.

**Evidence from trace**: Weak. The trace shows monotonic failure (all LR variants had val_bpb ~7-8), not oscillation. The single good point (iter4, val_bpb=0.126) is an *outlier*, not part of a trend. The mechanism would fail to distinguish between "oscillating around good point" and "oscillating between uniformly bad points."

**Score**: impact (2) × feasibility (4) ÷ complexity (2) = **4.0**

---

### Hypothesis 2: Loss-Landscape Memorization via Exponential Decay Cache

**Most likely failure mode**: Premature pruning of promising regions due to poor distance metric. The config space mixes continuous (LR, weight_decay), discrete (batch_size), and categorical (activation functions). An L2 distance on normalized values would treat "LR=0.003, activation='relu'" as near "LR=0.003, activation='tanh'" even if the loss landscape is sharply different. The cache would reject good proposals near bad ones based on spurious similarity. Additionally, if the best point (iter4) is a narrow spike, caching it would cause rejection of all nearby variants, preventing refinement.

**Implementation trap**: The inverse-distance weighting requires a distance function across heterogeneous types. Encoding categorical variables as one-hot and computing Euclidean distance gives meaningless numbers. You'd need domain-specific normalization (e.g., LR scaled log, batch_size scaled log2) and a kernel that respects categorical separations. The "3 standard deviations" threshold assumes Gaussian loss distribution, which is violated when losses range 0.126 to 8.0.

**Evidence from trace**: Moderate. The trace confirms repeated exploration of the same bad region. However, only 15 iterations means the cache would have at most 5-6 distinct configs—too few for reliable interpolation. The mechanism needs 50+ samples to be useful.

**Score**: impact (3) × feasibility (3) ÷ complexity (3) = **3.0**

---

### Hypothesis 3: Autoregressive Proposal Targeting (APT) - Multi-Step Lookahead

**Most likely failure mode**: Overfitting the linear predictor to 15 data points across a high-dimensional space (LR, batch_size, hidden_dim, weight_decay...). With ~5 parameters to predict val_bpb_change, 15 samples gives 3:1 sample-to-parameter ratio—guaranteed overfitting. The model would learn spurious correlations (e.g., "all proposals failing → predict all proposals fail") and never propose anything. Worse, 2-step lookahead doubles the combinatorics (15^2 = 225 possible pairs), and the model would hallucinate paths that look good in simulation but fail in reality.

**Implementation trap**: Training a regression model online where the input space changes dimensionality every iteration (some configs have HIDDEN_DIM, others don't). Need fixed-length feature encoding with missing value handling. The "gradient in hyperparameter space" is ill-defined—discrete parameters have no gradient. The linear model would require one-hot encoding, exploding dimension count to 10+ with 15 samples.

**Evidence from trace**: Poor. The trace shows no evidence of beneficial multi-step paths—the best point was found by *random* change (iter4 used different batch_size AND lr_scaler), not by gradient following. Multi-step planning assumes smoothness that the trace disproves.

**Score**: impact (2) × feasibility (2) ÷ complexity (4) = **1.0**

---

### Hypothesis 4: Staged Exploration with Progressive Constriction

**Most likely failure mode**: The 5-iteration exploration phase is too short given the dimensionality. Random search over 5+ parameters in 5 iterations covers < 1% of any reasonable space. The constriction then shrinks to a hypercube around the best of 5 random points, which is likely far from the global optimum. The trace shows the runner already explores ALL iterations—the problem isn't lack of exploration but *what* it explores (only LR). Forced random exploration of HIDDEN_DIM with 1-2 samples is unlikely to find the 0.126 valley, which requires specific joint combination (lr=0.002, lr_scaler=2.0, batch_size=64).

**Implementation trap**: The "variance of recent losses" constriction rate is a circular dependency—if all losses are uniformly bad (variance low), constriction accelerates, trapping search in bad region. Need to track *best* loss variance separately. Also, switching from exploration to exploitation abruptly at iter 5 creates a discontinuity that discards exploration momentum.

**Evidence from trace**: Moderate. The trace does show pathological over-focus on LR. However, the single good point involved LR *and* batch_size change simultaneously—random search would need to hit that exact combination by chance, which is improbable in 5 tries. The trace suggests the optimal basin is so narrow that brute-force exploration won't find it.

**Score**: impact (2) × feasibility (4) ÷ complexity (2) = **4.0**

---

## Synthesis

| Mechanism | Score | Key Weakness |
|-----------|-------|-------------|
| 1 (Adaptive Scaling) | 4.0 | Diagnoses oscillation but trace shows flat failure, not oscillation |
| 2 (Memorization Cache) | 3.0 | Needs 50+ samples; heterogeneous distance metric is fragile |
| 3 (Multi-Step Lookahead) | 1.0 | Overfits on 15 samples; linear model can't capture joint interactions |
| 4 (Staged Exploration) | 4.0 | 5 iterations too few; constriction rate depends on misleading variance |

**Selected**: None—all four mechanisms fail to address the core issue revealed by the trace: the optimal basin (LR=0.002, lr_scaler=2.0, batch_size=64) requires *joint* modification of multiple hyperparameters simultaneously, but the runner only proposes single-dimension changes. The trace's failure mode is **pathological locality in proposal generation**, not insufficient exploration or poor gradient tracking. A better mechanism would be **joint perturbation sampling**—instead of proposing changes to one parameter, sample from a multivariate Gaussian over all continuous parameters with a covariance structure learned from past successful proposals.