Based on the trace analysis, the main bottleneck is that the runner's proposal strategy is too unstable—most changes discard the best config (val_bpb ~0.1248) and yield much worse values (~7-8). This suggests the inner loop lacks a mechanism to stabilize exploration around promising regions and exploits random walks instead of evidence-driven search. Below are 4 mechanism improvements.

---

### **Hypothesis 1: Trust-Region Proposal Constraint**
1. **Domain**: Bayesian optimization (trust-region methods)
2. **Core idea**: Proposals that deviate too far from the current best hyperparameters should be rejected or penalized until evidence supports broader exploration.
3. **Implementation target**: `propose_hyperparameters()` method in the runner's proposal logic (likely in `runner.py` or a parent search controller). Add a check: if any hyperparameter change exceeds a distance threshold (e.g., `|new_lr - best_lr| > 0.001` *or* `batch_size` changes by more than 2x relative to current best), then generate a new proposal.
4. **Why it helps**: The trace shows that almost all changes that alter more than one parameter (e.g., iter 2, 5, 9, 19) produce high val_bpb. Constraining to small, single-parameter changes near the best config prevents large jumps that destabilize the MLP training on RX 580. It forces the search to remain in a "trust region" around the current optimum until we have evidence to expand.
5. **Implementation complexity**: 2 (add a threshold comparator in the proposal loop)
6. **Risk**: low (it only rejects proposals, not modifies the search distribution; can be tuned dynamically)

---

### **Hypothesis 2: Deterministic Replication Check (Reproducibility Gate)**
1. **Domain**: Experimental design / reproducibility
2. **Core idea**: Before any new proposal is evaluated, check if exactly the same hyperparameter set has been evaluated before—if so, skip and force a different proposal.
3. **Implementation target**: `runner.py` before launching subprocess; maintain a set of `frozenset` of evaluated hyperparameter dicts. If new proposal is a duplicate, regenerate.
4. **Why it helps**: The trace shows iter 17, 18, 21 are repeats of iter 16's config (`{'LR': 0.003, 'BATCH_SIZE': 128}`)—but they get discarded because the runner doesn't know they are duplicates and wastes resources. Avoiding repeats frees iterations for genuinely new explorations and prevents the runner from getting stuck in a "re-evaluate the best" loop. This directly reduces the discard rate.
5. **Implementation complexity**: 1 (add a hash set)
6. **Risk**: low (no negative impact—just skips known points)

---

### **Hypothesis 3: Early-Stopping Based on Degradation Ratio**
1. **Domain**: Online learning / bandit algorithms
2. **Core idea**: If a new proposal's validation loss after N steps (e.g., 50% of total inner steps) exceeds a threshold ratio (e.g., 3x the current best val_bpb), terminate the evaluation early and discard without completing all steps.
3. **Implementation target**: The inner-loop evaluation function (or the subprocess callback) that computes `val_bpb` over epochs. Add an early-termination check at epoch 5/10 or after 50% of batch updates.
4. **Why it helps**: Many proposals (e.g., iter 7–10, 12–15) eventually yield val_bpb ~7–8, but we waste time completing full training. Early termination reduces iteration wasted on clearly bad configurations, allowing more iterations for promising ones. This amplifies the effective search depth around the best region.
5. **Implementation complexity**: 3 (requires passing current best val_bpb to the evaluation function and modifying the HIP runtime to accept a termination signal)
6. **Risk**: medium (early termination could cut off a configuration that would improve later in training; but on this trace, none of the high-discard configurations ever recovered within the recorded iterations)

---

### **Hypothesis 4: Exponential Weighted Regret Penalty on Proposal Sampler**
1. **Domain**: Reinforcement learning / adversarial search
2. **Core idea**: Proposals that result in a high regret (val_bpb - best_val_bpb) should reduce the probability of sampling similar hyperparameter changes in future iterations using an exponential decay factor.
3. **Implementation target**: `propose_hyperparameters()` method—maintain a weight vector for each hyperparameter dimension (e.g., LR, batch_size). After each discard, multiply the weight of the changed dimension by a penalty factor (e.g., 0.9). After each keep, boost it slightly (e.g., 1.05). Normalize before next proposal.
4. **Why it helps**: Currently, the runner keeps re-proposing `LR: 0.003` (iters 16, 17, 18, 21) even though earlier non-best versions (iter 7, 9) were bad. A regret-weighted approach would learn that changing LR from 0.001 to 0.003 often fails, and thus penalize that change direction. This directly reduces the probability of proposing high-risk, high-regret changes, focusing search on more promising dimensions.
5. **Implementation complexity**: 4 (requires maintaining a stateful weight table across iterations and modifying the proposal distribution)
6. **Risk**: medium (could over-penalize a dimension that just had bad luck; but exponential decay is robust if decay is not too aggressive)