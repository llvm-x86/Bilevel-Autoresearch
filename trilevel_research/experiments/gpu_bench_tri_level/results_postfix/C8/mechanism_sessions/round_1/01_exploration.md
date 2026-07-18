## Proposed Mechanism Improvements for GpuBenchRunner

### Hypothesis 1: Adaptive Mutation Rate Based on Improvement Gradient

1. **Domain**: Evolutionary Optimization / Bayesian Optimization
2. **Core idea**: Dynamically adjust the mutation magnitude (step size) based on the slope of recent improvements, shrinking when close to optimum and expanding when plateauing.
3. **Implementation target**: `ProposalGenerator.generate_proposal()` method in runner.py
4. **Why it helps**: The current trace shows the runner repeatedly tweaking LR by ±0.0001 around 0.003 (iterations 12-16), indicating it's near a local optimum but applying tiny perturbations that rarely improve. By tracking the improvement gradient, the mechanism would detect when small steps stop yielding gains and either:
   - Shrink step size further for finer-grained search around promising regions
   - Or inject a larger "escape" mutation when plateauing, reducing the >50% discard rate
5. **Implementation complexity**: 3 (moderate - requires state tracking across iterations)
6. **Risk**: low/medium (well-understood technique, but gradient estimation adds noise)

### Hypothesis 2: Crossover Between Top-K Configurations

1. **Domain**: Genetic Algorithms / Mix-up Regularization
2. **Core idea**: Instead of mutating a single best config, generate proposals by combining hyperparameters from the top 3-5 historical configurations using weighted interpolation.
3. **Implementation target**: `ProposalGenerator.__init__()` and `generate_proposal()` methods
4. **Why it helps**: The trace shows distinct clusters of good configurations: (LR=0.003, WD=1e-6, HIDDEN=768) at iter 9, and (LR=0.003, WD=1e-5, HIDDEN=512) at iter 4. Crossover could produce hybrids like LR=0.003, WD=5e-6, HIDDEN=640 which neither configuration explored, potentially finding better trade-offs. This directly addresses the "too many discards" problem by exploiting known good hyperparameter combinations rather than random perturbations.
5. **Implementation complexity**: 4 (requires maintaining a hall-of-fame archive, interpolation logic)
6. **Risk**: medium (crossover can produce degenerate configurations if not bounded properly)

### Hypothesis 3: Automatic Learning Rate Schedule Injection

1. **Domain**: Meta-Learning / Learning Rate Scheduling
2. **Core idea**: Instead of keeping LR fixed during training, inject a cosine annealing schedule where the searched LR becomes the maximum learning rate, automatically decaying to 0 over the training run.
3. **Implementation target**: `TrainingLoop.create_optimizer()` or the launch script that passes config to gpu_bench
4. **Why it helps**: The trace shows the runner getting stuck in a narrow LR range (0.0025-0.003) with tiny improvements. A cosine schedule starting at the searched LR would:
   - Automatically explore lower LRs (0.001-0.002) later in training without requiring new proposals
   - Potentially find better minima through simulated annealing-like effects
   - Reduce the need for the outer loop to exhaustively search LR, since the schedule does temporal exploration
5. **Implementation complexity**: 2 (simple - wrap LR in a schedule object)
6. **Risk**: low (cosine annealing is well-studied and rarely harmful)

### Hypothesis 4: Validation-Based Early Termination with Warm Restarts

1. **Domain**: Hyperband / Successive Halving / Learning Rate Warm Restarts
2. **Core idea**: Run each configuration for multiple short "heat-up" cycles with warm restarts (reset optimizer, restore best weights), terminating the run early if validation loss doesn't improve within the first 20% of training.
3. **Implementation target**: `GpuBenchRunner.run_trial()` method and the subprocess timeout mechanism
4. **Why it helps**: Many discarded proposals (iterations 3, 8, 13, 18) converge to val_bpb ~7-8 within a few iterations of training. Early termination would:
   - Free up GPU compute for more promising proposals
   - Allow the runner to test 2-3x more configurations in the same time budget
   - The warm restart component helps escape the current narrow search basin (0.0438 bpb) by periodically resetting and trying different initializations
5. **Implementation complexity**: 5 (requires modifying the external gpu_bench binary or adding validation monitoring in the subprocess)
6. **Risk**: medium/high (risk of prematurely discarding configurations that would improve later, and requires binary modification)

---

**Recommended priority**: Implement Hypothesis 1 (adaptive mutation) and Hypothesis 3 (LR schedule) first as they are complementary and low-risk. Then add Hypothesis 2 (crossover) if improvements plateau. Hypothesis 4 should be last due to implementation complexity.