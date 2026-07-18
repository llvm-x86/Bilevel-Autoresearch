Based on the trace analysis, the core problem is that 21 out of 22 proposals are discarded because they fail to improve val_bpb. The optimizer is stuck in a narrow region near hidden_dim=384, lr=0.005 with high variance proposals.

Here are 4 concrete mechanism changes:

## 1. Multi-fidelity Early Rejection with Warmup Checkpoint

**Domain**: Bayesian optimization with learning curves

**Core idea**: Validate proposals at iteration 50% instead of 100% to discard hopeless configurations early, saving budget for more dense exploration in the 384-dim region.

**Implementation target**: `GpuBenchRunner._run_single_config()` → add early-stopping callback

**Why it helps**: Currently each discarded run wastes 100% of the compute budget. By assessing val_bpb at halfway and rejecting configurations with val_bpb > 4.0 (the observed discard floor), we increase the effective number of trials by ~2x, allowing tighter search around the best configurations.

**Implementation complexity**: 2 (insert early-termination logic with threshold)

**Risk**: Low — early rejection is standard practice; threshold 4.0 is safe given no discarded run was below 0.7653 in the trace

## 2. Local Perturbation Surrogate (Adaptive Search Radius)

**Domain**: Evolutionary strategies with mutation scaling

**Core idea**: Dynamically shrink the perturbation delta for each hyperparameter when recent proposals are all discarded, to sample more densely near the best point.

**Implementation target**: `GpuBenchRunner._generate_proposal()` — add exponential moving average of discard rate

**Why it helps**: The trace shows extreme convergence — best val_bpb=0.0416 with HIDDEN_DIM=384, but all surrounding proposals (256, 512, 640) fail. Current mechanism likely perturbs too aggressively. By scaling search radius by `exp(-0.5 * discard_rate_ma)` each iteration, we force the optimizer to sample at 10-20% the original step size, revealing hidden local structure.

**Implementation complexity**: 3 (add tracking state, modify proposal step sizes)

**Risk**: Medium — too aggressive shrinking could miss global optima, but the trace suggests we are already in a narrow basin, so local focus is justified

## 3. Variance-Aware Acceptance Threshold (Annealing)

**Domain**: Simulated annealing with adaptive temperature

**Core idea**: Adjust the "keep" threshold based on recent proposal variance — accept more configurations when variance is low, reject more aggressively when variance is high.

**Implementation target**: `GpuBenchRunner.run()` — modify acceptance logic in the main loop

**Why it helps**: The trace shows a pattern: iteration 12 (HIDDEN_DIM=384) achieved a breakthrough (0.0416), but subsequent iterations 13–21 all failed despite tiny perturbations. This implies the acceptance threshold is too rigid — we keep only improvements, but in a flat region, minor perturbations drop performance slightly. By accepting proposals within 2 standard deviations of recent keeps when variance < 0.1, we capture near-optimal configurations that current logic discards (e.g., iter 17 at 0.0901 with HIDDEN_DIM=640 — only 2x worse than best but discarded).

**Implementation complexity**: 4 (requires maintaining running variance of keep values, dual threshold logic)

**Risk**: Medium — could accept worse configurations if variance estimation is noisy, but annealing schedule (reduce acceptance width over time) mitigates this

## 4. Warm-Start Restart with Best-Seed Ensemble

**Domain**: Population-based training with checkpoint reuse

**Core idea**: When 3 consecutive proposals are discarded, restart the next 3 proposals from the best configuration but with different random seeds to combat initialization noise.

**Implementation target**: `GpuBenchRunner.run()` — add restart counter and seed perturbation

**Why it helps**: The trace shows deterministic exploration of hyperparameter space. Given the RX 580's non-deterministic GPU kernels, val_bpb has inherent variance from seed effects. The current mechanism assumes configurations determine outcomes, but seed-differences at the same config could yield ±5-10% variance. By restarting with seeds 42, 73, 101 after discards, we test whether iter 12's value (0.0416) was a fluke or repeatable. If repeatable, we trust local exploration more.

**Implementation complexity**: 2 (add seed override buffer, reset proposals)

**Risk**: Low — consumes 2-3 extra trials but no worse than random discards we currently pay