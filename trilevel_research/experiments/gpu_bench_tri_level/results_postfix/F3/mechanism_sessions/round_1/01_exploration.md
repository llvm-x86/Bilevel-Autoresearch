Looking at the trace, the core problem is clear: you found a strong configuration (BATCH_SIZE=128, LR=0.003) at iteration 4 (val_bpb=0.0296), but subsequent proposals keep regressing to ~7-8 bpb. This suggests the search is failing to do local refinement around promising configurations and is wasting iterations on large, disruptive moves. Here are four targeted mechanism improvements:

## 1. **Local Perturbation Budget After Finding a Leader**

- **Domain**: Bayesian Optimization (trust region methods)
- **Core idea**: After discovering a configuration with val_bpb < 0.1, allocate the next N iterations exclusively to small (≤10%) single-parameter perturbations of that leader before allowing exploration of new hyperparameter subspaces.
- **Implementation target**: `GpuBenchRunner._search_update()` — add a `leader_lock` counter that blocks large parameter changes when a strong leader exists
- **Why it helps**: The trace shows you found a great configuration at iter 4, then immediately jumped to HIDDEN_DIM=320 (30% change), LR=0.001 (67% change), and weight_decay variations that destroyed performance. A forced local search phase would try BATCH_SIZE=112, 144, 160; LR=0.002, 0.004, 0.005 — small steps that maintain good bpb while searching for improvements.
- **Implementation complexity**: 2
- **Risk**: low — can be deactivated if local search plateaus

## 2. **Retry Validation on Outlier Configurations**

- **Domain**: Experimental reproducibility (ML benchmarking)
- **Core idea**: When a parameter change produces a val_bpb regression >5x from the current best, automatically re-run that configuration with 3 different random seeds and take the median, to distinguish true configuration weakness from bad initialization luck.
- **Implementation target**: `GpuBenchRunner._run_single()` — add an `is_retry` mode triggered by large performance drops
- **Why it helps**: In a 0.03 to 7.5 bpb drop, there's a non-trivial chance of pathological initialization. The trace never retries — it accepts single-run results as definitive. On AMD RX 580 with small MLPs, initialization variance is high. Retrying could reveal that some configurations are actually competitive but got unlucky.
- **Implementation complexity**: 3
- **Risk**: low — extra compute per candidate, but only triggered for severe drops

## 3. **Multi-Modal Proposal Generation (3 candidates per iteration)**

- **Domain**: Parallel tempering / genetic algorithms
- **Core idea**: Instead of proposing one change per iteration, generate and test 3 candidates simultaneously: one conservative (5% change from best), one medium (10-20% change), one exploratory (new parameter subspace), then keep the best result.
- **Implementation target**: `GpuBenchRunner._propose_config()` — return a list of 3 configs; `_run_single()` — accept multiple configs
- **Why it helps**: Your current search alternates between wild guesses (HIDDEN_DIM 288→384) and repeating the same config (BATCH_SIZE=128 four times). Three-track generation ensures you always have a "safe" refinement candidate, a moderate exploration, and a creative option. This parallel efficient experimentation is standard in auto-ML systems.
- **Implementation complexity**: 4
- **Risk**: medium — 3x compute per iteration, but dramatically better coverage

## 4. **Adaptive Parameter Sensitivity Scoring**

- **Domain**: Gradient-free optimization (CMA-ES)
- **Core idea**: Maintain a per-parameter "sensitivity score" based on how much each hyperparameter affected val_bpb in previous iterations, then use inverse sensitivity to set proposal step sizes (high sensitivity → smaller steps).
- **Implementation target**: `GpuBenchRunner` — new `ParamSensitivityTracker` class; integrate into `_propose_config()`
- **Why it helps**: The trace shows you tried WEIGHT_DECAY at 0.001, 0.0005, 0.0001 — all terrible. This parameter is apparently extremely sensitive on your problem. A sensitivity tracker would learn that WEIGHT_DECAY needs 0.00001-scale steps or should be frozen once its sensitivity exceeds a threshold. Conversely, BATCH_SIZE changes from 64-256 all gave good results (0.03-0.13 bpb), suggesting it has low sensitivity and can be explored more aggressively.
- **Implementation complexity**: 3
- **Risk**: low — purely additive observation layer that doesn't change search logic