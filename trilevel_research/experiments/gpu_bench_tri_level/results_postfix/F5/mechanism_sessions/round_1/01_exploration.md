Based on the trace, the runner is stuck in a local optimum at LR=0.005 (the best iter 4) and repeatedly testing LR values around 0.003 with no success. The core failure is an inability to **escape the local well** or **adjust other knobs synergistically** when LR fine-tuning fails repeatedly after 10+ attempts. Below are four targeted mechanism changes:

---

### Mechanism 1: Adaptive LR Search Radius Shrink with Stagnation Counters

1. **Domain**: Bayesian optimization / multi-armed bandit  
2. **Core idea**: After `N` consecutive discards for LR-only proposals, reduce the LR search radius (log step size) by half and inject a random jitter to escape local plateaus.  
3. **Implementation target**: `GpuBenchRunner._propose_hyperparams()` — specifically the section that samples `LR` from a log-uniform distribution around `best_params['LR']`.  
4. **Why it helps**: The trace shows 10+ discards testing LR in [0.002, 0.008] without improvement. A fixed search radius keeps sampling near the known failure zone. Shrinking the radius prevents wasteful re-sampling of LR values that demonstrably fail, while jitter provides a controlled exploration–exploitation transition.  
5. **Implementation complexity**: 2 (add a stagnation counter, one `if` block, and a scaling factor)  
6. **Risk**: low (only changes the proposal distribution, not the evaluation; stagnation tracking is monotonic and self-limiting)

---

### Mechanism 2: Co-adaptation Leash — Fractional-Mutation of Paired Parameters

1. **Domain**: Evolutionary strategies / co-variance matrix adaptation  
2. **Core idea**: When discarding an LR-only proposal, force the next proposal to also mutate a parameter that was *unchanged* in the discarded trial (e.g., BATCH_SIZE or HIDDEN_DIM) by a small fraction, breaking the fixation on single-parameter sweeps.  
3. **Implementation target**: `GpuBenchRunner._propose_hyperparams()` — after generating the base candidate, apply a probabilistic co-mutation to at least one non-LR parameter sampled from the current best set.  
4. **Why it helps**: The trace reveals that almost all unsuccessful proposals only change LR (with occasional minor WEIGHT_DECAY shifts). This suggests that LR alone is insufficient — other parameters like batch size or hidden dim may need slight adjustments to work with the new LR. Forcing fractional co-adaptation prevents myopic single-dimension search.  
5. **Implementation complexity**: 3 (track unchanging params in the rejected proposal, add mutation logic with small step sizes for discrete params)  
6. **Risk**: medium (might initially increase discard rate if the co-mutation magnitude is poorly chosen, but reverse is also possible)

---

### Mechanism 3: Memory-Aware BATCH_SIZE Grid Anchoring

1. **Domain**: Hardware-aware ML / GPU memory constraints  
2. **Core idea**: Anchor BATCH_SIZE proposals only to divisor values of the GPU’s effective memory footprint (e.g., 64, 128, 256) instead of arbitrary small values (48, 64, 128), and reject proposals that use non-power-of-two batch sizes.  
3. **Implementation target**: `GpuBenchRunner._propose_hyperparams()` — add a `valid_batch_sizes` list and a rejection/remapping step when a candidate batch size is not in the anchor set.  
4. **Why it helps**: The trace shows batch sizes of 48 and 64 used interchangeably. On GCN/RDNA architectures, non-power-of-two batch sizes can cause warp occupancy penalties and memory coalescing inefficiencies, artificially inflating val_bpb. Anchoring to GPU-friendly sizes ensures hardware efficiency and reduces variance, making LR sweeps more interpretable.  
5. **Implementation complexity**: 1 (add a mapping list and a simple validator)  
6. **Risk**: low (reduces search space, may exclude some optimal values but power-of-two batch sizes are well-justified for these GPUs)

---

### Mechanism 4: Discard-Aware Warm Restart (DAWR) Trigger

1. **Domain**: Optimizer restart schedules (e.g., SGDR / cosine annealing)  
2. **Core idea**: After `K` discards (e.g., 5) without improvement, reset the search by sampling *all* hyperparameters from scratch (including LR) using a broader prior, and push the new baseline as a temporary candidate, forcing the runner to explore a new region.  
3. **Implementation target**: `GpuBenchRunner._propose_hyperparams()` — add a global `stall_counter`, and when it exceeds a threshold, call a `_warm_restart_sample()` that ignores the current best and samples from the original prior distribution.  
4. **Why it helps**: The trace shows that after ~iter 10, the runner is repeatedly revisiting LR=0.003 even though it has been tried 5+ times with consistent failure. A full restart breaks the inertia and re-initializes the search state, mimicking how cosine annealing restarts escape local minima in loss landscapes.  
5. **Implementation complexity**: 2 (add a global counter, a restart method, and a reset flag)  
6. **Risk**: medium (may temporarily discard a promising region but is self-correcting — once the restart candidate is kept, it becomes the new best and the runner continues from there)

---

**Summary Recommendation**: Implement Mechanisms 1 and 4 immediately (low complexity, high impact on search stagnation). Add Mechanism 3 as a safety net. Defer Mechanism 2 until after establishing that LR-only focus is indeed the bottleneck — the trace strongly suggests it is, so implement 2 in parallel.