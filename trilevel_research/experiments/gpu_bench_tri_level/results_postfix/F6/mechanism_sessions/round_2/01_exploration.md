Based on the trace analysis, the core issue is that the search quickly finds a good point (val_bpb=0.1314) but then fails to explore effectively around it or escape to better regions. All subsequent proposals are discarded or lead to much worse performance. The current mechanism lacks **exploitation refinement** and **robust change magnitude control**.

Here are 4 concrete mechanism improvements:

---

### Hypothesis 1: Adaptive Perturbation Magnitude Scaling

1. **Domain**: Bayesian Optimization / Trust Region Methods  
2. **Core idea**: Scale the magnitude of hyperparameter changes based on the relative improvement of the last kept configuration, shrinking perturbations near good minima and expanding them when stuck.  
3. **Implementation target**: `_propose_changes()` method in `GpuBenchRunner` — add a `perturbation_scale` attribute that is updated each iteration.  
4. **Why it helps**: Currently, the same fixed-size changes (e.g., LR from 0.003→0.001 or 0.01) are too large when near the 0.1314 optimum, causing catastrophic jumps to 7.9+ val_bpb. Smaller, refined steps (e.g., LR: 0.003→0.0025) would allow fine-tuning. Conversely, when no improvement is found for 5+ iterations, scaling up perturbations helps escape plateaus.  
5. **Implementation complexity**: 2  
6. **Risk**: low

---

### Hypothesis 2: Elite Buffer Replay with Semi-Random Regeneration

1. **Domain**: Evolutionary Strategies / Population-Based Training  
2. **Core idea**: Maintain a small buffer (size 3) of the best-performing configurations and occasionally reintroduce mutated variants of elite configurations rather than always starting from the single best.  
3. **Implementation target**: `GpuBenchRunner.__init__()` — add `self.elite_buffer = []`. Modify `_propose_changes()` to sample from the buffer (with decay) 40% of the time.  
4. **Why it helps**: The current mechanism keeps returning to `{'LR': 0.003}` (iter 14, 17, 18, 19) which repeatedly fails to improve. Mixing in perturbations of iter 4's config (`{'LR': 0.003, 'BATCH_SIZE': 32}`) or crossing it with other near-optimal points (e.g., iter 16's batch 64) could discover better local neighborhoods without catastrophic drift.  
5. **Implementation complexity**: 3  
6. **Risk**: medium

---

### Hypothesis 3: Gradient-Aware Exploration via Validatioño Trace Momentum

1. **Domain**: Meta-Learning / Gradient-Based Hyperparameter Optimization  
2. **Core idea**: Track the direction of val_bpb changes over the last 3 kept iterations and extend/contract the search in the direction of improvement using a momentum-like update rule.  
3. **Implementation target**: `GpuBenchRunner._update_and_decide()` — after each `keep`, compute `delta_val = previous_val_bpb - current_val_bpb` and update a momentum vector that biases next proposals toward parameters that previously caused improvement.  
4. **Why it helps**: Iter 2 (LR 3e-3) improved val_bpb from 6.4116 to 3.0210, and iter 4 (batch 32) improved to 0.1314 — these were directional improvements in the `(LR, batch)` plane. A momentum mechanism would detect that both `LR=0.003` and `batch=32` contributed positively and preferentially explore nearby values (LR=0.004 with batch=40) rather than reverting to baseline or making random changes.  
5. **Implementation complexity**: 4  
6. **Risk**: medium

---

### Hypothesis 4: Robustification Through Conditional Warmup Reset

1. **Domain**: Learning Rate Scheduling / Curriculum Learning  
2. **Core idea**: When a proposed change results in val_bpb > 2× the best observed value and all metrics indicate severe divergence, perform a soft-reset of the optimizer state (warmup restart on the subprocess) before continuing the search.  
3. **Implementation target**: `GpuBenchRunner.run_iteration()` — add a divergence-detection hook that passes a `--warmup_steps=50` flag to the `gpu_bench` binary when the val_bpb spike is > 7.0 (clearly chaotic).  
4. **Why it helps**: Most discards (val_bpb 7.9–8.0) are catastrophically worse than the baseline (6.4116), indicating the model enters a bad loss landscape or NaN territory. A 50-step linear LR warmup would allow the optimizer to stabilize gradually, potentially preventing the 7-fold degradation and enabling discovery of new stable configurations. This is especially important for the RX 580's HIP backend which may have numerical sensitivity.  
5. **Implementation complexity**: 3  
6. **Risk**: low