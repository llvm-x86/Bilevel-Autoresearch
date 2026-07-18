Based on the iteration trace analysis, here are 4 mechanism improvements to help GpuBenchRunner find lower val_bpb:

## Mechanism 1: Adaptive Hidden Dimension Backtracking

**Domain**: Bayesian optimization with exploration-exploitation balance

**Core idea**: When increasing hidden dimension from a peak performer (512 at iter 4) fails repeatedly, expand the search space to *lower* dimensions with higher learning rates instead of continuing to increase dimensions.

**Implementation target**: `_propose_experiment()` method in GpuBenchRunner (currently creates changes dict)

**Why it helps**: The trace shows hidden_dim increased from 512 to 544, 576, 640, 480, and back to 544 repeatedly—all discards. This suggests the model is overfitting at higher capacities. By forcing exploration of hidden_dim=384 or 256 *combined with lower* learning rates (0.001-0.002), you test whether regularization + lower capacity = better generalization. The fixed LR=0.003 is likely too high for larger models, causing divergence.

**Implementation complexity**: 3 (adds logic to detect repeated dimension increases from a known-best point and flip search direction)

**Risk**: medium (may overcorrect if the true optimum is actually at higher dimensions, but current approach is failing)

## Mechanism 2: Dual-Learning Rate Decay Scheduling

**Domain**: Learning rate schedulers in NN training

**Core idea**: Instead of a fixed learning rate per experiment, implement a two-phase schedule that starts with the best LR (0.003) and decays to a secondary candidate (0.001) midway through training, measured by validation bpb improvements.

**Implementation target**: Modify the `run_single_experiment()` method (or equivalent) to pass an LR schedule flag instead of a fixed LR to the gpu_bench binary

**Why it helps**: The trace shows LR=0.003 gave the best result (iter 4), but all subsequent attempts with LR=0.003 + other changes failed (exceeding 2 bpb vs 0.09). This suggests the model converges very quickly then overfits. A decay to 0.001 after, say, 50% of epochs could prevent validation loss from bouncing away from the 0.09 optimum. Subtle weight decay (0.0001) might help but needs *diminishing impact* as training progresses—decay achieves this.

**Implementation complexity**: 4 (requires passing schedule parameters to gpu_bench or adding a wrapper that splits training into phases)

**Risk**: low (decaying LR is standard practice, and you already have two LR values proven viable)

## Mechanism 3: Perturbative Hidden Dimension Around Known Best

**Domain**: Local derivative-free optimization (Nelder-Mead style)

**Core idea**: When hidden_dim=512 gave val_bpb=0.0916 (iter 4), search within ±10% of this dimension (461-563) before exploring far ranges like 640 or 480, to build a local response surface.

**Implementation target**: `_generate_proposal()` or equivalent method that selects which hyperparameter changes to test. Currently appears to pick arbitrary deltas (512→544, 576, 640, 480) without systematic scanning.

**Why it helps**: The best performance was at 512; scanning 544, 576, 640 in sequence is coarse. You missed testing 520, 496, 504, 528—any of which could yield 0.08 bpb with minimal change. The current jumps (e.g., 512→480, 512→576) are 6-12% changes; test 3% increments (±16) around 512 first. This reduces the risk of overshooting a narrow optimum that exists at slightly different dimensions.

**Implementation complexity**: 2 (replace arbitrary increment with bounded random perturbation ±10% of current best's value, with multiples of 16/32)

**Risk**: low (simply adds granularity; worst case you discard a few more but find the real peak)

## Mechanism 4: Batch Size-Adaptive Validation Interval

**Domain**: Memory-constrained GPU training

**Core idea**: Dynamically increase validation bpb computation frequency when batch size exceeds GPU memory limits or when training loss diverges, preventing wasted iterations where validation measurements are inaccurate due to gradient accumulation issues on AMD RX 580.

**Implementation target**: `_run_experiment()` method (or equivalent subprocess call). Add a pre-flight check that monitors GPU memory usage via `rocm-smi` or traps CUDA out-of-memory errors, then adjusts the validation interval (e.g., every 50 steps instead of 10).

**Why it helps**: The trace shows batch_size=128 (iter 5) and 256 (iter 10) both produced very high val_bpbs (5.98 and 9.51). On an RX 580 (4-8GB), larger batches likely cause memory thrashing, gradient accumulation errors, or system-level swaps that corrupt the validation process. By detecting memory pressure and either reducing batch size automatically or spacing out expensive validation calls, you avoid incorrectly assuming the hyperparameters are bad when the GPU is simply overloaded.

**Implementation complexity**: 5 (requires GPU memory monitoring, possibly AMD-specific ROCm calls, and conditional validation schedule)

**Risk**: high (AMD GPU monitoring tools may be inconsistent; could introduce race conditions)

**Recommendation**: Start with Mechanism 1 and 3 (low risk, directly address the trace pattern), then add Mechanism 2 if improvements stagnate. Avoid Mechanism 4 unless you can confirm the RX 580 has memory issues with batch sizes >64—the trace suggests high bpb rather than crashes, so the problem may be algorithmic, not hardware.