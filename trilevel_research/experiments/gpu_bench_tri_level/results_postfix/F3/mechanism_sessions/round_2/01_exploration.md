Based on the trace analysis, the main bottleneck is that the runner is generating too many low-quality proposals that don't improve val_bpb. Here are 4 mechanism improvements:

---

## 1. **Learning Rate Warmup Reintroduction**

**Domain**: Training dynamics / optimization

**Core idea**: Prepend a short linear warmup phase (~50 steps) to avoid large gradient updates during early training that destabilize convergence when hidden dim is increased.

**Implementation target**: `GpuBenchRunner._run_config` or `construct_command` method — add `--lr-warmup-steps 50` flag to the command line when launch config contains `LR` or `HIDDEN_DIM` changes from baseline.

**Why it helps**: The trace shows iter 2 (LR: 0.003) got 3.0 bpb, but iter 3 (LR: 0.001) got 7.83 bpb — this suggests sensitivity to initial LR. Warmup smooths the transition, especially when hidden dim increases from 256 to 512+, preventing sudden gradient spikes that cause convergence failures (iter 8, iter 13 all failed with larger hidden dims). This directly addresses the "too many discards" problem.

**Implementation complexity**: 2 (simple flag addition, no new dependencies)

**Risk**: low (warmup is standard practice; at worst adds a few seconds to training)

---

## 2. **Configuration Space Upper Bound Enforcement**

**Domain**: Search space design / safety constraints

**Core idea**: Impose a runtime upper bound on `HIDDEN_DIM` relative to input/output dimensions (e.g., HIDDEN_DIM ≤ 4 × input_dim) to prevent hidden layer sizes that cause memory thrashing or gradient explosion on AMD RX 580.

**Implementation target**: `GpuBenchRunner._generate_config` or `_validate_config` — add check `if config['HIDDEN_DIM'] > 4 * input_dim: config['HIDDEN_DIM'] = 4 * input_dim` and log warning.

**Why it helps**: The trace shows HIDDEN_DIM=512 was the only working value, but 768 (iter 5) and 448 (iter 21) failed. The AMD RX 580 has limited VRAM and compute; oversizing hidden dim causes memory pressure that degrades training quality. By capping hidden dim, proposals stay in the feasible region, reducing wasted iterations.

**Implementation complexity**: 1 (simple condition + clamping)

**Risk**: low (reduces proposal diversity but eliminates many discard-worthy configs)

---

## 3. **Gradient Scaling for Weight Decay Sensitivity**

**Domain**: Numerical stability / regularization

**Core idea**: Automatically scale weight decay by `1 / batch_size` when batch size changes, to prevent regularization from dominating or being negligible as batch size varies.

**Implementation target**: `GpuBenchRunner._generate_config` — after computing LR and BATCH_SIZE changes, set `WEIGHT_DECAY = 0.001 * (64 / BATCH_SIZE)` to maintain constant effective regularization.

**Why it helps**: The trace shows BATCH_SIZE=128 (iter 6) failed despite same LR as winner, and BATCH_SIZE=64 (iter 9) also failed. Weight decay interacts multiplicatively with batch size; without scaling, larger batches see proportionally less regularization, causing overfitting. This linkage reduces the effective search space dimension, making proposals more likely to succeed.

**Implementation complexity**: 2 (simple scaling formula, no new hyperparameters)

**Risk**: medium (changes trade-off surface; may need re-tuning but likely positive)

---

## 4. **Relative Validation BPD Offset Reporting**

**Domain**: Evaluation metrics / search guidance

**Core idea**: Report relative improvement (`val_bpd - best_seen_val_bpd`) instead of absolute val_bpd to the proposer, enabling the proposer to focus on configuration improvements rather than absolute values.

**Implementation target**: `GpuBenchRunner._run_config` — after reading val_bpd from binary stdout, subtract `self.best_val_bpd_so_far` before passing to proposer's `record(...)` call.

**Why it helps**: The trace shows proposer keeps trying to validate the same config (iter 7, 14, 18, 20) because it sees absolute values ~7.9 and cannot distinguish "slightly worse" from "better." Offsetting to relative values allows the proposer to see even tiny improvements or degradations, enabling finer-grained search and avoiding exploration of dead zones.

**Implementation complexity**: 3 (requires modifying proposer interface or adding a normalization method)

**Risk**: medium (changes semantics; proposer might misinterpret negative values, but trace shows it already handles 0.09 → 7.9 jumps)