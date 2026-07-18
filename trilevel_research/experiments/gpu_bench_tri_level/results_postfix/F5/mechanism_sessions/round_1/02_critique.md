## Critique of Hypotheses

### Mechanism 1: Adaptive LR Search Radius Shrink with Stagnation Counters

**Most likely failure mode**: Premature search radius collapse. If the stagnation counter triggers after 3-5 discards (common default), the LR search radius shrinks too quickly, trapping the runner in an even narrower region. The runner then cannot escape because the jitter amplitude is now too small relative to the local well's width. This creates a **double trap**: stuck first in LR space, then stuck in an even smaller LR space.

**Implementation trap**: The stagnation counter must persist across *different proposal types*. The trace shows LR proposals interleaved with occasional WEIGHT_DECAY or BATCH_SIZE changes. If the counter resets on non-LR proposals, the radius never shrinks. If it doesn't reset, a single good non-LR proposal followed by 5 LR failures triggers the shrink—but the good proposal may have changed the landscape, making the shrink premature. **Tracking what "stagnation" means across mixed proposal types is fragile**.

**Evidence from trace**: Partially supported. The trace does show repeated LR sampling near 0.003-0.005. However, the best iteration 4 (LR=0.005) is *not* revisited after iter 10. The runner drifts toward 0.003, not remaining near 0.005. This suggests the radius is already adapting—or the runner is exploring downhill, not stagnating near the best. The "local optimum at LR=0.005" claim is unsupported; the trace shows the runner *left* 0.005 and is now stuck near 0.003. A radius shrink near 0.003 would *confirm* the failure, not escape it.

**Score**: impact 3 × feasibility 4 ÷ complexity 2 = **6.0**

---

### Mechanism 2: Co-adaptation Leash — Fractional-Mutation of Paired Parameters

**Most likely failure mode**: Indiscriminate co-mutation destroys remembered good configurations. If the runner has found a working BATCH_SIZE (e.g., 64) after 20 iterations, forcing a random mutation to it each time LR fails will corrupt that memory. The runner will oscillate between two bad parameter sets: (good batch, bad LR) and (mutated batch, mutated LR). The discard rate increases further because *both* are now bad.

**Implementation trap**: Defining "unchanged parameter" is ambiguous. The runner's proposal log shows BATCH_SIZE=48 or 64, HIDDEN_DIM=64, WEIGHT_DECAY=1e-5 or 1e-4. But some of these are *already* being changed occasionally. The mechanism must track which parameters were *intentionally left equal to the current best* in the rejected trial—not merely which have low variance. This requires diffing the proposed candidate against the current best, but the runner's internal state may not preserve the exact best record at proposal time. **Knowning which parameter was "unchanged by choice" vs. "unchanged because it's already optimal" requires deep runner instrumentation.**

**Evidence from trace**: Weak. The trace shows 8 proposals with only LR changed (iters 5,7,8,9,10,12,14,15). But we also see BATCH_SIZE changes (iter 6: 64→48, iter 13: 64→48). This suggests the runner *can* change BATCH_SIZE but chooses not to—possibly because BATCH_SIZE=64 is known good. Forcing co-mutation would break this known good setting. The trace does *not* show that co-adaptation would help; it shows the runner is exploring LR *while keeping BATCH_SIZE=64*, which is a rational choice if BATCH_SIZE=64 is near-optimal.

**Score**: impact 2 × feasibility 3 ÷ complexity 3 = **2.0**

---

### Mechanism 3: Memory-Aware BATCH_SIZE Grid Anchoring

**Most likely failure mode**: Eliminating BATCH_SIZE=48 from the search space when 48 might be optimal. The trace shows BATCH_SIZE=48 used in iters 6 and 13. If 48 is actually better on this GPU (e.g., due to memory constraints where 64 causes OOM or swapping), forcing power-of-two batch sizes will *increase* val_bpb or cause memory errors. The trace does not show BATCH_SIZE=48's performance; it could be the silent destroyer or the hidden savior.

**Implementation trap**: Mapping "GPU effective memory footprint" to valid batch sizes is hardware-dependent and requires runtime memory queries. Different GPUs (RDNA2 vs RDNA3, Navi21 vs Navi31) have different warp sizes, L1/L2 geometries, and register file sizes. A hardcoded `valid_batch_sizes = [64, 128, 256]` list will fail on GPUs where 48 is more efficient (e.g., due to 24-CU groupings in some architectures). **Hardware-specific anchor lists that are not queried dynamically will break on future GPUs.**

**Evidence from trace**: Weak. BATCH_SIZE=48 appears twice, BATCH_SIZE=64 appears 6+ times. No evidence that 48 is worse or better—the trace doesn't report val_bpb per iteration. We don't know if using 48 was a failed experiment or a successful adaptation. The hypothesis that 48 causes "warp occupancy penalties" is plausible but unverified for the specific GPU (AMD RDNA3 likely has 32-wide warps, so 48=1.5 warps is indeed inefficient, but some architectures handle odd sizes via remainder warps fine).

**Score**: impact 2 × feasibility 4 ÷ complexity 1 = **8.0**

---

### Mechanism 4: Discard-Aware Warm Restart (DAWR) Trigger

**Most likely failure mode**: Restart destroys genuine progress toward a distant optimum. If the runner is correctly exploring LR space via a random walk and will reach 0.001 in 3 more iterations, a restart at iter 15 resets to LR=0.01 and wastes 15 iterations of Bayesian posterior. The runner enters a cycle: explore → restart → explore → restart, never converging. This is catastrophic if the restart threshold (5 discards) is too low relative to the search space dimension.

**Implementation trap**: The "broader prior" must be *strictly broader* than the current search distribution, or the restart samples the same region. If the runner's prior is already log-uniform over [0.001, 0.1] (as is typical), a restart resamples from the same distribution—no benefit. If the prior is widened (e.g., [0.0001, 1.0]), most samples will produce OOM or NaN gradients, creating instant discards and immediate retrigger of restart. **Designing a "broader prior" that is both exploratory and stable requires meta-knowledge of the problem's Lipschitz constant.**

**Evidence from trace**: Strong. The trace shows a clear pattern: after iter 10, the runner cycles through LR values near 0.003 with no improvement. A restart would break this cycle. However, note that iteration 13 changes BATCH_SIZE—the runner *does* escape occasionally. The real issue is that the escape doesn't persist. A restart would force all parameters to change simultaneously, which is more aggressive than what the trace suggests is needed. The trace shows LR is the stuck variable; other parameters change naturally.

**Score**: impact 4 × feasibility 3 ÷ complexity 2 = **6.0**

---

## Final Scores

| Mechanism | Impact | Feasibility | Complexity | Score |
|-----------|--------|-------------|------------|-------|
| 1 (LR radius shrink) | 3 | 4 | 2 | 6.0 |
| 2 (Co-adaptation leash) | 2 | 3 | 3 | 2.0 |
| 3 (Batch size anchor) | 2 | 4 | 1 | 8.0 |
| 4 (Warm restart) | 4 | 3 | 2 | 6.0 |

**Selected**: **Mechanism 3** — highest score, lowest risk, and directly addresses a known hardware inefficiency that could mask LR effects; combine with Mechanism 1 as a secondary safety net if batch anchoring alone doesn't break the stall.