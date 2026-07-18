## Rigorous Critique of Proposed Mechanisms

### Mechanism 1: Non-Determinism Detection & Compensation

**Most likely failure mode**: The 10x threshold is arbitrary and will fire constantly in early exploration (comparing random configs to each other), creating false positives that desensitize the team to real warnings. Worse, it could trigger a "slow path" that makes everything slower.

**Implementation trap**: The `history.get(config)` lookup requires exact config equality. Floating-point LR values like `0.003` vs `0.0030000001` would miss the match entirely, causing the detector to silently fail. You need a config hashing scheme with tolerance.

**Evidence from trace**: Partially supports. Iterations 12,14,15,16,20,21 do revisit iter 4's config and get vastly different results. But the trace doesn't show *other* configs being non-reproducible — this could be a single pathological config.

**Score**: impact=3 × feasibility=4 ÷ complexity=2 = **6.0**

---

### Mechanism 2: Warm-Start Replay with Multiple Seeds

**Most likely failure mode**: Running 3 seeds per revisit inflates iteration count by 3x for those cases, causing the search to spend 80% of its budget re-running the same config instead of exploring. The trace already shows 6 revisits wasted; this would make it 18 wasted runs.

**Implementation trap**: The environment variable `RANDOM_SEED` must actually control the GPU operations. PyTorch/HIP require `torch.manual_seed()`, `torch.cuda.manual_seed_all()`, plus CUBLAS deterministic flags, plus DataLoader shuffle seeds. If even one seed hook is missing, all 3 seeds produce identical results, wasting compute.

**Evidence from trace**: The non-reproducibility is 0.1286 vs 7.9 — a 60x gap. If this were seed variance, you'd expect a distribution like 0.12, 0.13, 7.9 — not two good and one bad. This looks like a corrupted checkpoint or hardware error, not seed sensitivity.

**Score**: impact=2 × feasibility=3 ÷ complexity=4 = **1.5**

---

### Mechanism 3: Local Refinement Around Best Config (Trust Region)

**Most likely failure mode**: The trust region assumes the best config is a *stable* optimum. If the 0.1286 result was a fluke (due to non-determinism), the trust region will tightly sample around a phantom optimum, never discovering the true good region. The search becomes a local optima trap.

**Implementation trap**: The lognormal proposal is mathematically dangerous — lognormal mean is `exp(μ + σ²/2)`, not `exp(μ)`. So `np.random.lognormal(mean=np.log(0.003), sigma=0.2)` has *expected value* of `0.003 * exp(0.2²/2) = 0.003 * 1.02 = 0.00306`, biasing proposals upward. Over 50 iterations, this drift compounds.

**Evidence from trace**: The current search *already* correctly identifies iter 4 as best and revisits it. The problem isn't exploration distance; it's evaluation fidelity. A trust region won't fix evaluation noise.

**Score**: impact=3 × feasibility=3 ÷ complexity=3 = **3.0**

---

### Mechanism 4: Cross-Validation on Last 3 Epochs to Reduce Variance

**Most likely failure mode**: If the training loss is monotonically decreasing (normal behavior), the last 3 epochs will show a *trend*, not stationary noise. The median of [7.8, 7.9, 8.0] is 7.9 — identical to the current single-epoch metric. The mechanism does nothing if the noise is between-run, not between-epoch.

**Implementation trap**: The `raw_epoch_bpbs` list must be parsed correctly from the binary output. If the binary only prints *best* epoch bpb (common in language modeling), there is no epoch-level data to median over. The sketch silently falls back to the single value, making the mechanism a no-op.

**Evidence from trace**: Iter 4 gets 0.1286, while revisits get ~7.9. This is an 60x *inter-run* variance, not epoch-to-epoch jitter. The same final epoch would be 0.1286 in one run and 7.9 in another. Epoch-level median can't fix this — the entire run is corrupted.

**Score**: impact=1 × feasibility=5 ÷ complexity=2 = **2.5**

---

## Synthesis

All four mechanisms miss the root cause: **the 0.1286 result is physically implausible**. A language model with LR=0.003 and WEIGHT_DECAY=1e-5 achieving 0.1286 val_bpb is either:
- A bug in the evaluation script (wrong data split, corrupted reference)
- A lucky initialization that memorized the validation set
- A numeric anomaly (NaN being ignored or clipped)

The 7.9 bpb results are the *correct* evaluations; iter 4's 0.1286 is the outlier. None of the proposed mechanisms detect *which* evaluation is the anomaly — they assume the best historical value is ground truth.

**Selected**: None. The hypotheses fail to address the fundamental plausibility problem. The high-priority item is **data validation**: verify that iter 4's 0.1286 result can be reproduced at all, with any seed, before deploying noise-detection mechanisms that will only reinforce a phantom optimum.