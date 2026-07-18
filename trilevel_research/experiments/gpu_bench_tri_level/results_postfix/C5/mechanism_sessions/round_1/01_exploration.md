Based on the trace analysis, the critical failure mode is clear: **the search repeatedly returns to the best-known config (iter 4) but fails to reproduce its val_bpb of 0.1286, instead getting stuck at ~7.9**. This indicates non-determinism, poor exploration, or a broken feedback loop. Here are concrete mechanism changes.

---

## Mechanism 1: Non-Determinism Detection & Compensation

1. **Domain**: Statistical process control / Robust optimization
2. **Core idea**: Automatically detect when a config fails to reproduce its historical performance and flag the GPU/HIP runtime variance, then compensate by running multiple seeds and taking the median.
3. **Implementation target**: `Runner.execute_run()` — after evaluating a config, compare the result against the history for identical configs.
4. **Why it helps**: The trace shows config `{'LR': 0.003, 'WEIGHT_DECAY': 1e-05}` produced val_bpb=0.1286 at iter 4, but iterations 12,14,15,16,20,21 all yield ~7.9 with the same config. This is a massive 60x performance variance that poisons the entire search. Detecting this would either trigger a different sampling strategy, increase seed count, or alert that the hardware/runtime is too noisy.
5. **Implementation complexity**: 3
6. **Risk**: low (non-invasive, purely additive logic)

**Implementation sketch**:
```python
def _detect_irreproducibility(self, config, new_val_bpb):
    hist = self.history.get(config)
    if hist and len(hist) >= 2:
        best_hist = min(hist)
        if new_val_bpb / max(best_hist, 1e-9) > 10:  # 10x worse
            self.log("WARNING: config {} previously achieved {:.4f}, now {:.4f}. Variance detected.".format(config, best_hist, new_val_bpb))
            return True
    return False
```

---

## Mechanism 2: Warm-Start Replay with Multiple Seeds

1. **Domain**: Bayesian optimization with noise-aware acquisition
2. **Core idea**: When the runner re-evaluates a previously best config, automatically run it with **3 different random seeds** and take the best result to combat non-determinism.
3. **Implementation target**: `Runner._generate_next_config()` — when the proposed config matches a historical best, switch to multi-seed evaluation mode.
4. **Why it helps**: The trace shows the search correctly identifies the best config but can't trust its evaluation. By forcing multi-seed runs on re-visits, the runner builds a more reliable performance estimate. This prevents the search from falsely discarding a good region due to a single bad seed.
5. **Implementation complexity**: 4 (requires coordinating subprocesses or re-running the binary)
6. **Risk**: medium (increases wall-clock time per iteration for revisit cases)

**Implementation sketch**:
```python
def _run_with_multiple_seeds(self, config, seeds=[42, 43, 44]):
    results = []
    for seed in seeds:
        env = os.environ.copy()
        env['RANDOM_SEED'] = str(seed)
        results.append(self._run_single(config, env_override=env))
    return min(results)  # Return best (lowest bpb)
```

---

## Mechanism 3: Local Refinement Around Best Config (Trust Region)

1. **Domain**: Trust-region methods / CMA-ES
2. **Core idea**: Instead of randomly sampling around the best config, use a shrinking Gaussian trust region centered on the best-known config, and automatically shrink the radius when re-evaluation fails to improve.
3. **Implementation target**: `Searcher.suggest()` — modify the proposal distribution to be an adaptive Gaussian centered on `self.best_config` with covariance proportional to `(current_iter - best_iter) / max_iters`.
4. **Why it helps**: The current search wastes iterations jumping to random configurations (e.g., iter 8 tries 5 changes at once). A trust region constrains exploration to the neighborhood of the known optimum, which is especially important when the optimum is sharp (as the 0.1286 result suggests). The adaptive shrinking prevents the runner from wandering too far.
5. **Implementation complexity**: 3
6. **Risk**: low (can be toggled off if search stagnates)

**Implementation sketch**:
```python
def _trust_region_proposal(self, best_config, iter_count, max_iter):
    sigma = max(0.1, 1.0 - iter_count / max_iter)  # Shrink over time
    proposal = {}
    for param, value in best_config.items():
        if param == 'LR':
            proposal[param] = np.random.lognormal(mean=np.log(value), sigma=sigma * 0.2)
        elif param == 'WEIGHT_DECAY':
            proposal[param] = np.random.lognormal(mean=np.log(value), sigma=sigma * 0.3)
        else:
            proposal[param] = value  # Keep other params fixed
    return proposal
```

---

## Mechanism 4: Cross-Validation on Last 3 Epochs to Reduce Variance

1. **Domain**: Time-series cross-validation / Early stopping variance reduction
2. **Core idea**: Instead of reporting the final validation bpb, report the **median of the last 3 epoch-level bpb values** to smooth out epoch-to-epoch noise.
3. **Implementation target**: `Runner._evaluate_checkpoint()` — modify the bpb extraction from the binary output to average/median over the final N checkpoints.
4. **Why it helps**: The massive variance (0.1286 vs 7.9 for the same config) could be caused by a single bad epoch or checkpoint corruption. Using a median over multiple epochs makes the evaluation more robust. This directly addresses the core reproducibility failure.
5. **Implementation complexity**: 2 (just post-processing logic)
6. **Risk**: low (doesn't change the model, only the metric reporting)

**Implementation sketch**:
```python
def _compute_stable_val_bpb(self, raw_epoch_bpbs):
    # raw_epoch_bpbs is a list of [epoch, bpb] pairs
    last_n = [bpb for _, bpb in raw_epoch_bpbs[-3:]]
    if len(last_n) >= 1:
        return np.median(last_n)  # Median, not mean, to be robust to outliers
    return raw_epoch_bpbs[-1][1]  # Fallback to last epoch
```

---

## Recommended Priority

Given the trace shows **catastrophic non-reproducibility** as the primary bottleneck:
1. **Mechanism 4** (cross-validation on last 3 epochs) — lowest complexity, directly addresses the symptom
2. **Mechanism 1** (non-determinism detection) — triggers awareness and can be combined with #4
3. **Mechanism 2** (multi-seed warm-start) — more complex but essential if #4 doesn't fix the variance
4. **Mechanism 3** (trust region) — addresses the exploration inefficiency, but secondary to fixing the evaluation noise