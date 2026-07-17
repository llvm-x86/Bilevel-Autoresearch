Looking at the trace, I see a critical pattern: the search is wasting compute repeating the same configuration (LR=0.003, WEIGHT_DECAY=1e-06, BATCH_SIZE=128) across iters 8, 9, 11 while getting worse results each time. This suggests the inner loop is stuck in a local minimum and needs Level-2 intervention sooner.

Here are 4 concrete mechanism improvements:

---

## Hypothesis 1: Stagnation Detection via Entropy Collapse

**Domain**: Statistical process control / early stopping theory

**Core idea**: Fire Level-2 when the inner loop repeatedly samples nearly identical configurations without improvement, indicating search space exhaustion.

**Implementation target**: `AdaptiveMechanismSchedule._detect_stagnation()` new method + enhanced `decide()` logic

```python
def _detect_stagnation(self, recent_trace: list[dict]) -> tuple[bool, str]:
    """Detect when inner loop is stuck repeating similar configs without improvement."""
    if len(recent_trace) < 4:
        return False, ""
    
    # Check config similarity in last N iterations
    recent_configs = [r.get("changes", {}) for r in recent_trace[-4:]]
    unique_keys = set()
    for cfg in recent_configs:
        unique_keys.update(cfg.keys())
    
    # Calculate entropy of parameter combinations
    value_combinations = [tuple(sorted(cfg.items())) for cfg in recent_configs]
    unique_combos = len(set(value_combinations))
    total_combos = len(value_combinations)
    
    # If >50% of recent iterations repeat exact configs without improvement
    if unique_combos / total_combos < 0.5:
        # Check if best value has stagnated or degraded
        best_val = min(r.get("val_bpb", float("inf")) for r in recent_trace[-4:])
        last_val = recent_trace[-1].get("val_bpb", float("inf"))
        
        if last_val > best_val * 1.05:  # 5% degradation
            return True, f"search stagnation: {unique_combos}/{total_combos} unique configs last 4 iters"
    
    return False, ""
```

**Why it helps**: The trace shows iters 8,9,11 all use identical config `{'LR': 0.003, 'WEIGHT_DECAY': 1e-06, 'BATCH_SIZE': 128}` with worsening val_bpb (7.22 → 4.96 → 8.11). This stagnation indicates the mutation operators have exhausted their local neighborhood and need Level-2's broader search space reset.

**Implementation complexity**: 2 (simple statistical computation)

**Risk**: Low - only triggers in clear stagnation cases, doesn't interfere with normal search

---

## Hypothesis 2: Degradation Trend Analysis with Weighted Momentum

**Domain**: Time series anomaly detection / momentum in optimization

**Core idea**: Detect when the inner loop's best performance is consistently degrading by computing an exponentially-weighted trend over recent iterations, firing Level-2 when trend turns negative.

**Implementation target**: `AdaptiveMechanismSchedule._compute_degradation_trend()` new method + modified fire_l2 decision logic

```python
def _compute_degradation_trend(self, recent_trace: list[dict]) -> tuple[float, bool]:
    """Compute weighted performance trend. Returns (trend, is_degrading)."""
    if len(recent_trace) < 4:
        return 0.0, False
    
    # Extract val_bpb with recency weighting (more recent = higher weight)
    values = []
    weights = []
    for i, r in enumerate(recent_trace[-6:]):  # max 6 iters lookback
        val = r.get("val_bpb")
        if val is not None:
            values.append(val)
            weights.append(1.5 ** i)  # exponential recency weighting
    
    if len(values) < 3:
        return 0.0, False
    
    # Simple linear regression with weighted points
    n = len(values)
    sum_w = sum(weights)
    sum_wx = sum(w * i for i, w in enumerate(weights))
    sum_wy = sum(w * v for v, w in zip(values, weights))
    sum_wxy = sum(w * i * v for i, (v, w) in enumerate(zip(values, weights)))
    sum_wx2 = sum(w * i * i for i, w in enumerate(weights))
    
    slope = (sum_w * sum_wxy - sum_wx * sum_wy) / (sum_w * sum_wx2 - sum_wx * sum_wx + 1e-10)
    
    # Positive slope means degrading performance (higher val_bpb is worse)
    is_degrading = slope > 0.05 and values[-1] > values[0] * 1.1
    return slope, is_degrading
```

**Why it helps**: The trace shows a clear degradation pattern: iter 5 (3.865 best), then 8 (7.22), 9 (4.96), 10 (5.07), 11 (8.11). A weighted trend would detect this monotonic degradation and trigger Level-2 before too many wasted iterations.

**Implementation complexity**: 3 (weighted regression computation)

**Risk**: Low-Medium - need to calibrate threshold to avoid false positives from natural variance

---

## Hypothesis 3: Discard Cascade Early Warning System

**Domain**: Sequential hypothesis testing / change point detection

**Core idea**: Instead of just counting discard rate, detect when discard quality is worsening (i.e., discards have progressively higher val_bpb), indicating the mutation space is systematically producing worse configurations.

**Implementation target**: `AdaptiveMechanismSchedule._analyze_discard_cascade()` new method

```python
def _analyze_discard_cascade(self, recent_trace: list[dict]) -> tuple[bool, str]:
    """Detect when discards are increasing in quality (getting worse)."""
    if len(recent_trace) < 5:
        return False, ""
    
    # Get last 5 discard events, not consecutive iterations
    discards = [r for r in recent_trace[-10:] if r.get("status") == "discard"]
    if len(discards) < 3:
        return False, ""
    
    # Check if discard values are monotonically increasing (getting worse)
    discard_values = [r.get("val_bpb", float("inf")) for r in discards[-3:]]
    is_monotonic_worsening = all(
        discard_values[i] > discard_values[i-1] 
        for i in range(1, len(discard_values))
    )
    
    # Check if the gap between best and current discards is growing
    best_val = min(r.get("val_bpb", float("inf")) for r in recent_trace)
    last_discard_val = discard_values[-1]
    gap_ratio = last_discard_val / (best_val + 1e-10)
    
    if is_monotonic_worsening and gap_ratio > 2.0:
        return True, f"discard cascade: last 3 discards worsening ({discard_values[-3]:.1f}→{discard_values[-1]:.1f}), gap={gap_ratio:.1f}x"
    
    return False, ""
```

**Why it helps**: In the trace, discards go 7.38 (iter 6) → 7.47 (iter 7) → 7.22 (iter 8) → 4.96 (iter 9) → 5.07 (iter 10) → 8.11 (iter 11). The last 3 consecutive discards (9→10→11) show 4.96→5.07→8.11 - a clear worsening cascade. This early warning would fire Level-2 after iter 9 or 10, saving 1-2 wasted iterations.

**Implementation complexity**: 2 (simple trend analysis)

**Risk**: Low - only triggers on clear cascade patterns with significant gap ratios

---

## Hypothesis 4: Exploitation-Exploration Phase Detection

**Domain**: Multi-armed bandit / explore-exploit tradeoff theory

**Core idea**: Dynamically adjust Level-2 firing based on whether the inner loop is in "exploitation" (repeating best config) vs. "exploration" (trying new combos) phase, firing Level-2 when exploitation exceeds a threshold without improvement.

**Implementation target**: `AdaptiveMechanismSchedule._classify_search_phase()` method + modified interval-based firing logic

```python
def _classify_search_phase(self, recent_trace: list[dict]) -> dict:
    """Classify current search phase and recommend Level-2 action."""
    if len(recent_trace) < 4:
        return {"phase": "warmup", "exploit_ratio": 0.0, "needs_l2": False}
    
    # Find best config in trace
    best_entry = min(recent_trace, key=lambda r: r.get("val_bpb", float("inf")))
    best_config = best_entry.get("changes", {})
    best_val = best_entry.get("val_bpb", float("inf"))
    
    # Count how many recent iterations exploit (repeat best config or small perturbation)
    exploitation_count = 0
    for r in recent_trace[-4:]:
        current_config = r.get("changes", {})
        # Count as exploitation if configs share >70% of parameters
        shared_params = len(set(current_config.keys()) & set(best_config.keys()))
        total_unique = len(set(list(current_config.keys()) + list(best_config.keys())))
        similarity = shared_params / (total_unique + 1e-10)
        if similarity > 0.7:
            exploitation_count += 1
    
    exploit_ratio = exploitation_count / min(4, len(recent_trace))
    
    # Phase determination
    if exploit_ratio > 0.75 and recent_trace[-1].get("val_bpb", 0) > best_val * 1.1:
        # Late-stage exploitation without improvement → needs Level-2 reset
        return {
            "phase": "stuck_exploitation",
            "exploit_ratio": exploit_ratio,
            "needs_l2": True,
            "reason": f"over-exploiting ({exploit_ratio:.0%}) best config without improvement"
        }
    elif exploit_ratio < 0.3:
        # Active exploration - no intervention needed
        return {
            "phase": "exploration",
            "exploit_ratio": exploit_ratio,
            "needs_l2": False
        }
    else:
        return {
            "phase": "balanced",
            "exploit_ratio": exploit_ratio,
            "needs_l2": False
        }
```

**Why it helps**: The trace shows heavy exploitation around the best config (iters 5-11 all use similar LR=0.003 combinations). When exploitation exceeds 75% without improving on the best val_bpb of 3.865, it indicates the mutation operators are stuck in a local neighborhood. Level-2 should fire to diversify the search space.

**Implementation complexity**: 3 (needs config similarity computation)

**Risk**: Low-Medium - similarity threshold may need tuning per problem domain

---

## Implementation Priority Recommendation

For maximum immediate impact with minimal complexity, implement in this order:

1. **Hypothesis 3** (Discard Cascade, complexity 2) - Quick win, detects the most obvious failure pattern in your trace
2. **Hypothesis 1** (Stagnation Detection, complexity 2) - Complements Hypothesis 3 with config-repeat detection
3. **Hypothesis 4** (Phase Detection, complexity 3) - Adds theoretical grounding to when Level-2 fires
4. **Hypothesis 2** (Degradation Trend, complexity 3) - Most sophisticated, but potentially most robust long-term

All four mechanisms can coexist in `decide()` - they provide independent signals that can be combined: fire Level-2 if ANY detection fires, or implement a voting system (e.g., fire if 2+ mechanisms agree).