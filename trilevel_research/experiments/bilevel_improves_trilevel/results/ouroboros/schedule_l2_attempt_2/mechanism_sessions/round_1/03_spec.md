## Implementation Specification for AdaptiveMechanismSchedule Improvement

### 1. Mechanism name
`clamp_discard_rate_decision`

### 2. Implementation strategy
`modify_init` + `modify_method` (modify existing `decide` method with new helper)

### 3. Target
`AdaptiveMechanismSchedule.decide` (modify existing method)

### 4. Step-by-step logic

**Step 1: Add class attribute**
Add to `AdaptiveMechanismSchedule` class:
```python
min_discard_rate_threshold: float = 0.30  # new
```

**Step 2: Clamp discard_rate in decide method (lines ~29-33)**
Replace the current discard rate calculation:
```python
discard_rate = discards / n
```

With clamped version:
```python
raw_discard_rate = discards / n
discard_rate = max(self.min_discard_rate_threshold, 
                   min(self.discard_rate_threshold, raw_discard_rate))
```

**Step 3: Add transition comment block**
Insert immediately after the clamping:
```python
# Discard rate is clamped to [min_discard_rate_threshold, discard_rate_threshold]
# This prevents:
#   - Spurious L2 suppression from temporary high discard rate (upper clamp)
#   - Over-suppression of exploration when discard rate is very low (lower clamp)
# The lower clamp ensures L2 fires at least min_discard_rate_threshold% of the time
```

**Step 4: Modify the conditional logic that uses discard_rate**
Update the existing condition:
```python
if discard_rate > self.discard_rate_threshold:
```
This condition now uses the clamped value—when discard_rate hits the upper clamp, it still triggers L2. When it hits the lower clamp, the `elif discard_rate < 0.5` check still works but now `discard_rate` never drops below 0.30, reducing false "improving" signals.

**Step 5: Add docstring update**
Append to existing `decide` method docstring:
```
Note: discard_rate is clamped to [min_discard_rate_threshold, discard_rate_threshold] 
to prevent oscillation and suppress false negative/positive signals from small windows.
```

### 5. Integration points

**A. Config integration**
- `min_discard_rate_threshold` should be accessible via `MechanismResearchConfig` if that config class has schedule-related fields. Otherwise, keep as class attribute.

**B. Existing logic preservation**
- All other conditions (zero keeps check, fixed interval override, L2/L3 revert logic) remain unchanged.
- The `elif completed_outer_cycles % interval != 0 and discard_rate < 0.5` condition still works but now `discard_rate < 0.5` is harder to trigger (since lower bound is 0.30). This is the intended behavior: prevent premature deferral.

**C. Performance impact**
- Negligible (single `max/min` operation per call).

**D. Testing hooks**
- Add test: `test_discard_rate_clamping` that verifies:
  - When raw discard rate = 0.10, clamped value = 0.30 (lower clamp)
  - When raw discard rate = 0.90, clamped value = 0.70 (upper clamp)
  - When raw discard rate = 0.50, clamped value = 0.50 (passthrough)

**E. Risk mitigation**
- Risk of suppressing beneficial exploration is now controlled by `min_discard_rate_threshold` (default 0.30). This ensures L2 still fires at least 30% of the time even if discard rate is very low, preventing the exact failure mode where the schedule stops firing L2 too early.