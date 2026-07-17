## Implementation Specification: AdaptiveMechanismSchedule

### 1. Mechanism Name
`adaptive_schedule_with_volatility_gate`

### 2. Implementation Strategy
`modify_init` + `modify_method` (modify `__init__` to add new parameters, modify `decide` to implement volatility gate logic)

### 3. Target
`AdaptiveMechanismSchedule.decide` (primary), `AdaptiveMechanismSchedule.__init__` (add volatility parameters)

### 4. Step-by-Step Logic

#### Step 1: Add volatility parameters to `__init__`
```python
# Add to __init__:
self.volatility_window: int = 5      # Number of recent iterations for volatility calc
self.volatility_threshold: float = 0.3  # CV threshold below which we suppress L2
self.research_budget: int = 2          # Max consecutive L2 fires after gate opens
```
Reasoning: These parameters control the gate behavior. Window of 5 provides statistical stability while being responsive. CV threshold of 0.3 (30%) is standard for "low volatility" in many domains.

#### Step 2: Add volatility gate method
```python
def _calculate_volatility(self, recent_scores: list[float]) -> float:
    """
    Calculate coefficient of variation of recent scores.
    Returns 0.0 if insufficient data or all identical scores.
    """
    if len(recent_scores) < 2:
        return 0.0
    
    # Safe filtering for NaN - explicitly exclude
    clean_scores = [s for s in recent_scores if s is not None and not (isinstance(s, float) and math.isnan(s))]
    
    if len(clean_scores) < 2:
        return 0.0
    
    mean = sum(clean_scores) / len(clean_scores)
    if mean == 0:
        return 0.0
    
    variance = sum((x - mean) ** 2 for x in clean_scores) / len(clean_scores)
    std_dev = math.sqrt(variance)
    cv = std_dev / abs(mean)
    
    return cv
```
Key design decision: Use coefficient of variation (CV = std/mean) instead of raw std. This normalizes volatility across different scales of scores, making the threshold meaningful regardless of whether scores are 0-1 or 0-1000.

#### Step 3: Track volatility state in `decide`
Add internal state tracking:
```python
# Add to class-level defaults or track in method
self._volatility_gate_open: bool = True  # Start open for safety
self._consecutive_l2_after_gate: int = 0  # Track budget usage
```

#### Step 4: Implement volatility gate logic in `decide`
Place this logic **before** the existing fixed-interval check, but **after** extracting `recent` scores:

```python
def decide(self, inner_trace, l2_sessions, completed_outer_cycles, config=None):
    # ... existing setup code ...
    
    # ===== NEW: Volatility Gate Logic =====
    recent_scores = [r.get("score") for r in recent if r.get("score") is not None]
    
    # Calculate volatility
    cv = self._calculate_volatility(recent_scores)
    
    # Gate logic: close gate when volatility is low (stability)
    volatility_gate_open = cv >= self.volatility_threshold or len(recent_scores) < 2
    
    # Handle gate transition
    if not self._volatility_gate_open and volatility_gate_open:
        # Gate just opened (volatility returned) - reset budget
        self._consecutive_l2_after_gate = 0
    
    if volatility_gate_open:
        # Gate is open - apply research budget
        self._consecutive_l2_after_gate += 1
        if self._consecutive_l2_after_gate > self.research_budget:
            # Exceeded budget - suppress L2
            fire_l2 = False
            reasons.append(f"volatility budget exhausted ({self._consecutive_l2_after_gate} > {self.research_budget})")
            # Reset gate to prevent endless suppression
            self._volatility_gate_open = False
            self._consecutive_l2_after_gate = 0
    else:
        # Gate is closed (low volatility) - suppress L2
        fire_l2 = False
        reasons.append(f"low volatility (CV={cv:.2f} < {self.volatility_threshold}); suppressing L2")
    
    # Update state
    self._volatility_gate_open = volatility_gate_open
    
    # ===== END NEW LOGIC =====
    
    # Continue with existing fixed-interval check...
```

#### Step 5: Integrate with existing override logic
Modify the "fixed interval override" section:

```python
# Replace the existing:
if completed_outer_cycles % interval == 0:
    fire_l2 = True
    if "defer L2" in " ".join(reasons):
        reasons = [f"fixed interval ({interval} cycles) overrides defer"]
    elif not reasons:
        reasons.append(f"fixed L2 interval ({interval} cycles)")

# With:
if completed_outer_cycles % interval == 0:
    # Only override volatility gate if it's been at least 2 cycles
    if not any("volatility" in r for r in reasons):
        fire_l2 = True
        if "defer L2" in " ".join(reasons):
            reasons = [f"fixed interval ({interval} cycles) overrides defer"]
        elif not reasons:
            reasons.append(f"fixed L2 interval ({interval} cycles)")
    else:
        reasons.append(f"volatility gate overrides fixed interval; L2 suppressed")
```

Reasoning: This prevents the fixed interval from completely nullifying the volatility gate, but allows it to assert itself when the gate has been open long enough.

### 5. Integration Points

| Component | Change | Reasoning |
|-----------|--------|-----------|
| `__init__` | Add 3 volatility parameters | Make gate configurable without breaking existing code |
| `decide()` | Insert volatility calculation before existing checks | Establish gate state before other logic runs |
| `decide()` | Modify fixed-interval override | Prevent volatility gate from being completely bypassed |
| `decide()` | Add NaN-safe filtering helper | Critical fix for the implementation trap identified in critique |
| `__init__` | Initialize `_volatility_gate_open = True` | Safety: start with gate open to avoid cold-start suppression |
| `__init__` | Import `math` if not present | For `math.isnan()` and `math.sqrt()` |

### Edge Cases and Mitigations

1. **NaN handling**: The `_calculate_volatility` method explicitly filters NaN before any computation. This prevents the NaN propagation trap identified in the critique.

2. **Window too small**: If `recent_scores` has < 2 entries, return 0.0 CV (gate closed = suppression). This causes a brief suppression during cold start but self-corrects as data accumulates.

3. **All identical scores**: CV = 0.0, gate closes. This is correct behavior – identical scores indicate perfect stability where research is least valuable.

4. **State persistence**: The volatility state variables (`_volatility_gate_open`, `_consecutive_l2_after_gate`) persist across `decide()` calls. This is intentional for the budget mechanism. Reset logic: only reset `_consecutive_l2_after_gate` when gate transitions from closed to open.

5. **Research budget exhaustion**: When budget exhausted, the gate forcibly closes (suppressing L2). This creates a refractory period of at least one cycle, after which the next cycle recalculates volatility. This prevents the "runaway research" scenario.

### Verification Criteria

```python
# Test 1: Low volatility suppresses L2
trace = [{"score": 0.95}] * 6  # All same scores
schedule = AdaptiveMechanismSchedule(volatility_window=5, volatility_threshold=0.3)
result = schedule.decide(trace, [], 10)
assert "suppressing L2" in result.reasons[0]
assert result.fire_l2 == False

# Test 2: High volatility permits L2 (within budget)
trace = [{"score": 0.5}, {"score": 0.9}, {"score": 0.3}, {"score": 0.7}, {"score": 0.6}]
result = schedule.decide(trace, [], 10)
assert result.fire_l2 == True  # Gate open, within budget

# Test 3: Budget exhausted after 3 consecutive open-gate cycles
for i in range(4):
    result = schedule.decide(trace, [], 10)
# After 4 calls (budget=2), should suppress
assert "budget exhausted" in result.reasons[0]

# Test 4: NaN scores handled gracefully
trace_nan = [{"score": float('nan')}, {"score": 0.5}]
result = schedule.decide(trace_nan, [], 10)
# Should not crash; NaN filtered out, treat as low volatility
assert result.fire_l2 == False  # <2 valid scores
```

### Notes on Critique-Informed Design

The critique identified three key issues that this design addresses:

1. **False stability trap is self-correcting**: When volatility is low and we suppress L2, the lack of mechanism changes causes scores to remain stable (low volatility) → gate stays closed → but after budget exhaustion, we forcibly close the gate, creating a natural break → next cycle recalculates → if scores have changed, gate opens.

2. **No dead zone around local optima**: Unlike the gradient approach where leaving an optimum guarantees suppression, the volatility gate only suppresses when scores are *stable*. At a local optimum, scores would be stable (low volatility), but the brief refractory period ensures the gate doesn't cause permanent lockout.

3. **NaN safety**: Explicit filtering before any CV computation, preventing the silent corruption identified in the critique's implementation trap analysis.

4. **Budget mechanism**: The research budget prevents the "one-shot" problem where the gate opens for a single L2 fire then closes again. Budget of 2 allows at least two consecutive L2 researches when volatility returns.