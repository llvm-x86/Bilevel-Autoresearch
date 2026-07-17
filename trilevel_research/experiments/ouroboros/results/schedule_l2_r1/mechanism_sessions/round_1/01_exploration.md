Based on the trace showing high discard rates (8/12 recent iterations discarded) and the system's tendency to fire Level-2 research during unstable periods, here are 3-4 concrete improvements to `AdaptiveMechanismSchedule.decide()`:

## Hypothesis 1: Volatility-Gated Research Scheduling

**Domain**: Financial market volatility trading systems

**Core idea**: Suppress Level-2/3 research when the inner loop shows high "volatility" (rapid flip-flopping between discard/keep statuses), only firing when the search has stabilized into productive exploration.

**Implementation target**: `AdaptiveMechanismSchedule.decide()` - add a volatility check before allowing L2/L3 fire decisions

**Why it helps**: In the trace, discards occur every 2-3 iterations after iteration 4's best result. This indicates the search is thrashing between degenerate configurations. Firing L2 during such volatility wastes semantic analysis on noise. Volatility gating ensures mechanisms only fire when the inner loop has established a clear improvement trajectory.

**Implementation complexity**: 2/5

**Risk**: Low - simple safeguard that doesn't block research during actual productive periods

**Pseudo-code**:
```python
def _compute_volatility(self, trace: list[dict]) -> float:
    """Compute status change frequency in recent iterations."""
    if len(trace) < 3:
        return 0.0
    changes = sum(1 for i in range(1, len(trace)) 
                  if trace[i].get("status") != trace[i-1].get("status"))
    return changes / (len(trace) - 1)

def decide(self, ...):
    volatility = self._compute_volatility(inner_trace)
    
    # Suppress L2 during high volatility periods
    if volatility > 0.6:  # Flip-flopping >60% of adjacent pairs
        fire_l2 = False
        reasons.append(f"high volatility ({volatility:.0%}); suspend L2")
```

## Hypothesis 2: Performance-Parameter Correlation Thresholding

**Domain**: Statistical process control / quality engineering

**Core idea**: Only fire Level-2 research when the correlation between hyperparameter changes and performance changes exceeds a threshold, indicating a signal worth investigating.

**Implementation target**: `AdaptiveMechanismSchedule.decide()` - add correlation check between parameter changes and val_bpb changes

**Why it helps**: The trace shows many discarded configurations with val_bpb > 7.0, while best value is 0.0998. When performance varies by 70x across configurations, most L2 analyses will be wasted on outliers. Correlation gating ensures research only fires when there's measurable impact from parameter changes.

**Implementation complexity**: 4/5

**Risk**: Medium - requires maintaining clean parameter change vectors

**Pseudo-code**:
```python
def _compute_parameter_impact_correlation(self, trace: list[dict]) -> float:
    """Compute correlation between parameter deltas and bpb changes."""
    if len(trace) < 5:
        return 0.0
    
    param_deltas = []
    perf_changes = []
    for i in range(1, len(trace)):
        prev_perf = trace[i-1].get("val_bpb", float('inf'))
        curr_perf = trace[i].get("val_bpb", float('inf'))
        if curr_perf < prev_perf:  # Only consider improvements
            perf_changes.append((prev_perf - curr_perf) / prev_perf)
            # Count parameters that changed
            delta_count = len(trace[i].get("changes", {}))
            param_deltas.append(delta_count)
    
    if len(param_deltas) < 3:
        return 0.0
    
    return np.corrcoef(param_deltas, perf_changes)[0, 1]

def decide(self, ...):
    # Only fire L2 if parameter changes actually affect performance
    correlation = self._compute_parameter_impact_correlation(inner_trace)
    if correlation < 0.3:  # Weak correlation
        fire_l2 = False
        reasons.append(f"weak param-perf correlation ({correlation:.2f})")
```

## Hypothesis 3: Temporal Reward-Consolidation Window

**Domain**: Reinforcement learning experience replay / consolidation scheduling

**Core idea**: Implement a "reward consolidation window" where L2 is automatically deferred for a configurable number of iterations after finding a new best configuration, allowing time for meaningful exploration around that point.

**Implementation target**: `AdaptiveMechanismSchedule.decide()` - add consolidation window tracking

**Why it helps**: The trace shows best result at iteration 4 (val_bpb=0.0998), followed by immediate degenerate exploration in iterations 5-11. Firing L2 research right after finding a new best wastes analysis on transient improvements. A consolidation window forces the inner loop to generate more context before triggering resource-intensive analysis.

**Implementation complexity**: 3/5

**Risk**: Low - configurable window prevents permanent blocking

**Pseudo-code**:
```python
class AdaptiveMechanismSchedule:
    def __init__(self, ...):
        self.consolidation_window = 3
        self.iteration_of_last_best = -float('inf')
    
    def decide(self, ...):
        # Track when we last found a new best
        if inner_trace:
            last_iter = inner_trace[-1]
            if last_iter.get("status") == "keep" and "is_new_best" in last_iter:
                self.iteration_of_last_best = completed_outer_cycles
        
        # Consolidation: skip L2 for N cycles after new best
        if completed_outer_cycles - self.iteration_of_last_best < self.consolidation_window:
            fire_l2 = False
            reasons.append(f"consolidating post-best (cycle {self.iteration_of_last_best})")
```

## Hypothesis 4: Gradient-Aware Research Scheduling

**Domain**: Gradient-based hyperparameter optimization / Bayesian optimization

**Core idea**: Use the "gradient" of performance changes across consecutive iterations to determine if the search is climbing a promising landscape (fire L2) or descending into noise (suppress L2).

**Implementation target**: `AdaptiveMechanismSchedule.decide()` - add performance gradient calculation

**Why it helps**: Iterations 5-11 show consistent performance degradation (val_bpb > 5.0) after the best at iteration 4. The "gradient" of val_bpb is positive (worsening). Firing L2 research during negative gradient periods wastes resources. By tracking the sliding window performance gradient, we can defer L2 when the search is clearly in a bad region (like iterations 5-11).

**Implementation complexity**: 3/5

**Risk**: Low-medium - might delay positive research during natural exploration phases

**Pseudo-code**:
```python
def _compute_performance_gradient(self, trace: list[dict]) -> float:
    """Compute slope of val_bpb over recent iterations."""
    if len(trace) < 5:
        return 0.0
    
    recent = [r.get("val_bpb", float('inf')) for r in trace[-5:]]
    # Remove outliers (inf/nan)
    recent = [x for x in recent if x < float('inf')]
    if len(recent) < 3:
        return 0.0
    
    # Simple slope using first/last diff
    slope = (recent[-1] - recent[0]) / len(recent)
    return slope

def decide(self, ...):
    gradient = self._compute_performance_gradient(inner_trace)
    
    # Positive gradient (worsening performance) suppresses L2
    if gradient > 0.5:  # Significant worsening
        fire_l2 = False
        reasons.append(f"negative gradient ({gradient:.2f} bpb/iter)")
    # Negative gradient (improving performance) encourages L2
    elif gradient < -0.5:  # Significant improvement
        fire_l3 = True  # Also consider L3 when on positive trajectory
        reasons.append(f"strong improvement gradient ({gradient:.2f} bpb/iter)")
```

**Recommended priority**: Hypothesis 1 (Volatility) → Hypothesis 3 (Consolidation) → Hypothesis 4 (Gradient) → Hypothesis 2 (Correlation)

The volatility gate is simplest and directly addresses the flip-flopping pattern in the trace. Consolidation window is complementary and prevents premature firing after new bests. Gradient awareness provides smooth gradation between "let's explore" and "let's analyze" modes. Correlation thresholding is most complex but provides the strongest signal-to-noise filter.