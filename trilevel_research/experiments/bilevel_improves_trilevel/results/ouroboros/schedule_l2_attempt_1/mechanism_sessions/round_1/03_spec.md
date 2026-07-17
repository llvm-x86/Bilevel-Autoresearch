## Patch Spec: AdaptiveMechanismSchedule — Capability-Value Cascade Detector

### 1. Mechanism Name
`capability_value_cascade_detector`

### 2. Implementation Strategy
`new_helper_class` + `modify_init` + `modify_decide_method`

### 3. Target
- **New helper class**: `ValueCascadeAnalyzer` (in `adaptive_mechanisms/analysis/`)
- **Modified method**: `AdaptiveMechanismSchedule.decide()`
- **Modified init**: `AdaptiveMechanismSchedule.__init__()`
- **New config fields**: Add to `MechanismResearchConfig`

### 4. Step-by-Step Logic

#### Step 1: Create `ValueCascadeAnalyzer` helper class

```python
class ValueCascadeAnalyzer:
    """
    Analyzes hyperparameter traces for value-repetition cascades
    
    Detects when the SAME values are being proposed repeatedly 
    despite worsening performance (not just same parameter keys).
    """
    
    def __init__(
        self,
        value_similarity_threshold: float = 0.85,
        cascade_window: int = 5,
        value_change_epsilon: dict[str, float] = None
    ):
        """
        Args:
            value_similarity_threshold: Jaccard+value similarity threshold for "exploiting"
            cascade_window: How many recent configs to check for cascading
            value_change_epsilon: Per-parameter relative change tolerance 
                (e.g., {'lr': 0.1, 'weight_decay': 0.05, 'batch_size': 0.1})
                Defaults to 10% for scalar params, 5% for decay params
        """
        self.threshold = value_similarity_threshold
        self.window = cascade_window
        self.epsilon = value_change_epsilon or {
            'lr': 0.1,          # 10% relative change
            'weight_decay': 0.05, # 5% relative change
            'batch_size': 0.1    # 10% relative change (categorical discretization)
        }
```

#### Step 2: Implement value-distance computation

```python
    def compute_value_similarity(
        self, 
        config_a: dict, 
        config_b: dict
    ) -> float:
        """
        Compute config similarity using normalized value distances.
        
        Returns 0.0 for completely different, 1.0 for identical values.
        Only considers parameters present in BOTH configs.
        """
        if not config_a or not config_b:
            return 0.0
            
        common_keys = set(config_a.keys()) & set(config_b.keys())
        if not common_keys:
            return 0.0
            
        # Compute per-parameter similarity score
        total_similarity = 0.0
        
        for key in common_keys:
            val_a = config_a[key]
            val_b = config_b[key]
            
            # Handle None values
            if val_a is None or val_b is None:
                continue
                
            # Type mismatch -> low similarity
            if type(val_a) != type(val_b):
                total_similarity += 0.1
                continue
                
            epsilon = self.epsilon.get(key, 0.1)
            
            if isinstance(val_a, (int, float)):
                # Avoid division by zero
                denominator = max(abs(val_a), abs(val_b), 1e-10)
                abs_diff = abs(val_a - val_b) / denominator
                
                # Convert to similarity score (inverse of normalized diff)
                param_sim = max(0.0, 1.0 - (abs_diff / epsilon))
                
            elif isinstance(val_a, str):
                param_sim = 1.0 if val_a == val_b else 0.0
                
            elif isinstance(val_a, (list, tuple)):
                # Jaccard similarity for sequences
                set_a, set_b = set(val_a), set(val_b)
                intersection = set_a & set_b
                union = set_a | set_b
                param_sim = len(intersection) / len(union) if union else 1.0
                
            else:
                # Fallback: exact match
                param_sim = 1.0 if val_a == val_b else 0.0
                
            total_similarity += param_sim
            
        # Normalize by number of common keys
        return total_similarity / len(common_keys) if common_keys else 0.0
```

#### Step 3: Implement cascade detection logic

```python
    def detect_value_cascade(
        self,
        recent_configs: list[dict],
        recent_metrics: list[float],  # val_bpb values
        best_metric: float
    ) -> CascadeResult:
        """
        Detect value-repetition cascades.
        
        A cascade occurs when:
        1. >= cascade_window consecutive configs have value_similarity > threshold
        2. AND performance is NOT improving (val_bpb not decreasing)
        
        Returns CascadeResult with:
        - is_cascade: bool
        - cascade_length: int
        - worst_metric_in_window: float
        - similarity_trace: list[float]
        """
        if len(recent_configs) < 2:
            return CascadeResult(is_cascade=False)
        
        # Compute pairwise similarities in the window
        similarities = []
        for i in range(1, len(recent_configs)):
            sim = self.compute_value_similarity(
                recent_configs[i-1], 
                recent_configs[i]
            )
            similarities.append(sim)
        
        # Check for sustained high similarity
        cascade_triggered = False
        cascade_length = 0
        
        for sim in reversed(similarities):
            if sim > self.threshold:
                cascade_length += 1
                if cascade_length >= self.window - 1:  # -1 for pairwise
                    cascade_triggered = True
                    break
            else:
                cascade_length = 0
        
        # Verify that performance isn't improving
        if cascade_triggered and recent_metrics:
            min_metric_in_window = min(
                recent_metrics[-self.window:], 
                default=float('inf')
            )
            improvement = best_metric - min_metric_in_window
            
            # Only fire if no significant improvement
            if improvement > 0.05 * best_metric:  # 5% improvement threshold
                cascade_triggered = False
                cascade_length = 0
        
        return CascadeResult(
            is_cascade=cascade_triggered,
            cascade_length=cascade_length,
            worst_metric_in_window=max(recent_metrics[-self.window:], default=float('inf')),
            similarity_trace=similarities[-self.window:]
        )
```

#### Step 4: Add config fields to `MechanismResearchConfig`

```python
class MechanismResearchConfig:
    # ... existing fields
    
    # New cascade detection fields
    value_cascade_enabled: bool = True
    value_similarity_threshold: float = 0.85
    value_cascade_window: int = 5
    value_change_epsilon: Optional[dict[str, float]] = None
```

#### Step 5: Modify `AdaptiveMechanismSchedule.__init__()`

```python
def __init__(
    self,
    level2_interval: int = 2,
    level3_interval: int = 2,
    discard_rate_threshold: float = 0.70,
    revert_rate_threshold: float = 0.50,
    lookback_iters: int = 10,
    value_cascade_analyzer: Optional[ValueCascadeAnalyzer] = None
):
    # ... existing init code
    
    self.value_cascade_analyzer = value_cascade_analyzer or ValueCascadeAnalyzer()
    
    # Track configs for value analysis
    self._config_history: list[dict] = []
```

#### Step 6: Modify `decide()` method — add value cascade detection

```python
def decide(
    self,
    inner_trace: list[dict],
    l2_sessions: list[MechanismSessionRecord],
    completed_outer_cycles: int,
    config: MechanismResearchConfig | None = None,
) -> ScheduleDecision:
    # ... existing code (interval checks, discard rate, etc.)
    
    # NEW: Value cascade detection
    fire_l2_from_cascade = False
    
    if config and config.value_cascade_enabled and recent:
        # Extract configs from trace
        recent_configs = [
            r.get("config", {}) for r in recent
            if isinstance(r.get("config"), dict)
        ]
        
        # Extract metrics
        recent_metrics = [
            r.get("val_bpb", float('inf')) for r in recent
            if r.get("val_bpb") is not None
        ]
        
        # Find best metric in trace
        all_metrics = [
            r.get("val_bpb", float('inf')) for r in inner_trace 
            if r.get("val_bpb") is not None
        ]
        best_metric = min(all_metrics) if all_metrics else float('inf')
        
        # Analyze for value cascades
        cascade_result = self.value_cascade_analyzer.detect_value_cascade(
            recent_configs=recent_configs,
            recent_metrics=recent_metrics,
            best_metric=best_metric
        )
        
        # Store for debugging
        self._last_cascade_result = cascade_result
        
        if cascade_result.is_cascade:
            fire_l2_from_cascade = True
            reasons.append(
                f"value cascade detected (len={cascade_result.cascade_length}, "
                f"sim={cascade_result.similarity_trace[-1]:.2f})"
            )
    
    # Integrate with existing logic
    if fire_l2_from_cascade:
        fire_l2 = True
        # Override defer if cascade detected
        # Keep existing fire_l3 logic unchanged
    
    # ... rest of existing decide() code
```

### 5. Integration Points

1. **Instantiation**: In the schedule creation point (likely in `run_adaptive_mechanism.py` or `runner.py`):
   ```python
   analyzer = ValueCascadeAnalyzer(
       value_similarity_threshold=0.85,
       cascade_window=5
   )
   schedule = AdaptiveMechanismSchedule(
       value_cascade_analyzer=analyzer
   )
   ```

2. **Trace logging**: In the inner loop runner, ensure that `config` is stored in trace entries:
   ```python
   # In inner loop execution
   trace_entry = {
       "iteration": iter_idx,
       "config": current_config,  # <-- CRITICAL: must be dict
       "val_bpb": val_bpb,
       "status": status,
       # ... other fields
   }
   inner_trace.append(trace_entry)
   ```

3. **Config field exposure**: Add cascade parameters to CLI/config loading:
   ```yaml
   adaptive_mechanisms:
     value_cascade:
       enabled: true
       similarity_threshold: 0.85
       window_size: 5
       epsilon:
         lr: 0.1
         weight_decay: 0.05
         batch_size: 0.1
   ```

4. **Testing hook**: Add a `CascadeResult` accessor for unit testing:
   ```python
   @property
   def last_cascade_analysis(self