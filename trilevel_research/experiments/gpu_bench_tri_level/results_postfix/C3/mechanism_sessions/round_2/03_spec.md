## Implementation Specification

**Mechanism name**: `joint_perturbation_sampling`

**Implementation strategy**: `new_helper_class` + `modify_init`

**Target**: 
- New class: `MultiVariatePerturbationExplorer`
- Modify: `GpuBenchRunner.__init__` to instantiate new explorer
- Modify: `StructuredRandomExplorer.propose_exploration` to delegate to new explorer when appropriate

**Step-by-step logic**:

1. **Data Structure** (`new_helper_class`):
```python
class MultiVariatePerturbationExplorer:
    def __init__(self, param_names: List[str], param_ranges: Dict[str, Tuple[float, float]]):
        self.param_names = param_names  # ['lr', 'lr_scaler', 'batch_size'] (continuous params)
        self.param_ranges = param_ranges
        self.success_history: List[Tuple] = []  # list of (config_tuple, loss)
        self.covariance_matrix = None
        self.mean_vector = None
        self.min_samples = 5  # minimum samples before fitting covariance
        self.scale_factor = 0.1  # initial perturbation scale
        
    def _normalize_config(self, config_dict: Dict) -> np.ndarray:
        """Normalize each param to [0,1] space using its range"""
        arr = []
        for name in self.param_names:
            lo, hi = self.param_ranges[name]
            val = config_dict[name]
            normalized = (val - lo) / (hi - lo)
            arr.append(normalized)
        return np.array(arr)
    
    def _denormalize_config(self, normalized: np.ndarray) -> Dict:
        """Convert back from [0,1] to original space"""
        config = {}
        for i, name in enumerate(self.param_names):
            lo, hi = self.param_ranges[name]
            val = normalized[i] * (hi - lo) + lo
            config[name] = val
        return config
    
    def add_observation(self, config_dict: Dict, loss: float):
        """Record a new configuration and its loss"""
        self.success_history.append((config_dict, loss))
        if len(self.success_history) >= self.min_samples:
            self._update_covariance()
            
    def _update_covariance(self):
        """Fit multivariate Gaussian to best 50% of observations"""
        # Sort by loss (ascending)
        sorted_history = sorted(self.success_history, key=lambda x: x[1])
        best_n = max(len(sorted_history) // 2, self.min_samples)
        best_configs = sorted_history[:best_n]
        
        # Extract normalized configs
        normalized_configs = np.array([
            self._normalize_config(c) for c, _ in best_configs
        ])
        
        # Compute empirical mean and covariance
        self.mean_vector = np.mean(normalized_configs, axis=0)
        centered = normalized_configs - self.mean_vector
        self.covariance_matrix = np.cov(centered, rowvar=False)
        
        # Add regularization to prevent singular matrices
        self.covariance_matrix += np.eye(len(self.param_names)) * 0.01
        
    def propose_perturbation(self, current_config: Dict) -> Dict:
        """Propose joint perturbation of all continuous params"""
        if len(self.success_history) < self.min_samples:
            # Not enough data: explore single param with larger step
            return self._propose_univariate_exploration(current_config)
        
        normalized_current = self._normalize_config(current_config)
        
        # Sample from multivariate Gaussian
        # If covariance is degenerate, fall back to uniform exploration
        try:
            perturbation = np.random.multivariate_normal(
                mean=np.zeros(len(self.param_names)),
                cov=self.covariance_matrix * self.scale_factor
            )
        except np.linalg.LinAlgError:
            perturbation = np.random.uniform(
                -0.1, 0.1, size=len(self.param_names)
            )
        
        # Apply perturbation (in normalized space)
        new_normalized = normalized_current + perturbation
        new_normalized = np.clip(new_normalized, 0, 1)  # Keep in bounds
        
        new_config = self._denormalize_config(new_normalized)
        return new_config
        
    def _propose_univariate_exploration(self, current_config: Dict) -> Dict:
        """Fallback: vary one parameter at a time"""
        param = np.random.choice(self.param_names)
        lo, hi = self.param_ranges[param]
        current_val = current_config[param]
        
        # Sample uniformly in a neighborhood
        step = (hi - lo) * np.random.uniform(-0.2, 0.2)
        new_val = np.clip(current_val + step, lo, hi)
        
        new_config = current_config.copy()
        new_config[param] = new_val
        return new_config
```

2. **Integration in `__init__`** (`modify_init`):
```python
# After self.config is set up
if not self.simple_mode:
    # Get continuous parameter names and ranges
    param_names = []
    param_ranges = {}
    if hasattr(self.config, 'learning_rate_range'):
        param_names.append('lr')
        param_ranges['lr'] = self.config.learning_rate_range
    if hasattr(self.config, 'lr_scaler_range'):
        param_names.append('lr_scaler')
        param_ranges['lr_scaler'] = self.config.lr_scaler_range
    if hasattr(self.config, 'batch_size_range'):
        param_names.append('batch_size')
        param_ranges['batch_size'] = self.config.batch_size_range
    if hasattr(self.config, 'hidden_dim_range'):
        param_names.append('hidden_dim')
        param_ranges['hidden_dim'] = self.config.hidden_dim_range
    
    self.multivariate_explorer = MultiVariatePerturbationExplorer(
        param_names=param_names,
        param_ranges=param_ranges
    )
```

3. **Modify `propose_exploration`** to use joint perturbation:
```python
def propose_exploration(self, iteration, best_config):
    if hasattr(self, 'multivariate_explorer') and iteration > 5:
        # After initial exploration, use joint perturbation
        return self.multivariate_explorer.propose_perturbation(best_config)
    
    # ... existing logic for early iterations ...
```

4. **Logging hook** in `run_trial` or wherever results are processed:
```python
# After computing loss, feed back to multivariate explorer
if hasattr(self, 'multivariate_explorer'):
    self.multivariate_explorer.add_observation(config_dict, new_loss)
```

**Integration points**:

1. **Constructor**: Add `multivariate_explorer` attribute when `simple_mode=False`
2. **`propose_exploration`**: Add delegation branch for iterations > 5
3. **Post-trial hook**: Feed results back to explorer for covariance learning
4. **Configuration**: Requires `continuous_param_ranges` to be defined in `SearchConfig` or inferred from `GpuBenchConfig`
5. **Fallback**: Preserve existing single-parameter behavior when sample count < 5 or when covariance matrix is degenerate

**Key Design Decisions**:
- Normalize to [0,1] space to avoid scale issues between parameters (LR vs batch size)
- Use best 50% of observations to learn "good" joint configurations
- Add regularization to covariance matrix (ridge-like) to handle low-sample scenarios
- Maintain backward compatibility by only activating after iteration 5 (when enough history exists)

This directly addresses the pathological locality problem by allowing simultaneous perturbation of multiple parameters, increasing the probability of finding the joint optimum (LR=0.002, lr_scaler=2.0, batch_size=64) rather than moving one axis at a time.