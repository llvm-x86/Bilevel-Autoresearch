## Implementation Specification: Nelder-Mead Local Optimization

### 1. Mechanism name
`nelder_mead_simplex_optimizer`

### 2. Implementation strategy
`new_helper_class` + `modify_init`

### 3. Target
New class `NelderMeadOptimizer` + modifications to `GpuBenchRunner.__init__` and `GpuBenchRunner.run_iteration`

### 4. Step-by-step logic

**Helper Class: `NelderMeadOptimizer`**

```python
class NelderMeadOptimizer:
    """Nelder-Mead simplex optimization for hyperparameter tuning.
    
    Maintains a simplex of n+1 points for n active parameters.
    Each point is a dict mapping param name -> value (normalized 0-1).
    The simplex evolves through reflection, expansion, contraction, and shrink steps.
    """
    
    def __init__(
        self,
        search_config: SearchConfig,
        param_ranges: dict[str, tuple[float, float]],
        alpha: float = 1.0,    # reflection coefficient
        gamma: float = 2.0,    # expansion coefficient
        rho: float = 0.5,      # contraction coefficient
        sigma: float = 0.5,    # shrink coefficient
    ):
        self.active_params = sorted(search_config.active_params)
        self.param_ranges = param_ranges
        self.n = len(self.active_params)
        self.alpha = alpha
        self.gamma = gamma
        self.rho = rho
        self.sigma = sigma
        
        # Simplex state: list of (params_dict, loss_value)
        self.simplex: list[tuple[dict[str, float], float | None]] = []
        
        # Current operation phase
        self.phase = "initialize"  # initialize | reflect | expand | contract | shrink
        self.current_iteration = 0
        
        # Track best point globally
        self.best_point: dict[str, float] | None = None
        self.best_loss = float('inf')
        
    def normalize(self, params: dict[str, float]) -> dict[str, float]:
        """Normalize params to [0,1] range for simplex operations."""
        normalized = {}
        for param, value in params.items():
            lo, hi = self.param_ranges.get(param, (0.0, 1.0))
            normalized[param] = (value - lo) / (hi - lo)
        return normalized
    
    def denormalize(self, normalized: dict[str, float]) -> dict[str, float]:
        """Convert normalized [0,1] values back to actual parameter values."""
        denormalized = {}
        for param, value in normalized.items():
            lo, hi = self.param_ranges.get(param, (0.0, 1.0))
            denormalized[param] = value * (hi - lo) + lo
        return denormalized
    
    def initialize(self, best_config: dict[str, float]) -> dict[str, float]:
        """Create initial simplex around best config.
        
        For n parameters, create n+1 points:
        - Point 0: current best config (from best_iteration)
        - Points 1..n: each has one parameter nudged by 5% (in normalized space)
        """
        normalized_best = self.normalize(best_config)
        
        # Start with the best point
        self.simplex = [(best_config.copy(), None)]
        
        # Generate n additional points, each varying one parameter
        for i in range(self.n):
            new_point = best_config.copy()
            param = self.active_params[i]
            lo, hi = self.param_ranges[param]
            step = 0.05 * (hi - lo)  # 5% step in original space
            
            # Nudge up, with bounds checking
            nudged = normalized_best[param] + 0.05
            if nudged > 1.0:
                nudged = normalized_best[param] - 0.05
            actual_value = nudged * (hi - lo) + lo
            new_point[param] = max(lo, min(hi, actual_value))
            self.simplex.append((new_point, None))
        
        self.phase = "reflect"
        return self.simplex[1][0]  # Return first additional point to evaluate
    
    def get_next_point(self, loss: float | None = None) -> dict[str, float] | None:
        """Return next point to evaluate, or None if optimization complete.
        
        Args:
            loss: Loss value for the last requested point, or None if trial failed
        """
        # Record loss for the last point we requested
        if loss is not None and self._last_requested_index is not None:
            point_index, _ = self.simplex[self._last_requested_index]
            self.simplex[self._last_requested_index] = (point_index, loss)
            self._last_requested_index = None
            
            if loss < self.best_loss:
                self.best_loss = loss
                self.best_point = point_index
        
        # Check if simplex is degenerate (all points too close)
        if self._is_degenerate():
            self._reinitialize_around_best()
        
        # Main Nelder-Mead logic
        if self.phase == "initialize":
            # Check if we've evaluated all initial points
            unevaluated = [(i, p) for i, (p, l) in enumerate(self.simplex) if l is None]
            if not unevaluated:
                self.phase = "reflect"
            else:
                return unevaluated[0][1]
        
        if self.phase == "reflect":
            return self._reflect_point()
        elif self.phase == "expand":
            return self._expand_point()
        elif self.phase == "contract":
            return self._contract_point()
        elif self.phase == "shrink":
            return self._shrink_point()
        
        return None
    
    def _reflect_point(self) -> dict[str, float]:
        """Calculate and return reflection point."""
        # Sort simplex by loss, worst first
        self.simplex.sort(key=lambda x: x[1] if x[1] is not None else float('inf'), reverse=True)
        
        # Calculate centroid of all points except worst
        centroid = self._calculate_centroid(exclude_last=1)
        
        # Reflection: x_r = centroid + alpha * (centroid - x_worst)
        worst_point_norm = self.normalize(self.simplex[-1][0])
        centroid_norm = self.normalize(centroid)
        reflected_norm = {}
        for param in self.active_params:
            reflected_norm[param] = centroid_norm[param] + self.alpha * (centroid_norm[param] - worst_point_norm[param])
            reflected_norm[param] = max(0.0, min(1.0, reflected_norm[param]))
        
        reflected = self.denormalize(reflected_norm)
        self._last_requested_index = -1  # Will replace worst
        self.phase = "evaluate_reflection"
        return reflected
    
    def _expand_point(self) -> dict[str, float]:
        """Calculate and return expansion point (if reflection was good)."""
        centroid = self._calculate_centroid(exclude_last=1)
        worst_point_norm = self.normalize(self.simplex[-1][0])
        centroid_norm = self.normalize(centroid)
        
        # Expansion: x_e = centroid + gamma * (x_r - centroid)
        reflected_norm = self.normalize(self._last_reflected)
        expanded_norm = {}
        for param in self.active_params:
            expanded_norm[param] = centroid_norm[param] + self.gamma * (reflected_norm[param] - centroid_norm[param])
            expanded_norm[param] = max(0.0, min(1.0, expanded_norm[param]))
        
        expanded = self.denormalize(expanded_norm)
        self._last_requested_index = -1  # Will replace worst if better
        self.phase = "reflect"  # Back to reflection after expansion
        return expanded
    
    def _contract_point(self) -> dict[str, float]:
        """Calculate and return contraction point."""
        centroid = self._calculate_centroid(exclude_last=1)
        worst_point_norm = self.normalize(self.simplex[-1][0])
        centroid_norm = self.normalize(centroid)
        
        # Contraction: x_c = centroid + rho * (x_worst - centroid)
        contracted_norm = {}
        for param in self.active_params:
            contracted_norm[param] = centroid_norm[param] + self.rho * (worst_point_norm[param] - centroid_norm[param])
            contracted_norm[param] = max(0.0, min(1.0, contracted_norm[param]))
        
        contracted = self.denormalize(contracted_norm)
        self._last_requested_index = -1
        self.phase = "evaluate_contraction"
        return contracted
    
    def _shrink_point(self) -> dict[str, float] | None:
        """Shrink simplex toward best point, return next point to evaluate."""
        # Sort by loss, best first
        self.simplex.sort(key=lambda x: x[1] if x[1] is not None else float('inf'))
        best_point = self.simplex[0][0]
        best_norm = self.normalize(best_point)
        
        # Replace all points except best with contracted versions
        for i in range(1, len(self.simplex)):
            point_norm = self.normalize(self.simplex[i][0])
            shrunk_norm = {}
            for param in self.active_params:
                shrunk_norm[param] = best_norm[param] + self.sigma * (point_norm[param] - best_norm[param])
                shrunk_norm[param] = max(0.0, min(1.0, shrunk_norm[param]))
            
            new_point = self.denormalize(shrunk_norm)
            self.simplex[i] = (new_point, None)
        
        self.phase = "reflect"
        # Return first shrunk point to evaluate
        return self.simplex[1][0] if len(self.simplex) > 1 else None
    
    def _calculate_centroid(self, exclude_last: int = 1) -> dict[str, float]:
        """Calculate centroid of simplex, excluding last n points (worst)."""
        points_to_use = self.simplex[:-exclude_last] if exclude_last > 0 else self.simplex
        centroid = {}
        for param in self.active_params:
            centroid[param] = sum(p[0][param] for p in points_to_use) / len(points_to_use)
        return centroid
    
    def _is_degenerate(self) -> bool:
        """Check if simplex has collapsed (all points too close together)."""
        if len(self.simplex) < 2:
            return False
        
        evaluated = [(p, l) for p, l in self.simplex if l is not None]
        if len(evaluated) < 2:
            return False
        
        # Check if all points are within 1% of each other in normalized space
        norms = [self