class AdaptivePerturbationController:
    """Controls adaptive perturbation magnitude based on iteration history."""

    def __init__(
        self,
        initial_perturbation_pct: float = 0.50,
        min_perturbation_pct: float = 0.05,
        max_perturbation_pct: float = 0.75,
        shrink_factor_keep: float = 0.85,
        grow_factor_discard: float = 0.50,
        recent_window: int = 3,
        oscillation_penalty: float = 0.10,
    ):
        self.current_pct = initial_perturbation_pct
        self.min_pct = min_perturbation_pct
        self.max_pct = max_perturbation_pct
        self.shrink_factor_keep = shrink_factor_keep
        self.grow_factor_discard = grow_factor_discard
        self.recent_window = recent_window
        self.oscillation_penalty = oscillation_penalty
        self.param_directions: dict[str, list[float]] = {}
        self.last_perturbation_info: dict[str, dict] = {}

    def get_perturbation_pct(self, param_name: str, best_config) -> float:
        pct = self.current_pct
        param_range = self._get_param_range(param_name)
        if param_range > 0:
            current_val = getattr(best_config, param_name.lower())
            min_val, max_val = self._get_param_bounds(param_name)
            edge_distance = min(current_val - min_val, max_val - current_val) / param_range
            if edge_distance < 0.15:
                pct *= 0.5
        return max(self.min_pct, min(pct, self.max_pct))

    def _get_param_range(self, param_name: str) -> float:
        return 10.0  # placeholder, should be computed from config

    def _get_param_bounds(self, param_name: str):
        return 0.0, 100.0  # placeholder

    def update_after_iteration(
        self,
        iteration: int,
        result,
        proposed_changes: dict[str, tuple],
        best_config,
    ):
        if not proposed_changes:
            return
        for param, (old_val, proposed_val) in proposed_changes.items():
            if old_val is not None and proposed_val is not None:
                direction = 1 if proposed_val > old_val else (-1 if proposed_val < old_val else 0)
                if param not in self.param_directions:
                    self.param_directions[param] = []
                self.param_directions[param].append(direction)
        oscillation_count = 0
        for param in proposed_changes:
            recent = self.param_directions.get(param, [])[-self.recent_window:]
            if len(recent) >= 2:
                non_zero = [d for d in recent if d != 0]
                sign_changes = sum(1 for i in range(1, len(non_zero)) if non_zero[i] != non_zero[i-1])
                if sign_changes >= 2:
                    oscillation_count += 1
        if result.accepted and result.status == "keep":
            self.current_pct *= self.shrink_factor_keep
        else:
            if oscillation_count >= 1:
                self.current_pct *= (1.0 - self.oscillation_penalty)
            else:
                self.current_pct *= self.grow_factor_discard
        self.current_pct = max(self.min_pct, min(self.current_pct, self.max_pct))