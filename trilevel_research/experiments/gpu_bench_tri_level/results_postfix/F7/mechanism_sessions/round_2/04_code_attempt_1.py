import random
import math
from typing import Any

class PerturbationSurrogate:
    """A local perturbation mechanism for generating proposals near the best configuration."""
    def __init__(self, radius_initial: float = 0.15, radius_min: float = 0.05, step_size_grid: dict | None = None):
        self.radius = radius_initial
        self.radius_min = radius_min
        self.step_size_grid = step_size_grid or {
            "HIDDEN_DIM": 64,
            "NUM_LAYERS": 1,
            "LEARNING_RATE": 0,
        }
        self.active = False

    def propose(self, best_config: Any, active_params: set[str], iteration: int) -> tuple[dict[str, Any], str]:
        changes = {}
        hypothesis_parts = [f"local_perturbation(radius={self.radius:.3f})"]
        for param_name in sorted(active_params):
            param_key = param_name.upper()
            current_val = getattr(best_config, param_key, None)
            if current_val is None:
                continue
            grid = self.step_size_grid.get(param_key, 0)
            if isinstance(current_val, int) and grid > 0:
                noise = random.gauss(0, self.radius * current_val)
                new_val = round((current_val + noise) / grid) * grid
                new_val = max(1, new_val)
                lower = getattr(type(best_config), param_key, None)
                upper = None
                if hasattr(best_config, f'_{param_key}_min'):
                    lower = getattr(best_config, f'_{param_key}_min')
                if hasattr(best_config, f'_{param_key}_max'):
                    upper = getattr(best_config, f'_{param_key}_max')
                if lower is not None:
                    new_val = max(lower, new_val)
                if upper is not None:
                    new_val = min(upper, new_val)
            elif isinstance(current_val, float):
                log_val = math.log(current_val) if current_val > 0 else -10
                noise = random.gauss(0, self.radius)
                new_log = log_val + noise
                new_val = math.exp(new_log)
                new_val = max(1e-6, min(1.0, new_val))
            else:
                continue
            if new_val != current_val:
                changes[param_key] = new_val
                hypothesis_parts.append(f"{param_key}={current_val}->{new_val}")
        if not changes and "HIDDEN_DIM" in active_params:
            step = self.step_size_grid.get("HIDDEN_DIM", 64)
            new_dim = getattr(best_config, "HIDDEN_DIM", 384) + step
            changes["HIDDEN_DIM"] = new_dim
            hypothesis_parts.append(f"HIDDEN_DIM=384->{new_dim}")
        hypothesis = " | ".join(hypothesis_parts)
        return changes, hypothesis

    def adjust_radius(self, result: Any) -> None:
        if result.status == "keep":
            self.radius = min(self.radius * 1.1, 0.30)
        elif result.status == "discard":
            self.radius = max(self.radius * 0.9, self.radius_min)