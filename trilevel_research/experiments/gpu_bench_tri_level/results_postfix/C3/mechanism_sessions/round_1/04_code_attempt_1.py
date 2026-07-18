from dataclasses import dataclass
import math
import random
from typing import Dict, Optional, Set, Tuple

@dataclass
class ExplorationState:
    dimension_attempts: Dict[str, int] = None
    dimension_successes: Dict[str, int] = None
    last_direction: Dict[str, float] = None
    consecutive_failures: int = 0
    stagnation_counter: int = 0
    exploration_budget: int = 10
    last_improvement_iteration: int = 0

    def __post_init__(self):
        if self.dimension_attempts is None:
            self.dimension_attempts = {}
        if self.dimension_successes is None:
            self.dimension_successes = {}
        if self.last_direction is None:
            self.last_direction = {}

class StructuredRandomExplorer:
    def __init__(self, config, trace):
        self.config = config
        self.trace = trace
        self.state = ExplorationState()
        self.param_bounds = self._get_param_bounds()

    def _get_param_bounds(self):
        bounds = {}
        if hasattr(self.config, 'lr_range'):
            bounds['lr'] = self.config.lr_range
        if hasattr(self.config, 'batch_size_range'):
            bounds['batch_size'] = self.config.batch_size_range
        if hasattr(self.config, 'hidden_dim_range'):
            bounds['hidden_dim'] = self.config.hidden_dim_range
        return bounds

    def propose_exploration(self, iteration, best_config):
        active_params = set(self.config.active_params)
        if iteration - self.state.last_improvement_iteration > 5:
            return self._propose_reset_exploration(active_params, best_config)
        least_explored = self._find_least_explored_param(active_params)
        if least_explored:
            return self._probe_parameter(least_explored, best_config, active_params)
        return self._propose_local_perturbation(active_params, best_config)

    def _find_least_explored_param(self, active_params):
        if not active_params:
            return None
        attempts = {p: self.state.dimension_attempts.get(p, 0) for p in active_params}
        min_attempts = min(attempts.values())
        least_explored = [p for p, a in attempts.items() if a == min_attempts]
        return least_explored[0] if least_explored else None

    def _propose_reset_exploration(self, active_params, best_config):
        param = self._find_least_explored_param(active_params)
        if not param:
            param = next(iter(active_params))
        changes = {}
        if param == 'lr' and 'lr' in self.param_bounds:
            min_lr, max_lr = self.param_bounds['lr']
            changes['lr'] = 10 ** ((math.log10(min_lr) + math.log10(max_lr)) / 2)
        elif param == 'batch_size' and 'batch_size' in self.param_bounds:
            min_bs, max_bs = self.param_bounds['batch_size']
            changes['batch_size'] = (min_bs + max_bs) // 2
        elif param == 'hidden_dim' and 'hidden_dim' in self.param_bounds:
            min_hd, max_hd = self.param_bounds['hidden_dim']
            changes['hidden_dim'] = (min_hd + max_hd) // 2
        return changes, f"reset_exploration:{param}:midpoint"

    def _probe_parameter(self, param, best_config, active_params):
        if param not in self.param_bounds:
            return self._propose_local_perturbation(active_params, best_config)
        min_val, max_val = self.param_bounds[param]
        current_val = getattr(best_config, param, None)
        if current_val is None:
            return self._propose_local_perturbation(active_params, best_config)
        last_dir = self.state.last_direction.get(param, 1)
        failures_in_dim = self.state.dimension_attempts.get(param, 0) - self.state.dimension_successes.get(param, 0)
        direction = -last_dir if failures_in_dim > 2 else last_dir
        if param == 'lr':
            step_factor = 1.5 if direction > 0 else 1/1.5
            new_val = current_val * step_factor
        elif param == 'batch_size':
            step = 16 * direction
            new_val = max(16, min(current_val + step, max_val))
        elif param == 'hidden_dim':
            step = 32 * direction
            new_val = max(32, min(current_val + step, max_val))
        new_val = max(min_val, min(new_val, max_val))
        self.state.last_direction[param] = direction
        changes = {param: new_val}
        return changes, f"probe:{param}:dir={direction}:val={new_val}"

    def _propose_local_perturbation(self, active_params, best_config):
        param = random.choice(list(active_params))
        if param == 'lr':
            last_dir = self.state.last_direction.get('lr', 1)
            bias = 0.2 * last_dir
            noise = random.gauss(bias, 0.3)
            changes = {'lr': best_config.lr * (10 ** noise)}
        elif param == 'batch_size':
            step = random.choice([-16, -8, 8, 16])
            changes = {'batch_size': max(16, min(best_config.batch_size + step, 256))}
        elif param == 'hidden_dim':
            step = random.choice([-32, -16, 16, 32])
            changes = {'hidden_dim': max(32, min(best_config.hidden_dim + step, 1024))}
        else:
            changes = {}
        return changes, f"local_perturbation:{param}:multi_modal"

    def update_state(self, changes, was_improvement):
        for param in changes:
            self.state.dimension_attempts[param] = self.state.dimension_attempts.get(param, 0) + 1
            if was_improvement:
                self.state.dimension_successes[param] = self.state.dimension_successes.get(param, 0) + 1
        if was_improvement:
            self.state.consecutive_failures = 0
            if self.trace.results:
                self.state.last_improvement_iteration = self.trace.results[-1].iteration
        else:
            self.state.consecutive_failures += 1

# Append to __init__:
        self.explorer = StructuredRandomExplorer(
            config=self.search_config,
            trace=self.trace
        )