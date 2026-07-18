import random
import numpy as np

class MultiAxisSampler:
    """Generates multi-axis perturbations randomly selecting multiple parameters.
    Scales perturbation magnitudes inversely to observed sensitivity."""
    
    def __init__(self, active_params, perturbation_scale=0.15, history_window=10, sample_size=3, p_random=0.3):
        self.active_params = active_params
        self.perturbation_scale = perturbation_scale
        self.history_window = history_window
        self.sample_size = min(sample_size, len(active_params))
        self.p_random = p_random
        self.param_history = {p: [] for p in active_params}
        self.param_sensitivity = {p: 1.0 for p in active_params}
        self._bound_map = {
            "LR": (1e-6, 0.01, 0.001),
            "BATCH_SIZE": (8, 128, 32),
            "HIDDEN_DIM": (32, 512, 128),
        }
    
    def propose(self, current_config):
        if random.random() < self.p_random:
            return self._random_proposal(current_config)
        return self._guided_proposal(current_config)
    
    def _random_proposal(self, config):
        selected = random.sample(list(self.active_params), self.sample_size)
        return {p: self._perturb_val(p, getattr(config, p.lower(), None)) for p in selected}
    
    def _guided_proposal(self, config):
        sorted_params = sorted(self.active_params, key=lambda p: self.param_sensitivity.get(p, 1.0))
        sens = [self.param_sensitivity[p] for p in sorted_params]
        total = sum(sens)
        weights = [total / max(s, 0.1) for s in sens]
        selected = random.choices(sorted_params, weights=weights, k=self.sample_size)
        changes = {}
        for p in selected:
            sf = 1.0 / max(self.param_sensitivity[p], 0.1)
            changes[p] = self._perturb_val(p, getattr(config, p.lower(), None), sf)
        return changes
    
    def _perturb_val(self, param, old_val, scale_factor=1.0):
        if old_val is None:
            return 0.0
        bounds = self._bound_map.get(param.upper())
        if bounds is None:
            return old_val
        min_v, max_v, base = bounds
        mag = self.perturbation_scale * base * scale_factor
        if random.random() < 0.5:
            new_val = old_val * (1 + mag * random.random())
        else:
            new_val = old_val * (1 - mag * random.random())
        if param.upper() in ("BATCH_SIZE", "HIDDEN_DIM"):
            new_val = int(round(new_val))
        return max(min_v, min(max_v, new_val))
    
    def update_sensitivity(self, changes, bpb_change):
        for param in changes:
            h = self.param_history.setdefault(param, [])
            h.append(abs(bpb_change))
            if len(h) > self.history_window:
                h.pop(0)
            self.param_sensitivity[param] = np.mean(h) if h else 1.0