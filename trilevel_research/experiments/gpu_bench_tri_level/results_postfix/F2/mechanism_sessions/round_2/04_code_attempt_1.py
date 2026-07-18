from dataclasses import dataclass, field
import math
import random
from typing import Optional

@dataclass
class HiddenDimPerturber:
    """Samples hidden_dim near user-provided values using ±10% perturbation."""
    
    base_config: object  # GpuBenchConfig
    rng: random.Random
    perturbation_range: float = 0.10
    active_params: set = field(default_factory=lambda: {"HIDDEN_DIM"})
    
    def perturb(self, user_changes: dict) -> tuple[dict, str]:
        """Apply ±10% perturbation to hidden_dim if present in changes."""
        hidden_key = None
        for key in user_changes:
            if key.upper() == "HIDDEN_DIM":
                hidden_key = key
                break
        
        if hidden_key is None:
            return user_changes, ""
        
        original_value = user_changes[hidden_key]
        log_factor = math.log1p(self.perturbation_range)
        perturbation = math.exp(self.rng.uniform(-log_factor, log_factor))
        new_value = int(round(original_value * perturbation))
        new_value = max(64, new_value)
        
        perturbed_changes = dict(user_changes)
        perturbed_changes[hidden_key] = new_value
        
        delta_pct = ((new_value - original_value) / original_value) * 100
        perturbation_note = f"±10%: {original_value}→{new_value} ({delta_pct:+.1f}%)"
        
        return perturbed_changes, perturbation_note
    
    def should_perturb(self, iteration: int, trace: object) -> bool:
        """Decision logic: perturb unless too many consecutive failures."""
        if iteration == 0:
            return False
        recent = [r for r in trace.results[-5:] if r.hypothesis and "±10%" in r.hypothesis]
        if len(recent) >= 3:
            failures = sum(1 for r in recent[-3:] if r.status in ("discard", "crash"))
            if failures >= 3:
                return False
        return True