self.discard_tracker = DiscardTracker()

# --- DiscardTracker helper class for expand_radius_on_discards mechanism ---
class DiscardTracker:
    """Tracks consecutive discards and expands search radius accordingly."""
    def __init__(self, base_radius: float = 0.1, expansion_factor: float = 1.5):
        self.consecutive_discards = 0
        self.current_radius = base_radius
        self.base_radius = base_radius
        self.expansion_factor = expansion_factor
        self.max_radius = 0.5

    def record_result(self, status: str, config: 'GpuBenchConfig') -> None:
        if status == "discard":
            self.consecutive_discards += 1
            expansion = self.expansion_factor ** min(self.consecutive_discards, 5)
            self.current_radius = min(
                self.base_radius * expansion,
                self.max_radius
            )
        else:  # keep or crash resets
            self.consecutive_discards = 0
            self.current_radius = self.base_radius

    def get_perturbation_range(self, param_name: str, config: 'GpuBenchConfig') -> float:
        ranges = {
            'LEARNING_RATE': (0.0001, 0.01),
            'HIDDEN_DIM': (32, 512),
            'BATCH_SIZE': (16, 128),
        }
        low, high = ranges.get(param_name, (0, 1))
        param_range = high - low
        return param_range * self.current_radius