class ConditionalWarmupReset:
    """Helper class for divergence detection and warmup reset logic."""

    def __init__(self, warmup_steps: int = 50, divergence_threshold_multiplier: float = 3.0,
                 recent_window_size: int = 5, max_consecutive_divergences: int = 3):
        self.warmup_steps = warmup_steps
        self.divergence_threshold_multiplier = divergence_threshold_multiplier
        self.recent_window_size = recent_window_size
        self.max_consecutive_divergences = max_consecutive_divergences
        self.recent_val_bpb_window: list[float] = []
        self.consecutive_divergences = 0

    def detect_divergence(self, val_bpb: float) -> bool:
        if val_bpb == float('inf') or val_bpb != val_bpb:
            return True
        if not self.recent_val_bpb_window:
            return False
        recent = self.recent_val_bpb_window[-self.recent_window_size:]
        moving_avg = sum(recent) / len(recent)
        threshold = moving_avg * self.divergence_threshold_multiplier
        return val_bpb > threshold

    def try_warmup_reset(self, runner, trial_config, iteration: int, hypothesis: str) -> object:
        if self.consecutive_divergences >= self.max_consecutive_divergences:
            return None
        modified = GpuBenchConfig()
        modified.learning_rate = getattr(trial_config, 'learning_rate', 0.001) * 0.1
        if hasattr(modified, 'warmup_steps'):
            modified.warmup_steps = self.warmup_steps
        if hasattr(modified, 'training_steps'):
            modified.training_steps = getattr(trial_config, 'training_steps', 100) + self.warmup_steps
        return runner._run_trial(modified, iteration=iteration, hypothesis=f"{hypothesis}_warmup")

    def record_success(self, val_bpb: float):
        self.consecutive_divergences = 0
        self.recent_val_bpb_window.append(val_bpb)
        if len(self.recent_val_bpb_window) > self.recent_window_size:
            self.recent_val_bpb_window.pop(0)

    def record_divergence(self):
        self.consecutive_divergences += 1

    def reset_window(self, val_bpb: float):
        self.recent_val_bpb_window = [val_bpb]