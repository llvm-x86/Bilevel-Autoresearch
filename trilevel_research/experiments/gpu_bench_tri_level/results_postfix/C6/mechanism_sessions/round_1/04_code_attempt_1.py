class ResampleDiverseMechanism:
    """Detects plateau and generates diverse configs for exploration."""

    def __init__(self, runner, window: int = 5, threshold: float = 0.01,
                 frequency: int = 3):
        self.runner = runner
        self.window = window
        self.threshold = threshold
        self.frequency = frequency
        self.last_resample_iter = -self.frequency

    def detect_plateau(self, iteration: int) -> bool:
        trace = self.runner.trace
        if len(trace.results) < self.window:
            return False
        recent = trace.results[-self.window:]
        if not recent:
            return False
        best_recent = min(r.val_bpb for r in recent if r.status in ("keep", "discard"))
        best_ever = trace.best_bpb
        if abs(best_recent - best_ever) / max(best_ever, 0.01) > self.threshold:
            return False
        last_configs = [r.config for r in recent[-3:] if hasattr(r, 'config')]
        if len(last_configs) < 3:
            return False
        lr_vals = [c.learning_rate for c in last_configs]
        hd_vals = [c.hidden_dim for c in last_configs]
        lr_narrow = (max(lr_vals) - min(lr_vals)) < 0.0005
        hd_narrow = (max(hd_vals) - min(hd_vals)) < 32 if hd_vals else False
        return lr_narrow and hd_narrow

    def resample_if_stuck(self, iteration: int):
        if not self.runner.search_config.resample_enabled:
            return None
        if iteration - self.last_resample_iter < self.frequency:
            return None
        if not self.detect_plateau(iteration):
            return None
        config = self.runner.config
        search_config = self.runner.search_config
        changes = {}
        parts = ["Resample: forced diverse exploration"]
        # Flip learning rate
        current_lr = config.learning_rate
        new_lr = 0.01 if current_lr < 0.005 else 0.001
        changes["learning_rate"] = new_lr
        parts.append(f"LR={new_lr}")
        # Flip hidden dim
        current_hd = config.hidden_dim
        new_hd = 128 if current_hd >= 256 else 512
        changes["hidden_dim"] = new_hd
        parts.append(f"HIDDEN_DIM={new_hd}")
        # Toggle batch size if active
        if hasattr(search_config, 'active_params') and 'BATCH_SIZE' in search_config.active_params:
            current_bs = config.batch_size
            new_bs = 64 if current_bs <= 32 else 16
            changes["batch_size"] = new_bs
            parts.append(f"BATCH_SIZE={new_bs}")
        # Toggle optimizer if active
        if hasattr(search_config, 'active_params') and 'OPTIMIZER' in search_config.active_params:
            current_opt = config.optimizer
            new_opt = "adam" if current_opt == "sgd" else "sgd"
            changes["optimizer"] = new_opt
            parts.append(f"OPTIM={new_opt}")
        hypothesis = " | ".join(parts)
        self.last_resample_iter = iteration
        return changes, hypothesis