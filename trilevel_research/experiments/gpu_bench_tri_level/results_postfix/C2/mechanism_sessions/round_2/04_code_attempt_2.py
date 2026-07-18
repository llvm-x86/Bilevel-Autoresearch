# ----- Stuck state tracker (restart_with_higher_variance) -----
self._iteration_since_improvement = 0
self._best_score = float('-inf')
self._stuck_threshold = 4
self._restart_count = 0
self._last_restart_iteration = -1
self._restart_applied = False

# Small helper to encapsulate stuck detection and restart logic
class _RestartHandler:
    def __init__(self, owner: 'GpuBenchRunner') -> None:
        self.owner = owner
        self.iteration_since_improvement = 0
        self.best_score = float('-inf')
        self.stuck_threshold = 4
        self.restart_count = 0
        self.last_restart_iteration = -1
        self.restart_applied = False

    def handle(self, current_score: float) -> bool:
        if current_score > self.best_score:
            self.best_score = current_score
            self.iteration_since_improvement = 0
            return False
        self.iteration_since_improvement += 1
        if self.iteration_since_improvement >= self.stuck_threshold:
            it = self.owner._iteration_number  # assume exists
            if it - self.last_restart_iteration >= 3:
                self._apply_restart()
                self.restart_count += 1
                self.last_restart_iteration = it
                self.restart_applied = True
                return True
        return False

    def _apply_restart(self) -> None:
        vm = min(4.0 * (1.25 ** self.restart_count), 20.0)
        cfg = self.owner.config
        # Perturb each parameter with increased variance
        import random, math
        lr = cfg.LR
        new_lr = lr + random.uniform(-0.5 * vm, 0.5 * vm) * lr
        cfg.LR = max(0.0001, min(0.1, new_lr))
        wd = cfg.WEIGHT_DECAY
        new_wd = 10 ** (-6 + random.uniform(0, 2 * vm))
        cfg.WEIGHT_DECAY = max(1e-8, min(0.1, new_wd))
        bs = cfg.BATCH_SIZE
        new_bs = bs + round(random.uniform(-32 * vm, 32 * vm))
        cfg.BATCH_SIZE = max(16, min(512, new_bs))
        hd = cfg.HIDDEN_DIM
        new_hd = hd + round(random.uniform(-64 * vm, 64 * vm))
        cfg.HIDDEN_DIM = max(64, min(1024, new_hd))
        # Reset dedup validator so the new config is not rejected
        if hasattr(self.owner, 'dedup_validator'):
            self.owner.dedup_validator.reset()
        self.restart_applied = True

self._restart_handler = _RestartHandler(self)