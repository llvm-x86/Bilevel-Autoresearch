class ConfigDedupParams:
    def __init__(self, min_hamming_distance: int = 2,
                 param_rounding: dict[str, float] | None = None,
                 jitter_std: float = 0.05) -> None:
        self.min_hamming_distance = min_hamming_distance
        self.param_rounding = param_rounding or {
            'LR': 0.001,
            'WEIGHT_DECAY': 1e-6,
            'BATCH_SIZE': 8,
            'HIDDEN_DIM': 128,
        }
        self.jitter_std = jitter_std


class ConfigDedupValidator:
    def __init__(self, params: ConfigDedupParams | None = None) -> None:
        self.params = params or ConfigDedupParams()
        self._seen_configs: list[dict[str, float]] = []

    def normalize_config(self, config: dict) -> dict:
        normalized: dict[str, float] = {}
        for key, value in config.items():
            rounding = self.params.param_rounding.get(key, 1.0)
            if rounding != 0:
                normalized[key] = round(value / rounding) * rounding
            else:
                normalized[key] = value
        # Return sorted by key for deterministic comparison
        return dict(sorted(normalized.items()))

    def is_novel(self, config: dict) -> tuple[bool, dict]:
        norm = self.normalize_config(config)
        for seen in self._seen_configs:
            distance = 0
            for key in set(norm.keys()) | set(seen.keys()):
                val_norm = norm.get(key, 0.0)
                val_seen = seen.get(key, 0.0)
                threshold = self.params.param_rounding.get(key, 1.0)
                if abs(val_norm - val_seen) > threshold:
                    distance += 1
            if distance < self.params.min_hamming_distance:
                return (False, {})
        self._seen_configs.append(norm)
        return (True, norm)

    def fallback_jitter(self, best_config: 'GpuBenchConfig') -> dict:
        import random
        import numpy as np
        perturbed: dict[str, float] = {}
        # When best_config is not yet set, use defaults
        if best_config is None:
            lr = 0.001
            weight_decay = 0.0
            batch_size = 64
            hidden_dim = 128
        else:
            lr = best_config.lr
            weight_decay = best_config.weight_decay
            batch_size = best_config.batch_size
            hidden_dim = best_config.hidden_dim

        perturbed['LR'] = lr * (1 + np.random.normal(0, self.params.jitter_std))
        perturbed['WEIGHT_DECAY'] = weight_decay * (1 + np.random.normal(0, self.params.jitter_std))
        perturbed['BATCH_SIZE'] = round(batch_size * (1 + np.random.normal(0, self.params.jitter_std)) / 8) * 8
        # HIDDEN_DIM: pick from [128, 256, 384, 512] not equal to the best (if possible)
        choices = [128, 256, 384, 512]
        if hidden_dim in choices:
            others = [c for c in choices if c != hidden_dim]
            if others:
                perturbed['HIDDEN_DIM'] = random.choice(others)
            else:
                perturbed['HIDDEN_DIM'] = random.choice(choices)
        else:
            perturbed['HIDDEN_DIM'] = random.choice(choices)
        return perturbed


self.dedup_validator = ConfigDedupValidator()