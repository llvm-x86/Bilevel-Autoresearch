from pathlib import Path
from typing import Any

class ConfigCache:
    """Stores multiple runs per unique config and provides aggregated metrics."""
    
    def __init__(self, min_replicates: int = 2, cache_dir: Path | None = None):
        self.min_replicates = max(1, min_replicates)
        self._cache: dict[frozenset, list[dict]] = {}
        self._cache_dir = cache_dir
        if cache_dir is not None:
            self._cache_file = cache_dir / "config_cache.json"
            self._load()
    
    def _config_key(self, config: 'GpuBenchConfig') -> frozenset:
        params = {
            k: getattr(config, k.lower())
            for k in ['HIDDEN_DIM', 'LR', 'BATCH_SIZE', 'GRAD_CLIP']
            if hasattr(config, k.lower())
        }
        return frozenset(params.items())
    
    def _load(self) -> None:
        import json
        if self._cache_file.exists():
            try:
                with open(self._cache_file, 'r') as f:
                    data = json.load(f)
                for key_str, entries in data.items():
                    key = frozenset(eval(key_str))
                    self._cache[key] = entries
            except (json.JSONDecodeError, ValueError):
                pass
    
    def _save(self) -> None:
        if self._cache_dir is None:
            return
        import json
        data = {str(list(key)): entries for key, entries in self._cache.items()}
        self._cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._cache_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def add_result(self, config: 'GpuBenchConfig', result: 'BenchResult') -> None:
        key = self._config_key(config)
        if key not in self._cache:
            self._cache[key] = []
        self._cache[key].append({
            'val_bpb': result.val_bpb,
            'status': result.status,
        })
        self._save()
    
    def get_aggregated(self, config: 'GpuBenchConfig') -> float | None:
        key = self._config_key(config)
        entries = self._cache.get(key, [])
        vals = [e['val_bpb'] for e in entries if e['status'] != 'crash']
        if len(vals) < self.min_replicates:
            return None
        return sum(vals) / len(vals)
    
    def needs_replicate(self, config: 'GpuBenchConfig') -> bool:
        key = self._config_key(config)
        entries = self._cache.get(key, [])
        valid = [e for e in entries if e['status'] != 'crash']
        return len(valid) < self.min_replicates