class SensitivityClock:
    """Tracks parameter sensitivity and enforces exploration fairness."""

    def __init__(self, active_params: set[str]):
        self._param_trials: dict[str, dict[float, list[float]]] = {
            p.upper(): {} for p in active_params
        }
        self._variation_count: dict[str, int] = {
            p.upper(): 0 for p in active_params
        }
        self._locked: dict[str, bool] = {
            p.upper(): False for p in active_params
        }
        self._min_variations = 5
        self._min_distinct_values = 3
        self._sensitivity_ratio = 0.02

    def record_trial(self, param: str, value: float, val_bpb: float) -> None:
        p = param.upper()
        if p not in self._param_trials:
            return
        if value not in self._param_trials[p]:
            self._param_trials[p][value] = []
        self._param_trials[p][value].append(val_bpb)
        self._variation_count[p] += 1

    def record_variation(self, param: str) -> None:
        p = param.upper()
        if p in self._variation_count:
            self._variation_count[p] += 1

    def is_sensitive(self, param: str, best_bpb: float) -> bool | None:
        p = param.upper()
        if p not in self._param_trials:
            return None
        values_tried = len(self._param_trials[p])
        if values_tried < self._min_distinct_values:
            return None
        if self._variation_count[p] < self._min_variations:
            return None
        value_means = []
        for v, results in self._param_trials[p].items():
            if results:
                value_means.append(sum(results) / len(results))
        if len(value_means) < 2:
            return None
        median_mean = sorted(value_means)[len(value_means) // 2]
        for mean in value_means:
            improvement_ratio = (median_mean - mean) / median_mean
            if improvement_ratio > 0.10:
                return True
        range_bpb = max(value_means) - min(value_means)
        threshold = best_bpb * self._sensitivity_ratio
        return range_bpb > threshold

    def lock_insensitive_params(self, best_bpb: float) -> list[str]:
        newly_locked = []
        for p in list(self._locked.keys()):
            if self._locked[p]:
                continue
            sensitivity = self.is_sensitive(p, best_bpb)
            if sensitivity is False:
                self._locked[p] = True
                newly_locked.append(p)
        return newly_locked

    def is_locked(self, param: str) -> bool:
        return self._locked.get(param.upper(), False)

    def count_variations(self, param: str) -> int:
        return self._variation_count.get(param.upper(), 0)