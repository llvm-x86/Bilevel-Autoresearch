class ValueCascadeAnalyzer:
    """..."""  # docstring as in spec

    def __init__(self, value_similarity_threshold=0.85, cascade_window=5, value_change_epsilon=None):
        self.threshold = value_similarity_threshold
        self.window = cascade_window
        self.epsilon = value_change_epsilon or {
            'lr': 0.1,
            'weight_decay': 0.05,
            'batch_size': 0.1,
        }

    def compute_value_similarity(self, config_a, config_b):
        if not config_a or not config_b:
            return 0.0
        common_keys = set(config_a.keys()) & set(config_b.keys())
        if not common_keys:
            return 0.0
        total_similarity = 0.0
        for key in common_keys:
            val_a = config_a[key]
            val_b = config_b[key]
            if val_a is None or val_b is None:
                continue
            if type(val_a) != type(val_b):
                total_similarity += 0.1
                continue
            epsilon = self.epsilon.get(key, 0.1)
            if isinstance(val_a, (int, float)):
                denominator = max(abs(val_a), abs(val_b), 1e-10)
                abs_diff = abs(val_a - val_b) / denominator
                param_sim = max(0.0, 1.0 - (abs_diff / epsilon))
            elif isinstance(val_a, str):
                param_sim = 1.0 if val_a == val_b else 0.0
            elif isinstance(val_a, (list, tuple)):
                set_a, set_b = set(val_a), set(val_b)
                intersection = set_a & set_b
                union = set_a | set_b
                param_sim = len(intersection) / len(union) if union else 1.0
            else:
                param_sim = 1.0 if val_a == val_b else 0.0
            total_similarity += param_sim
        return total_similarity / len(common_keys) if common_keys else 0.0

    def detect_value_cascade(self, recent_configs, recent_metrics, best_metric):
        from collections import namedtuple
        CascadeResult = namedtuple('CascadeResult', ['is_cascade', 'cascade_length', 'worst_metric_in_window', 'similarity_trace'])
        if len(recent_configs) < 2:
            return CascadeResult(is_cascade=False, cascade_length=0, worst_metric_in_window=float('inf'), similarity_trace=[])
        similarities = []
        for i in range(1, len(recent_configs)):
            sim = self.compute_value_similarity(recent_configs[i-1], recent_configs[i])
            similarities.append(sim)
        cascade_triggered = False
        cascade_length = 0
        for sim in reversed(similarities):
            if sim > self.threshold:
                cascade_length += 1
                if cascade_length >= self.window - 1:
                    cascade_triggered = True
                    break
            else:
                cascade_length = 0
        if cascade_triggered and recent_metrics:
            min_metric_in_window = min(recent_metrics[-self.window:], default=float('inf'))
            improvement = best_metric - min_metric_in_window
            if improvement > 0.05 * best_metric:
                cascade_triggered = False
                cascade_length = 0
        return CascadeResult(
            is_cascade=cascade_triggered,
            cascade_length=cascade_length,
            worst_metric_in_window=max(recent_metrics[-self.window:], default=float('inf')),
            similarity_trace=similarities[-self.window:]
        )


class CascadeResult:
    def __init__(self, is_cascade=False, cascade_length=0, worst_metric_in_window=float('inf'), similarity_trace=None):
        self.is_cascade = is_cascade
        self.cascade_length = cascade_length
        self.worst_metric_in_window = worst_metric_in_window
        self.similarity_trace = similarity_trace or []


def decide(
    self,
    inner_trace: list[dict],
    l2_sessions: list[MechanismSessionRecord],
    completed_outer_cycles: int,
    config: MechanismResearchConfig | None = None,
) -> ScheduleDecision:
    interval = config.level2_interval if config else self.level2_interval
    l3_interval = config.level3_interval if config else self.level3_interval
    batch_size = interval

    fire_l2 = True
    fire_l3 = False
    reasons: list[str] = []

    recent = inner_trace[-self.lookback_iters:] if inner_trace else []
    if recent:
        n = len(recent)
        discards = sum(1 for r in recent if r.get("status") == "discard")
        keeps = sum(1 for r in recent if r.get("status") == "keep")
        discard_rate = discards / n
        if discard_rate > self.discard_rate_threshold:
            reasons.append(f"high discard rate ({discard_rate:.0%})")
            fire_l2 = True
        elif keeps == 0 and n >= 3:
            reasons.append("zero keeps in lookback window")
            fire_l2 = True
        elif completed_outer_cycles % interval != 0 and discard_rate < 0.5:
            fire_l2 = False
            reasons.append("inner loop improving; defer L2")

    if completed_outer_cycles % interval == 0:
        fire_l2 = True
        if "defer L2" in " ".join(reasons):
            reasons = [f"fixed interval ({interval} cycles) overrides defer"]
        elif not reasons:
            reasons.append(f"fixed L2 interval ({interval} cycles)")

    if l2_sessions:
        attempted = [s for s in l2_sessions if not s.blocked_by_tabu]
        if attempted:
            reverts = sum(1 for s in attempted if s.applied and s.validated is False)
            revert_rate = reverts / len(attempted)
            if revert_rate >= self.revert_rate_threshold:
                reasons.append(f"high revert rate ({revert_rate:.0%})")
                fire_l2 = True

    # NEW: value cascade detection
    fire_l2_from_cascade = False
    if config and getattr(config, 'value_cascade_enabled', True) and recent:
        recent_configs = [r.get("config", {}) for r in recent if isinstance(r.get("config"), dict)]
        recent_metrics = [r.get("val_bpb", float('inf')) for r in recent if r.get("val_bpb") is not None]
        all_metrics = [r.get("val_bpb", float('inf')) for r in inner_trace if r.get("val_bpb") is not None]
        best_metric = min(all_metrics) if all_metrics else float('inf')
        cascade_result = self.value_cascade_analyzer.detect_value_cascade(
            recent_configs=recent_configs,
            recent_metrics=recent_metrics,
            best_metric=best_metric
        )
        self._last_cascade_result = cascade_result
        if cascade_result.is_cascade:
            fire_l2_from_cascade = True
            reasons.append(
                f"value cascade detected (len={cascade_result.cascade_length}, "
                f"sim={cascade_result.similarity_trace[-1]:.2f})"
            )

    if fire_l2_from_cascade:
        fire_l2 = True

    if config and config.exploitation_window is not None and fire_l2:
        ...  # existing exploitation_window code (unchanged)

    return ScheduleDecision(
        fire_level2=fire_l2,
        fire_level3=fire_l3,
        batch_size=batch_size,
        reason="; ".join(reasons) if reasons else "nominal schedule"
    )