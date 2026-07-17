def decide(
    self,
    inner_trace: list[dict],
    l2_sessions: list[MechanismSessionRecord],
    completed_outer_cycles: int,
    config: MechanismResearchConfig | None = None,
) -> ScheduleDecision:
    interval = config.level2_interval if config else self.level2_interval
    if not inner_trace or len(inner_trace) < 2:
        self.fire_level2 = False
        self.fire_level3 = False
        return ScheduleDecision.STAY
    recent_configs = [entry.get("config", {}) for entry in inner_trace[-self.window:]]
    recent_metrics = [entry.get("metric", float('inf')) for entry in inner_trace[-self.window:] if entry.get("metric") is not None]
    best_metric = min(recent_metrics) if recent_metrics else float('inf')
    cascade_result = self.detect_value_cascade(recent_configs, recent_metrics, best_metric)
    if cascade_result.is_cascade:
        self.fire_level2 = True
        self.fire_level3 = False
        return ScheduleDecision.LEVEL2
    self.fire_level2 = False
    self.fire_level3 = False
    return ScheduleDecision.STAY