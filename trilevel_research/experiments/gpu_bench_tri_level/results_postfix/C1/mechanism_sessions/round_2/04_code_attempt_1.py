class AdaptiveProposalScaling:
    """Helper class for adaptive proposal scaling mechanism used by GpuBenchRunner."""

    def __init__(self):
        self.phase = "exploration"          # "exploration" | "exploitation"
        self.trigger_count = 0              # consecutive proposals without improvement
        self.recent_improvements = []       # rolling window of val_bpb improvements
        self.best_val_bpb_initialized = False

    def should_expand_search(self) -> bool:
        return self.trigger_count >= 10 and self.phase == "exploitation"

    def should_contract_search(self) -> bool:
        if len(self.recent_improvements) >= 5:
            return all(imp < 0.0 for imp in self.recent_improvements[-5:])
        return False

    def update_state(self, result: "BenchResult", trace_best_val_bpb: float) -> None:
        if not self.best_val_bpb_initialized and result.iteration > 0:
            self.best_val_bpb_initialized = True
            self.phase = "exploitation"
            self.recent_improvements.append(0.0)  # placeholder
            return
        if result.iteration == 0:
            return
        improvement = trace_best_val_bpb - result.val_bpb if result.val_bpb is not None else 0.0
        self.recent_improvements.append(improvement)
        if len(self.recent_improvements) > 10:
            self.recent_improvements.pop(0)
        if result.accepted and result.val_bpb < trace_best_val_bpb:
            self.trigger_count = 0
            self.phase = "exploitation"
        else:
            self.trigger_count += 1
        if self.should_expand_search():
            self.phase = "exploration"
            self.trigger_count = 0
        if self.should_contract_search():
            self.phase = "exploitation"

    @staticmethod
    def apply_scaling(changes: dict, phase: str, config) -> dict:
        """Apply adaptive scaling to numeric percentage changes. Returns scaled dict."""
        import re
        CORE = {"LEARNING_RATE", "BATCH_SIZE", "WEIGHT_DECAY"}
        SECONDARY = {"MOMENTUM", "ADAM_BETA1", "ADAM_BETA2"}
        scale_map = {}
        for param in CORE:
            scale_map[param] = 2.0 if phase == "exploration" else 0.75
        scale_map["LEARNING_RATE"] = 2.0 if phase == "exploration" else 0.5
        for param in SECONDARY:
            scale_map[param] = 1.5 if phase == "exploration" else 0.9
        scaled = {}
        for param, value in changes.items():
            param_upper = param.upper()
            if param_upper in scale_map and isinstance(value, str) and "%" in value:
                match = re.match(r"([+-])(\d+(?:\.\d+)?)%", value)
                if match:
                    sign = match.group(1)
                    pct = float(match.group(2)) * scale_map[param_upper]
                    scaled[param] = f"{sign}{pct:.1f}%"
                else:
                    scaled[param] = value
            elif param_upper in scale_map and isinstance(value, (int, float)):
                scaled[param] = value * scale_map[param_upper] if phase == "exploration" else value * (1.0 - 0.5 * (1 - scale_map[param_upper]))
            else:
                scaled[param] = value
        return scaled