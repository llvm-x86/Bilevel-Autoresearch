class DiscreteCliffExplorer:
    """Handles discrete optimization cliffs where continuous perturbations fail."""

    def __init__(self, search_config):
        self.search_config = search_config
        self.cliff_detected = False
        self.cliff_param = None
        self.cliff_values = []
        self.exploration_phase = 0
        self.alternatives_tried = set()
        self.precision_tried = set()
        self.batch_size_alternatives = {
            256: [320, 384, 448],
            512: [384, 448, 576, 640],
            128: [160, 192, 224],
            1024: [768, 896, 1152, 1280]
        }
        self.precision_modes = ['fp16', 'mixed', 'tf32']

    def detect_cliff(self, trace):
        if len(trace.results) < 3:
            return False
        param_values = {}
        for result in trace.results:
            if hasattr(result, 'changes') and result.changes:
                for param, value in result.changes.items():
                    param_values.setdefault(param, {})[value] = result.val_bpb
        for param, values_dict in param_values.items():
            sorted_vals = sorted(values_dict.items())
            if len(sorted_vals) >= 2:
                for i in range(len(sorted_vals) - 1):
                    val1, bpb1 = sorted_vals[i]
                    val2, bpb2 = sorted_vals[i + 1]
                    if abs(bpb1 - bpb2) > 0.003:
                        self.cliff_param = param
                        self.cliff_values = sorted_vals
                        self.cliff_detected = True
                        return True
        return False

    def propose_alternatives(self, current_config):
        if self.exploration_phase == 0:
            if self.cliff_param == 'BATCH_SIZE':
                cliff_vals = [v for v, _ in self.cliff_values]
                if len(cliff_vals) >= 2:
                    low_val = min(cliff_vals)
                    high_val = max(cliff_vals)
                    for power_val in (low_val, high_val):
                        for alt in self.batch_size_alternatives.get(power_val, []):
                            if alt not in self.alternatives_tried:
                                self.alternatives_tried.add(alt)
                                return {'BATCH_SIZE': alt}
            self.exploration_phase = 1
        if self.exploration_phase == 1:
            for precision in self.precision_modes:
                if precision not in self.precision_tried:
                    self.precision_tried.add(precision)
                    return {'PRECISION': precision}
            self.exploration_phase = 2
        if self.exploration_phase == 2:
            cliff_vals = [v for v, _ in self.cliff_values]
            if cliff_vals:
                high_val = max(cliff_vals)
                for alt in self.batch_size_alternatives.get(high_val, []):
                    for precision in self.precision_modes:
                        combo = (alt, precision)
                        if combo not in self.alternatives_tried:
                            self.alternatives_tried.add(combo)
                            return {'BATCH_SIZE': alt, 'PRECISION': precision}
        return {}

    def should_activate(self, trace, current_val_bpb, best_val_bpb):
        recent_failures = 0
        for result in reversed(trace.results[-10:]):
            if result.status == "discard" and not result.accepted:
                recent_failures += 1
            else:
                break
        stagnant = (len(trace.results) >= 5 and
                    trace.results[-1].val_bpb >= best_val_bpb * 1.001)
        return (recent_failures >= 3) or (stagnant and self.detect_cliff(trace))