class NelderMeadOptimizer:
    """Nelder-Mead simplex optimization for hyperparameter tuning."""

    def __init__(
        self,
        search_config,
        param_ranges,
        alpha=1.0,
        gamma=2.0,
        rho=0.5,
        sigma=0.5,
    ):
        self.active_params = sorted(search_config.active_params)
        self.param_ranges = param_ranges
        self.n = len(self.active_params)
        self.alpha = alpha
        self.gamma = gamma
        self.rho = rho
        self.sigma = sigma
        self.simplex = []
        self.phase = "initialize"
        self._last_requested_index = None
        self._last_reflected = None
        self.best_point = None
        self.best_loss = float("inf")

    def normalize(self, params):
        normalized = {}
        for param, value in params.items():
            lo, hi = self.param_ranges.get(param, (0.0, 1.0))
            normalized[param] = (value - lo) / (hi - lo)
        return normalized

    def denormalize(self, normalized):
        denormalized = {}
        for param, value in normalized.items():
            lo, hi = self.param_ranges.get(param, (0.0, 1.0))
            denormalized[param] = value * (hi - lo) + lo
        return denormalized

    def initialize(self, best_config):
        normalized_best = self.normalize(best_config)
        self.simplex = [(best_config.copy(), None)]
        for i in range(self.n):
            new_point = best_config.copy()
            param = self.active_params[i]
            lo, hi = self.param_ranges.get(param, (0.0, 1.0))
            step = 0.05 * (hi - lo)
            nudged = normalized_best[param] + 0.05
            if nudged > 1.0:
                nudged = normalized_best[param] - 0.05
            actual_value = nudged * (hi - lo) + lo
            new_point[param] = max(lo, min(hi, actual_value))
            self.simplex.append((new_point, None))
        self.phase = "reflect"
        return self.simplex[1][0]

    def get_next_point(self, loss=None):
        if loss is not None and self._last_requested_index is not None:
            old_point = self.simplex[self._last_requested_index][0]
            self.simplex[self._last_requested_index] = (old_point, loss)
            if loss < self.best_loss:
                self.best_loss = loss
                self.best_point = old_point
            self._last_requested_index = None
        if self.phase == "initialize":
            for i, (p, l) in enumerate(self.simplex):
                if l is None:
                    return p
            self.phase = "reflect"
        if self.phase == "reflect":
            return self._reflect()
        if self.phase == "expand":
            return self._expand()
        if self.phase == "contract":
            return self._contract()
        if self.phase == "shrink":
            return self._shrink()
        return None

    def _centroid(self, exclude_last=1):
        points = self.simplex[:-exclude_last] if exclude_last else self.simplex
        centroid = {}
        for param in self.active_params:
            centroid[param] = sum(p[0][param] for p in points) / len(points)
        return centroid

    def _reflect(self):
        self.simplex.sort(key=lambda x: x[1] if x[1] is not None else float("inf"), reverse=True)
        centroid = self._centroid(exclude_last=1)
        worst = self.simplex[-1][0]
        c_norm = self.normalize(centroid)
        w_norm = self.normalize(worst)
        r_norm = {}
        for p in self.active_params:
            r_norm[p] = max(0.0, min(1.0, c_norm[p] + self.alpha * (c_norm[p] - w_norm[p])))
        reflected = self.denormalize(r_norm)
        self._last_reflected = reflected
        self._last_requested_index = -1
        self.phase = "evaluate_reflection"
        return reflected

    def _expand(self):
        centroid = self._centroid(exclude_last=1)
        c_norm = self.normalize(centroid)
        r_norm = self.normalize(self._last_reflected)
        e_norm = {}
        for p in self.active_params:
            e_norm[p] = max(0.0, min(1.0, c_norm[p] + self.gamma * (r_norm[p] - c_norm[p])))
        expanded = self.denormalize(e_norm)
        self._last_requested_index = -1
        self.phase = "reflect"
        return expanded

    def _contract(self):
        centroid = self._centroid(exclude_last=1)
        c_norm = self.normalize(centroid)
        w_norm = self.normalize(self.simplex[-1][0])
        ct_norm = {}
        for p in self.active_params:
            ct_norm[p] = max(0.0, min(1.0, c_norm[p] + self.rho * (w_norm[p] - c_norm[p])))
        contracted = self.denormalize(ct_norm)
        self._last_requested_index = -1
        self.phase = "evaluate_contraction"
        return contracted

    def _shrink(self):
        self.simplex.sort(key=lambda x: x[1] if x[1] is not None else float("inf"))
        best = self.simplex[0][0]
        b_norm = self.normalize(best)
        for i in range(1, len(self.simplex)):
            p_norm = self.normalize(self.simplex[i][0])
            s_norm = {}
            for p in self.active_params:
                s_norm[p] = max(0.0, min(1.0, b_norm[p] + self.sigma * (p_norm[p] - b_norm[p])))
            self.simplex[i] = (self.denormalize(s_norm), None)
        self.phase = "reflect"
        return self.simplex[1][0] if len(self.simplex) > 1 else None