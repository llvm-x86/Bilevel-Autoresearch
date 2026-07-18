def run_iteration(
    self,
    iteration: int,
    *,
    changes: dict | None = None,
    hypothesis: str = "",
) -> BenchResult:
    """Run one inner iteration. Uses LLM when changes is None."""
    if changes is None:
        if self.client is None:
            raise ValueError("LLM client required when changes not provided")
        changes, hypothesis = self._propose(iteration)

    active = set(self.search_config.active_params)
    filtered = {k: v for k, v in changes.items() if k.upper() in active}
    if not filtered:
        return BenchResult(
            val_bpb=self.trace.best_bpb,
            train_bpb=self.trace.best_bpb,
            elapsed_s=0.0,
            status="discard",
            changes={},
            hypothesis=hypothesis or "no active param changes",
            iteration=iteration,
        )

    trial = self._trial_config(iteration)
    trial.apply_changes(filtered)
    result = self._run_trial(trial, iteration=iteration, hypothesis=hypothesis)
    result.changes = filtered

    # ---- adaptive acceptance logic ----
    self.acceptance_stats["total_proposals"] += 1

    if result.status == "crash":
        result.accepted = False
        self.consecutive_rejections += 1
        self.acceptance_stats["crashes"] += 1
    else:
        dynamic_threshold = max(
            self.base_acceptance_threshold * (self.acceptance_decay ** self.consecutive_rejections),
            self.min_acceptance_threshold,
        )
        current_best = self.trace.best_val_bpb
        relative_improvement = (current_best - result.val_bpb) / current_best

        if relative_improvement > dynamic_threshold:
            result.status = "keep"
            result.accepted = True
            self.trace.best_val_bpb = result.val_bpb
            self.trace.best_bpb = result.val_bpb
            self.trace.best_iteration = iteration
            self.trace.best_config = trial
            self.config = trial
            self.consecutive_rejections = 0
            self.acceptance_stats["major_accepts"] += 1
        elif relative_improvement > 0:
            result.status = "keep_minor"
            result.accepted = True
            self.consecutive_rejections = max(0, self.consecutive_rejections - 1)
            self.acceptance_stats["minor_accepts"] += 1
        else:
            result.status = "discard"
            result.accepted = False
            self.consecutive_rejections += 1
            self.acceptance_stats["rejects"] += 1

    # stagnation detection
    self.stagnation_scores.append((iteration, self.trace.best_val_bpb))
    if len(self.stagnation_scores) > self.stagnation_window:
        self.stagnation_scores.pop(0)
    if len(self.stagnation_scores) == self.stagnation_window:
        scores = [s[1] for s in self.stagnation_scores]
        if max(scores) - min(scores) < 1e-4:
            self.consecutive_rejections = max(0, self.consecutive_rejections - 2)

    return result