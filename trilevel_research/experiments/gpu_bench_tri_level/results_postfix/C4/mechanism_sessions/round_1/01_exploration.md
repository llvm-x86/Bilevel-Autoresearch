Based on the trace analysis, the runner is stuck in a local optimum (val_bpb=0.0296) and all subsequent proposals are discarded. The problem is not slow convergence but **complete search failure** after iteration 4.

Here are 4 mechanism improvements to fix this:

---

## Mechanism 1: Hyperparameter Decay Scheduling

**Domain:** Optimization / Learning Rate Schedules

**Core idea:** Automatically decay `LR` by sqrt(10) every 5 iterations without improvement, giving systematic exploration at lower learning rates rather than randomly trying 0.0015 repeatedly.

**Implementation target:** `_select_next_config()` method — add a `_decay_lr()` helper that checks `self.best_val_bpb_age` and applies geometric decay when stagnation > 5 iterations.

**Why it helps:** The trace shows `LR=0.0015` was tried 3 times as a manual guess but never combined with `BATCH_SIZE=128`. Systematic decay forces testing `LR=0.001, 0.0003, 0.0001` with your known-good batch size, exploring regions that might yield lower bpb.

**Implementation complexity:** 2 (add 10 lines for counter + decay logic)

**Risk:** Low — Preserves ability to find better solutions; safe because learning rates monotonically decrease.

---

## Mechanism 2: Adaptive Perturbation Range

**Domain:** Evolutionary Strategies / ES

**Core idea:** When the last N proposals are all discarded, expand the hyperparameter perturbation range (e.g., multiply step sizes by 2) to escape the local basin.

**Implementation target:** `_get_next_changes()` — add `self._consecutive_discards` counter that, when >= 5, doubles the step size for `BATCH_SIZE` (e.g., from ±16 to ±32) and `HIDDEN_DIM` (from adjust to ±128).

**Why it helps:** Current changes are too conservative (BATCH_SIZE 112, 128, 144, 160). All are near 128. The runner needs to jump to BATCH_SIZE=64 or 256 with confidence. Wider exploration increases chance of finding different valley.

**Implementation complexity:** 3 (need to track `consecutive_discards`, modify perturbation logic, reset on any keep)

**Risk:** Medium — Could temporarily overshoot, but automatically resets when improvement found. Worth the risk given current failure.

---

## Mechanism 3: Best Config Cache with Restart

**Domain:** Iterative Optimization / Restart Strategies

**Core idea:** After 10 consecutive discards from a given best config, temporarily revert to that best config but apply a **different single hyperparameter change** (randomly chosen, not greedy).

**Implementation target:** `run_iteration()` — before calling `_select_next_config()`, check `self._discards_since_keep`. If >= 10, set a flag that forces next proposal to be "best + 1 random change" using `random.choice(hp_keys)`.

**Why it helps:** Current search is path-dependent and forgot the good valley (BATCH_SIZE=128). This mechanism forces revisiting that valley with fresh random perturbations, breaking the unsuccessful exploration chain.

**Implementation complexity:** 4 (need state machine: normal mode vs. restart mode)

**Risk:** Low-Medium — Still exploring; just resets direction to a known-good point. Guarantees the runner doesn't wander too far.

---

## Mechanism 4: Adaptive Loss Landscape Smoothing

**Domain:** Bayesian Optimization / Acquisition Function

**Core idea:** Before proposing changes, roughly estimate the loss landscape curvature around the current best by running 2 **smaller** trials (fewer epochs) with perturbed hyperparameters, then only propose changes where the estimate suggests improvement.

**Implementation target:** New method `_quick_estimate_curvature(self, base_config)` — runs 20% of normal training epochs for `LR * 0.5` and `LR * 2.0`, compares their BPB. If both are worse than best known, flag a "plateau" and trigger larger perturbations.

**Why it helps:** Many discards (LR=0.001) are obviously worse. A quick estimate would save iterations and focus exploration on promising directions. The trace shows the runner wastes iterations on clearly worse configs.

**Implementation complexity:** 5 (requires subprocess for quick trials, stopping early, storing partial results)

**Risk:** High — Quick estimates could be noisy; might miss good configs that need full training. But could drastically reduce wasted iterations.

---

## Recommended Priority

1. **Mechanism 1** (Decay) — Simple, safe, addresses the clear LR=0.003 stagnation
2. **Mechanism 2** (Adaptive Range) — Directly solves the "too conservative" problem
3. **Mechanism 3** (Restart) — Safety net for when both above fail
4. **Mechanism 4** (Curvature) — Only if exploration budget is tight

The core problem is **exploration trap**: the runner keeps trying configurations near the current best but never finds anything better, nor does it systematically explore different orders of magnitude. Mechanisms 1+2 directly fix this.