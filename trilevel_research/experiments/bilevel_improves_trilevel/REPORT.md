# bilevel_improves_trilevel — Run Report

**Date:** 2026-07-17  
**Host:** ASUS (a112@100.78.97.35)  
**Driver PID:** 35450 (completed normally)  
**Verdict:** **SUCCESS**

## Summary

The bilevel→tri-level hypothesis run completed in ~10.5 minutes. Ouroboros failed to validate LLM-generated schedule L2 patches on both attempts and fell back to the hand-written **bootstrap** patch (`bootstrap_improving_streak_defer`), which applied and validated (L2 apply rate = 1.0). The promoted schedule was used for a paired 4×4 ablation. Group **F** (tri-level, L3 enabled) beat Group **C** (bi-level, L3 off) on mean Δval_bpb by **0.0123**, exceeding the required **0.01** margin.

## Ouroboros (schedule L2)

| Attempt | Mechanism | Applied | Validated | Error |
|---------|-----------|---------|-----------|-------|
| 1 | `schedule_mechanism_20260717_130027` | No | No | import_fail |
| 2 | `schedule_mechanism_20260717_130254` | No | No | import_fail |

After 2 failed LLM attempts, bootstrap fallback was applied:

- **Status:** `bootstrap`
- **Mechanism:** `bootstrap_improving_streak_defer` (hand-written `BOOTSTRAP_DECIDE_CODE`)
- **Schedule path:** `results/ouroboros/bootstrap_patch/adaptive_mechanism_schedule.py`
- **Ouroboros L2 apply rate:** 1.0 (1/1 bootstrap session applied)
- **Schedule promoted:** yes → `trilevel_research/core/adaptive_mechanism_schedule.py`

## Ablation: Group C vs Group F

| Group | Meaning | L3 | Repeats OK | Mean Δval_bpb | Stdev |
|-------|---------|----|------------|---------------|-------|
| **C** | Bi-level (L1 + L1.5 + L2) | off | 4/4 | **6.338030** | 0.047 |
| **F** | Tri-level (+ L3 meta-loop) | on | 4/4 | **6.350361** | 0.035 |

**Margin:** F − C = **+0.012331** (required > +0.010) ✓

### Per-repeat improvements (Δval_bpb)

| Repeat | C | F |
|--------|---|---|
| 1 | 6.296324 | 6.382066 |
| 2 | 6.285347 | 6.296324 |
| 3 | 6.388383 | 6.340989 |
| 4 | 6.382066 | 6.382066 |

### L2 apply rates (ablation inner-loop mechanisms)

All ablation L2 mechanism proposals failed to apply (`applied=False`):

| Group | Repeat | L2 apply rate |
|-------|--------|---------------|
| C | 1–4 | 0.0 each |
| F | 1–4 | 0.0 each |

Notable ablation L2 attempts (all `applied=False`): `coord_descent_round`, `bayesian_optimization_acquisition`, `exploration_bonus_on_stagnation`, `repeat_detection_and_cooling`, and several LLM-generated gpu_bench mechanisms.

**Success criterion note:** Overall verdict uses **ouroboros** L2 apply rate (1.0 from bootstrap), not per-ablation L2 rates (all 0.0). The MANAGER success gate is F mean > C mean + 0.01 **and** ouroboros L2 apply > 0.

## Success criteria check

| Criterion | Result |
|-----------|--------|
| F mean Δval_bpb > C mean + 0.01 | ✓ (6.350361 > 6.348030) |
| L2 apply rate > 0 (ouroboros) | ✓ (1.0 bootstrap) |

## Artifacts

- Lock: `.run.lock` → PID 35450 (dead, run finished)
- Log: `run.log`
- Summary: `results/bilevel_improves_trilevel_summary.json`
- Started: 2026-07-17T17:59:35Z
- Finished: 2026-07-17T18:09:58Z
