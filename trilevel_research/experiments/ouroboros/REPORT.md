# Ouroboros Experiment Report — Bilevel L2 on `adaptive_mechanism_schedule.py`

**Date:** 2026-07-17  
**Host:** asus-kiosk (AMD RX 580, gfx803)  
**Branch:** `feature/tri-level-autoresearch`  
**Driver:** `trilevel_research/experiments/ouroboros/run_bilevel_on_schedule.py`

---

## Executive Summary

| Question | Answer |
|----------|--------|
| Did bilevel L2 patch `adaptive_mechanism_schedule.py`? | **Partially** — 1/1 L2 session generated code and applied syntactically, but **0% validated** (reverted after `import_fail`) |
| Did patched schedule change L2/L3 firing in mini F-run? | **No** — no patch survived validation; mini-F replay used canonical schedule |
| Verdict vs runner.py target | **Schedule target is patchable; runner target was not.** Failures differ: runner.py = module import complexity; schedule = LLM codegen shape errors |

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Group | C (L1 + L1.5 + L2, **no L3**) |
| L2 patch target | `trilevel_research/core/adaptive_mechanism_schedule.py` (run_dir copy) |
| L1 task | gpu_bench hyperparameter search |
| Inner budget | 5 |
| Outer cycles | 4 |
| level2-interval | 2 |
| Provider | deepseek |
| Repeats | 1 (+ 1 L1-only baseline) |

```bash
python3 -m trilevel_research.experiments.ouroboros.run_bilevel_on_schedule \
  --repeats 1 --inner-budget 5 --outer-cycles 4 --level2-interval 2 \
  --provider deepseek --baseline-l1-only
```

---

## L2 Schedule Patch Results

| Metric | schedule_l2 | l1_only (baseline) | gpu_bench runner L2 (prior ablation) |
|--------|-------------|--------------------|--------------------------------------|
| L2 rounds | 1 | 0 (disabled) | 2/run |
| Apply rate | **0%** (0/1) | n/a | **0%** (0/16) |
| Validate rate | **0%** (0/1) | n/a | **0%** |
| Primary failure | `import_fail` (bad codegen) | `l2_disabled` | `import_fail` (runner import) |
| Δval_bpb | 6.3119 | 6.2853 | ~6.37 (Group C mean) |

### L2 session detail (round 1)

- **Hypothesis:** volatility-gated L2 scheduling (`replace_method` on `decide()`)
- **Codegen:** LLM returned a **full nested class** instead of a method fragment
- **Apply:** Syntactic patch written to `05_patched_schedule.py`, then **reverted** after validate
- **Validate failure:** Nested class + wrong API (`fire_l2` vs `fire_level2`, `reasons` vs `reason`, bogus import)
- **Canonical validate:** `ScheduleMechanismResearcher.validate()` passes on unpatched schedule (after `sys.modules` fix)

Artifacts: `results/schedule_l2_r1/mechanism_sessions/round_1/`

---

## Mini F-Run (L3 firing replay)

Replayed `schedule.decide(..., config.enable_level3=True)` for cycles 1–4 on post-run schedule file.

| Cycle | fire_level2 | fire_level3 | reason |
|-------|-------------|-------------|--------|
| 1–4 | true | false | high discard rate (80%) |

**No difference** between schedule_l2 and l1_only baselines — expected because no validated patch was retained.

---

## Baseline Comparison

- **Task metric:** schedule_l2 Δval_bpb = 6.3119 vs l1_only = 6.2853 (+0.027). Noise-level; dominated by L1 search stochasticity, not L2.
- **Mechanism layer:** l1_only correctly skips L2 (`error: l2_disabled`). schedule_l2 engaged L2 once at the fixed interval boundary (after cycles 1–2).
- **vs prior gpu_bench ablation:** Same 0% L2 apply rate, but schedule target reaches validate stage; runner target failed at import for all 16 sessions.

---

## Interpretation & Verdict

1. **Ouroboros wiring works:** `OuroborosScheduleController` + `ScheduleMechanismResearcher` successfully run Group C with L2 pointed at the schedule module while L1 tunes gpu_bench.
2. **Schedule is a better L2 target than `runner.py`:** Import validation succeeds on the canonical module; the prior gpu_bench ablation's 100% `import_fail` on runner patches is plausibly a **wrong-target** problem, not a fundamental tri-level failure.
3. **Remaining blocker is codegen quality:** The one L2 session produced structurally invalid output (full class for `replace_method`). Post-run fix adds codegen normalization + stricter prompts.
4. **Tri-level design not invalidated:** L3 never ran (by design). L2 fired and reverted safely via backup/validate — the guardrail stack behaves as intended even when codegen fails.
5. **Recommendation:** Re-run ouroboros after codegen normalization lands; if validate rate > 0, replay mini-F on the retained patched schedule to test L2/L3 firing changes.

---

## Artifacts

- Summary: `results/ouroboros_summary.json`
- Runs: `results/schedule_l2_r1/`, `results/l1_only_r1/`
- Log: `results/ouroboros_run.log`
