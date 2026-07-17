# GPU Bench Tri-Level vs Bi-Level Ablation Report

**Date:** 2026-07-17  
**Host:** asus-kiosk (AMD RX 580, gfx803)  
**Protocol:** PROTOCOL.md v1.0  
**Driver:** `run_ablation_parallel.py` (8 workers; C1–C2 from prior sequential run, `--skip-existing`)

---

## Executive Summary

| Verdict | Result |
|---------|--------|
| **Overall** | **INCONCLUSIVE** |
| Tier A (Mechanistic) | **FAIL** |
| Tier B (Task) | **FAIL** |

Tri-level (Group F) does **not** meaningfully improve over bi-level (Group C) on this run. All 16 repeats completed successfully, but **zero L2 mechanism patches were applied** in either group (100% `import_fail`). L3 fired once per F repeat but also failed to apply meta-mechanism patches. Without successful L2/L3 engagement, the primary mechanistic hypothesis could not be tested.

---

## Preflight

| Check | Result |
|-------|--------|
| `gpu_bench` smoke (seed 42) | PASS — val_bpb=6.412, elapsed=0.14s |
| Import smoke (4 modules) | PASS |
| `pytest tests/test_gpu_bench_opt.py` | **9/9 passed** (0.54s) |
| pytest `--cov` | Not run (preflight used `-v` only) |

---

## Run Configuration

| Parameter | Value |
|-----------|-------|
| Groups | C (bi-level) vs F (+L3) |
| Repeats per group | 8 (paired by repeat index) |
| Inner budget | 10 / outer cycle |
| Outer cycles | 6 (60 L1 iterations) |
| level2-interval | 2 |
| level3-interval | 2 |
| Provider | deepseek |
| GPU | HIP_VISIBLE_DEVICES=0 |
| Total runs | 16/16 ok |

---

## Task Outcome (Δval_bpb = baseline_bpb − best_val_bpb)

| Group | Mean ± Std | Median | Best repeat |
|-------|-----------|--------|-------------|
| **C** (bi-level) | 6.3715 ± 0.0372 | 6.3874 | C2 (6.3980) |
| **F** (tri-level) | 6.3686 ± 0.0422 | 6.3940 | F2 (6.4071) |

### Paired Comparison (F − C by repeat index)

| Repeat | C | F | Δ(F−C) | Winner |
|--------|---|---|--------|--------|
| 1 | 6.3910 | 6.3999 | +0.0089 | F |
| 2 | 6.3980 | 6.4071 | +0.0091 | F |
| 3 | 6.3838 | 6.3437 | −0.0401 | C |
| 4 | 6.3837 | 6.3245 | −0.0592 | C |
| 5 | 6.3955 | 6.3974 | +0.0019 | F |
| 6 | 6.3385 | 6.3999 | +0.0614 | F |
| 7 | 6.3964 | 6.3907 | −0.0057 | C |
| 8 | 6.2853 | 6.2853 | +0.0000 | tie |

- **Paired wins (F > C):** 4/8 (need ≥6/8 for Tier B)
- **Mean paired diff:** −0.0030 ± 0.0336
- **Median paired diff:** +0.0009 (need ≥0.005 for Tier B)
- **Wilcoxon signed-rank (one-sided F > C):** W⁺=15.0, **p ≈ 0.433**

---

## L2 Mechanism Efficiency

| Metric | Group C | Group F |
|--------|---------|---------|
| L2 rounds / run | 2 | 2 |
| L2 apply rate | **0%** (0/16 sessions) | **0%** (0/16 sessions) |
| L2 revert rate | 0% (0 applied) | 0% (0 applied) |
| L2 validated rate | 0% | 0% |
| Primary failure mode | `import_fail` (16/16) | `import_fail` (15/16) |
| Tabu blocks | 0 | 0 |

All L2 sessions generated mechanism code that failed import validation. Tabu lists grew (size 1–3) but never blocked a proposal (`blocked_by_tabu=false` on all sessions).

---

## L3 Diagnostics (Group F only)

| Metric | Value |
|--------|-------|
| L3 fires / run | 1 (8/8 repeats) |
| L3 apply rate | **0%** (0/8 sessions applied) |
| L3 failure modes | `import_fail` (5/8), `none` (3/8) |
| Harness blocks | 0 |
| Tabu blocks on L2 | 0 |

L3 engaged (fired every repeat) but could not apply meta-mechanism patches.

---

## Wall Time

| Scope | Value |
|-------|-------|
| Per-run mean (C) | 397s ± 55s (~6.6 min) |
| Per-run mean (F) | 412s ± 18s (~6.9 min) |
| Sequential total (if serial) | C: 3174s + F: 3297s ≈ **107 min** |
| Actual parallel batch | 11:43:30 → 11:57 ≈ **14 min** (8 workers) |
| GPU eval fraction | ≪1% of wall (dominated by LLM latency) |

Note: `wall_time_s` field not populated in report.json (driver gap); times derived from `run.log` timestamps.

---

## Tier A / Tier B Verdict (PROTOCOL §6)

### Tier A — Mechanistic (required for L3 claim)

| Criterion | Required | Observed | Pass? |
|-----------|----------|----------|-------|
| L2 revert rate reduction | median(F) ≤ median(C) − 0.15, p < 0.05 | Both 0%; no reduction testable | **FAIL** |
| L3 engagement | L3 fires ≥1 every F repeat | 1/1 every repeat | PASS |
| Tabu/harness blocks | ≥50% F repeats with blocks | 0% (0/8) | **FAIL** |
| No L2 collapse | apply(F) ≥ apply(C) − 0.20 | 0% vs 0% | PASS (vacuous) |

**Tier A: FAIL** — L3 never successfully modified L2 behavior; mechanistic hypothesis untestable.

### Tier B — Task (meaningful improvement)

| Criterion | Required | Observed | Pass? |
|-----------|----------|----------|-------|
| Absolute gain | median(F−C) ≥ 0.005 | +0.0009 | **FAIL** |
| Win count | F wins ≥ 6/8 | 4/8 | **FAIL** |
| Wilcoxon | p < 0.10 (one-sided) | p ≈ 0.433 | **FAIL** |

**Tier B: FAIL** — No detectable task improvement at n=8.

---

## Interpretation

1. **Mechanism research pipeline broken on gpu_bench:** Every L2 session failed with `import_fail` after code generation. This prevents testing whether L3 reduces L2 revert cycles (the paper's primary claim).
2. **L3 fired but inert:** Meta-mechanism sessions ran but did not apply patches, so tri-level and bi-level runs were effectively identical at the mechanism layer.
3. **Task metrics dominated by L1 search:** Large Δval_bpb (~6.4) reflects inner-loop hyperparameter search, not L2/L3 effects. Paired differences are noise-level (|Δ| < 0.06).
4. **C8 ≡ F8:** Identical improvement (6.285347) suggests shared seed/state artifact in parallel execution.

### Recommended follow-ups

- Fix L2 `import_fail` root cause (validate generated helper classes against `GpuBenchRunner` API).
- Re-run ablation after L2 apply rate > 0 in smoke test.
- Add `wall_time_s` to report.json in driver.
- Consider `--interleave` sequential mode to avoid C8/F8 seed collision.

---

## Artifacts

- Summary: `results/ablation_summary.json`
- Per-run: `results/{C|F}{1-8}/report.json`, `run.log`, `mechanism_sessions/`
- Logs: `parallel_run.log`, `ablation_run.log`
