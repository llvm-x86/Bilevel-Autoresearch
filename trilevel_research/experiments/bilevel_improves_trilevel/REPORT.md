# bilevel_improves_trilevel — 20% margin experiment report

**Date:** 2026-07-17  
**Target:** `(F_mean - C_mean) / C_mean × 100 ≥ 20%`  
**Verdict:** **FAIL** — driver crashed twice before completing 8 iterations; best margin **+9.26%** (iteration 1)

## Run summary

| Item | Value |
|------|-------|
| Driver PID (primary) | **38608** |
| Restart PID (MANAGER rule) | **44049** |
| Iterations completed | 3 logged (1 from run 1, 2 from run 2); run 2 iteration 3 crashed mid-ablation |
| Best margin_pct | **+9.26%** |
| Winning iteration | **1** (driver 38608) |
| Target | 20% |

## Iteration margins

| Iter | Driver | C_mean | F_mean | margin_pct | Ouroboros |
|------|--------|--------|--------|------------|-----------|
| 1 | 38608 | 5.8232 | 6.3625 | **+9.26%** | bootstrap |
| 2 | 38608 | — | — | (crashed mid-ablation) | — |
| 1 | 44049 | 6.3260 | 6.3071 | -0.30% | bootstrap |
| 2 | 44049 | 6.3480 | 6.3171 | -0.49% | bootstrap |
| 3 | 44049 | — | — | (crashed mid-ablation) | bootstrap |

## L3 round counts (Group F vs C)

From last partial ablation (iteration 3, driver 44049):

- **Group C (bi-level):** L3 = 0 on all repeats (expected)
- **Group F (tri-level):** L3 = 2 rounds per completed repeat (C5/C6, F5/F6 incomplete at crash)

L3 meta-patches fired but **almost all failed to apply**:
- Dominant error: `Could not find 'class GpuBenchMechanismResearcher:' in mechanism_research.py`
- One success: `add_l2_failure_logging applied=True`
- ~15 patch attempts observed; apply rate ≈ 7%

## Ouroboros: bootstrap vs LLM

| Source | Result |
|--------|--------|
| LLM L2 schedule patches | **Never validated** (2 attempts × every iteration) |
| Bootstrap schedule patch | **Used every iteration** after LLM failure |

All ouroboros cycles ended with: `Ouroboros LLM L2 did not validate; applying bootstrap schedule patch`.

## Failure modes

1. **Driver crash (no summary written):** Both PIDs died abruptly during parallel ablation (4 workers × 6 repeats), with no traceback in log. Likely external kill or worker pool failure; not OOM (31 GiB RAM, no dmesg OOM).
2. **L3 patch anchor mismatch:** L3 codegen targets `class GpuBenchMechanismResearcher:` but post-L2 `mechanism_research.py` copies use `class GpuBenchMechanismResearcher(` without trailing colon — all meta-patches fail to apply.
3. **L2 apply in ablation:** L2 rounds execute (2 per repeat) but tri-level margin remains below 20%; restart run showed **negative** margins.

## Recommended next fixes

1. **L3 meta patch apply:** Fix anchor string in L3 applier to match actual class declaration (`GpuBenchMechanismResearcher(` or regex), and patch the **canonical** `mechanism_research.py` under `domains/gpu_bench_opt/`, not stale ouroboros copies.
2. **Runner L2 apply:** Ensure importlib validation runs against promoted schedule + live mechanism module; confirm L2 patches persist across ablation worker subprocesses.
3. **Driver resilience:** Write `iterative_summary.json` after each iteration (checkpoint); on crash, resume from last iteration instead of full restart.
4. **Ouroboros LLM L2:** Bootstrap-only schedule is insufficient for 20% margin; tighten L2 validation criteria or increase ouroboros attempts so LLM schedule patches actually validate.
