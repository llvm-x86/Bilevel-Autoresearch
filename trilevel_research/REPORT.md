# Tri-Level Autoresearch — Ablation Report

**This is the primary artifact for this PR.** Read it before reviewing code.

| Verdict | Status |
|---------|--------|
| **Overall** | **NEGATIVE / INCONCLUSIVE** — no validated LLM/L3 task win; bi-level remains recommended default |
| GPU bench 16-run ablation | **INCONCLUSIVE** — 0% inner-loop L2 apply (`import_fail`); L3 fired, 0% apply; Δval_bpb noise-level |
| `bilevel_improves_trilevel` 4×4 ablation | **MARGINAL / NOT ATTRIBUTABLE** — +0.0123 Δval_bpb via bootstrap schedule, not LLM/L3 |
| `run_iterative` (20% margin target) | **FAIL** — best +9.26% (unreliable); restart −0.30% / −0.49%; driver crashed twice |
| CPU counterfactual (paper fixtures) | **Design-only** — +0.0067 estimated Δval_bpb; not live training evidence |

---

## Executive summary

Tri-level autoresearch adds Level 3 meta-mechanism research on top of the bilevel stack (Groups C/F): tabu registry, adaptive L2/L3 schedule, and validation harness gating Level-2 patches. The extension lives entirely under `trilevel_research/`; upstream `core/` and `domains/train_opt/` are **identical to `main`**.

**Conclusion:** Tri-level does not demonstrate validated task superiority over bi-level. Added complexity (L3 meta-loop, tabu, schedule, iterative optimization) did not produce reliable live gains attributable to LLM-driven or L3 meta-mechanism research. **Recommendation: use bi-level as the default stack; enable tri-level only as optional research.**

No validated LLM/L3 win; one bootstrap-assisted marginal gain (+0.0123 Δval_bpb); iterative 20% target failed (best +9.26%, restart runs negative, driver crashed).

### Evidence (all key experiments)

| Experiment | Δval_bpb / margin | L2 apply | L3 | Verdict |
|------------|-------------------|----------|-----|---------|
| Initial gpu_bench ablation (16 paired runs, AMD RX 580) | C 6.3715 ± 0.0372 vs F 6.3686 ± 0.0422; 4/8 paired wins for F | **0%** inner-loop (100% `import_fail`) | fired, **0% apply** | **INCONCLUSIVE** — mechanism untested; task diff noise-level |
| `bilevel_improves_trilevel` 4×4 paired ablation | F − C = **+0.0123** (C mean 6.3380, F mean 6.3504) | **0%** on all 4×4 ablation repeats; ouroboros schedule L2 **1.0** (bootstrap only); LLM schedule **0/2** | meta-loop did not deliver validated win | **MARGINAL / NOT ATTRIBUTABLE** — bootstrap patch, not LLM/L3 |
| `run_iterative` 20% relative margin | Iter 1 (PID 38608): **+9.26%** (**unreliable** — polluted by C outliers); restart (PID 44049): **−0.30%**, **−0.49%**; target 20% | bootstrap ouroboros each iter | L3 ~7% apply (anchor mismatch) | **FAIL** — driver crashed twice; no `iterative_summary.json` |
| CPU counterfactual (paper Group C fixtures) | estimated **+0.0067** from reduced apply/revert | simulated 83% → 33% apply | 5 L3 fires, 3 tabu blocks | **Design-only** — not live training |

→ Full gpu_bench write-up: [experiments/gpu_bench_tri_level/REPORT.md](experiments/gpu_bench_tri_level/REPORT.md)  
→ Iterative 20% margin: [experiments/bilevel_improves_trilevel/REPORT.md](experiments/bilevel_improves_trilevel/REPORT.md)  
→ CPU simulation detail: [experiments/tri_level_ablation/REPORT.md](experiments/tri_level_ablation/REPORT.md)

---

## Experiment detail

### CPU counterfactual (paper Group C fixtures)

Replay of published Group C L2 session artifacts — no GPU, no LLM.

| Metric | Group C (actual) | Group F (L3 policies simulated) |
|--------|------------------|----------------------------------|
| L2 apply rate | 83% | 33% |
| L2 revert rate | 100% | 67% |
| Tabu blocks | — | 3 |
| L3 fire decisions | — | 5 |
| Estimated Δval_bpb gain | — | +0.0067 (heuristic) |

Counterfactual replay only — supports guardrail *design*, not end-to-end efficacy.

### GPU bench tri-level vs bi-level (initial 16-run ablation)

16-run paired ablation (8× Group C vs 8× Group F) on HIP MLP `gpu_bench` (AMD RX 580).

Every L2 session failed `import_fail` after codegen. L3 engaged on F repeats but could not apply patches. Inner-loop ablation L2 apply rate **0%** both groups; tri-level and bi-level runs were effectively identical at the mechanism layer.

### `bilevel_improves_trilevel` (post-bootstrap 4×4)

After LLM schedule L2 failed 2/2, a **hand-written bootstrap schedule patch** was promoted (ouroboros L2 apply rate 1.0). The subsequent 4×4 paired ablation showed F beating C by +0.0123 Δval_bpb, but inner-loop ablation L2 apply remained **0% on all repeats** — the margin is not evidence of LLM-driven or L3 meta-mechanism superiority.

### `run_iterative` (20% margin target)

Driver crashed twice before completing 8 iterations. Iteration 1 reported **+9.26%** margin (PID 38608) but used bootstrap ouroboros with **unreliable C outliers** — do not treat as validated success. Restart run (PID 44049) showed F trailing C: **−0.30%** then **−0.49%**. No trajectory toward the 20% goal; no `iterative_summary.json` written.

---

## Conclusions & limitations

- **Bi-level is the recommended default.** Tri-level adds tabu, adaptive schedule, and L3 gating without proven live task benefit over bi-level simplicity.
- **Merge only as optional, removable research extension** with documented negative/inconclusive evidence — not for performance gains.
- **Inner-loop ablation L2: 0% apply** throughout gpu_bench ablations; **ouroboros schedule L2: bootstrap only** (LLM 0/2).
- **CPU simulation** supports L3 policy *design* but does not re-run training or measure live val_bpb gains.
- Full live Group F on Karpathy `train.py` (RTX 5090 + DeepSeek) remains future work.

---

## How to reproduce

```bash
pip install -e ".[trilevel,dev]"

# CPU counterfactual (no GPU/API)
python -m trilevel_research.experiments.tri_level_ablation.simulate_from_fixtures --write-report

# GPU bench ablation (needs gpu_bench binary + API key)
python -m trilevel_research.experiments.gpu_bench_tri_level.run_ablation --group all --repeats 2

# Extension tests
pytest trilevel_research/tests/ -v
```

---

## Code map (secondary — see REPORT first)

| Component | Path |
|-----------|------|
| L3 controller (train) | `trilevel_research/domains/train_opt/tri_level_controller.py` |
| L2 + tabu subclass | `trilevel_research/domains/train_opt/l2_mechanism_research.py` |
| Tabu / schedule / harness | `trilevel_research/core/` |
| GPU bench domain | `trilevel_research/domains/gpu_bench_opt/` |
| Config surface | `trilevel_research/config.py` |
| Meta / uninstall | [META.md](./META.md) · [UNINSTALL.md](./UNINSTALL.md) |
