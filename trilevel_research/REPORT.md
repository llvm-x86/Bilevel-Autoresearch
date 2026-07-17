# Tri-Level Autoresearch — Ablation Report

**This is the primary artifact for this PR.** Read it before reviewing code.

| Verdict | Status |
|---------|--------|
| **Overall** | **INCONCLUSIVE** — L2 never applied on live gpu_bench; CPU simulation is suggestive only |
| GPU bench (Group C vs F) | **INCONCLUSIVE** — 100% L2 `import_fail`; L3 fired but 0% apply |
| CPU counterfactual (paper Group C fixtures) | **Suggestive, not measured** — design validation only |

---

## Executive summary

Tri-level autoresearch adds Level 3 meta-mechanism research on top of the bilevel stack (Groups C/F): a tabu registry, adaptive L2/L3 schedule, and validation harness that gate Level-2 mechanism patches. The extension lives entirely under `trilevel_research/`; upstream `core/` and `domains/train_opt/` are **reverted to pre–Level-3 state** (deletions only in the diff).

**What we can say honestly today:**

1. **GPU bench (AMD RX 580, 16 paired runs):** Completed successfully but is **INCONCLUSIVE**. Zero L2 mechanism patches were applied in either group (100% `import_fail`). L3 fired once per F repeat but also failed to apply meta-mechanism patches. Without successful L2/L3 engagement, neither mechanistic nor task-level hypotheses could be tested. Mean Δval_bpb was 6.3715 ± 0.0372 (C) vs 6.3686 ± 0.0422 (F); paired wins for F were 4/8 (need ≥6/8 for significance).

2. **CPU counterfactual (paper Group C fixtures):** Offline replay suggests tri-level policies *would* cut L2 apply rate from 83% → 33% and revert rate from 100% → 67%, with 3 tabu blocks and 5 L3 escalations. A conservative heuristic estimates **+0.0067 Δval_bpb** from avoided bad L2 patches — **not measured on live training**. This supports the *design* of L3 guardrails, not end-to-end efficacy.

3. **Blocker for a decisive verdict:** Fix L2 codegen/import validation on the `gpu_bench` domain (and smoke-test L2 apply rate > 0) before claiming mechanistic or task superiority.

→ Full gpu_bench write-up: [experiments/gpu_bench_tri_level/REPORT.md](experiments/gpu_bench_tri_level/REPORT.md)  
→ CPU simulation detail: [experiments/tri_level_ablation/REPORT.md](experiments/tri_level_ablation/REPORT.md)

---

## Experiment 1: CPU counterfactual (paper Group C fixtures)

Replay of published Group C L2 session artifacts — no GPU, no LLM.

| Metric | Group C (actual) | Group F (L3 policies simulated) |
|--------|------------------|----------------------------------|
| L2 apply rate | 83% | 33% |
| L2 revert rate | 100% | 67% |
| Tabu blocks | — | 3 |
| L3 fire decisions | — | 5 |
| Estimated Δval_bpb gain | — | +0.0067 (heuristic) |

**Interpretation:** Tabu + adaptive schedule + harness would have blocked repeat failures and reduced wasteful L2 applies. This is a **counterfactual replay**, not a re-run of training.

---

## Experiment 2: GPU bench tri-level vs bi-level

16-run paired ablation (8× Group C vs 8× Group F) on HIP MLP `gpu_bench` (AMD RX 580).

| Verdict | Result |
|---------|--------|
| Overall | **INCONCLUSIVE** |
| Tier A (Mechanistic) | **FAIL** — 0% L2 apply rate both groups |
| Tier B (Task) | **FAIL** — 4/8 paired wins for F |

**Root cause:** Every L2 session failed `import_fail` after codegen. L3 engaged (1 fire/repeat on F) but could not apply patches either. Tri-level and bi-level runs were effectively identical at the mechanism layer; task metrics reflect L1 hyperparameter search noise only.

---

## Conclusions & limitations

- **Do not merge expecting proven L3 gains.** The PR ships an isolated, test-covered extension plus honest negative/inconclusive evidence.
- **CPU simulation** supports L3 policy *design* (tabu + schedule) but does not re-run training or measure live val_bpb gains.
- **GPU bench** failed to validate the stack end-to-end because generated L2/L3 code did not pass import validation — domain-specific codegen/import paths need hardening.
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
