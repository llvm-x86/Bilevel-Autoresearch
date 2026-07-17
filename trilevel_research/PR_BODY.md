<!-- PR title: Optional tri-level research extension (negative / inconclusive results) -->

**Read the report first:** [trilevel_research/REPORT.md](trilevel_research/REPORT.md)

This PR adds Level 3 meta-mechanism research (tabu registry, adaptive L2/L3 schedule, validation harness) as an **isolated optional extension** under `trilevel_research/`. Upstream `core/` and `domains/train_opt/` are identical to `main`; tri-level code lives only under the extension. [REPORT.md](trilevel_research/REPORT.md) is updated to match the evidence below.

**Verdict:** No validated LLM/L3 task win. One bootstrap-assisted marginal gain (+0.0123 Δval_bpb); iterative 20% target failed (best +9.26%, restart runs negative, driver crashed). Do not merge expecting performance gains — merge for the isolated scaffold and honest negative/inconclusive record.

### Evidence

| Experiment | Result |
|------------|--------|
| Initial 16-run gpu_bench ablation | **INCONCLUSIVE** — 0% inner-loop L2 apply (100% `import_fail`); L3 fired, 0% apply; C vs F Δval_bpb noise-level (6.3715 ± 0.0372 vs 6.3686 ± 0.0422; 4/8 paired wins) |
| `bilevel_improves_trilevel` 4×4 paired ablation | **MARGINAL / NOT ATTRIBUTABLE** — F beat C by **+0.0123** Δval_bpb via hand-written bootstrap schedule patch; LLM schedule L2 **0/2** failed; inner-loop ablation L2 **0% apply on all 4×4 repeats**; ouroboros schedule L2 apply **1.0** (bootstrap only) |
| `run_iterative` (20% margin target) | **FAIL** — best **+9.26%** (iter 1, PID 38608; **unreliable** — polluted by C outliers); restart margins **−0.30%** / **−0.49%**; driver crashed twice; no `iterative_summary.json` |
| CPU counterfactual (paper Group C fixtures) | **Design-only** — estimated **+0.0067** Δval_bpb from avoided reverts; not live training evidence |

### What's in `trilevel_research/` (removable)

Self-contained directory. Delete per [UNINSTALL.md](trilevel_research/UNINSTALL.md) to revert to pure bilevel.

| Path | Purpose |
|------|---------|
| [REPORT.md](trilevel_research/REPORT.md) | Primary artifact — verdict, experiments, reproduce commands |
| [META.md](trilevel_research/META.md) | Process notes, ouroboros hook |
| [README.md](trilevel_research/README.md) | Install, CLI, architecture |
| `core/` | Tabu, adaptive schedule, validation harness, session trace |
| `domains/train_opt/`, `domains/gpu_bench_opt/` | Tri-level controllers and mechanism researchers |
| `experiments/tri_level_ablation/` | CPU fixture replay |
| `experiments/gpu_bench_tri_level/` | gpu_bench paired ablation (16 runs) |
| `experiments/bilevel_improves_trilevel/` | Bootstrap ablation + iterative 20% margin driver |
| `tests/` | Extension unit/smoke tests (126 tests) |
| `cli.py`, `config.py` | Entry points and config |

### Upstream changes (minimal)

| File | Change |
|------|--------|
| [README.md](README.md) | One paragraph + link to `trilevel_research/REPORT.md` |
| [pyproject.toml](pyproject.toml) | Optional extra `[trilevel]` |
| `core/`, `domains/train_opt/` | Identical to `main` (tri-level code re-homed under extension) |
| `experiments/ablations/tri_level_ablation/` | Removed (relocated under `trilevel_research/`) |
| `tests/test_level3_*`, `tests/test_mechanism_*` | Removed (relocated under `trilevel_research/tests/`) |

### Why not merge for performance

- Inner-loop ablation L2: **0% apply** throughout gpu_bench ablations; ouroboros schedule L2 is **bootstrap only** (LLM 0/2).
- No validated LLM/L3 win; the +0.0123 margin traces to a hand-written bootstrap patch, not meta-mechanism superiority.
- Iterative 20% target failed; restart runs showed F trailing C; driver crashed before completion.

### Why merge anyway

- **Isolated scaffold:** Entire extension under `trilevel_research/`; upstream paths match `main`; uninstall documented.
- **Test coverage:** 126 extension + 110 upstream tests pass; CPU counterfactual reproducible without GPU/API.
- **Honest negative/inconclusive record:** Documents what was tried and why bi-level remains the recommended default.

### Test plan

Required items pass; optional gpu smoke skipped/blocked (no `GPU_BENCH_BIN` locally; ASUS blocked on `DEEPSEEK_API_KEY`).

- [x] `pip install -e ".[trilevel,dev]"` succeeds from repo root
- [x] `pytest trilevel_research/tests/ -v` — 126 passed
- [x] `python -m trilevel_research.experiments.tri_level_ablation.simulate_from_fixtures --write-report` — CPU counterfactual matches REPORT
- [x] Import smoke: `from trilevel_research.domains.train_opt.tri_level_controller import TriLevelController`
- [x] `ruff check trilevel_research/` clean
- [x] Upstream CI unchanged: `pytest tests/` — 110 passed, no tri-level imports
- [x] `git diff main -- core/ domains/train_opt/` — empty (upstream identical to `main`)
- [ ] (Optional) gpu_bench ablation smoke — SKIPPED locally; BLOCKED on ASUS (`DEEPSEEK_API_KEY` unset)
