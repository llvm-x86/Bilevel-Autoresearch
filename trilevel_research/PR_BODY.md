# Tri-Level Autoresearch (optional extension)

**Read the report first:** [trilevel_research/REPORT.md](trilevel_research/REPORT.md)

This PR adds Level 3 meta-mechanism research (tabu registry, adaptive L2/L3 schedule, validation harness) as an **isolated optional extension** under `trilevel_research/`. Upstream bilevel code is unchanged in spirit: tri-level modules that previously lived in `core/` and `domains/train_opt/` are **removed from upstream** and re-homed under the extension. Evidence from a 16-run gpu_bench ablation is **INCONCLUSIVE** — every L2 session failed with `import_fail`, so mechanistic and task hypotheses could not be tested. A CPU counterfactual replay of paper Group C fixtures is **suggestive only** (estimated +0.0067 Δval_bpb from avoided reverts, not live measurement). Do not merge expecting proven L3 gains; merge for the isolated, test-covered scaffold and honest negative result.

## What's in `trilevel_research/` (removable)

The entire directory is self-contained. Delete it (see [UNINSTALL.md](trilevel_research/UNINSTALL.md)) to revert to pure bilevel with no Level-3 surface area.

| Path | Purpose |
|------|---------|
| [REPORT.md](trilevel_research/REPORT.md) | **Primary artifact** — verdict, experiments, reproduce commands |
| [META.md](trilevel_research/META.md) | Process notes, ouroboros hook (bilevel → optimize trilevel) |
| [README.md](trilevel_research/README.md) | Install, CLI, architecture |
| `core/` | Tabu, adaptive schedule, validation harness, session trace |
| `domains/train_opt/` | Tri-level controller, L2/L3 mechanism researchers |
| `domains/gpu_bench_opt/` | GPU bench domain + tri-level controller |
| `experiments/tri_level_ablation/` | CPU fixture replay + live ablation driver |
| `experiments/gpu_bench_tri_level/` | gpu_bench paired ablation (16 runs, INCONCLUSIVE) |
| `tests/` | Extension unit/smoke tests |
| `cli.py`, `config.py` | Entry points and config |

## Upstream changes (minimal)

| File | Change |
|------|--------|
| [README.md](README.md) | One paragraph + link to `trilevel_research/REPORT.md` |
| [pyproject.toml](pyproject.toml) | Optional extra `[trilevel]`; package find includes `trilevel_research*` |
| `core/`, `domains/train_opt/` | **Deletions only** — Level-3 code reverted/moved to extension |
| `experiments/ablations/tri_level_ablation/` | **Removed** (relocated under `trilevel_research/`) |
| `tests/test_level3_*`, `tests/test_mechanism_*` | **Removed** (relocated under `trilevel_research/tests/`) |

Verified diff stat on upstream paths: **~1962 deletions, 0 additions** in `core/` + `domains/train_opt/`.

## Test plan

- [ ] `pip install -e ".[trilevel,dev]"` succeeds from repo root
- [ ] `pytest trilevel_research/tests/ -v` — all extension tests pass
- [ ] `python -m trilevel_research.experiments.tri_level_ablation.simulate_from_fixtures --write-report` — CPU counterfactual reproduces summary in REPORT
- [ ] Import smoke: `from trilevel_research.domains.train_opt.tri_level_controller import TriLevelController` (no GPU)
- [ ] `ruff check trilevel_research/` clean (or document exceptions)
- [ ] Upstream CI unchanged: `pytest tests/` still passes without tri-level imports
- [ ] (Optional, needs GPU + API) gpu_bench ablation smoke: `--group C --repeats 1` — expect L2 `import_fail` until blocker fixed
- [ ] Confirm `git diff --stat core/ domains/train_opt/` shows deletions only

## Verdict

**INCONCLUSIVE until L2 apply is fixed on gpu_bench.**

- **GPU bench:** 16/16 runs completed; 0% L2 apply rate (100% `import_fail`); L3 fired but 0% apply. Tier A and Tier B both FAIL. Task Δval_bpb differences are noise-level.
- **CPU simulation:** Supports L3 *policy design* (tabu blocks, schedule escalations, lower revert rate in replay) but is not live training evidence.
- **Next blocker:** Harden L2 codegen/import validation for `gpu_bench_opt`, re-run ablation with L2 apply rate > 0, then revisit mechanistic claims.

Meta follow-up (not in scope for decisive verdict): [META.md — Using bilevel-research to optimize trilevel-research](trilevel_research/META.md).
