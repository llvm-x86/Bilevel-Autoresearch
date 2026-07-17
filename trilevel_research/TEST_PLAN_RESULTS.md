# Test Plan Results — `feature/tri-level-autoresearch`

**Date:** 2026-07-17  
**Branch:** `feature/tri-level-autoresearch`  
**Repo:** `/home/a112/Bilevel-Autoresearch`  
**Overall:** **PASS** (all required items pass; optional gpu_bench skipped/blocked)

---

## Checklist

| # | Item | Status |
|---|------|--------|
| 1 | `pip install -e ".[trilevel,dev]"` | [x] PASS |
| 2 | `pytest trilevel_research/tests/ -v` | [x] PASS |
| 3 | CPU counterfactual `simulate_from_fixtures --write-report` | [x] PASS |
| 4 | Import smoke `TriLevelController` | [x] PASS |
| 5 | `ruff check trilevel_research/` | [x] PASS |
| 6 | Upstream `pytest tests/` | [x] PASS |
| 7 | gpu_bench ablation smoke (optional) | [ ] SKIPPED / BLOCKED |
| 8 | `git diff main -- core/ domains/train_opt/` empty (upstream == main) | [x] PASS |

---

## 1. Editable install

```bash
python3 -m pip install -e ".[trilevel,dev]"
```

**Result:** PASS — `Successfully installed bilevel-autoresearch-0.3.0` (venv Python 3.11.14).  
**Note:** System `pip` not on PATH; used `python3 -m pip` from project `.venv`.

---

## 2. Extension tests

```bash
pytest trilevel_research/tests/ -v
```

**Result:** PASS — **126 passed** in 0.55s.

---

## 3. CPU counterfactual simulation

```bash
python -m trilevel_research.experiments.tri_level_ablation.simulate_from_fixtures --write-report
```

**Result:** PASS — summary matches REPORT.md expectations:

- Group C actual: L2 apply 83%, revert 100%
- Group F simulated: L2 apply 33%, revert 67%, tabu blocks 3, L3 fires 5
- `estimated_task_gain_from_l3`: **0.006667** (~**+0.0067** in REPORT)
- Wrote `simulation_results/simulation_summary.json` and `experiments/tri_level_ablation/REPORT.md`

---

## 4. Import smoke

```python
from trilevel_research.domains.train_opt.tri_level_controller import TriLevelController
```

**Result:** PASS — `OK: <class 'trilevel_research.domains.train_opt.tri_level_controller.TriLevelController'>`

---

## 5. Ruff lint

```bash
ruff check trilevel_research/
```

**Result:** PASS — `All checks passed!`

---

## 6. Upstream CI (no tri-level imports)

```bash
pytest tests/ -v
```

**Result:** PASS — **110 passed** in 0.44s. No tri-level imports in upstream test suite.

---

## 7. gpu_bench ablation smoke (optional)

### Local

`GPU_BENCH_BIN` unset → **SKIPPED** (per test plan: skip if no `GPU_BENCH_BIN`).

### ASUS (100.78.97.35 via Tailscale SSH)

```bash
GPU_BENCH_BIN=/home/a112/gpu-bench/build/gpu_bench \
  python3 -m trilevel_research.experiments.gpu_bench_tri_level.run_ablation \
  --group C --repeats 1 --inner-budget 3 --outer-cycles 2
```

**Result:** BLOCKED — module imports and `gpu_bench` binary OK; run failed immediately:

```
OSError: API key not set: DEEPSEEK_API_KEY
```

`trilevel_research/` synced to ASUS; upstream `core/` and `domains/` restored after rsync layout fix. Full live smoke requires `DEEPSEEK_API_KEY` in `/home/a112/Bilevel-Autoresearch/.env` on ASUS.

---

## 8. Upstream path diff (deletions only)

```bash
git diff --stat main...HEAD -- core/ domains/train_opt/
```

**Before fix:** FAIL — `domains/train_opt/mechanism_research.py` had 7 insertions / 6 deletions (unrelated `explore_prompt` refactor from first tri-level commit, not reverted in refactor).

**Fix applied:** Reverted `domains/train_opt/mechanism_research.py` Round-1 explore block to match `main` exactly.

**After fix:**

```bash
git diff main -- core/ domains/train_opt/   # working tree vs main
# (empty — zero net changes in upstream paths)
```

**Result:** PASS — upstream `core/` and `domains/train_opt/` are identical to `main` (net zero diff; tri-level code fully re-homed under `trilevel_research/`).

---

## Fixes applied during test run

1. **`domains/train_opt/mechanism_research.py`** — reverted incidental formatting change so upstream paths match `main` (item 8).

---

## Ready for review?

**Yes** — all **required** items pass. Optional gpu_bench live smoke skipped locally (`GPU_BENCH_BIN` unset) and blocked on ASUS (`DEEPSEEK_API_KEY` unset); not required for merge decision given documented negative/inconclusive live results in REPORT.md.
