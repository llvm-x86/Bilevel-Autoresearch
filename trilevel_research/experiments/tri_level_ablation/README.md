# Tri-Level Ablation (CPU simulation + live driver)

Compares **Group C** (L1 + L1.5 + L2) against **Group F** (L1 + L1.5 + L2 + L3).

Level 3 meta-optimizes Level 2's `mechanism_research.py` process using:

- **Mechanism tabu registry** — blocks repeated failed L2 proposals
- **Adaptive scheduling** — decides when to fire L2/L3 based on inner trace
- **Validation harness** — import-checks L3 patches before activation

**Primary results:** [../../REPORT.md](../../REPORT.md) · CPU detail: [REPORT.md](./REPORT.md)

## Groups

| Group | Levels | L3 |
|-------|--------|-----|
| C | L1 + L1.5 + L2 | off |
| F | L1 + L1.5 + L2 + L3 | on |

Shared settings (match paper ablation):

- 30 inner iterations, 6 outer cycles
- L2 every 2 outer cycles (adaptive schedule may override)
- L3 every 2 L2 rounds (Group F only)
- 300s training budget per run
- `DEPTH` and `ASPECT_RATIO` frozen

## Commands

```bash
cd Bilevel-Autoresearch
pip install -e ".[trilevel]"

export AUTORESEARCH_DIR="$HOME/karpathy_autoresearch"

# CPU counterfactual from paper Group C fixtures (no GPU/API)
python -m trilevel_research.experiments.tri_level_ablation.simulate_from_fixtures --write-report

# Group C only (3 repeats) — live, needs GPU + API
python -m trilevel_research.experiments.tri_level_ablation.run_ablation \
  --group C --repeats 3 --iterations 30 --outer-cycles 6

# Group F only (tri-level)
python -m trilevel_research.experiments.tri_level_ablation.run_ablation \
  --group F --repeats 3 --iterations 30 --outer-cycles 6

# Both groups
python -m trilevel_research.experiments.tri_level_ablation.run_ablation --group all --repeats 3

# CLI tri-level sanity run
python -m trilevel_research train trilevel \
  --inner-budget 5 --outer-cycles 4 --enable-level3
```

## Results

Reports are written to `trilevel_research/experiments/tri_level_ablation/results/{C|F}{repeat}/report.json`.

Each run directory contains:

- `runner.py` / `runner_final.py` — L2-patched inner loop
- `mechanism_research.py` — L3-patched Level-2 researcher (Group F)
- `mechanism_sessions/round_N/` — L2 session artifacts
- `meta_mechanism_sessions/round_M/` — L3 session artifacts (Group F)
- `mechanism_tabu.json` — tabu registry state

Simulation output: `trilevel_research/experiments/tri_level_ablation/simulation_results/simulation_summary.json`

## Metrics

### Primary (task outcome)

- **Δval_bpb** — `baseline_bpb - best_val_bpb`
- **best_val_bpb**, **time-to-best**

### L2 efficiency (L3 claim)

- **L2 apply rate** — patches applied / L2 rounds
- **L2 revert rate** — import-fail reverts
- **Unique mechanisms** — distinct applied mechanism names

### L3 diagnostics (Group F)

- **L3 apply rate**
- **tabu_stats**, **schedule_stats** in report.json

## Import smoke test (no GPU)

```bash
python -c "
from trilevel_research.domains.train_opt.tri_level_controller import TriLevelController
from trilevel_research.domains.train_opt.meta_mechanism_research import TrainMetaMechanismResearcher
from trilevel_research.core.mechanism_tabu_registry import MechanismTabuRegistry
print('OK')
"
```
