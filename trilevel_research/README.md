# Tri-Level Autoresearch (Optional Extension)

> **Start here:** [REPORT.md](./REPORT.md) — ablation results and interpretation.

## What this adds

Level 3: meta-mechanism research that patches Level-2's `mechanism_research.py`.
Built on bilevel Groups C/F; does not modify upstream bilevel code paths.

## Install

```bash
pip install -e ".[trilevel]"    # from repo root
```

## Quick start

```bash
# Karpathy train.py domain (Group F)
python -m trilevel_research train trilevel --inner-budget 5 --outer-cycles 6 --enable-level3

# GPU bench domain (optional experiment)
python -m trilevel_research gpu-bench trilevel --inner-budget 10 --outer-cycles 6 --enable-level3
```

## Reproduce ablations

```bash
python -m trilevel_research.experiments.tri_level_ablation.simulate_from_fixtures --write-report
python -m trilevel_research.experiments.gpu_bench_tri_level.run_ablation --group all --repeats 2
```

## Architecture

```
[L1 inner] → [L1.5 config outer] → [L2 mechanism research] → [L3 meta-mechanism research]
```

Components: tabu registry · adaptive schedule · validation harness · session trace

## Tests

```bash
pytest trilevel_research/tests/ -v
```

## Future work

See [META.md](./META.md) — *Using bilevel-research to optimize trilevel-research* (ouroboros / Group C ablation with L2 target = `adaptive_mechanism_schedule.py`).

## Uninstall

See [UNINSTALL.md](./UNINSTALL.md).
