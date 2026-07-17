# Tri-Level CPU Simulation Report

CPU-only counterfactual replay of paper ablation **Group C** fixtures
(`experiments/ablations/paper_ablation/run2_results/results_C`).

No GPU or LLM calls. Simulates Level 3 policies (tabu registry, adaptive
schedule, validation harness) against historical L2 session artifacts.

**Verified on:** local dev machine + `asus-kiosk` (Tailscale `100.78.97.35`, ethernet
`10.42.0.92`) — 52 unit tests passed, simulation reproduced identically on both.

## Summary

| Metric | Group C (actual) | Group F (L3 policies simulated) |
|--------|------------------|----------------------------------|
| Mean Δval_bpb | 0.0448 ± 0.0296 | (same task trace — meta-layer only) |
| L2 apply rate | 83% | 33% |
| L2 revert rate | 100% | 67% |
| L2 validated rate | 0% | 0% |
| Tabu blocks | — | 3 |
| Harness blocks | — | 0 |
| L3 fire decisions | — | 5 |

## Interpretation

- **L2 revert rate drops** from 100% to 67% — tabu + harness prevent re-applying broken mechanisms.
- **Tabu registry** would have blocked 3 duplicate/failed mechanism proposals across repeats.
- **Adaptive schedule** triggered 5 Level 3 escalations (high L2 revert rate in all C repeats).

## Counterfactual task estimate

- Revert rate reduction: 33%
- Estimated additional Δval_bpb from avoided bad L2 patches: **+0.0067**
- Conservative heuristic: each avoided L2 revert saves ~1 wasted outer batch; 0.02 val_bpb per revert based on C1 batch-3 breakthrough pattern.

## Per-repeat detail

### C1
- Δval_bpb: 0.0654
- C revert rate: 100% → F: 100%
- Tabu blocks: 1, harness blocks: 0, L3 fires: 2

### C2
- Δval_bpb: 0.0109
- C revert rate: 100% → F: 100%
- Tabu blocks: 1, harness blocks: 0, L3 fires: 2

### C3
- Δval_bpb: 0.0580
- C revert rate: 100% → F: 0%
- Tabu blocks: 1, harness blocks: 0, L3 fires: 1

## Limitations

- CPU simulation replays historical traces; does not re-run training.
- Full GPU ablation (Group F live) requires RTX 5090 + DeepSeek API.
- Counterfactual task gain is a heuristic, not measured val_bpb.

## Next steps

```bash
python -m experiments.ablations.tri_level_ablation.run_ablation \
  --group all --repeats 3 --iterations 30 --outer-cycles 6
```