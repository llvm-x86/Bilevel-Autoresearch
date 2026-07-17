# Tri-Level Research — Meta Notes

Companion to [REPORT.md](./REPORT.md). Process and future-work context for reviewers.

---

## Using bilevel-research to optimize trilevel-research

**The ouroboros idea:** run the *upstream* bilevel stack (Groups C/D from the paper) with an L2 patch target pointed at tri-level code, so the framework that researches itself can research the framework that researches the framework.

### Concrete hook

Use paper **Group C** ablation settings (L1 + L1.5 + L2) with:

| Setting | Value |
|---------|--------|
| L2 patch target | `trilevel_research/core/adaptive_mechanism_schedule.py` |
| L1 task | Ablation hyperparameters (inner budget, outer cycles, tabu tenure, L3 intervals) |
| Domain | `train_opt` (Karpathy `train.py`) or `gpu_bench_opt` once L2 apply is fixed |

Group C's Level 2 already code-generates Python mechanisms. Pointing it at `adaptive_mechanism_schedule.py` asks bilevel autoresearch to *invent better L3 scheduling policies* — tabu escalation thresholds, L2/L3 cadence, revert-rate triggers — instead of hand-tuning them.

### Minimal repro sketch

```bash
pip install -e ".[trilevel,dev]"

# After L2 import_fail is fixed on gpu_bench — or on train_opt with AUTORESEARCH_DIR set:
# Configure mechanism research canonical file to trilevel schedule module, then:
python -m domains.train_opt.cli mechanism \
  --canonical-file trilevel_research/core/adaptive_mechanism_schedule.py \
  --iterations 30 --outer-cycles 6
```

(Full wiring depends on exposing `--canonical-file` or equivalent in the ablation driver; the *intent* is: L2 patches the schedule module while L1 tunes search.)

### Why this is interesting

- **Dogfooding:** Validates bilevel recursion on a real, test-covered extension rather than toy code.
- **Actionable output:** Successful L2 patches would directly improve the tri-level stack under test in [REPORT.md](./REPORT.md).
- **Honest scope:** Only meaningful once gpu_bench L2 `import_fail` is resolved (see REPORT §Experiment 2); until then, CPU fixture replay and unit tests are the evidence base.

### Related docs

- Extension install/usage: [README.md](./README.md)
- Remove extension: [UNINSTALL.md](./UNINSTALL.md)
- Ablation evidence: [REPORT.md](./REPORT.md)
