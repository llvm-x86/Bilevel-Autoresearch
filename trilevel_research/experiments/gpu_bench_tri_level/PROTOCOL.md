# gpu_bench Tri-Level vs Bi-Level Ablation Protocol

**Goal:** Full paired comparison of **Group C** (L1 + L1.5 + L2) vs **Group F** (same + L3) on the HIP MLP `gpu_bench` domain (AMD RX 580 on ASUS), with equivalent statistical rigor to the train_opt paper ablation but scaled for ~0.14 s/eval.

**Status:** Protocol v1.0 — driver: `trilevel_research/experiments/gpu_bench_tri_level/run_ablation.py` (mirrors `trilevel_research/experiments/tri_level_ablation/run_ablation.py` with `GpuBenchTriLevelController`).

---

## 1. Background & Hypothesis

| Layer | Role on gpu_bench |
|-------|-------------------|
| **L1** | Inner loop: LLM proposes LR / WEIGHT_DECAY / BATCH_SIZE / HIDDEN_DIM → `GpuBenchRunner` eval |
| **L1.5** | Outer loop: `GpuBenchOuterLoop` freezes/unfreezes params, strategy, guidance |
| **L2** | `GpuBenchMechanismResearcher` patches `runner.py` every 2 outer cycles |
| **L3** (F only) | `GpuBenchMetaMechanismResearcher` patches `mechanism_research.py`; tabu + adaptive schedule + validation harness |

**Primary hypothesis (mechanistic):** L3 reduces wasteful L2 apply/revert cycles without harming task performance.

**Secondary hypothesis (task):** L3 yields higher Δval_bpb than bi-level alone, via better L2 search mechanisms and scheduling.

**Reference (train_opt paper ablation):** 30 inner iterations, 6 outer cycles, 3 repeats, 300 s training budget/eval. Group C mean Δval_bpb = 0.0448 ± 0.030 (n=3); L2 revert rate = 100%.

---

## 2. Hardware & Environment (ASUS / RX 580)

### Prerequisites

```bash
# ROCm / HIP visible to gpu_bench
export HIP_VISIBLE_DEVICES=0
export GPU_BENCH_BIN="$HOME/gpu-bench/build/gpu_bench"

# DeepSeek (all levels share one client)
export DEEPSEEK_API_KEY="..."   # or via Bilevel-Autoresearch/.env

cd ~/Bilevel-Autoresearch
pip install -e ".[trilevel]"
```

### Preflight (required before ablation)

```bash
# 1. Binary smoke test
$GPU_BENCH_BIN --json --lr 0.003 --batch-size 64 --hidden-dim 256 --train-steps 150 --seed 42
# Expect: JSON line with val_bpb, elapsed_s ≈ 0.12–0.18 s on RX 580

# 2. Import smoke (no GPU)
python -c "
from trilevel_research.domains.gpu_bench_opt.runner import GpuBenchRunner
from trilevel_research.domains.gpu_bench_opt.outer import GpuBenchOuterLoop
from trilevel_research.domains.gpu_bench_opt.mechanism_research import GpuBenchMechanismResearcher
from trilevel_research.domains.gpu_bench_opt.meta_mechanism_research import GpuBenchMetaMechanismResearcher
print('OK')
"

# 3. Pilot baseline variance (Phase 0, below)
python -m trilevel_research.experiments.gpu_bench_tri_level.run_ablation --pilot-baseline --repeats 5
```

---

## 3. Statistical Power & Scale Settings

### 3.1 Why gpu_bench allows more repeats (not more GPU time)

| Quantity | train_opt (paper) | gpu_bench (this protocol) |
|----------|-------------------|---------------------------|
| GPU/eval wall | ~300 s | ~0.14 s (~**2140×** faster) |
| Bottleneck | GPU training | **LLM latency** (~3–8 s/call) |
| Paper repeats | 3 | **8 paired** (recommended) |
| Inner iterations | 30 | **60** (2× decision depth) |
| Outer cycles | 6 | **6** (unchanged — preserves L2 cadence) |
| L2 rounds / run | 2 | **2** (interval=2 cycles) |
| L3 rounds / run (F) | ~2 | **~2** (interval=2 L2 rounds) |

GPU time is negligible (~9 s/run for 61 evals). Statistical power is limited by **repeat count** and **LLM stochasticity**, not eval throughput. We trade saved GPU hours into **more paired repeats** and **2× inner iterations** while keeping the same outer/L2/L3 schedule structure as the paper.

### 3.2 Power analysis (paired C vs F)

**Endpoints:**

1. **Mechanistic (primary):** L2 revert rate — paper simulation showed C=100%, F=67% (Δ=33 pp).
2. **Task (secondary):** paired Δval_bpb difference δ = median(F) − median(C).

**Assumed effect sizes (conservative, from train_opt fixtures):**

| Endpoint | Expected δ | Expected σ_diff (paired) | n for 80% power (α=0.05) |
|----------|------------|--------------------------|----------------------------|
| L2 revert rate | −0.25 to −0.33 | 0.15 | **5–6** repeats |
| Δval_bpb (task) | +0.006 to +0.015 | 0.012–0.020 | **8–12** repeats |

**Recommended design:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Repeats per group** | **8** (paired by repeat index) | 80% power on mechanistic endpoint; ~80% on task if δ≥0.012 |
| **Extended option** | 12 repeats | Use if Phase 0 pilot σ_bpb > 0.015 |
| **Inner iterations** | **60** | 10/cycle; doubles L1 decisions vs paper |
| **Outer cycles** | **6** | Matches paper; yields exactly **2 L2 rounds** |
| **Randomization** | Repeat *r* uses `base_seed=42+r` for gpu_bench seeds | Same seed policy for C*r* and F*r* |

**Significance tests (pre-registered):**

- **Mechanistic:** Wilcoxon signed-rank on paired revert rates (F vs C); one-sided H₁: revert(F) < revert(C).
- **Task:** Wilcoxon signed-rank on paired Δval_bpb (F − C); one-sided H₁: Δ(F) > Δ(C).
- **Multiplicity:** Holm correction across the two primary families (mechanistic + task); exploratory metrics uncorrected.
- **Minimum n for reporting:** ≥6 successful paired repeats per group (else report descriptive only).

### 3.3 Phase 0 — Pilot (optional but recommended)

Run 5 baseline-only evals + 5 single-group C mini-runs (15 iters, 3 cycles) to estimate:

- `σ_bpb` (baseline val_bpb noise)
- Typical Δval_bpb range per 60-iter run
- Actual mean `elapsed_s` on RX 580

If pilot `σ_bpb > 0.02`, increase repeats to **12** or inner iterations to **90**.

---

## 4. Exact Group Configurations

Groups C and F are **identical except L3**. Both use `GpuBenchTriLevelController` (same driver, same tabu + adaptive schedule on L2 path).

### 4.1 Shared settings

```yaml
domain: gpu_bench_opt
provider: deepseek
model: ""                    # provider default (deepseek-chat)
iterations: 60
outer_cycles: 6
inner_per_cycle: 10          # iterations // outer_cycles
eval_timeout_s: 120
forbidden_params: [TRAIN_STEPS]   # frozen at 150 (analogous to DEPTH/ASPECT_RATIO)
editable_params: [LR, WEIGHT_DECAY, BATCH_SIZE, HIDDEN_DIM]
seed_policy: "base_seed + iteration"   # GpuBenchConfig.seed + iteration
canonical_runner: trilevel_research/domains/gpu_bench_opt/runner.py
canonical_mech_research: trilevel_research/domains/gpu_bench_opt/mechanism_research.py
validation_harness_domain: gpu_bench
```

### 4.2 Group C — Bi-level (L1 + L1.5 + L2)

```python
MechanismResearchConfig(
    level2_interval=2,
    level3_interval=2,          # ignored when L3 off
    enable_level3=False,
    enable_tabu=True,
    enable_adaptive_schedule=True,
    max_code_retries=3,
    tabu_max_size=20,
    tabu_default_tenure=3,
    validation_strict=True,
)

GpuBenchTriLevelController(
    enable_level3=False,
    inner_budget=10,
    outer_cycles=6,
    forbidden_params={"TRAIN_STEPS"},
    mech_config=<above>,
)
```

**Expected per run:** 6 outer cycles → 2 L2 rounds (after cycles 2 and 4), 0 L3 rounds.

### 4.3 Group F — Tri-level (+ L3)

```python
MechanismResearchConfig(
    level2_interval=2,
    level3_interval=2,
    enable_level3=True,
    enable_tabu=True,
    enable_adaptive_schedule=True,
    max_code_retries=3,
    tabu_max_size=20,
    tabu_default_tenure=3,
    validation_strict=True,
)

GpuBenchTriLevelController(
    enable_level3=True,
    inner_budget=10,
    outer_cycles=6,
    forbidden_params={"TRAIN_STEPS"},
    mech_config=<above>,
)
```

**Expected per run:** 2 L2 rounds, **1–2 L3 fires** (every 2 L2 rounds + adaptive escalation on high revert rate).

### 4.4 Isolation checklist (each repeat)

1. Reset `runner.py` / `mechanism_research.py` from canonical copies.
2. Delete prior run artifacts under `results/{C|F}{repeat}/`.
3. Use same `base_seed = 42 + repeat` for C and F.
4. Do not share tabu/schedule JSON across groups (per-run `mechanism_tabu.json`, `schedule_config.json`).

---

## 5. Primary Metrics

All metrics extracted from `results/{C|F}{repeat}/report.json` plus session artifacts.

### 5.1 Task outcome

| Metric | Definition | Direction |
|--------|------------|-----------|
| **Δval_bpb** | `baseline_bpb − best_val_bpb` | Higher = better |
| **best_val_bpb** | Best inner-loop metric | Lower = better |
| **time_to_best** | Wall seconds from run start to iteration `best_iteration` | Lower = better (exploratory) |

### 5.2 L2 efficiency (L3 mechanistic claim)

| Metric | Definition |
|--------|------------|
| **L2 apply rate** | `#sessions applied=True / level2_rounds` |
| **L2 revert rate** | `#sessions applied=True ∧ validated=False / #applied` |
| **L2 validated rate** | `#sessions validated=True / level2_rounds` |
| **Tabu blocks** | `tabu_stats.blocked` or count of `blocked_by_tabu` in sessions |
| **Unique mechanisms** | Distinct `mechanism_name` among applied sessions |

### 5.3 L3 diagnostics (Group F only)

| Metric | Definition |
|--------|------------|
| **L3 fires** | `level3_rounds` |
| **L3 apply rate** | `#L3 sessions applied=True / level3_rounds` |
| **Schedule overrides** | `schedule_stats` + `schedule_decisions` in report |
| **Harness blocks** | L3 patches rejected by validation harness |

### 5.4 Cost

| Metric | Definition |
|--------|------------|
| **Wall time** | `report.wall_time_s` (driver must log monotonic start/end) |
| **GPU eval time** | Sum of `elapsed_s` from trace (expect ≪ wall time) |
| **LLM calls** | Count from experiment.log (inner + outer + L2 + L3) |

---

## 6. Success Criteria

### 6.1 Tier A — Mechanistic success (required for L3 claim)

Group **F** wins Tier A if **all** hold (n ≥ 6 paired repeats):

1. **L2 revert rate:** median(F) ≤ median(C) − **0.15** (15 pp absolute reduction), AND Wilcoxon one-sided **p < 0.05**.
2. **L3 engagement:** every F repeat has **L3 fires ≥ 1** and **≥ 50%** of F repeats have tabu or harness blocks on at least one L2 proposal.
3. **No regression guard:** L2 apply rate(F) ≥ L2 apply rate(C) − 0.20 (L3 may block bad applies — not a collapse of L2 activity).

### 6.2 Tier B — Task success (meaningful improvement)

Group **F** wins Tier B if **any** of:

1. **Absolute gain:** median Δval_bpb(F) − median Δval_bpb(C) ≥ **0.005** val_bpb (~0.5% on typical baseline ~1.0 bpb), AND F wins **≥ 6/8** paired repeats on Δval_bpb, AND Wilcoxon one-sided **p < 0.10** (exploratory α; report exact p).
2. **Relative gain:** paired mean improvement ratio ≥ **1.15×** (15% relative) with same win-count rule.

**Interpretation:** Tier A alone supports the paper’s L3 efficiency narrative (matches train_opt simulation). Tier B is required to claim end-to-end task superiority on gpu_bench.

### 6.3 Failure / inconclusive

- **LLM outage:** >2 failed repeats per group → trigger fallback (Section 8).
- **GPU failure:** any repeat with >10% eval crashes → discard repeat, rerun.
- **Inconclusive task, positive mechanistic:** Report as “L3 improves L2 hygiene; task gain not detected at n=8”.

---

## 7. Execution Plan

### 7.1 Run order

1. **Phase 0:** Pilot baseline (5×) — ~2 min.
2. **Phase 1:** Group C, repeats 1–8 — ~80 min.
3. **Phase 2:** Group F, repeats 1–8 — ~100 min (L3 overhead).
4. **Phase 3:** Aggregate + stats script → `RESULTS.md`.

Alternate: interleave repeats (`C1→F1→C2→F2…`) to control for API rate limits / thermal drift.

### 7.2 Recommended ASUS command

```bash
cd ~/Bilevel-Autoresearch
export HIP_VISIBLE_DEVICES=0
export GPU_BENCH_BIN="$HOME/gpu-bench/build/gpu_bench"
export DEEPSEEK_API_KEY="$(grep DEEPSEEK_API_KEY .env | cut -d= -f2)"

# Full paired ablation (8 repeats × 2 groups)
python -m trilevel_research.experiments.gpu_bench_tri_level.run_ablation \
  --group all \
  --repeats 8 \
  --iterations 60 \
  --outer-cycles 6 \
  --level3-interval 2 \
  --provider deepseek \
  --interleave \
  --results-dir trilevel_research/experiments/gpu_bench_tri_level/results
```

**Single-group debugging:**

```bash
python -m trilevel_research.experiments.gpu_bench_tri_level.run_ablation \
  --group C --repeats 1 --iterations 30 --outer-cycles 6
```

### 7.3 Wall-time estimate (RX 580 + DeepSeek)

Per-repeat breakdown (60 iters, 6 outer, 2 L2):

| Component | Count | Est. time |
|-----------|-------|-----------|
| GPU evals | 61 × 0.14 s | ~9 s |
| L1 LLM + overhead | 60 × ~4 s | ~240 s |
| L1.5 outer LLM | 6 × ~5 s | ~30 s |
| L2 sessions | 2 × ~60 s | ~120 s |
| L3 sessions (F only) | 1–2 × ~60 s | ~60–120 s |

| Run scope | Estimated wall |
|-----------|----------------|
| 1× Group C repeat | **~7–9 min** |
| 1× Group F repeat | **~9–12 min** |
| **Full study (8 C + 8 F)** | **~2.5–3.5 h** |
| Extended (12 + 12) | **~4–5 h** |

GPU compute is <2% of total wall time; schedule assumes stable DeepSeek API (~4 s/call). Add ~15 min for Phase 0 pilot and aggregation.

### 7.4 Outputs

```
trilevel_research/experiments/gpu_bench_tri_level/
├── PROTOCOL.md
├── run_ablation.py
├── run_ablation_parallel.py
├── aggregate_results.py
├── simulate_from_fixtures.py      # CPU fallback
└── results/
    ├── C1/ … C8/
    │   ├── report.json
    │   ├── experiment.log
    │   ├── runner_final.py
    │   ├── mechanism_sessions/round_*/
    │   └── meta_mechanism_sessions/   (F only)
    ├── F1/ … F8/
    ├── summary.json
    └── RESULTS.md
```

---

## 8. Fallback — CPU-Only Fixture Simulation

If DeepSeek API fails, rate-limits, or ASUS GPU is unavailable, run counterfactual replay **without LLM or GPU** (same pattern as `tri_level_ablation/simulate_from_fixtures.py`).

### 8.1 Trigger conditions

- `DEEPSEEK_API_KEY` missing or 3 consecutive LLM hard failures.
- `gpu_bench` binary missing / HIP error on preflight.
- User explicitly requests dry-run validation of L3 policies.

### 8.2 Procedure

**Option A — Replay completed C fixtures (preferred after partial live run):**

```bash
python -m trilevel_research.experiments.gpu_bench_tri_level.simulate_from_fixtures \
  --fixture-root trilevel_research/experiments/gpu_bench_tri_level/results \
  --groups C \
  --write-report
```

Replays actual L2 session artifacts from live Group C runs; applies tabu + adaptive schedule + harness policies to estimate Group F counterfactual L2 metrics. **Does not re-measure val_bpb** — task gain uses the train_opt-derived heuristic (0.02 val_bpb per avoided revert) unless gpu_bench traces are available.

**Option B — Bootstrap from train_opt paper fixtures (no live data yet):**

```bash
python -m trilevel_research.experiments.tri_level_ablation.simulate_from_fixtures \
  --write-report
```

Sanity-checks the simulation pipeline; documents expected L3 policy behavior (revert ↓, tabu blocks ↑). Label results **“train_opt proxy — not gpu_bench task metrics”** in `RESULTS.md`.

**Option C — Mock LLM harness (CI / import validation):**

```bash
pytest trilevel_research/tests/test_level3_smoke.py -v
python -c "
from trilevel_research.core.mechanism_validation_harness import MechanismValidationHarness, MockLLMClient
from trilevel_research.domains.gpu_bench_opt.mechanism_research import GpuBenchMechanismResearcher
h = MechanismValidationHarness(domain='gpu_bench')
# dry-run with mocked responses
print('harness OK')
"
```

### 8.3 Fallback success criteria (relaxed)

Report **simulated** L2 revert rate reduction ≥ 15 pp and L3 fires ≥ 1 per repeat. Mark task Δval_bpb as **counterfactual estimate only** — do not claim live gpu_bench task superiority.

---

## 9. Analysis Script (post-run)

```bash
python -m trilevel_research.experiments.gpu_bench_tri_level.aggregate_results \
  --results-dir trilevel_research/experiments/gpu_bench_tri_level/results \
  --alpha 0.05 \
  --output trilevel_research/experiments/gpu_bench_tri_level/results/RESULTS.md
```

Must emit:

- Per-group mean ± std Δval_bpb
- Paired F−C differences with Wilcoxon p-values
- L2 apply/revert/validated rates
- L3 fire counts, tabu/harness blocks
- Total wall time and GPU eval time
- Tier A / Tier B pass/fail against Section 6

---

## 10. Comparison to train_opt Paper Settings

| Setting | Paper (train_opt) | This protocol (gpu_bench) |
|---------|-------------------|---------------------------|
| Inner iterations | 30 | **60** |
| Outer cycles | 6 | 6 |
| Repeats / group | 3 | **8** |
| Eval cost | ~300 s | ~0.14 s |
| Frozen capacity param | DEPTH, ASPECT_RATIO | **TRAIN_STEPS** |
| L2 interval | 2 cycles | 2 cycles |
| L3 interval | 2 L2 rounds | 2 L2 rounds |
| Tabu + adaptive | yes (F) | yes (F) |
| Controller | TriLevelController | GpuBenchTriLevelController |
| Primary claim | L2 mechanism research | **L3 meta-mechanism over L2** |

---

## 11. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| LLM stochasticity dominates variance | 8 paired repeats; interleaved C/F |
| L2 patches break imports | Validation + revert (same as paper); tabu in F |
| val_bpb noise from short MLP train | 150 train_steps fixed; 60 inner decisions |
| API rate limits | `--interleave`; exponential backoff in driver |
| RX 580 thermal throttle | 0.14 s/eval unlikely to throttle; monitor `elapsed_s` drift |
| Small absolute Δval_bpb | Pre-register mechanistic Tier A as primary |

---

## Appendix A — report.json schema (required fields)

```json
{
  "group": "C",
  "repeat": 1,
  "status": "ok",
  "baseline_bpb": 1.012345,
  "best_val_bpb": 0.998765,
  "improvement": 0.013580,
  "total_iterations": 61,
  "outer_cycles": 6,
  "level2_rounds": 2,
  "level3_rounds": 0,
  "level2_sessions": [{"round": 1, "applied": true, "validated": false, ...}],
  "level3_sessions": [],
  "tabu_stats": {},
  "schedule_stats": {},
  "wall_time_s": 520.3,
  "gpu_eval_time_s": 8.7,
  "trace": [{"iter": 0, "bpb": 1.01, "status": "keep", ...}],
  "outer_trace": [...]
}
```

---

## Appendix B — Quick reference

**Recommended production run:** `--group all --repeats 8 --iterations 60 --outer-cycles 6`  
**Estimated wall on ASUS:** **~3 hours** (8 paired C+F repeats)  
**Primary readout:** L2 revert rate (Tier A) + Δval_bpb (Tier B)  
**Fallback:** `simulate_from_fixtures.py` (CPU, no LLM/GPU)
