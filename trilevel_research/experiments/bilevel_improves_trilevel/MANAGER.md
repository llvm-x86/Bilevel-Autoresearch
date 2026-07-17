# bilevel_improves_trilevel — single-run manager

**Do not start a second instance.** Only one driver may run at a time.

## Canonical paths (ASUS)

| Item | Path |
|------|------|
| Driver PID lock | `trilevel_research/experiments/bilevel_improves_trilevel/.run.lock` |
| Log | `trilevel_research/experiments/bilevel_improves_trilevel/run.log` |
| Iterative summary | `trilevel_research/experiments/bilevel_improves_trilevel/results/iterative_summary.json` |
| Report | `trilevel_research/experiments/bilevel_improves_trilevel/REPORT.md` |

## Manager rules

1. **Monitor only** — read `.run.lock` for the canonical PID; poll `run.log`. Never start a new driver if lock exists and PID is alive.
2. **If PID dead and no summary** — restart once using the command below, overwrite lock.
3. **If `iterative_summary.json` exists with `success: true`** — write REPORT.md, rsync to local, stop.
4. **Never run two drivers** — no parallel ouroboros or ablation.

## Start command — 20% target (only when no live lock)

```bash
cd /home/a112/Bilevel-Autoresearch
export GPU_BENCH_BIN=/home/a112/gpu-bench/build/gpu_bench
export PYTHONPATH=/home/a112/Bilevel-Autoresearch HIP_VISIBLE_DEVICES=0
nohup python3 -m trilevel_research.experiments.bilevel_improves_trilevel.run_iterative \
  --target-margin-pct 20 --max-iterations 8 --repeats 6 --workers 4 \
  --outer-cycles 6 --level3-interval 1 \
  > trilevel_research/experiments/bilevel_improves_trilevel/run.log 2>&1 &
echo $! > trilevel_research/experiments/bilevel_improves_trilevel/.run.lock
```

## Success criteria

- `(F_mean - C_mean) / C_mean * 100 >= 20`
- Ouroboros schedule L2 apply rate > 0 (LLM or bootstrap)
- Group F runs with L3 enabled; Group C is bi-level only

## Stack fixes in this phase

- Runner L2 validate uses importlib (not broken subprocess import)
- L3 fires after every L2 round in Group F (and on L2 failure)
- Schedule triggers L3 on first consecutive L2 failure
- Iterative bilevel re-runs ouroboros + ablation until 20% margin or max iterations
