# bilevel_improves_trilevel — single-run manager

**Do not start a second instance.** Only one driver may run at a time.

## Canonical paths (ASUS)

| Item | Path |
|------|------|
| Driver PID lock | `trilevel_research/experiments/bilevel_improves_trilevel/.run.lock` |
| Log | `trilevel_research/experiments/bilevel_improves_trilevel/run.log` |
| Summary | `trilevel_research/experiments/bilevel_improves_trilevel/results/bilevel_improves_trilevel_summary.json` |
| Report | `trilevel_research/experiments/bilevel_improves_trilevel/REPORT.md` |

## Manager rules

1. **Monitor only** — read `.run.lock` for the canonical PID; poll `run.log`. Never `nohup` a new driver if lock exists and PID is alive.
2. **If PID dead and no summary** — restart once using the command below, overwrite lock.
3. **If summary exists** — write REPORT.md, rsync to local, stop.
4. **Never run two drivers** — no parallel ouroboros or ablation.

## Start command (only when no live lock)

```bash
cd /home/a112/Bilevel-Autoresearch
export GPU_BENCH_BIN=/home/a112/gpu-bench/build/gpu_bench
export PYTHONPATH=/home/a112/Bilevel-Autoresearch HIP_VISIBLE_DEVICES=0
nohup python3 -m trilevel_research.experiments.bilevel_improves_trilevel.run \
  --repeats 4 --workers 4 --inner-budget 5 --outer-cycles 4 --ouroboros-attempts 2 \
  > trilevel_research/experiments/bilevel_improves_trilevel/run.log 2>&1 &
echo $! > trilevel_research/experiments/bilevel_improves_trilevel/.run.lock
```

## Success criteria

- F mean Δval_bpb > C mean + 0.01
- L2 apply rate > 0
