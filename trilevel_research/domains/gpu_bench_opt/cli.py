"""CLI for gpu_bench autoresearch (inner, bilevel, trilevel)."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO))

_env = REPO / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from core.llm_client import PROVIDERS, LLMClient, parse_json_response
from trilevel_research.config import MechanismResearchConfig
from trilevel_research.domains.gpu_bench_opt.outer import GpuBenchOuterLoop
from trilevel_research.domains.gpu_bench_opt.runner import DEFAULT_BIN, GpuBenchRunner
from trilevel_research.domains.gpu_bench_opt.search_config import SearchConfig
from trilevel_research.domains.gpu_bench_opt.tri_level_controller import (
    GpuBenchTriLevelController,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("gpu_bench_opt")

PROPOSE_PROMPT = """\
You optimize a small GPU MLP byte-predictor (HIP on AMD RX 580).
Metric: val_bpb — lower is better.

Current best config:
{config}

Recent trace:
{trace}

Editable params: LR, WEIGHT_DECAY, BATCH_SIZE, HIDDEN_DIM

Respond JSON only:
{{"changes": {{"LR": 0.003}}, "hypothesis": "one sentence why"}}
"""


def get_llm(provider: str = "deepseek", model: str = "") -> LLMClient:
    pinfo = PROVIDERS.get(provider)
    if not pinfo:
        sys.exit(f"ERROR: Unknown provider '{provider}'")
    api_key = os.environ.get(pinfo["api_key_env"], "")
    if not api_key:
        sys.exit(f"ERROR: {pinfo['api_key_env']} not set")
    return LLMClient(provider, api_key, model or pinfo["default_model"])


def _resolve_bench_bin(path: str | None) -> Path:
    if path:
        return Path(path)
    env = os.environ.get("GPU_BENCH_BIN")
    return Path(env) if env else DEFAULT_BIN


def cmd_inner(args: argparse.Namespace) -> None:
    client = get_llm(args.provider, args.model)
    runner = GpuBenchRunner(
        bench_bin=_resolve_bench_bin(args.bench_bin),
        llm_client=client,
        search_config=SearchConfig(inner_budget=args.iterations),
    )

    baseline = runner.run_baseline()
    logger.info("Baseline val_bpb=%.6f (%.2fs)", baseline.val_bpb, baseline.elapsed_s)

    for i in range(1, args.iterations + 1):
        prompt = PROPOSE_PROMPT.format(
            config=json.dumps(runner.config.snapshot(), indent=2),
            trace=runner.trace.summary(last_n=5),
        )
        resp = client.call(prompt, system="You are a hyperparameter researcher.")
        parsed = parse_json_response(resp)
        changes = parsed.get("changes", {}) if isinstance(parsed, dict) else {}
        hypothesis = parsed.get("hypothesis", "") if isinstance(parsed, dict) else ""

        result = runner.run_iteration(i, changes=changes, hypothesis=hypothesis)
        tag = result.status.upper()
        logger.info("[%s] iter=%d val_bpb=%.6f changes=%s", tag, i, result.val_bpb, changes)

    report = {
        "baseline_val_bpb": baseline.val_bpb,
        "best_val_bpb": runner.trace.best_val_bpb,
        "improvement": baseline.val_bpb - runner.trace.best_val_bpb,
        "iterations": args.iterations,
        "best_config": runner.trace.best_config.snapshot() if runner.trace.best_config else {},
    }
    print(json.dumps(report, indent=2))


def cmd_bilevel(args: argparse.Namespace) -> None:
    client = get_llm(args.provider, args.model)
    search_cfg = SearchConfig(inner_budget=args.inner_budget)
    runner = GpuBenchRunner(
        bench_bin=_resolve_bench_bin(args.bench_bin),
        llm_client=client,
        search_config=search_cfg,
        artifacts_dir=REPO / "artifacts" / "gpu_bench_opt" / "bilevel",
    )
    outer = GpuBenchOuterLoop(
        runner=runner,
        llm_client=client,
        max_outer_cycles=args.outer_cycles,
    )
    report = outer.run()

    print("\n" + "=" * 60)
    print("BILEVEL GPU_BENCH COMPLETE (L1 + L1.5)")
    print("=" * 60)
    print(f"Baseline:     {report['baseline_bpb']:.6f}")
    print(f"Best:         {report['best_val_bpb']:.6f} (iter {report['best_iteration']})")
    print(f"Improvement:  {report['improvement']:.6f}")
    print(f"Outer cycles: {report['outer_cycles']}")

    report_path = runner.artifacts_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport saved to: {report_path}")


def cmd_trilevel(args: argparse.Namespace) -> None:
    client = get_llm(args.provider, args.model)
    mech_config = MechanismResearchConfig(
        level2_interval=args.level2_interval,
        level3_interval=args.level3_interval,
        enable_level3=args.enable_level3,
        enable_tabu=not args.no_tabu,
        enable_adaptive_schedule=not args.fixed_schedule,
    )
    run_dir = REPO / "artifacts" / "gpu_bench_opt" / "trilevel" / (
        f"run_{args.outer_cycles}c_{args.inner_budget}i"
        f"_{'F' if args.enable_level3 else 'C'}"
    )
    controller = GpuBenchTriLevelController(
        run_dir=run_dir,
        bench_bin=_resolve_bench_bin(args.bench_bin),
        llm_client=client,
        inner_budget=args.inner_budget,
        outer_cycles=args.outer_cycles,
        enable_level3=args.enable_level3,
        mech_config=mech_config,
        search_config=SearchConfig(inner_budget=args.inner_budget),
        timeout_s=args.timeout,
    )
    report = controller.run()
    report_dict = report.to_dict()

    print("\n" + "=" * 60)
    print(f"TRI-LEVEL GPU_BENCH COMPLETE (group {report.group})")
    print("=" * 60)
    print(f"Baseline:      {report_dict.get('baseline_bpb')}")
    best = report_dict.get("best_val_bpb")
    if best != float("inf"):
        print(f"Best:          {best:.6f} (iter {report_dict.get('best_iteration')})")
    print(f"Improvement:   {report_dict.get('improvement')}")
    print(f"L2 rounds:     {report_dict['level2_rounds']}")
    print(f"L3 rounds:     {report_dict['level3_rounds']}")

    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
    print(f"\nReport saved to: {report_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="GPU bench autoresearch")
    sub = p.add_subparsers(dest="cmd", required=True)

    inner = sub.add_parser("inner", help="Inner loop only (L1)")
    inner.add_argument("--iterations", type=int, default=5)
    inner.add_argument("--provider", default="deepseek")
    inner.add_argument("--model", default="")
    inner.add_argument("--bench-bin", default=None, help="Path to gpu_bench binary")
    inner.set_defaults(func=cmd_inner)

    bilevel = sub.add_parser("bilevel", help="Bilevel experiment (L1 + L1.5)")
    bilevel.add_argument("--inner-budget", type=int, default=5)
    bilevel.add_argument("--outer-cycles", type=int, default=4)
    bilevel.add_argument("--provider", default="deepseek")
    bilevel.add_argument("--model", default="")
    bilevel.add_argument("--bench-bin", default=None)
    bilevel.set_defaults(func=cmd_bilevel)

    tri = sub.add_parser("trilevel", help="Tri-level experiment (L1+L1.5+L2+L3)")
    tri.add_argument("--inner-budget", type=int, default=5)
    tri.add_argument("--outer-cycles", type=int, default=4)
    tri.add_argument("--level2-interval", type=int, default=2)
    tri.add_argument("--level3-interval", type=int, default=2)
    tri.add_argument("--enable-level3", action="store_true")
    tri.add_argument("--no-tabu", action="store_true")
    tri.add_argument("--fixed-schedule", action="store_true")
    tri.add_argument("--provider", default="deepseek")
    tri.add_argument("--model", default="")
    tri.add_argument("--bench-bin", default=None)
    tri.add_argument("--timeout", type=int, default=120)
    tri.set_defaults(func=cmd_trilevel)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
