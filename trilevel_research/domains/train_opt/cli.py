"""Tri-level train_opt CLI — `train trilevel` subcommand."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            _v = _v.strip().strip('"').strip("'")
            os.environ.setdefault(_k.strip(), _v)

from core.llm_client import PROVIDERS, LLMClient
from trilevel_research.config import MechanismResearchConfig
from trilevel_research.domains.train_opt.tri_level_controller import TriLevelController

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

AUTORESEARCH_DIR = Path(
    os.environ.get("AUTORESEARCH_DIR", str(Path.home() / "karpathy_autoresearch"))
)


def get_llm_client(provider: str = "deepseek", model: str = "") -> LLMClient:
    pinfo = PROVIDERS.get(provider)
    if not pinfo:
        sys.exit(f"ERROR: Unknown provider '{provider}'")
    api_key = os.environ.get(pinfo["api_key_env"], "")
    if not api_key:
        sys.exit(f"ERROR: {pinfo['api_key_env']} not set")
    return LLMClient(provider, api_key, model or pinfo["default_model"])


def cmd_trilevel(args: argparse.Namespace) -> None:
    """Tri-level experiment (Level 1 + 1.5 + 2 + optional 3)."""
    client = get_llm_client(args.provider, args.model)

    train_py = AUTORESEARCH_DIR / "train.py"
    if not train_py.exists():
        sys.exit(f"ERROR: train.py not found at {train_py}")

    mech_config = MechanismResearchConfig(
        level2_interval=args.level2_interval,
        level3_interval=args.level3_interval,
        enable_level3=args.enable_level3,
        enable_tabu=not args.no_tabu,
        enable_adaptive_schedule=not args.fixed_schedule,
    )

    artifacts_base = PROJECT_ROOT / "artifacts" / "train_opt" / "trilevel"
    run_dir = artifacts_base / f"run_{args.outer_cycles}c_{args.inner_budget}i"

    controller = TriLevelController(
        run_dir=run_dir,
        train_py=train_py,
        work_dir=AUTORESEARCH_DIR,
        llm_client=client,
        inner_budget=args.inner_budget,
        outer_cycles=args.outer_cycles,
        time_budget=args.time_budget,
        enable_level3=args.enable_level3,
        mech_config=mech_config,
    )

    report = controller.run()
    report_dict = report.to_dict()

    print("\n" + "=" * 60)
    print(f"TRI-LEVEL EXPERIMENT COMPLETE (group {report.group})")
    print("=" * 60)
    print(f"Baseline:      {report_dict.get('baseline_bpb')}")
    best = report_dict.get("best_val_bpb")
    print(
        f"Best:          {best:.6f} (iter {report_dict.get('best_iteration')})"
        if best != float("inf")
        else "Best:          (none)"
    )
    imp = report_dict.get("improvement")
    print(f"Improvement:   {imp:.6f}" if imp is not None else "Improvement:   N/A")
    print(f"Total iters:   {report_dict['total_iterations']}")
    print(f"Outer cycles:  {report_dict['outer_cycles']}")
    print(f"L2 rounds:     {report_dict['level2_rounds']}")
    print(f"L3 rounds:     {report_dict['level3_rounds']}")

    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(report_dict, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport saved to: {report_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Tri-level training optimization")
    parser.add_argument("--provider", default="deepseek", help="LLM provider (default: deepseek)")
    parser.add_argument("--model", default="", help="LLM model")
    parser.add_argument("--time-budget", type=int, default=300, help="Training time per run in seconds")

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_tri = sub.add_parser("trilevel", help="Tri-level experiment (L1 + L1.5 + L2 + L3)")
    p_tri.add_argument("--inner-budget", type=int, default=5, help="Inner iterations per outer cycle")
    p_tri.add_argument("--outer-cycles", type=int, default=6, help="Number of outer cycles")
    p_tri.add_argument("--level2-interval", type=int, default=2, help="Outer cycles per L2 batch")
    p_tri.add_argument("--level3-interval", type=int, default=2, help="Fire L3 every N L2 rounds")
    p_tri.add_argument("--enable-level3", action="store_true", help="Enable Level-3 meta-mechanism research")
    p_tri.add_argument("--no-tabu", action="store_true", help="Disable mechanism tabu registry")
    p_tri.add_argument("--fixed-schedule", action="store_true", help="Use fixed L2 interval only")
    p_tri.set_defaults(func=cmd_trilevel)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
