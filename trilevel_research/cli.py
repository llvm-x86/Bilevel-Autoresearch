"""Top-level CLI router for the tri-level extension."""
from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Tri-level autoresearch extension (optional Level 3)",
    )
    sub = parser.add_subparsers(dest="domain", required=True)

    train = sub.add_parser("train", help="Karpathy train.py domain (Group F)")
    train_sub = train.add_subparsers(dest="cmd", required=True)
    train_tri = train_sub.add_parser("trilevel", help="Run tri-level train experiment")
    train_tri.set_defaults(_handler="train_trilevel")

    gpu = sub.add_parser("gpu-bench", help="GPU bench domain experiment")
    gpu_sub = gpu.add_subparsers(dest="cmd", required=True)
    gpu_tri = gpu_sub.add_parser("trilevel", help="Run tri-level gpu_bench experiment")
    gpu_tri.set_defaults(_handler="gpu_trilevel")

    ablation = sub.add_parser("ablation", help="Offline ablation drivers")
    ablation_sub = ablation.add_subparsers(dest="cmd", required=True)
    ablation_sim = ablation_sub.add_parser("simulate", help="CPU counterfactual from Group C fixtures")
    ablation_sim.add_argument("--write-report", action="store_true")
    ablation_sim.set_defaults(_handler="ablation_simulate")
    ablation_gpu = ablation_sub.add_parser("gpu-bench", help="GPU bench C vs F ablation")
    ablation_gpu.add_argument("--group", default="all")
    ablation_gpu.add_argument("--repeats", type=int, default=2)
    ablation_gpu.set_defaults(_handler="ablation_gpu_bench")

    args, remainder = parser.parse_known_args(argv)
    handler = getattr(args, "_handler", None)

    if handler == "train_trilevel":
        from trilevel_research.domains.train_opt.cli import main as train_main

        train_main(["trilevel", *remainder])
    elif handler == "gpu_trilevel":
        from trilevel_research.domains.gpu_bench_opt.cli import main as gpu_main

        gpu_main(["trilevel", *remainder])
    elif handler == "ablation_simulate":
        import sys

        from trilevel_research.experiments.tri_level_ablation.simulate_from_fixtures import (
            main as sim_main,
        )

        argv = ["simulate_from_fixtures"]
        if args.write_report:
            argv.append("--write-report")
        argv.extend(remainder)
        old_argv = sys.argv
        sys.argv = argv
        try:
            sim_main()
        finally:
            sys.argv = old_argv
    elif handler == "ablation_gpu_bench":
        import sys

        from trilevel_research.experiments.gpu_bench_tri_level.run_ablation import (
            main as gpu_ablation_main,
        )

        argv = [
            "run_ablation",
            "--group",
            args.group,
            "--repeats",
            str(args.repeats),
            *remainder,
        ]
        old_argv = sys.argv
        sys.argv = argv
        try:
            gpu_ablation_main()
        finally:
            sys.argv = old_argv
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
