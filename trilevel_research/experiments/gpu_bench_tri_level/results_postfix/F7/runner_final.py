"""Inner-loop runner: subprocess to HIP gpu_bench binary."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from core.llm_client import LLMClient, parse_json_response
from trilevel_research.domains.gpu_bench_opt.config import GpuBenchConfig
from trilevel_research.domains.gpu_bench_opt.search_config import SearchConfig

logger = logging.getLogger(__name__)

DEFAULT_BIN = Path.home() / "gpu-bench" / "build" / "gpu_bench"

PROPOSE_PROMPT = """\
You optimize a small GPU MLP byte-predictor (HIP on AMD RX 580).
Metric: val_bpb — lower is better.

Current best config:
{config}

Recent trace:
{trace}

Search strategy: {strategy}
Active parameters (you may change ONLY these): {active_params}
Frozen parameters (do NOT change): {frozen_params}
{guidance}

Respond JSON only:
{{"changes": {{"LR": 0.003}}, "hypothesis": "one sentence why"}}
"""


@dataclass
class BenchResult:
    val_bpb: float
    train_bpb: float
    elapsed_s: float
    status: str
    changes: dict = field(default_factory=dict)
    hypothesis: str = ""
    iteration: int = 0
    accepted: bool = False


@dataclass
class BenchTrace:
    results: list[BenchResult] = field(default_factory=list)
    best_val_bpb: float = float("inf")
    best_bpb: float = float("inf")
    best_iteration: int = 0
    best_config: GpuBenchConfig | None = None

    def record(self, result: BenchResult) -> None:
        if result.status == "keep" and result.val_bpb < self.best_bpb:
            self.best_bpb = result.val_bpb
            self.best_val_bpb = result.val_bpb
            self.best_iteration = result.iteration

    def summary(self, last_n: int = 10) -> str:
        lines = [f"Best val_bpb: {self.best_bpb:.6f} (iter {self.best_iteration})"]
        for r in self.results[-last_n:]:
            lines.append(
                f"  iter {r.iteration} [{r.status}] val_bpb={r.val_bpb:.4f} "
                f"changes={r.changes} — {r.hypothesis[:80]}"
            )
        return "\n".join(lines)



class DeduplicateProposals:
    """Helper class for deduplicating LLM-proposed changes in GpuBenchRunner."""

    def __init__(self, trace):
        self.trace = trace

    def deduplicate(self, proposed_changes: dict, iteration: int) -> dict:
        """
        Remove changes that have been tried before with similar or worse results.

        Args:
            proposed_changes: dict of {param: value} from LLM proposal
            iteration: current iteration number for logging

        Returns:
            Filtered dict with duplicates removed, or empty dict if all are duplicates
        """
        if not proposed_changes or not self.trace.results:
            return proposed_changes

        filtered_changes = {}
        rejected_params = []

        for param, value in proposed_changes.items():
            is_duplicate = False
            for result in self.trace.results:
                if result.changes and param in result.changes:
                    existing_value = result.changes[param]
                    try:
                        if float(existing_value) == float(value):
                            is_duplicate = True
                            break
                    except (ValueError, TypeError):
                        if str(existing_value) == str(value):
                            is_duplicate = True
                            break

            if is_duplicate:
                rejected_params.append(param)
            else:
                filtered_changes[param] = value

        if rejected_params:
            import logging
            logging.info(
                f"Iteration {iteration}: Rejected duplicate proposals: "
                f"{', '.join(rejected_params)}"
            )

        return filtered_changes

    def is_all_duplicates(self, proposed_changes: dict, iteration: int) -> bool:
        """Check if all proposed changes are duplicates (returns True if all rejected)."""
        if not proposed_changes or not self.trace.results:
            return False
        deduped = self.deduplicate(proposed_changes, iteration)
        return not deduped



import random
import math
from typing import Any

class PerturbationSurrogate:
    """A local perturbation mechanism for generating proposals near the best configuration."""
    def __init__(self, radius_initial: float = 0.15, radius_min: float = 0.05, step_size_grid: dict | None = None):
        self.radius = radius_initial
        self.radius_min = radius_min
        self.step_size_grid = step_size_grid or {
            "HIDDEN_DIM": 64,
            "NUM_LAYERS": 1,
            "LEARNING_RATE": 0,
        }
        self.active = False

    def propose(self, best_config: Any, active_params: set[str], iteration: int) -> tuple[dict[str, Any], str]:
        changes = {}
        hypothesis_parts = [f"local_perturbation(radius={self.radius:.3f})"]
        for param_name in sorted(active_params):
            param_key = param_name.upper()
            current_val = getattr(best_config, param_key, None)
            if current_val is None:
                continue
            grid = self.step_size_grid.get(param_key, 0)
            if isinstance(current_val, int) and grid > 0:
                noise = random.gauss(0, self.radius * current_val)
                new_val = round((current_val + noise) / grid) * grid
                new_val = max(1, new_val)
                lower = getattr(type(best_config), param_key, None)
                upper = None
                if hasattr(best_config, f'_{param_key}_min'):
                    lower = getattr(best_config, f'_{param_key}_min')
                if hasattr(best_config, f'_{param_key}_max'):
                    upper = getattr(best_config, f'_{param_key}_max')
                if lower is not None:
                    new_val = max(lower, new_val)
                if upper is not None:
                    new_val = min(upper, new_val)
            elif isinstance(current_val, float):
                log_val = math.log(current_val) if current_val > 0 else -10
                noise = random.gauss(0, self.radius)
                new_log = log_val + noise
                new_val = math.exp(new_log)
                new_val = max(1e-6, min(1.0, new_val))
            else:
                continue
            if new_val != current_val:
                changes[param_key] = new_val
                hypothesis_parts.append(f"{param_key}={current_val}->{new_val}")
        if not changes and "HIDDEN_DIM" in active_params:
            step = self.step_size_grid.get("HIDDEN_DIM", 64)
            new_dim = getattr(best_config, "HIDDEN_DIM", 384) + step
            changes["HIDDEN_DIM"] = new_dim
            hypothesis_parts.append(f"HIDDEN_DIM=384->{new_dim}")
        hypothesis = " | ".join(hypothesis_parts)
        return changes, hypothesis

    def adjust_radius(self, result: Any) -> None:
        if result.status == "keep":
            self.radius = min(self.radius * 1.1, 0.30)
        elif result.status == "discard":
            self.radius = max(self.radius * 0.9, self.radius_min)


class GpuBenchRunner:
    def __init__(
        self,
        bench_bin: Path | None = None,
        timeout_s: int = 120,
        llm_client: LLMClient | None = None,
        search_config: SearchConfig | None = None,
        artifacts_dir: Path | None = None,
        simple_mode: bool = True,
    ) -> None:
        env_bin = os.environ.get("GPU_BENCH_BIN")
        self.bench_bin = Path(env_bin) if env_bin else (bench_bin or DEFAULT_BIN)
        self.timeout_s = timeout_s
        self.client = llm_client
        self.search_config = search_config or SearchConfig()
        self.artifacts_dir = artifacts_dir or Path("artifacts/gpu_bench_opt")
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.simple_mode = simple_mode
        self.config = GpuBenchConfig()
        self.trace = BenchTrace()

    def run_baseline(self) -> BenchResult:
        result = self._run_trial(self.config, iteration=0, hypothesis="baseline")
        self.trace.best_val_bpb = result.val_bpb
        self.trace.best_bpb = result.val_bpb
        self.trace.best_iteration = 0
        self.trace.best_config = GpuBenchConfig()
        result.status = "keep"
        result.accepted = True
        self.trace.results.append(result)
        return result

    def run_iteration(
        self,
        iteration: int,
        *,
        changes: dict | None = None,
        hypothesis: str = "",
    ) -> BenchResult:
        """Run one inner iteration. Uses LLM when changes is None."""
        if changes is None:
            if self.client is None:
                raise ValueError("LLM client required when changes not provided")
            changes, hypothesis = self._propose(iteration)

        active = set(self.search_config.active_params)
        filtered = {k: v for k, v in changes.items() if k.upper() in active}
        if not filtered:
            return BenchResult(
                val_bpb=self.trace.best_bpb,
                train_bpb=self.trace.best_bpb,
                elapsed_s=0.0,
                status="discard",
                changes={},
                hypothesis=hypothesis or "no active param changes",
                iteration=iteration,
            )

        trial = self._trial_config(iteration)
        trial.apply_changes(filtered)
        result = self._run_trial(trial, iteration=iteration, hypothesis=hypothesis)
        result.changes = filtered

        if result.status == "crash":
            result.accepted = False
        elif result.val_bpb < self.trace.best_bpb:
            result.status = "keep"
            result.accepted = True
            self.trace.best_val_bpb = result.val_bpb
            self.trace.best_bpb = result.val_bpb
            self.trace.best_iteration = iteration
            self.trace.best_config = trial
            self.config = trial
        else:
            result.status = "discard"
            result.accepted = False

        self.trace.record(result)
        self.trace.results.append(result)
        return result

    def _trial_config(self, iteration: int) -> GpuBenchConfig:
        base = self.trace.best_config or self.config
        return GpuBenchConfig(
            lr=base.lr,
            weight_decay=base.weight_decay,
            batch_size=base.batch_size,
            hidden_dim=base.hidden_dim,
            train_steps=base.train_steps,
            seed=self.config.seed + iteration,
        )

    def _propose(self, iteration: int) -> tuple[dict, str]:
        prompt = PROPOSE_PROMPT.format(
            config=json.dumps(
                (self.trace.best_config or self.config).snapshot(), indent=2
            ),
            trace=self.trace.summary(last_n=5),
            strategy=self.search_config.strategy,
            active_params=", ".join(self.search_config.active_params),
            frozen_params=", ".join(self.search_config.frozen_params) or "(none)",
            guidance=self.search_config.guidance or "",
        )
        resp = self.client.call(prompt, system="You are a hyperparameter researcher.")
        parsed = parse_json_response(resp)
        changes = parsed.get("changes", {}) if isinstance(parsed, dict) else {}
        hypothesis = parsed.get("hypothesis", "") if isinstance(parsed, dict) else ""
        return changes, hypothesis

    def _run_trial(
        self,
        cfg: GpuBenchConfig,
        iteration: int,
        hypothesis: str,
    ) -> BenchResult:
        if not self.bench_bin.is_file():
            raise FileNotFoundError(f"gpu_bench not found: {self.bench_bin}")

        cmd = [str(self.bench_bin), "--json"]
        for k, v in cfg.as_cli_args().items():
            cmd.extend([f"--{k}", str(v)])

        logger.info("Running: %s", " ".join(cmd))
        t0 = time.monotonic()
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.timeout_s,
        )
        wall = time.monotonic() - t0

        if proc.returncode != 0:
            logger.error("gpu_bench failed rc=%s stderr=%s", proc.returncode, proc.stderr)
            return BenchResult(
                val_bpb=99.0,
                train_bpb=99.0,
                elapsed_s=wall,
                status="crash",
                hypothesis=hypothesis,
                iteration=iteration,
            )

        line = proc.stdout.strip().splitlines()[-1]
        payload = json.loads(line)
        return BenchResult(
            val_bpb=float(payload["val_bpb"]),
            train_bpb=float(payload.get("train_bpb", payload["val_bpb"])),
            elapsed_s=float(payload.get("elapsed_s", wall)),
            status=payload.get("status", "ok"),
            hypothesis=hypothesis,
            iteration=iteration,
        )
