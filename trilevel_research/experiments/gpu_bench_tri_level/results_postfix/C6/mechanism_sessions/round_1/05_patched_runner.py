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



class ResampleDiverseMechanism:
    """Detects plateau and generates diverse configs for exploration."""

    def __init__(self, runner, window: int = 5, threshold: float = 0.01,
                 frequency: int = 3):
        self.runner = runner
        self.window = window
        self.threshold = threshold
        self.frequency = frequency
        self.last_resample_iter = -self.frequency

    def detect_plateau(self, iteration: int) -> bool:
        trace = self.runner.trace
        if len(trace.results) < self.window:
            return False
        recent = trace.results[-self.window:]
        if not recent:
            return False
        best_recent = min(r.val_bpb for r in recent if r.status in ("keep", "discard"))
        best_ever = trace.best_bpb
        if abs(best_recent - best_ever) / max(best_ever, 0.01) > self.threshold:
            return False
        last_configs = [r.config for r in recent[-3:] if hasattr(r, 'config')]
        if len(last_configs) < 3:
            return False
        lr_vals = [c.learning_rate for c in last_configs]
        hd_vals = [c.hidden_dim for c in last_configs]
        lr_narrow = (max(lr_vals) - min(lr_vals)) < 0.0005
        hd_narrow = (max(hd_vals) - min(hd_vals)) < 32 if hd_vals else False
        return lr_narrow and hd_narrow

    def resample_if_stuck(self, iteration: int):
        if not self.runner.search_config.resample_enabled:
            return None
        if iteration - self.last_resample_iter < self.frequency:
            return None
        if not self.detect_plateau(iteration):
            return None
        config = self.runner.config
        search_config = self.runner.search_config
        changes = {}
        parts = ["Resample: forced diverse exploration"]
        # Flip learning rate
        current_lr = config.learning_rate
        new_lr = 0.01 if current_lr < 0.005 else 0.001
        changes["learning_rate"] = new_lr
        parts.append(f"LR={new_lr}")
        # Flip hidden dim
        current_hd = config.hidden_dim
        new_hd = 128 if current_hd >= 256 else 512
        changes["hidden_dim"] = new_hd
        parts.append(f"HIDDEN_DIM={new_hd}")
        # Toggle batch size if active
        if hasattr(search_config, 'active_params') and 'BATCH_SIZE' in search_config.active_params:
            current_bs = config.batch_size
            new_bs = 64 if current_bs <= 32 else 16
            changes["batch_size"] = new_bs
            parts.append(f"BATCH_SIZE={new_bs}")
        # Toggle optimizer if active
        if hasattr(search_config, 'active_params') and 'OPTIMIZER' in search_config.active_params:
            current_opt = config.optimizer
            new_opt = "adam" if current_opt == "sgd" else "sgd"
            changes["optimizer"] = new_opt
            parts.append(f"OPTIM={new_opt}")
        hypothesis = " | ".join(parts)
        self.last_resample_iter = iteration
        return changes, hypothesis


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
