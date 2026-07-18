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



@dataclass
class BatchAnchorState:
    nominal_batch: int = 16
    anchor_count: int = 3
    cooldown: int = 5
    active: bool = False
    remaining_anchors: int = 3
    cooldown_counter: int = 0
    original_batch: int | None = None


class BatchAnchorManager:
    def __init__(self, nominal_batch: int = 16, anchor_count: int = 3, cooldown: int = 5):
        self.nominal_batch = nominal_batch
        self.anchor_count = anchor_count
        self.cooldown = cooldown
        self.state = BatchAnchorState(
            nominal_batch=nominal_batch,
            anchor_count=anchor_count,
            cooldown=cooldown,
        )

    def should_anchor(self, iteration: int, trace: "BenchTrace") -> bool:
        if self.state.active:
            return True
        if self.state.cooldown_counter > 0:
            self.state.cooldown_counter -= 1
            return False
        recent = [r for r in trace.results if r.status in ("keep", "discard")][-5:]
        if len(recent) < 5:
            return False
        lr_only_changes = sum(
            1
            for r in recent
            if r.changes
            and set(r.changes.keys()) == {"LEARNING_RATE"}
            and r.status == "discard"
        )
        return lr_only_changes >= 3

    def anchor(self, current_config: "GpuBenchConfig") -> None:
        self.state.active = True
        self.state.remaining_anchors = self.anchor_count
        self.state.original_batch = current_config.BATCH_SIZE
        current_config.BATCH_SIZE = self.nominal_batch

    def release(self, current_config: "GpuBenchConfig") -> None:
        if self.state.original_batch is not None:
            current_config.BATCH_SIZE = self.state.original_batch
        self.state.active = False
        self.state.cooldown_counter = self.cooldown
        self.state.original_batch = None

    def on_iteration_end(self, result: "BenchResult", current_config: "GpuBenchConfig") -> None:
        if not self.state.active:
            return
        if result.status == "keep":
            self.release(current_config)
            return
        self.state.remaining_anchors -= 1
        if self.state.remaining_anchors <= 0:
            self.release(current_config)

    def reset(self) -> None:
        self.state = BatchAnchorState(
            nominal_batch=self.nominal_batch,
            anchor_count=self.anchor_count,
            cooldown=self.cooldown,
        )



import random
import logging
from copy import deepcopy as deepest_copy
from typing import Optional

class StagnationRestartController:
    def __init__(
        self,
        window_size: int = 5,
        stagnation_threshold_bpb: float = 0.001,
        top_k: int = 3,
        perturbation_std: float = 0.01,
        restart_cooldown: int = 3,
    ) -> None:
        self.window_size = window_size
        self.stagnation_threshold_bpb = stagnation_threshold_bpb
        self.top_k = top_k
        self.perturbation_std = perturbation_std
        self.restart_cooldown = restart_cooldown

        self.best_bpb_history: list[float] = []
        self.best_configs: list[tuple[float, 'GpuBenchConfig']] = []
        self.restart_count: int = 0
        self.last_restart_iteration: int = -restart_cooldown
        self.stagnant_iterations: int = 0
        self.restart_active: bool = False

    def update(self, iteration: int, config: 'GpuBenchConfig', val_bpb: float) -> bool:
        if not self.best_bpb_history or val_bpb < self.best_bpb_history[-1]:
            self.best_bpb_history.append(val_bpb)
        else:
            self.best_bpb_history.append(self.best_bpb_history[-1])

        if len(self.best_bpb_history) > self.window_size:
            self.best_bpb_history.pop(0)

        self.best_configs.append((val_bpb, config))
        self.best_configs.sort(key=lambda x: x[0])
        self.best_configs = self.best_configs[:self.top_k]

        if iteration - self.last_restart_iteration < self.restart_cooldown:
            return False

        if len(self.best_bpb_history) < self.window_size:
            return False

        improvement = self.best_bpb_history[-1] - self.best_bpb_history[0]
        if improvement < self.stagnation_threshold_bpb:
            self.stagnant_iterations += 1
            if self.stagnant_iterations >= 2:
                self.restart_count += 1
                self.last_restart_iteration = iteration
                self.stagnant_iterations = 0
                return True
        else:
            self.stagnant_iterations = 0

        return False

    def generate_restart_config(self) -> 'GpuBenchConfig':
        if not self.best_configs:
            from domains.gpu_bench_opt.runner import GpuBenchConfig
            return GpuBenchConfig()

        best_val, best_config = self.best_configs[0]
        restart_config = deepest_copy(best_config)

        for param_name, param_value in vars(restart_config).items():
            if isinstance(param_value, float):
                noise = random.gauss(0, self.perturbation_std)
                new_value = param_value * (1.0 + noise)
                setattr(restart_config, param_name, new_value)

        return restart_config


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
