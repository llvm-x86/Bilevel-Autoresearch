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



class SensitivityClock:
    """Tracks parameter sensitivity and enforces exploration fairness."""

    def __init__(self, active_params: set[str]):
        self._param_trials: dict[str, dict[float, list[float]]] = {
            p.upper(): {} for p in active_params
        }
        self._variation_count: dict[str, int] = {
            p.upper(): 0 for p in active_params
        }
        self._locked: dict[str, bool] = {
            p.upper(): False for p in active_params
        }
        self._min_variations = 5
        self._min_distinct_values = 3
        self._sensitivity_ratio = 0.02

    def record_trial(self, param: str, value: float, val_bpb: float) -> None:
        p = param.upper()
        if p not in self._param_trials:
            return
        if value not in self._param_trials[p]:
            self._param_trials[p][value] = []
        self._param_trials[p][value].append(val_bpb)
        self._variation_count[p] += 1

    def record_variation(self, param: str) -> None:
        p = param.upper()
        if p in self._variation_count:
            self._variation_count[p] += 1

    def is_sensitive(self, param: str, best_bpb: float) -> bool | None:
        p = param.upper()
        if p not in self._param_trials:
            return None
        values_tried = len(self._param_trials[p])
        if values_tried < self._min_distinct_values:
            return None
        if self._variation_count[p] < self._min_variations:
            return None
        value_means = []
        for v, results in self._param_trials[p].items():
            if results:
                value_means.append(sum(results) / len(results))
        if len(value_means) < 2:
            return None
        median_mean = sorted(value_means)[len(value_means) // 2]
        for mean in value_means:
            improvement_ratio = (median_mean - mean) / median_mean
            if improvement_ratio > 0.10:
                return True
        range_bpb = max(value_means) - min(value_means)
        threshold = best_bpb * self._sensitivity_ratio
        return range_bpb > threshold

    def lock_insensitive_params(self, best_bpb: float) -> list[str]:
        newly_locked = []
        for p in list(self._locked.keys()):
            if self._locked[p]:
                continue
            sensitivity = self.is_sensitive(p, best_bpb)
            if sensitivity is False:
                self._locked[p] = True
                newly_locked.append(p)
        return newly_locked

    def is_locked(self, param: str) -> bool:
        return self._locked.get(param.upper(), False)

    def count_variations(self, param: str) -> int:
        return self._variation_count.get(param.upper(), 0)



class ConditionalWarmupReset:
    """Helper class for divergence detection and warmup reset logic."""

    def __init__(self, warmup_steps: int = 50, divergence_threshold_multiplier: float = 3.0,
                 recent_window_size: int = 5, max_consecutive_divergences: int = 3):
        self.warmup_steps = warmup_steps
        self.divergence_threshold_multiplier = divergence_threshold_multiplier
        self.recent_window_size = recent_window_size
        self.max_consecutive_divergences = max_consecutive_divergences
        self.recent_val_bpb_window: list[float] = []
        self.consecutive_divergences = 0

    def detect_divergence(self, val_bpb: float) -> bool:
        if val_bpb == float('inf') or val_bpb != val_bpb:
            return True
        if not self.recent_val_bpb_window:
            return False
        recent = self.recent_val_bpb_window[-self.recent_window_size:]
        moving_avg = sum(recent) / len(recent)
        threshold = moving_avg * self.divergence_threshold_multiplier
        return val_bpb > threshold

    def try_warmup_reset(self, runner, trial_config, iteration: int, hypothesis: str) -> object:
        if self.consecutive_divergences >= self.max_consecutive_divergences:
            return None
        modified = GpuBenchConfig()
        modified.learning_rate = getattr(trial_config, 'learning_rate', 0.001) * 0.1
        if hasattr(modified, 'warmup_steps'):
            modified.warmup_steps = self.warmup_steps
        if hasattr(modified, 'training_steps'):
            modified.training_steps = getattr(trial_config, 'training_steps', 100) + self.warmup_steps
        return runner._run_trial(modified, iteration=iteration, hypothesis=f"{hypothesis}_warmup")

    def record_success(self, val_bpb: float):
        self.consecutive_divergences = 0
        self.recent_val_bpb_window.append(val_bpb)
        if len(self.recent_val_bpb_window) > self.recent_window_size:
            self.recent_val_bpb_window.pop(0)

    def record_divergence(self):
        self.consecutive_divergences += 1

    def reset_window(self, val_bpb: float):
        self.recent_val_bpb_window = [val_bpb]


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
