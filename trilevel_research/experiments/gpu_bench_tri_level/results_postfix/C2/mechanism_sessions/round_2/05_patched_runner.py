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
        class ConfigDedupParams:
            def __init__(self, min_hamming_distance: int = 2,
                         param_rounding: dict[str, float] | None = None,
                         jitter_std: float = 0.05) -> None:
                self.min_hamming_distance = min_hamming_distance
                self.param_rounding = param_rounding or {
                    'LR': 0.001,
                    'WEIGHT_DECAY': 1e-6,
                    'BATCH_SIZE': 8,
                    'HIDDEN_DIM': 128,
                }
                self.jitter_std = jitter_std


        class ConfigDedupValidator:
            def __init__(self, params: ConfigDedupParams | None = None) -> None:
                self.params = params or ConfigDedupParams()
                self._seen_configs: list[dict[str, float]] = []

            def normalize_config(self, config: dict) -> dict:
                normalized: dict[str, float] = {}
                for key, value in config.items():
                    rounding = self.params.param_rounding.get(key, 1.0)
                    if rounding != 0:
                        normalized[key] = round(value / rounding) * rounding
                    else:
                        normalized[key] = value
                # Return sorted by key for deterministic comparison
                return dict(sorted(normalized.items()))

            def is_novel(self, config: dict) -> tuple[bool, dict]:
                norm = self.normalize_config(config)
                for seen in self._seen_configs:
                    distance = 0
                    for key in set(norm.keys()) | set(seen.keys()):
                        val_norm = norm.get(key, 0.0)
                        val_seen = seen.get(key, 0.0)
                        threshold = self.params.param_rounding.get(key, 1.0)
                        if abs(val_norm - val_seen) > threshold:
                            distance += 1
                    if distance < self.params.min_hamming_distance:
                        return (False, {})
                self._seen_configs.append(norm)
                return (True, norm)

            def fallback_jitter(self, best_config: 'GpuBenchConfig') -> dict:
                import random
                import numpy as np
                perturbed: dict[str, float] = {}
                # When best_config is not yet set, use defaults
                if best_config is None:
                    lr = 0.001
                    weight_decay = 0.0
                    batch_size = 64
                    hidden_dim = 128
                else:
                    lr = best_config.lr
                    weight_decay = best_config.weight_decay
                    batch_size = best_config.batch_size
                    hidden_dim = best_config.hidden_dim

                perturbed['LR'] = lr * (1 + np.random.normal(0, self.params.jitter_std))
                perturbed['WEIGHT_DECAY'] = weight_decay * (1 + np.random.normal(0, self.params.jitter_std))
                perturbed['BATCH_SIZE'] = round(batch_size * (1 + np.random.normal(0, self.params.jitter_std)) / 8) * 8
                # HIDDEN_DIM: pick from [128, 256, 384, 512] not equal to the best (if possible)
                choices = [128, 256, 384, 512]
                if hidden_dim in choices:
                    others = [c for c in choices if c != hidden_dim]
                    if others:
                        perturbed['HIDDEN_DIM'] = random.choice(others)
                    else:
                        perturbed['HIDDEN_DIM'] = random.choice(choices)
                else:
                    perturbed['HIDDEN_DIM'] = random.choice(choices)
                return perturbed


        self.dedup_validator = ConfigDedupValidator()
        # ----- Stuck state tracker (restart_with_higher_variance) -----
        self._iteration_since_improvement = 0
        self._best_score = float('-inf')
        self._stuck_threshold = 4
        self._restart_count = 0
        self._last_restart_iteration = -1
        self._restart_applied = False

        # Small helper to encapsulate stuck detection and restart logic
        class _RestartHandler:
            def __init__(self, owner: 'GpuBenchRunner') -> None:
                self.owner = owner
                self.iteration_since_improvement = 0
                self.best_score = float('-inf')
                self.stuck_threshold = 4
                self.restart_count = 0
                self.last_restart_iteration = -1
                self.restart_applied = False

            def handle(self, current_score: float) -> bool:
                if current_score > self.best_score:
                    self.best_score = current_score
                    self.iteration_since_improvement = 0
                    return False
                self.iteration_since_improvement += 1
                if self.iteration_since_improvement >= self.stuck_threshold:
                    it = self.owner._iteration_number  # assume exists
                    if it - self.last_restart_iteration >= 3:
                        self._apply_restart()
                        self.restart_count += 1
                        self.last_restart_iteration = it
                        self.restart_applied = True
                        return True
                return False

            def _apply_restart(self) -> None:
                vm = min(4.0 * (1.25 ** self.restart_count), 20.0)
                cfg = self.owner.config
                # Perturb each parameter with increased variance
                import random, math
                lr = cfg.LR
                new_lr = lr + random.uniform(-0.5 * vm, 0.5 * vm) * lr
                cfg.LR = max(0.0001, min(0.1, new_lr))
                wd = cfg.WEIGHT_DECAY
                new_wd = 10 ** (-6 + random.uniform(0, 2 * vm))
                cfg.WEIGHT_DECAY = max(1e-8, min(0.1, new_wd))
                bs = cfg.BATCH_SIZE
                new_bs = bs + round(random.uniform(-32 * vm, 32 * vm))
                cfg.BATCH_SIZE = max(16, min(512, new_bs))
                hd = cfg.HIDDEN_DIM
                new_hd = hd + round(random.uniform(-64 * vm, 64 * vm))
                cfg.HIDDEN_DIM = max(64, min(1024, new_hd))
                # Reset dedup validator so the new config is not rejected
                if hasattr(self.owner, 'dedup_validator'):
                    self.owner.dedup_validator.reset()
                self.restart_applied = True

        self._restart_handler = _RestartHandler(self)

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
