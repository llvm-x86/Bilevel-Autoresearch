"""Search space for gpu_bench HIP MLP micro-benchmark."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GpuBenchConfig:
    lr: float = 3e-3
    weight_decay: float = 0.0
    batch_size: int = 64
    hidden_dim: int = 256
    train_steps: int = 150
    seed: int = 42

    editable_params: list[str] = field(
        default_factory=lambda: ["LR", "WEIGHT_DECAY", "BATCH_SIZE", "HIDDEN_DIM"]
    )

    def as_cli_args(self) -> dict[str, str | int | float]:
        return {
            "lr": self.lr,
            "weight-decay": self.weight_decay,
            "batch-size": self.batch_size,
            "hidden-dim": self.hidden_dim,
            "train-steps": self.train_steps,
            "seed": self.seed,
        }

    def apply_changes(self, changes: dict) -> None:
        mapping = {
            "LR": "lr",
            "WEIGHT_DECAY": "weight_decay",
            "BATCH_SIZE": "batch_size",
            "HIDDEN_DIM": "hidden_dim",
            "TRAIN_STEPS": "train_steps",
        }
        for key, val in changes.items():
            attr = mapping.get(key.upper())
            if attr is None:
                continue
            if attr in ("batch_size", "hidden_dim", "train_steps", "seed"):
                setattr(self, attr, int(val))
            else:
                setattr(self, attr, float(val))

    def snapshot(self) -> dict:
        return {
            "LR": self.lr,
            "WEIGHT_DECAY": self.weight_decay,
            "BATCH_SIZE": self.batch_size,
            "HIDDEN_DIM": self.hidden_dim,
            "TRAIN_STEPS": self.train_steps,
        }
