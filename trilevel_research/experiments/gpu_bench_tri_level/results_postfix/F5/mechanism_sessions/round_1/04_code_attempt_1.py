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