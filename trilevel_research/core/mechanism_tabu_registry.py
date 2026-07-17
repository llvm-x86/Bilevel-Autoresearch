"""Tabu registry over mechanism names / targets (Level-2 proposals)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from trilevel_research.core.mechanism_session_trace import MechanismSessionRecord


@dataclass
class MechanismTabuEntry:
    mechanism_name: str
    target: str
    reason: str
    added_at_round: int
    expires_at_round: int
    strategy: str = ""


@dataclass
class MechanismTabuRegistry:
    max_size: int = 20
    default_tenure: int = 3
    entries: list[MechanismTabuEntry] = field(default_factory=list)
    strategy_entries: list[MechanismTabuEntry] = field(default_factory=list)

    def is_tabu(
        self,
        name: str,
        target: str,
        round_num: int,
        strategy: str = "",
    ) -> tuple[bool, str]:
        """Return (is_tabu, reason) for a proposed mechanism."""
        self._prune_expired(round_num)
        name_l = name.lower().strip()
        target_l = target.lower().strip()
        strategy_l = strategy.lower().strip()

        for entry in self.strategy_entries:
            if strategy_l and entry.strategy.lower() == strategy_l:
                return True, f"strategy tabu ({entry.reason})"

        for entry in self.entries:
            if entry.mechanism_name.lower() == name_l:
                return True, f"mechanism_name tabu ({entry.reason})"
            if entry.target.lower() == target_l:
                return True, f"target tabu ({entry.reason})"
            if name_l and name_l in entry.mechanism_name.lower():
                return True, f"similar mechanism tabu ({entry.reason})"
            if target_l.startswith("generatedmechanism_") and entry.target.lower().startswith(
                "generatedmechanism_"
            ):
                return True, f"generated helper class tabu ({entry.reason})"

        return False, ""

    def record_failure(
        self,
        mechanism_name_or_session: str | MechanismSessionRecord,
        target: str | None = None,
        round_num: int | None = None,
        reason: str = "failure",
    ) -> None:
        if isinstance(mechanism_name_or_session, str):
            assert target is not None and round_num is not None
            self._add_entry(mechanism_name_or_session, target, round_num, reason)
        else:
            session = mechanism_name_or_session
            rnd = round_num if round_num is not None else (session.round or 0)
            self._add_entry(session.mechanism_name, session.target, rnd, reason)
            if reason in (
                "import_fail",
                "harness_syntax_fail",
                "not_applied",
                "validate_fail",
                "validate_attr_error",
            ):
                self._add_strategy_entry(
                    session.implementation_strategy, rnd, reason,
                )

    def record_success(
        self,
        mechanism_name_or_session: str | MechanismSessionRecord,
        target: str | None = None,
        round_num: int | None = None,
    ) -> None:
        """Record success — short tenure to avoid immediate duplicates."""
        if isinstance(mechanism_name_or_session, str):
            assert target is not None and round_num is not None
            self._add_entry(
                mechanism_name_or_session, target, round_num,
                reason="recent_success", tenure=1,
            )
        else:
            session = mechanism_name_or_session
            rnd = round_num if round_num is not None else (session.round or 0)
            self._add_entry(
                session.mechanism_name, session.target, rnd,
                reason="recent_success", tenure=1,
            )

    def to_prompt_block(self) -> str:
        if not self.entries:
            return "(none — no mechanisms are currently tabu)"
        lines = [
            "Do NOT propose mechanisms matching these forbidden entries:",
        ]
        for entry in self.entries[-self.max_size :]:
            lines.append(
                f"- {entry.mechanism_name} (target: {entry.target}) — {entry.reason}"
            )
        return "\n".join(lines)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "max_size": self.max_size,
            "default_tenure": self.default_tenure,
            "entries": [asdict(e) for e in self.entries],
            "strategy_entries": [asdict(e) for e in self.strategy_entries],
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> MechanismTabuRegistry:
        path = Path(path)
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        entries = [MechanismTabuEntry(**e) for e in data.get("entries", [])]
        strategy_entries = [
            MechanismTabuEntry(**e) for e in data.get("strategy_entries", [])
        ]
        return cls(
            max_size=int(data.get("max_size", 20)),
            default_tenure=int(data.get("default_tenure", 3)),
            entries=entries,
            strategy_entries=strategy_entries,
        )

    def stats(self) -> dict:
        return {"size": len(self.entries), "max_size": self.max_size}

    def _add_entry(
        self,
        mechanism_name: str,
        target: str,
        round_num: int,
        reason: str,
        tenure: int | None = None,
    ) -> None:
        tenure = tenure if tenure is not None else self.default_tenure
        self.entries.append(
            MechanismTabuEntry(
                mechanism_name=mechanism_name,
                target=target,
                reason=reason,
                added_at_round=round_num,
                expires_at_round=round_num + tenure,
            )
        )
        if len(self.entries) > self.max_size:
            self.entries = self.entries[-self.max_size :]

    def _prune_expired(self, round_num: int) -> None:
        self.entries = [e for e in self.entries if round_num <= e.expires_at_round]
        self.strategy_entries = [
            e for e in self.strategy_entries if round_num <= e.expires_at_round
        ]

    def _add_strategy_entry(
        self,
        strategy: str,
        round_num: int,
        reason: str,
        tenure: int | None = None,
    ) -> None:
        if not strategy:
            return
        tenure = tenure if tenure is not None else self.default_tenure
        self.strategy_entries.append(
            MechanismTabuEntry(
                mechanism_name="",
                target="",
                strategy=strategy,
                reason=reason,
                added_at_round=round_num,
                expires_at_round=round_num + tenure,
            )
        )
