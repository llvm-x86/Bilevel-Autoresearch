"""Parse Level-2 mechanism research session directories into structured records."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MechanismSessionRecord:
    """Structured summary of one Level-2 research session."""

    round: int
    session_id: str
    mechanism_name: str
    implementation_strategy: str
    target: str
    hypothesis: str = ""
    code_retries: int = 0
    applied: bool = False
    validated: bool = False
    blocked_by_tabu: bool = False
    error: str = ""
    session_dir: Path = field(default_factory=lambda: Path("."))
    exploration: str = ""
    critique: str = ""
    spec: str = ""
    code: str = ""
    patched_artifact: str = ""

    def to_dict(self) -> dict:
        return {
            "round": self.round,
            "session_id": self.session_id,
            "mechanism_name": self.mechanism_name,
            "implementation_strategy": self.implementation_strategy,
            "target": self.target,
            "hypothesis": self.hypothesis[:300],
            "code_retries": self.code_retries,
            "applied": self.applied,
            "validated": self.validated,
            "blocked_by_tabu": self.blocked_by_tabu,
            "error": self.error,
            "session_dir": str(self.session_dir),
        }


@dataclass
class MechanismSessionTrace:
    """Aggregated trace over multiple L2 sessions."""

    sessions: list[MechanismSessionRecord] = field(default_factory=list)

    def to_text(self, max_hypothesis_chars: int = 200) -> str:
        return MechanismSessionTraceBuilder.to_text(
            self.sessions,
            max_hypothesis_chars=max_hypothesis_chars,
        )

    def mechanism_names(self) -> list[str]:
        return [s.mechanism_name for s in self.sessions]

    def revert_rate(self) -> float:
        applied_sessions = [s for s in self.sessions if s.applied]
        if not applied_sessions:
            return 0.0
        reverts = sum(1 for s in applied_sessions if s.validated is False)
        return reverts / len(applied_sessions)

    def apply_rate(self) -> float:
        if not self.sessions:
            return 0.0
        return sum(1 for s in self.sessions if s.applied) / len(self.sessions)


def _read_text(path: Path) -> str:
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return ""


def _read_code_attempt(session_dir: Path) -> str:
    attempts = sorted(session_dir.glob("04_code_attempt_*.py"))
    if attempts:
        return _read_text(attempts[-1])
    final = session_dir / "04_code_final.py"
    if final.is_file():
        return _read_text(final)
    return ""


def parse_session_dir(
    session_dir: Path,
    *,
    round: int | None = None,
    report_entry: dict | None = None,
) -> MechanismSessionRecord:
    """Parse a single L2 session directory (01_exploration.md … 06_summary.json)."""
    session_dir = Path(session_dir)
    rnd = round
    if rnd is None:
        match = re.search(r"round_(\d+)", session_dir.name)
        rnd = int(match.group(1)) if match else 0

    return MechanismSessionTraceBuilder.parse_session_dir(
        session_dir,
        round_num=rnd,
        report_entry=report_entry,
    )


class MechanismSessionTraceBuilder:
    """Build L2 session records from on-disk session artifacts."""

    @staticmethod
    def parse_session_dir(
        session_dir: Path,
        round_num: int = 0,
        report_entry: dict | None = None,
    ) -> MechanismSessionRecord:
        session_dir = Path(session_dir)
        summary_path = session_dir / "06_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing 06_summary.json in {session_dir}")

        record = MechanismSessionRecord(
            round=round_num,
            session_id=session_dir.name,
            mechanism_name="unknown",
            implementation_strategy="unknown",
            target="unknown",
            session_dir=session_dir,
        )

        if summary_path.exists():
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            record.session_id = data.get("session_id", record.session_id)
            record.mechanism_name = data.get("mechanism_name", record.mechanism_name)
            record.implementation_strategy = data.get(
                "implementation_strategy", record.implementation_strategy
            )
            record.target = data.get("target", record.target)
            record.hypothesis = data.get("hypothesis", "")
            record.code_retries = int(data.get("code_retries", 0))

        record.exploration = _read_text(session_dir / "01_exploration.md")
        record.critique = _read_text(session_dir / "02_critique.md")
        record.spec = _read_text(session_dir / "03_spec.md")
        record.code = _read_code_attempt(session_dir)

        patched_path = session_dir / "05_patched_runner.py"
        if not patched_path.is_file():
            patched_path = session_dir / "05_patched_mechanism_research.py"
        record.patched_artifact = _read_text(patched_path)

        if report_entry:
            record.applied = bool(report_entry.get("applied", record.applied))
            record.validated = bool(report_entry.get("validated", record.validated))
            record.blocked_by_tabu = bool(report_entry.get("blocked_by_tabu", False))
            record.error = report_entry.get("error", "") or ""
        else:
            record.applied = patched_path.is_file() or record.applied

        return record

    @classmethod
    def build(
        cls,
        sessions_root: Path,
        inner_trace: list[dict] | None = None,
        report_sessions: list[dict] | None = None,
        report: dict | None = None,
    ) -> list[MechanismSessionRecord]:
        """Parse all round_* session dirs under sessions_root."""
        if report is not None and report_sessions is None:
            report_sessions = report.get("level2_sessions")

        sessions_root = Path(sessions_root)
        records: list[MechanismSessionRecord] = []

        if sessions_root.is_dir():
            round_dirs = sorted(
                d for d in sessions_root.iterdir()
                if d.is_dir() and d.name.startswith("round_")
            )
            report_by_round: dict[int, dict] = {}
            if report_sessions:
                for item in report_sessions:
                    report_by_round[int(item.get("round", 0))] = item

            for rd in round_dirs:
                try:
                    round_num = int(rd.name.split("_", 1)[1])
                except (IndexError, ValueError):
                    round_num = len(records) + 1
                records.append(
                    cls.parse_session_dir(
                        rd,
                        round_num=round_num,
                        report_entry=report_by_round.get(round_num),
                    )
                )

        if report_sessions:
            by_round = {r.round: r for r in records}
            for item in report_sessions:
                rnd = int(item.get("round", len(records) + 1))
                if rnd in by_round:
                    rec = by_round[rnd]
                    rec.applied = bool(item.get("applied", rec.applied))
                    rec.validated = bool(item.get("validated", rec.validated))
                    rec.blocked_by_tabu = bool(item.get("blocked_by_tabu", False))
                    rec.error = item.get("error", "") or ""
                    if item.get("mechanism_name"):
                        rec.mechanism_name = item["mechanism_name"]
                    if item.get("target"):
                        rec.target = item["target"]
                    if item.get("strategy"):
                        rec.implementation_strategy = item["strategy"]
                else:
                    records.append(
                        MechanismSessionRecord(
                            round=rnd,
                            session_id=item.get("session_id", f"round_{rnd}"),
                            mechanism_name=item.get("mechanism_name", "unknown"),
                            implementation_strategy=item.get("strategy", "unknown"),
                            target=item.get("target", "unknown"),
                            applied=bool(item.get("applied", False)),
                            validated=bool(item.get("validated", False)),
                            blocked_by_tabu=bool(item.get("blocked_by_tabu", False)),
                            error=item.get("error", "") or "",
                        )
                    )

        records.sort(key=lambda r: r.round)
        return records

    @classmethod
    def build_trace(
        cls,
        sessions_root: Path,
        inner_trace: list[dict] | None = None,
        report: dict | None = None,
    ) -> MechanismSessionTrace:
        sessions = cls.build(
            sessions_root,
            inner_trace=inner_trace,
            report=report,
        )
        return MechanismSessionTrace(sessions=sessions)

    @staticmethod
    def to_text(
        records: list[MechanismSessionRecord],
        inner_trace: list[dict] | None = None,
        max_hypothesis_chars: int = 200,
    ) -> str:
        """Human-readable summary for Level-3 explore prompts."""
        lines = [f"## L2 Session Trace ({len(records)} sessions)", ""]
        if not records:
            lines.append("(no Level-2 sessions yet)")
        else:
            for rec in records:
                status = "applied" if rec.applied else "not applied"
                if rec.blocked_by_tabu:
                    status = "blocked_by_tabu"
                if rec.validated is False:
                    status = "import_failed"
                if rec.error:
                    status = f"error: {rec.error[:80]}"
                lines.append(
                    f"- Round {rec.round}: {rec.mechanism_name} "
                    f"({rec.implementation_strategy} → {rec.target}) [{status}]"
                )
                if rec.hypothesis:
                    lines.append(
                        f"  hypothesis: {rec.hypothesis[:max_hypothesis_chars].replace(chr(10), ' ')}"
                    )

        if inner_trace:
            keeps = sum(1 for r in inner_trace if r.get("status") == "keep")
            discards = sum(1 for r in inner_trace if r.get("status") == "discard")
            crashes = sum(1 for r in inner_trace if r.get("status") == "crash")
            lines.extend([
                "",
                "## Inner loop stats (context)",
                f"Iterations: {len(inner_trace)}, keeps: {keeps}, "
                f"discards: {discards}, crashes: {crashes}",
            ])

        return "\n".join(lines)
