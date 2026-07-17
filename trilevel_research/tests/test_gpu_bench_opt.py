"""Offline tests for gpu_bench_opt tri-level integration."""
from __future__ import annotations

import json
import shutil
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

from trilevel_research.config import MechanismResearchConfig
from trilevel_research.core.adaptive_mechanism_schedule import AdaptiveMechanismSchedule
from trilevel_research.domains.gpu_bench_opt.outer import GpuBenchOuterLoop
from trilevel_research.domains.gpu_bench_opt.runner import GpuBenchRunner
from trilevel_research.domains.gpu_bench_opt.search_config import SearchConfig
from trilevel_research.domains.gpu_bench_opt.tri_level_controller import (
    GpuBenchTriLevelController,
)


def _mock_bench_payload(val_bpb: float, status: str = "ok") -> str:
    return json.dumps({"val_bpb": val_bpb, "train_bpb": val_bpb, "elapsed_s": 1.0, "status": status})


def _make_fake_bin(tmp_path: Path) -> Path:
    """Create a stub executable that prints JSON like gpu_bench --json."""
    script = tmp_path / "gpu_bench"
    script.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import json, sys
            val = 2.5
            if "--lr" in sys.argv:
                idx = sys.argv.index("--lr")
                lr = float(sys.argv[idx + 1])
                val = max(1.0, 2.5 - lr * 100)
            print(json.dumps({"val_bpb": val, "train_bpb": val, "elapsed_s": 0.5, "status": "ok"}))
            """
        ),
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


class MockLLMClient:
    """Deterministic LLM for offline controller tests."""

    def __init__(self, proposals: list[dict] | None = None, outer_responses: list[dict] | None = None):
        self._model = "mock"
        self._api_key = "test"
        self._provider = "mock"
        self.proposals = proposals or [
            {"changes": {"LR": 0.005}, "hypothesis": "lower LR"},
            {"changes": {"BATCH_SIZE": 128}, "hypothesis": "larger batch"},
            {"changes": {"HIDDEN_DIM": 512}, "hypothesis": "wider net"},
            {"changes": {"WEIGHT_DECAY": 0.01}, "hypothesis": "regularize"},
            {"changes": {"LR": 0.002}, "hypothesis": "even lower LR"},
        ]
        self.outer_responses = outer_responses or [
            {
                "diagnosis": "ok",
                "strategy": "focused",
                "freeze_params": [],
                "unfreeze_params": [],
                "guidance": "focus on LR",
                "reasoning": "LR helped",
            }
        ]
        self._proposal_idx = 0
        self._outer_idx = 0

    def call(self, prompt: str, system: str = "", max_tokens: int = 4000) -> str:
        if "freeze_params" in prompt or "Outer Loop Trace" in prompt:
            resp = self.outer_responses[min(self._outer_idx, len(self.outer_responses) - 1)]
            self._outer_idx += 1
            return json.dumps(resp)
        proposal = self.proposals[min(self._proposal_idx, len(self.proposals) - 1)]
        self._proposal_idx += 1
        return json.dumps(proposal)


class TestGpuBenchRunner:
    def test_subprocess_mock_baseline_and_iteration(self, tmp_path):
        fake_bin = _make_fake_bin(tmp_path)
        client = MockLLMClient()

        with patch("trilevel_research.domains.gpu_bench_opt.runner.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=_mock_bench_payload(2.5) + "\n",
                stderr="",
            )
            runner = GpuBenchRunner(
                bench_bin=fake_bin,
                llm_client=client,
                search_config=SearchConfig(inner_budget=2),
            )
            baseline = runner.run_baseline()
            assert baseline.val_bpb == 2.5
            assert baseline.status == "keep"

            mock_run.return_value.stdout = _mock_bench_payload(2.0) + "\n"
            result = runner.run_iteration(1)
            assert result.val_bpb == 2.0
            assert result.status == "keep"
            assert runner.trace.best_bpb == 2.0

    def test_crash_handling(self, tmp_path):
        fake_bin = _make_fake_bin(tmp_path)
        with patch("trilevel_research.domains.gpu_bench_opt.runner.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="OOM")
            runner = GpuBenchRunner(bench_bin=fake_bin)
            result = runner._run_trial(runner.config, 1, "test")
            assert result.status == "crash"
            assert result.val_bpb == 99.0

    def test_frozen_params_filtered(self, tmp_path):
        fake_bin = _make_fake_bin(tmp_path)
        client = MockLLMClient(proposals=[{"changes": {"LR": 0.001, "BATCH_SIZE": 32}, "hypothesis": "x"}])
        with patch("trilevel_research.domains.gpu_bench_opt.runner.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=_mock_bench_payload(2.0) + "\n", stderr="")
            runner = GpuBenchRunner(
                bench_bin=fake_bin,
                llm_client=client,
                search_config=SearchConfig(frozen_params=["BATCH_SIZE"]),
            )
            runner.run_baseline()
            result = runner.run_iteration(1)
            assert "BATCH_SIZE" not in result.changes
            assert "LR" in result.changes


class TestGpuBenchOuterLoop:
    def test_outer_loop_offline(self, tmp_path):
        fake_bin = _make_fake_bin(tmp_path)
        client = MockLLMClient()
        with patch("trilevel_research.domains.gpu_bench_opt.runner.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=_mock_bench_payload(2.5) + "\n",
                stderr="",
            )
            runner = GpuBenchRunner(
                bench_bin=fake_bin,
                llm_client=client,
                search_config=SearchConfig(inner_budget=2),
                artifacts_dir=tmp_path / "artifacts",
            )
            outer = GpuBenchOuterLoop(runner=runner, llm_client=client, max_outer_cycles=2)
            report = outer.run()

        assert report["outer_cycles"] == 2
        assert report["total_iterations"] >= 2
        assert report["baseline_bpb"] is not None
        assert len(report["outer_trace"]) == 2


class TestGpuBenchTriLevelController:
    def _setup_controller(self, tmp_path, enable_level3: bool, fake_bin: Path, client: MockLLMClient):
        run_dir = tmp_path / ("F" if enable_level3 else "C")
        mech_config = MechanismResearchConfig(
            level2_interval=2,
            level3_interval=2,
            enable_level3=enable_level3,
            enable_tabu=True,
            validation_strict=False,
        )
        return GpuBenchTriLevelController(
            run_dir=run_dir,
            bench_bin=fake_bin,
            llm_client=client,
            inner_budget=2,
            outer_cycles=4,
            enable_level3=enable_level3,
            mech_config=mech_config,
            search_config=SearchConfig(inner_budget=2),
        )

    def test_group_c_schedule_fires_l2_not_l3(self, tmp_path):
        from trilevel_research.core.mechanism_session_trace import (
            MechanismSessionRecord,
        )

        fake_bin = _make_fake_bin(tmp_path)
        client = MockLLMClient()
        controller = self._setup_controller(tmp_path, enable_level3=False, fake_bin=fake_bin, client=client)

        def fake_l2(runner, l2_round, batch_size):
            return MechanismSessionRecord(
                round=l2_round,
                session_id=f"round_{l2_round}",
                mechanism_name="stub_mech",
                implementation_strategy="new_helper_class",
                target="Stub",
                session_dir=tmp_path / "mech" / f"round_{l2_round}",
                applied=True,
                validated=True,
            )

        with patch("trilevel_research.domains.gpu_bench_opt.runner.subprocess.run") as mock_run, patch.object(
            controller, "_run_level2", side_effect=fake_l2
        ) as mock_l2, patch.object(controller, "_run_level3") as mock_l3:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=_mock_bench_payload(2.5) + "\n",
                stderr="",
            )
            report = controller.run()

        assert report.group == "C"
        assert mock_l2.called
        assert report.level2_rounds >= 1
        assert report.level3_rounds == 0
        assert not mock_l3.called
        assert report.total_iterations > 0
        assert (controller.run_dir / "runner_final.py").exists()

    def test_group_f_enables_l3_flag(self, tmp_path):
        from trilevel_research.core.mechanism_session_trace import (
            MechanismSessionRecord,
        )

        fake_bin = _make_fake_bin(tmp_path)
        client = MockLLMClient()
        run_dir = tmp_path / "F"
        mech_config = MechanismResearchConfig(
            level2_interval=2,
            level3_interval=1,
            enable_level3=True,
            enable_tabu=True,
            validation_strict=False,
        )
        controller = GpuBenchTriLevelController(
            run_dir=run_dir,
            bench_bin=fake_bin,
            llm_client=client,
            inner_budget=2,
            outer_cycles=4,
            enable_level3=True,
            mech_config=mech_config,
            search_config=SearchConfig(inner_budget=2),
        )

        def fake_l2(runner, l2_round, batch_size):
            return MechanismSessionRecord(
                round=l2_round,
                session_id=f"round_{l2_round}",
                mechanism_name="stub_mech",
                implementation_strategy="new_helper_class",
                target="Stub",
                session_dir=tmp_path / "mech" / f"round_{l2_round}",
            )

        with patch("trilevel_research.domains.gpu_bench_opt.runner.subprocess.run") as mock_run, patch.object(
            controller, "_run_level2", side_effect=fake_l2
        ), patch.object(
            controller, "_run_level3", return_value={"round": 1, "applied": False}
        ) as mock_l3:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=_mock_bench_payload(2.5) + "\n",
                stderr="",
            )
            report = controller.run()

        assert report.group == "F"
        assert mock_l3.called
        assert report.level3_rounds >= 1

    def test_schedule_l2_every_2_cycles(self, tmp_path):
        schedule = AdaptiveMechanismSchedule(level2_interval=2, level3_interval=2)
        decision = schedule.decide(
            inner_trace=[],
            l2_sessions=[],
            completed_outer_cycles=2,
            config=MechanismResearchConfig(level2_interval=2),
        )
        assert decision.batch_size == 2
        assert decision.fire_level2 is True


class TestGpuBenchMechanismResearch:
    def test_mechanism_researcher_dry_run(self, tmp_path):
        from trilevel_research.domains.gpu_bench_opt.mechanism_research import (
            GpuBenchMechanismResearcher,
        )

        class MechMockLLM:
            def call(self, prompt, system="", max_tokens=4000):
                if "failure mode" in prompt.lower():
                    return "**Selected**: 1 — best."
                if "implementation spec" in prompt.lower() and "reference code" not in prompt.lower():
                    return (
                        "1. **Mechanism name**: tabu_filter\n"
                        "2. **Implementation strategy**: new_helper_class\n"
                        "3. **Target**: TabuFilter\n"
                    )
                if "reference code" in prompt.lower():
                    return "class TabuFilter:\n    pass\n"
                return "Hypothesis 1: tabu filter"

        runner_stub = (Path(__file__).parent.parent / "domains/gpu_bench_opt/runner.py").read_text()
        researcher = GpuBenchMechanismResearcher(api_key="test")
        researcher.client = MechMockLLM()
        session_dir = tmp_path / "l2_session"
        result = researcher.research(
            trace_summary="iter 1 [discard] val_bpb=2.5",
            runner_code=runner_stub,
            session_dir=session_dir,
        )
        assert result.mechanism_name == "tabu_filter"
        assert result.implementation_strategy == "new_helper_class"
        assert (session_dir / "01_exploration.md").exists()
        assert (session_dir / "03_spec.md").exists()
        assert "class TabuFilter" in result.code

    def test_apply_helper_class(self, tmp_path):
        from trilevel_research.domains.gpu_bench_opt.mechanism_research import (
            GpuBenchMechanismResearcher,
            GpuBenchMechanismResult,
        )

        canonical = Path(__file__).parent.parent / "domains/gpu_bench_opt/runner.py"
        runner_copy = tmp_path / "runner.py"
        shutil.copy2(canonical, runner_copy)

        researcher = GpuBenchMechanismResearcher(api_key="test")
        result = GpuBenchMechanismResult(
            session_id="test001",
            hypothesis="add helper",
            mechanism_name="test_helper",
            implementation_strategy="new_helper_class",
            target="TestHelper",
            spec="spec",
            code="class TestHelper:\n    value = 1\n",
            session_dir=tmp_path / "session",
        )
        result.session_dir.mkdir(parents=True, exist_ok=True)

        assert researcher.apply(runner_copy, result) is True
        patched = runner_copy.read_text(encoding="utf-8")
        assert "class TestHelper" in patched
        assert "class GpuBenchRunner" in patched
