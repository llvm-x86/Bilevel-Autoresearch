"""Tests for domains/train_opt/tri_level_controller.py (mocked LLM/subprocess)."""
from __future__ import annotations

import json
import textwrap
from unittest.mock import MagicMock, patch

import pytest

from domains.train_opt.config import SearchConfig
from trilevel_research.config import MechanismResearchConfig
from trilevel_research.core.base_meta_mechanism_research import MetaMechanismResult
from trilevel_research.core.mechanism_session_trace import MechanismSessionRecord
from trilevel_research.domains.train_opt.tri_level_controller import (
    TriLevelController,
    TriLevelReport,
)

MINIMAL_RUNNER = textwrap.dedent(
    '''\
    """Minimal runner stub for tri-level controller tests."""
    from domains.train_opt.config import SearchConfig


    class TrainTrace:
        def __init__(self):
            self.results = []

        def summary(self, last_n=10):
            return "mock trace summary"


    class TrainRunner:
        def __init__(self, train_py, work_dir, llm_client, search_config, artifacts_dir, simple_mode=False):
            self.train_py = train_py
            self.work_dir = work_dir
            self.search_config = search_config
            self.artifacts_dir = artifacts_dir
            self.trace = TrainTrace()
    '''
)

MINIMAL_MECH_RESEARCH = textwrap.dedent(
    '''\
    """Minimal mechanism_research stub for tri-level controller tests."""
    class TrainMechanismResearcher:
        def __init__(self, model="", api_key="", provider="deepseek", max_code_retries=3):
            self.tabu_registry = None

        def research(self, trace_summary, runner_code, session_dir, bottleneck=""):
            raise NotImplementedError("mocked in tests")

        def apply(self, path, result):
            return False

        def validate(self, path):
            return False
    '''
)


def _mock_llm_client():
    client = MagicMock()
    client._model = "mock-model"
    client._api_key = "mock-key"
    client._provider = "deepseek"
    return client


def _batch_report(*, trace=None, baseline=3.0, best=2.9):
    return {
        "baseline_bpb": baseline,
        "best_val_bpb": best,
        "best_iteration": 1,
        "trace": trace or [{"status": "keep"}, {"status": "discard"}],
        "outer_trace": [{"cycle": 1}],
    }


@pytest.fixture
def tri_env(tmp_path):
    run_dir = tmp_path / "run"
    work_dir = tmp_path / "work"
    train_py = tmp_path / "train.py"
    train_py.write_text("# stub train script\n", encoding="utf-8")
    work_dir.mkdir()

    canonical_runner = tmp_path / "canonical_runner.py"
    canonical_runner.write_text(MINIMAL_RUNNER, encoding="utf-8")
    canonical_mech = tmp_path / "canonical_mech.py"
    canonical_mech.write_text(MINIMAL_MECH_RESEARCH, encoding="utf-8")

    return {
        "run_dir": run_dir,
        "work_dir": work_dir,
        "train_py": train_py,
        "canonical_runner": canonical_runner,
        "canonical_mech": canonical_mech,
    }


def _make_controller(env, **kwargs):
    defaults = dict(
        run_dir=env["run_dir"],
        train_py=env["train_py"],
        work_dir=env["work_dir"],
        llm_client=_mock_llm_client(),
        inner_budget=2,
        outer_cycles=4,
        time_budget=60,
        enable_level3=False,
        canonical_runner_py=env["canonical_runner"],
        canonical_mech_research_py=env["canonical_mech"],
    )
    defaults.update(kwargs)
    return TriLevelController(**defaults)


class TestTriLevelReport:
    def test_to_dict_includes_group(self):
        report = TriLevelReport(group="C", baseline_bpb=3.0, best_val_bpb=2.5)
        data = report.to_dict()
        assert data["group"] == "C"
        assert data["baseline_bpb"] == 3.0
        assert "level2_sessions" in data


class TestTriLevelControllerRun:
    @patch("trilevel_research.domains.train_opt.tri_level_controller.TrainOuterLoop")
    def test_bilevel_group_c_completes(self, mock_outer_cls, tri_env):
        mock_outer = MagicMock()
        mock_outer.run.return_value = _batch_report()
        mock_outer_cls.return_value = mock_outer

        controller = _make_controller(tri_env, outer_cycles=4, enable_level3=False)

        l2_record = MechanismSessionRecord(
            round=1,
            session_id="r1",
            mechanism_name="test_mech",
            implementation_strategy="new_helper_class",
            target="TestHelper",
            applied=True,
            validated=True,
        )
        with patch.object(controller, "_run_level2", return_value=l2_record):
            report = controller.run()

        assert report.group == "C"
        assert report.level2_rounds == 1
        assert report.total_iterations == 4
        assert report.improvement == pytest.approx(0.1)
        assert (tri_env["run_dir"] / "runner_final.py").is_file()

    @patch("trilevel_research.domains.train_opt.tri_level_controller.TrainOuterLoop")
    def test_skips_l2_when_schedule_defers(self, mock_outer_cls, tri_env):
        mock_outer = MagicMock()
        mock_outer.run.return_value = _batch_report(trace=[{"status": "keep"}] * 5)
        mock_outer_cls.return_value = mock_outer

        controller = _make_controller(tri_env, outer_cycles=4)
        defer = MagicMock(
            fire_level2=False,
            fire_level3=False,
            reason="inner loop improving; defer L2",
            batch_size=2,
        )
        with patch.object(controller.schedule, "decide", return_value=defer):
            with patch.object(controller, "_run_level2") as mock_l2:
                report = controller.run()

        mock_l2.assert_not_called()
        assert report.level2_rounds == 0

    @patch("trilevel_research.domains.train_opt.tri_level_controller.TrainOuterLoop")
    def test_level3_fires_when_enabled(self, mock_outer_cls, tri_env):
        mock_outer = MagicMock()
        mock_outer.run.return_value = _batch_report()
        mock_outer_cls.return_value = mock_outer

        mech_config = MechanismResearchConfig(enable_level3=True, level3_interval=1)
        controller = _make_controller(
            tri_env,
            outer_cycles=4,
            enable_level3=True,
            mech_config=mech_config,
        )

        l2_record = MechanismSessionRecord(
            round=1,
            session_id="r1",
            mechanism_name="m",
            implementation_strategy="new_helper_class",
            target="T",
            applied=True,
            validated=True,
        )
        l3_info = {"round": 1, "applied": True, "patch_name": "p1"}

        with patch.object(controller, "_run_level2", return_value=l2_record):
            with patch.object(controller, "_run_level3", return_value=l3_info) as mock_l3:
                report = controller.run()

        assert report.group == "F"
        mock_l3.assert_called_once()
        assert report.level3_rounds == 1
        assert report.level3_sessions[0]["patch_name"] == "p1"

    @patch("trilevel_research.domains.train_opt.tri_level_controller.TrainOuterLoop")
    def test_tabu_and_schedule_persisted(self, mock_outer_cls, tri_env):
        mock_outer = MagicMock()
        mock_outer.run.return_value = _batch_report()
        mock_outer_cls.return_value = mock_outer

        controller = _make_controller(tri_env, outer_cycles=4)
        l2_record = MechanismSessionRecord(
            round=1,
            session_id="r1",
            mechanism_name="m",
            implementation_strategy="new_helper_class",
            target="T",
        )
        with patch.object(controller, "_run_level2", return_value=l2_record):
            controller.run()

        assert (tri_env["run_dir"] / "mechanism_tabu.json").is_file()


class TestTriLevelHelpers:
    def test_make_search_config_from_template(self, tri_env):
        template = SearchConfig(inner_budget=7, time_budget=120, strategy="exploit")
        controller = _make_controller(
            tri_env,
            search_config=template,
            forbidden_params={"DEPTH"},
        )
        cfg = controller._make_search_config()
        assert cfg.inner_budget == 7
        assert cfg.strategy == "exploit"
        assert "DEPTH" not in cfg.editable_params

    def test_make_search_config_defaults(self, tri_env):
        controller = _make_controller(tri_env, inner_budget=3, time_budget=90)
        cfg = controller._make_search_config()
        assert cfg.inner_budget == 3
        assert cfg.time_budget == 90

    def test_restore_backup(self, tri_env):
        tri_env["run_dir"].mkdir(parents=True, exist_ok=True)
        target = tri_env["run_dir"] / "runner.py"
        target.write_text("original\n", encoding="utf-8")
        backup = target.with_suffix(".py.bak_session123")
        backup.write_text("restored\n", encoding="utf-8")
        TriLevelController._restore_backup(target, "session123")
        assert target.read_text(encoding="utf-8") == "restored\n"

    def test_load_runner_class(self, tri_env):
        controller = _make_controller(tri_env)
        tri_env["run_dir"].mkdir(parents=True, exist_ok=True)
        runner_path = tri_env["run_dir"] / "runner.py"
        runner_path.write_text(MINIMAL_RUNNER, encoding="utf-8")
        cls = controller._load_runner_class(runner_path)
        assert cls.__name__ == "TrainRunner"


class TestRunLevel2:
    def test_tabu_blocks_before_apply(self, tri_env):
        controller = _make_controller(tri_env)
        tri_env["run_dir"].mkdir(parents=True, exist_ok=True)
        tri_env["run_dir"].joinpath("runner.py").write_text("# runner\n", encoding="utf-8")

        mock_result = MagicMock(
            session_id="s1",
            mechanism_name="blocked_mech",
            implementation_strategy="new_helper_class",
            target="Blocked",
            hypothesis="h",
            code_retries=0,
        )
        mock_researcher = MagicMock()
        mock_researcher.research.return_value = mock_result

        controller.tabu.record_failure("blocked_mech", "Blocked", 0, reason="fail")

        runner = MagicMock()
        runner.trace.summary.return_value = "trace"

        with patch.object(controller, "_load_mechanism_researcher_class", return_value=lambda **kw: mock_researcher):
            record = controller._run_level2(runner=runner, l2_round=1, batch_size=2)

        assert record.blocked_by_tabu is True
        mock_researcher.apply.assert_not_called()

    def test_import_fail_records_tabu_and_restores(self, tri_env):
        controller = _make_controller(tri_env)
        tri_env["run_dir"].mkdir(parents=True, exist_ok=True)
        runner_py = tri_env["run_dir"] / "runner.py"
        runner_py.write_text("# runner\n", encoding="utf-8")

        mock_result = MagicMock(
            session_id="fail001",
            mechanism_name="bad_mech",
            implementation_strategy="new_helper_class",
            target="Bad",
            hypothesis="h",
            code_retries=0,
            validation_error="import_fail",
        )
        mock_researcher = MagicMock()
        mock_researcher.research.return_value = mock_result
        mock_researcher.apply.return_value = True
        mock_researcher.validate.return_value = False

        runner = MagicMock()
        runner.trace.summary.return_value = "trace"

        with patch.object(controller, "_load_mechanism_researcher_class", return_value=lambda **kw: mock_researcher):
            with patch.object(TriLevelController, "_restore_backup") as mock_restore:
                record = controller._run_level2(runner=runner, l2_round=1, batch_size=2)

        assert record.applied is False
        assert record.validated is False
        mock_restore.assert_called_once()
        assert controller.tabu.is_tabu("bad_mech", "Bad", 2)[0] is True

    def test_l2_success_records_tabu(self, tri_env):
        controller = _make_controller(tri_env)
        tri_env["run_dir"].mkdir(parents=True, exist_ok=True)
        tri_env["run_dir"].joinpath("runner.py").write_text("# runner\n", encoding="utf-8")

        mock_result = MagicMock(
            session_id="ok001",
            mechanism_name="good_mech",
            implementation_strategy="new_helper_class",
            target="Good",
            hypothesis="h",
            code_retries=0,
        )
        mock_researcher = MagicMock()
        mock_researcher.research.return_value = mock_result
        mock_researcher.apply.return_value = True
        mock_researcher.validate.return_value = True

        runner = MagicMock()
        runner.trace.summary.return_value = "trace"

        with patch.object(controller, "_load_mechanism_researcher_class", return_value=lambda **kw: mock_researcher):
            record = controller._run_level2(runner=runner, l2_round=1, batch_size=2)

        assert record.applied is True
        assert record.validated is True
        assert controller.tabu.is_tabu("good_mech", "Good", 2)[0] is True

    def test_l2_exception_sets_error(self, tri_env):
        controller = _make_controller(tri_env)
        tri_env["run_dir"].mkdir(parents=True, exist_ok=True)
        tri_env["run_dir"].joinpath("runner.py").write_text("# runner\n", encoding="utf-8")

        mock_researcher = MagicMock()
        mock_researcher.research.side_effect = RuntimeError("research blew up")

        runner = MagicMock()
        runner.trace.summary.return_value = "trace"

        with patch.object(controller, "_load_mechanism_researcher_class", return_value=lambda **kw: mock_researcher):
            record = controller._run_level2(runner=runner, l2_round=1, batch_size=2)

        assert "research blew up" in record.error

    def test_load_module_rewrites_relative_imports(self, tri_env):
        controller = _make_controller(tri_env)
        tri_env["run_dir"].mkdir(parents=True, exist_ok=True)
        runner_path = tri_env["run_dir"] / "runner.py"
        runner_path.write_text(
            MINIMAL_RUNNER.replace(
                "from domains.train_opt.config import SearchConfig",
                "from .config import SearchConfig",
            ),
            encoding="utf-8",
        )
        cls = controller._load_runner_class(runner_path)
        assert cls.__name__ == "TrainRunner"
        assert "from domains.train_opt.config import" in runner_path.read_text(encoding="utf-8")

    def test_load_mechanism_researcher_class(self, tri_env):
        controller = _make_controller(tri_env)
        tri_env["run_dir"].mkdir(parents=True, exist_ok=True)
        controller.run_mech_py.write_text(MINIMAL_MECH_RESEARCH, encoding="utf-8")
        cls = controller._load_mechanism_researcher_class()
        assert cls.__name__ == "TriLevelTrainMechanismResearcher"


class TestRunLevel3:
    @patch("trilevel_research.domains.train_opt.meta_mechanism_research.TrainMetaMechanismResearcher")
    def test_applies_schedule_patch(self, mock_meta_cls, tri_env):
        controller = _make_controller(tri_env, enable_level3=True)
        tri_env["run_dir"].mkdir(parents=True, exist_ok=True)
        tri_env["run_dir"].joinpath("mechanism_research.py").write_text(
            MINIMAL_MECH_RESEARCH, encoding="utf-8"
        )
        tri_env["run_dir"].joinpath("mechanism_sessions").mkdir()
        (tri_env["run_dir"] / "mechanism_sessions" / "round_1").mkdir()
        (tri_env["run_dir"] / "mechanism_sessions" / "round_1" / "06_summary.json").write_text(
            json.dumps({"session_id": "r1", "mechanism_name": "m"}),
            encoding="utf-8",
        )

        mock_result = MetaMechanismResult(
            session_id="l3_001",
            hypothesis="h",
            patch_name="sched_patch",
            implementation_strategy="new_helper_class",
            target="Helper",
            spec="spec",
            code="class Helper:\n    pass\n",
            schedule_patch={"level2_interval": 5},
        )
        mock_meta = MagicMock()
        mock_meta.research.return_value = mock_result
        mock_meta.apply.return_value = True
        mock_meta.validate.return_value = True
        mock_meta_cls.return_value = mock_meta

        info = controller._run_level3(l3_round=1, inner_trace=[])

        assert info["applied"] is True
        assert controller.mech_config.level2_interval == 5
        assert (tri_env["run_dir"] / "schedule_config.json").is_file()

    @patch("trilevel_research.domains.train_opt.meta_mechanism_research.TrainMetaMechanismResearcher")
    def test_validation_fail_restores_backup(self, mock_meta_cls, tri_env):
        controller = _make_controller(tri_env, enable_level3=True)
        tri_env["run_dir"].mkdir(parents=True, exist_ok=True)
        tri_env["run_dir"].joinpath("mechanism_research.py").write_text(
            MINIMAL_MECH_RESEARCH, encoding="utf-8"
        )
        tri_env["run_dir"].joinpath("mechanism_sessions").mkdir()

        mock_result = MetaMechanismResult(
            session_id="l3_fail",
            hypothesis="h",
            patch_name="p",
            implementation_strategy="new_helper_class",
            target="Helper",
            spec="spec",
            code="class Helper:\n    pass\n",
            validation_error="import_fail",
        )
        mock_meta = MagicMock()
        mock_meta.research.return_value = mock_result
        mock_meta.apply.return_value = True
        mock_meta.validate.return_value = False
        mock_meta_cls.return_value = mock_meta

        with patch.object(TriLevelController, "_restore_backup") as mock_restore:
            info = controller._run_level3(l3_round=1, inner_trace=[])

        assert info["applied"] is False
        assert "error" in info
        mock_restore.assert_called_once()

    @patch("trilevel_research.domains.train_opt.meta_mechanism_research.TrainMetaMechanismResearcher")
    def test_l3_exception_sets_error(self, mock_meta_cls, tri_env):
        controller = _make_controller(tri_env, enable_level3=True)
        tri_env["run_dir"].mkdir(parents=True, exist_ok=True)
        tri_env["run_dir"].joinpath("mechanism_research.py").write_text(
            MINIMAL_MECH_RESEARCH, encoding="utf-8"
        )
        tri_env["run_dir"].joinpath("mechanism_sessions").mkdir()

        mock_meta = MagicMock()
        mock_meta.research.side_effect = RuntimeError("L3 failed")
        mock_meta_cls.return_value = mock_meta

        info = controller._run_level3(l3_round=1, inner_trace=[])
        assert "L3 failed" in info["error"]
