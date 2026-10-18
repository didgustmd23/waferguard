"""계획 배정·중단·다른 부모 거부와 읽기 전용 조회를 합성 입력으로 검사한다."""

from contextlib import ExitStack
from copy import deepcopy
from importlib.metadata import version
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from src.agents.wm_module_plan import load_wm_modules
from src.agents.wm_run_bindings import inspect_bindings, load_bindings
from src.agents.wm_run_evidence import RunNotComplete, WmRunReader, _scope


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "configs/agents/pipelines/wm_development_example.json"


class WmRunBindingsTest(unittest.TestCase):
    def setUp(self):
        self.plan = load_wm_modules(PLAN)
        self.runs = dict(zip((step.step_id for step in self.plan.steps),
                             ("eda", "proposal", "split", "transform", "cache", "dataset")))

    def test_real_example_matches_current_plan(self):
        runs = load_bindings(ROOT / "configs/agents/pipelines/wm_completed_runs_example.json", self.plan)
        self.assertEqual(runs["dataset"], "wm-dataset-check01")

    def test_invalid_binding_file_is_rejected(self):
        """계획 변경·알 수 없는 단계·경로·중복 키·과대 파일을 거부한다."""
        value = {"schema_version": "wm_run_bindings_v1", "pipeline_signature": self.plan.signature, "runs": self.runs}
        changes = (
            lambda item: item.update(pipeline_signature="0" * 64),
            lambda item: item["runs"].update(unknown="run"),
            lambda item: item["runs"].update(eda="../run"),
            lambda item: item["runs"].update(eda=None),
        )

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bindings.json"

            for change in changes:
                item = deepcopy(value)
                change(item)
                path.write_text(json.dumps(item), encoding="utf-8")

                with self.assertRaises(ValueError):
                    load_bindings(path, self.plan)

            for raw in ('{"runs":{},"runs":{}}', ' ' * 65537):
                path.write_text(raw, encoding="utf-8")

                with self.assertRaises(ValueError):
                    load_bindings(path, self.plan)

    def test_unassigned_and_incomplete_stop(self):
        """미배정·미완료에서는 뒤 단계를 확인하거나 실행하지 않는다."""
        with patch("src.agents.wm_run_bindings.WmRunReader") as factory:
            result = inspect_bindings(self.plan, {})
            factory.return_value.read.assert_not_called()

        self.assertEqual(result["next_action"]["code"], "RUN_NOT_ASSIGNED")
        self.assertEqual(result["steps"][-1]["status"], "not_checked")

        for state in ("awaiting_approval", "blocked_input", "failed_runtime", "cancelled"):
            with patch("src.agents.wm_run_bindings.WmRunReader") as factory:
                factory.return_value.read.side_effect = RunNotComplete(state, "run_eda")
                result = inspect_bindings(self.plan, self.runs)

            self.assertEqual(result["next_action"]["run_status"], state)
            self.assertFalse(result["approval_consumed"])

    def test_wrong_parent_and_changes_never_pass(self):
        """같은 데이터라도 계획과 다른 부모 작업이면 거부하고 상세 데이터는 반환하지 않는다."""
        eda = SimpleNamespace(name="eda", tool=SimpleNamespace())
        proposal = SimpleNamespace(name="proposal", tool=SimpleNamespace(eda_run="different-eda"))

        with patch("src.agents.wm_run_bindings.WmRunReader") as factory:
            factory.return_value.read.side_effect = [eda, proposal]
            result = inspect_bindings(self.plan, self.runs)

        self.assertEqual(result["steps"][1]["status"], "blocked")
        self.assertEqual(result["steps"][2]["status"], "not_checked")

        with patch("src.agents.wm_run_bindings.WmRunReader") as factory:
            factory.return_value.read.side_effect = ValueError("민감한 상세 경로")
            result = inspect_bindings(self.plan, self.runs)
            self.assertNotIn("민감한", json.dumps(result, ensure_ascii=False))
            factory.return_value.check_unchanged.side_effect = ValueError()
            result = inspect_bindings(self.plan, self.runs)

        self.assertEqual(result["status"], "verification_failed")


class WmRunReadonlyIntegrationTest(unittest.TestCase):
    """기존 합성 6단계를 실제 코어로 준비한 뒤 조회 시 쓰기 API를 차단한다."""

    def test_complete_chain_and_tampering(self):
        from tests.agents import test_wm_cache as fixtures
        from tests.agents import test_wm_dataset as dataset_fixtures
        from src.agents import wm_dataset_cli
        from src.agents.artifacts import _fingerprint
        from src.agents.worker_contract import WorkerLimits
        from src.agents.wm_canonical_contract import file_hash

        # 합성 fixture에서만 승인과 모의 transport를 사용한다.
        # 조회 구간에서는 ApprovalStore 생성·승인·worker 시작이 모두 금지된다.
        for name in ("transport", "proposal_transport", "commit_transport", "check_transport", "cache_transport"):
            setattr(self, name, getattr(fixtures.WmCacheTest, name).__get__(self))

        self.approval = fixtures.WmCacheTest.approval.__get__(self)
        fixtures.WmCacheTest.setUp(self)
        approval = self.approval()

        with patch("src.agents.wm_cache.run_cache_worker", side_effect=self.cache_transport):
            self.assertEqual(self.manager.run_next(approval_id=approval.approval_id).status, "completed")

        self.cache_tool = self.tool
        dataset_dir = self.root / "runs/dataset"
        context = patch("src.agents.wm_dataset_cli._directory", return_value=dataset_dir)
        context.start()
        self.addCleanup(context.stop)
        torch_file = Path(__import__("sys").prefix) / "Lib/site-packages/torch/__init__.py"
        runtime = {"Lib/site-packages/torch/__init__.py": file_hash(torch_file, 1024 * 1024)}

        with patch("src.agents.wm_dataset_cli.runtime_manifest", return_value=(runtime, version("torch"))):
            wm_dataset_cli.create("dataset", "cache", WorkerLimits(300, 8192, 2),
                                  max_input_bytes=32 * 1024 * 1024, confirmed=True)

        self.tool = wm_dataset_cli.open_run("dataset")
        self.manager = wm_dataset_cli.executor(self.tool)
        approval = dataset_fixtures.WmDatasetTest.approval(self)
        transport = dataset_fixtures.WmDatasetTest.transport.__get__(self)

        with patch("src.agents.wm_dataset.run_dataset_worker", side_effect=transport):
            self.assertEqual(self.manager.run_next(approval_id=approval.approval_id).status, "completed")

        plan = load_wm_modules(PLAN)
        runs = dict(zip((step.step_id for step in plan.steps),
                        ("example", "proposal", "commit", "check", "cache", "dataset")))
        def fingerprints():
            return {str(path): _fingerprint(path) for path in self.root.rglob("*") if path.is_file()}

        before = fingerprints()
        protected = ["src.agents.approval_store.ApprovalStore.__init__",
                     "src.agents.executor.RulesExecutor.run_next"]

        with ExitStack() as stack:
            for target in protected:
                stack.enter_context(patch(target, side_effect=AssertionError("조회 중 쓰기 API 금지")))

            reader = WmRunReader()
            verified = reader.read("wm_dataset_check", "dataset")
            self.assertEqual(verified.report["status"], "dataset_batch_passed")
            self.assertEqual(_scope(verified.tool, "validate_contract"), self.manager.scope_for("validate_contract"))
            result = inspect_bindings(plan, runs)
            self.assertEqual(result["status"], "bindings_verified", result)
            self.assertEqual(len(result["steps"]), 6)
            self.assertEqual(before, fingerprints())

            # 완료 보고서가 바뀌면 이미 만든 reader도 성공 결과를 재사용하지 않는다.
            report = verified.attempt / "dataset/dataset_check.json"
            original = report.read_bytes()
            report.write_bytes(b"{}")

            with self.assertRaises(ValueError):
                reader.read("wm_dataset_check", "dataset")

            report.write_bytes(original)

            # 파일이 정상이어도 실제 커밋이 실패 상태라면 새 조회는 거부한다.
            import sqlite3
            from dataclasses import asdict
            database = self.tool.states.directory / "approvals.sqlite3"
            saved_database = database.read_bytes()

            with sqlite3.connect(database) as connection:
                connection.execute("UPDATE task_runs SET status='failed' WHERE scope_json=?", (
                    json.dumps(asdict(_scope(self.tool, "validate_contract")), ensure_ascii=False, sort_keys=True),))

            with self.assertRaises(ValueError):
                WmRunReader().read("wm_dataset_check", "dataset")

            database.write_bytes(saved_database)


if __name__ == "__main__":
    unittest.main()
