"""공통 실행의 한 단계 제한·사용자 승인·범위 변경 거부를 합성 검사한다."""

from dataclasses import replace
from datetime import datetime, timezone
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
from unittest.mock import patch
import unittest

from tests.agents import test_wm_canonical as fixtures
from src.agents.wm_module_plan import load_wm_modules
from src.agents.wm_workflow_execution import dispatch_stage


ROOT = Path(__file__).resolve().parents[2]


class WmWorkflowExecutionTest(unittest.TestCase):
    transport = fixtures.WmCanonicalTest.transport

    def setUp(self):
        """단일 EDA 계획에 합성 작업을 배정한다. 실제 격리 worker는 쓰지 않는다."""
        fixtures.WmCanonicalTest.setUp(self)
        output = redirect_stdout(io.StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)
        value = json.loads((ROOT / "configs/agents/pipelines/wm_development_example.json").read_text(encoding="utf-8"))
        value["steps"] = value["steps"][:1]
        self.pipeline = self.root / "pipeline.json"
        self.pipeline.write_text(json.dumps(value), encoding="utf-8")
        plan = load_wm_modules(self.pipeline)
        self.bindings = self.root / "bindings.json"
        self.bindings.write_text(json.dumps({"schema_version": "wm_run_bindings_v1",
            "pipeline_signature": plan.signature, "runs": {"eda": "example"}}), encoding="utf-8")

    def dispatch(self, stage, command="next", **kwargs):
        return dispatch_stage(self.pipeline, self.bindings, "eda", "example", stage, command, **kwargs)

    def pending(self):
        # 실제 문서 명령처럼 상대경로를 지정해도 엄격한 경로 검사 전에 보정한다.
        previous = Path.cwd()

        try:
            os.chdir(self.root)
            result = dispatch_stage(self.pipeline.name, self.bindings.name, "eda", "example", "preflight", "next")
        finally:
            os.chdir(previous)

        self.assertEqual(result["code"], "STEP_COMPLETED")
        self.assertEqual(self.dispatch("run_eda")["code"], "APPROVAL_REQUIRED")

    def test_approval_execution_and_no_rerun(self):
        """승인 요청·직접 확인·승인 1회 소비·완료 후 재실행 금지를 구분한다."""
        self.pending()

        with patch("src.agents.wm_canonical_cli.os.isatty", return_value=True), \
                patch("src.agents.wm_canonical_cli.secrets.token_hex", return_value="1234abcd"), \
                patch("builtins.input", return_value="WM-1234ABCD"):
            approval = self.dispatch("run_eda", "approve")

        identifier = approval["approval_id"]
        self.assertFalse(approval["worker_started"])

        with patch("src.agents.wm_canonical.run_wm_worker", side_effect=self.transport):
            result = self.dispatch("run_eda", approval_id=identifier)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["code"], "STEP_COMPLETED")
        self.assertEqual(self.manager.approvals.get(identifier).status, "consumed")

        with patch("src.agents.wm_canonical.run_wm_worker") as worker:
            with self.assertRaises(ValueError):
                self.dispatch("run_eda", approval_id=identifier)

        worker.assert_not_called()

    def test_declined_and_noninteractive_approval(self):
        """거절·비대화형 호출은 승인 발급이나 worker 시작으로 이어지지 않는다."""
        self.pending()

        with patch("src.agents.approval_store.ApprovalStore.record_confirmed_approval") as record, \
                patch("src.agents.wm_canonical_cli.os.isatty", return_value=True), \
                patch("builtins.input", return_value="거절"):
            self.assertEqual(self.dispatch("run_eda", "approve")["code"], "APPROVAL_DECLINED")

        record.assert_not_called()

        with patch("src.agents.wm_canonical_cli.os.isatty", return_value=False):
            with self.assertRaises(ValueError):
                self.dispatch("run_eda", "approve")

    def test_wrong_stage_and_approval_are_rejected(self):
        """다른 단계·다른 작업·다른 승인 범위를 실행하지 않는다."""
        with self.assertRaises(ValueError):
            self.dispatch("run_eda")

        self.pending()
        scope = self.manager.scope_for("run_eda")
        valid = self.manager.approvals.record_confirmed_approval(scope)
        other = self.manager.approvals.record_confirmed_approval(replace(scope, config_id="different"))

        with patch.object(self.manager.approvals, "clock", return_value=datetime(2020, 1, 1, tzinfo=timezone.utc)):
            expired = self.manager.approvals.record_confirmed_approval(scope)

        for identifier in (None, "0" * 32, other.approval_id, expired.approval_id):
            with self.assertRaises(ValueError):
                self.dispatch("run_eda", approval_id=identifier)

        self.assertEqual(self.manager.approvals.get(valid.approval_id).status, "pending")

        with self.assertRaises(ValueError):
            dispatch_stage(self.pipeline, self.bindings, "eda", "different", "run_eda", "next")

        state_path = self.directory / "state.json"
        original = state_path.read_bytes()

        for changes in ({"status": "failed_runtime"}, {"cancel_requested": True},
                        {"blockers": ["SECURITY_LOCKED"]}, {"status": "blocked_input"}):
            state = json.loads(original)
            state.update(changes)
            state_path.write_text(json.dumps(state), encoding="utf-8")

            with patch("src.agents.wm_canonical_cli.executor") as factory:
                with self.assertRaises(ValueError):
                    self.dispatch("run_eda", approval_id=valid.approval_id)

            factory.assert_not_called()

        state_path.write_bytes(original)

    def test_changes_during_approval_and_before_lock(self):
        """확인 도중 배정 변경·잠금 전 상태 변경은 승인 저장·실행 전에 거부한다."""
        self.pending()
        original = self.bindings.read_bytes()

        def changed_input(prompt):
            self.bindings.write_bytes(original + b"\n")
            return "WM-1234ABCD"

        with patch("src.agents.wm_canonical_cli.os.isatty", return_value=True), \
                patch("src.agents.wm_canonical_cli.secrets.token_hex", return_value="1234abcd"), \
                patch("builtins.input", side_effect=changed_input), \
                patch("src.agents.approval_store.ApprovalStore.record_confirmed_approval") as record:
            with self.assertRaises(ValueError):
                self.dispatch("run_eda", "approve")

        record.assert_not_called()
        self.bindings.write_bytes(original)
        approval = self.manager.approvals.record_confirmed_approval(self.manager.scope_for("run_eda"))
        from src.agents import wm_canonical_cli
        factory = wm_canonical_cli.executor

        def changed_state(tool):
            manager = factory(tool)
            state = tool.states.load()
            state.cancel_requested = True
            tool.states.save(state, expected_revision=state.revision)
            return manager

        with patch("src.agents.wm_canonical_cli.executor", side_effect=changed_state), \
                patch("src.agents.wm_canonical.run_wm_worker") as worker:
            result = self.dispatch("run_eda", approval_id=approval.approval_id)

        self.assertEqual(result["code"], "WM_SELECTION_CHANGED")
        worker.assert_not_called()
        self.assertEqual(self.manager.approvals.get(approval.approval_id).status, "pending")


class WmCacheExecutionTest(unittest.TestCase):
    """검토 두 개와 실제 캐시 생성 승인이 공통 연결에서도 분리되는지 확인한다."""

    def test_reviews_remain_separate(self):
        from tests.agents import test_wm_cache as cache_fixtures
        from src.agents import wm_cache_cli

        for name in ("transport", "proposal_transport", "commit_transport", "check_transport", "cache_transport"):
            setattr(self, name, getattr(cache_fixtures.WmCacheTest, name).__get__(self))

        cache_fixtures.WmCacheTest.setUp(self)
        pipeline = ROOT / "configs/agents/pipelines/wm_development_example.json"
        plan = load_wm_modules(pipeline)
        bindings = self.root / "bindings.json"
        bindings.write_text(json.dumps({"schema_version": "wm_run_bindings_v1", "pipeline_signature": plan.signature,
            "runs": {"eda": "example", "proposal": "proposal", "split": "commit", "transform": "check", "cache": "cache"}}), encoding="utf-8")

        def dispatch(stage, command="next", **kwargs):
            return dispatch_stage(pipeline, bindings, "cache", "cache", stage, command, **kwargs)

        with redirect_stdout(io.StringIO()):
            for stage in ("preflight", "run_eda", "review_eda", "propose_split", "commit_split", "prepare_data"):
                with patch("src.agents.wm_cache.run_cache_worker") as worker:
                    result = dispatch(stage)

                worker.assert_not_called()

                if stage not in ("review_eda", "commit_split", "prepare_data"):
                    self.assertEqual(result["code"], "STEP_COMPLETED")
                    continue

                self.assertEqual(result["code"], "APPROVAL_REQUIRED")

                with patch("src.agents.wm_cache_cli.os.isatty", return_value=True), \
                        patch("src.agents.wm_cache_cli.secrets.token_hex", return_value="1234abcd"), \
                        patch("builtins.input", return_value="WM-CACHE-1234ABCD"):
                    approval = dispatch(stage, "approve")

                identifier = approval["approval_id"]

                with patch("src.agents.wm_cache.run_cache_worker", side_effect=self.cache_transport) as worker:
                    result = dispatch(stage, approval_id=identifier)

                self.assertEqual(result["code"], "STEP_COMPLETED")
                self.assertEqual(worker.call_count, 1 if stage == "prepare_data" else 0)
                self.assertEqual(wm_cache_cli.executor(self.tool).approvals.get(identifier).status, "consumed")

        self.assertEqual(result["status"], "completed")


if __name__ == "__main__":
    unittest.main()
