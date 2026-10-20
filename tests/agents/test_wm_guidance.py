"""안내만 수행하고 실제 생성·승인·실행은 호출하지 않는지 확인한다."""

from contextlib import closing, ExitStack
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
import unittest
from unittest.mock import patch

from src.agents.wm_module_plan import load_wm_modules, validate_wm_modules
from src.agents.wm_workflow_guidance import recommend_next


PLAN_FILE = Path(__file__).resolve().parents[2] / "configs/agents/pipelines/wm_development_example.json"


class WmGuidanceTest(unittest.TestCase):
    def setUp(self):
        self.plan = load_wm_modules(PLAN_FILE)
        self.runs = {"eda": "example"}

    def test_completed_never_opens_pending_run(self):
        inspected = {"status": "bindings_verified", "steps": [], "next_action": None}

        with patch("src.agents.wm_workflow_guidance.inspect_bindings", return_value=inspected), \
                patch("src.agents.wm_workflow_guidance.load_run_tool", side_effect=AssertionError()):
            result = recommend_next(self.plan, self.runs)

        self.assertEqual(result["guidance"]["action"], "finish")
        self.assertEqual(result["status"], "completed")
        self.assertNotIn("command", result["guidance"])

    def test_unassigned_only_provides_help(self):
        result = recommend_next(self.plan, {})
        self.assertEqual(result["guidance"]["action"], "select_or_create_run")
        self.assertEqual(result["guidance"]["command"], "create --help")
        self.assertFalse(result["worker_started"])
        self.assertTrue(result["advisory_only"])

    def test_changed_or_mismatched_results_block(self):
        for inspected in (
            {"status": "verification_failed", "steps": [], "next_action": "repeat_readonly_verification"},
            {"status": "attention_required", "steps": [],
             "next_action": {"step_id": "eda", "code": "RUN_EVIDENCE_MISMATCH"}},
        ):
            with patch("src.agents.wm_workflow_guidance.inspect_bindings", return_value=inspected):
                result = recommend_next(self.plan, self.runs)

            self.assertEqual(result["status"], "blocked")
            self.assertNotIn("command", result["guidance"])

    def test_invalid_clock_is_rejected(self):
        with self.assertRaises(ValueError):
            recommend_next(self.plan, self.runs, now=datetime(2026, 1, 1))


class WmGuidanceReadonlyTest(unittest.TestCase):
    def setUp(self):
        from tests.agents import test_wm_canonical as fixtures
        fixtures.WmCanonicalTest.setUp(self)
        config = json.loads(PLAN_FILE.read_text(encoding="utf-8"))
        config["steps"] = config["steps"][:1]
        self.plan = validate_wm_modules(config)
        self.runs = {"eda": "example"}

    def advise(self, *, now=None):
        """모든 파일을 전후 비교하고 쓰기 권한을 가진 API는 테스트 중 차단한다."""
        from src.agents.artifacts import _fingerprint

        def fingerprints():
            return {str(path): _fingerprint(path) for path in self.root.rglob("*") if path.is_file()}

        before = fingerprints()
        blocked = ("src.agents.approval_store.ApprovalStore.__init__",
                   "src.agents.executor.RulesExecutor.run_next",
                   "src.agents.wm_canonical_cli.create", "src.agents.wm_canonical_cli.approve",
                   "src.agents.wm_canonical.run_wm_worker")

        with ExitStack() as stack:
            for name in blocked:
                stack.enter_context(patch(name, side_effect=AssertionError("안내 중 실행·쓰기 API 금지")))

            result = recommend_next(self.plan, self.runs, now=now)

        self.assertEqual(before, fingerprints())
        self.assertFalse(result["worker_started"])
        self.assertFalse(result["approval_consumed"])
        return result

    def test_approval_lifecycle_and_no_reexecution(self):
        """준비 → 승인 요청 → 사용자 승인 → 조건부 실행 안내를 구분하고 예약 후에는 중단한다."""
        self.assertEqual(self.advise()["guidance"]["action"], "continue_local_stage")
        self.assertEqual(self.manager.run_next().code, "STEP_COMPLETED")
        self.assertEqual(self.advise()["guidance"]["action"], "request_approval")
        self.assertEqual(self.manager.run_next().code, "APPROVAL_REQUIRED")
        self.assertEqual(self.advise()["guidance"]["action"], "approve_stage")
        scope = self.manager.scope_for("run_eda")
        approval = self.manager.approvals.record_confirmed_approval(scope)
        result = self.advise(now=approval.created_at + timedelta(seconds=1))
        self.assertEqual(result["guidance"]["action"], "execute_approved_stage")
        self.assertTrue(result["guidance"]["approval_id_required"])
        self.assertNotIn(approval.approval_id, json.dumps(result))

        # 만료된 승인을 조회하더라도 DB status를 expired로 자동 갱신하지 않는다.
        self.assertEqual(self.advise(now=approval.expires_at)["guidance"]["action"], "approve_stage")

        with closing(
            sqlite3.connect((self.directory / "approvals.sqlite3").as_uri() + "?mode=ro", uri=True)
        ) as connection, connection:
            self.assertEqual(connection.execute("SELECT status FROM approvals WHERE approval_id=?",
                                                (approval.approval_id,)).fetchone()[0], "pending")

        # 다른 범위의 유효한 승인은 현재 작업에 사용할 수 없다.
        self.manager.approvals.record_confirmed_approval(replace(scope, config_id="different"))
        self.assertEqual(self.advise(now=approval.expires_at)["guidance"]["action"], "approve_stage")

        # 예약은 합성 fixture에서만 생성하며 실제 worker를 실행하지 않는다.
        self.manager.approvals.consume_and_reserve(approval.approval_id, scope)
        result = self.advise()
        self.assertEqual(result["guidance"]["action"], "wait_or_review_task")
        self.assertEqual(result["status"], "blocked")

    def test_cancel_security_failure_and_stale_code(self):
        """중단 상태·설정 변경을 우회하는 실행 안내를 하지 않는다."""
        state_path = self.directory / "state.json"
        selection_path = self.directory / "selection.json"
        saved_state = state_path.read_bytes()
        saved_selection = selection_path.read_bytes()
        cases = (
            ({"status": "failed_runtime"}, "review_local_report"),
            ({"status": "blocked_input"}, "review_local_report"),
            ({"status": "cancelled"}, "stop_cancelled"),
            ({"cancel_requested": True}, "stop_cancelled"),
            ({"blockers": ["SECURITY_LOCKED"]}, "security_review"),
        )

        for changes, action in cases:
            state = json.loads(saved_state)
            state.update(changes)
            state_path.write_text(json.dumps(state), encoding="utf-8")
            result = self.advise()
            self.assertEqual(result["guidance"]["action"], action)
            self.assertEqual(result["status"], "blocked")

        state_path.write_bytes(saved_state)
        marker = self.directory / "CANCEL_REQUESTED"
        marker.write_text("cancel", encoding="utf-8")
        self.assertEqual(self.advise()["guidance"]["action"], "stop_cancelled")
        marker.unlink()
        selection = json.loads(saved_selection)
        selection["code_id"] = "0" * 64
        selection_path.write_text(json.dumps(selection), encoding="utf-8")
        self.assertEqual(self.advise()["guidance"]["action"], "review_local_evidence")
        selection_path.write_bytes(saved_selection)

    def test_missing_previous_result_blocks(self):
        self.manager.run_next()
        stage = self.directory / "attempts"
        artifact = next(stage.glob("*/AGENT_COMPLETE.json"))
        saved = artifact.read_bytes()
        artifact.write_bytes(b"{}")
        self.assertEqual(self.advise()["guidance"]["action"], "review_local_evidence")
        artifact.write_bytes(saved)


if __name__ == "__main__":
    unittest.main()
