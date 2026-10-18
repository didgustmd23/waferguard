"""실제 원본 없이 등록된 합성 도구로 실행·중단·재개를 검증한다."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.agents.approval_store import ApprovalStore
from src.agents.contracts import ToolResult
from src.agents.executor import (
    ExecutionPlan, ExecutionStep, RegisteredTool, RulesExecutor, _run_lock,
)
from src.agents.state import LocalStateStore


class SimulatedCrash(BaseException):
    """예외 복구보다 먼저 프로세스가 종료된 상황을 합성한다."""


class SyntheticTool:
    """실행 횟수를 기록하고 독립 합성 파일 하나를 만드는 도구."""

    def __init__(self):
        self.calls = 0

    def execute(self, output, arguments):
        self.calls += 1
        (output / "result.json").write_text(json.dumps(arguments), encoding="utf-8")

        return ToolResult(status="passed")

    def validate(self, output, arguments):
        return json.loads((output / "result.json").read_text(encoding="utf-8")) == arguments


class ExecutorTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)

        self.root = Path(temporary.name)
        self.states = LocalStateStore(self.root / "run")
        self.states.create("test_operator", policy_version="policy_v1")
        self.approvals = ApprovalStore(self.root / "approvals.sqlite3")
        self.preflight = SyntheticTool()
        self.audit = SyntheticTool()
        self.plan = ExecutionPlan(
            source_id="synthetic_source",
            config_id="synthetic_config",
            policy_version="policy_v1",
            steps=(
                ExecutionStep("preflight", "run_preflight"),
                ExecutionStep("audit", "run_audit", '{"test": true}'),
            ),
        )

    def executor(self, *, plan=None, protected=False, preflight=None):
        """테스트에 필요한 고정 도구만 등록한다."""
        tool = preflight or self.preflight

        return RulesExecutor(
            self.states, self.approvals, plan or self.plan,
            (
                RegisteredTool("run_preflight", tool.execute, tool.validate, protected),
                RegisteredTool("run_audit", self.audit.execute, self.audit.validate),
            ),
        )

    def test_fixed_steps_and_completion(self):
        """단계 순서대로 한 번씩 실행하고 완료 후 재실행하지 않는다."""
        executor = self.executor()

        self.assertEqual(executor.run_next().code, "STEP_COMPLETED")
        self.assertEqual(self.states.load().current_stage, "audit")
        self.assertEqual(executor.run_next().status, "completed")
        self.assertEqual(executor.run_next().code, "RUN_NOT_EXECUTABLE")
        self.assertEqual((self.preflight.calls, self.audit.calls), (1, 1))

    def test_approval_wait_and_resume(self):
        """승인 전에는 실행하지 않고 정확한 승인을 한 번 소비한 뒤 진행한다."""
        executor = self.executor(protected=True)

        self.assertEqual(executor.run_next().code, "APPROVAL_REQUIRED")
        self.assertEqual(self.preflight.calls, 0)

        scope = executor.scope_for("preflight")
        record = self.approvals.record_confirmed_approval(scope)
        result = executor.run_next(approval_id=record.approval_id)

        self.assertEqual(result.code, "STEP_COMPLETED")
        self.assertEqual(self.preflight.calls, 1)
        self.assertEqual(self.approvals.get(record.approval_id).status, "consumed")

    def test_wrong_approval_locks_execution(self):
        """다른 설정에 발급된 승인으로는 도구를 실행하지 않는다."""
        executor = self.executor(protected=True)
        executor.run_next()
        scope = replace(executor.scope_for("preflight"), config_id="other")
        record = self.approvals.record_confirmed_approval(scope)

        result = executor.run_next(approval_id=record.approval_id)

        self.assertEqual(result.code, "APPROVAL_SCOPE_MISMATCH")
        self.assertEqual(self.preflight.calls, 0)
        self.assertIn("SECURITY_LOCKED", self.states.load().blockers)

    def test_mandatory_ingest_approval(self):
        """도구 등록 옵션이 False여도 ingestion 승인을 생략하지 않는다."""
        plan = replace(self.plan, steps=(
            ExecutionStep("preflight", "run_preflight"),
            ExecutionStep("ingest", "run_ingest"),
        ))
        executor = RulesExecutor(
            self.states, self.approvals, plan,
            (
                RegisteredTool("run_preflight", self.preflight.execute, self.preflight.validate),
                RegisteredTool("run_ingest", self.audit.execute, self.audit.validate),
            ),
        )
        executor.run_next()

        self.assertEqual(executor.run_next().code, "APPROVAL_REQUIRED")
        self.assertEqual(self.audit.calls, 0)

    def test_cancel_and_general_blocker(self):
        """취소 요청과 일반 blocker도 실제 도구 실행을 막는다."""
        executor = self.executor()
        state = self.states.load()
        state.cancel_requested = True
        self.states.save(state, expected_revision=state.revision)

        self.assertEqual(executor.run_next().code, "RUN_CANCEL_REQUESTED")
        self.assertEqual(self.preflight.calls, 0)

        state = self.states.load()
        state.cancel_requested = False
        state.blockers = ["INPUT_REVIEW_REQUIRED"]
        self.states.save(state, expected_revision=state.revision)

        self.assertEqual(executor.run_next().code, "BLOCKERS_PRESENT")
        self.assertEqual(self.preflight.calls, 0)

    def test_changed_plan_and_skipped_stage(self):
        """실행 중 계획 변경을 차단한다."""
        self.executor().run_next()
        changed = replace(self.plan, config_id="changed")

        result = self.executor(plan=changed).run_next()

        self.assertEqual(result.code, "PLAN_MISMATCH")
        self.assertEqual(self.audit.calls, 0)

    def test_state_cannot_skip_prerequisite(self):
        """상태 파일의 현재 단계만 바꿔 선행 작업을 생략하지 못한다."""
        state = self.states.load()
        state.current_stage = "audit"
        self.states.save(state, expected_revision=state.revision)

        self.assertEqual(self.executor().run_next().code, "PREREQUISITE_NOT_COMMITTED")
        self.assertEqual(self.audit.calls, 0)

    def test_failed_tool_retains_partial_output(self):
        """실패한 도구의 부분 파일은 보존하고 완료 표식은 만들지 않는다."""
        def fail(output, arguments):
            (output / "partial.txt").write_text("synthetic", encoding="utf-8")

            raise ValueError("CANARY_RAW_ERROR")

        self.preflight.execute = fail
        executor = self.executor()

        self.assertEqual(executor.run_next().code, "TOOL_RUNTIME_FAILED")
        self.assertEqual(len(list(self.states.directory.rglob("partial.txt"))), 1)
        self.assertEqual(list(self.states.directory.rglob("AGENT_COMPLETE.json")), [])
        self.assertNotIn("CANARY_RAW_ERROR", (self.states.directory / "events.jsonl").read_text())

    def test_content_validation_failure(self):
        """도구의 passed 응답만으로 완료를 인정하지 않는다."""
        self.preflight.validate = lambda output, arguments: False

        self.assertEqual(self.executor().run_next().code, "TOOL_CONTENT_INVALID")
        self.assertEqual(list(self.states.directory.rglob("AGENT_COMPLETE.json")), [])

    def test_audit_failure_prevents_tool_execution(self):
        """감사 기록 실패 시 새로운 도구 실행을 차단한다."""
        executor = self.executor()

        with patch("src.agents.executor.append_event", side_effect=RuntimeError("synthetic")):
            result = executor.run_next()

        self.assertEqual(result.code, "EXECUTOR_RUNTIME_FAILED")
        self.assertEqual(self.preflight.calls, 0)
        self.assertIn("SECURITY_LOCKED", self.states.load().blockers)

    def test_verified_result_recovers_without_refit(self):
        """완료 표식과 verified 이력이 있으면 도구를 다시 실행하지 않는다."""
        executor = self.executor()

        with patch.object(executor, "_advance", side_effect=SimulatedCrash):
            with self.assertRaises(SimulatedCrash):
                executor.run_next()

        result = self.executor().run_next()

        self.assertEqual(result.code, "ARTIFACT_REUSED")
        self.assertEqual(self.preflight.calls, 1)
        self.assertEqual(self.states.load().current_stage, "audit")

    def test_committed_result_recovers_after_state_save_crash(self):
        """DB 커밋 후 상태 저장 전 종료돼도 완료 도구를 재실행하지 않는다."""
        executor = self.executor()
        original_save = self.states.save

        def crash_before_advance(state, **kwargs):
            if state.current_stage == "audit":
                raise SimulatedCrash()

            return original_save(state, **kwargs)

        with patch.object(self.states, "save", side_effect=crash_before_advance):
            with self.assertRaises(SimulatedCrash):
                executor.run_next()

        self.assertEqual(self.executor().run_next().code, "ARTIFACT_REUSED")
        self.assertEqual(self.preflight.calls, 1)

    def test_cancel_during_tool_does_not_publish_result(self):
        """도구 실행 중 들어온 취소는 완료 표식·다음 단계 진행을 막는다."""
        original_execute = self.preflight.execute

        def execute_then_cancel(output, arguments):
            result = original_execute(output, arguments)
            latest = self.states.load()
            latest.cancel_requested = True
            self.states.save(latest, expected_revision=latest.revision)

            return result

        self.preflight.execute = execute_then_cancel
        result = self.executor().run_next()

        self.assertEqual(result.code, "STATE_CHANGED_AFTER_EXECUTION")
        self.assertEqual(list(self.states.directory.rglob("AGENT_COMPLETE.json")), [])
        self.assertEqual(self.audit.calls, 0)

    def test_running_result_never_reexecutes_automatically(self):
        """실행 중 종료된 작업은 부분 결과가 있어도 자동 재실행하지 않는다."""
        def crash(output, arguments):
            raise SimulatedCrash()

        self.preflight.execute = crash
        executor = self.executor()

        with self.assertRaises(SimulatedCrash):
            executor.run_next()

        self.assertEqual(self.executor().run_next().code, "TASK_REQUIRES_REVIEW")

    def test_modified_prerequisite_prevents_next_tool(self):
        """이전 단계 결과가 바뀌면 다음 단계를 실행하지 않는다."""
        executor = self.executor()
        executor.run_next()
        state = self.states.load()
        output = self.states.directory / state.artifact_refs["preflight"]
        (output / "result.json").write_text("changed", encoding="utf-8")

        self.assertEqual(executor.run_next().code, "PREREQUISITE_ARTIFACT_INVALID")
        self.assertEqual(self.audit.calls, 0)

    def test_run_lock_blocks_competing_executor(self):
        """같은 실행 폴더의 OS 잠금을 다른 실행기가 얻지 못한다."""
        executor = self.executor()

        with _run_lock(self.states.directory) as acquired:
            self.assertTrue(acquired)
            self.assertEqual(executor.run_next().code, "EXECUTOR_BUSY")

        self.assertEqual(self.preflight.calls, 0)

    def test_invalid_plan_and_registry(self):
        """잘못된 단계 순서·EDA 검토 생략·미등록 도구를 거부한다."""
        with self.assertRaises(ValueError):
            replace(self.plan, steps=tuple(reversed(self.plan.steps)))

        with self.assertRaises(ValueError):
            replace(self.plan, steps=(
                ExecutionStep("preflight", "run_preflight"),
                ExecutionStep("commit_split", "commit_split"),
            ))

        with self.assertRaises(ValueError):
            RegisteredTool("unknown", self.audit.execute, self.audit.validate)


if __name__ == "__main__":
    unittest.main()
