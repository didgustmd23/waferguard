"""등록 도구를 고정 계획에 따라 실행하는 로컬 rules_only 실행기."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Callable

from .approval_store import ApprovalStore
from .approvals import ApprovalScope
from .artifacts import (
    ArtifactContext, _fingerprint, _is_link, _root, check_artifacts, complete_artifacts,
)
from .audit import append_event
from .contracts import ToolResult
from .policy import ACTION_IDS, ALLOWED_ACTIONS_BY_STAGE, decide_action
from .state import LocalStateStore
from .workflow import STAGE_ORDER, WORKFLOW_TEMPLATES


# 도구 등록에서 승인이 불필요하다고 지정해도 민감 단계의 승인을 생략하지 않는다.
MANDATORY_APPROVAL_ACTIONS = frozenset({
    "derive_preparation",
    "apply_preparation",
    "run_ingest", "run_pipeline", "review_eda", "commit_split", "prepare_data", "prepare_cache",
})


def _json(value):
    """로컬 계획과 인자를 비교할 수 있도록 일관된 JSON으로 표현한다."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _digest(value):
    """식별값을 코드에서 자동 생성한다. 인증 서명으로 사용하지 않는다."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExecutionStep:
    """로컬 계획에 고정된 단계·도구·인자. 외부 LLM 입력으로 사용하지 않는다."""

    stage: str
    action_id: str
    arguments_json: str = "{}"

    def __post_init__(self):
        if self.action_id not in ALLOWED_ACTIONS_BY_STAGE.get(self.stage, ()):
            raise ValueError("실행 단계와 도구 ID가 일치하지 않습니다.")

        arguments = json.loads(self.arguments_json)

        if not isinstance(arguments, dict):
            raise ValueError("도구 인자는 JSON 객체여야 합니다.")

        object.__setattr__(self, "arguments_json", _json(arguments))


@dataclass(frozen=True)
class ExecutionPlan:
    """한 실행에서 변경 없이 사용할 로컬 계획."""

    source_id: str
    config_id: str
    policy_version: str
    steps: tuple[ExecutionStep, ...]
    profile_id: str = ""
    workflow_id: str = ""

    def __post_init__(self):
        identifiers = (self.source_id, self.config_id, self.policy_version)

        if any(not isinstance(value, str) or not value.strip() for value in identifiers):
            raise ValueError("실행 계획 식별값이 필요합니다.")

        if not isinstance(self.steps, tuple) or not self.steps:
            raise ValueError("실행 계획에는 고정된 단계 목록이 필요합니다.")

        if any(not isinstance(step, ExecutionStep) for step in self.steps):
            raise TypeError("실행 단계는 ExecutionStep이어야 합니다.")

        stages = [step.stage for step in self.steps]
        indices = [STAGE_ORDER.index(stage) for stage in stages]

        if stages[0] != "preflight" or indices != sorted(set(indices)):
            raise ValueError("실행 단계는 preflight부터 중복 없이 순서대로 구성해야 합니다.")

        if self.profile_id or self.workflow_id:
            expected = WORKFLOW_TEMPLATES.get(self.workflow_id)

            if not self.profile_id or expected is None or tuple(stages) != expected:
                raise ValueError("프로파일 실행 계획이 등록된 템플릿과 일치하지 않습니다.")

        # 분할안 작성 전에 EDA 실행과 사람의 검토 단계를 반드시 포함한다.
        if any(stage in stages for stage in ("propose_split", "commit_split")):
            if not {"run_eda", "review_eda", "propose_split"}.issubset(stages):
                raise ValueError("분할 계획에는 EDA·검토·분할 제안 단계가 필요합니다.")

        if any(stage in stages for stage in ("prepare_data", "prepare_cache")):
            if "commit_split" not in stages:
                raise ValueError("학습 입력 준비 전에 분할 확정 단계가 필요합니다.")

        if "publish" in stages and "validate_contract" not in stages:
            raise ValueError("산출물 게시 전에 입력 계약 검증 단계가 필요합니다.")

    @property
    def plan_id(self):
        """단계·인자 변경도 반영하는 실행 계획 식별값."""
        return _digest(_json(self.to_dict()))

    def to_dict(self):
        """기존 비프로파일 계획의 저장 형식과 식별값을 보존한다."""
        result = asdict(self)

        if not self.profile_id and not self.workflow_id:
            result.pop("profile_id")
            result.pop("workflow_id")

        return result


@dataclass(frozen=True)
class RegisteredTool:
    """신뢰하는 애플리케이션 코드가 등록하는 실행·내용 검증 함수."""

    action_id: str
    execute: Callable[[Path, dict], ToolResult]
    validate: Callable[[Path, dict], bool]
    requires_approval: bool = False
    dependency_stage: str | None = None

    def __post_init__(self):
        if self.action_id not in ACTION_IDS:
            raise ValueError("등록되지 않은 도구 ID입니다.")

        if not callable(self.execute) or not callable(self.validate):
            raise TypeError("도구 실행·검증 함수가 필요합니다.")

        if type(self.requires_approval) is not bool:
            raise TypeError("승인 필요 여부는 bool이어야 합니다.")

        # 검토된 의존 도구에만 검증된 선행 폴더를 별도 인자로 전달한다.
        # JSON 인자로 파일 경로를 주입하는 기능은 만들지 않는다.
        allowed_dependencies = {
            ("commit_split", "propose_split"), ("prepare_data", "commit_split"),
            ("validate_contract", "prepare_data"), ("render_report", "validate_contract"),
        }

        if self.dependency_stage is not None and (self.action_id, self.dependency_stage) not in allowed_dependencies:
            raise ValueError("등록되지 않은 도구 의존 관계입니다.")


@dataclass(frozen=True)
class ExecutionResult:
    """실행기의 일반 상태와 고정 결과 코드. 상세 데이터는 포함하지 않는다."""

    status: str
    code: str
    stage: str


@contextmanager
def _run_lock(directory):
    """한 실행 폴더에 대한 동시 실행을 OS 파일 잠금으로 차단한다."""
    path = directory / "executor.lock"

    if path.exists() and _is_link(path):
        raise ValueError("실행 잠금 경로에 링크를 사용할 수 없습니다.")

    # 잠금 파일은 삭제하지 않는다. 삭제·재생성에 의한 잠금 우회를 피한다.
    with path.open("a+b") as stream:
        if stream.seek(0, os.SEEK_END) == 0:
            stream.write(b"0")
            stream.flush()

        stream.seek(0)
        acquired = False

        try:
            try:
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

                acquired = True
            except OSError:
                pass

            yield acquired
        finally:
            if acquired:
                stream.seek(0)

                if os.name == "nt":
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


# ==========================================
# 규칙 기반 실행기
# - 한 번의 run_next 호출은 현재 단계 하나만 처리한다.
# - 정책·선행 작업·승인·취소를 실제 도구 실행 직전에 검사한다.
# - 예약만 남거나 실행 여부가 불명확하면 자동 재실행하지 않는다.
# - verified/committed 작업은 파일·내용 검증 후 재사용한다.
# - 원본 경로와 도메인별 처리는 신뢰하는 등록 도구에서 담당한다.
# ==========================================
class RulesExecutor:
    """로컬 합성 도구부터 연결하는 실행 관리자. OS worker 격리는 후속 범위다."""

    def __init__(self, state_store, approval_store, plan, tools):
        if not isinstance(state_store, LocalStateStore):
            raise TypeError("로컬 상태 저장소가 필요합니다.")

        if not isinstance(approval_store, ApprovalStore) or not isinstance(plan, ExecutionPlan):
            raise TypeError("승인 저장소와 실행 계획이 필요합니다.")

        registered = tuple(tools)

        if any(not isinstance(tool, RegisteredTool) for tool in registered):
            raise TypeError("등록 도구 목록이 올바르지 않습니다.")

        self.tools = {tool.action_id: tool for tool in registered}

        if len(self.tools) != len(registered):
            raise ValueError("도구 ID를 중복 등록할 수 없습니다.")

        if any(step.action_id not in self.tools for step in plan.steps):
            raise ValueError("실행 계획에 필요한 도구가 등록되지 않았습니다.")

        self.states = state_store
        self.approvals = approval_store
        self.plan = plan
        self.directory = _root(state_store.directory)
        self.events = self.directory / "events.jsonl"

        for step in plan.steps:
            dependency = self.tools[step.action_id].dependency_stage

            if dependency is not None and dependency not in [item.stage for item in plan.steps[:plan.steps.index(step)]]:
                raise ValueError("등록 도구의 선행 단계가 실행 계획에 없습니다.")

    def scope_for(self, stage):
        """승인 화면 연결 시 사용할 현재 계획의 정확한 작업 범위를 구성한다."""
        state = self.states.load()
        step = next((item for item in self.plan.steps if item.stage == stage), None)

        if step is None:
            raise ValueError("실행 계획에 없는 단계입니다.")

        return self._scope(state, step)

    def _scope(self, state, step):
        """인자와 계획 식별값은 자동 계산하며 작업 대상은 실행·단계에 결합한다."""
        arguments_id = _digest(step.arguments_json)
        dependency = self._dependency(state, step)

        if dependency is not None:
            # 완료 표식은 파일 목록·해시를 포함한다. 승인 후 제안이 바뀌면
            # 이전 승인을 사용할 수 없도록 그 식별값까지 승인 범위에 묶는다.
            arguments_id = _digest(_json({
                "arguments": json.loads(step.arguments_json),
                "dependency": _fingerprint(dependency / "AGENT_COMPLETE.json"),
            }))

        return ApprovalScope(
            actor_id=state.actor_id, run_id=state.run_id,
            stage=step.stage, tool_id=step.action_id,
            target_id=f"{state.run_id}:{step.stage}",
            arguments_id=arguments_id,
            source_id=self.plan.source_id, config_id=self.plan.config_id,
            plan_id=self.plan.plan_id, policy_version=self.plan.policy_version,
        )

    def _dependency(self, state, step):
        """DB의 커밋 이력으로만 선행 산출물을 찾는다. 상태 파일 경로를 신뢰하지 않는다."""
        stage = self.tools[step.action_id].dependency_stage

        if stage is None:
            return None

        previous = next(item for item in self.plan.steps if item.stage == stage)
        task = self.approvals.get_task(self._scope(state, previous))

        if task is None or task["status"] != "committed":
            raise ValueError("선행 단계의 커밋 이력이 필요합니다.")

        output = self._output(task)

        if not check_artifacts(output, self._context(previous)).reusable:
            raise ValueError("선행 산출물의 정합성 검증에 실패했습니다.")

        return output

    def _call_tool(self, callback, state, step, output):
        """기존 두 인자 도구는 보존하고 등록된 의존 도구에만 세 번째 인자를 제공한다."""
        arguments = json.loads(step.arguments_json)
        dependency = self._dependency(state, step)

        if dependency is None:
            return callback(output, arguments)

        return callback(output, arguments, dependency)

    def _context(self, step):
        """파일 정합성 검사를 현재 실행 계획에 연결한다."""
        return ArtifactContext(
            self.plan.source_id, self.plan.config_id, self.plan.plan_id,
            self.plan.policy_version, step.action_id,
        )

    def _event(self, state, event_type, code):
        """감사 실패는 예외로 전파하여 후속 도구 실행을 막는다."""
        if self.events.exists() and _is_link(self.events):
            raise ValueError("감사 기록 경로에 링크를 사용할 수 없습니다.")

        append_event(
            self.events, event_type, code,
            stage=state.current_stage, revision=state.revision,
        )

    def _save(self, state):
        self.states.save(state, expected_revision=state.revision)
        self._event(state, "state_saved", "STATE_SAVED")

    def _stop(self, state, status, code, *, lock=False):
        """상세 오류 원문 없이 중단 상태와 고정 사유를 기록한다."""
        state.status = status

        if lock and "SECURITY_LOCKED" not in state.blockers:
            state.blockers.append("SECURITY_LOCKED")

        event_type = "security_lock" if lock else "action_denied"
        self._event(state, event_type, code)
        self._save(state)

        return ExecutionResult(status, code, state.current_stage)

    def _bind_plan(self, state):
        """최초 계획을 로컬에 고정하여 재개 시 계획 바꿔치기를 탐지한다."""
        path = self.directory / "execution_plan.json"
        payload = _json({
            "run_id": state.run_id,
            "actor_id": state.actor_id,
            "plan": self.plan.to_dict(),
        })

        if path.exists():
            if _is_link(path) or path.read_text(encoding="utf-8") != payload:
                return False
        else:
            with path.open("x", encoding="utf-8") as stream:
                stream.write(payload)

        return state.policy_version == self.plan.policy_version

    def _output(self, task):
        """DB 식별값을 경로로 사용하기 전에 내부 UUID 형식을 검사한다."""
        if not re.fullmatch(r"[a-f0-9]{32}", task["task_id"]):
            raise ValueError("작업 식별값 형식이 올바르지 않습니다.")

        return self.directory / "attempts" / task["task_id"]

    def _advance(self, state, step, task, output, *, reused):
        """검증된 작업의 완료 이력과 상태 파일을 연결한다."""
        scope = self._scope(state, step)

        if task["status"] == "verified":
            self.approvals.transition_task(
                scope, task["task_id"], expected="verified", status="committed",
            )

        state.artifact_refs[step.stage] = output.relative_to(self.directory).as_posix()
        index = self.plan.steps.index(step)

        if index + 1 == len(self.plan.steps):
            state.status = "completed"
        else:
            state.current_stage = self.plan.steps[index + 1].stage
            state.status = "running"

        self._save(state)

        code = "ARTIFACT_REUSED" if reused else "STEP_COMPLETED"

        return ExecutionResult(state.status, code, step.stage)

    def run_next(self, *, approval_id=None):
        """다음 한 단계를 실행한다. 승인 발급과 자동 재시도는 수행하지 않는다."""
        _root(self.directory)

        if _is_link(self.states.path):
            raise ValueError("실행 상태 경로에 링크를 사용할 수 없습니다.")

        with _run_lock(self.directory) as acquired:
            if not acquired:
                return ExecutionResult("running", "EXECUTOR_BUSY", "preflight")

            state = self.states.load()

            try:
                return self._run_step(state, approval_id)
            except Exception:
                # DB·감사·파일 저장 실패 뒤에는 신규 도구를 실행하지 않는다.
                # 상세 예외 메시지를 상태나 외부 응답에 복사하지 않는다.
                state.status = "failed_runtime"

                if "SECURITY_LOCKED" not in state.blockers:
                    state.blockers.append("SECURITY_LOCKED")

                try:
                    self.states.save(state, expected_revision=state.revision)
                except Exception:
                    pass

                return ExecutionResult(
                    "failed_runtime", "EXECUTOR_RUNTIME_FAILED", state.current_stage,
                )

    def _run_step(self, state, approval_id):
        """정책·이력·승인 검사 후 등록된 도구만 실행한다."""
        if not self._bind_plan(state):
            return self._stop(state, "blocked_input", "PLAN_MISMATCH", lock=True)

        step = next((item for item in self.plan.steps if item.stage == state.current_stage), None)

        if step is None:
            return self._stop(state, "blocked_input", "STATE_PLAN_MISMATCH", lock=True)

        # 승인 대기 재개는 정책 검사용 복사본만 running으로 바꾼다.
        # 실제 상태 변경·도구 실행은 승인 소비 또는 검증된 재사용 이후에만 가능하다.
        candidate = replace(state, status="running") if state.status == "awaiting_approval" else state
        decision = decide_action(candidate, step.action_id)

        if not decision.allowed:
            self._event(state, "action_denied", decision.code)

            return ExecutionResult(state.status, decision.code, step.stage)

        if state.blockers:
            return self._stop(state, "blocked_input", "BLOCKERS_PRESENT")

        # 상태 파일의 단계만 건너뛰어도 선행 작업 이력이 없으면 실행하지 않는다.
        for previous in self.plan.steps[:self.plan.steps.index(step)]:
            previous_task = self.approvals.get_task(self._scope(state, previous))

            if previous_task is None or previous_task["status"] != "committed":
                return self._stop(state, "blocked_input", "PREREQUISITE_NOT_COMMITTED", lock=True)

            previous_output = self._output(previous_task)
            check = check_artifacts(previous_output, self._context(previous))

            if not check.reusable:
                return self._stop(state, "blocked_input", "PREREQUISITE_ARTIFACT_INVALID", lock=True)

        scope = self._scope(state, step)
        tool = self.tools[step.action_id]
        task = self.approvals.get_task(scope)

        if task is not None:
            return self._reuse(state, step, tool, task)

        needs_approval = tool.requires_approval or step.action_id in MANDATORY_APPROVAL_ACTIONS

        if needs_approval and approval_id is None:
            state.status = "awaiting_approval"
            self._event(state, "approval_requested", "APPROVAL_REQUIRED")
            self._save(state)

            return ExecutionResult(state.status, "APPROVAL_REQUIRED", step.stage)

        self._event(state, "action_allowed", "ACTION_ALLOWED")

        if needs_approval:
            reservation = self.approvals.consume_and_reserve(approval_id, scope)
        else:
            reservation = self.approvals.reserve_task(scope)

        if reservation.code != "TASK_RESERVED":
            return self._stop(state, "blocked_input", reservation.code, lock=True)

        if needs_approval:
            self._event(state, "approval_consumed", "APPROVAL_CONSUMED")

        task = {"task_id": reservation.reservation_id, "status": "reserved"}
        output = self._output(task)
        output.parent.mkdir(exist_ok=True)
        _root(output.parent)
        output.mkdir(exist_ok=False)

        # 도구 시작 직전에 최신 취소·revision·잠금 상태를 다시 검사한다.
        latest = self.states.load()

        if latest.revision != state.revision or latest.cancel_requested or latest.blockers:
            return self._stop(latest, "blocked_input", "STATE_CHANGED_BEFORE_EXECUTION")

        self.approvals.transition_task(scope, task["task_id"], expected="reserved", status="running")
        state.status = "running"
        self._save(state)

        return self._execute(state, step, tool, task, output)

    def _execute(self, state, step, tool, task, output):
        """도구 실행·내용 검증·완료 표식·작업 커밋을 순서대로 수행한다."""
        scope = self._scope(state, step)

        try:
            result = self._call_tool(tool.execute, state, step, output)
        except Exception:
            self.approvals.transition_task(scope, task["task_id"], expected="running", status="failed")

            return self._stop(state, "failed_runtime", "TOOL_RUNTIME_FAILED")

        if not isinstance(result, ToolResult) or result.status != "passed":
            self.approvals.transition_task(scope, task["task_id"], expected="running", status="failed")

            code = "TOOL_RESULT_NOT_PASSED"
            status = "blocked_input"

            if isinstance(result, ToolResult):
                # 격리 worker가 종료 확인 후 반환하는 고정 상태만 반영한다.
                if result.errors == ("WORKER_CANCELLED",):
                    state.cancel_requested = True
                    return self._stop(state, "cancelled", "WORKER_CANCELLED")

                code = {
                    "review_required": "TOOL_REVIEW_REQUIRED",
                    "blocked_input": "TOOL_INPUT_BLOCKED",
                    "failed_runtime": "TOOL_RUNTIME_FAILED",
                }.get(result.status, code)

                if result.status == "failed_runtime":
                    status = "failed_runtime"

                    if result.errors in (("WORKER_TIMED_OUT",), ("WORKER_RESOURCE_EXCEEDED",), ("WORKER_FAILED",)):
                        code = result.errors[0]

            return self._stop(state, status, code)

        if self._call_tool(tool.validate, state, step, output) is not True:
            self.approvals.transition_task(scope, task["task_id"], expected="running", status="failed")

            return self._stop(state, "blocked_input", "TOOL_CONTENT_INVALID")

        latest = self.states.load()

        if latest.cancel_requested or latest.revision != state.revision or latest.blockers:
            return self._stop(latest, "blocked_input", "STATE_CHANGED_AFTER_EXECUTION")

        complete_artifacts(output, self._context(step))

        if not check_artifacts(output, self._context(step)).reusable:
            return self._stop(state, "blocked_input", "ARTIFACT_CHECK_FAILED", lock=True)

        self.approvals.transition_task(scope, task["task_id"], expected="running", status="verified")
        task["status"] = "verified"

        return self._advance(state, step, task, output, reused=False)

    def _reuse(self, state, step, tool, task):
        """검증 이력이 있는 결과만 재사용하고 불명확한 작업은 중단한다."""
        if task["status"] not in {"verified", "committed"}:
            return self._stop(state, "blocked_input", "TASK_REQUIRES_REVIEW")

        output = self._output(task)
        check = check_artifacts(output, self._context(step))

        if not check.reusable:
            return self._stop(state, "blocked_input", check.code, lock=True)

        if self._call_tool(tool.validate, state, step, output) is not True:
            return self._stop(state, "blocked_input", "TOOL_CONTENT_INVALID", lock=True)

        self._event(state, "action_allowed", "ARTIFACT_REUSED")

        return self._advance(state, step, task, output, reused=True)
