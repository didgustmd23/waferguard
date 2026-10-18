"""입력 형식에 의존하지 않는 파이프라인 승인 기록 실행기."""

# Shared approval plumbing; input-format rules stay in each adapter binding.
import json
from typing import Protocol

from .approval_store import ApprovalStore
from .artifacts import _root
from .contracts import ToolResult
from .executor import RegisteredTool, RulesExecutor
from .state import LocalStateStore
from .pipeline_binding_registry import build_pipeline_execution
from .worker_contract import _decode


class PipelineBinding(Protocol):
    """각 입력 어댑터 바인더가 공통 승인 계층에 제공할 계약 표면."""

    policy_version: str

    @property
    def source_id(self): ...

    @property
    def binding_id(self): ...

    def to_dict(self): ...

    def execution_plan(self): ...

    def approval_summary(self): ...


def _preflight(output, arguments):
    """고정 실행 계약이 다시 확인된 뒤 사전 검사 완료 표식을 남긴다."""
    if set(arguments) != {"binding_id", "binding"}:
        return ToolResult("blocked_input", errors=("INVALID_BINDING_ARGUMENTS",))

    payload = {
        "schema_version": "pipeline_preflight_v1",
        "binding_id": arguments["binding_id"],
    }
    (output / "preflight.json").write_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    return ToolResult("passed")


def _validate_preflight(output, arguments):
    """사전 검사 산출물이 현재 실행 계약 ID와 일치하는지 확인한다."""
    path = output / "preflight.json"

    if not path.is_file() or path.is_symlink():
        return False

    value = _decode(path.read_text(encoding="utf-8"))
    return value == {
        "schema_version": "pipeline_preflight_v1",
        "binding_id": arguments.get("binding_id"),
    }


def _json_value(value):
    """저장된 JSON과 비교할 객체를 JSON 기본 자료형으로 정규화한다."""
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))


class PipelineApprovalRun:
    """검토된 실행 계약을 기존 규칙 실행기의 승인 범위에 연결한다."""

    def __init__(self, states, binding: PipelineBinding):
        self.states = states
        self.binding = binding
        self.plan = binding.execution_plan()
        self.arguments = json.loads(self.plan.steps[-1].arguments_json)
        self.approvals = ApprovalStore(states.directory / "approvals.sqlite3")
        self.pipeline_execution = build_pipeline_execution(binding.adapter_id, states, binding)
        self.executor = RulesExecutor(
            states,
            self.approvals,
            self.plan,
            (
                RegisteredTool("run_preflight", _preflight, _validate_preflight),
                RegisteredTool(
                    "run_pipeline", self.pipeline_execution.execute, self.pipeline_execution.validate,
                    requires_approval=True,
                ),
            ),
        )

    def next(self, *, approval_id=None):
        """한 단계를 진행한다. worker는 유효한 1회 승인 ID가 있을 때만 호출된다."""
        return self.executor.run_next(approval_id=approval_id)


def open_pipeline_approval_run(directory, binding):
    """저장한 실행 상태와 현재 바인딩이 같은지 확인해 승인 도구를 연다."""
    directory = _root(directory)
    states = LocalStateStore(directory)
    state = states.load()

    if state.actor_id == "" or state.policy_version != binding.policy_version:
        raise ValueError("실행 소유자 또는 정책 버전이 일치하지 않습니다.")

    run = PipelineApprovalRun(states, binding)

    if run.plan.source_id != binding.source_id:
        raise ValueError("승인 실행 계획의 입력 출처가 바인딩과 다릅니다.")

    plan_path = directory / "execution_plan.json"

    if plan_path.exists():
        recorded = _decode(plan_path.read_text(encoding="utf-8"))

        if recorded != {
            "run_id": state.run_id,
            "actor_id": state.actor_id,
            "plan": _json_value(run.plan.to_dict()),
        }:
            raise ValueError("저장된 실행 계획이 현재 계약과 다릅니다.")

    return run
