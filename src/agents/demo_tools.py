"""실제 데이터와 독립된 고정 합성 도구와 데모 실행 계획."""

import json

from .contracts import ToolResult
from .executor import ExecutionPlan, ExecutionStep, RegisteredTool


DEMO_VERSION = "agent_synthetic_demo_v1"


def demo_plan():
    """실제 입력 경로 없이 생성·승인·검증 흐름을 확인하는 계획."""
    return ExecutionPlan(
        source_id="independent_synthetic_fixture_v1",
        config_id=DEMO_VERSION,
        policy_version="agent_demo_policy_v1",
        steps=(
            ExecutionStep("preflight", "run_preflight", '{"kind":"preflight"}'),
            ExecutionStep("ingest", "run_ingest", '{"kind":"synthetic_input"}'),
            ExecutionStep("audit", "run_audit", '{"kind":"synthetic_audit"}'),
            ExecutionStep("validate_contract", "validate_contract", '{"kind":"synthetic_contract"}'),
        ),
    )


def _execute(output, arguments):
    """고정 합성 JSON만 쓰며 실제 원본·pickle·NPZ·모델을 읽지 않는다."""
    record = {"demo_version": DEMO_VERSION, "kind": arguments["kind"]}

    with (output / "synthetic_result.json").open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, allow_nan=False)

    return ToolResult(status="passed")


def _validate(output, arguments):
    """파일 내용이 해당 단계의 고정 합성 계약과 일치하는지 검사한다."""
    path = output / "synthetic_result.json"
    record = json.loads(path.read_text(encoding="utf-8"))

    return record == {"demo_version": DEMO_VERSION, "kind": arguments["kind"]}


def demo_tools():
    """계획에 필요한 고정 도구만 등록한다. 외부 설정으로 함수를 import하지 않는다."""
    return tuple(
        RegisteredTool(step.action_id, _execute, _validate)
        for step in demo_plan().steps
    )
