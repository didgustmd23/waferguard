"""데이터셋과 독립적인 작업 ID·단계·검토된 실행 템플릿."""

from src.preprocessing.workflows import STAGE_ACTION, STAGE_ORDER, WORKFLOW_TEMPLATES


def build_profile_plan(profile, source_id):
    """검증된 프로파일에서 실행 계획을 생성한다. 미등록 도구를 자동 구현하지 않는다."""
    from src.preprocessing.profiles import DatasetProfile
    from .executor import ExecutionPlan, ExecutionStep

    if not isinstance(profile, DatasetProfile):
        raise TypeError("검증된 DatasetProfile이 필요합니다.")

    arguments = profile.tool_arguments_json
    steps = tuple(
        ExecutionStep(stage, STAGE_ACTION[stage], arguments)
        for stage in WORKFLOW_TEMPLATES[profile.workflow_id]
    )

    return ExecutionPlan(
        source_id=source_id,
        config_id=profile.signature,
        policy_version="preprocess_policy_v1",
        steps=steps,
        profile_id=profile.profile_id,
        workflow_id=profile.workflow_id,
    )
