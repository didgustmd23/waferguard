"""전처리 에이전트 내부 계약.

이 모듈의 객체는 로컬 실행기와 전처리 코어 사이에서만 사용한다.
원본 배열·DataFrame·상세 통계·파일 내용은 상태나 선택기 입력에 넣지 않는다.
"""

from dataclasses import dataclass, field
from typing import Mapping

from .workflow import STAGE_ORDER


# ==========================================
# 실행 상태·단계·도구 결과의 고정 열거값
# - 외부 입력 문자열을 그대로 상태나 도구 ID로 사용하지 않는다.
# - LLM 선택기는 이후 별도 최소 DTO로만 연결한다.
# ==========================================
RUN_STATUSES = frozenset({
    "created", "running", "awaiting_approval", "blocked_input",
    "failed_runtime", "cancelled", "completed",
})

STAGES = frozenset(STAGE_ORDER)

TOOL_STATUSES = frozenset({
    "passed", "review_required", "blocked_input", "failed_runtime",
})


@dataclass(frozen=True)
class ToolResult:
    """등록된 로컬 도구가 반환하는 요약 결과.

    artifact_refs에는 실행 하위의 승인된 상대 참조만 두며, 원본 내용·절대 경로·
    예외 원문은 넣지 않는다. metrics 역시 로컬 보고서의 상세값을 복사하지 않은
    고정 키의 일반 상태에 한정한다.
    """

    status: str
    artifact_refs: tuple[str, ...] = ()
    metrics: Mapping[str, int | bool] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def __post_init__(self):
        if self.status not in TOOL_STATUSES:
            raise ValueError("허용하지 않는 도구 결과 상태입니다.")

        invalid_artifact_reference = any(
            not isinstance(item, str) or not item
            for item in self.artifact_refs
        )

        if invalid_artifact_reference:
            raise ValueError("산출물 참조 형식이 올바르지 않습니다.")

        invalid_message_code = any(
            not isinstance(item, str) or not item
            for item in self.warnings + self.errors
        )

        if invalid_message_code:
            raise ValueError("경고·오류 코드는 문자열이어야 합니다.")

        for key, value in self.metrics.items():
            if not isinstance(key, str) or type(value) not in (int, bool):
                raise ValueError("도구 요약 지표 형식이 올바르지 않습니다.")

