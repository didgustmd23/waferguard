"""전처리 에이전트의 규칙 기반 실행 정책."""

from dataclasses import dataclass

from .state import RunState
from .workflow import STAGE_ACTION


# ==========================================
# 실행 작업과 판단 결과의 고정 값
# - 외부 입력 문자열을 그대로 실행 명령으로 사용하지 않는다.
# - 이후 승인·실행기는 이 작업 ID만 받는다.
# ==========================================
ACTION_IDS = frozenset(STAGE_ACTION.values())


DECISION_CODES = frozenset({
    "ACTION_ALLOWED",
    "ACTION_NOT_ALLOWED",
    "RUN_CANCEL_REQUESTED",
    "SECURITY_LOCKED",
    "RUN_NOT_EXECUTABLE",
    "APPROVAL_PENDING",
})


# ==========================================
# 단계별 허용 작업
# - 한 단계에서 다음 단계의 작업을 실행하지 못하게 제한한다.
# - 승인 자체의 발급·소비는 다음 approvals.py에서 담당한다.
# ==========================================
ALLOWED_ACTIONS_BY_STAGE = {
    stage: frozenset({action})
    for stage, action in STAGE_ACTION.items()
}


@dataclass(frozen=True)
class PolicyDecision:
    """정책 판단의 최소 결과.

    원본 데이터·경로·예외 원문은 포함하지 않고,
    고정된 허용 여부와 코드만 전달한다.
    """

    allowed: bool
    code: str

    def __post_init__(self):
        """허용 여부와 판단 코드의 형식, 일치 여부를 검증"""

        # 0, 1이나 문자열 대신 실제 bool 값만 허용
        if type(self.allowed) is not bool:
            raise ValueError("정책 허용 여부는 bool이어야 합니다.")

        # 정의된 고정 판단 코드만 허용
        if self.code not in DECISION_CODES:
            raise ValueError("허용하지 않는 정책 판단 코드입니다.")

        # 허용 여부와 판단 코드가 서로 일치하는지 확인
        expected_allowed = self.code == "ACTION_ALLOWED"

        if self.allowed != expected_allowed:
            raise ValueError("정책 허용 여부와 판단 코드가 일치하지 않습니다.")

# ==========================================
# 현재 단계에서 작업 실행이 가능한지 판단
# - 실행 상태의 기본 형식을 먼저 검증
# - 취소된 실행과 취소 요청을 우선 차단
# - 등록된 작업인지 확인한 뒤 단계별 허용 목록 검사
# - 판단 결과는 허용 여부와 고정 코드로 반환
# ==========================================
def decide_action(state: RunState, action_id: str) -> PolicyDecision:
    state.validate()

    # 보안 잠금일 경우 사전 차단
    if "SECURITY_LOCKED" in state.blockers:
        return PolicyDecision(
            allowed=False,
            code="SECURITY_LOCKED",
        )

    # 취소 요청이 들어왔거나 이미 취소된 실행을 차단
    if state.cancel_requested or state.status == 'cancelled':
        return PolicyDecision(
            allowed=False,
            code="RUN_CANCEL_REQUESTED",
        )
    
    # 중단되거나 완료된 실행은 상태 정리 후 재개
    blocked_statuses = frozenset({
        "completed",
        "failed_runtime",
        "blocked_input",
    })

    if state.status in blocked_statuses:
        return PolicyDecision(
            allowed=False,
            code="RUN_NOT_EXECUTABLE",
        )

    # 사용자 승인 기다리는 동안 새작업 실행 금지
    if state.status == "awaiting_approval":
        return PolicyDecision(
            allowed=False,
            code="APPROVAL_PENDING",
        )

    # 미등록 작업 실행 대상으로 인정 안함
    if action_id not in ACTION_IDS:
        return PolicyDecision(
            allowed=False,
            code="ACTION_NOT_ALLOWED",
        )

    # 허용 목록 가져오기
    allowed_actions = ALLOWED_ACTIONS_BY_STAGE.get(
        state.current_stage,
        frozenset(),
    )

    # 등록 작업이라도 현재 단계아니라면 거부
    if action_id not in allowed_actions:
        return PolicyDecision(
            allowed=False,
            code="ACTION_NOT_ALLOWED",
        )

    return PolicyDecision(
        allowed=True,
        code="ACTION_ALLOWED",
    )
