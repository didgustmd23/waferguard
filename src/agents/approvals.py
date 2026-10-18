"""전처리 작업의 승인 대상과 승인 이력을 관리한다."""

from datetime import datetime
from dataclasses import dataclass

# ==========================================
# 승인 대상 계약
# - 승인받을 실행·작업·입력·설정을 하나로 묶는다.
# - 실제 데이터나 설정 내용 대신 로컬 식별값을 저장한다.
# - frozen=True로 생성 후 필드 변경을 막는다.
# - 식별값의 실제 일치 여부는 이후 승인 소비 시 확인한다.
# ==========================================
@dataclass(frozen=True)
class ApprovalScope:
    """하나의 승인이 적용되는 정확한 작업 범위."""

    actor_id: str
    run_id: str
    stage: str
    tool_id: str
    target_id: str
    arguments_id: str
    source_id: str
    config_id: str
    plan_id: str
    policy_version: str

    def __post_init__(self):
        """승인 대상의 필수 식별값이 비어 있지 않은지 확인한다."""

        identifiers = (
            self.actor_id,
            self.run_id,
            self.stage,
            self.tool_id,
            self.target_id,
            self.arguments_id,
            self.source_id,
            self.config_id,
            self.plan_id,
            self.policy_version,
        )

        # 누락되거나 공백뿐인 식별값으로 승인을 만들지 못하게 한다.
        invalid_identifier = any(
            not isinstance(value, str) or not value.strip()
            for value in identifiers
        )

        if invalid_identifier:
            raise ValueError("승인 대상의 필수 식별값이 올바르지 않습니다.")


# ==========================================
# 승인 기록과 사용 상태
# - pending: 사용자 확인을 거쳐 발급됐으며 아직 사용하지 않은 승인
# - consumed: 작업 예약에 이미 사용한 승인
# - revoked: 취소된 승인
# - expired: 유효기간이 지난 승인
# - 실제 사용 가능 여부는 저장소의 최신 기록으로 판단한다.
# ==========================================
APPROVAL_STATUSES = frozenset({
    "pending",
    "consumed",
    "revoked",
    "expired",
})


@dataclass(frozen=True)
class ApprovalRecord:
    """발급된 승인의 대상·유효기간·사용 상태."""

    approval_id: str
    scope: ApprovalScope
    created_at: datetime
    expires_at: datetime
    status: str = "pending"

    def __post_init__(self):
        """승인 기록의 기본 형식과 유효기간을 검증한다."""

        if not isinstance(self.approval_id, str) or not self.approval_id.strip():
            raise ValueError("승인 ID가 필요합니다.")

        if not isinstance(self.scope, ApprovalScope):
            raise ValueError("승인 대상은 ApprovalScope여야 합니다.")

        if self.status not in APPROVAL_STATUSES:
            raise ValueError("허용하지 않는 승인 상태입니다.")

        timestamps = (
            self.created_at,
            self.expires_at,
        )

        # 시간대가 없는 시각은 컴퓨터 설정에 따라 해석이 달라지므로 거부한다.
        invalid_timestamp = any(
            not isinstance(value, datetime)
            or value.utcoffset() is None
            for value in timestamps
        )

        if invalid_timestamp:
            raise ValueError("승인 시각에는 시간대 정보가 필요합니다.")

        if self.expires_at <= self.created_at:
            raise ValueError("승인 만료 시각은 생성 시각보다 늦어야 합니다.")

# ==========================================
# 승인 사용 가능 여부 검사
# - 이미 사용했거나 취소된 승인은 거부
# - 생성 이전·만료 시각 이후에는 사용 불가
# - 현재 작업의 대상·설정·정책이 승인 내용과 일치해야 통과
# - 검사 결과는 고정 코드로 반환
# ==========================================
def validate_approval(
    record: ApprovalRecord,
    scope: ApprovalScope,
    *,
    now: datetime,
) -> str:
    """지정한 시각과 작업 범위에 대한 승인 검사 코드를 반환한다."""

    # 호출 인자의 기본 형식을 확인한다.
    if not isinstance(record, ApprovalRecord):
        raise TypeError("승인 기록은 ApprovalRecord여야 합니다.")

    if not isinstance(scope, ApprovalScope):
        raise TypeError("현재 작업 대상은 ApprovalScope여야 합니다.")

    if not isinstance(now, datetime) or now.utcoffset() is None:
        raise ValueError("검사 시각에는 시간대 정보가 필요합니다.")

    # 이미 사용했거나 취소된 승인은 다시 사용할 수 없다.
    if record.status == "consumed":
        return "APPROVAL_ALREADY_CONSUMED"

    if record.status == "revoked":
        return "APPROVAL_REVOKED"

    # 만료 시각과 정확히 같아도 만료된 것으로 처리한다.
    if record.status == "expired" or now >= record.expires_at:
        return "APPROVAL_EXPIRED"

    if now < record.created_at:
        return "APPROVAL_NOT_YET_VALID"

    # dataclass 비교는 모든 필드를 비교한다.
    # 사용자·원본·설정·계획 중 하나라도 다르면 거부한다.
    if record.scope != scope:
        return "APPROVAL_SCOPE_MISMATCH"

    return "APPROVAL_VALID"
