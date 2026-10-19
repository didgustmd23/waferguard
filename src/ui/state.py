"""Streamlit에 의존하지 않는 화면 초안·세션 설정 검증."""

from dataclasses import asdict, dataclass

from src.agents.llm_config import LocalModelConfig, load_local_config
from src.agents.worker_contract import WorkerLimits


SCREENS = ("1 요청·대화", "2 제안 검토", "3 입력 선택", "4 실행 승인", "5 결과 확인", "설정")
MODES = ("로컬 LLM 제안", "규칙 기반")

# 제안·저장 파일·후보 생성 영수증을 순서대로 확인한다. 실행 자체는 별도 승인 경계에서 검사한다.
SCREEN_LOCKS = {
    SCREENS[1]: "요청 화면에서 로컬 모델의 검증된 제안을 먼저 생성하세요.",
    SCREENS[2]: "검증된 제안을 검토·저장한 후 입력을 선택할 수 있습니다.",
    SCREENS[3]: "입력과 실행안이 결합된 실행 후보를 먼저 준비해야 합니다.",
    SCREENS[4]: "실행을 시작한 작업이 있어야 진행 상태와 결과를 확인할 수 있습니다.",
}


def resolve_screen(requested, *, proposal_available=False, saved_available=False, candidate_available=False,
                   execution_available=False):
    """메뉴 비활성화와 별개로 화면 진입을 검사한다. 실행 권한 검증은 대신하지 않는다."""
    allowed = (SCREENS[0], SCREENS[-1])

    # 결과 이력은 새 제안·설정과 독립적으로 조회한다. 실행 권한은 부여하지 않는다.
    if execution_available is True:
        allowed += (SCREENS[4],)

    if proposal_available is True:
        allowed += (SCREENS[1],)

        if saved_available is True:
            allowed += (SCREENS[2],)

            if candidate_available is True:
                allowed += (SCREENS[3],)

    if requested not in allowed:
        return SCREENS[0]

    return requested


@dataclass(frozen=True)
class SessionSettings:
    """이번 브라우저 세션에서만 적용하는 설정. 전송·실행 허가를 포함하지 않는다."""

    model: LocalModelConfig
    limits: WorkerLimits
    mode: str = MODES[0]

    def __post_init__(self):
        if not isinstance(self.model, LocalModelConfig) or not isinstance(self.limits, WorkerLimits):
            raise ValueError("검증된 모델·자원 설정이 필요합니다.")

        if self.mode not in MODES:
            raise ValueError("지원하지 않는 화면 모드입니다.")

    def to_dict(self):
        """UI 세션에 기본 자료형만 전달하고 호출·승인 상태는 만들지 않는다."""
        return asdict(self)


def default_settings():
    """기존 모델 JSON을 읽기만 한다. Ollama 접속·모델 목록 조회는 하지 않는다."""
    return SessionSettings(load_local_config(), WorkerLimits(300, 2048, 2)).to_dict()


def validate_settings(value):
    """폼 전체를 검증한 후에만 세션 적용을 허용한다. 추가 비밀 필드는 받지 않는다."""
    if type(value) is not dict or set(value) != {"model", "limits", "mode"}:
        raise ValueError("정확한 설정 항목을 지정하세요.")

    return SessionSettings(
        LocalModelConfig(**value["model"]), WorkerLimits(**value["limits"]), value["mode"],
    ).to_dict()


def validate_draft(text):
    """요청을 화면에 보관하기 전에 길이·제어 문자를 검사한다. 모델로 보내지 않는다."""
    if type(text) is not str or not text.strip():
        raise ValueError("요청 문장을 입력하세요.")

    if len(text) > 4000 or len(text.encode("utf-8")) > 8192:
        raise ValueError("요청은 4000자·UTF-8 8 KiB 이하여야 합니다.")

    if any((ord(char) < 32 and char not in "\n\t") or ord(char) == 127 for char in text):
        raise ValueError("제어 문자를 넣을 수 없습니다.")

    return text
