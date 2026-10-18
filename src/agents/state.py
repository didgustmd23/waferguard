"""규칙 기반 에이전트의 로컬 상태 저장.

실제 데이터의 내용은 저장하지 않는다. 상태 파일은 재개 판단용 메타데이터이며,
전처리 산출물의 완료 표식이나 파일 무결성을 대체하지 않는다.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import uuid

from .contracts import RUN_STATUSES, STAGES


RUN_ID = re.compile(r"^[a-f0-9]{32}$")


def utc_now():
    """재현 가능한 형식의 UTC 시각을 생성한다."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ==========================================
# JSON 상태 스키마
# - 상태 revision은 저장 성공 때마다 하나 증가한다.
# - artifact_refs는 상대 참조만 허용하며 원문 데이터는 기록하지 않는다.
# ==========================================
@dataclass
class RunState:
    run_id: str
    actor_id: str
    status: str = "created"
    current_stage: str = "preflight"
    policy_version: str = "preprocess_policy_v1"
    revision: int = 0
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    artifact_refs: dict[str, str] = field(default_factory=dict)
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    approval_ids: list[str] = field(default_factory=list)
    cancel_requested: bool = False

    def validate(self):
        if not RUN_ID.fullmatch(self.run_id):
            raise ValueError("실행 ID 형식이 올바르지 않습니다.")

        if not isinstance(self.actor_id, str) or not self.actor_id:
            raise ValueError("실행 주체 식별자가 필요합니다.")

        if self.status not in RUN_STATUSES or self.current_stage not in STAGES:
            raise ValueError("실행 상태 또는 단계가 올바르지 않습니다.")

        if not isinstance(self.policy_version, str) or not self.policy_version:
            raise ValueError("정책 버전이 필요합니다.")

        if type(self.revision) is not int or self.revision < 0:
            raise ValueError("상태 revision이 올바르지 않습니다.")

        if type(self.cancel_requested) is not bool:
            raise ValueError("취소 상태 형식이 올바르지 않습니다.")

        invalid_artifact_references = (
            not isinstance(self.artifact_refs, dict)
            or any(
                not isinstance(key, str)
                or not key
                or not isinstance(value, str)
                or not value
                for key, value in self.artifact_refs.items()
            )
        )

        if invalid_artifact_references:
            raise ValueError("산출물 참조 형식이 올바르지 않습니다.")

        code_lists = (
            self.blockers,
            self.warnings,
            self.approval_ids,
        )

        if not all(isinstance(values, list) for values in code_lists):
            raise ValueError("상태 코드 목록 형식이 올바르지 않습니다.")

        invalid_state_code = any(
            not isinstance(value, str) or not value
            for value in self.blockers + self.warnings + self.approval_ids
        )

        if invalid_state_code:
            raise ValueError("상태 코드 형식이 올바르지 않습니다.")


class LocalStateStore:
    """한 실행의 상태 파일을 원자적으로 저장·복원한다."""

    def __init__(self, run_directory):
        self.directory = Path(run_directory).resolve()
        self.path = self.directory / "state.json"

    def create(self, actor_id, policy_version="preprocess_policy_v1"):
        """새 실행의 초기 상태를 만들며 기존 상태 파일은 덮어쓰지 않는다."""
        if self.path.exists():
            raise FileExistsError("이미 상태 파일이 있는 실행 경로입니다.")

        state = RunState(
            run_id=uuid.uuid4().hex,
            actor_id=actor_id,
            policy_version=policy_version,
        )

        self.save(state, expected_revision=0, create_directory=True)

        return state

    def load(self):
        """신뢰하지 않는 JSON을 엄격하게 복원한다."""
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("상태 파일을 읽을 수 없습니다.") from error

        if not isinstance(raw, dict) or set(raw) != set(RunState.__dataclass_fields__):
            raise ValueError("상태 파일 구조가 올바르지 않습니다.")

        state = RunState(**raw)
        state.validate()

        return state

    def save(self, state, *, expected_revision, create_directory=False):
        """revision을 확인한 뒤 임시 파일 교체로 상태를 저장한다."""
        if not isinstance(state, RunState):
            raise TypeError("RunState만 저장할 수 있습니다.")

        state.validate()

        if state.revision != expected_revision:
            raise ValueError("오래된 상태 revision으로 저장할 수 없습니다.")

        if self.path.exists() and self.load().revision != expected_revision:
            raise ValueError("다른 변경으로 상태 revision이 달라졌습니다.")

        if not self.path.parent.exists():
            if not create_directory:
                raise FileNotFoundError("실행 상태 폴더가 없습니다.")

            self.path.parent.mkdir(parents=True, exist_ok=False)

        state.revision += 1
        state.updated_at = utc_now()

        temporary = self.path.with_suffix(".json.tmp")

        try:
            temporary.write_text(
                json.dumps(asdict(state), ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8",
            )

            temporary.replace(self.path)
        except OSError as error:
            raise RuntimeError("상태 파일 저장에 실패했습니다.") from error

        return state

