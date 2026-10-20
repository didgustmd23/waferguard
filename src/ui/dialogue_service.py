"""UI 세션과 기존 제안 대화를 연결한다. 전처리 승인·worker와는 연결하지 않는다."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from threading import Lock
from uuid import uuid4

from src.agents.llm_config import LocalModelConfig
from src.agents.llm_proposal import ALLOWED, SOURCE
from src.agents.llm_proposal_contract import validate_proposal
from src.agents.llm_proposal_store import STORE_ROOT, load_proposal, save_proposal
from .model_commands import ModelCommandLedger
from .state import MODES, validate_draft, validate_settings


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def create_dialogue(config):
    # UI를 열거나 설정을 보는 것만으로 선택 의존성·모델 운송을 초기화하지 않는다.
    from src.agents.llm_dialogue import ProposalDialogue

    return ProposalDialogue(config)


class UiDialogue:
    """브라우저 세션별 대화. 재접속·서버 재시작 시 자동 복원·전송하지 않는다."""

    def __init__(self, settings, *, factory=create_dialogue, ledger=None, proposal_root=STORE_ROOT):
        settings = validate_settings(settings)

        if settings["mode"] != MODES[0]:
            raise ValueError("로컬 LLM 모드에서만 대화할 수 있습니다.")

        self.settings_id = fingerprint(settings)
        self._dialogue = factory(LocalModelConfig(**settings["model"]))
        self._ledger = ModelCommandLedger() if ledger is None else ledger
        self._lock = Lock()
        self.pending = None
        self.result = None
        self.history = []
        self.phase = "new"
        self.error = None
        self.saved = None
        self._proposal_root = proposal_root

    def matches(self, settings):
        return self.settings_id == fingerprint(validate_settings(settings))

    def _validate(self, proposal):
        return validate_proposal(json.dumps(proposal, ensure_ascii=False),
                                 expected_input=dict(SOURCE), allowed_modules=ALLOWED).to_dict()

    def prepare(self, text, settings):
        """누적 문장·공개 명세·모델 설정의 정확한 미리보기를 만들며 호출은 하지 않는다."""
        with self._lock:
            if not self.matches(settings) or self.phase not in ("new", "awaiting_answer"):
                raise ValueError("설정이 바뀌었거나 대화가 종료되었습니다. 새 대화를 시작하세요.")

            text = validate_draft(text)
            preview = self._dialogue.preview(text)
            binding = fingerprint({"settings": settings, "preview": preview})
            self.pending = {"command_id": uuid4().hex, "binding": binding,
                            "text": text, "preview": deepcopy(preview)}
            return deepcopy(self.pending)

    def send(self, command_id, settings, *, confirmed=False):
        """확인한 명령만 한 번 소비한다. 오류 이후 동일 대화를 자동 재개하지 않는다."""
        if confirmed is not True:
            raise ValueError("전송 내용을 확인해야 합니다.")

        if not self._lock.acquire(blocking=False):
            return {"status": "busy"}

        try:
            pending = self.pending

            if (pending is None or pending["command_id"] != command_id
                    or not self.matches(settings)):
                raise ValueError("전송 확인이 만료되었습니다. 내용을 다시 확인하세요.")

            # 표시 후 문장·공개 명세·설정이 달라졌다면 이전 확인을 사용할 수 없다.
            preview = self._dialogue.preview(pending["text"])
            binding = fingerprint({"settings": settings, "preview": preview})

            if binding != pending["binding"] or preview != pending["preview"]:
                self.pending = None
                raise ValueError("전송 내용이 변경되었습니다. 미리보기를 다시 만드세요.")

            # DB 오류 시 호출하지 않는다. 한 번 소비된 명령은 성공 여부와 무관하게 재사용 금지.
            if not self._ledger.claim(command_id, binding):
                self.pending = None
                return {"status": "already_consumed"}

            self.pending = None
            self.phase = "sending"

            try:
                result = self._dialogue.send(pending["text"], confirmed=True)
                checked = self._validate(result["proposal"])
                proposed = checked["proposal"]["status"] == "proposed"
                expected = "proposal_validated" if proposed else "awaiting_answer"

                if (result["status"] not in (expected, "clarification_limit_reached")
                        or result["plan_signature"] != checked["plan_signature"]):
                    raise ValueError("대화 응답 상태가 검증 결과와 다릅니다.")

                self.result = checked
                self.phase = result["status"]
                self.history.append({"user": pending["text"],
                                     "questions": checked["proposal"]["questions"]})
            except BaseException:
                self.error = "MODEL_CALL_FAILED_OR_RESPONSE_REJECTED"
                self.result = None
                self.phase = "failed"
                self._dialogue.close()
                self._ledger.finish(command_id, binding, "unknown")
                raise

            self._ledger.finish(command_id, binding, "completed")
            return {"status": self.phase}
        finally:
            self._lock.release()

    def validated_proposal(self, settings):
        """검토 화면의 선행 조건. 세션 불리언이 아니라 현재 계약으로 재검사한 제안만 반환한다."""
        if self.phase != "proposal_validated" or self.result is None or not self.matches(settings):
            return None

        try:
            checked = self._validate(self.result["proposal"])
        except (ValueError, TypeError, KeyError):
            return None

        return checked if checked == self.result else None

    def close(self):
        self._dialogue.close()
        self.pending = None
        self.result = None
        self.saved = None
        self.phase = "closed"

    def saved_proposal(self, settings):
        """저장 완료 플래그 대신 파일의 현재 계약·지문과 이번 제안을 대조한다."""
        checked = self.validated_proposal(settings)

        if checked is None or self.saved is None:
            return None

        try:
            restored = load_proposal(self.saved["name"], root=self._proposal_root)
        except (OSError, ValueError, TypeError, KeyError, UnicodeError):
            return None

        if (restored["plan_signature"] != checked["plan_signature"]
                or restored["proposal"] != checked["proposal"]):
            return None

        return restored

    def restore_saved(self, name, settings):
        """저장 제안을 기존 복원 검사와 현재 계약으로 다시 확인해 새 대화의 검토 상태로 둔다.

        모델 호출·명령 소비·승인 발급·작업 생성은 하지 않는다. 불러온 제안도 실행 승인이 아니다.
        """
        with self._lock:
            if (not self.matches(settings) or self.phase != "new" or self.pending is not None
                    or self.result is not None or self.saved is not None):
                raise ValueError("새 대화에서만 저장 제안을 불러올 수 있습니다.")

            restored = load_proposal(name, root=self._proposal_root)
            checked = self._validate(restored["proposal"])

            if checked["plan_signature"] != restored["plan_signature"] or checked["proposal"] != restored["proposal"]:
                raise ValueError("저장 제안이 현재 계약과 일치하지 않습니다.")

            self.result = checked
            self.phase = "proposal_validated"
            self.saved = {"status": "proposal_restored", "name": name,
                          "local_file": str(Path(self._proposal_root) / f"{name}.json"),
                          "plan_signature": checked["plan_signature"],
                          "execution_approved": False, "worker_started": False}

            if self.saved_proposal(settings) is None:
                self.result, self.saved, self.phase = None, None, "new"
                raise ValueError("불러온 제안의 복원 검증을 통과하지 못했습니다.")

            return deepcopy(self.saved)

    def save(self, name, settings, *, confirmed=False):
        """명시적 검토 확인 후 새 파일로 저장한다. 동일 저장 제출은 재검증만 한다."""
        if confirmed is not True:
            raise ValueError("제안 내용을 검토하고 저장에 동의해야 합니다.")

        with self._lock:
            checked = self.validated_proposal(settings)

            if checked is None:
                raise ValueError("현재 설정에서 검증된 제안이 필요합니다.")

            if self.saved is not None:
                if self.saved["name"] != name or self.saved_proposal(settings) is None:
                    raise ValueError("저장 기록이 달라졌습니다. 기존 파일은 덮어쓰지 않습니다.")

                return deepcopy(self.saved)

            result = dict(checked, status="proposal_validated")
            receipt = save_proposal(name, result, root=self._proposal_root)
            self.saved = receipt

            if self.saved_proposal(settings) is None:
                self.saved = None
                raise ValueError("저장 후 복원 검증을 통과하지 못했습니다.")

            return deepcopy(receipt)
