"""확인한 입력 미리보기를 기존 미승인 작업 생성기에 연결한다. 승인·worker는 호출하지 않는다."""

from copy import deepcopy
from pathlib import Path
from threading import Lock

from src.agents.llm_execution_candidate import _fingerprint as candidate_fingerprint, create_candidate
from src.agents.pipeline_approval_cli import MAX_SELECTION_BYTES, _read_json
from src.agents.state import RunState
from src.agents.worker_contract import WorkerLimits
from src.agents.worker_paths import _path
from src.preprocessing.module_contracts import check_id
from .dialogue_service import fingerprint


class UiCandidate:
    """세션의 작업 준비 요청. 실패·중단 후 자동 재시도하지 않으며 기존 파일을 보존한다."""

    def __init__(self, preview, dialogue, settings):
        self.preview = deepcopy(preview)
        self.preview_id = fingerprint(preview)
        self.source_id = getattr(dialogue, "ui_source_id", None)
        self.phase = "ready"
        self.receipt = None
        self.run_selection_id = None
        self._lock = Lock()

        if not self.matches(dialogue, settings):
            raise ValueError("현재 제안·설정에 해당하는 입력 미리보기가 필요합니다.")

    def matches(self, dialogue, settings):
        """UI 플래그가 아닌 현재 저장 제안·설정·선택 범위를 비교한다."""
        saved = dialogue.saved_proposal(settings)
        preview = self.preview
        result = preview["result"]
        selection = result["selection"]
        return (fingerprint(preview) == self.preview_id
                and saved is not None and preview["settings_id"] == dialogue.settings_id
                and getattr(dialogue, "ui_source_id", None) == self.source_id
                and preview["plan_signature"] == saved["plan_signature"]
                and preview["proposal_name"] == saved["name"]
                and selection["proposal_name"] == saved["name"]
                and selection["proposal_plan_signature"] == saved["plan_signature"]
                and selection["pipeline"] == saved["proposal"]["pipeline"]
                and selection["limits"] == settings["limits"]
                and result["status"] == "candidate_preview"
                and result["selection_id"] == candidate_fingerprint(selection))

    def _read_run_selection(self):
        directory = _path(Path(self.receipt["run_dir"]), directory=True)
        selection_file = _path(directory / "selection.json", directory=False)
        return directory, _read_json(selection_file, MAX_SELECTION_BYTES)

    def verified_receipt(self, dialogue, settings):
        """작은 준비 기록만 읽어 화면 선행 조건을 확인한다. 실행 전 전체 재검사를 대신하지 않는다."""
        if self.phase != "created" or self.receipt is None:
            return None

        try:
            if not self.matches(dialogue, settings) or fingerprint(self.preview) != self.preview_id:
                return None

            receipt = self.receipt
            candidate = _path(Path(receipt["candidate_directory"]), directory=True)
            stored = _read_json(_path(candidate / "CREATED.json", directory=False), MAX_SELECTION_BYTES)
            original = self.preview["result"]["selection"]

            if (stored != receipt
                    or _read_json(_path(candidate / "candidate.json", directory=False), MAX_SELECTION_BYTES) != original
                    or _read_json(_path(candidate / "pipeline.json", directory=False), MAX_SELECTION_BYTES) != original["pipeline"]
                    or _read_json(_path(candidate / "profile.json", directory=False), MAX_SELECTION_BYTES) != original["profile"]):
                return None

            directory, selection = self._read_run_selection()
            state = RunState(**_read_json(_path(directory / "state.json", directory=False), MAX_SELECTION_BYTES))
            state.validate()

            if (fingerprint(selection) != self.run_selection_id
                    or state.run_id != selection["run_id"] or state.actor_id != selection["actor_id"]):
                return None

            return {**deepcopy(receipt), "recorded_status": state.status,
                    "recorded_stage": state.current_stage}
        except (OSError, ValueError, TypeError, KeyError, UnicodeError):
            return None

    def create(self, name, dialogue, settings, *, confirmed=False):
        """확인한 범위를 재검사해 배타 생성한다. 같은 완료 요청은 읽기 검증만 수행한다."""
        if confirmed is not True:
            raise ValueError("작업 준비 범위를 확인해야 합니다.")

        check_id(name)
        if len(name) > 48:
            raise ValueError("실행 후보 이름은 48자 이하여야 합니다.")

        if not self._lock.acquire(blocking=False):
            return {"status": "busy"}

        try:
            if self.phase == "created":
                receipt = self.verified_receipt(dialogue, settings)
                if receipt is None or name != receipt["name"]:
                    raise ValueError("기존 작업과 요청이 다릅니다. 다시 생성하지 않습니다.")
                return receipt

            if self.phase != "ready" or not self.matches(dialogue, settings):
                raise ValueError("현재 미리보기로 작업을 준비할 수 없습니다. 내용을 다시 검사하세요.")

            self.phase = "creating"
            result = self.preview["result"]
            selection = result["selection"]
            try:
                # 이 기존 생성기가 현재 파일·프로파일·코드를 다시 검사하고 변경 시 거부한다.
                receipt = create_candidate(
                    name, selection["proposal_name"], selection["snapshot_directory"], selection["profile_path"],
                    WorkerLimits(**settings["limits"]), expected_selection_id=result["selection_id"], confirmed=True,
                )

                if (receipt["status"] != "execution_candidate_created"
                        or receipt["name"] != name or receipt["selection_id"] != result["selection_id"]
                        or receipt["binding_id"] != result["binding_id"]
                        or receipt["proposal_name"] != selection["proposal_name"]
                        or receipt["worker_started"] is not False or receipt["model_called"] is not False
                        or receipt["execution_approval_issued"] is not False):
                    raise ValueError("실행 후보 생성 결과가 확인한 범위와 다릅니다.")

                self.receipt = deepcopy(receipt)
                _, actual = self._read_run_selection()
                if any(actual[key] != selection[key] for key in ("binding", "pipeline", "profile", "limits", "snapshot_directory")):
                    raise ValueError("생성한 작업 선택이 미리보기와 다릅니다.")

                self.run_selection_id = fingerprint(actual)
                self.phase = "created"
                verified = self.verified_receipt(dialogue, settings)
                if verified is None:
                    raise ValueError("생성 영수증과 작업 기록을 확인할 수 없습니다.")
                return verified
            except BaseException:
                self.phase = "failed"
                raise
        finally:
            self._lock.release()
