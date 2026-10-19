"""UI 작업 연결과 명시적 명령 경계. 실제 승인은 기존 저장소·실행기가 검증한다."""

from copy import deepcopy
from pathlib import Path
from threading import Lock

from src.agents import pipeline_approval_cli as backend
from src.agents.cli import _current_actor
from src.agents.state import RunState
from src.agents.worker_paths import _path
from .dialogue_service import fingerprint


class UiExecution:
    """확인된 후보 하나에 결합한다. 입력 화면이 바뀌어도 진행 이력은 보존한다.

    명령은 버튼 제출에서만 호출한다. 같은 revision의 같은 명령을 재전송하지
    않으며 예외 후에는 결과 불명 상태로 멈춘다. worker 중복 실행의 최종 방어는
    기존 실행기의 프로세스 잠금·일회 승인·SQLite 작업 예약이 담당한다.
    """

    def __init__(self, receipt, selection_id):
        self.name = receipt["name"]
        self.directory = _path(Path(receipt["run_dir"]), directory=True)
        self.binding_id = receipt["binding_id"]
        self.selection_id = selection_id
        self.preview = None
        self.approval_id = None
        self.last_result = None
        self.error = False
        self.started = False
        self.busy = False
        self.worker_started = False
        self._commands = set()
        self._lock = Lock()
        self.read_state()

    def read_state(self):
        """화면 표시용 작은 JSON만 읽는다. DB 생성·모델 호출·worker 시작이 없다."""
        selection = backend._read_json(
            _path(self.directory / "selection.json", directory=False), backend.MAX_SELECTION_BYTES,
        )
        state = RunState(**backend._read_json(
            _path(self.directory / "state.json", directory=False), backend.MAX_SELECTION_BYTES,
        ))
        state.validate()

        if (fingerprint(selection) != self.selection_id or state.run_id != selection["run_id"]
                or state.actor_id != selection["actor_id"] or state.actor_id != _current_actor()):
            raise ValueError("작업 소유자·입력 선택 기록이 변경되었습니다.")

        return state

    def _check_current(self):
        """명령 전에는 입력·코드까지 기존 계약 검사기로 다시 확인한다."""
        self.read_state()
        directory, selection, run = backend._open(self.name)

        if (directory != self.directory or fingerprint(selection) != self.selection_id
                or run.binding.binding_id != self.binding_id):
            raise ValueError("현재 명령이 연결된 작업과 다릅니다.")

        return run

    def command(self, action, *, confirmed=False):
        """명시적인 사용자 제출만 처리한다. 승인과 실행은 별도 명령이다."""
        if action not in {"preflight", "preview", "approve", "execute"}:
            raise ValueError("지원하지 않는 UI 명령입니다.")

        if action in {"approve", "execute"} and confirmed is not True:
            raise ValueError("이번 승인 또는 실행 범위를 확인해야 합니다.")

        if not self._lock.acquire(blocking=False):
            return {"status": "busy"}

        try:
            state = self.read_state()
            key = (action, state.revision)

            if self.error or key in self._commands:
                raise ValueError("실패·중복 명령은 자동 재시도하지 않습니다.")

            self._check_current()
            self._commands.add(key)
            self.busy = True
            self.started = True

            try:
                if action == "preflight":
                    if state.current_stage == "preflight" and state.status in {"created", "running"}:
                        result = backend.next_step(self.name)
                    elif state.current_stage == "run_pipeline" and state.status == "running":
                        # worker 승인 ID를 전달하지 않으므로 승인 요청까지만 진행한다.
                        result = backend.next_step(self.name)
                    else:
                        raise ValueError("사전 검사 진행 상태가 아닙니다.")

                elif action == "preview":
                    self.preview = backend.preview_approval(self.name)
                    result = {"status": "approval_preview"}

                elif action == "approve":
                    if self.preview is None:
                        raise ValueError("먼저 현재 승인 범위를 검사하세요.")

                    result = backend.record_approval(self.name, deepcopy(self.preview), confirmed=True)
                    self.approval_id = result["approval_id"]
                    self.preview = None

                else:
                    current = backend.status(self.name)

                    if (self.approval_id is None or current.get("approval_id") != self.approval_id
                            or current.get("approval_status") != "pending"
                            or state.status != "awaiting_approval"):
                        raise ValueError("현재 범위의 유효한 미사용 승인이 필요합니다.")

                    result = backend.next_step(self.name, self.approval_id)
                    self.worker_started = result.get("worker_started", False)

                self.last_result = deepcopy(result)
                return result
            except BaseException:
                self.error = True

                if action == "execute":
                    # 호출이 중단되었다면 자식 시작 여부를 False로 단정하지 않는다.
                    self.worker_started = None
                    self.last_result = {"status": "unknown", "code": "UI_EXECUTION_INTERRUPTED",
                                        "worker_started": None}

                raise
            finally:
                self.busy = False
        finally:
            self._lock.release()
