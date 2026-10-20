"""공통 작업 UI의 세션 상태. 렌더링과 실제 서비스 호출을 분리한다."""

from copy import deepcopy
from threading import Lock

from src.agents.workflow_service_registry import workflow_service, workflow_specifications
from src.agents.worker_contract import WorkerLimits
from .dialogue_service import fingerprint


def workflow_modules():
    """명세만 조회한다. 원본·작업 DB·worker는 열지 않는다."""
    return workflow_specifications()


class UiWorkflow:
    """공통 서비스의 명시적 호출만 허용한다. 화면 재실행은 메서드를 호출하지 않는다.

    서비스는 호출할 때마다 현재 파일·승인·단계를 재검사한다. UI의 마지막 상태는
    표시용 복사본일 뿐 실행 권한이 아니다. 실패 후에는 사용자가 명시적으로 상태를
    다시 확인해야 하며 자동 재시도하거나 승인 ID를 다시 발급하지 않는다.
    """

    def __init__(self, module_id, version):
        self.module_id = module_id
        self.version = version
        self.reviewed = False
        self.creation_preview = None
        self.creation_selection = None
        self.creation_receipt = None
        self.creation_busy = False
        self.creation_requested = None
        self.creation_failure = None
        self.path_diagnostics = []
        self.name = None
        self.state = None
        self.approval_preview = None
        self.approval_id = None
        self.last_result = None
        self.error = False
        self.busy = False
        self.execution_requested = False
        self._command_lock = Lock()

    def service(self):
        return workflow_service(self.module_id, self.version)

    def preview_creation(self, values, limits):
        self.creation_preview = None
        self.creation_selection = None
        self.creation_preview = self.service().preview_form(values, WorkerLimits(**limits))
        self.creation_selection = fingerprint({"values": values, "limits": limits})
        return self.creation_preview.packet

    def create(self, values, limits, *, confirmed=False):
        if (confirmed is not True or self.creation_preview is None or self.creation_receipt is not None
                or self.creation_selection != fingerprint({"values": values, "limits": limits})):
            raise ValueError("현재 입력의 생성 미리보기와 명시적 확인이 필요합니다.")

        # 생성 결과가 불명확해도 같은 세션에서 무심코 다시 생성하지 않는다.
        preview = self.creation_preview
        self.creation_preview = None
        result = self.service().create(preview, confirmed=True)
        self.creation_receipt = deepcopy(result)
        self.name = result["run_name"]
        self.state = None
        return result

    def connect(self, name):
        """기존 작업 재연결도 사용자가 제출할 때만 정밀 조회한다."""
        state = self.service().status(name)
        self.name = name
        self.state = state
        self.reviewed = True
        self.error = False
        self.approval_preview = None
        self.approval_id = None

    def command(self, action, *, confirmed=False):
        """UI 복수 제출을 같은 컨트롤러 잠금으로 거부한다. backend 잠금도 유지한다."""
        if not self._command_lock.acquire(blocking=False):
            return {"status": "busy"}

        self.busy = True
        self.execution_requested = False
        try:
            return self._perform_command(action, confirmed=confirmed)
        except BaseException:
            # 승인 ID가 이미 소진된 실행 중단은 결과 불명으로 취급하고 재시도하지 않는다.
            if action == "execute" and self.approval_id is None:
                self.error = True
            raise
        finally:
            self.busy = False
            self._command_lock.release()

    def _perform_command(self, action, *, confirmed=False):
        if self.name is None:
            raise ValueError("먼저 작업을 준비하거나 연결하세요.")

        service = self.service()
        if action == "refresh":
            self.state = service.status(self.name)
            self.error = False
            return deepcopy(self.state)

        if self.error or self.state is None:
            raise ValueError("작업 상태를 명시적으로 다시 확인하세요.")

        actions = {item["id"] for item in self.state["actions"]}
        required = {"next": "next", "preview": "approve", "approve": "approve", "execute": "execute"}
        if action not in required or required[action] not in actions:
            raise ValueError("현재 표시 단계에서 허용되지 않는 행동입니다.")

        if action in {"approve", "execute"} and confirmed is not True:
            raise ValueError("승인과 실행에는 각각 별도의 사용자 확인이 필요합니다.")

        try:
            if action == "preview":
                self.approval_preview = service.preview_approval(self.name)
                return self.approval_preview.to_dict()

            if action == "approve":
                if self.approval_preview is None:
                    raise ValueError("먼저 승인 범위를 검사하세요.")

                preview = self.approval_preview
                self.approval_preview = None
                result = service.record_approval(self.name, preview, confirmed=True)
                self.approval_id = result["approval_id"]
            else:
                approval_id = None
                if action == "execute":
                    if self.approval_id is None:
                        raise ValueError("재연결한 작업은 유효 승인 ID를 직접 확인해 입력하세요.")

                    # 실행 결과를 받지 못해도 같은 승인으로 자동 재실행하지 않는다.
                    approval_id = self.approval_id
                    self.approval_id = None

                result = service.next(self.name, expected_stage=self.state["stage"],
                                      expected_revision=self.state["revision"], approval_id=approval_id)

            self.last_result = deepcopy(result)
            self.state = None
            self.state = service.status(self.name)
            return result
        except Exception:
            self.error = True
            raise
