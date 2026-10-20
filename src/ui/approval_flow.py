"""명시적으로 시작한 작업만 다음 승인 경계까지 진행하는 공통 UI 제어기."""

from threading import Lock


TERMINAL = {"completed", "blocked_input", "failed_runtime", "security_locked", "cancelled"}


class ApprovalFlow:
    """승인 없는 진행과 현재 승인 한 건의 실행을 묶는다. 재시도·자동 승인은 없다.

    화면 재표시에서는 run을 호출하지 않는다. 별도 잠금은 복수 클릭 전체를 막고,
    각 명령의 계약·revision·승인 검증은 기존 컨트롤러와 backend가 그대로 담당한다.
    """

    def __init__(self, controller, *, workflow):
        self.controller = controller
        self.workflow = workflow
        self.busy = False
        self.pending = None
        self.started = False
        self.error = False
        self.ready = False
        self.state = None
        self.summary = None
        self.review_key = None
        self._lock = Lock()

    def _snapshot(self):
        if self.workflow:
            state = self.controller.state
            if state is None:
                self.controller.command("refresh")
                state = self.controller.state
            return dict(state)

        state = self.controller.read_state()
        return {"status": state.status, "stage": state.current_stage, "revision": state.revision}

    def _key(self, state):
        return (self.controller.name, state["stage"], state["revision"])

    def _command(self, action, *, confirmed=False):
        result = self.controller.command(action, confirmed=confirmed)
        if result.get("code") == "EXECUTOR_BUSY":
            raise ValueError("다른 실행이 진행 중입니다. 상태를 확인하세요.")
        if result.get("status") == "busy":
            raise ValueError("다른 요청이 진행 중입니다. 자동 재시도하지 않습니다.")
        return result

    def _advance(self):
        seen = set()
        for _ in range(16):
            self.state = self._snapshot()
            state = self.state
            if state["status"] in TERMINAL:
                return

            key = self._key(state)
            if key in seen:
                raise ValueError("작업 상태가 진행되지 않았습니다. 자동 재시도하지 않습니다.")
            seen.add(key)

            if self.workflow:
                actions = {item["id"] for item in state["actions"]}
                if "execute" in actions:
                    # 이전 세션의 승인은 새 클릭으로 자동 소비하지 않는다.
                    raise ValueError("이미 저장된 승인이 있습니다. 기존 승인과 상태를 별도로 확인하세요.")
                awaiting = "approve" in actions
                can_advance = "next" in actions
            else:
                awaiting = state["status"] == "awaiting_approval"
                can_advance = state["status"] in {"created", "running"}

            if awaiting:
                self._command("preview")
                preview = self.controller.approval_preview if self.workflow else self.controller.preview
                self.summary = preview.to_dict()["summary"] if self.workflow else preview["summary"]
                self.review_key = key
                self.ready = True
                return

            if not can_advance:
                raise ValueError("자동 진행할 수 없는 상태입니다. 로컬 기록을 확인하세요.")

            self._command("next" if self.workflow else "preflight")

        raise ValueError("자동 진행 단계 상한을 넘었습니다.")

    def run(self, action):
        if not self._lock.acquire(blocking=False):
            return {"status": "busy"}

        self.busy = True
        self.pending = None  # 명시적 클릭을 backend 호출 전에 소비한다.
        try:
            if self.error or self.controller.error:
                raise ValueError("이전 요청이 실패했습니다. 자동 재시도하지 않습니다.")

            if action == "start":
                if self.started:
                    raise ValueError("이미 시작한 작업을 다시 진행하지 않습니다.")
                self.started = True
            elif action == "approve":
                if not self.ready or self._key(self._snapshot()) != self.review_key:
                    raise ValueError("화면에 표시한 승인 범위와 현재 단계가 다릅니다.")

                self.ready = False
                self.summary = None
                self._command("approve", confirmed=True)
                self._command("execute", confirmed=True)
            else:
                raise ValueError("허용되지 않은 진행 요청입니다.")

            self._advance()
            return {"status": self.state["status"]}
        except BaseException:
            self.error = True
            self.ready = False
            raise
        finally:
            self.busy = False
            self._lock.release()


def mark_flow_pending(st, flow, action, confirmation_key=None):
    """클릭 callback은 화면 잠금만 예약하며 backend를 호출하지 않는다."""
    if flow.busy or flow.error:
        return
    if confirmation_key is not None and st.session_state.get(confirmation_key) is not True:
        return

    flow.pending = action
    flow.busy = True
