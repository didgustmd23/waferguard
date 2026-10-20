"""부모 실행기의 관찰 전용 진행 신호. 승인·DB·worker·LLM을 호출하지 않는다."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


STAGES = {
    "contract_check": "입력·코드·작업 계약 확인",
    "runtime_inventory": "격리 라이브러리 목록·해시 확인",
    "profile_create": "AppContainer 프로필 생성",
    "permissions": "격리 폴더 권한 설정",
    "runtime_copy": "격리 라이브러리 복사·검증",
    "tools_copy": "실행 코드 복사·검증",
    "worker_wait": "격리 worker 실행·결과 대기",
    "result_check": "실행 결과·자원·저장 계약 확인",
    "cleanup": "AppContainer 프로필 정리",
}
_observer = ContextVar("operation_progress_observer", default=None)


@dataclass(frozen=True)
class ProgressEvent:
    stage: str
    completed: int | None = None
    total: int | None = None
    bytes_done: int | None = None


@contextmanager
def observe_progress(callback):
    """현재 호출 문맥에만 연결한다. 다른 UI 세션·스레드와 구독자를 공유하지 않는다."""
    token = _observer.set(callback)
    try:
        yield
    finally:
        _observer.reset(token)


def report_progress(stage, *, completed=None, total=None, bytes_done=None):
    """고정 단계 ID와 비민감 처리량만 전달한다. 오류·진행 신호는 성공 증거가 아니다."""
    callback = _observer.get()
    if callback is None or stage not in STAGES:
        return

    if any(value is not None and (type(value) is not int or value < 0)
           for value in (completed, total, bytes_done)):
        return

    if (completed is None) != (total is None) or (total is not None and completed > total):
        return

    try:
        callback(ProgressEvent(stage, completed, total, bytes_done))
    except Exception:
        # 표시 장애만으로 승인·처리 결과를 바꾸거나 같은 작업을 재시도하지 않는다.
        pass
