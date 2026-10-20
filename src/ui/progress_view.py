"""동기 실행을 그대로 유지하는 공통 UI 진행 표시. 별도 실행 스레드·polling 없음."""

from contextlib import contextmanager
import time

from src.agents.operation_progress import STAGES, observe_progress, report_progress


def progress_text(event):
    text = STAGES[event.stage]
    if event.total is not None:
        text += f" · 파일 {event.completed:,}/{event.total:,}"

    if event.bytes_done is not None:
        text += f" · 처리 {event.bytes_done / 1024 ** 2:,.1f} MiB"

    return text


@contextmanager
def show_operation_progress(st, label):
    """spinner의 프런트엔드 타이머로 경과 시간을 표시한다. rerun을 예약하지 않는다."""
    placeholder = st.empty()
    last_stage, last_update = None, 0.0

    def display(event):
        nonlocal last_stage, last_update
        now = time.monotonic()
        finished = event.total is not None and event.completed == event.total
        if event.stage == last_stage and now - last_update < 0.5 and not finished:
            return

        placeholder.info(progress_text(event))
        last_stage, last_update = event.stage, now

    try:
        with st.spinner(label, show_time=True), observe_progress(display):
            report_progress("contract_check")
            yield
    finally:
        placeholder.empty()
