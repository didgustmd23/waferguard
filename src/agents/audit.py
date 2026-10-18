"""원문 데이터 없이 남기는 로컬 보안 감사 이벤트."""

from datetime import datetime, timezone
import json
from pathlib import Path


EVENT_TYPES = frozenset({
    "run_created", "state_saved", "action_allowed", "action_denied",
    "approval_requested", "approval_consumed", "security_lock",
})


def append_event(path, event_type, code, *, stage, revision):
    """고정 스키마 이벤트를 JSONL 한 행으로 추가한다.

    호출자는 원본·경로·예외 원문·provider 응답을 인자로 넘길 수 없다.
    감사 기록 자체가 실패하면 실행기는 새 작업을 시작하면 안 된다.
    """
    if event_type not in EVENT_TYPES:
        raise ValueError("허용하지 않는 감사 이벤트입니다.")

    if not all(isinstance(value, str) and value for value in (code, stage)):
        raise ValueError("감사 이벤트 코드 형식이 올바르지 않습니다.")

    if type(revision) is not int or revision < 0:
        raise ValueError("감사 revision 형식이 올바르지 않습니다.")

    record = {
        "event_type": event_type,
        "code": code,
        "stage": stage,
        "revision": revision,
        "recorded_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }

    target = Path(path)
    write_mode = "x" if not target.exists() else "a"

    try:
        with target.open(write_mode, encoding="utf-8") as stream:

            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    except OSError as error:
        raise RuntimeError("감사 기록 저장에 실패했습니다.") from error

