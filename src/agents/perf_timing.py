"""측정용 호출 시간 기록. 환경 변수 WAFERGUARD_TIMING_LOG에 파일 경로가 있을 때만 쓰고, 없으면 아무 일도 하지 않는다.

기록은 고정된 호출 종류·단계 이름·시작 벽시계 시각·단조 시계 경과 초뿐이다. 인자·토큰·데이터 값·자유 문자열은 쓰지 않는다.
승인·검증·실행 순서는 바꾸지 않으며, 기록 실패는 원래 호출의 결과를 바꾸지 않는다.
"""

from contextlib import contextmanager
import json
import os
import time

KINDS = frozenset({"status", "preview_approval", "record_delegated_approval", "next", "create_step", "report_sources"})
STEPS = frozenset({"split", "export", "candidate", "apply", "handoff", "-"})


@contextmanager
def timed(kind, step="-"):
    path = os.environ.get("WAFERGUARD_TIMING_LOG")
    if not path or kind not in KINDS or step not in STEPS:
        yield
        return
    wall, start = time.time(), time.perf_counter()
    failed = True
    try:
        yield
        failed = False
    finally:
        record = {"kind": kind, "step": step, "start_wall": round(wall, 3),
                  "elapsed_s": round(time.perf_counter() - start, 4), "failed": failed}
        try:
            with open(path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
        except OSError:
            pass
