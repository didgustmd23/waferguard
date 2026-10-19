"""모델 전송 명령의 단일 소비 기록. 대화 원문·응답·승인 토큰은 저장하지 않는다."""

from contextlib import closing
from pathlib import Path
import sqlite3


DEFAULT_LEDGER = Path(__file__).resolve().parents[2] / "logs/agents/ui/model_commands.sqlite3"


class ModelCommandLedger:
    """명령 ID당 호출을 최대 한 번 시도한다. 원격 처리 성공의 exactly-once 보장은 아니다."""

    def __init__(self, path=DEFAULT_LEDGER):
        self.path = Path(path)

    def claim(self, command_id, binding):
        """HTTP 전에 원자적으로 소비한다. 중단된 pending도 자동 재실행하지 않는다."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

        with closing(sqlite3.connect(self.path, timeout=5)) as connection:
            with connection:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS model_commands ("
                    "command_id TEXT PRIMARY KEY, binding TEXT NOT NULL, "
                    "status TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
                )
                cursor = connection.execute(
                    "INSERT OR IGNORE INTO model_commands(command_id, binding, status) VALUES (?, ?, 'pending')",
                    (command_id, binding),
                )
                return cursor.rowcount == 1

    def finish(self, command_id, binding, status):
        """원문 대신 완료/확인 불가만 기록한다. 결과 기록 실패도 재호출 근거가 아니다."""
        if status not in ("completed", "unknown"):
            raise ValueError("지원하지 않는 모델 명령 상태입니다.")

        with closing(sqlite3.connect(self.path, timeout=5)) as connection:
            with connection:
                connection.execute(
                    "UPDATE model_commands SET status = ? WHERE command_id = ? AND binding = ? AND status = 'pending'",
                    (status, command_id, binding),
                )
