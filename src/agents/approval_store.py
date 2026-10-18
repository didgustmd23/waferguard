"""로컬 SQLite에서 승인 단일 소비와 작업 예약을 함께 관리한다."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
import uuid

from .approvals import ApprovalRecord, ApprovalScope, validate_approval


@dataclass(frozen=True)
class ReservationResult:
    """예약 결과 코드와 성공한 작업의 로컬 식별값."""

    code: str
    reservation_id: str | None = None


def _utc_now():
    """실행기의 현재 UTC 시각을 가져온다."""
    return datetime.now(timezone.utc)


def _scope_json(scope):
    """동일한 승인 대상은 항상 같은 문자열로 저장한다."""
    if not isinstance(scope, ApprovalScope):
        raise TypeError("승인 대상은 ApprovalScope여야 합니다.")

    return json.dumps(asdict(scope), ensure_ascii=False, sort_keys=True)


# ==========================================
# 승인 저장소
# - 각 호출은 별도 DB 연결을 사용하여 동시 접근을 처리한다.
# - BEGIN IMMEDIATE로 쓰기 권한을 얻은 뒤 최신 승인을 검사한다.
# - 승인 소비와 작업 예약은 함께 커밋하거나 함께 롤백한다.
# - 소비된 승인은 장애가 나더라도 자동 복구·재발급하지 않는다.
# ==========================================
class ApprovalStore:
    """신뢰하는 로컬 실행기가 사용하는 승인 영속 저장소.

    clock 주입은 합성 테스트용이다. 실제 실행에서는 기본 UTC 시계를 사용한다.
    사용자 인증·승인 화면·DB 접근 권한은 상위 실행기에서 연결해야 한다.
    """

    def __init__(self, path, *, clock=_utc_now):
        self.path = Path(path).resolve()
        self.clock = clock

        self.path.parent.mkdir(parents=True, exist_ok=True)

        with self._transaction() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS approvals (
                    approval_id TEXT PRIMARY KEY,
                    scope_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('pending', 'consumed', 'revoked', 'expired')
                    )
                )
            """)

            connection.execute("""
                CREATE TABLE IF NOT EXISTS reservations (
                    reservation_id TEXT PRIMARY KEY,
                    approval_id TEXT NOT NULL UNIQUE REFERENCES approvals(approval_id),
                    scope_json TEXT NOT NULL UNIQUE,
                    reserved_at TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status = 'reserved')
                )
            """)

            # 기존 승인 예약 테이블을 유지하면서 실행 진행 이력을 별도로 확장한다.
            connection.execute("""
                CREATE TABLE IF NOT EXISTS task_runs (
                    task_id TEXT PRIMARY KEY,
                    scope_json TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL CHECK (
                        status IN ('reserved', 'running', 'verified', 'committed', 'failed')
                    )
                )
            """)

    @contextmanager
    def _transaction(self):
        """DB 연결을 닫고 실패 시 미확정 쓰기를 취소한다."""
        connection = None

        try:
            connection = sqlite3.connect(self.path, timeout=10, isolation_level=None)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("BEGIN IMMEDIATE")

            yield connection

            connection.commit()
        except Exception as error:
            if connection is not None:
                connection.rollback()

            if isinstance(error, sqlite3.Error):
                raise RuntimeError("승인 저장소 작업에 실패했습니다.") from error

            raise
        finally:
            if connection is not None:
                connection.close()

    def _now(self):
        """시계가 시간대 있는 시각을 반환하는지 검사한다."""
        now = self.clock()

        if not isinstance(now, datetime) or now.utcoffset() is None:
            raise ValueError("승인 저장소 시각에는 시간대 정보가 필요합니다.")

        return now

    @staticmethod
    def _record(row):
        """저장된 필드를 검증 가능한 승인 객체로 복원한다."""
        try:
            scope = ApprovalScope(**json.loads(row["scope_json"]))

            return ApprovalRecord(
                approval_id=row["approval_id"],
                scope=scope,
                created_at=datetime.fromisoformat(row["created_at"]),
                expires_at=datetime.fromisoformat(row["expires_at"]),
                status=row["status"],
            )
        except (TypeError, ValueError) as error:
            raise ValueError("저장된 승인 기록이 올바르지 않습니다.") from error

    def record_confirmed_approval(self, scope, *, lifetime=timedelta(minutes=30)):
        """사용자 확인이 끝난 승인을 기록한다. 호출 자체는 사용자 인증이 아니다."""
        scope_json = _scope_json(scope)

        if not isinstance(lifetime, timedelta) or lifetime <= timedelta(0):
            raise ValueError("승인 유효기간은 양수여야 합니다.")

        with self._transaction() as connection:
            now = self._now()
            record = ApprovalRecord(
                approval_id=uuid.uuid4().hex,
                scope=scope,
                created_at=now,
                expires_at=now + lifetime,
            )

            connection.execute(
                "INSERT INTO approvals VALUES (?, ?, ?, ?, ?)",
                (
                    record.approval_id, scope_json,
                    record.created_at.isoformat(), record.expires_at.isoformat(),
                    record.status,
                ),
            )

        return record

    def get(self, approval_id):
        """현재 저장된 승인을 조회한다. 존재하지 않으면 None을 반환한다."""
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM approvals WHERE approval_id = ?", (approval_id,),
            ).fetchone()

            return None if row is None else self._record(row)

    def revoke(self, approval_id):
        """미사용 승인만 취소한다. 소비된 승인은 되돌리지 않는다."""
        with self._transaction() as connection:
            updated = connection.execute(
                "UPDATE approvals SET status = 'revoked' "
                "WHERE approval_id = ? AND status = 'pending'",
                (approval_id,),
            )

            return updated.rowcount == 1

    def revoke_run(self, run_id):
        """취소된 실행에 속한 미사용 승인만 한 트랜잭션에서 폐기한다."""
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("취소할 실행 ID가 필요합니다.")

        with self._transaction() as connection:
            rows = connection.execute(
                "SELECT * FROM approvals WHERE status = 'pending'",
            ).fetchall()
            approval_ids = [
                row["approval_id"]
                for row in rows
                if self._record(row).scope.run_id == run_id
            ]

            connection.executemany(
                "UPDATE approvals SET status = 'revoked' WHERE approval_id = ?",
                ((approval_id,) for approval_id in approval_ids),
            )

            return len(approval_ids)

    def consume_and_reserve(self, approval_id, scope):
        """최신 승인 검사·단일 소비·작업 예약을 하나의 트랜잭션으로 수행한다."""
        scope_json = _scope_json(scope)

        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM approvals WHERE approval_id = ?", (approval_id,),
            ).fetchone()

            if row is None:
                return ReservationResult("APPROVAL_NOT_FOUND")

            record = self._record(row)
            now = self._now()
            code = validate_approval(record, scope, now=now)

            if code != "APPROVAL_VALID":
                if code == "APPROVAL_EXPIRED" and record.status == "pending":
                    connection.execute(
                        "UPDATE approvals SET status = 'expired' WHERE approval_id = ?",
                        (approval_id,),
                    )

                return ReservationResult(code)

            # 새 승인을 발급받아도 동일한 작업 예약을 중복 생성하지 않는다.
            existing = connection.execute(
                "SELECT reservation_id FROM reservations WHERE scope_json = ?",
                (scope_json,),
            ).fetchone()

            if existing is not None:
                return ReservationResult("TASK_ALREADY_RESERVED")

            reservation_id = uuid.uuid4().hex

            connection.execute(
                "UPDATE approvals SET status = 'consumed' WHERE approval_id = ?",
                (approval_id,),
            )

            connection.execute(
                "INSERT INTO reservations VALUES (?, ?, ?, ?, 'reserved')",
                (reservation_id, approval_id, scope_json, now.isoformat()),
            )

            connection.execute(
                "INSERT INTO task_runs VALUES (?, ?, 'reserved')",
                (reservation_id, scope_json),
            )

            return ReservationResult("TASK_RESERVED", reservation_id)

    def get_task(self, scope):
        """정확히 같은 작업 범위의 진행 이력을 조회한다."""
        scope_json = _scope_json(scope)

        with self._transaction() as connection:
            row = connection.execute(
                "SELECT task_id, status FROM task_runs WHERE scope_json = ?",
                (scope_json,),
            ).fetchone()

            if row is not None:
                return dict(row)

            # 이전 버전에서 예약만 남은 경우에도 자동 재실행하지 않는다.
            legacy = connection.execute(
                "SELECT reservation_id FROM reservations WHERE scope_json = ?",
                (scope_json,),
            ).fetchone()

            if legacy is not None:
                return {"task_id": legacy["reservation_id"], "status": "reserved"}

            return None

    def reserve_task(self, scope):
        """승인이 불필요한 등록 도구도 작업 예약을 한 번만 생성한다."""
        scope_json = _scope_json(scope)

        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT task_id FROM task_runs WHERE scope_json = ?",
                (scope_json,),
            ).fetchone()
            legacy = connection.execute(
                "SELECT reservation_id FROM reservations WHERE scope_json = ?",
                (scope_json,),
            ).fetchone()

            if existing is not None or legacy is not None:
                return ReservationResult("TASK_ALREADY_RESERVED")

            task_id = uuid.uuid4().hex

            connection.execute(
                "INSERT INTO task_runs VALUES (?, ?, 'reserved')",
                (task_id, scope_json),
            )

            return ReservationResult("TASK_RESERVED", task_id)

    def transition_task(self, scope, task_id, *, expected, status):
        """기대 상태가 일치할 때만 작업 진행 상태를 변경한다."""
        allowed = {
            ("reserved", "running"),
            ("running", "verified"),
            ("verified", "committed"),
            ("running", "failed"),
        }

        if (expected, status) not in allowed:
            raise ValueError("허용하지 않는 작업 상태 전이입니다.")

        scope_json = _scope_json(scope)

        with self._transaction() as connection:
            updated = connection.execute(
                "UPDATE task_runs SET status = ? "
                "WHERE task_id = ? AND scope_json = ? AND status = ?",
                (status, task_id, scope_json, expected),
            )

            if updated.rowcount != 1:
                raise ValueError("작업 상태가 기대한 값과 일치하지 않습니다.")
