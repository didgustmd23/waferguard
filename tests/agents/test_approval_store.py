"""합성 승인으로 SQLite의 단일 소비·동시성·롤백을 확인한다."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import tempfile
from threading import Barrier
import unittest
from unittest.mock import patch

from src.agents.approvals import ApprovalScope
from src.agents.approval_store import ApprovalStore


class ApprovalStoreTest(unittest.TestCase):
    def setUp(self):
        """테스트마다 독립된 임시 DB와 가상 입력을 만든다."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)

        self.path = Path(temporary.name) / "approvals.sqlite3"
        self.now = datetime(2026, 10, 15, 12, tzinfo=timezone.utc)
        self.store = ApprovalStore(self.path, clock=lambda: self.now)
        self.scope = ApprovalScope(
            actor_id="test_operator", run_id="a" * 32,
            stage="ingest", tool_id="run_ingest",
            target_id="test_target", arguments_id="test_arguments",
            source_id="test_source", config_id="test_config",
            plan_id="test_plan", policy_version="test_policy",
        )

    def reservation_count(self):
        """내부 DB를 직접 조회해 실제 작업 예약 수를 확인한다."""
        # SQLite 문맥은 트랜잭션만 끝낸다. closing으로 파일 연결도 닫는다.
        with closing(sqlite3.connect(self.path)) as connection, connection:
            return connection.execute("SELECT COUNT(*) FROM reservations").fetchone()[0]

    def test_reservation_count_closes_connection_even_on_query_failure(self):
        """조회 성공·실패 모두 연결을 닫고 GC나 임시 폴더 삭제에 기대지 않는다."""
        for missing_table in (False, True):
            with self.subTest(missing_table=missing_table):
                database = ":memory:" if missing_table else self.path
                connection = sqlite3.connect(database)

                with patch.object(sqlite3, "connect", return_value=connection):
                    if missing_table:
                        with self.assertRaises(sqlite3.OperationalError):
                            self.reservation_count()
                    else:
                        self.assertEqual(self.reservation_count(), 0)

                with self.assertRaises(sqlite3.ProgrammingError):
                    connection.execute("SELECT 1")

    def test_roundtrip_and_single_consumption(self):
        """재연결 후에도 승인과 소비 상태가 유지되며 재사용을 거부한다."""
        record = self.store.record_confirmed_approval(self.scope)
        reopened = ApprovalStore(self.path, clock=lambda: self.now)

        self.assertEqual(reopened.get(record.approval_id), record)

        first = reopened.consume_and_reserve(record.approval_id, self.scope)
        second = self.store.consume_and_reserve(record.approval_id, self.scope)

        self.assertEqual(first.code, "TASK_RESERVED")
        self.assertIsNotNone(first.reservation_id)
        self.assertEqual(second.code, "APPROVAL_ALREADY_CONSUMED")
        self.assertEqual(self.reservation_count(), 1)
        self.assertEqual(self.store.get(record.approval_id).status, "consumed")
        self.assertFalse(self.store.revoke(record.approval_id))

    def test_rejected_scopes_do_not_consume(self):
        """대상 불일치와 미등록 승인은 예약을 만들지 않는다."""
        record = self.store.record_confirmed_approval(self.scope)
        changed = replace(self.scope, config_id="changed_config")

        mismatch = self.store.consume_and_reserve(record.approval_id, changed)
        missing = self.store.consume_and_reserve("unknown", self.scope)

        self.assertEqual(mismatch.code, "APPROVAL_SCOPE_MISMATCH")
        self.assertEqual(missing.code, "APPROVAL_NOT_FOUND")
        self.assertEqual(self.store.get(record.approval_id).status, "pending")
        self.assertEqual(self.reservation_count(), 0)

    def test_expiry_and_revocation(self):
        """만료 경계와 취소 상태를 DB에 반영하고 실행을 거부한다."""
        expired = self.store.record_confirmed_approval(self.scope)
        revoked = self.store.record_confirmed_approval(self.scope)

        self.assertTrue(self.store.revoke(revoked.approval_id))

        self.now = expired.expires_at
        expired_result = self.store.consume_and_reserve(expired.approval_id, self.scope)
        revoked_result = self.store.consume_and_reserve(revoked.approval_id, self.scope)

        self.assertEqual(expired_result.code, "APPROVAL_EXPIRED")
        self.assertEqual(revoked_result.code, "APPROVAL_REVOKED")
        self.assertEqual(self.store.get(expired.approval_id).status, "expired")
        self.assertEqual(self.reservation_count(), 0)

    def test_two_connections_reserve_only_once(self):
        """별도 연결의 동시 소비 요청 중 하나만 작업을 예약한다."""
        record = self.store.record_confirmed_approval(self.scope)
        other = ApprovalStore(self.path, clock=lambda: self.now)
        barrier = Barrier(2)

        def consume(store):
            barrier.wait(timeout=10)

            return store.consume_and_reserve(record.approval_id, self.scope).code

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(consume, store) for store in (self.store, other)]
            codes = [future.result(timeout=15) for future in futures]

        self.assertCountEqual(codes, ["TASK_RESERVED", "APPROVAL_ALREADY_CONSUMED"])
        self.assertEqual(self.reservation_count(), 1)

    def test_reservation_failure_rolls_back_consumption(self):
        """예약 저장 실패를 강제하면 앞서 수행한 승인 소비도 롤백된다."""
        record = self.store.record_confirmed_approval(self.scope)

        # 승인 UPDATE 뒤 예약 INSERT가 실패하도록 합성 DB에만 장애를 주입한다.
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("""
                CREATE TRIGGER fail_reservation BEFORE INSERT ON reservations
                BEGIN SELECT RAISE(ABORT, 'synthetic_failure'); END
            """)

        with self.assertRaisesRegex(RuntimeError, "승인 저장소"):
            self.store.consume_and_reserve(record.approval_id, self.scope)

        self.assertEqual(self.store.get(record.approval_id).status, "pending")
        self.assertEqual(self.reservation_count(), 0)

    def test_new_approval_cannot_duplicate_reserved_task(self):
        """동일 범위에 새 승인을 발급해도 예약된 작업을 중복 예약하지 않는다."""
        first = self.store.record_confirmed_approval(self.scope)
        second = self.store.record_confirmed_approval(self.scope)

        self.store.consume_and_reserve(first.approval_id, self.scope)
        result = self.store.consume_and_reserve(second.approval_id, self.scope)

        self.assertEqual(result.code, "TASK_ALREADY_RESERVED")
        self.assertEqual(self.store.get(second.approval_id).status, "pending")
        self.assertEqual(self.reservation_count(), 1)

    def test_invalid_lifetime_and_corrupted_record(self):
        """잘못된 유효기간과 변조된 기록에서는 작업을 예약하지 않는다."""
        with self.assertRaises(ValueError):
            self.store.record_confirmed_approval(self.scope, lifetime=timedelta(0))

        record = self.store.record_confirmed_approval(self.scope)

        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute(
                "UPDATE approvals SET scope_json = '{}' WHERE approval_id = ?",
                (record.approval_id,),
            )

        with self.assertRaisesRegex(ValueError, "저장된 승인 기록"):
            self.store.consume_and_reserve(record.approval_id, self.scope)

        self.assertEqual(self.reservation_count(), 0)


if __name__ == "__main__":
    unittest.main()
