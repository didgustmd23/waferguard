"""승인의 유효기간·대상 변경·사용 상태를 검증한다."""

from dataclasses import fields, replace
from datetime import datetime, timedelta, timezone
import unittest

from src.agents.approvals import (
    ApprovalRecord,
    ApprovalScope,
    validate_approval,
)


class ApprovalTest(unittest.TestCase):
    def setUp(self):
        """실제 데이터 없이 고정 시각과 가상 승인 대상을 준비한다."""
        self.now = datetime(2026, 10, 15, 12, tzinfo=timezone.utc)

        self.scope = ApprovalScope(
            actor_id="local_operator",
            run_id="a" * 32,
            stage="ingest",
            tool_id="run_ingest",
            target_id="target_v1",
            arguments_id="arguments_v1",
            source_id="source_v1",
            config_id="config_v1",
            plan_id="plan_v1",
            policy_version="policy_v1",
        )

        self.record = ApprovalRecord(
            approval_id="approval_test",
            scope=self.scope,
            created_at=self.now,
            expires_at=self.now + timedelta(minutes=30),
        )

    # ==========================================
    # 승인 유효기간 경계 검사
    # - 생성 시각부터 사용 가능
    # - 만료 시각과 같으면 사용 불가
    # ==========================================
    def test_time_boundaries(self):
        cases = [
            (self.now - timedelta(seconds=1), "APPROVAL_NOT_YET_VALID"),
            (self.now, "APPROVAL_VALID"),
            (self.record.expires_at, "APPROVAL_EXPIRED"),
        ]

        for check_time, expected in cases:
            with self.subTest(check_time=check_time):
                result = validate_approval(
                    self.record,
                    self.scope,
                    now=check_time,
                )

                self.assertEqual(result, expected)

    def test_each_scope_field_is_bound(self):
        """승인 대상의 어느 필드든 바뀌면 기존 승인을 거부한다."""
        for field in fields(ApprovalScope):
            changed_scope = replace(
                self.scope,
                **{field.name: "different_value"},
            )

            with self.subTest(field=field.name):
                result = validate_approval(
                    self.record,
                    changed_scope,
                    now=self.now,
                )

                self.assertEqual(result, "APPROVAL_SCOPE_MISMATCH")

    def test_unusable_statuses(self):
        """소비·취소·만료 상태의 승인을 거부한다."""
        cases = [
            ("consumed", "APPROVAL_ALREADY_CONSUMED"),
            ("revoked", "APPROVAL_REVOKED"),
            ("expired", "APPROVAL_EXPIRED"),
        ]

        for status, expected in cases:
            record = replace(self.record, status=status)

            with self.subTest(status=status):
                result = validate_approval(
                    record,
                    self.scope,
                    now=self.now,
                )

                self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()