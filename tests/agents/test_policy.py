"""정책의 단계 제한과 보안 차단 우선순위를 검증한다."""

import unittest

from src.agents.policy import PolicyDecision, decide_action
from src.agents.state import RunState


# ==========================================
# 규칙 기반 실행 정책 검증
# - 현재 단계에 맞는 작업만 허용하는지 확인
# - 보안 잠금이 취소 요청보다 우선하는지 확인
# - 모순된 판단 결과의 생성을 차단하는지 확인
# ==========================================
class PolicyTest(unittest.TestCase):
    def setUp(self):
        """각 테스트에서 사용할 독립적인 실행 상태를 준비한다."""
        self.state = RunState(
            run_id="a" * 32,
            actor_id="local_operator",
        )

    def test_stage_restriction(self):
        """현재 단계 작업은 허용하고 다른 단계 작업은 거부한다."""
        allowed = decide_action(self.state, "run_preflight")
        denied = decide_action(self.state, "run_ingest")

        self.assertTrue(allowed.allowed)
        self.assertEqual(allowed.code, "ACTION_ALLOWED")

        self.assertFalse(denied.allowed)
        self.assertEqual(denied.code, "ACTION_NOT_ALLOWED")

    def test_security_lock_has_priority(self):
        """보안 잠금과 취소 요청이 함께 있으면 잠금 코드를 반환한다."""
        self.state.blockers = ["SECURITY_LOCKED"]
        self.state.cancel_requested = True

        decision = decide_action(self.state, "run_preflight")

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.code, "SECURITY_LOCKED")

    def test_invalid_decisions_are_rejected(self):
        """잘못된 타입·미등록 코드·모순된 판단을 거부한다."""
        invalid_cases = [
            (1, "ACTION_ALLOWED"),
            (False, "UNKNOWN"),
            (True, "SECURITY_LOCKED"),
            (False, "ACTION_ALLOWED"),
        ]

        for allowed, code in invalid_cases:
            with self.subTest(allowed=allowed, code=code):
                with self.assertRaises(ValueError):
                    PolicyDecision(allowed=allowed, code=code)


if __name__ == "__main__":
    unittest.main()