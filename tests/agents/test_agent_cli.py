"""실제 데이터 없이 합성 CLI 명령과 대화형 승인 경계를 검증한다."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.agents.approval_store import ApprovalStore
from src.agents.cli import main
from src.agents.state import LocalStateStore


class AgentCliTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)

        self.root = Path(temporary.name)
        self.run = self.root / "demo"
        actor = patch("src.agents.cli._current_actor", return_value="synthetic_os_actor")
        actor.start()
        self.addCleanup(actor.stop)

    def command(self, name, *extra):
        """콘솔 출력을 캡처하며 명령의 종료 코드도 함께 확인한다."""
        output = io.StringIO()
        arguments = [name]

        if name != "plan":
            arguments.extend(("--run-dir", str(self.run)))

        arguments.extend(extra)

        with redirect_stdout(output):
            code = main(arguments)

        return code, output.getvalue()

    def waiting_demo(self):
        """두 번의 단계 호출로 ingestion 승인 대기에 도달한다."""
        self.assertEqual(self.command("init-demo")[0], 0)
        self.assertEqual(self.command("step")[0], 0)
        code, output = self.command("step")

        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output)["code"], "APPROVAL_REQUIRED")

    def approve(self):
        """실제 사용자 대신 독립 합성 응답으로 승인 화면을 검증한다."""
        with patch("src.agents.cli.os.isatty", return_value=True):
            with patch("src.agents.cli.secrets.token_hex", return_value="1234abcd"):
                with patch("builtins.input", return_value="DEMO-1234ABCD"):
                    code, output = self.command("approve")

        self.assertEqual(code, 0)

        decoder = json.JSONDecoder()
        _, position = decoder.raw_decode(output)

        return decoder.raw_decode(output[position:].strip())[0]

    def test_plan_and_status(self):
        """계획 조회는 실행을 만들지 않으며 상태 조회는 revision을 바꾸지 않는다."""
        self.assertEqual(self.command("plan")[0], 0)
        self.assertFalse(self.run.exists())

        self.assertEqual(self.command("init-demo")[0], 0)
        code, output = self.command("status")
        before = json.loads(output)

        self.assertEqual(code, 0)
        self.assertEqual(before["stage"], "preflight")
        self.assertEqual(self.command("status")[1], output)

    def test_profile_plan_is_read_only(self):
        """프로파일 단계 조회는 실제 입력을 읽거나 실행 폴더를 생성하지 않는다."""
        root = Path(__file__).resolve().parents[2]
        profile = root / "configs/agents/profiles/tabular_example.json"
        code, output = self.command("plan", "--profile", str(profile))
        result = json.loads(output)

        self.assertEqual(code, 0)
        self.assertTrue(result["planning_only"])
        self.assertFalse(result["source_bound"])
        self.assertEqual(result["workflow_id"], "eda_review_v1")
        self.assertNotIn("prepare_cache", result["stages"])
        self.assertFalse(self.run.exists())

    def test_interactive_approval_and_completion(self):
        """생성·대기·승인·재개·최종 검증의 전체 합성 흐름을 수행한다."""
        self.waiting_demo()
        approval = self.approve()
        code, output = self.command("resume", "--approval-id", approval["approval_id"])

        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output)["code"], "STEP_COMPLETED")
        self.assertEqual(self.command("step")[0], 0)
        self.assertEqual(self.command("step")[0], 0)
        self.assertEqual(json.loads(self.command("status")[1])["status"], "completed")

    def test_declined_and_noninteractive_approval(self):
        """대화형 확인 거절과 파이프 입력은 승인을 발급하지 않는다."""
        self.waiting_demo()

        with patch("src.agents.cli.os.isatty", return_value=False):
            self.assertEqual(self.command("approve")[0], 1)

        with patch("src.agents.cli.os.isatty", return_value=True):
            with patch("builtins.input", return_value="wrong"):
                code, output = self.command("approve")

        self.assertEqual(code, 0)
        self.assertIn("APPROVAL_DECLINED", output)

    def test_different_actor_and_duplicate_init(self):
        """다른 계정의 실행 조작과 기존 폴더 덮어쓰기를 거부한다."""
        self.command("init-demo")
        self.assertEqual(self.command("init-demo")[0], 1)

        with patch("src.agents.cli._current_actor", return_value="other_actor"):
            self.assertEqual(self.command("status")[0], 1)

    def test_cancel_revokes_issued_approval(self):
        """취소된 실행의 미사용 승인만 폐기하고 재개를 차단한다."""
        self.waiting_demo()
        approval = self.approve()
        store = ApprovalStore(self.run / "approvals.sqlite3")

        self.assertEqual(self.command("cancel")[0], 0)
        self.assertEqual(store.get(approval["approval_id"]).status, "revoked")
        code, output = self.command("resume", "--approval-id", approval["approval_id"])

        self.assertEqual(json.loads(output)["code"], "RUN_CANCEL_REQUESTED")

    def test_plain_directory_is_not_a_demo(self):
        """데모 표식이 없는 실제 데이터·임의 폴더는 실행 대상으로 받지 않는다."""
        self.run.mkdir()

        self.assertEqual(self.command("step")[0], 1)

    def test_state_change_during_confirmation_is_rejected(self):
        """사용자가 화면을 확인하는 도중 상태가 바뀌면 승인을 발급하지 않는다."""
        self.waiting_demo()

        def cancel_while_reading(prompt):
            states = LocalStateStore(self.run)
            state = states.load()
            state.cancel_requested = True
            states.save(state, expected_revision=state.revision)

            return "DEMO-1234ABCD"

        with patch("src.agents.cli.os.isatty", return_value=True):
            with patch("src.agents.cli.secrets.token_hex", return_value="1234abcd"):
                with patch("builtins.input", side_effect=cancel_while_reading):
                    self.assertEqual(self.command("approve")[0], 1)


if __name__ == "__main__":
    unittest.main()
