"""실제 LangGraph 중단·재개를 사용하고 모델 운송은 합성 응답으로 대체한다."""

from contextlib import redirect_stdout
from io import StringIO
import json
import os
import unittest
from unittest.mock import patch

from langsmith import tracing_context

from src.agents import langchain_local, llm_dialogue_cli
from src.agents.llm_dialogue import MAX_MODEL_CALLS, ProposalDialogue
from src.agents.local_ollama import LocalProposalError
from tests.agents.test_llm_proposal import valid_content


def clarification():
    return json.dumps({"schema_version": "llm_proposal_v1", "status": "needs_clarification",
                       "pipeline": None, "questions": ["원하시는 검사를 알려주세요."]}, ensure_ascii=False)


class ProposalDialogueTest(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(langchain_local, "request_local_proposal", return_value=valid_content())
        self.model = patcher.start()
        self.addCleanup(patcher.stop)
        self.session = ProposalDialogue()
        self.addCleanup(self.session.close)

    def send(self, text="원하는 검사 종류는 아직 정하지 않았습니다."):
        return self.session.send(text, confirmed=True)

    def test_preview_and_missing_confirmation_do_not_call_model(self):
        self.assertFalse(self.session.preview("품질 검사를 해주세요.")["model_called"])

        with self.assertRaises(LocalProposalError):
            self.session.send("품질 검사를 해주세요.")

        self.model.assert_not_called()

    def test_direct_proposal_has_no_execution_authority_or_rerun(self):
        result = self.send()
        self.assertEqual(result["status"], "proposal_validated")
        self.assertEqual(result["model_calls"], 1)
        self.assertFalse(result["execution_approved"])
        self.assertFalse(result["worker_started"])

        with self.assertRaises(LocalProposalError):
            self.send()

        self.model.assert_called_once()

    def test_interrupt_resume_preserves_context_without_repeating_previous_model_call(self):
        self.model.side_effect = [clarification(), valid_content()]
        first = self.send("알아서 전처리해 주세요.")
        self.assertEqual(first["status"], "awaiting_answer")
        preview = self.session.preview("품질 검사 후 결측률을 확인해 주세요.")
        self.assertEqual(self.model.call_count, 1)
        result = self.send("품질 검사 후 결측률을 확인해 주세요.")
        self.assertEqual(result["status"], "proposal_validated")
        self.assertEqual(self.model.call_count, 2)
        sent = self.model.call_args.args[0]
        self.assertEqual(sent, preview["payload"])
        text = sent["messages"][1]["content"]
        self.assertIn("알아서 전처리해 주세요.", text)
        self.assertIn("원하시는 검사를 알려주세요.", text)
        self.assertIn("품질 검사 후 결측률을 확인해 주세요.", text)

    def test_clarification_calls_are_bounded(self):
        self.model.return_value = clarification()

        for index in range(MAX_MODEL_CALLS):
            result = self.send("원하는 작업을 더 확인하고 싶습니다.")
            expected = "clarification_limit_reached" if index == MAX_MODEL_CALLS - 1 else "awaiting_answer"
            self.assertEqual(result["status"], expected)

        with self.assertRaises(LocalProposalError):
            self.send()

        self.assertEqual(self.model.call_count, MAX_MODEL_CALLS)

    def test_accumulated_text_limit_blocks_before_next_model_call(self):
        self.model.return_value = clarification()
        self.send("가" * 1500)

        with self.assertRaises(ValueError):
            self.send("나" * 1500)

        self.model.assert_called_once()
        self.model.return_value = valid_content()
        self.assertEqual(self.send("결측 분석만 요청합니다.")["status"], "proposal_validated")

    def test_model_failure_closes_session_and_never_retries(self):
        self.model.side_effect = LocalProposalError("OLLAMA_CONNECTION_FAILED")

        with self.assertRaises(LocalProposalError):
            self.send()

        with self.assertRaises(LocalProposalError):
            self.send()

        self.model.assert_called_once()

    def test_invalid_contract_closes_session(self):
        self.model.return_value = '{"approved":true}'

        with self.assertRaises(LocalProposalError) as caught:
            self.send()

        self.assertEqual(caught.exception.code, "PROPOSAL_CONTRACT_REJECTED")
        self.model.assert_called_once()

    def test_sessions_do_not_share_history(self):
        self.model.side_effect = [clarification(), valid_content()]
        self.send("첫 세션의 합성 요청입니다.")
        other = ProposalDialogue()
        self.addCleanup(other.close)
        other.send("두 번째 세션의 합성 요청입니다.", confirmed=True)
        text = self.model.call_args.args[0]["messages"][1]["content"]
        self.assertNotIn("첫 세션", text)
        self.assertEqual(text, "두 번째 세션의 합성 요청입니다.")

    def test_graph_and_model_both_disable_tracing(self):
        with patch.dict(os.environ, {"LANGSMITH_TRACING": "true"}), tracing_context(enabled=True):
            with patch("langsmith.Client.create_run") as trace:
                self.send()
                trace.assert_not_called()

    def test_returned_questions_cannot_mutate_graph_history(self):
        self.model.side_effect = [clarification(), valid_content()]
        result = self.send()
        result["proposal"]["questions"][0] = "외부에서 변조한 질문"
        self.send("결측 분석을 요청합니다.")
        text = self.model.call_args.args[0]["messages"][1]["content"]
        self.assertNotIn("외부에서 변조한 질문", text)


class DialogueCliTest(unittest.TestCase):
    def run_cli(self, answers):
        with patch("sys.argv", ["dialogue", "--example", "clarify"]):
            with patch("builtins.input", side_effect=answers), redirect_stdout(StringIO()) as output:
                with patch("sys.stdin.isatty", return_value=True), patch("sys.stdout.isatty", return_value=True):
                    code = llm_dialogue_cli.main()

        return code, output.getvalue()

    def test_decline_never_calls_model(self):
        with patch.object(langchain_local, "request_local_proposal") as model:
            code, output = self.run_cli(["취소"])

        self.assertEqual(code, 0)
        self.assertIn("dialogue_cancelled", output)
        model.assert_not_called()

    def test_each_answer_requires_send_and_no_automatic_save(self):
        with patch.object(langchain_local, "request_local_proposal", side_effect=[clarification(), valid_content()]) as model:
            with patch.object(llm_dialogue_cli, "_save_selected_result") as save:
                code, output = self.run_cli(["SEND", "결측률을 확인해 주세요.", "SEND"])

        self.assertEqual(code, 0)
        self.assertEqual(model.call_count, 2)
        self.assertIn("proposal_validated", output)
        save.assert_not_called()

    def test_cancel_after_question_never_sends_answer(self):
        with patch.object(langchain_local, "request_local_proposal", return_value=clarification()) as model:
            code, _output = self.run_cli(["SEND", "결측률 확인", "취소"])

        self.assertEqual(code, 0)
        model.assert_called_once()


if __name__ == "__main__":
    unittest.main()
