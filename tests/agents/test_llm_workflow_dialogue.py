"""실제 LangGraph를 사용하되 모델 운송은 합성 응답으로 대체한다."""

from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
import os
import unittest
from unittest.mock import patch

from langsmith import tracing_context

from src.agents import langchain_local, llm_workflow_cli, llm_workflow_proposal
from src.agents.llm_dialogue import MAX_MODEL_CALLS
from src.agents.llm_workflow_contract import SCHEMA_VERSION
from src.agents.llm_workflow_dialogue import WorkflowProposalDialogue
from src.agents.local_ollama import LocalProposalError
from tests.agents.test_llm_workflow_contract import EXAMPLE


def clarification():
    return json.dumps({"schema_version": SCHEMA_VERSION, "status": "needs_clarification",
                       "workflow": None, "questions": ["분할부터 적용까지 제안할까요?"]}, ensure_ascii=False)


class WorkflowDialogueTest(unittest.TestCase):
    def setUp(self):
        self.example = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        self.context = {
            "allowed_modules": tuple((step["module_id"], step["version"]) for step in self.example["workflow"]["steps"]),
            "selected_settings": {step["module_id"]: deepcopy(step["settings"])
                                  for step in self.example["workflow"]["steps"]},
        }
        patcher = patch.object(langchain_local, "request_local_proposal", return_value=json.dumps(self.example))
        self.model = patcher.start()
        self.addCleanup(patcher.stop)
        self.session = WorkflowProposalDialogue(**self.context)
        self.addCleanup(self.session.close)

    def send(self, text="선택한 규칙으로 분할부터 적용까지 제안해 주세요."):
        return self.session.send(text, confirmed=True)

    def test_preview_requires_confirmation_and_has_only_public_selection(self):
        preview = self.session.preview("선택 규칙으로 제안하세요.")
        self.assertFalse(preview["model_called"])
        self.assertEqual(preview["payload"]["format"]["properties"]["schema_version"]["const"], SCHEMA_VERSION)
        self.assertNotIn("snapshot_dir", preview["payload"]["messages"][0]["content"])

        with self.assertRaises(LocalProposalError):
            self.session.send("제안하세요.")

        self.model.assert_not_called()

    def test_user_five_stage_request_marks_rules_fixed_and_defers_input_checks(self):
        request = (
            "1. 데이터 품질 검사와 EDA 후 분할 검토·확정\n"
            "2. Train·Validation 데이터 내보내기\n"
            "3. Train에서만 전처리 기준 후보 산출\n"
            "4. 후보 검토 후 저장된 기준을 Train·Validation에 적용\n"
            "5. 로컬 전달 묶음 생성"
        )
        context = deepcopy(self.context)
        context["allowed_modules"] += (("csv_local_handoff", "1.0.0"),)
        context["selected_settings"]["csv_local_handoff"] = {}
        session = WorkflowProposalDialogue(**context)
        self.addCleanup(session.close)
        preview = session.preview(request)
        system = preview["payload"]["messages"][0]["content"]
        instructions, context_json = system.split("\n", 1)
        context = json.loads(context_json)
        self.assertEqual(context["settings_status"], "fixed_user_selection")
        self.assertEqual(context["input_validation_owner"], "local_validator_after_proposal")
        self.assertIn("csv_local_handoff", context["selected_settings"])
        self.assertEqual(context["selected_settings"]["csv_prepare_candidate"],
                         self.context["selected_settings"]["csv_prepare_candidate"])
        self.assertIn("proposed와 questions:[]", instructions)
        self.assertIn("none, false, null도 명시적으로 선택한 값", instructions)
        self.assertIn("시간 컬럼이 있는지 재질문하거나 random 변경을 권하지 마세요", instructions)
        self.assertIn("time에서 stratify=true나 require_all_classes=false를 권하지 마세요", instructions)
        self.assertNotIn("설정 변경이 필요하면 질문으로 알리세요", instructions)
        self.model.assert_not_called()

    def test_prompt_limits_questions_without_suppressing_genuine_clarification(self):
        system = self.session.preview("어떤 작업을 할지는 아직 정하지 않았습니다.")["payload"]["messages"][0]["content"]
        for decision in ("작업 종류가 미정", "요청 작업의 전체 규칙이 없", "명시적 변경 요청", "미지원 작업"):
            self.assertIn(decision, system)

        self.assertIn("binning은 수치형 구간화", system)
        self.assertIn("iqr_clip은 클리핑이지 행 삭제가 아닙니다", system)
        self.model.return_value = clarification()
        self.assertEqual(self.send("어떤 작업을 할지는 아직 정하지 않았습니다.")["status"], "awaiting_answer")
        self.model.assert_called_once()

    def test_proposal_contains_no_execution_authority_and_cannot_be_resent(self):
        result = self.send()
        self.assertEqual(result["status"], "proposal_validated")
        self.assertTrue(result["requires_local_binding"])
        self.assertFalse(result["worker_started"])
        self.assertFalse(result["execution_approved"])
        self.assertEqual(result["model_calls"], 1)

        with self.assertRaises(LocalProposalError):
            self.send()

        self.model.assert_called_once()

    def test_interrupt_resume_matches_preview_without_repeating_call(self):
        self.model.side_effect = [clarification(), json.dumps(self.example)]
        self.assertEqual(self.send("어떤 작업인지 확인해 주세요.")["status"], "awaiting_answer")
        preview = self.session.preview("네. 선택 규칙 그대로 제안하세요.")
        self.assertEqual(self.model.call_count, 1)
        self.assertEqual(self.send("네. 선택 규칙 그대로 제안하세요.")["model_calls"], 2)
        self.assertEqual(preview["payload"], self.model.call_args.args[0])
        self.assertIn("분할부터 적용까지 제안할까요?", preview["payload"]["messages"][1]["content"])

    def test_selected_settings_are_frozen_despite_caller_mutation(self):
        self.context["selected_settings"]["csv_prepare_candidate"]["numeric_imputation"] = "mean"
        self.assertEqual(self.send()["status"], "proposal_validated")

    def test_missing_selection_is_not_filled_from_model_defaults(self):
        session = WorkflowProposalDialogue(**{**self.context, "selected_settings": {}})
        self.addCleanup(session.close)

        with self.assertRaises(LocalProposalError) as caught:
            session.send("기본값으로 진행하세요.", confirmed=True)

        self.assertEqual(caught.exception.code, "WORKFLOW_PROPOSAL_CONTRACT_REJECTED")
        self.assertEqual(session._phase, "closed")

    def test_missing_selection_can_ask_questions_and_has_null_only_schema(self):
        payload = llm_workflow_proposal.build_request_payload("규칙을 먼저 확인하세요.",
                                                             **{**self.context, "selected_settings": {}})
        self.assertEqual(payload["format"]["properties"]["workflow"], {"type": "null"})
        self.model.return_value = clarification()
        session = WorkflowProposalDialogue(**{**self.context, "selected_settings": {}})
        self.addCleanup(session.close)
        self.assertEqual(session.send("규칙을 확인하세요.", confirmed=True)["status"], "awaiting_answer")

    def test_changed_rules_skipped_dependency_and_forged_approval_rejected(self):
        for alteration in ("rule", "dependency", "approval", "test", "legacy"):
            with self.subTest(alteration=alteration):
                value = deepcopy(self.example)

                if alteration == "rule":
                    value["workflow"]["steps"][2]["settings"]["scaling"] = "standard"
                elif alteration == "dependency":
                    value["workflow"]["steps"][1]["depends_on"] = []
                elif alteration == "approval":
                    value["execution_approved"] = True
                elif alteration == "test":
                    value["workflow"]["steps"][3]["settings"]["test_used"] = True
                else:
                    value["schema_version"] = "llm_proposal_v1"

                self.model.return_value = json.dumps(value)
                session = WorkflowProposalDialogue(**self.context)
                self.addCleanup(session.close)

                with self.assertRaises(LocalProposalError):
                    session.send("제안하세요.", confirmed=True)

    def test_questions_bounded_and_failure_never_retries(self):
        self.model.return_value = clarification()

        for _ in range(MAX_MODEL_CALLS):
            result = self.send("추가 확인해 주세요.")

        self.assertEqual(result["status"], "clarification_limit_reached")
        self.assertEqual(self.model.call_count, MAX_MODEL_CALLS)

        with self.assertRaises(LocalProposalError):
            self.send()

    def test_transport_failure_closes_without_retry(self):
        self.model.side_effect = LocalProposalError("OLLAMA_CONNECTION_FAILED")

        for _ in range(2):
            with self.assertRaises(LocalProposalError):
                self.send()

        self.model.assert_called_once()

    def test_accumulated_text_limits_apply_before_second_call(self):
        self.model.return_value = clarification()
        self.send("가" * 1500)

        with self.assertRaises(ValueError):
            self.send("나" * 1500)

        self.model.assert_called_once()

    def test_tracing_disabled_even_when_inherited(self):
        with patch.dict(os.environ, {"LANGSMITH_TRACING": "true"}), tracing_context(enabled=True):
            with patch("langsmith.Client.create_run") as trace:
                self.send()
                trace.assert_not_called()

    def test_http_and_langchain_share_contract_and_one_shot_transport(self):
        with patch.object(llm_workflow_proposal, "request_local_proposal", return_value=json.dumps(self.example)) as model:
            result = llm_workflow_proposal.propose_request("선택 규칙으로 제안하세요.", **self.context)
            self.assertTrue(result.to_dict()["proposal_only"])
            model.assert_called_once()

    def test_invalid_context_and_text_do_not_call_transport(self):
        for text in ("", "x" * 4001, "bad\x1b"):
            with self.assertRaises(ValueError):
                self.session.preview(text)

        with self.assertRaises(ValueError):
            WorkflowProposalDialogue(**{**self.context, "allowed_modules": (("shell", "1.0.0"),)})

        self.model.assert_not_called()

    def test_graph_revalidates_even_if_proposal_helper_is_bypassed(self):
        forged = deepcopy(self.example)
        forged["execution_approved"] = True

        with patch.object(self.session, "_propose", return_value={"proposal": forged}):
            with self.assertRaises(LocalProposalError):
                self.send()

        self.model.assert_not_called()
        self.assertEqual(self.session._phase, "closed")

    def test_independent_sessions_do_not_share_questions(self):
        self.model.return_value = clarification()
        self.send("첫 세션 전용 문장입니다.")
        other = WorkflowProposalDialogue(**self.context)
        self.addCleanup(other.close)
        self.model.return_value = json.dumps(self.example)
        other.send("두 번째 세션 전용 문장입니다.", confirmed=True)
        self.assertNotIn("첫 세션", self.model.call_args.args[0]["messages"][1]["content"])

    def test_cli_each_call_requires_send_and_does_not_create_jobs(self):
        with patch("sys.argv", ["workflow", "--example", "prepare", "--use-example-settings"]):
            with patch("builtins.input", return_value="취소"), redirect_stdout(StringIO()) as output:
                with patch("sys.stdin.isatty", return_value=True), patch("sys.stdout.isatty", return_value=True):
                    self.assertEqual(llm_workflow_cli.main(), 0)

        self.assertIn("dialogue_cancelled", output.getvalue())
        self.model.assert_not_called()

    def test_cli_question_answer_calls_only_after_each_confirmation(self):
        self.model.side_effect = [clarification(), json.dumps(self.example)]

        with patch("sys.argv", ["workflow", "--example", "clarify", "--use-example-settings"]):
            with patch("builtins.input", side_effect=["SEND", "예제 규칙 그대로 적용까지 제안하세요.", "SEND"]):
                with redirect_stdout(StringIO()) as output:
                    with patch("sys.stdin.isatty", return_value=True), patch("sys.stdout.isatty", return_value=True):
                        self.assertEqual(llm_workflow_cli.main(), 0)

        self.assertEqual(self.model.call_count, 2)
        self.assertIn("proposal_validated", output.getvalue())
        self.assertIn('"worker_started": false', output.getvalue())


if __name__ == "__main__":
    unittest.main()
