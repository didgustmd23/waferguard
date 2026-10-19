"""실제 모델·네트워크 없이 LangChain 연결과 기존 보호 규칙 재사용을 검사한다."""

from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
import json
import os
import unittest
from unittest.mock import Mock, patch

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.runnables.config import var_child_runnable_config
from langsmith import get_tracing_context, tracing_context

from src.agents import langchain_local as adapter
from src.agents import llm_proposal, llm_proposal_cli
from src.agents.llm_config import load_local_config
from src.agents.local_ollama import LocalProposalError
from tests.agents.test_llm_proposal import valid_content


class LangChainLocalTest(unittest.TestCase):
    def setUp(self):
        self.config = load_local_config()
        self.payload = llm_proposal.build_example_payload("missing", self.config)
        patcher = patch.object(adapter, "request_local_proposal", return_value=valid_content())
        self.transport = patcher.start()
        self.addCleanup(patcher.stop)

    def call(self):
        return adapter.request_langchain_proposal(self.payload, self.config)

    def test_langchain_invokes_existing_transport_once_with_unchanged_payload(self):
        self.assertEqual(self.call(), valid_content())
        self.transport.assert_called_once_with(self.payload, self.config)

    def test_settings_are_taken_from_config(self):
        self.config = replace(self.config, model="synthetic:local", port=11435, think=True,
                              temperature=0.2, num_ctx=4096, num_predict=512)
        self.payload = llm_proposal.build_example_payload("missing", self.config)
        self.call()
        self.transport.assert_called_once_with(self.payload, self.config)

    def test_environment_tracing_and_parent_callbacks_are_not_inherited(self):
        callback = Mock(spec=BaseCallbackHandler)
        token = var_child_runnable_config.set({"callbacks": [callback]})
        seen = []

        def answer(*args):
            seen.append(get_tracing_context()["enabled"])
            return valid_content()

        self.transport.side_effect = answer

        try:
            with patch.dict(os.environ, {"LANGSMITH_TRACING": "true", "HTTP_PROXY": "http://untrusted.invalid"}):
                with tracing_context(enabled=True), patch("langsmith.Client.create_run") as trace:
                    self.call()
                    trace.assert_not_called()
        finally:
            var_child_runnable_config.reset(token)

        self.assertEqual(seen, [False])
        self.assertEqual(callback.mock_calls, [])

    def test_global_debug_or_verbose_blocks_before_transport(self):
        for name in ("get_debug", "get_verbose"):
            with self.subTest(name=name), patch.object(adapter, name, return_value=True):
                with self.assertRaises(LocalProposalError) as caught:
                    self.call()

                self.assertEqual(caught.exception.code, "LANGCHAIN_DIAGNOSTICS_ENABLED")

        self.transport.assert_not_called()

    def test_additional_callback_is_rejected(self):
        manager = Mock(handlers=[Mock()], inheritable_handlers=[])

        with patch.object(adapter.CallbackManager, "configure", return_value=manager):
            with self.assertRaises(LocalProposalError) as caught:
                self.call()

        self.assertEqual(caught.exception.code, "LANGCHAIN_CALLBACKS_REJECTED")
        self.transport.assert_not_called()

    def test_request_size_model_and_extra_tools_are_rejected(self):
        for mutation in (
            {"model": "different"}, {"tools": [{"name": "execute"}]},
            {"stream": True}, {"options": {}},
            {"messages": [{"role": "system", "content": "x" * 70000},
                          {"role": "user", "content": "synthetic"}]},
        ):
            self.payload = llm_proposal.build_example_payload("missing", self.config)
            self.payload.update(mutation)

            with self.subTest(mutation=list(mutation)), self.assertRaises(LocalProposalError):
                self.call()

        self.transport.assert_not_called()

    def test_images_tools_and_extra_messages_are_rejected(self):
        for messages in (
            [{"role": "user", "content": []}],
            [{"role": "system", "content": "s"}, {"role": "tool", "content": "t"}],
            [{"role": "system", "content": "s"}, {"role": "user", "content": "t", "images": []}],
        ):
            self.payload["messages"] = messages

            with self.subTest(messages=messages), self.assertRaises(LocalProposalError):
                self.call()

        self.transport.assert_not_called()

    def test_error_is_sanitized_and_not_retried(self):
        self.transport.side_effect = RuntimeError("synthetic_private_detail")

        with self.assertRaises(LocalProposalError) as caught:
            self.call()

        self.assertEqual(str(caught.exception), "LANGCHAIN_CALL_FAILED")
        self.transport.assert_called_once()

    def test_transport_failure_code_is_preserved(self):
        self.transport.side_effect = LocalProposalError("OLLAMA_CONNECTION_FAILED")

        with self.assertRaises(LocalProposalError) as caught:
            self.call()

        self.assertEqual(caught.exception.code, "OLLAMA_CONNECTION_FAILED")
        self.transport.assert_called_once()

    def test_proposal_and_clarification_use_existing_contract(self):
        contents = [valid_content(), json.dumps({
            "schema_version": "llm_proposal_v1", "status": "needs_clarification",
            "pipeline": None, "questions": ["원하시는 검사를 알려주세요."],
        })]

        for content in contents:
            self.transport.return_value = content
            result = llm_proposal.propose_example("missing", backend="langchain").to_dict()
            self.assertTrue(result["proposal_only"])
            self.assertFalse(result["execution_approved"])
            self.assertFalse(result["worker_started"])

    def test_bad_contract_and_unknown_backend_never_fall_back(self):
        self.transport.return_value = '{"approved":true}'

        with patch.object(llm_proposal, "request_local_proposal") as fallback:
            with self.assertRaises(LocalProposalError) as caught:
                llm_proposal.propose_example("missing", backend="langchain")

            self.assertEqual(caught.exception.code, "PROPOSAL_CONTRACT_REJECTED")

            with self.assertRaises(LocalProposalError):
                llm_proposal.propose_example("missing", backend="unknown")

            fallback.assert_not_called()

    def test_cli_preview_does_not_invoke_transport(self):
        arguments = ["proposal", "--example", "missing", "--backend", "langchain", "--preview"]

        with patch("sys.argv", arguments), redirect_stdout(StringIO()) as output:
            self.assertEqual(llm_proposal_cli.main(), 0)

        result = json.loads(output.getvalue())
        self.assertEqual(result["backend"], "langchain")
        self.assertFalse(result["model_called"])
        self.transport.assert_not_called()

    def test_cli_confirmation_calls_langchain(self):
        arguments = ["proposal", "--example", "missing", "--backend", "langchain",
                     "--confirm-local-model-call"]

        with patch("sys.argv", arguments), redirect_stdout(StringIO()) as output:
            self.assertEqual(llm_proposal_cli.main(), 0)

        result = json.loads(output.getvalue())
        self.assertEqual(result["status"], "proposal_validated")
        self.assertEqual(result["backend"], "langchain")
        self.transport.assert_called_once()


if __name__ == "__main__":
    unittest.main()
