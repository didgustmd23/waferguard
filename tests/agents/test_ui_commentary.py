"""결과 해설의 정책·자료 최소화·단발 호출 경계. 실제 모델·worker를 실행하지 않는다."""

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from src.ui.commentary_policy import (default_policy, disclosure_decision, load_policy,
                                     model_binding, save_policy, validate_policy, fingerprint)
from src.ui.commentary_service import build_packet, failure_code, request_commentary, send_once
from src.ui.commentary_view import after_completion, freeze_classification, scope
from src.ui.model_commands import ModelCommandLedger
from src.ui.state import default_settings
from src.ui.commentary_transport import CommentaryHTTPError, request_images
from src.agents.llm_config import LocalModelConfig


class CommentaryPolicyTest(unittest.TestCase):
    def setUp(self):
        self.model = default_settings()["model"]
        self.policy = default_policy() | {"local_default_allow": True,
                                          "model_binding": model_binding(self.model)}

    def test_default_and_unknown_block_automatic_transmission(self):
        self.assertFalse(default_policy()["local_default_allow"])
        self.assertEqual(disclosure_decision(self.policy, "local", "unknown", self.model),
                         "CLASSIFICATION_REQUIRED")
        self.assertEqual(disclosure_decision(self.policy, "local", "general", self.model), "AUTO_ALLOWED")

    def test_sensitive_exclusion_and_individual_confirmation(self):
        self.assertEqual(disclosure_decision(self.policy, "local", "sensitive", self.model), "SENSITIVE_EXCLUDED")
        self.policy["sensitive_action"] = "confirm"
        self.assertEqual(disclosure_decision(self.policy, "local", "sensitive", self.model),
                         "INDIVIDUAL_CONFIRMATION_REQUIRED")

    def test_external_never_inherits_local_permission(self):
        self.assertEqual(disclosure_decision(self.policy, "external", "general", self.model),
                         "INDIVIDUAL_CONFIRMATION_REQUIRED")
        self.assertEqual(disclosure_decision(self.policy, "remote", "general", self.model), "DESTINATION_REJECTED")

    def test_model_changes_revoke_default_permission(self):
        changed = self.model | {"model": "different:local"}
        self.assertEqual(disclosure_decision(self.policy, "local", "general", changed),
                         "MODEL_CONFIRMATION_REQUIRED")

    def test_strict_policy_and_persistence(self):
        for change in ({"version": True}, {"graphs": 1}, {"api_default": True}):
            with self.assertRaises(ValueError):
                validate_policy(self.policy | change)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            self.assertEqual(load_policy(path), default_policy())
            save_policy(self.policy, path)
            self.assertEqual(load_policy(path), self.policy)
            path.write_text('{"broken":true}', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_policy(path)


class CommentaryServiceTest(unittest.TestCase):
    def setUp(self):
        self.model = default_settings()["model"]
        self.policy = default_policy() | {"local_default_allow": True,
                                          "model_binding": model_binding(self.model)}
        self.result = {"status": "completed", "files": [
            {"name": "eda/eda.md", "data": b"secret-path\n| class | count |\n| --- | --- |\n| A | 3 |\n"},
            {"name": "train.csv", "data": b"raw-secret"},
            {"name": "preprocessor.json", "data": b"private"},
            {"name": "examples.png", "data": b"private image"}]}

    def test_only_tables_not_prose_raw_rows_or_examples(self):
        packet = build_packet(self.result, self.policy)
        self.assertEqual(len(packet["assets"]), 1)
        self.assertNotIn("secret", str(packet))
        self.assertNotIn("private", str(packet))

    def test_incomplete_or_oversized_not_silently_truncated(self):
        with self.assertRaises(ValueError):
            build_packet(self.result | {"status": "review_required"}, self.policy)
        large = {"status": "completed", "files": [{"name": "missing.csv", "data": b"x\n" * 202}]}
        with self.assertRaises(ValueError):
            build_packet(large, self.policy)

    def test_graphs_require_explicit_permission_and_valid_png(self):
        graph = {"status": "completed", "files": [{"name": "distributions.png", "data": b"not-png"}]}
        with self.assertRaises(ValueError):
            build_packet(graph, self.policy)
        with self.assertRaisesRegex(ValueError, "GRAPH_REJECTED"):
            build_packet(graph, self.policy | {"graphs": True})

    def test_ledger_prevents_repeat_calls_across_instances(self):
        packet = build_packet(self.result, self.policy)
        requester = MagicMock(return_value={"observations": ["관찰"], "limitations": ["한계"]})
        with TemporaryDirectory() as directory:
            path = Path(directory) / "commands.sqlite3"
            first = send_once(packet, self.policy, self.model, "general", "test",
                              ledger=ModelCommandLedger(path), requester=requester)
            second = send_once(packet, self.policy, self.model, "general", "test",
                               ledger=ModelCommandLedger(path), requester=requester)
            self.assertEqual(first["status"], "completed")
            self.assertEqual(second["status"], "already_attempted")
            requester.assert_called_once()

    def test_concurrent_claims_only_one_model_call(self):
        packet = build_packet(self.result, self.policy)
        requester = MagicMock(return_value={"observations": ["관찰"], "limitations": ["한계"]})
        with TemporaryDirectory() as directory:
            path = Path(directory) / "commands.sqlite3"
            def run(_):
                return send_once(packet, self.policy, self.model, "general", "parallel",
                                 ledger=ModelCommandLedger(path), requester=requester)

            with ThreadPoolExecutor(max_workers=2) as executor:
                replies = list(executor.map(run, range(2)))
            self.assertEqual(sorted(reply["status"] for reply in replies), ["already_attempted", "completed"])
            requester.assert_called_once()

    def test_sensitive_consent_is_bound_to_exact_payload(self):
        packet = build_packet(self.result, self.policy)
        policy = self.policy | {"sensitive_action": "confirm"}
        consent = fingerprint({"packet": packet, "policy": policy,
                               "model": self.model, "sensitivity": "sensitive"})
        ledger = MagicMock()
        with self.assertRaises(ValueError):
            send_once(packet | {"purpose": "changed"}, policy, self.model, "sensitive", "test",
                      consent=consent, ledger=ledger)
        ledger.claim.assert_not_called()
        ledger.claim.return_value = True
        self.assertEqual(send_once(packet, policy, self.model, "sensitive", "test", consent=consent,
                                   ledger=ledger, requester=lambda *args: {})["status"], "completed")

    def test_model_failure_has_no_retry_or_preprocessing_mutation(self):
        packet = build_packet(self.result, self.policy)
        before = deepcopy(self.result)
        ledger = MagicMock()
        ledger.claim.return_value = True
        reply = send_once(packet, self.policy, self.model, "general", "test", ledger=ledger,
                          requester=MagicMock(side_effect=TimeoutError))
        self.assertEqual(reply["status"], "failed_no_retry")
        self.assertEqual(self.result, before)
        self.assertEqual(ledger.finish.call_args.args[-1], "unknown")

    def test_response_contract_rejects_extra_actions(self):
        packet = build_packet(self.result, self.policy)
        with patch("src.ui.commentary_service.request_local_proposal", return_value='{"execute":"bad"}'):
            with self.assertRaises(ValueError):
                request_commentary(packet, self.model)

    def test_failure_codes_never_expose_exception_text(self):
        self.assertEqual(failure_code(ValueError("secret raw response")), "COMMENTARY_RESPONSE_INVALID")
        self.assertEqual(failure_code(RuntimeError("secret path")), "COMMENTARY_REQUEST_FAILED")
        self.assertEqual(failure_code(TimeoutError("private")), "COMMENTARY_TIMEOUT")

    def test_attempt_records_actual_pair_payload_version_not_render_time(self):
        packet = {"purpose": "result_commentary", "assets": [{"name": "correlations.png", "image": "synthetic",
                  "graph_context": {"pairs": [{"feature_a": "a", "feature_b": "b", "pearson": -0.2,
                                                "observed_count": 8}]}}]}
        ledger = MagicMock()
        ledger.claim.return_value = True
        requester = MagicMock(return_value={"observations": ["관찰"], "limitations": []})
        reply = send_once(packet, self.policy | {"graphs": True}, self.model, "general", "version-test", ledger=ledger, requester=requester)
        self.assertEqual(reply["commentary_version"], "pairs-v2")
        self.assertEqual(reply["input_format"], "sensor_pairs")
        self.assertEqual(reply["correlation_pair_count"], 1)
        requester.assert_called_once_with(packet, self.model)

    def test_http_failure_keeps_only_status_and_fixed_reason(self):
        packet = build_packet(self.result, self.policy)
        ledger = MagicMock()
        ledger.claim.return_value = True
        reply = send_once(packet, self.policy, self.model, "general", "http-test", ledger=ledger,
                          requester=MagicMock(side_effect=CommentaryHTTPError(400, "CONTEXT_LIMIT")))
        self.assertEqual(reply["status"], "failed_no_retry")
        self.assertEqual(reply["code"], "COMMENTARY_HTTP_REJECTED")
        self.assertEqual(reply["http_status"], 400)
        self.assertEqual(reply["reason"], "CONTEXT_LIMIT")
        self.assertEqual(reply["request_context"], self.model["num_ctx"])
        self.assertTrue(reply["attempted_at"].endswith("+00:00"))
        ledger.finish.assert_called_once_with("commentary:http-test", unittest.mock.ANY, "unknown")

    def test_completion_does_not_read_results_without_policy_and_classification(self):
        controller = SimpleNamespace(name="synthetic", module_id="tabular")
        st = MagicMock()
        st.session_state = {"ui_settings": {"model": self.model}}
        with patch("src.ui.commentary_view.load_policy", return_value=self.policy), \
                patch("src.ui.workflow_results.load_workflow_results") as loader:
            after_completion(st, controller, workflow=True)
            loader.assert_not_called()
            st.session_state["commentary_class_" + scope(controller)] = "일반 자료"
            freeze_classification(st, controller)
            st.session_state.pop("commentary_class_" + scope(controller))
            loader.return_value = self.result
            with patch("src.ui.commentary_view.send_once", return_value={"status": "completed"}) as send:
                after_completion(st, controller, workflow=True)
                send.assert_called_once()

    def test_policy_revoked_during_result_validation_prevents_send(self):
        controller = SimpleNamespace(name="synthetic", module_id="tabular")
        st = MagicMock()
        st.session_state = {"ui_settings": {"model": self.model},
                            "commentary_frozen_" + scope(controller): "general"}
        with patch("src.ui.commentary_view.load_policy", side_effect=[self.policy, default_policy()]), \
                patch("src.ui.workflow_results.load_workflow_results", return_value=self.result), \
                patch("src.ui.commentary_view.send_once") as send:
            after_completion(st, controller, workflow=True)
            send.assert_not_called()

    def test_commentary_allows_no_limitations_and_instructs_non_repetition(self):
        packet = build_packet(self.result, self.policy)
        raw = json.dumps({"observations": ["A는 3건입니다."], "limitations": []}, ensure_ascii=False)
        with patch("src.ui.commentary_service.request_local_proposal", return_value=raw) as request:
            reply = request_commentary(packet, self.model)
            self.assertEqual(reply["limitations"], [])
            instructions = request.call_args.args[0]["messages"][0]["content"]
            self.assertIn("반복하지", instructions)
            self.assertIn("반올림", instructions)
            self.assertIn("그림이 없으면", instructions)

    def test_commentary_removes_only_literal_whitespace_duplicates(self):
        packet = build_packet(self.result, self.policy)
        raw = json.dumps({"observations": ["A는 3건입니다.", " A는  3건입니다. "], "limitations": []})
        with patch("src.ui.commentary_service.request_local_proposal", return_value=raw):
            self.assertEqual(request_commentary(packet, self.model)["observations"], ["A는 3건입니다."])

    def test_commentary_rejects_excess_items_or_blank_observation(self):
        packet = build_packet(self.result, self.policy)
        for reply in ({"observations": ["x"] * 4, "limitations": []},
                      {"observations": ["x"], "limitations": ["x"] * 3},
                      {"observations": ["  "], "limitations": []}):
            with patch("src.ui.commentary_service.request_local_proposal", return_value=json.dumps(reply)):
                with self.assertRaises(ValueError):
                    request_commentary(packet, self.model)

    def test_simple_split_guidance_does_not_require_three_observations(self):
        packet = {"purpose": "result_commentary", "assets": [{"name": "split_summary.md", "tables": [
            "| split | rows | ratio |\n| --- | --- | --- |\n| train | 7 | 70% |\n"
            "| validation | 2 | 20% |\n| test | 1 | 10% |"]}]}
        raw = json.dumps({"observations": ["총 10행을 Train 7행, Validation 2행, Test 1행으로 나눴습니다."],
                          "limitations": []})
        with patch("src.ui.commentary_service.request_local_proposal", return_value=raw) as request:
            reply = request_commentary(packet, self.model)
            self.assertEqual(len(reply["observations"]), 1)
            instructions = request.call_args.args[0]["messages"][0]["content"]
            for rule in ("단순 분할 표는 관찰 한 문장", "재표현을 추가 관찰", "목표 비율이 자료에 없으면"):
                self.assertIn(rule, instructions)


class CommentaryTransportTest(unittest.TestCase):
    def setUp(self):
        self.config = LocalModelConfig(**default_settings()["model"])
        self.payload = {"model": self.config.model, "messages": [{"role": "user", "images": ["image"]}]}
        self.response = MagicMock()
        self.response.status = 200
        self.response.getheader.return_value = None
        self.response.read.return_value = json.dumps({"done": True, "done_reason": "stop",
            "model": self.config.model, "message": {"role": "assistant", "content": "reply"}}).encode()
        self.connection = MagicMock()
        self.connection.getresponse.return_value = self.response

    def test_loopback_only_and_no_redirect(self):
        with patch("src.ui.commentary_transport.http.client.HTTPConnection", return_value=self.connection) as http:
            self.assertEqual(request_images(self.payload, self.config), "reply")
            self.assertEqual(http.call_args.args, (self.config.host, self.config.port))
            self.connection.close.assert_called_once()
            self.response.status = 302
            with self.assertRaises(ValueError):
                request_images(self.payload, self.config)

    def test_oversized_or_tool_reply_rejected_and_connection_closed(self):
        with patch("src.ui.commentary_transport.http.client.HTTPConnection", return_value=self.connection):
            self.response.read.return_value = b"x" * 65537
            with self.assertRaises(ValueError):
                request_images(self.payload, self.config)
            self.response.read.return_value = json.dumps({"done": True, "done_reason": "stop",
                "model": self.config.model, "message": {"role": "assistant", "content": "reply",
                                                        "tool_calls": ["forbidden"]}}).encode()
            with self.assertRaises(ValueError):
                request_images(self.payload, self.config)
            self.assertEqual(self.connection.close.call_count, 2)

    def test_http_diagnostics_are_bounded_and_do_not_expose_server_body(self):
        cases = ((400, "input length exceeds context window: secret", "CONTEXT_LIMIT"),
                 (400, "request (8302 tokens) exceeds the available context size (8192 tokens), try increasing it", "CONTEXT_LIMIT"),
                 (500, "CUDA error: out of memory private-data", "MEMORY_EXHAUSTED"),
                 (400, "model does not support images private-data", "IMAGES_UNSUPPORTED"),
                 (404, "private model path", "MODEL_OR_ENDPOINT_NOT_FOUND"),
                 (400, "private raw error", "INVALID_REQUEST"),
                 (500, "private raw error", "SERVER_ERROR"))
        for status, message, expected in cases:
            with self.subTest(status=status, reason=expected), patch(
                    "src.ui.commentary_transport.http.client.HTTPConnection", return_value=self.connection):
                self.response.status = status
                self.response.read.return_value = json.dumps({"error": message}).encode()
                with self.assertRaises(CommentaryHTTPError) as caught:
                    request_images(self.payload, self.config)
                self.assertEqual(caught.exception.reason, expected)
                self.assertEqual(caught.exception.http_status, status)
                self.assertNotIn("private", str(caught.exception))
                self.assertEqual(self.response.read.call_args.args, (8193,))

    def test_oversized_error_does_not_get_parsed_or_echoed(self):
        with patch("src.ui.commentary_transport.http.client.HTTPConnection", return_value=self.connection):
            self.response.status = 500
            self.response.read.return_value = b"private" * 2000
            with self.assertRaises(CommentaryHTTPError) as caught:
                request_images(self.payload, self.config)
            self.assertEqual(caught.exception.reason, "SERVER_ERROR")
            self.assertNotIn("private", str(caught.exception))

    def test_context_error_keeps_only_bounded_token_counts(self):
        with patch("src.ui.commentary_transport.http.client.HTTPConnection", return_value=self.connection):
            self.response.status = 400
            self.response.read.return_value = json.dumps({"error":
                "request (8302 tokens) exceeds the available context size (8192 tokens), try increasing it private"}).encode()
            with self.assertRaises(CommentaryHTTPError) as caught:
                request_images(self.payload, self.config)
            self.assertEqual(caught.exception.requested_tokens, 8302)
            self.assertEqual(caught.exception.context_tokens, 8192)
            self.assertNotIn("private", str(caught.exception))
            self.response.read.return_value = json.dumps({"error":
                "request (999999999999 tokens) exceeds the available context size (8192 tokens)"}).encode()
            with self.assertRaises(CommentaryHTTPError) as oversized:
                request_images(self.payload, self.config)
            self.assertIsNone(oversized.exception.requested_tokens)

    def test_context_setting_survives_policy_save_and_rerender(self):
        from streamlit.testing.v1 import AppTest
        script = """
import streamlit as st
from src.ui.state import default_settings
from src.ui.settings import render_settings
if 'ui_settings' not in st.session_state:
    st.session_state['ui_settings'] = default_settings()
    st.session_state['ui_settings_revision'] = 0
render_settings(st)
"""
        with patch("src.ui.commentary_view.load_policy", return_value=default_policy()), \
                patch("src.ui.commentary_view.save_policy") as save:
            app = AppTest.from_string(script).run()
            next(widget for widget in app.number_input if widget.label == "문맥 길이").set_value(16384)
            next(button for button in app.button if button.label == "이번 세션에 적용").click().run()
            self.assertEqual(app.session_state["ui_settings"]["model"]["num_ctx"], 16384)
            next(button for button in app.button if button.label == "해설 전달 정책 저장").click().run()
            self.assertEqual(save.call_args.args[0]["model_binding"], model_binding(app.session_state["ui_settings"]["model"]))
            app.run()
            self.assertEqual(app.session_state["ui_settings"]["model"]["num_ctx"], 16384)
            self.assertFalse(app.exception)

    def test_sensitive_result_preview_render_never_calls_model(self):
        from streamlit.testing.v1 import AppTest
        script = """
import streamlit as st
from types import SimpleNamespace
from src.ui.commentary_view import render_commentary
render_commentary(st, SimpleNamespace(name='synthetic', module_id='tabular'), st.session_state['result'])
"""
        model = default_settings()["model"]
        policy = default_policy() | {"sensitive_action": "confirm", "model_binding": model_binding(model)}
        with patch("src.ui.commentary_view.load_policy", return_value=policy), \
                patch("src.ui.commentary_view.send_once") as send:
            app = AppTest.from_string(script)
            app.session_state["ui_settings"] = {"model": model}
            app.session_state["result"] = {"status": "completed", "files": [
                {"name": "missing.csv", "data": b"feature,missing_ratio\nx,0.1\n"}]}
            app.run()
            app.selectbox[0].select("민감 자료").run()
            app.run()
            self.assertFalse(app.exception)
            send.assert_not_called()


    def test_failed_request_requires_explicit_new_request_and_fresh_confirmation(self):
        from streamlit.testing.v1 import AppTest
        script = """
import streamlit as st
from types import SimpleNamespace
from src.ui.commentary_view import render_commentary
render_commentary(st, SimpleNamespace(name='synthetic', module_id='tabular'), st.session_state['result'])
"""
        model = default_settings()["model"]
        policy = default_policy() | {"model_binding": model_binding(model)}
        controller = SimpleNamespace(name="synthetic", module_id="tabular")
        result_key = "commentary_result_" + scope(controller)
        with patch("src.ui.commentary_view.load_policy", return_value=policy), \
                patch("src.ui.commentary_view.send_once", return_value={"status": "completed"}) as send:
            app = AppTest.from_string(script)
            app.session_state["ui_settings"] = {"model": model}
            app.session_state["result"] = {"status": "completed", "files": [
                {"name": "missing.csv", "data": b"feature,missing_ratio\nx,0.1\n"}]}
            app.session_state[result_key] = {"status": "failed_no_retry", "code": "COMMENTARY_TIMEOUT"}
            app.run()
            app.selectbox[0].select("일반 자료").run()
            self.assertTrue(next(item for item in app.button if item.label == "확인한 자료로 해설 요청").disabled)
            next(item for item in app.checkbox if "이전 요청" in item.label).check().run()
            next(item for item in app.button if item.label == "새 해설 요청 준비").click().run()
            self.assertEqual(app.session_state[result_key]["status"], "new_request_prepared")
            send.assert_not_called()
            next(item for item in app.button if item.label == "확인한 자료로 해설 요청").click().run()
            self.assertTrue(any("먼저" in item.value for item in app.warning))
            send.assert_not_called()
            next(item for item in app.checkbox if "표시된 자료" in item.label).check()
            next(item for item in app.button if item.label == "확인한 자료로 해설 요청").click().run()
            self.assertFalse(app.exception)
            send.assert_called_once()
            app.run()
            send.assert_called_once()


class AutoCommentaryIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.model = default_settings()["model"]
        self.policy = default_policy() | {"local_default_allow": True, "model_binding": model_binding(self.model)}
        self.controller = SimpleNamespace(name="synthetic-auto", module_id="tabular")
        self.st = MagicMock()
        self.st.session_state = {"ui_settings": {"model": self.model},
                                 "commentary_frozen_" + scope(self.controller): "general"}
        self.result = {"status": "completed", "files": [
            {"name": "missing.csv", "data": b"feature,missing_ratio\na,0.1\n"}]}

    def test_auto_hook_real_ledger_calls_once_and_preserves_first_reply(self):
        with TemporaryDirectory() as directory, \
                patch("src.ui.commentary_view.load_policy", return_value=self.policy), \
                patch("src.ui.workflow_results.load_workflow_results", return_value=self.result), \
                patch("src.ui.commentary_service.ModelCommandLedger",
                      side_effect=lambda: ModelCommandLedger(Path(directory) / "calls.sqlite3")), \
                patch("src.ui.commentary_service.request_local_proposal",
                      return_value=json.dumps({"observations": ["결측률은 0.1입니다."], "limitations": []})) as model:
            after_completion(self.st, self.controller, workflow=True)
            first = deepcopy(self.st.session_state["commentary_result_" + scope(self.controller)])
            after_completion(self.st, self.controller, workflow=True)
            after_completion(self.st, self.controller, workflow=True)
            model.assert_called_once()
            self.assertEqual(self.st.session_state["commentary_result_" + scope(self.controller)], first)

    def test_auto_policy_denials_do_not_even_load_results(self):
        cases = (("unknown", self.policy), ("sensitive", self.policy),
                 ("sensitive", self.policy | {"sensitive_action": "confirm"}),
                 ("general", self.policy | {"local_default_allow": False}),
                 ("general", self.policy | {"model_binding": "0" * 64}))
        for sensitivity, policy in cases:
            with self.subTest(sensitivity=sensitivity, policy=policy), \
                    patch("src.ui.commentary_view.load_policy", return_value=policy), \
                    patch("src.ui.workflow_results.load_workflow_results") as results, \
                    patch("src.ui.commentary_service.request_local_proposal") as model:
                self.st.session_state["commentary_frozen_" + scope(self.controller)] = sensitivity
                after_completion(self.st, self.controller, workflow=True)
                results.assert_not_called()
                model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
