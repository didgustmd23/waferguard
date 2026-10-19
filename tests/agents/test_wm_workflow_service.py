"""공통 서비스·기존 WM CLI 호환성. 합성 맵·모의 운송만 사용한다."""

from contextlib import closing
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.agents.agent_module_registry import build_agent_registry
from src.agents.workflow_approval import WorkflowApprovalPreview
from src.agents.wm_module_catalogue import _cli, wm_modules
from src.agents.wm_workflow_service import WmWorkflowService
from src.agents.wm_runtime import runtime_status
from tests.agents import test_wm_canonical as fixtures


class WmWorkflowServiceTest(unittest.TestCase):
    transport = fixtures.WmCanonicalTest.transport

    def setUp(self):
        fixtures.WmCanonicalTest.setUp(self)
        # Python 3.12에서도 코어·서비스만 검사한다. native 호환성 통과로 해석하지 않는다.
        runtime = patch("src.agents.wm_runtime.runtime_status", return_value={"supported": True,
                        "code": "SYNTHETIC_TEST_ONLY", "python": [3, 10]})
        runtime.start()
        self.addCleanup(runtime.stop)
        self.service = build_agent_registry().service("wm_canonical_eda", "1.0.0")

    def advance(self, **options):
        state = self.tool.states.load()
        return self.service.next("example", expected_stage=state.current_stage,
                                 expected_revision=state.revision, **options)

    def pending(self):
        self.assertEqual(self.advance()["code"], "STEP_COMPLETED")
        self.assertEqual(self.advance()["code"], "APPROVAL_REQUIRED")

    def test_registry_resolves_all_wm_services_without_opening_runs(self):
        registry = build_agent_registry()
        with patch("src.agents.wm_module_catalogue._cli", side_effect=AssertionError("작업 열기 금지")):
            for module in wm_modules():
                service = registry.service(module.contract.module_id, module.contract.version)
                self.assertIsInstance(service, WmWorkflowService)
                self.assertEqual(service.module_id, module.contract.module_id)
        with self.assertRaises(ValueError):
            registry.service("eda_missing", "1.0.0")

    def test_no_terminal_approval_completion_and_common_result_views(self):
        self.pending()
        with patch("builtins.input", side_effect=AssertionError("터미널 입력 금지")), patch(
                "src.agents.wm_canonical.run_wm_worker", side_effect=self.transport) as worker:
            preview = self.service.preview_approval("example")
            self.assertEqual(self.service.status("example")["actions"][0]["id"], "approve")
            with self.assertRaises(ValueError):
                self.service.record_approval("example", preview)
            approval = self.service.record_approval("example", preview, confirmed=True)
            worker.assert_not_called()
            self.assertEqual(self.service.status("example")["actions"][0]["id"], "execute")
            with self.assertRaises(ValueError):
                self.advance()
            result = self.advance(approval_id=approval["approval_id"])
            self.assertEqual(result["status"], "completed")
            worker.assert_called_once()
            report = self.service.results("example")
            self.assertEqual({item["kind"] for item in report["items"]}, {"table", "image", "report"})
            self.assertFalse(report["llm_transmitted"])
            self.assertTrue(all(item["verified"] for item in report["items"]))
            self.assertEqual(self.service.status("example")["actions"], [])

    def test_duplicate_approval_and_changed_preview_are_rejected(self):
        self.pending()
        preview = self.service.preview_approval("example")
        changed = preview.to_dict()
        changed["summary"]["limits"]["processes"] = 8
        with self.assertRaises(ValueError):
            self.service.record_approval("example", WorkflowApprovalPreview(json.dumps(changed)), confirmed=True)
        self.service.record_approval("example", preview, confirmed=True)
        with self.assertRaises(ValueError):
            self.service.record_approval("example", preview, confirmed=True)
        with closing(sqlite3.connect(self.directory / "approvals.sqlite3")) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM approvals").fetchone()[0], 1)

    def test_state_changed_after_preview_rejected(self):
        self.pending()
        preview = self.service.preview_approval("example")
        state = self.tool.states.load()
        self.tool.states.save(state, expected_revision=state.revision)
        with self.assertRaises(ValueError):
            self.service.record_approval("example", preview, confirmed=True)

    def test_stale_next_does_not_advance_a_later_stage(self):
        state = self.tool.states.load()
        self.advance()
        with patch("src.agents.wm_canonical.run_wm_worker") as worker:
            with self.assertRaises(ValueError):
                self.service.next("example", expected_stage=state.current_stage, expected_revision=state.revision)
        worker.assert_not_called()
        self.assertEqual(self.tool.states.load().status, "running")

    def test_cancelled_run_has_no_actions_or_approval(self):
        from src.agents.isolated_eda_cli import cancel
        self.pending()
        preview = self.service.preview_approval("example")
        cancel(self.tool)
        self.assertEqual(self.service.status("example")["actions"], [])
        with self.assertRaises(ValueError):
            self.service.record_approval("example", preview, confirmed=True)

    def test_quality_review_keeps_only_verified_quality_results(self):
        from src.agents.wm_canonical_cli import create_demo, open_run
        directory = self.root / "runs/review"
        with patch("src.agents.wm_canonical_cli._directory", return_value=directory):
            create_demo("review", self.tool.limits, review=True)
            self.tool = open_run("review")
            for _ in range(2):
                state = self.tool.states.load()
                self.service.next("review", expected_stage=state.current_stage, expected_revision=state.revision)
            preview = self.service.preview_approval("review")
            approval = self.service.record_approval("review", preview, confirmed=True)
            with patch("src.agents.wm_canonical.run_wm_worker", side_effect=self.transport), patch(
                    "src.wafer_dl.eda.run_eda", side_effect=AssertionError("검토 중단 뒤 EDA 금지")):
                state = self.tool.states.load()
                result = self.service.next("review", expected_stage=state.current_stage,
                                           expected_revision=state.revision, approval_id=approval["approval_id"])
            self.assertEqual(result["code"], "TOOL_REVIEW_REQUIRED")
            report = self.service.results("review")
            self.assertFalse(report["details"]["eda_generated"])
            self.assertTrue(all(item["kind"] == "report" for item in report["items"]))
            self.assertEqual(self.service.status("review")["actions"], [])

    def test_resource_failure_never_becomes_success(self):
        self.pending()
        preview = self.service.preview_approval("example")
        approval = self.service.record_approval("example", preview, confirmed=True)

        def failed(request, root, signal):
            result = self.transport(request, root, signal)
            result["launch"]["status"] = "resource_exceeded"
            return result

        with patch("src.agents.wm_canonical.run_wm_worker", side_effect=failed):
            self.assertEqual(self.advance(approval_id=approval["approval_id"])["status"], "failed_runtime")
        self.assertEqual(self.service.status("example")["actions"], [])
        with self.assertRaises(ValueError):
            self.service.results("example")

    def test_creation_delegates_existing_preview_and_rejects_other_modules(self):
        from src.agents.wm_run_creation import CreationPreview
        preview = CreationPreview(json.dumps({"module_id": "wm_canonical_eda"}))
        with patch("src.agents.wm_run_creation.preview_creation", return_value=preview) as prepare:
            returned = self.service.preview_creation("plan.json", "bindings.json", "eda", "new", self.tool.limits)
            self.assertIs(returned, preview)
            prepare.assert_called_once()
        with patch("src.agents.wm_run_creation.create_and_assign", return_value={"worker_started": False}) as create:
            for confirmation in (False, "yes", 1):
                with self.assertRaises(ValueError):
                    self.service.create(preview, confirmed=confirmation)
            create.assert_not_called()
            self.assertFalse(self.service.create(preview, confirmed=True)["worker_started"])
            create.assert_called_once_with(preview, confirmed=True)
        with self.assertRaises(ValueError):
            WmWorkflowService("wm_conflict_apply").preview_creation("p", "b", "s", "n", self.tool.limits)
        with self.assertRaises(ValueError):
            self.service.create(CreationPreview(json.dumps({"module_id": "wm_cache_prepare"})), confirmed=True)

    def test_common_results_reject_modified_files(self):
        self.pending()
        approval = self.service.record_approval("example", self.service.preview_approval("example"), confirmed=True)
        with patch("src.agents.wm_canonical.run_wm_worker", side_effect=self.transport):
            self.advance(approval_id=approval["approval_id"])
        report = self.service.results("example")
        table = next(item for item in report["items"] if item["kind"] == "table")
        Path(table["path"]).write_bytes(b"changed")
        with self.assertRaises(ValueError):
            self.service.results("example")


class WmRuntimePolicyTest(unittest.TestCase):
    def test_runtime_policy_keeps_unverified_312_blocked(self):
        with patch("src.agents.wm_runtime.sys.version_info", (3, 12, 0)), \
                patch("src.agents.wm_runtime._installed_torch_build", return_value="2.14.2+cu130"):
            self.assertFalse(runtime_status()["supported"])
            from src.agents.wm_canonical_cli import _preflight
            self.assertEqual(_preflight(None, {}).status, "blocked_input")

    def test_312_support_is_limited_to_six_modules_and_pinned_cpu_build(self):
        from src.agents.wm_runtime import CPU_312_MODULES

        with patch("src.agents.wm_runtime.os.name", "nt"), \
                patch("src.agents.wm_runtime.sys.version_info", (3, 12, 15)), \
                patch("src.agents.wm_runtime._installed_torch_build", return_value="2.14.1+cpu"):
            for module in wm_modules():
                name = module.contract.module_id
                self.assertEqual(runtime_status(name)["supported"], name in CPU_312_MODULES)
            self.assertFalse(runtime_status()["supported"])
            self.assertFalse(runtime_status("unknown")["supported"])
            self.assertFalse(runtime_status([])["supported"])

    def test_unknown_build_inspection_failure_and_future_python_remain_blocked(self):
        with patch("src.agents.wm_runtime.os.name", "nt"), \
                patch("src.agents.wm_runtime.sys.version_info", (3, 12, 15)):
            for build in ("2.14.1+cu128", "2.14.2+cpu", None):
                with patch("src.agents.wm_runtime._installed_torch_build", return_value=build):
                    self.assertFalse(runtime_status("wm_dataset_check")["supported"])
            with patch("src.agents.wm_runtime._installed_torch_build", side_effect=ValueError("synthetic")):
                self.assertFalse(runtime_status("wm_dataset_check")["supported"])

        with patch("src.agents.wm_runtime.sys.version_info", (3, 13, 0)):
            self.assertFalse(runtime_status("wm_dataset_check")["supported"])

    def test_cuda_installation_uses_same_environment_but_cpu_preprocessing(self):
        from src.agents.wm_runtime import CPU_312_MODULES

        with patch("src.agents.wm_runtime.os.name", "nt"), \
                patch("src.agents.wm_runtime.sys.version_info", (3, 12, 15)), \
                patch("src.agents.wm_runtime._installed_torch_build", return_value="2.14.1+cu130"):
            for module in wm_modules():
                name = module.contract.module_id
                result = runtime_status(name)
                self.assertEqual(result["supported"], name in CPU_312_MODULES)
                self.assertEqual(result["worker_device"], "cpu")

            self.assertFalse(runtime_status()["supported"])

    def test_all_eight_clis_expose_shared_approval_surface_and_runtime_policy(self):
        with patch("src.agents.wm_runtime.runtime_status", return_value={"supported": False}):
            for module in wm_modules():
                cli = _cli(module.cli_id)
                self.assertTrue(callable(cli.approval_summary))
                self.assertTrue(callable(cli.preview_approval))
                self.assertTrue(callable(cli.record_approval))
                self.assertEqual(cli._preflight(None, {}).status, "blocked_input")

    def test_existing_confirmation_tokens_delegate_shared_saver(self):
        prefixes = {"wm_canonical_eda": "WM-", "wm_conflict_candidate": "CONFLICT-",
                    "wm_conflict_apply": "EXCLUDE-", "wm_split_propose": "WM-SPLIT-",
                    "wm_split_commit": "WM-COMMIT-", "wm_transform_check": "WM-TRANSFORM-",
                    "wm_cache_prepare": "WM-CACHE-", "wm_dataset_check": "WM-DATASET-"}
        for module in wm_modules():
            cli = _cli(module.cli_id)
            tool = SimpleNamespace(states=SimpleNamespace(load=lambda: SimpleNamespace(current_stage=module.worker_stage)))
            preview = WorkflowApprovalPreview(json.dumps({"summary": {}}))
            with patch.object(cli, "preview_approval", return_value=preview), patch.object(
                    cli, "record_approval", return_value={"approval_id": "fixture"}) as save, patch.object(
                    cli.os, "isatty", return_value=True), patch.object(
                    cli.secrets, "token_hex", return_value="1234abcd"), patch(
                    "builtins.input", return_value=prefixes[module.contract.module_id] + "1234ABCD"), patch("builtins.print"):
                self.assertEqual(cli.approve(tool)["approval_id"], "fixture")
                save.assert_called_once_with(tool, preview, confirmed=True)

    def test_all_wm_code_manifests_cover_common_control_without_worker_permissions(self):
        import importlib
        required = {"src/agents/workflow_approval.py", "src/agents/wm_runtime.py",
                    "src/agents/wm_workflow_service.py", "src/agents/wm_workflow_execution.py",
                    "src/agents/wm_module_catalogue.py", "src/agents/agent_module_registry.py",
                    "src/agents/module_specification.py", "configs/agents/modules/catalogue.json"}
        for module in wm_modules():
            transport = importlib.import_module("src.agents." + module.cli_id.removesuffix("_cli") + "_transport")
            self.assertTrue(required <= set(transport.CONTROL_FILES))
            self.assertIn("src/agents/wm_dataset_runtime.py", transport.CONTROL_FILES)
            self.assertFalse(required & set(transport.WORKER_FILES))


if __name__ == "__main__":
    unittest.main()
