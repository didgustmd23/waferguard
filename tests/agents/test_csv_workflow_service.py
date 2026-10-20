"""CSV 공통 서비스 전체 연결. 합성 자료와 모의 OS 운송만 사용한다."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.agents.csv_workflow_creation import preview_creation
from src.agents.csv_workflow_service import CsvWorkflowService
from src.agents.workflow_service_registry import CSV_IDS, workflow_specifications, workflow_service
from src.agents.worker_contract import WorkerLimits
from src.ui.approval_flow import ApprovalFlow
from src.ui.workflow_results import load_workflow_results
from src.ui.workflow_service import UiWorkflow
from tests.agents import test_snapshot_split_execution as split_fixtures


class CatalogueTest(unittest.TestCase):
    def test_csv_and_wm_are_in_same_ui_list_without_changing_legacy_registry(self):
        from src.agents.agent_module_registry import build_agent_registry

        self.assertEqual(len(build_agent_registry().contracts()), 16)
        specs = workflow_specifications()
        self.assertEqual(len(specs), 13)
        self.assertTrue(set(CSV_IDS) <= {item["contract"]["module_id"] for item in specs})
        for module in CSV_IDS:
            service = workflow_service(module, "1.0.0")
            self.assertTrue(service.creation_schema()["supported"])
        with self.assertRaises(ValueError):
            workflow_service(CSV_IDS[0], "2.0.0")

    def test_verified_versions_are_enabled_and_other_platforms_are_blocked(self):
        for system, version, expected in (("nt", (3, 10, 0), True), ("nt", (3, 12, 15), True),
                                          ("nt", (3, 11, 0), False), ("nt", (3, 13, 0), False),
                                          ("posix", (3, 12, 15), False)):
            with self.subTest(system=system, version=version), patch(
                    "src.agents.csv_workflow_service.os.name", system), patch(
                    "src.agents.csv_workflow_service.sys.version_info", version):
                for module in CSV_IDS:
                    self.assertEqual(CsvWorkflowService(module).runtime()["supported"], expected)

    def test_four_cli_preflights_accept312_but_reject_unknown_versions_and_arguments(self):
        from src.agents import snapshot_split_cli, partition_export_cli
        from src.agents import preparation_candidate_cli, preparation_apply_cli

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for cli in (snapshot_split_cli, partition_export_cli, preparation_candidate_cli, preparation_apply_cli):
                output = root / cli.__name__.split(".")[-1]
                output.mkdir()

                with patch.object(cli.os, "name", "nt"), patch.object(cli.sys, "version_info", (3, 12, 15)):
                    self.assertEqual(cli._preflight(output, {}).status, "passed")
                    self.assertTrue(cli._validate_preflight(output, {}))
                    self.assertEqual(cli._preflight(output, {"unexpected": True}).status, "blocked_input")

                for version in ((3, 11, 0), (3, 13, 0)):
                    with patch.object(cli.sys, "version_info", version):
                        self.assertEqual(cli._preflight(output, {}).status, "blocked_input")

    def test_unsupported_runtime_never_opens_or_executes_a_run(self):
        service = CsvWorkflowService("csv_split_review")
        with patch.object(service, "runtime", return_value={"supported": False}), patch.object(service, "_open") as opened:
            with self.assertRaises(ValueError):
                service.next("ignored", expected_stage="preflight", expected_revision=0)
            with self.assertRaises(ValueError):
                service.record_approval("ignored", None, confirmed=True)
            opened.assert_not_called()


class CsvWorkflowIntegrationTest(unittest.TestCase):
    transport = split_fixtures.SnapshotSplitExecutionTest.transport

    def setUp(self):
        split_fixtures.SnapshotSplitExecutionTest.setUp(self)
        self.limits = WorkerLimits(60, 2048, 2)
        self.handoff_runs = self.root / "handoff-runs"
        self.handoff_runs.mkdir()
        self.bundles = self.root / "bundles"
        self.bundles.mkdir()
        self.pipeline_path = Path(__file__).resolve().parents[2] / "configs/agents/pipelines/tabular_preparation_example.json"

        for identifier, directory in (("snapshot_split", "split"), ("partition_export", "export"),
                                      ("preparation_candidate", "candidate"), ("preparation_apply", "apply"),
                                      ("preparation_pipeline", "pipeline")):
            parent = self.root / directory
            parent.mkdir()
            self.start_patch("src.agents." + identifier + "_cli._directory",
                             side_effect=lambda name, parent=parent, **kw: parent / name)
            self.start_patch("src.agents." + identifier + "_cli._current_actor", return_value="synthetic_actor")
            if identifier != "preparation_pipeline":
                import importlib

                cli = importlib.import_module("src.agents." + identifier + "_cli")
                original = cli._preflight

                def preflight(output, arguments, original=original):
                    # 실제 OS 운송은 모의하되 3.12 사전 검사와 승인 흐름을 확인한다.
                    with patch("sys.version_info", (3, 12, 15)):
                        return original(output, arguments)

                self.start_patch("src.agents." + identifier + "_cli._preflight", side_effect=preflight)

        self.start_patch("src.agents.csv_workflow_creation._current_actor", return_value="synthetic_actor")
        self.start_patch("src.agents.csv_handoff_workflow._current_actor", return_value="synthetic_actor")
        self.start_patch("src.agents.csv_handoff_workflow.directory_for", side_effect=lambda name: self.handoff_runs / name)
        self.start_patch("src.agents.csv_handoff_cli._directory", side_effect=lambda name, **kw: self.bundles / name)
        self.start_patch("src.agents.csv_handoff_workflow._directory", side_effect=lambda name, **kw: self.bundles / name)
        self.start_patch("src.agents.csv_workflow_service.CsvWorkflowService.runtime",
                         return_value={"supported": True, "notice": "synthetic OS only"})

        from src.agents.snapshot_split_worker import execute_request as split_worker
        from src.agents.partition_export_worker import execute_request as export_worker
        from src.agents.preparation_candidate_worker import execute_request as candidate_worker
        from src.agents.preparation_apply_worker import execute_request as apply_worker

        for name, function, worker in (("snapshot_split", "run_split_worker", split_worker),
                                       ("partition_export", "run_export_worker", export_worker),
                                       ("preparation_candidate", "run_candidate_worker", candidate_worker),
                                       ("preparation_apply", "run_apply_worker", apply_worker)):
            self.start_patch("src.agents." + name + "." + function,
                side_effect=lambda request, root, signal, worker=worker: self.transport(request, root, signal, worker))

    def start_patch(self, target, **kwargs):
        patcher = patch(target, **kwargs)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def prepare(self, module, values):
        service = CsvWorkflowService(module)
        preview = service.preview_form(values, self.limits)
        result = service.create(preview, confirmed=True)
        controller = UiWorkflow(module, "1.0.0")
        controller.connect(result["run_name"])
        return controller, ApprovalFlow(controller, workflow=True)

    def finish(self, flow):
        flow.run("start")
        approvals = 0
        while flow.ready:
            approvals += 1
            self.assertLess(approvals, 8)
            flow.run("approve")
        self.assertFalse(flow.error)
        self.assertEqual(flow.state["status"], "completed")
        return approvals

    def test_five_workflows_with_separate_approvals_and_verified_results(self):
        original = self.source.read_bytes()
        split, split_flow = self.prepare("csv_split_review", {"name": "split01", "source": str(self.source),
            "profile": str(self.profile_path), "max_input_bytes": 65536, "allow_feature_findings": False})
        self.assertEqual(self.finish(split_flow), 5)
        self.assertTrue(load_workflow_results(split)["files"])
        state = split.state
        with self.assertRaises(ValueError):
            split.service().next("split01", expected_stage=state["stage"], expected_revision=state["revision"] - 1)
        self.assertEqual(split.service().status("split01")["revision"], state["revision"])

        export, export_flow = self.prepare("csv_partition_export", {"name": "export01", "split_name": "split01", "max_partition_bytes": 65536})
        self.assertEqual(self.finish(export_flow), 1)
        self.assertTrue(load_workflow_results(export)["directories"])

        candidate, candidate_flow = self.prepare("csv_prepare_candidate", {"name": "prep01", "split_name": "split01",
            "export_name": "export01", "pipeline": str(self.pipeline_path), "max_candidate_bytes": 65536,
            "max_partition_bytes": 65536})
        self.assertEqual(self.finish(candidate_flow), 1)
        self.assertTrue(load_workflow_results(candidate)["files"])

        application, apply_flow = self.prepare("csv_prepare_apply", {"pipeline_name": "prep01"})
        self.assertEqual(self.finish(apply_flow), 1)
        self.assertTrue(load_workflow_results(application)["files"])

        handoff, handoff_flow = self.prepare("csv_local_handoff", {"name": "handoff01", "pipeline_name": "prep01", "max_bundle_bytes": 1024 ** 2})
        self.assertEqual(self.finish(handoff_flow), 1)
        self.assertTrue(load_workflow_results(handoff)["files"])
        self.assertEqual(original, self.source.read_bytes())
        self.assertEqual(self.calls.count("apply_preparation"), 1)
        from src.agents.csv_handoff_contract import verify_bundle

        self.assertEqual(verify_bundle(self.bundles / "handoff01")["status"], "handoff_verified")

    def test_changed_source_rejected_and_creation_requires_confirmation(self):
        service = CsvWorkflowService("csv_split_review")
        values = {"name": "split01", "source": str(self.source), "profile": str(self.profile_path),
                  "max_input_bytes": 65536, "allow_feature_findings": False}
        preview = service.preview_form(values, self.limits)
        with self.assertRaises(ValueError):
            service.create(preview)
        self.source.write_bytes(self.source.read_bytes() + b"\n")
        with self.assertRaises(ValueError):
            service.create(preview, confirmed=True)
        self.assertFalse((self.root / "split/split01").exists())
