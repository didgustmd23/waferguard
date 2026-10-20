"""합성 CSV·기존 승인 실행기·모의 OS 운송으로 제안 바인더를 검사한다."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
import unittest
from unittest.mock import patch

from src.agents.csv_workflow_service import CsvWorkflowService
from src.agents.llm_workflow_binding import WorkflowStepBinding, bind_workflow_step, create_bound_step
from src.agents.llm_workflow_contract import validate_workflow_proposal
from src.agents.preparation_module_plan import validate_preparation_modules
from src.agents.workflow_service_registry import CSV_IDS
from src.ui.approval_flow import ApprovalFlow
from src.ui.workflow_service import UiWorkflow
from tests.agents import test_csv_workflow_service as fixtures
from tests.agents.test_llm_workflow_contract import EXAMPLE


class WorkflowBindingTest(unittest.TestCase):
    # 기존 통합 검사의 합성 fixture만 공유한다. 기존 검사 메서드를 중복 상속하지 않는다.
    setUp = fixtures.CsvWorkflowIntegrationTest.setUp
    start_patch = fixtures.CsvWorkflowIntegrationTest.start_patch
    transport = fixtures.CsvWorkflowIntegrationTest.transport
    finish = fixtures.CsvWorkflowIntegrationTest.finish

    def selection(self):
        value = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        value["workflow"]["steps"][0]["settings"] = asdict(self.profile.split_rules)
        rules = validate_preparation_modules(json.loads(self.pipeline_path.read_text(encoding="utf-8"))).rules_json
        value["workflow"]["steps"][2]["settings"] = json.loads(rules)
        value["workflow"]["steps"].append({
            "step_id": "handoff", "module_id": "csv_local_handoff", "version": "1.0.0",
            "inputs": {"prepared": "prepared.saved"}, "settings": {},
            "depends_on": [value["workflow"]["steps"][-1]["step_id"]],
        })
        self.raw = value
        self.context = {"allowed_modules": tuple((name, "1.0.0") for name in CSV_IDS),
                        "selected_settings": {step["module_id"]: step["settings"] for step in value["workflow"]["steps"]}}
        self.proposal = validate_workflow_proposal(json.dumps(value), **self.context)
        self.steps = value["workflow"]["steps"]
        self.completed = {}
        self.values = [
            {"name": "split01", "source": str(self.source), "profile": str(self.profile_path),
             "max_input_bytes": 65536, "allow_feature_findings": False},
            {"name": "export01", "split_name": "split01", "max_partition_bytes": 65536},
            {"name": "prep01", "split_name": "split01", "export_name": "export01",
             "pipeline": str(self.pipeline_path), "max_candidate_bytes": 65536, "max_partition_bytes": 65536},
            {"pipeline_name": "prep01"},
            {"name": "handoff01", "pipeline_name": "prep01", "max_bundle_bytes": 1024 ** 2},
        ]

    def bind(self, index, **overrides):
        args = {**self.context, "step_id": self.steps[index]["step_id"], "values": self.values[index],
                "limits": self.limits, "completed_runs": self.completed}
        args.update(overrides)
        return bind_workflow_step(self.proposal, **args)

    def advance(self, index):
        binding = self.bind(index)
        result = create_bound_step(binding, confirmed=True)
        self.assertFalse(result["worker_started"])
        self.assertFalse(result["execution_approval_issued"])
        self.assertEqual(result["binding_id"], binding.binding_id)
        controller = UiWorkflow(self.steps[index]["module_id"], "1.0.0")
        controller.connect(result["run_name"])
        self.finish(ApprovalFlow(controller, workflow=True))
        self.completed[self.steps[index]["step_id"]] = result["run_name"]
        return binding

    def test_five_steps_use_existing_services_and_separate_approvals(self):
        self.selection()
        before = self.source.read_bytes()

        for index in range(5):
            binding = self.advance(index)

            if index == 3:
                proof = binding.packet["evidence"]["candidate.saved"]
                self.assertEqual(len(proof["lineage"]["candidate"]), 64)
                self.assertIn("candidate", proof["local_reports"])

        self.assertEqual(before, self.source.read_bytes())
        self.assertEqual(self.calls.count("derive_preparation"), 1)
        self.assertEqual(self.calls.count("apply_preparation"), 1)

    def test_preview_has_no_creation_approval_worker_or_model_effect(self):
        self.selection()

        with patch.object(CsvWorkflowService, "create") as create, patch("http.client.HTTPConnection") as model:
            binding = self.bind(0)

        create.assert_not_called()
        model.assert_not_called()
        self.assertEqual(self.calls, [])
        self.assertFalse((self.root / "split/split01").exists())
        self.assertTrue(binding.packet["local_only"])

        with self.assertRaises(ValueError):
            create_bound_step(binding)

    def test_changed_source_and_rules_reject_before_creation(self):
        self.selection()
        binding = self.bind(0)
        self.source.write_bytes(self.source.read_bytes() + b"\n")

        with self.assertRaises(ValueError):
            create_bound_step(binding, confirmed=True)

        profile = self.profile.to_dict()
        profile["split_rules"]["train_ratio"] = 0.6
        profile["split_rules"]["validation_ratio"] = 0.2
        profile["split_rules"]["test_ratio"] = 0.2
        self.profile_path.write_text(json.dumps(profile), encoding="utf-8")

        with self.assertRaises(ValueError):
            self.bind(0)

        self.assertFalse((self.root / "split/split01").exists())

    def test_forged_plan_clarification_and_skipped_steps_are_rejected(self):
        self.selection()

        with self.assertRaises(ValueError):
            bind_workflow_step(replace(self.proposal, plan_signature="0" * 64), **self.context,
                               step_id=self.steps[0]["step_id"], values=self.values[0], limits=self.limits)

        with self.assertRaises(ValueError):
            self.bind(2)

        with self.assertRaises(ValueError):
            self.bind(0, completed_runs={"fake": "split01"})

        with self.assertRaises(ValueError):
            self.bind(0, step_id="not_registered")

    def test_code_change_and_existing_run_cannot_be_reused(self):
        self.selection()
        binding = self.bind(0)

        with patch("src.agents.llm_workflow_binding.binding_code_id", return_value="changed"):
            with self.assertRaises(ValueError):
                create_bound_step(binding, confirmed=True)

        create_bound_step(binding, confirmed=True)

        with self.assertRaises(ValueError):
            create_bound_step(binding, confirmed=True)

    def test_created_but_unfinished_parent_does_not_enable_next_step(self):
        self.selection()
        create_bound_step(self.bind(0), confirmed=True)
        self.completed[self.steps[0]["step_id"]] = "split01"

        with self.assertRaises(ValueError):
            self.bind(1)

        self.assertEqual(self.calls, [])

    def test_candidate_plan_settings_must_match_proposal(self):
        self.selection()
        self.advance(0)
        self.advance(1)
        changed = deepcopy(self.context)
        changed["selected_settings"]["csv_prepare_candidate"]["scaling"] = "standard"
        raw = deepcopy(self.raw)
        raw["workflow"]["steps"][2]["settings"] = changed["selected_settings"]["csv_prepare_candidate"]
        self.proposal = validate_workflow_proposal(json.dumps(raw), **changed)

        with self.assertRaises(ValueError):
            self.bind(2, **changed)

        self.assertFalse((self.root / "pipeline/prep01").exists())

    def test_completed_parent_file_tamper_blocks_creation(self):
        self.selection()
        self.advance(0)
        binding = self.bind(1)
        from src.agents.snapshot_split_cli import open_run

        tool = open_run("split01")
        output = tool.states.directory / tool.states.load().artifact_refs["commit_split"]
        (output / "split.json").write_text("{}", encoding="utf-8")

        with self.assertRaises(ValueError):
            create_bound_step(binding, confirmed=True)

        self.assertFalse((self.root / "export/export01").exists())

    def test_middle_start_requires_actual_matching_parent_evidence(self):
        self.selection()

        for index in range(3):
            self.advance(index)

        raw = deepcopy(self.raw)
        raw["workflow"]["steps"] = [deepcopy(self.steps[3])]
        raw["workflow"]["steps"][0]["depends_on"] = []
        context = {"allowed_modules": (("csv_prepare_apply", "1.0.0"),),
                   "selected_settings": {"csv_prepare_apply": {}},
                   "existing_inputs": ("split.confirmed", "export.development", "candidate.saved")}
        proposal = validate_workflow_proposal(json.dumps(raw), **context)
        runs = {"split.confirmed": "split01", "export.development": "export01", "candidate.saved": "prep01"}
        binding = bind_workflow_step(proposal, **context, step_id=self.steps[3]["step_id"],
                                    values=self.values[3], limits=self.limits, existing_runs=runs)
        self.assertEqual(binding.packet["creation"]["parents"]["candidate"],
                         binding.packet["evidence"]["candidate.saved"]["lineage"]["candidate"])

        with self.assertRaises(ValueError):
            bind_workflow_step(proposal, **context, step_id=self.steps[3]["step_id"],
                               values=self.values[3], limits=self.limits, existing_runs={})

    def test_conflicting_ancestry_rejected_before_creation_preview(self):
        self.selection()
        self.completed = {self.steps[0]["step_id"]: "split01", self.steps[1]["step_id"]: "export01"}
        with patch("src.agents.llm_workflow_binding.completed_workflow", side_effect=[
            {"lineage": {"split": "first"}}, {"lineage": {"split": "different"}},
        ]), patch.object(CsvWorkflowService, "preview_form") as preview:
            with self.assertRaises(ValueError):
                self.bind(2)

        preview.assert_not_called()

    def test_native_parent_identity_must_equal_bound_evidence(self):
        self.selection()
        self.completed = {self.steps[0]["step_id"]: "split01"}
        from src.agents.csv_workflow_creation import CsvCreationPreview

        native = CsvCreationPreview(json.dumps({"values": self.values[1], "parents": {"split": "different"}}))
        with patch("src.agents.llm_workflow_binding.completed_workflow", return_value={"lineage": {"split": "first"}}):
            with patch.object(CsvWorkflowService, "preview_form", return_value=native):
                with self.assertRaises(ValueError):
                    self.bind(1)

    def test_clarification_is_not_a_creation_plan(self):
        self.selection()
        response = {"schema_version": "llm_workflow_proposal_v1", "status": "needs_clarification",
                    "workflow": None, "questions": ["원하는 작업을 알려주세요."]}
        self.proposal = validate_workflow_proposal(json.dumps(response), **self.context)

        with self.assertRaises(ValueError):
            self.bind(0)

    def test_mutated_binding_limits_rejected_without_creation(self):
        self.selection()
        packet = self.bind(0).packet
        packet["limits"]["memory_mib"] += 1
        forged = WorkflowStepBinding(json.dumps(packet, ensure_ascii=False, sort_keys=True))

        with patch.object(CsvWorkflowService, "create") as create:
            with self.assertRaises(ValueError):
                create_bound_step(forged, confirmed=True)

        create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
