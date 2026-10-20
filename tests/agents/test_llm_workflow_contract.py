"""신규 CSV 제안의 정적 검증. 실제 파일·모델·worker·승인은 사용하지 않는다."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from src.agents.csv_workflow_proposal_catalogue import settings_schema, workflow_proposal_specifications
from src.agents.llm_workflow_contract import validate_workflow_proposal
from src.agents.workflow_service_registry import CSV_IDS, workflow_specifications
from src.preprocessing.split import SplitRules
from src.preprocessing.transform import PreparationRules


EXAMPLE = Path(__file__).resolve().parents[2] / "configs/agents/llm/workflow_proposal_example.json"


class WorkflowContractTest(unittest.TestCase):
    def setUp(self):
        self.example = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        self.allowed = tuple((name, "1.0.0") for name in CSV_IDS)
        self.selected = {name: {} for name in CSV_IDS}
        self.selected["csv_split_review"] = asdict(SplitRules(strategy="time", timestamp_format="%d/%m/%Y %H:%M:%S"))
        self.selected["csv_prepare_candidate"] = asdict(PreparationRules(missing_ratio_threshold=0.5, drop_constant=True))

    def check(self, value=None, **kwargs):
        return validate_workflow_proposal(json.dumps(self.example if value is None else value, ensure_ascii=False),
                                         allowed_modules=self.allowed, selected_settings=self.selected, **kwargs)

    def test_full_plan_is_proposal_only_and_retains_approval_boundaries(self):
        result = self.check().to_dict()
        self.assertTrue(result["proposal_only"])
        self.assertTrue(result["requires_local_binding"])
        self.assertFalse(result["execution_approved"])
        self.assertFalse(result["worker_started"])
        self.assertEqual(result["contracts"][3]["approval_boundaries"], ["review_candidate", "apply_preparation"])
        self.assertEqual(result["contracts"][2]["criteria_policy"], "derive_on_train")

    def test_public_specs_reuse_titles_not_local_creation_forms(self):
        specs = workflow_proposal_specifications()
        ui = {item["contract"]["module_id"]: item for item in workflow_specifications()}
        self.assertEqual([item["module_id"] for item in specs], list(CSV_IDS))
        for spec in specs:
            self.assertEqual(spec["title"], ui[spec["module_id"]]["title"])
        text = json.dumps(specs)
        for private in ("snapshot_dir", "source_path", "bundle_dir", "approval_id", "python -m", "json_file"):
            self.assertNotIn(private, text)
        self.assertEqual(set(settings_schema("csv_prepare_candidate")["required"]), set(asdict(PreparationRules())))

    def test_clarification_has_no_plan_and_rejects_mixed_confirmation(self):
        value = {"schema_version": "llm_workflow_proposal_v1", "status": "needs_clarification",
                 "workflow": None, "questions": ["분할 비율을 선택해 주세요."]}
        self.assertIsNone(self.check(value).plan_signature)
        value["workflow"] = self.example["workflow"]
        with self.assertRaises(ValueError):
            self.check(value)

    def test_apply_can_start_only_with_explicit_logical_predecessors(self):
        value = deepcopy(self.example)
        value["workflow"]["steps"] = [value["workflow"]["steps"][-1]]
        value["workflow"]["steps"][0]["depends_on"] = []
        with self.assertRaises(ValueError):
            self.check(value)
        result = self.check(value, existing_inputs=("split.confirmed", "export.development", "candidate.saved"))
        self.assertIsNotNone(result.plan_signature)

    def test_missing_skipped_reordered_duplicate_and_forged_dependencies_rejected(self):
        changes = [lambda s: s.pop(1), lambda s: s.reverse(), lambda s: s.append(deepcopy(s[-1])),
                   lambda s: s[2].update(depends_on=["split"]), lambda s: s[0].update(depends_on=["apply"]),
                   lambda s: s[1].update(step_id=s[0]["step_id"])]
        for change in changes:
            value = deepcopy(self.example)
            change(value["workflow"]["steps"])
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.check(value)

    def test_test_refit_paths_and_quality_bypass_rejected(self):
        for section, field, malicious in ((0, "settings", {"allow_feature_findings": True}),
                                          (2, "settings", {"partition": "validation"}),
                                          (3, "settings", {"refit": True}),
                                          (3, "settings", {"partitions": ["test"]}),
                                          (2, "settings", {"source": "D:/private/input.csv"})):
            value = deepcopy(self.example)
            value["workflow"]["steps"][section][field].update(malicious)
            with self.subTest(malicious=malicious), self.assertRaises(ValueError):
                self.check(value)
        value = deepcopy(self.example)
        value["workflow"]["steps"][0]["inputs"] = {"data": "D:/private/input.csv"}
        with self.assertRaises(ValueError):
            self.check(value)

    def test_selected_settings_cannot_be_changed_or_omitted_by_model(self):
        value = deepcopy(self.example)
        value["workflow"]["steps"][2]["settings"]["missing_ratio_threshold"] = 0.9
        with self.assertRaises(ValueError):
            self.check(value)
        value["workflow"]["steps"][2]["settings"].pop("scaling")
        with self.assertRaises(ValueError):
            self.check(value)

    def test_core_rule_combinations_reject_bad_ratios_and_stratified_time(self):
        for change in ({"train_ratio": 0.9}, {"stratify": True}, {"random_state": True},
                       {"timestamp_format": "D:/private/data"}):
            selected = deepcopy(self.selected)
            selected["csv_split_review"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_workflow_proposal(json.dumps(self.example), allowed_modules=self.allowed, selected_settings=selected)

    def test_signature_is_canonical_and_copies_are_independent(self):
        first = self.check()
        reordered = json.loads(json.dumps(self.example, sort_keys=True))
        self.assertEqual(first.plan_signature, self.check(reordered).plan_signature)
        copy = first.to_dict()
        copy["proposal"]["workflow"]["steps"].clear()
        self.assertEqual(len(first.to_dict()["proposal"]["workflow"]["steps"]), 4)

    def test_optional_handoff_and_prefix_plan_are_explicit(self):
        value = deepcopy(self.example)
        value["workflow"]["steps"].append({"step_id": "handoff", "module_id": "csv_local_handoff",
            "version": "1.0.0", "inputs": {"prepared": "prepared.saved"}, "settings": {}, "depends_on": ["apply"]})
        self.assertEqual(len(self.check(value).to_dict()["contracts"]), 5)
        value["workflow"]["steps"] = value["workflow"]["steps"][:1]
        self.assertEqual(len(self.check(value).to_dict()["contracts"]), 1)

    def test_invalid_existing_refs_and_old_downstream_outputs_rejected(self):
        for refs in (("D:/private/candidate.json",), ("candidate.saved", "candidate.saved"), ("prepared.saved",)):
            with self.subTest(refs=refs), self.assertRaises(ValueError):
                self.check(existing_inputs=refs)

    def test_request_allowlist_is_not_the_entire_registry(self):
        with self.assertRaises(ValueError):
            validate_workflow_proposal(json.dumps(self.example), allowed_modules=(("csv_split_review", "1.0.0"),),
                                       selected_settings={"csv_split_review": self.selected["csv_split_review"]})
        for allowed in ((), (("csv_split_review", "latest"),), (("csv_split_review", "1.0.0"),) * 2):
            with self.subTest(allowed=allowed), self.assertRaises(ValueError):
                validate_workflow_proposal(json.dumps(self.example), allowed_modules=allowed, selected_settings={})

    def test_model_cannot_forge_review_or_local_binding_fields(self):
        for extra in ({"approved": True}, {"candidate_id": "forged"}, {"code": "print('unsafe')"}):
            value = deepcopy(self.example)
            value["workflow"]["steps"][-1].update(extra)
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.check(value)

    def test_all_selected_rule_combinations_must_pass_core_validation(self):
        selected = deepcopy(self.selected)
        selected["csv_prepare_candidate"].update(binning="uniform", scaling="standard")
        with self.assertRaises(ValueError):
            validate_workflow_proposal(json.dumps(self.example), allowed_modules=self.allowed, selected_settings=selected)

    def test_legacy_response_unknown_version_fields_and_oversized_json_rejected(self):
        for mutation in ({"schema_version": "llm_proposal_v1"}, {"execution_approved": True}, {"questions": ["x"]}):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.check(self.example | mutation)
        value = deepcopy(self.example)
        value["workflow"]["steps"][0]["version"] = "latest"
        with self.assertRaises(ValueError):
            self.check(value)
        for raw in ('{"schema_version":1,"schema_version":2}', '{"x":NaN}', '"' + 'x' * 65536 + '"'):
            with self.subTest(raw=raw[:50]), self.assertRaises(ValueError):
                validate_workflow_proposal(raw, allowed_modules=self.allowed, selected_settings=self.selected)

    def test_validation_does_not_execute_or_read_real_data(self):
        with patch("http.client.HTTPConnection", side_effect=AssertionError("모델 호출 금지")), \
                patch("subprocess.Popen", side_effect=AssertionError("worker 금지")), \
                patch("pandas.read_csv", side_effect=AssertionError("데이터 읽기 금지")), \
                patch("sqlite3.connect", side_effect=AssertionError("승인 DB 접근 금지")), \
                patch("src.agents.workflow_service_registry.workflow_service", side_effect=AssertionError("실행 서비스 금지")):
            self.check()


if __name__ == "__main__":
    unittest.main()
