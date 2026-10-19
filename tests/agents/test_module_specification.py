"""공통 명세 검증. 데이터 처리·승인·worker 없이 기존 계약과 대조한다."""

from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import io
import json
import unittest
from unittest.mock import patch

from src.agents.agent_module_registry import AgentModuleRegistry, build_agent_registry
from src.agents.module_specification import CATALOGUE_PATH, ModuleCatalogue, load_catalogue


class ModuleSpecificationTest(unittest.TestCase):
    def setUp(self):
        self.registry = build_agent_registry()
        self.document = json.loads(CATALOGUE_PATH.read_text(encoding="utf-8"))

    def make(self, document=None):
        return ModuleCatalogue(self.document if document is None else document, self.registry)

    def test_all_registered_contracts_have_one_specification(self):
        specs = self.make().list()
        self.assertEqual(len(specs), 16)
        self.assertEqual({(item["contract"]["module_id"], item["contract"]["version"]) for item in specs},
                         {(item.module_id, item.version) for item in self.registry.contracts()})
        for spec in specs:
            self.assertEqual(spec["contract"]["network"], "deny_all")
            self.assertEqual(spec["contract"]["original_policy"], "preserve")
            expected = "service_registered" if spec["execution"]["kind"] == "workflow" else "declaration_only"
            self.assertEqual(spec["execution"]["connection_status"], expected)

    def test_workflow_stages_are_inherited_not_flattened(self):
        cache = self.registry.specification("wm_cache_prepare", "1.0.0")
        self.assertEqual(cache["execution"]["kind"], "workflow")
        self.assertEqual(cache["execution"]["review_stages"], ["review_eda", "commit_split"])
        self.assertEqual(cache["execution"]["worker_stage"], "prepare_data")
        self.assertFalse(cache["settings"]["form_ready"])
        self.assertEqual(cache["settings"]["scope"], "workflow_creation")
        self.assertEqual(cache["settings"]["properties"], {})

    def test_settings_defaults_come_from_registered_validators(self):
        for identifier in ("tabular_quality", "tabular_prepare_propose"):
            module = self.registry.resolve(identifier, "1.0.0")
            fields = self.registry.specification(identifier, "1.0.0")["settings"]["properties"]
            expected = json.loads(json.dumps(asdict(module.validate_settings()["rules"])))
            self.assertEqual({name: rule["default"] for name, rule in fields.items()}, expected)
            self.assertEqual(set(fields), set(module.contract.settings_fields))
            module.validate_settings(expected)

    def test_options_and_bounds_agree_with_existing_validators(self):
        for identifier in ("tabular_quality", "tabular_prepare_propose"):
            module = self.registry.resolve(identifier, "1.0.0")
            fields = self.registry.specification(identifier, "1.0.0")["settings"]["properties"]
            for name, rule in fields.items():
                for value in rule.get("enum", []):
                    module.validate_settings({name: value})
                for key, direction in (("minimum", -1), ("maximum", 1)):
                    if key in rule:
                        with self.subTest(module=identifier, field=name), self.assertRaises(ValueError):
                            module.validate_settings({name: rule[key] + direction})

    def test_exposure_filter_matches_current_llm_scope(self):
        from src.agents.llm_proposal import ALLOWED
        specs = self.registry.specifications(audience="llm")
        self.assertEqual({(item["contract"]["module_id"], item["contract"]["version"]) for item in specs}, set(ALLOWED))
        self.assertEqual(len(self.registry.specifications(audience="ui")), 16)
        for audience in ("unknown", [], None):
            with self.assertRaises(ValueError):
                self.make().list(audience=audience)

    def test_returned_data_cannot_mutate_registry(self):
        catalogue = self.make()
        before = catalogue.signature
        returned = catalogue.list()
        returned[0]["execution"]["service_id"] = "changed"
        returned[0]["contract"]["inputs"].clear()
        self.assertEqual(catalogue.signature, before)
        self.assertTrue(catalogue.list()[0]["contract"]["inputs"])
        self.assertEqual(before, self.make().signature)
        self.document["modules"][0]["title"] += " 수정"
        self.assertNotEqual(before, self.make().signature)

    def test_duplicate_missing_unknown_versions_rejected(self):
        cases = []
        changed = deepcopy(self.document)
        changed["modules"].append(deepcopy(changed["modules"][0]))
        cases.append(changed)
        changed = deepcopy(self.document)
        changed["modules"].pop()
        cases.append(changed)
        changed = deepcopy(self.document)
        changed["modules"][0]["version"] = "latest"
        cases.append(changed)
        for document in cases:
            with self.assertRaises(ValueError):
                self.make(document)

    def test_invalid_fields_import_paths_display_policy_rejected(self):
        for field, value in (("module_id", "os.system"), ("execution_profile", "src.anywhere"),
                             ("settings_schema", ["empty"]), ("result_views", ["html"]),
                             ("result_views", ["table", "table"]), ("ui_visible", "true"),
                             ("llm_proposable", 1), ("title", "bad\x1b"), ("description", ""),
                             ("import_path", "arbitrary.module")):
            document = deepcopy(self.document)
            document["modules"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.make(document)

    def test_wrong_execution_or_settings_bindings_rejected(self):
        cases = (("wm_cache_prepare", "execution_profile", "tabular_analysis"),
                 ("wm_cache_prepare", "settings_schema", "empty"),
                 ("eda_missing", "settings_schema", "workflow_creation"),
                 ("eda_missing", "settings_schema", "quality_rules"),
                 ("tabular_quality", "settings_schema", "preparation_rules"),
                 ("tabular_quality", "settings_schema", "empty"),
                 ("tabular_prepare_apply", "execution_profile", "tabular_candidate"))
        for identifier, field, value in cases:
            document = deepcopy(self.document)
            next(item for item in document["modules"] if item["module_id"] == identifier)[field] = value
            with self.subTest(module=identifier, field=field), self.assertRaises(ValueError):
                self.make(document)

    def test_requires_sealed_registry_and_exact_lookup(self):
        with self.assertRaises(ValueError):
            AgentModuleRegistry().catalogue()
        for identifier, version in (("missing", "1.0.0"), ("eda_missing", "latest"), ("eda_missing", [])):
            with self.assertRaises(ValueError):
                self.registry.specification(identifier, version)

    def test_duplicate_keys_and_nonstandard_json_rejected(self):
        for payload in (b'{"schema_version":1,"schema_version":2}', b'{"modules":NaN}'):
            with patch("src.agents.module_specification._read_file", return_value=payload):
                with self.assertRaises(ValueError):
                    load_catalogue(self.registry)

    def test_listing_does_not_start_services_or_write_files(self):
        with patch("subprocess.Popen", side_effect=AssertionError("프로세스 금지")), patch(
                "sqlite3.connect", side_effect=AssertionError("DB 접근 금지")), patch(
                "pathlib.Path.write_text", side_effect=AssertionError("파일 저장 금지")), patch(
                "src.agents.wm_module_catalogue._cli", side_effect=AssertionError("실행 연결 금지")), patch(
                "src.preprocessing.eda.analyze_missing", side_effect=AssertionError("계산 금지")):
            self.assertEqual(len(build_agent_registry().specifications()), 16)

    def test_cli_list_and_show_are_declarations_only(self):
        from src.agents.module_specification_cli import main
        for arguments in (["list"], ["list", "--audience", "llm"],
                          ["show", "--module", "wm_cache_prepare"]):
            stream = io.StringIO()
            with patch("sys.argv", ["catalogue", *arguments]), redirect_stdout(stream):
                main()
            result = json.loads(stream.getvalue())
            self.assertEqual(result["status"], "catalogue_validated")
            self.assertTrue(result["declaration_only"])
            self.assertFalse(result["worker_started"])
            self.assertFalse(result["approval_issued"])

    def test_existing_reference_module_cli_list_and_check_are_preserved(self):
        from src.agents.module_catalogue_cli import main
        from src.agents.module_catalogue import MODULE_ID, VERSION

        for arguments in (["list"], ["check", "--module-id", MODULE_ID, "--version", VERSION]):
            stream = io.StringIO()
            with patch("sys.argv", ["catalogue", *arguments]), redirect_stdout(stream):
                main()
            result = json.loads(stream.getvalue())
            self.assertEqual(result["status"], "module_spec_verified")
            self.assertEqual(result["module_id"], MODULE_ID)
            self.assertFalse(result["module_imported"])
            self.assertFalse(result["data_read"])


if __name__ == "__main__":
    unittest.main()
