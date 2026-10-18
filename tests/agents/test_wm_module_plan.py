"""데이터·worker 없이 등록 계약·계획·기존 승인 실행기로의 연결만 검사한다."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src.agents.agent_module_registry import AgentModuleRegistry, build_agent_registry
from src.agents.wm_module_catalogue import _cli, open_wm_execution, wm_modules
from src.agents.wm_module_plan import load_wm_modules, validate_wm_modules


EXAMPLE = Path(__file__).resolve().parents[2] / "configs/agents/pipelines/wm_development_example.json"


class WmModulePlanTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(EXAMPLE.read_text(encoding="utf-8"))

    def test_common_registry_preserves_tabular_and_adds_wm(self):
        """기존 표형 8개와 WM 8개가 중복 없이 조회되고 봉인 후 변경되지 않는다."""
        registry = build_agent_registry()
        self.assertEqual(len(registry.contracts()), 16)
        self.assertEqual(registry.resolve("eda_missing", "1.0.0").contract.kind, "analysis")
        self.assertEqual(registry.resolve("tabular_prepare_apply", "1.0.0").contract.criteria_policy, "apply_saved")

        with self.assertRaises(ValueError):
            registry.register(wm_modules()[0])

        local = AgentModuleRegistry()
        local.register(wm_modules()[0])

        with self.assertRaises(ValueError):
            local.register(wm_modules()[0])

    def test_plan_dependencies_and_no_execution(self):
        """6단계 계획과 검토 게이트를 기록하지만 실행기 import·worker 호출은 하지 않는다."""
        with patch("src.agents.wm_module_catalogue._cli", side_effect=AssertionError("실행기 접근 금지")):
            plan = load_wm_modules(EXAMPLE)

        self.assertEqual(len(plan.steps), 6)
        self.assertEqual(plan.steps[-1].depends_on, ("cache",))
        record = plan.to_dict()
        self.assertFalse(record["automatic_execution"])
        self.assertFalse(record["test_maps_allowed"])
        self.assertTrue(record["plan_only"])
        self.assertEqual(record["execution_routes"][4]["review_stages"], ["review_eda", "commit_split"])
        self.assertEqual(record["execution_routes"][-1]["cli"], "src.agents.wm_dataset_cli")

    def test_invalid_contract_and_privileges(self):
        """Test·표형 입력·미등록 버전·설정 주입·승인 생략 요청을 거부한다."""
        changes = (
            lambda item: item["input"].update(partition="test"),
            lambda item: item["input"].update(contract_id="tabular_data"),
            lambda item: item["input"].update(contract_version=True),
            lambda item: item["steps"][0].update(version="latest"),
            lambda item: item["steps"][0].update(settings={"command": "python"}),
            lambda item: item["steps"][0].update(approval=False),
            lambda item: item["steps"][0].update(module_id="eda_missing"),
        )

        for change in changes:
            with self.subTest(change=change):
                config = deepcopy(self.config)
                change(config)

                with self.assertRaises(ValueError):
                    validate_wm_modules(config)

    def test_invalid_ports_and_dependencies(self):
        """생략·단계 건너뛰기·미래 참조·중복 ID·중복 의존성을 거부한다."""
        changes = (
            lambda item: item["steps"][0].update(inputs={}),
            lambda item: item["steps"][1].update(inputs={"evidence": "source.evidence"}),
            lambda item: item["steps"][0].update(inputs={"evidence": "cache.result"}),
            lambda item: item["steps"][1].update(step_id="eda"),
            lambda item: item["steps"][1].update(depends_on=["eda", "eda"]),
            lambda item: item["steps"][1].update(depends_on=["dataset"]),
        )

        for change in changes:
            config = deepcopy(self.config)
            change(config)

            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_wm_modules(config)

    def test_conflict_branch_requires_new_canonical_eda(self):
        """품질 검토 증거는 정상 EDA 통과로 바꾸지 않고 별도 제외 경로를 거친다."""
        config = {"schema_version": "wm_module_pipeline_v1", "pipeline_id": "conflict_example",
                  "input": {"contract_id": "wm_quality_review", "contract_version": 1, "partition": "unsplit"},
                  "steps": []}
        modules = ("wm_conflict_candidate", "wm_conflict_apply", "wm_canonical_eda", "wm_split_propose")

        for number, module in enumerate(modules):
            reference = "source.evidence" if number == 0 else f"step_{number - 1}.result"
            config["steps"].append({"step_id": f"step_{number}", "module_id": module,
                                    "version": "1.0.0", "inputs": {"evidence": reference}})

        self.assertEqual(len(validate_wm_modules(config).steps), 4)
        config["steps"][-1]["inputs"]["evidence"] = "step_1.result"

        with self.assertRaises(ValueError):
            validate_wm_modules(config)

    def test_plan_signature_and_input_independence(self):
        """키 순서에는 독립적이며 입력 dict 변경은 이미 만든 계획에 영향을 주지 않는다."""
        first = validate_wm_modules(self.config)
        reordered = dict(reversed(list(self.config.items())))
        self.assertEqual(first.signature, validate_wm_modules(reordered).signature)
        self.config["pipeline_id"] = "changed"
        self.assertNotEqual(first.signature, validate_wm_modules(self.config).signature)
        self.assertEqual(first.pipeline_id, "wm_development_example")

    def test_loader_rejects_duplicate_keys_and_oversize(self):
        """설정 파서도 중복 키·비표준 숫자·과대 파일을 거부한다."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "plan.json"

            for payload in ('{"steps": [], "steps": []}', '{"steps": NaN}', ' ' * 65537):
                path.write_text(payload, encoding="utf-8")

                with self.assertRaises(ValueError):
                    load_wm_modules(path)

    def test_existing_execution_reused_without_run(self):
        """기존 입력 검사와 실행 관리자는 재사용하되 승인·실행 호출은 하지 않는다."""
        for module in wm_modules():
            with self.subTest(module=module.contract.module_id):
                tool = Mock()
                tool.registered_tool.return_value = SimpleNamespace(action_id=module.worker_stage, requires_approval=True)
                cli = Mock()
                cli.open_run.return_value = tool

                with patch("src.agents.wm_module_catalogue._cli", return_value=cli) as resolver:
                    actual, manager = open_wm_execution(module.contract.module_id, "1.0.0", "existing-run")

                resolver.assert_called_once_with(module.cli_id)
                self.assertIs(actual, tool)
                self.assertIs(manager, cli.executor.return_value)
                tool._check_arguments.assert_called_once_with(tool.arguments)
                cli.approve.assert_not_called()
                manager.run_next.assert_not_called()

    def test_invalid_execution_route_and_approval_boundary(self):
        """임의 import·작업 경로와 필수 승인이 빠진 실행기를 거부한다."""
        with self.assertRaises(ValueError):
            _cli("os.system")

        for identifier, version, name in (("unknown", "1.0.0", "run"),
                                           ("wm_dataset_check", "2.0.0", "run"),
                                           ("wm_dataset_check", "1.0.0", "../run")):
            with self.assertRaises(ValueError):
                open_wm_execution(identifier, version, name)

        for action, approval in (("validate_contract", False), ("run_eda", True)):
            cli = Mock()
            cli.open_run.return_value.registered_tool.return_value = SimpleNamespace(
                action_id=action, requires_approval=approval)

            with patch("src.agents.wm_module_catalogue._cli", return_value=cli), self.assertRaises(ValueError):
                open_wm_execution("wm_dataset_check", "1.0.0", "run")

            cli.executor.assert_not_called()

    def test_native_routes_match_registered_tools(self):
        """실제 기존 도구의 종류·승인 선언을 확인한다. 생성자·worker는 호출하지 않는다."""
        types = ("WmCanonicalTool", "WmConflictTool", "WmExclusionTool", "WmSplitTool",
                 "WmSplitCommitTool", "WmTransformTool", "WmCacheTool", "WmDatasetTool")

        for module, name in zip(wm_modules(), types):
            cli = _cli(module.cli_id)
            factory = getattr(cli, name)
            # registered_tool은 바인딩된 메서드와 승인 선언만 만든다.
            # 데이터·상태·DB를 읽는 실제 도구 생성자는 의도적으로 실행하지 않는다.
            registered = factory.__new__(factory).registered_tool()
            self.assertEqual(registered.action_id, module.worker_stage)
            self.assertIs(registered.requires_approval, True)


if __name__ == "__main__":
    unittest.main()
