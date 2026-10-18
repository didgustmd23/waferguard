"""사용자 파이프라인의 정적 계약·순서 검증. 합성 설정만 사용하며 모듈을 실행하지 않는다."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.preprocessing.builtin_modules import build_builtin_registry
from src.preprocessing.module_contracts import ModulePort
from src.preprocessing.module_registry import ModuleRegistry
from src.preprocessing.pipeline import load_pipeline, validate_pipeline


def example_config():
    """실제 데이터 경로·컬럼·통계가 없는 최소 2단계 설정."""
    return {
        "schema_version": "module_pipeline_v1", "pipeline_id": "example",
        "input": {"contract_id": "tabular_data", "contract_version": 1, "partition": "unsplit"},
        "steps": [
            {"step_id": "quality", "module_id": "tabular_quality", "version": "1.0.0", "inputs": {"data": "source.data"}},
            {"step_id": "missing", "module_id": "eda_missing", "version": "1.0.0", "inputs": {"data": "source.data"}, "depends_on": ["quality"]},
        ],
    }


class ModulePipelineTest(unittest.TestCase):
    def setUp(self):
        self.config = example_config()

    def test_plan_creation_does_not_execute(self):
        """계획 생성은 모듈 계산·데이터 읽기를 호출하지 않는다."""
        with patch("src.preprocessing.eda.analyze_missing", side_effect=AssertionError("계획 중 계산")):
            plan = validate_pipeline(self.config)

        self.assertEqual(tuple(step.step_id for step in plan.steps), ("quality", "missing"))
        self.assertEqual(plan.steps[1].depends_on, ("quality",))
        self.assertEqual(plan.steps[0].module_id, "tabular_quality")
        self.assertTrue(plan.to_dict()["stop_on_review_or_block"])

    def test_signature_and_source_copy(self):
        """설정 사후 변경이 기존 계획에 영향을 주지 않고 유효 설정 변경은 식별값을 바꾼다."""
        before = deepcopy(self.config)
        plan = validate_pipeline(self.config)
        signature = plan.signature
        self.assertEqual(self.config, before)

        self.config["steps"][0]["settings"] = {"missing_ratio_threshold": 0.4}
        self.assertNotEqual(validate_pipeline(self.config).signature, signature)
        self.assertEqual(plan.signature, signature)

        reordered = dict(reversed(list(before.items())))
        self.assertEqual(validate_pipeline(reordered).signature, signature)

    def test_schema_fields_and_limits(self):
        """알 수 없는 실행 옵션·입력 계약·구간·빈/과다 단계는 차단한다."""
        variants = []

        for key, value in (
            ("schema_version", "v2"), ("pipeline_id", "../path"),
            ("command", "run"), ("stop_on_review_or_block", False), ("steps", []),
        ):
            config = deepcopy(self.config)
            config[key] = value
            variants.append(config)

        for key, value in (("contract_id", "image"), ("contract_version", True), ("partition", "future")):
            config = deepcopy(self.config)
            config["input"][key] = value
            variants.append(config)

        config = deepcopy(self.config)
        config["steps"] *= 17
        variants.append(config)

        for config in variants:
            with self.subTest(config=config), self.assertRaises(ValueError):
                validate_pipeline(config)

    def test_ids_versions_and_settings(self):
        """예약 단계명·중복 단계·미지원 모듈·설정은 실행 전에 거부한다."""
        for changes in (
            {"step_id": "source"}, {"step_id": "quality"}, {"module_id": "unknown"},
            {"version": "latest"}, {"settings": {"command": "run"}}, {"enabled": False},
        ):
            config = deepcopy(self.config)
            config["steps"][1].update(changes)

            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_pipeline(config)

    def test_dependency_order_and_cycles(self):
        """자기 참조·뒤 단계·순환·중복 의존성은 허용하지 않는다."""
        for dependencies in (["missing"], ["unknown"], ["quality", "quality"], "quality"):
            config = deepcopy(self.config)
            config["steps"][1]["depends_on"] = dependencies

            with self.subTest(dependencies=dependencies), self.assertRaises(ValueError):
                validate_pipeline(config)

        config = deepcopy(self.config)
        config["steps"][0]["depends_on"] = ["missing"]

        with self.assertRaises(ValueError):
            validate_pipeline(config)

    def test_input_ports_and_contract_mismatch(self):
        """필수 포트·이전 출력 존재·계약 일치를 확인한다."""
        for inputs in ({}, {"unknown": "source.data"}, {"data": "quality.report"}, {"data": "later.result"}):
            config = deepcopy(self.config)
            config["steps"][1]["inputs"] = inputs

            with self.subTest(inputs=inputs), self.assertRaises(ValueError):
                validate_pipeline(config)

    def test_inferred_dependency_from_output_binding(self):
        """앞 단계 출력에 연결하면 명시 목록이 없어도 실행 의존성이 생긴다."""
        builtin = build_builtin_registry()
        quality = builtin.resolve("tabular_quality", "1.0.0")
        missing = builtin.resolve("eda_missing", "1.0.0")
        report_reader = replace(missing, contract=replace(
            missing.contract, module_id="report_reader",
            inputs=(ModulePort("data", "quality_report"),),
        ))
        registry = ModuleRegistry()
        registry.register(quality)
        registry.register(report_reader)
        registry.seal()
        config = deepcopy(self.config)
        config["steps"][1].update(module_id="report_reader", inputs={"data": "quality.report"}, depends_on=[])

        # 계산 함수는 호출하지 않는다. 이 합성 등록은 연결 검증만 확인한다.
        plan = validate_pipeline(config, registry)
        self.assertEqual(plan.steps[1].depends_on, ("quality",))

    def test_runtime_checks_and_train_only_derivation(self):
        """미지원 선행 증거와 평가 구간 기준 산출을 계획 시 차단한다."""
        base = build_builtin_registry().resolve("eda_missing", "1.0.0")

        for changes in (
            {"preconditions": ("unknown_evidence",)},
            {"kind": "transformation", "criteria_policy": "derive_on_train", "derivation_partition": "train"},
        ):
            registry = ModuleRegistry()
            registry.register(replace(base, contract=replace(base.contract, **changes)))
            config = deepcopy(self.config)
            config["steps"] = [config["steps"][1]]
            config["steps"][0]["depends_on"] = []
            config["input"]["partition"] = "validation"

            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_pipeline(config, registry)

    def test_json_rejects_duplicates_nonfinite_and_oversize(self):
        """파일 파서는 중복 키·NaN·과대 입력을 제한한다."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pipeline.json"

            for payload in ('{"x":1,"x":2}', '{"x":NaN}', " " * (64 * 1024 + 1)):
                path.write_text(payload, encoding="utf-8")

                with self.subTest(payload_size=len(payload)), self.assertRaises(ValueError):
                    load_pipeline(path)

            path.write_text(json.dumps(self.config), encoding="utf-8")
            self.assertEqual(load_pipeline(path).signature, validate_pipeline(self.config).signature)

    def test_shipped_example(self):
        """배포 예제는 실제 데이터 없이 정적으로 검증할 수 있다."""
        project = Path(__file__).resolve().parents[2]
        plan = load_pipeline(project / "configs/agents/pipelines/tabular_eda_example.json")
        self.assertEqual(len(plan.steps), 3)


if __name__ == "__main__":
    unittest.main()
