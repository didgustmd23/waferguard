"""고정 모듈 계약·조회·설정·결과 검증의 정상/예외 입력을 확인한다."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.preprocessing.builtin_modules import build_builtin_registry
from src.preprocessing.contracts import ColumnRoles, TabularInput
from src.preprocessing.module_contracts import ModulePort
from src.preprocessing.module_registry import ModuleRegistry
from src.preprocessing.quality import QualityReport, QualityRules


class ModuleRegistryTest(unittest.TestCase):
    def setUp(self):
        self.registry = build_builtin_registry()
        self.frame = pd.DataFrame({
            "x": [1.0, 2.0, np.nan, 4.0], "y": [2.0, 4.0, 6.0, 8.0],
            "category": ["a", "b", "a", None], "label": [0, 1, 0, 1],
        })
        self.data = TabularInput(
            self.frame, ColumnRoles(numeric=("x", "y"), categorical=("category",), target="label"),
        )

    def test_catalog_creation_does_not_calculate(self):
        """등록부 생성만으로 데이터 계산을 시작하지 않는다."""
        with patch("src.preprocessing.eda.analyze_missing", side_effect=AssertionError("등록 중 계산")):
            registry = build_builtin_registry()

        self.assertEqual({item.module_id for item in registry.contracts()}, {
            "eda_missing", "eda_numeric", "eda_categories", "eda_target",
            "eda_correlations", "tabular_quality",
        })
        self.assertTrue(all(item.network == "deny_all" for item in registry.contracts()))

    def test_exact_lookup(self):
        """미등록 ID·버전·import 경로와 자동 최신 버전 선택을 거부한다."""
        for name, version in (
            ("unknown", "1.0.0"), ("eda_missing", "2.0.0"),
            ("src.preprocessing.eda", "1.0.0"), ("eda_missing", "latest"), ("eda_missing", None),
        ):
            with self.subTest(name=name, version=version), self.assertRaises(ValueError):
                self.registry.resolve(name, version)

    def test_duplicate_and_sealed_registration(self):
        """같은 ID·버전의 덮어쓰기와 봉인된 목록 변경을 거부한다."""
        module = self.registry.resolve("eda_missing", "1.0.0")
        registry = ModuleRegistry()
        registry.register(module)

        with self.assertRaises(ValueError):
            registry.register(module)

        with self.assertRaises(ValueError):
            self.registry.register(module)

    def test_required_callbacks_and_default_settings(self):
        """필수 함수 누락과 기본 설정 검증 실패를 등록 시 거부한다."""
        module = self.registry.resolve("eda_missing", "1.0.0")

        for field in ("calculate", "settings_validator", "result_validator"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                replace(module, **{field: None})

        def reject(settings):
            raise ValueError("잘못된 기본 설정")

        with self.assertRaises(ValueError):
            ModuleRegistry().register(replace(module, settings_validator=reject))

    def test_invalid_contract_and_security_policy(self):
        """식별값·버전·포트·보안 정책·기준 산출 구간의 잘못된 조합을 거부한다."""
        contract = self.registry.resolve("eda_missing", "1.0.0").contract
        invalid = (
            {"module_id": "../code"}, {"version": "01.0.0"}, {"version": "1.0"},
            {"kind": "training"}, {"network": "allow"}, {"approval": "none"},
            {"original_policy": "overwrite"}, {"result_validator_id": ""},
            {"inputs": []}, {"outputs": ()}, {"settings_fields": ("a", "a")},
            {"criteria_policy": "derive_on_train", "derivation_partition": "validation"},
            {"inputs": (ModulePort("data", "tabular_data"), ModulePort("data", "tabular_data"))},
        )

        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(contract, **changes)

        with self.assertRaises(ValueError):
            ModulePort("data", "tabular_data", contract_version=True)

    def test_strict_settings_and_copy(self):
        """미지원 키·잘못된 기준을 거부하고 사용자 설정은 변경하지 않는다."""
        module = self.registry.resolve("tabular_quality", "1.0.0")
        settings = {"allowed_labels": [0, 1], "missing_ratio_threshold": 0.5}
        normalized = module.validate_settings(settings)

        self.assertIsInstance(normalized["rules"], QualityRules)
        self.assertEqual(normalized["rules"].allowed_labels, (0, 1))
        self.assertEqual(settings["allowed_labels"], [0, 1])

        for invalid in (
            {"command": "run"}, {"missing_ratio_threshold": True},
            {"missing_ratio_threshold": float("nan")}, {"high_missing": "delete"},
            {"allowed_labels": [1, 1]}, [], {1: "x"},
        ):
            with self.subTest(settings=invalid), self.assertRaises(ValueError):
                module.validate_settings(invalid)

        with self.assertRaises(ValueError):
            self.registry.resolve("eda_missing", "1.0.0").validate_settings({"path": "input.csv"})

    def test_builtin_results_preserve_source(self):
        """모든 내장 계산이 결과 검증을 통과하고 원본을 보존한다."""
        before = self.frame.copy(deep=True)

        for contract in self.registry.contracts():
            module = self.registry.resolve(contract.module_id, contract.version)
            result = module.calculate(self.data, **module.validate_settings())
            self.assertTrue(module.result_validator(result), contract.module_id)

        pd.testing.assert_frame_equal(before, self.frame)

    def test_required_structure_and_quality_diagnosis(self):
        """EDA 필수 검사는 생략할 수 없으며 품질 모듈은 같은 오류를 진단한다."""
        bad = TabularInput(self.frame.drop(columns="x"), self.data.roles)

        with self.assertRaises(ValueError):
            self.registry.resolve("eda_missing", "1.0.0").calculate(bad)

        quality = self.registry.resolve("tabular_quality", "1.0.0")
        report = quality.calculate(bad, **quality.validate_settings())
        self.assertTrue(quality.result_validator(report))
        self.assertEqual(report.status, "blocked_input")

    def test_invalid_results(self):
        """잘못된 형태·범위·상태는 검증에 실패한다."""
        missing = self.registry.resolve("eda_missing", "1.0.0")
        result = missing.calculate(self.data)
        result.loc["x", "missing_ratio"] = 2.0
        self.assertFalse(missing.result_validator(result))
        self.assertFalse(missing.result_validator({"status": "passed"}))

        quality = self.registry.resolve("tabular_quality", "1.0.0")
        self.assertFalse(quality.result_validator(QualityReport("passed", (None,))))
        self.assertFalse(quality.result_validator(QualityReport("review_required", ())))

        correlations = self.registry.resolve("eda_correlations", "1.0.0")
        matrices = correlations.calculate(self.data)
        matrices["pearson"].iloc[0, 1] = 2.0
        self.assertFalse(correlations.result_validator(matrices))


if __name__ == "__main__":
    unittest.main()
