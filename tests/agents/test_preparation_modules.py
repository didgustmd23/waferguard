"""변환 모듈 등록·계획을 합성 데이터로 확인한다. 실제 worker는 실행하지 않는다."""

from copy import deepcopy
from pathlib import Path
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.agents.preparation_module_plan import validate_preparation_modules
from src.preprocessing.contracts import ColumnRoles, TabularInput
from src.preprocessing.preparation_modules import build_preparation_registry


class PreparationModulesTest(unittest.TestCase):
    def setUp(self):
        # 데이터가 아니라 검토된 설정 예시만 읽는다.
        root = Path(__file__).resolve().parents[2]
        self.config = json.loads((root / "configs/agents/pipelines/tabular_preparation_example.json").read_text(encoding="utf-8"))
        self.registry = build_preparation_registry()

    def test_plan_does_not_compute(self):
        """계획·등록부 생성은 전처리를 실행하지 않고 두 새 승인 경계를 보존한다."""
        with patch("src.preprocessing.preparation_modules.build_preparation", side_effect=AssertionError("계획 중 계산")):
            build_preparation_registry()
            plan = validate_preparation_modules(self.config).to_dict()

        self.assertFalse(plan["automatic_execution"])
        self.assertFalse(plan["test_allowed"])
        self.assertEqual([item["stage"] for item in plan["steps"]], ["derive_preparation", "apply_preparation"])
        self.assertTrue(all(item["approval"] == "new_explicit_approval" for item in plan["steps"]))

    def test_core_reuse_without_refit(self):
        """정확히 50%인 특징 유지·중앙값 재사용·상수 제거·행 순서 보존을 확인한다."""
        roles = ColumnRoles(numeric=("x", "constant"), target="label")
        frame = pd.DataFrame({"x": [1.0, np.nan, 3.0, np.nan], "constant": [7.0] * 4, "label": [0, 1, 0, 1]})
        before = frame.copy(deep=True)
        propose = self.registry.resolve("tabular_prepare_propose", "1.0.0")
        candidate = propose.calculate(TabularInput(frame, roles), **propose.validate_settings(self.config["steps"][0]["settings"]))
        self.assertTrue(propose.result_validator(candidate))
        validation = TabularInput(pd.DataFrame({"x": [np.nan, 1000.0], "constant": [8.0, 9.0], "label": [1, 0]}, index=[10, 20]), roles)
        apply = self.registry.resolve("tabular_prepare_apply", "1.0.0")

        with patch.object(candidate, "fit", side_effect=AssertionError("적용 중 재산출")):
            result = apply.calculate(validation, candidate)

        self.assertTrue(apply.result_validator(result))
        self.assertEqual(result.iloc[:, 0].tolist(), [2.0, 1000.0])
        self.assertEqual(result.index.tolist(), [10, 20])
        self.assertEqual(result.shape[1], 1)
        pd.testing.assert_frame_equal(frame, before)

    def test_invalid_plans(self):
        """순서·버전·승인 경계·규칙 변경·임의 실행 경로를 거부한다."""
        variants = []
        changed = deepcopy(self.config)
        changed["steps"].reverse()
        variants.append(changed)

        for key, value in (("approval", "automatic"), ("version", "latest"), ("command", "run")):
            changed = deepcopy(self.config)
            changed["steps"][0][key] = value
            variants.append(changed)

        changed = deepcopy(self.config)
        changed["steps"][1]["settings"] = {"numeric_imputation": "mean"}
        variants.append(changed)
        changed = deepcopy(self.config)
        changed["test"] = "data/test.csv"
        variants.append(changed)

        for config in variants:
            with self.subTest(config=config), self.assertRaises(ValueError):
                validate_preparation_modules(config)

    def test_settings_and_signature(self):
        """규칙 변경을 식별하고 계획 생성 후 원본 설정 변경에 영향받지 않는다."""
        plan = validate_preparation_modules(self.config)
        signature = plan.signature
        self.config["steps"][0]["settings"]["numeric_imputation"] = "mean"
        self.assertNotEqual(validate_preparation_modules(self.config).signature, signature)
        self.assertEqual(plan.signature, signature)
        self.config["steps"][0]["settings"]["missing_ratio_threshold"] = 2

        with self.assertRaises(ValueError):
            validate_preparation_modules(self.config)

    def test_sealed_registry_and_invalid_results(self):
        """등록부 변경과 미산출 상태·무한대 출력은 거부한다."""
        propose = self.registry.resolve("tabular_prepare_propose", "1.0.0")

        with self.assertRaises(ValueError):
            self.registry.register(propose)

        self.assertFalse(propose.result_validator(object()))
        apply = self.registry.resolve("tabular_prepare_apply", "1.0.0")
        self.assertFalse(apply.result_validator(pd.DataFrame({"x": [np.inf]})))


if __name__ == "__main__":
    unittest.main()
