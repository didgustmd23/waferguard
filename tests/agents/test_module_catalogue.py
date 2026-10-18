"""사용자 모듈의 명세·변조 차단·합성 계산·실행 경계를 검증한다."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.agents import module_catalogue as catalogue
from src.agents.module_pipeline_contract import bind_pipeline
from src.preprocessing.contracts import ColumnRoles, TabularInput
from src.preprocessing.pipeline import validate_pipeline


class ModuleCatalogueTest(unittest.TestCase):
    def setUp(self):
        self.project = Path(__file__).resolve().parents[2]

    def copied_project(self):
        """변조 검사는 실제 프로젝트 대신 임시 복사본에서 수행한다."""
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)

        for relative in catalogue.SOURCE_FILES + (catalogue.DESCRIPTOR,):
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.project / relative, target)

        return root

    def test_inspect_without_import(self):
        """명세 검사는 새 프로세스에서도 확장 Python을 import하지 않는다."""
        code = (
            "import sys; from src.agents.module_catalogue import inspect_module; "
            "name = 'src.preprocessing.extensions.missing_row_histogram'; "
            "assert name not in sys.modules; inspect_module('user_missing_row_histogram', '1.0.0'); "
            "assert name not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-X", "utf8", "-c", code], cwd=self.project, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_exact_id_and_version(self):
        """미등록 ID·버전·함수 경로·latest는 거부한다."""
        for name, version in (("unknown", "1.0.0"), (catalogue.MODULE_ID, "latest"), ("src.custom.function", "1.0.0")):
            with self.subTest(name=name, version=version), self.assertRaises(ValueError):
                catalogue.inspect_module(name, version)

    def test_source_and_dependency_changes(self):
        """모듈과 선언한 의존 소스의 변경을 검출한다."""
        for relative in ("src/preprocessing/extensions/missing_row_histogram.py", "src/preprocessing/contracts.py"):
            root = self.copied_project()
            path = root / relative
            path.write_bytes(path.read_bytes() + b"\n# changed\n")

            with self.assertRaises(ValueError):
                catalogue.inspect_module(catalogue.MODULE_ID, catalogue.VERSION, project_root=root)

    def test_descriptor_cannot_approve_itself(self):
        """명세를 바꾸고 소스 해시를 함께 바꿔도 고정 목록이 다르면 거부한다."""
        root = self.copied_project()
        path = root / catalogue.DESCRIPTOR
        value = json.loads(path.read_text(encoding="utf-8"))
        value["source_files"]["src/preprocessing/contracts.py"] = "0" * 64
        path.write_text(json.dumps(value), encoding="utf-8")

        with self.assertRaises(ValueError):
            catalogue.inspect_module(catalogue.MODULE_ID, catalogue.VERSION, project_root=root)

    def test_line_endings(self):
        """CRLF 체크아웃만으로는 검토 내용이 바뀐 것으로 처리하지 않는다."""
        root = self.copied_project()

        for relative in catalogue.SOURCE_FILES + (catalogue.DESCRIPTOR,):
            path = root / relative
            path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))

        self.assertEqual(catalogue.inspect_module(catalogue.MODULE_ID, catalogue.VERSION, project_root=root).module_id, catalogue.MODULE_ID)

    def test_contract_and_scope(self):
        """새 명세를 검토 목록에서 선택했어도 권한·실행 범위 위반은 거부한다."""
        original = json.loads((self.project / catalogue.DESCRIPTOR).read_text(encoding="utf-8"))
        variants = []

        for key, value in (("network", "allow_all"), ("original_policy", "overwrite"), ("approval", "none")):
            changed = deepcopy(original)
            changed["contract"][key] = value
            variants.append(changed)

        changed = deepcopy(original)
        changed["execution_scope"] = "production"
        variants.append(changed)
        changed = deepcopy(original)
        changed["source_files"]["../outside.py"] = "0" * 64
        variants.append(changed)

        for value in variants:
            root = self.copied_project()
            path = root / catalogue.DESCRIPTOR
            path.write_text(json.dumps(value), encoding="utf-8")

            with patch.object(catalogue, "DESCRIPTOR_SHA256", catalogue._source_hash(path)), self.assertRaises(ValueError):
                catalogue.inspect_module(catalogue.MODULE_ID, catalogue.VERSION, project_root=root)

    def test_synthetic_calculation(self):
        """특징만 집계하고 원본 보존·설정·결과 형식을 확인한다."""
        registry = catalogue.build_reviewed_registry()
        module = registry.resolve(catalogue.MODULE_ID, catalogue.VERSION)
        frame = pd.DataFrame({"x": [1.0, np.nan, np.nan], "y": [1.0, 2.0, np.nan],
                              "category": ["a", None, "b"], "label": [None, 1, 0]})
        before = frame.copy(deep=True)
        data = TabularInput(frame, ColumnRoles(numeric=("x", "y"), categorical=("category",), target="label"))
        result = module.calculate(data, **module.validate_settings())
        self.assertEqual(result.to_dict("list"), {"missing_feature_count": [0, 2], "row_count": [1, 2]})
        self.assertTrue(module.result_validator(result))
        filtered = module.calculate(data, **module.validate_settings({"include_complete": False}))
        self.assertEqual(filtered["missing_feature_count"].tolist(), [2])
        pd.testing.assert_frame_equal(frame, before)

        for settings in ({"include_complete": 1}, {"command": "run"}):
            with self.assertRaises(ValueError):
                module.validate_settings(settings)

        self.assertFalse(module.result_validator(pd.DataFrame({"missing_feature_count": [-1], "row_count": [1]})))

        with self.assertRaises(ValueError):
            registry.register(module)

    def test_static_plan_and_native_denial(self):
        """검토 등록부로 정적 계획만 허용하고 기존 네이티브 실행은 거부한다."""
        config = json.loads((self.project / "configs/agents/pipelines/user_module_example.json").read_text(encoding="utf-8"))
        plan = validate_pipeline(config, catalogue.build_reviewed_registry())
        self.assertEqual(plan.steps[-1].module_id, catalogue.MODULE_ID)

        with self.assertRaises(ValueError):
            bind_pipeline(config, None)


if __name__ == "__main__":
    unittest.main()
