"""공통 등록부와 에이전트 조립 계층의 의존성·검토 경계를 확인한다."""

from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from src.agents.module_catalogue import MODULE_ID, VERSION, inspect_module
from src.preprocessing.metadata_registry import build_metadata_registry


class MetadataRegistryTest(unittest.TestCase):
    def test_builtin_registry_without_agents(self):
        """새 프로세스에서 공통 등록부가 에이전트·도메인 모듈 없이 동작한다."""
        script = (
            "import sys; "
            "from src.preprocessing.metadata_registry import build_metadata_registry; "
            "registry = build_metadata_registry(); "
            "assert registry.resolve('tabular_quality', '1.0.0'); "
            # src.__init__의 기존 호환 namespace 생성은 허용하되 도메인 구현은 로딩하지 않는다.
            "assert not any(name.startswith(('src.agents', 'src.wafer_dl', 'src.sensor_ml.')) "
            "for name in sys.modules)"
        )
        root = Path(__file__).resolve().parents[2]
        result = subprocess.run([sys.executable, "-B", "-c", script], cwd=root,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_no_automatic_user_registration(self):
        """검토 계약을 주입하지 않으면 사용자 모듈을 자동 등록하지 않는다."""
        registry = build_metadata_registry()

        with self.assertRaises(ValueError):
            registry.resolve(MODULE_ID, VERSION)

    def test_reviewed_metadata_keeps_settings_checks(self):
        """검토된 계약을 전달해도 계산 코드는 로딩하지 않고 설정 검증만 수행한다."""
        contract = inspect_module(MODULE_ID, VERSION)

        with patch("src.agents.module_catalogue.inspect_module", side_effect=AssertionError("코어 역참조 금지")):
            registry = build_metadata_registry(reviewed_contract=contract)

        module = registry.resolve(MODULE_ID, VERSION)
        self.assertEqual(module.contract, contract)
        self.assertEqual(module.validate_settings(), {"include_complete": True})
        self.assertEqual(module.validate_settings({"include_complete": False}), {"include_complete": False})

        for settings in ({"include_complete": 1}, {"unknown": True}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                module.validate_settings(settings)

    def test_contract_schema_mismatches_rejected(self):
        """다른 ID·버전·설정 필드나 계약 아닌 객체를 조용히 등록하지 않는다."""
        contract = inspect_module(MODULE_ID, VERSION)

        for changed in (replace(contract, module_id="other_module"), replace(contract, version="2.0.0"),
                        replace(contract, settings_fields=("other_setting",))):
            with self.subTest(module=changed.module_id), self.assertRaises(ValueError):
                build_metadata_registry(reviewed_contract=changed)

        with self.assertRaises(TypeError):
            build_metadata_registry(reviewed_contract={"module_id": MODULE_ID})


if __name__ == "__main__":
    unittest.main()
