"""AppContainer 판정·복사 범위를 합성 입력으로 검사한다. OS 프로필·ACL은 변경하지 않는다."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.agents.appcontainer_probe import DENIALS, REQUIRED, evaluate_child, probe_environment, runtime_files, summarize_components
from src.agents.appcontainer_native import SecurityCapabilities, StartupInfoEx
from src.agents.appcontainer_child import denied


class AppContainerProbeTest(unittest.TestCase):
    def test_experimental_312_requires_explicit_opt_in(self):
        from src.agents.appcontainer_probe import check_probe_runtime

        with patch("src.agents.appcontainer_probe.os.name", "nt"), patch("src.agents.appcontainer_probe.sys.version_info", (3, 12)):
            with self.assertRaises(ValueError):
                check_probe_runtime()
            check_probe_runtime(experimental_python312=True)

        with patch("src.agents.appcontainer_probe.os.name", "nt"), patch("src.agents.appcontainer_probe.sys.version_info", (3, 13)):
            with self.assertRaises(ValueError):
                check_probe_runtime(experimental_python312=True)

        with self.assertRaises(ValueError):
            check_probe_runtime(experimental_python312=1)

    def result(self):
        checks = {name: True for name in REQUIRED}
        checks.update({name: {"denied": True, "error_code": 10013 if "network" in name else 5} for name in DENIALS})

        return {"schema_version": "appcontainer_child_v1", "checks": checks, "real_data_read": False}

    def test_all_checks_required(self):
        """토큰·허용 접근·네이티브 라이브러리·그림 검사를 모두 요구한다."""
        self.assertTrue(evaluate_child(self.result()))

        for name in REQUIRED:
            value = self.result()
            value["checks"][name] = False
            self.assertFalse(evaluate_child(value), name)

    def test_component_status_is_separate(self):
        """런타임 통과와 로컬 통신 미확인을 분리하고 비정상 종료는 통과하지 않는다."""
        launch = {"status": "exited", "exit_code": 0}
        loopback = {"status": "review_required"}
        result = summarize_components(self.result(), launch, loopback)
        self.assertEqual(result["runtime"], "passed")
        self.assertEqual(result["file_access"], "passed")
        self.assertEqual(result["loopback"], "review_required")
        failed = summarize_components(self.result(), {"status": "exited", "exit_code": 1}, loopback)
        self.assertEqual(failed["runtime"], "review_required")
        malformed = self.result()
        malformed["checks"]["protected_read"] = []
        self.assertEqual(summarize_components(malformed, launch, loopback)["file_access"], "review_required")

    def test_timeout_and_missing_are_not_denial(self):
        """연결 시간 초과·파일 없음·연결 거부를 보안 차단 성공으로 인정하지 않는다."""
        for code in (2, 10060, 10061, None, True):
            value = self.result()
            value["checks"]["external_network"] = {"denied": True, "error_code": code}
            self.assertFalse(evaluate_child(value), code)

    def test_runtime_and_token_failures(self):
        """환경이 일반 프로세스이거나 검사 도중 예외가 있으면 통과하지 않는다."""
        value = self.result()
        value["runtime_error"] = "PROBE_RUNTIME_FAILED"
        self.assertFalse(evaluate_child(value))
        value = self.result()
        value["checks"]["appcontainer_token"] = False
        self.assertFalse(evaluate_child(value))

    def test_malformed_results(self):
        """누락·잘못된 타입·원본 접근 표시가 있는 결과를 거부한다."""
        for value in (None, [], {}, {**self.result(), "real_data_read": True}, {**self.result(), "checks": []}):
            self.assertFalse(evaluate_child(value))

        for name in DENIALS:
            value = deepcopy(self.result())
            del value["checks"][name]
            self.assertFalse(evaluate_child(value))

    def test_runtime_copy_allowlist(self):
        """프로젝트·임의 패키지·pycache·설치 스크립트는 런타임 복사에 포함하지 않는다."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = (
                "python.exe", "python310.dll", "Lib/os.py", "Lib/encodings/__init__.py",
                "Lib/__pycache__/os.pyc", "Lib/site-packages/numpy/__init__.py",
                "Lib/site-packages/cloudpickle/__init__.py",
                "Lib/site-packages/not_allowed/private.txt", "Lib/site-packages/private.pth",
                "Lib/site-packages/narwhals/__init__.py",
                "Lib/site-packages/narwhals-2.27.1.dist-info/METADATA",
                "Library/bin/synthetic.dll",
            )

            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"synthetic")

            before = {path.relative_to(root) for path in root.rglob("*")}
            selected = {path.relative_to(root).as_posix() for path in runtime_files(root)}
            self.assertEqual(selected, {
                "python.exe", "python310.dll", "Lib/os.py", "Lib/encodings/__init__.py",
                "Lib/site-packages/numpy/__init__.py", "Library/bin/synthetic.dll",
                "Lib/site-packages/cloudpickle/__init__.py",
                "Lib/site-packages/narwhals/__init__.py",
                "Lib/site-packages/narwhals-2.27.1.dist-info/METADATA",
            })
            self.assertEqual(before, {path.relative_to(root) for path in root.rglob("*")})

    def test_native_capabilities_default(self):
        """네이티브 구조는 기본 네트워크 capability가 없는 형태로 초기화된다."""
        capabilities = SecurityCapabilities()
        startup = StartupInfoEx()
        self.assertEqual(capabilities.count, 0)
        self.assertFalse(capabilities.capabilities)
        self.assertFalse(startup.startup.stdin)

    def test_environment_allowlist(self):
        """프로필 조회 변수는 보존하되 키·PYTHONPATH·부모 PATH는 상속하지 않는다."""
        parent = {
            "SystemRoot": "C:/Windows", "LOCALAPPDATA": "C:/synthetic/Local",
            "USERPROFILE": "C:/synthetic", "APPDATA": "C:/synthetic/Roaming",
            "OPENAI_API_KEY": "synthetic_secret", "PYTHONPATH": "private", "PATH": "private_bin",
        }
        runtime, output = Path("runtime"), Path("output")
        environment = probe_environment(runtime, output, parent)
        self.assertEqual(environment["LOCALAPPDATA"], parent["LOCALAPPDATA"])
        self.assertEqual(environment["USERPROFILE"], parent["USERPROFILE"])
        self.assertEqual(environment["PATH"], str(runtime))
        self.assertEqual(environment["TEMP"], str(output))
        self.assertNotIn("OPENAI_API_KEY", environment)
        self.assertNotIn("PYTHONPATH", environment)
        self.assertEqual(parent["PATH"], "private_bin")

    def test_missing_os_environment_rejected(self):
        """필수 OS 변수 누락을 프로세스 생성 전에 명확히 중단한다."""
        parent = {"SystemRoot": "C:/Windows", "LOCALAPPDATA": "C:/synthetic/Local", "USERPROFILE": "C:/synthetic"}

        for name in parent:
            changed = dict(parent)
            del changed[name]

            with self.assertRaises(ValueError):
                probe_environment(Path("runtime"), Path("output"), changed)

    def test_denial_diagnostic_distinguishes_timeout(self):
        """로컬 연결 시간 초과를 차단 성공이나 연결 성공으로 표시하지 않는다."""
        with patch("builtins.open", side_effect=TimeoutError()):
            timeout = denied(lambda: open("synthetic"))

        self.assertFalse(timeout["denied"])
        self.assertEqual(timeout["outcome"], "timeout")

        with patch("builtins.open", side_effect=PermissionError(13, "synthetic")):
            permission = denied(lambda: open("synthetic"))

        self.assertTrue(permission["denied"])
        self.assertEqual(permission["outcome"], "access_denied")
        self.assertNotIn("synthetic", str(permission))


if __name__ == "__main__":
    unittest.main()
