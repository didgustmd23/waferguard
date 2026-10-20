"""계산용 환경 모의와 운영 사전 검사의 미지원 환경 차단을 구분한다."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src.agents import isolated_eda_cli, snapshot_eda_cli, snapshot_audit_cli
from src.agents import module_pipeline_cli, reviewed_module_cli
from tests.agents.runtime_fixtures import run_supported_preflight


class LegacyCliRuntimeTest(unittest.TestCase):
    def test_original_preflights_keep_their_platform_version_and_argument_gates(self):
        modules = (isolated_eda_cli, snapshot_eda_cli, snapshot_audit_cli,
                   module_pipeline_cli, reviewed_module_cli)
        cases = (
            ("nt", (3, 10, 0), {}, True),
            ("nt", (3, 12, 15), {}, False),
            ("nt", (3, 11, 0), {}, False),
            ("nt", (3, 13, 0), {}, False),
            ("posix", (3, 10, 0), {}, False),
            ("nt", (3, 10, 0), {"unexpected": True}, False),
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            for cli in modules:
                for index, (system, version, arguments, supported) in enumerate(cases):
                    output = root / cli.__name__ / str(index)
                    output.mkdir(parents=True)

                    with self.subTest(cli=cli.__name__, system=system, version=version, arguments=arguments), patch.object(
                        cli, "os", SimpleNamespace(name=system)
                    ), patch.object(cli, "sys", SimpleNamespace(version_info=version)):
                        result = cli._preflight(output, arguments)
                        self.assertEqual(result.status, "passed" if supported else "blocked_input")
                        self.assertEqual((output / "preflight.json").exists(), supported)

                        if supported:
                            self.assertTrue(cli._validate_preflight(output, arguments))

    def test_fixture_restores_cli_references_on_success_and_exception(self):
        cli = snapshot_eda_cli
        original_os, original_sys = cli.os, cli.sys
        original_version = original_sys.version_info

        for failure in (False, True):
            with self.subTest(failure=failure):
                manager = Mock()
                manager.states.load.return_value = SimpleNamespace(current_stage="preflight")

                def run():
                    self.assertEqual(cli.os.name, "nt")
                    self.assertEqual(cli.sys.version_info, (3, 10, 0))
                    self.assertIs(original_sys.version_info, original_version)

                    if failure:
                        raise RuntimeError("synthetic_failure")

                    return "synthetic_result"

                manager.run_next.side_effect = run

                if failure:
                    with self.assertRaisesRegex(RuntimeError, "synthetic_failure"):
                        run_supported_preflight(manager, cli)
                else:
                    self.assertEqual(run_supported_preflight(manager, cli), "synthetic_result")

                self.assertIs(cli.os, original_os)
                self.assertIs(cli.sys, original_sys)
                manager.run_next.assert_called_once_with()

    def test_fixture_cannot_be_used_for_worker_stage(self):
        manager = Mock()
        manager.states.load.return_value = SimpleNamespace(current_stage="run_eda")

        with self.assertRaises(AssertionError):
            run_supported_preflight(manager, snapshot_eda_cli)

        manager.run_next.assert_not_called()
