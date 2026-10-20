"""캐시·Dataset 조회는 전용 검증기를 사용하고 일반 분할 검증을 우회 확장하지 않는다."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src.ui.workflow_results import _result_references


class ResultRoutesTest(unittest.TestCase):
    def test_specialized_commit_and_domain_inspection_are_required(self):
        for module, target, folder, report in (
            ("wm_cache_prepare", "src.agents.wm_cache_evidence.committed_cache", "cache", "cache.md"),
            ("wm_dataset_check", "src.agents.wm_dataset_contract.committed_dataset", "dataset", "dataset_check.md"),
        ):
            with self.subTest(module=module):
                state = SimpleNamespace(status="completed", revision=6)
                tool, manager = Mock(), Mock()
                tool.states.load.return_value = state
                service = Mock(module_id=module)
                service._open.return_value = (Mock(), tool, manager)
                output = Path("synthetic-attempt")
                with patch(target, return_value=output) as committed, patch(
                        "src.ui.workflow_results._path", side_effect=lambda path, **kw: path):
                    result = _result_references(service, "synthetic")
                    committed.assert_called_once_with(tool, manager)
                    tool.inspect_attempt.assert_called_once_with(output)
                    self.assertEqual(result["items"][1]["label"], folder + "/" + report)
                    service.results.assert_not_called()

                with patch(target, side_effect=ValueError("commit rejected")):
                    with self.assertRaises(ValueError):
                        _result_references(service, "synthetic")

                with patch(target, return_value=output), patch(
                        "src.ui.workflow_results._path", side_effect=lambda path, **kw: path):
                    tool.inspect_attempt.side_effect = ValueError("domain rejected")
                    with self.assertRaises(ValueError):
                        _result_references(service, "synthetic")

    def test_other_modules_and_noncompleted_states_keep_service_route(self):
        for module, status in (("wm_canonical_eda", "completed"), ("wm_cache_prepare", "blocked_input")):
            service = Mock(module_id=module)
            tool = Mock()
            tool.states.load.return_value = SimpleNamespace(status=status)
            service._open.return_value = (Mock(), tool, Mock())
            self.assertEqual(_result_references(service, "synthetic"), service.results.return_value)
            service.results.assert_called_once_with("synthetic")
