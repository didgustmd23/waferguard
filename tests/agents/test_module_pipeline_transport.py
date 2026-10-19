"""실제 AppContainer·ACL·worker 없이 운송 준비 실패 진단과 정리를 검사한다."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src.agents import module_pipeline_transport as transport
from src.agents.writable_directory_acl import DirectoryAclError


class PipelineTransportDiagnosticTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

        for name in ("runtime", "input", "tools", "output", "scratch"):
            (self.root / name).mkdir()

        (self.root / "output/result").mkdir()
        (self.root / "scratch/task").mkdir()
        self.request = SimpleNamespace(
            task_id="a" * 32, request_id="synthetic_request",
            output_directory=str(self.root / "output/result"),
            scratch_directory=str(self.root / "scratch/task"), limits=Mock(),
        )
        self.container = Mock()
        self.container.create.return_value = "synthetic_sid"
        self.container.close.return_value = True
        self.container.launch.return_value = {"status": "exited", "exit_code": 0}
        self.factory = self.replace("AppContainer", return_value=self.container)
        self.inventory = self.replace("runtime_files", return_value=[])
        self.manifest = self.replace("code_manifest", return_value={"entry.py": "fixed"})
        self.acl = self.replace("_acl")
        self.writable_acl = self.replace("prepare_writable_directory")
        self.replace("WORKER_FILES", new=())
        self.replace("ENTRY", new="entry.py")
        self.replace("_fingerprint", return_value={"sha256": "fixed"})
        self.replace("probe_environment", return_value={})
        patcher = patch.object(transport.shutil, "copyfile")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.cancel = Mock()
        self.cancel.is_set.return_value = False

    def replace(self, name, **kwargs):
        patcher = patch.object(transport, name, **kwargs)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def run_transport(self):
        return transport.run_pipeline_worker(self.request, self.root, self.cancel)

    def diagnostic(self):
        return json.loads((self.root / "transport.json").read_text(encoding="utf-8"))

    def test_inventory_failure_has_no_private_message(self):
        self.inventory.side_effect = ValueError("synthetic_private_path")
        with self.assertRaises(ValueError):
            self.run_transport()
        result = self.diagnostic()
        self.assertEqual(result["diagnostic"]["stage"], "RUNTIME_INVENTORY")
        self.assertNotIn("synthetic_private_path", json.dumps(result))
        self.container.launch.assert_not_called()
        self.container.close.assert_called_once()

    def test_container_failure_captures_numeric_error(self):
        self.container.create.side_effect = OSError(5, "synthetic_private_detail")
        with self.assertRaises(OSError):
            self.run_transport()
        result = self.diagnostic()
        self.assertEqual(result["diagnostic"]["stage"], "CONTAINER_CREATE")
        self.assertEqual(result["diagnostic"]["errno"], 5)
        self.assertNotIn("synthetic_private_detail", json.dumps(result))
        self.acl.assert_not_called()

    def test_acl_failure_identifies_fixed_section(self):
        self.acl.side_effect = PermissionError(13, "synthetic_private_path")
        with self.assertRaises(PermissionError):
            self.run_transport()
        self.assertEqual(self.diagnostic()["diagnostic"]["stage"], "ACL_RUNTIME")
        self.container.launch.assert_not_called()

    def test_initialization_failure_still_leaves_diagnostic(self):
        self.factory.side_effect = OSError(5, "synthetic_private_detail")
        with self.assertRaises(OSError):
            self.run_transport()
        self.assertEqual(self.diagnostic()["diagnostic"]["stage"], "CONTAINER_INITIALIZE")
        self.assertTrue(self.diagnostic()["profile_cleanup"])

    def test_writable_acl_failure_stops_before_launch(self):
        self.writable_acl.side_effect = DirectoryAclError("LOW_INTEGRITY", 5)

        with self.assertRaises(DirectoryAclError):
            self.run_transport()

        result = self.diagnostic()
        self.assertEqual(result["diagnostic"]["stage"], "ACL_OUTPUT")
        self.assertEqual(result["diagnostic"]["acl_operation"], "LOW_INTEGRITY")
        self.assertNotIn("launch", result)
        self.container.launch.assert_not_called()
        self.container.close.assert_called_once()

    def test_only_output_and_scratch_receive_writable_preparation(self):
        self.run_transport()
        self.assertEqual(self.writable_acl.call_count, 2)
        self.assertEqual(
            [call.args[1] for call in self.writable_acl.call_args_list],
            [self.root / "output/result", self.root / "scratch/task"],
        )
        self.assertTrue(all(call.args[2] == "RX" for call in self.acl.call_args_list))

    def test_cleanup_failure_preserves_original_diagnostic(self):
        self.inventory.side_effect = ValueError("synthetic")
        self.container.close.side_effect = OSError(5, "synthetic")
        with self.assertRaises(ValueError):
            self.run_transport()
        result = self.diagnostic()
        self.assertEqual(result["diagnostic"]["stage"], "RUNTIME_INVENTORY")
        self.assertEqual(result["cleanup_diagnostic"]["stage"], "CONTAINER_CLEANUP")
        self.assertFalse(result["profile_cleanup"])

    def test_success_and_cleanup_failure_remain_distinct(self):
        result = self.run_transport()
        self.assertNotIn("diagnostic", result)
        self.assertTrue(result["profile_cleanup"])
        self.container.close.side_effect = OSError(5, "synthetic")
        result = self.run_transport()
        self.assertFalse(result["profile_cleanup"])
        self.assertIn("cleanup_diagnostic", result)

    def test_code_change_stops_before_launch(self):
        self.manifest.side_effect = [{"entry.py": "fixed"}, {"entry.py": "changed"}]
        with self.assertRaises(ValueError):
            self.run_transport()
        self.assertEqual(self.diagnostic()["diagnostic"]["stage"], "CODE_RECHECK")
        self.container.launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
