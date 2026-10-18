"""완료 분할에서 검사 작업만 준비하며 설정·부모 변경은 거부한다."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

from tests.agents import test_wm_transform as fixtures
from src.agents.wm_module_plan import load_wm_modules
from src.agents.wm_run_bindings import load_bindings
from src.agents.wm_run_creation import create_and_assign, preview_creation
from src.agents.wm_run_evidence import load_run_tool
from src.agents.worker_contract import WorkerLimits


ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "configs/agents/pipelines/wm_development_example.json"


class WmTransformCreationTest(unittest.TestCase):
    transport = fixtures.WmTransformTest.transport
    proposal_transport = fixtures.WmTransformTest.proposal_transport
    commit_transport = fixtures.WmTransformTest.commit_transport

    def setUp(self):
        """부모 분할까지만 합성 완료하고 새 검사 생성 경로를 격리한다."""
        fixtures.WmTransformTest.setUp(self)
        self.plan = load_wm_modules(PIPELINE)
        self.bindings = self.root / "transform_bindings.json"
        self.bindings.write_text(json.dumps({"schema_version": "wm_run_bindings_v1",
            "pipeline_signature": self.plan.signature,
            "runs": {"eda": "example", "proposal": "proposal", "split": "commit"}}), encoding="utf-8")
        self.config_path = self.root / "transform.json"
        self.config_path.write_bytes((ROOT / "configs/wm811k/transform.json").read_bytes())

        def directory(name, **kwargs):
            return self.root / "runs" / name

        def creation_root(*, create=False):
            directory = self.root / "creations"

            if create:
                directory.mkdir(exist_ok=True)

            return directory

        for target, replacement in (("src.agents.wm_transform_cli._directory", directory),
                                    ("src.agents.wm_run_creation.creation_root", creation_root)):
            mocking = patch(target, side_effect=replacement)
            mocking.start()
            self.addCleanup(mocking.stop)

    def preview(self, *, split_config=None, maximum=None):
        return preview_creation(PIPELINE, self.bindings, "transform", "new-transform", split_config,
            WorkerLimits(300, 2048, 2), maximum, transform_config=self.config_path)

    def test_creation_preserves_config_and_approval(self):
        """새 배정과 설정만 저장하고 실제 실행·승인 발급·기존 변경은 하지 않는다."""
        from src.agents.artifacts import _fingerprint

        paths = [path for path in self.root.rglob("*") if path.is_file()]
        before = {str(path): _fingerprint(path) for path in paths}
        preview = self.preview()
        self.assertEqual(before, {str(path): _fingerprint(path) for path in paths})
        self.assertEqual(preview.packet["max_output_bytes"], 256 * 1024 * 1024)

        with patch("src.agents.executor.RulesExecutor.run_next", side_effect=AssertionError("실행 금지")), \
                patch("src.agents.approval_store.ApprovalStore.record_confirmed_approval", side_effect=AssertionError("승인 발급 금지")), \
                patch("src.agents.wm_transform.run_check_worker", side_effect=AssertionError("worker 금지")):
            result = create_and_assign(preview, confirmed=True)

        self.assertEqual(result["status"], "created_and_assigned")
        self.assertFalse(result["worker_started"])
        self.assertFalse(result["execution_approval_issued"])
        self.assertEqual(load_bindings(result["bindings"], self.plan)["transform"], "new-transform")
        self.assertNotIn("transform", load_bindings(self.bindings, self.plan))
        saved_config = Path(result["bindings"]).parent / "transform_config.json"
        self.assertEqual(json.loads(saved_config.read_text(encoding="utf-8")), preview.packet["transform_config"])
        tool, _, _, _ = load_run_tool("wm_transform_check", "new-transform", require_complete=False)
        self.assertEqual(tool.states.load().status, "created")
        self.assertEqual(tool.states.load().current_stage, "preflight")
        self.assertEqual(tool.split_run, "commit")
        self.assertEqual(tool.evidence["transform_config"], preview.packet["transform_config"])
        self.assertTrue(tool.registered_tool().requires_approval)
        self.assertEqual(before, {str(path): _fingerprint(path) for path in paths})

    def test_config_and_parent_changes_rejected(self):
        """잘못된 설정·과대 출력·확인 후 설정/분할 변경은 생성 전에 거부한다."""
        for options in ({"split_config": self.config_path}, {"maximum": 1024 ** 3 + 1}):
            with self.assertRaises(ValueError):
                self.preview(**options)

        preview = self.preview()
        original = self.config_path.read_bytes()
        config = json.loads(original)
        config["target_size"] = [64, 64]
        self.config_path.write_text(json.dumps(config), encoding="utf-8")

        with patch("src.agents.wm_transform_cli.create") as create:
            with self.assertRaises(ValueError):
                create_and_assign(preview, confirmed=True)

        create.assert_not_called()
        self.config_path.write_bytes(original)
        # 파일 열거 순서가 아니라 실제 분할 결과의 완료 표식을 선택한다.
        marker = next(path for path in (self.split_tool.states.directory / "attempts").glob("*/AGENT_COMPLETE.json")
                      if (path.parent / "split").is_dir())
        marker.write_bytes(b"{}")

        with patch("src.agents.wm_transform_cli.create") as create:
            with self.assertRaises(ValueError):
                create_and_assign(preview, confirmed=True)

        create.assert_not_called()
        self.assertFalse((self.root / "runs/new-transform").exists())


if __name__ == "__main__":
    unittest.main()
