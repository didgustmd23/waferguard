"""분할 확정 준비만 생성하고 검토·확정 실행 승인을 보존하는지 합성 입력으로 확인한다."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

from tests.agents import test_wm_split_commit as fixtures
from src.agents.wm_module_plan import load_wm_modules
from src.agents.wm_run_bindings import load_bindings
from src.agents.wm_run_creation import create_and_assign, preview_creation
from src.agents.wm_run_evidence import load_run_tool
from src.agents.worker_contract import WorkerLimits


ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "configs/agents/pipelines/wm_development_example.json"


class WmCommitCreationTest(unittest.TestCase):
    transport = fixtures.WmSplitCommitTest.transport
    proposal_transport = fixtures.WmSplitCommitTest.proposal_transport

    def setUp(self):
        fixtures.WmSplitCommitTest.setUp(self)
        self.plan = load_wm_modules(PIPELINE)
        self.bindings = self.root / "commit_bindings.json"
        self.bindings.write_text(json.dumps({"schema_version": "wm_run_bindings_v1",
                "pipeline_signature": self.plan.signature, "runs": {"eda": "example", "proposal": "proposal"}}), encoding="utf-8")

        def directory(name, **kwargs):
            return self.root / "runs" / name

        def creation_root(*, create=False):
            path = self.root / "creations"

            if create:
                path.mkdir(exist_ok=True)

            return path

        for target, replacement in (("src.agents.wm_split_commit_cli._directory", directory),
                                    ("src.agents.wm_run_creation.creation_root", creation_root)):
            context = patch(target, side_effect=replacement)
            context.start()
            self.addCleanup(context.stop)

    def preview(self, *, config=None, maximum=None):
        return preview_creation(PIPELINE, self.bindings, "split", "new-commit", config,
                                WorkerLimits(300, 2048, 2), maximum)

    def test_creation_keeps_review_and_execution_approval(self):
        from src.agents.artifacts import _fingerprint
        paths = [path for path in self.root.rglob("*") if path.is_file()]
        before = {str(path): _fingerprint(path) for path in paths}
        preview = self.preview()
        self.assertEqual(before, {str(path): _fingerprint(path) for path in paths})
        self.assertTrue(preview.packet["review_required_before_execution"])
        self.assertIsNone(preview.packet["split_config"])
        self.assertEqual(preview.packet["max_output_bytes"], 2 * 1024 ** 3)

        with patch("src.agents.executor.RulesExecutor.run_next", side_effect=AssertionError("자동 실행 금지")), \
                patch("src.agents.approval_store.ApprovalStore.record_confirmed_approval", side_effect=AssertionError("검토·실행 승인 발급 금지")), \
                patch("src.agents.wm_split_commit.run_commit_worker", side_effect=AssertionError("worker 금지")):
            result = create_and_assign(preview, confirmed=True)

        self.assertEqual(result["status"], "created_and_assigned")
        self.assertTrue(result["review_required_before_execution"])
        self.assertFalse(result["worker_started"])
        self.assertFalse(result["execution_approval_issued"])
        self.assertEqual(load_bindings(result["bindings"], self.plan)["split"], "new-commit")
        self.assertNotIn("split", load_bindings(self.bindings, self.plan))
        self.assertFalse((Path(result["bindings"]).parent / "split_config.json").exists())
        tool, _, _, _ = load_run_tool("wm_split_commit", "new-commit", require_complete=False)
        self.assertEqual(tool.states.load().status, "created")
        self.assertEqual(tool.proposal_run, "proposal")
        stages = [item.stage for item in tool.plan.steps]
        self.assertLess(stages.index("review_eda"), stages.index("commit_split"))
        review = next(item for item in tool.reference_tools() if item.action_id == "review_eda")
        self.assertTrue(review.requires_approval)
        self.assertTrue(tool.registered_tool().requires_approval)
        self.assertEqual(before, {str(path): _fingerprint(path) for path in paths})

    def test_config_assignment_and_parent_changes_rejected(self):
        for values in ({"config": ROOT / "configs/wm811k/split.json"}, {"maximum": 2 * 1024 ** 3 + 1}):
            with self.assertRaises(ValueError):
                self.preview(**values)

        preview = self.preview()
        # 파일 열거 순서에 기대지 않고 실제 제안 결과의 표식만 변경한다.
        # preflight 표식을 고르면 부모 제안 변조 검사가 되지 않는다.
        marker = next(path for path in (self.proposal_dir / "attempts").glob("*/AGENT_COMPLETE.json")
                      if (path.parent / "proposal").is_dir())
        saved = marker.read_bytes()
        marker.write_bytes(b"{}")

        with patch("src.agents.wm_split_commit_cli.create") as create:
            with self.assertRaises(ValueError):
                create_and_assign(preview, confirmed=True)

        create.assert_not_called()
        self.assertFalse((self.root / "runs/new-commit").exists())
        marker.write_bytes(saved)
        bindings = json.loads(self.bindings.read_text())
        bindings["runs"]["split"] = "commit"
        self.bindings.write_text(json.dumps(bindings), encoding="utf-8")

        with self.assertRaises(ValueError):
            self.preview()


if __name__ == "__main__":
    unittest.main()
