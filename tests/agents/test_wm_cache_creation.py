"""완료 변환 검사에서 캐시 작업만 준비하고 검토·실행 승인은 유지한다."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

from tests.agents import test_wm_cache as fixtures
from src.agents.wm_module_plan import load_wm_modules
from src.agents.wm_run_bindings import load_bindings
from src.agents.wm_run_creation import create_and_assign, preview_creation
from src.agents.wm_run_evidence import load_run_tool
from src.agents.worker_contract import WorkerLimits


ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "configs/agents/pipelines/wm_development_example.json"


class WmCacheCreationTest(unittest.TestCase):
    transport = fixtures.WmCacheTest.transport
    proposal_transport = fixtures.WmCacheTest.proposal_transport
    commit_transport = fixtures.WmCacheTest.commit_transport
    check_transport = fixtures.WmCacheTest.check_transport

    def setUp(self):
        """부모 변환 검사까지만 합성 완료하고 새 캐시 생성 경로를 격리한다."""
        fixtures.WmCacheTest.setUp(self)
        self.plan = load_wm_modules(PIPELINE)
        self.bindings = self.root / "cache_bindings.json"
        self.bindings.write_text(json.dumps({"schema_version": "wm_run_bindings_v1",
            "pipeline_signature": self.plan.signature,
            "runs": {"eda": "example", "proposal": "proposal", "split": "commit", "transform": "check"}}), encoding="utf-8")
        self.config_path = self.root / "cache.json"
        self.config_path.write_bytes((ROOT / "configs/wm811k/cache.json").read_bytes())

        def directory(name, **kwargs):
            return self.root / "runs" / name

        def creation_root(*, create=False):
            directory = self.root / "creations"

            if create:
                directory.mkdir(exist_ok=True)

            return directory

        for target, replacement in (("src.agents.wm_cache_cli._directory", directory),
                                    ("src.agents.wm_run_creation.creation_root", creation_root)):
            mocking = patch(target, side_effect=replacement)
            mocking.start()
            self.addCleanup(mocking.stop)

    def preview(self, *, maximum=None, **options):
        return preview_creation(PIPELINE, self.bindings, "cache", "new-cache", None,
            WorkerLimits(300, 2048, 2), maximum, cache_config=self.config_path, **options)

    def test_creation_keeps_reviews_and_no_execution(self):
        """설정 복사·새 배정만 만들며 시각 검토·분할 재확인·실행 승인을 남긴다."""
        from src.agents.artifacts import _fingerprint

        paths = [path for path in self.root.rglob("*") if path.is_file()]
        before = {str(path): _fingerprint(path) for path in paths}
        preview = self.preview()
        self.assertEqual(before, {str(path): _fingerprint(path) for path in paths})
        self.assertEqual(preview.packet["max_output_bytes"], 8 * 1024 ** 3)
        self.assertGreater(preview.packet["estimated_output_bytes"], 0)
        self.assertTrue(preview.packet["review_required_before_execution"])

        with patch("src.agents.executor.RulesExecutor.run_next", side_effect=AssertionError("실행 금지")), \
                patch("src.agents.approval_store.ApprovalStore.record_confirmed_approval", side_effect=AssertionError("승인 발급 금지")), \
                patch("src.agents.wm_cache.run_cache_worker", side_effect=AssertionError("worker 금지")):
            result = create_and_assign(preview, confirmed=True)

        self.assertEqual(result["status"], "created_and_assigned")
        self.assertFalse(result["worker_started"])
        self.assertFalse(result["execution_approval_issued"])
        self.assertTrue(result["review_required_before_execution"])
        self.assertEqual(load_bindings(result["bindings"], self.plan)["cache"], "new-cache")
        self.assertNotIn("cache", load_bindings(self.bindings, self.plan))
        saved_config = Path(result["bindings"]).parent / "cache_config.json"
        self.assertEqual(json.loads(saved_config.read_text(encoding="utf-8")), preview.packet["cache_config"])
        tool, _, _, _ = load_run_tool("wm_cache_prepare", "new-cache", require_complete=False)
        self.assertEqual(tool.states.load().status, "created")
        self.assertEqual(tool.states.load().current_stage, "preflight")
        self.assertEqual(tool.check_run, "check")
        self.assertEqual(tool.evidence["cache_config"], preview.packet["cache_config"])
        reviews = {item.action_id: item for item in tool.reference_tools()}

        for stage in ("review_eda", "commit_split"):
            self.assertTrue(reviews[stage].requires_approval)

        self.assertTrue(tool.registered_tool().requires_approval)
        self.assertEqual(before, {str(path): _fingerprint(path) for path in paths})

    def test_limits_and_changed_inputs_rejected(self):
        """출력 부족·상한 초과·설정 재지정·확인 후 설정/부모 변경은 거부한다."""
        for options in ({"maximum": 1}, {"maximum": 32 * 1024 ** 3 + 1},
                        {"transform_config": ROOT / "configs/wm811k/transform.json"}):
            with self.assertRaises(ValueError):
                self.preview(**options)

        preview = self.preview()
        original = self.config_path.read_bytes()
        config = json.loads(original)
        config["shard_max_rows"] = 0
        self.config_path.write_text(json.dumps(config), encoding="utf-8")

        with self.assertRaises(ValueError):
            self.preview()

        config["shard_max_rows"] = 2048
        self.config_path.write_text(json.dumps(config), encoding="utf-8")

        with patch("src.agents.wm_cache_cli.create") as create:
            with self.assertRaises(ValueError):
                create_and_assign(preview, confirmed=True)

        create.assert_not_called()
        self.config_path.write_bytes(original)
        # 실제 변환 검사 결과의 완료 표식만 변조한다. preflight 표식과 구별한다.
        marker = next(path for path in (self.check_dir / "attempts").glob("*/AGENT_COMPLETE.json")
                      if (path.parent / "check").is_dir())
        marker.write_bytes(b"{}")

        with patch("src.agents.wm_cache_cli.create") as create:
            with self.assertRaises(ValueError):
                create_and_assign(preview, confirmed=True)

        create.assert_not_called()
        self.assertFalse((self.root / "runs/new-cache").exists())


if __name__ == "__main__":
    unittest.main()
