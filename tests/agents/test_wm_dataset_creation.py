"""완료 캐시에서 Dataset 검사 작업만 준비하고 실행·학습은 하지 않는다."""

from copy import deepcopy
from importlib.metadata import version
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from tests.agents import test_wm_cache as fixtures
from src.agents.wm_canonical_contract import file_hash
from src.agents.wm_dataset_creation import DEFAULT_OPTIONS
from src.agents.wm_module_plan import load_wm_modules
from src.agents.wm_run_bindings import load_bindings
from src.agents.wm_run_creation import create_and_assign, preview_creation
from src.agents.wm_run_evidence import load_run_tool
from src.agents.worker_contract import WorkerLimits


ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "configs/agents/pipelines/wm_development_example.json"


class WmDatasetCreationTest(unittest.TestCase):
    transport = fixtures.WmCacheTest.transport
    proposal_transport = fixtures.WmCacheTest.proposal_transport
    commit_transport = fixtures.WmCacheTest.commit_transport
    check_transport = fixtures.WmCacheTest.check_transport
    cache_transport = fixtures.WmCacheTest.cache_transport

    def setUp(self):
        """부모 캐시까지만 합성 완료하고 설치 런타임 수집은 작은 대역을 쓴다."""
        fixtures.WmCacheTest.setUp(self)
        approval = fixtures.WmCacheTest.approval(self)

        with patch("src.agents.wm_cache.run_cache_worker", side_effect=self.cache_transport):
            self.assertEqual(self.manager.run_next(approval_id=approval.approval_id).status, "completed")

        self.plan = load_wm_modules(PIPELINE)
        self.bindings = self.root / "dataset_bindings.json"
        self.bindings.write_text(json.dumps({"schema_version": "wm_run_bindings_v1",
            "pipeline_signature": self.plan.signature,
            "runs": {"eda": "example", "proposal": "proposal", "split": "commit", "transform": "check", "cache": "cache"}}), encoding="utf-8")
        torch_path = Path(sys.prefix) / "Lib/site-packages/torch/__init__.py"
        self.runtime = {"Lib/site-packages/torch/__init__.py": file_hash(torch_path, 1024 * 1024)}

        def directory(name, **kwargs):
            return self.root / "runs" / name

        def creation_root(*, create=False):
            directory = self.root / "creations"

            if create:
                directory.mkdir(exist_ok=True)

            return directory

        for target, replacement in (("src.agents.wm_dataset_cli._directory", directory),
                                    ("src.agents.wm_run_creation.creation_root", creation_root)):
            mocking = patch(target, side_effect=replacement)
            mocking.start()
            self.addCleanup(mocking.stop)

        mocking = patch("src.agents.wm_dataset_cli.runtime_manifest", return_value=(self.runtime, version("torch")))
        self.runtime_mock = mocking.start()
        self.addCleanup(mocking.stop)

    def preview(self, *, options=None, maximum=None, **other):
        return preview_creation(PIPELINE, self.bindings, "dataset", "new-dataset", None,
            WorkerLimits(300, 2048, 2), maximum, dataset_options=options, **other)

    def test_creation_keeps_runtime_and_execution_approval(self):
        """입력·런타임·배치 계약과 새 배정만 저장한다. 배열이나 모델은 실행하지 않는다."""
        from src.agents.artifacts import _fingerprint

        paths = [path for path in self.root.rglob("*") if path.is_file()]
        before = {str(path): _fingerprint(path) for path in paths}
        preview = self.preview()
        self.assertEqual(before, {str(path): _fingerprint(path) for path in paths})
        self.assertEqual(preview.packet["dataset_options"]["batch_size"], 32)
        self.assertEqual(preview.packet["max_output_bytes"], 1024 * 1024)

        with patch("src.agents.executor.RulesExecutor.run_next", side_effect=AssertionError("실행 금지")), \
                patch("src.agents.approval_store.ApprovalStore.record_confirmed_approval", side_effect=AssertionError("승인 발급 금지")), \
                patch("src.agents.wm_dataset.run_dataset_worker", side_effect=AssertionError("worker 금지")):
            result = create_and_assign(preview, confirmed=True)

        self.assertEqual(result["status"], "created_and_assigned")
        self.assertFalse(result["worker_started"])
        self.assertFalse(result["execution_approval_issued"])
        self.assertEqual(load_bindings(result["bindings"], self.plan)["dataset"], "new-dataset")
        self.assertNotIn("dataset", load_bindings(self.bindings, self.plan))
        tool, _, _, _ = load_run_tool("wm_dataset_check", "new-dataset", require_complete=False)
        self.assertEqual(tool.states.load().status, "created")
        self.assertEqual(tool.states.load().current_stage, "preflight")
        self.assertEqual(tool.cache_run, "cache")

        for key in ("batch_size", "max_input_bytes", "max_runtime_bytes", "runtime_sha256", "torch_version"):
            self.assertEqual(getattr(tool.contract, key), preview.packet["dataset_options"][key])

        self.assertTrue(tool.registered_tool().requires_approval)
        self.assertEqual(before, {str(path): _fingerprint(path) for path in paths})

    def test_limits_runtime_and_parent_changes_rejected(self):
        """잘못된 배치·복사 예산·설정 재지정과 확인 후 런타임/부모 변경을 거부한다."""
        for change in ({"batch_size": 0}, {"batch_size": True}, {"batch_size": 257},
                       {"max_input_bytes": 1}, {"max_runtime_bytes": 32 * 1024 ** 3 + 1}, {"unknown": 1}):
            with self.assertRaises(ValueError):
                self.preview(options={**DEFAULT_OPTIONS, **change})

        for other in ({"maximum": 64 * 1024 * 1024 + 1},
                      {"cache_config": ROOT / "configs/wm811k/cache.json"}):
            with self.assertRaises(ValueError):
                self.preview(**other)

        preview = self.preview()
        changed = deepcopy(self.runtime)
        changed[next(iter(changed))]["sha256"] = "0" * 64
        self.runtime_mock.return_value = (changed, version("torch"))

        with patch("src.agents.wm_dataset_cli.create") as create:
            with self.assertRaises(ValueError):
                create_and_assign(preview, confirmed=True)

        create.assert_not_called()
        self.runtime_mock.return_value = (self.runtime, version("torch"))
        marker = next(path for path in (self.cache_dir / "attempts").glob("*/AGENT_COMPLETE.json")
                      if (path.parent / "cache").is_dir())
        marker.write_bytes(b"{}")

        with patch("src.agents.wm_dataset_cli.create") as create:
            with self.assertRaises(ValueError):
                create_and_assign(preview, confirmed=True)

        create.assert_not_called()
        self.assertFalse((self.root / "runs/new-dataset").exists())


if __name__ == "__main__":
    unittest.main()
