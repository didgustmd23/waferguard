"""첫 EDA 준비의 바이트 복사·원본 보존·승인 및 배열 파싱 금지를 확인한다."""

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from tests.agents import test_wm_canonical as fixtures
from src.agents import wm_creation_cli, wm_canonical_cli
from src.agents.wm_module_plan import load_wm_modules
from src.agents.wm_run_bindings import load_bindings
from src.agents.wm_run_creation import create_and_assign, preview_creation
from src.agents.wm_run_evidence import load_run_tool
from src.agents.worker_contract import WorkerLimits


ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "configs/agents/pipelines/wm_development_example.json"


class WmEdaCreationTest(unittest.TestCase):
    def setUp(self):
        """작은 canonical fixture와 미배정 계획만 준비한다."""
        fixtures.WmCanonicalTest.setUp(self)
        self.source = Path(self.tool.contract.snapshot_directory)
        self.plan = load_wm_modules(PIPELINE)
        self.bindings = self.root / "empty_bindings.json"
        self.bindings.write_text(json.dumps({"schema_version": "wm_run_bindings_v1",
            "pipeline_signature": self.plan.signature, "runs": {}}), encoding="utf-8")

        def directory(name, **kwargs):
            return self.root / "runs" / name

        def creation_root(*, create=False):
            directory = self.root / "creations"

            if create:
                directory.mkdir(exist_ok=True)

            return directory

        for target, replacement in (("src.agents.wm_canonical_cli._directory", directory),
                                    ("src.agents.wm_run_creation.creation_root", creation_root)):
            mocking = patch(target, side_effect=replacement)
            mocking.start()
            self.addCleanup(mocking.stop)

    def preview(self, *, options=None, **kwargs):
        if options is None:
            options = {"max_input_bytes": 1024 * 1024, "include_examples": False, "max_group_ratio": 0.2}

        return preview_creation(PIPELINE, self.bindings, "eda", "new-eda", None, WorkerLimits(300, 2048, 2),
                                canonical_dir=self.source, canonical_options=options, **kwargs)

    def test_copy_assignment_and_no_execution(self):
        """미리보기는 무변경이며 사용자 확인 후에도 복사·준비만 한다."""
        from src.agents.artifacts import _fingerprint

        paths = [path for path in self.root.rglob("*") if path.is_file()]
        before = {str(path): _fingerprint(path) for path in paths}

        with patch("numpy.load", side_effect=AssertionError("배열 파싱 금지")), \
                patch("pandas.read_pickle", side_effect=AssertionError("pickle 금지")), \
                patch("src.agents.executor.RulesExecutor.run_next", side_effect=AssertionError("실행 금지")), \
                patch("src.agents.approval_store.ApprovalStore.record_confirmed_approval", side_effect=AssertionError("승인 발급 금지")):
            preview = self.preview()
            self.assertEqual(before, {str(path): _fingerprint(path) for path in paths})

            with self.assertRaises(ValueError):
                create_and_assign(preview)

            result = create_and_assign(preview, confirmed=True)

        self.assertEqual(result["status"], "created_and_assigned")
        self.assertFalse(result["worker_started"])
        self.assertFalse(result["execution_approval_issued"])
        self.assertEqual(load_bindings(result["bindings"], self.plan), {"eda": "new-eda"})
        self.assertEqual(load_bindings(self.bindings, self.plan), {})
        tool, _, _, _ = load_run_tool("wm_canonical_eda", "new-eda", require_complete=False)
        self.assertEqual(tool.states.load().status, "created")
        self.assertEqual(tool.states.load().current_stage, "preflight")
        self.assertNotEqual(Path(tool.contract.snapshot_directory), self.source)
        self.assertEqual(tool.contract.files, self.tool.contract.files)
        self.assertTrue(tool.registered_tool().requires_approval)
        self.assertEqual(before, {str(path): _fingerprint(path) for path in paths})

    def test_invalid_and_changed_source_rejected(self):
        """잘못된 설정·복사 상한 부족과 확인 후 원본 변경을 생성 전에 거부한다."""
        for changes in ({"max_input_bytes": 1}, {"max_input_bytes": True},
                        {"include_examples": 1}, {"max_group_ratio": 0}, {"unknown": 1}):
            options = {"max_input_bytes": 1024 * 1024, "include_examples": False, "max_group_ratio": 0.2, **changes}

            with self.assertRaises(ValueError):
                self.preview(options=options)

        preview = self.preview()
        shard = next((self.source / "maps").iterdir())
        shard.write_bytes(shard.read_bytes() + b"changed")

        with patch("src.agents.wm_canonical_cli.create") as create:
            with self.assertRaises(ValueError):
                create_and_assign(preview, confirmed=True)

        create.assert_not_called()
        self.assertFalse((self.root / "runs/new-eda").exists())

    def test_changed_copy_is_not_assigned(self):
        """바뀐 복사본은 준비 기록을 보존하되 배정 완료로 등록하지 않는다."""
        preview = self.preview()
        original = wm_canonical_cli.create

        def changed_copy(*args, **kwargs):
            result = original(*args, **kwargs)
            shard = next((self.root / "runs/new-eda/snapshot/maps").iterdir())
            shard.write_bytes(b"changed")
            return result

        with patch("src.agents.wm_canonical_cli.create", side_effect=changed_copy):
            with self.assertRaises(ValueError):
                create_and_assign(preview, confirmed=True)

        self.assertTrue((self.root / "runs/new-eda").exists())
        self.assertFalse((self.root / "creations/new-eda/bindings.json").exists())
        self.assertFalse((self.root / "creations/new-eda/CREATION_COMPLETE.json").exists())

    def test_cli_relative_paths_and_options(self):
        """상대경로와 입력 상한을 Dataset이 아닌 첫 EDA 설정으로 전달한다."""
        previous = Path.cwd()
        arguments = ["wm_creation_cli", "preview", "--pipeline", str(PIPELINE), "--bindings", self.bindings.name,
                     "--step", "eda", "--name", "new-eda", "--canonical-dir", "runs/example/snapshot",
                     "--max-input-bytes", "1048576"]

        try:
            os.chdir(self.root)

            with patch("sys.argv", arguments), redirect_stdout(io.StringIO()) as output:
                self.assertEqual(wm_creation_cli.main(), 0)
        finally:
            os.chdir(previous)

        selection = json.loads(output.getvalue())["selection"]
        self.assertIsNone(selection["dataset_options"])
        self.assertEqual(selection["canonical_options"]["max_input_bytes"], 1048576)
        self.assertFalse((self.root / "runs/new-eda").exists())


if __name__ == "__main__":
    unittest.main()
