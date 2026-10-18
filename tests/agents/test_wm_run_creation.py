"""합성 완료 EDA에서 작업 준비·배정만 생성하고 worker와 실행 승인은 금지한다."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from tests.agents import test_wm_split as fixtures
from src.agents import wm_creation_cli
from src.agents import wm_split_cli
from src.agents.wm_module_plan import load_wm_modules
from src.agents.wm_run_bindings import load_bindings
from src.agents.wm_run_creation import create_and_assign, preview_creation
from src.agents.worker_contract import WorkerLimits


ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "configs/agents/pipelines/wm_development_example.json"


class WmCreationTest(unittest.TestCase):
    transport = fixtures.WmSplitTest.transport

    def setUp(self):
        fixtures.WmSplitTest.setUp(self)
        self.plan = load_wm_modules(PIPELINE)
        self.bindings = self.root / "bindings.json"
        self.config = self.root / "split_config.json"
        self.config.write_bytes((ROOT / "configs/wm811k/split.json").read_bytes())
        self.bindings.write_text(json.dumps({"schema_version": "wm_run_bindings_v1",
                "pipeline_signature": self.plan.signature, "runs": {"eda": "example"}}), encoding="utf-8")

        def directory(name, **kwargs):
            return self.root / "runs" / name

        def creation_root(*, create=False):
            path = self.root / "creations"

            if create:
                path.mkdir(exist_ok=True)

            return path

        for target, replacement in (("src.agents.wm_split_cli._directory", directory),
                                    ("src.agents.wm_run_creation.creation_root", creation_root)):
            context = patch(target, side_effect=replacement)
            context.start()
            self.addCleanup(context.stop)

    def preview(self, name="new-proposal"):
        return preview_creation(PIPELINE, self.bindings, "proposal", name, self.config, WorkerLimits(300, 2048, 2))

    def test_preview_confirmation_and_created_assignment(self):
        from src.agents.artifacts import _fingerprint
        from src.agents.wm_run_evidence import load_run_tool

        paths = [path for path in self.root.rglob("*") if path.is_file()]
        before = {str(path): _fingerprint(path) for path in paths}

        with patch("src.agents.wm_split_cli.create", side_effect=AssertionError("미리보기 중 생성 금지")):
            preview = self.preview()

        self.assertEqual(before, {str(path): _fingerprint(path) for path in paths})
        self.assertFalse((self.root / "creations").exists())

        with self.assertRaises(ValueError):
            create_and_assign(preview)

        self.assertFalse((self.root / "creations").exists())

        with patch("src.agents.executor.RulesExecutor.run_next", side_effect=AssertionError("자동 실행 금지")), \
                patch("src.agents.approval_store.ApprovalStore.record_confirmed_approval", side_effect=AssertionError("실행 승인 발급 금지")), \
                patch("src.agents.wm_split.run_proposal_worker", side_effect=AssertionError("worker 금지")):
            result = create_and_assign(preview, confirmed=True)

        self.assertEqual(result["status"], "created_and_assigned")
        self.assertFalse(result["execution_approval_issued"])
        self.assertFalse(result["worker_started"])
        self.assertEqual(load_bindings(result["bindings"], self.plan), {"eda": "example", "proposal": "new-proposal"})
        self.assertEqual(load_bindings(self.bindings, self.plan), {"eda": "example"})
        self.assertTrue((self.root / "creations/new-proposal/CREATION_COMPLETE.json").is_file())
        tool, _, _, _ = load_run_tool("wm_split_propose", "new-proposal", require_complete=False)
        self.assertEqual(tool.states.load().status, "created")
        self.assertEqual(tool.evidence["split_config"], preview.packet["split_config"])

        # 기존 EDA·기존 제안·원래 배정은 바뀌지 않는다. 새 작업과 새 기록만 생긴다.
        self.assertEqual(before, {str(path): _fingerprint(path) for path in paths})

        with self.assertRaises(ValueError):
            create_and_assign(preview, confirmed=True)

    def test_changed_selection_and_invalid_requests(self):
        preview = self.preview()
        changed = json.loads(self.config.read_text())
        changed["seed"] += 1
        self.config.write_text(json.dumps(changed), encoding="utf-8")

        with patch("src.agents.wm_split_cli.create") as create:
            with self.assertRaises(ValueError):
                create_and_assign(preview, confirmed=True)

        create.assert_not_called()
        self.assertFalse((self.root / "runs/new-proposal").exists())
        self.assertFalse((self.root / "creations/new-proposal").exists())

        for step, name, maximum in (("cache", "new-cache", 65536),
                                     ("proposal", "../escape", 65536),
                                     ("proposal", "new-proposal", 1024 ** 3 + 1)):
            with self.subTest(step=step, name=name), self.assertRaises(ValueError):
                preview_creation(PIPELINE, self.bindings, step, name, self.config,
                                 WorkerLimits(300, 2048, 2), maximum)

        assigned = json.loads(self.bindings.read_text())
        assigned["runs"]["proposal"] = "proposal"
        self.bindings.write_text(json.dumps(assigned), encoding="utf-8")

        with self.assertRaises(ValueError):
            self.preview()

    def test_partial_creation_is_preserved_unassigned(self):
        preview = self.preview()
        original_create = wm_split_cli.create

        def create_then_change(*args, **kwargs):
            result = original_create(*args, **kwargs)
            self.bindings.write_text(self.bindings.read_text() + " ", encoding="utf-8")
            return result

        with patch("src.agents.wm_split_cli.create", side_effect=create_then_change) as create:
            with self.assertRaises(ValueError):
                create_and_assign(preview, confirmed=True)

        self.assertEqual(create.call_count, 1)
        self.assertTrue((self.root / "runs/new-proposal/state.json").is_file())
        self.assertTrue((self.root / "creations/new-proposal/run_created.json").is_file())
        self.assertFalse((self.root / "creations/new-proposal/CREATION_COMPLETE.json").exists())
        self.assertNotIn("proposal", load_bindings(self.bindings, self.plan))


class WmCreationCliTest(unittest.TestCase):
    def test_interactive_confirmation_required(self):
        preview = Mock(signature="preview", packet={"worker_started": False})
        args = ["wm_creation_cli", "create", "--pipeline", "plan", "--bindings", "bindings",
                "--step", "proposal", "--name", "example"]

        for interactive, answer, expected_calls in ((False, "CREATE-ABCD", 0), (True, "wrong", 0),
                                                    (True, "CREATE-ABCD", 1)):
            with patch("sys.argv", args), patch("src.agents.wm_creation_cli.preview_creation", return_value=preview), \
                    patch("src.agents.wm_creation_cli.os.isatty", return_value=interactive), \
                    patch("src.agents.wm_creation_cli.secrets.token_hex", return_value="abcd"), \
                    patch("builtins.input", return_value=answer), \
                    patch("src.agents.wm_creation_cli.create_and_assign", return_value={"status": "created_and_assigned"}) as create, \
                    redirect_stdout(io.StringIO()):
                code = wm_creation_cli.main()

            self.assertEqual(create.call_count, expected_calls)
            self.assertEqual(code, 0 if expected_calls else 2)


if __name__ == "__main__":
    unittest.main()
