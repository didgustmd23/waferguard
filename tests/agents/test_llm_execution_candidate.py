"""합성 snapshot과 임시 승인 DB만 사용해 제안→기존 작업 생성 연결을 확인한다."""

from contextlib import closing, redirect_stdout
from copy import deepcopy
from dataclasses import replace
from io import StringIO
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from src.agents import llm_execution_candidate as candidate
from src.agents import llm_execution_candidate_cli as cli
from src.agents.input_snapshot import create_snapshot
from src.agents.llm_proposal import ALLOWED, SOURCE
from src.agents.llm_proposal_contract import validate_proposal
from src.agents.synthetic_eda_data import CSV_BYTES
from src.agents.worker_contract import WorkerLimits
from tests.agents.test_llm_proposal import valid_content


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ExecutionCandidateTest(unittest.TestCase):
    """결합 검증·변경 중단·기존 생성기 재사용을 검사한다. worker 실행 검사가 아니다."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        source = self.root / "synthetic.csv"
        source.write_bytes(CSV_BYTES)
        self.snapshot = self.root / "snapshot"
        create_snapshot(source, self.snapshot, max_input_bytes=4096)
        self.profile = self.root / "profile.json"
        self.profile.write_bytes((PROJECT_ROOT / "configs/agents/profiles/tabular_example.json").read_bytes())
        self.restored = validate_proposal(valid_content(), expected_input=dict(SOURCE), allowed_modules=ALLOWED).to_dict()
        self.limits = WorkerLimits(60, 2048, 2)
        self.drafts = self.root / "candidates"
        self.runs = self.root / "runs"
        self.runs.mkdir()
        patcher = patch.object(candidate, "load_proposal", side_effect=lambda name: deepcopy(self.restored))
        patcher.start()
        self.addCleanup(patcher.stop)

    def preview(self):
        return candidate.preview_candidate("saved", self.snapshot, self.profile, self.limits)

    def prepare(self, expected):
        with patch("src.agents.pipeline_approval_cli._directory",
                   side_effect=lambda name, **kwargs: self.runs / name):
            return candidate.create_candidate(
                "prepared01", "saved", self.snapshot, self.profile, self.limits,
                expected_selection_id=expected, confirmed=True, root=self.drafts,
            )

    def test_preview_is_read_only_and_binds_profile(self):
        with patch.object(candidate, "create", side_effect=AssertionError("생성 금지")):
            preview = self.preview()
        self.assertFalse(self.drafts.exists())
        self.assertFalse(preview["worker_started"])
        self.assertFalse(preview["execution_approval_issued"])
        self.assertEqual(preview["selection"]["limits"]["processes"], 2)
        self.assertIn("quality_rules", preview["effective_execution"])

    def test_create_reuses_existing_run_without_approval(self):
        preview = self.preview()
        result = self.prepare(preview["selection_id"])
        self.assertEqual(result["status"], "execution_candidate_created")
        run = self.runs / "prepared01"
        self.assertEqual(json.loads((run / "state.json").read_text())["status"], "created")
        with closing(sqlite3.connect(f"{(run / 'approvals.sqlite3').as_uri()}?mode=ro", uri=True)) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM approvals").fetchone()[0], 0)
        self.assertTrue((self.drafts / "prepared01/CREATED.json").is_file())
        self.assertFalse(result["worker_started"])
        self.assertFalse(result["execution_approval_issued"])

    def test_profile_change_invalidates_preview(self):
        preview = self.preview()
        profile = json.loads(self.profile.read_text())
        profile["max_plot_features"] = profile.get("max_plot_features", 3) + 1
        self.profile.write_text(json.dumps(profile), encoding="utf-8")
        with patch.object(candidate, "create") as create, self.assertRaises(ValueError):
            self.prepare(preview["selection_id"])
        create.assert_not_called()
        self.assertFalse(self.drafts.exists())

    def test_proposal_change_invalidates_preview(self):
        preview = self.preview()
        self.restored["proposal"]["pipeline"]["pipeline_id"] = "changed"
        with patch.object(candidate, "create") as create, self.assertRaises(ValueError):
            self.prepare(preview["selection_id"])
        create.assert_not_called()

    def test_snapshot_tampering_is_rejected(self):
        (self.snapshot / "input.bin").write_bytes(b"changed synthetic input")
        with self.assertRaises(ValueError):
            self.preview()

    def test_limits_and_confirmation_are_required(self):
        preview = self.preview()
        self.limits = replace(self.limits, wall_seconds=61)
        with self.assertRaises(ValueError):
            self.prepare(preview["selection_id"])
        with self.assertRaises(ValueError):
            candidate.create_candidate("prepared01", "saved", self.snapshot, self.profile, self.limits,
                                       expected_selection_id=preview["selection_id"], root=self.drafts)
        self.assertFalse(self.drafts.exists())

    def test_duplicate_draft_not_overwritten(self):
        preview = self.preview()
        self.prepare(preview["selection_id"])
        with self.assertRaises(FileExistsError):
            self.prepare(preview["selection_id"])

    def test_unknown_profile_adapter_rejected(self):
        profile = json.loads(self.profile.read_text())
        profile["adapter_id"] = "unregistered"
        self.profile.write_text(json.dumps(profile), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.preview()

    def test_cli_decline_does_not_create(self):
        argv = ["candidate", "create", "--proposal-name", "saved", "--snapshot-dir", "synthetic",
                "--profile", "synthetic.json", "--name", "newrun", "--memory-mib", "2048",
                "--processes", "2", "--wall-seconds", "60"]
        with patch("sys.argv", argv), redirect_stdout(StringIO()):
            with patch("sys.stdin.isatty", return_value=True), patch("sys.stdout.isatty", return_value=True):
                with patch("builtins.input", return_value="아니요"):
                    with patch.object(cli, "preview_candidate", return_value={"selection_id": "example"}):
                        with patch.object(cli, "create_candidate") as create:
                            self.assertEqual(cli.main(), 0)
        create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
