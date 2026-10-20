"""합성 실행안만 저장·변경·복원한다. 모델·데이터·승인 DB·worker는 사용하지 않는다."""

from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.agents import llm_proposal_cli, llm_proposal_store as store
from src.agents.llm_proposal import ALLOWED, SOURCE
from src.agents.llm_proposal_contract import validate_proposal
from tests.agents.test_llm_proposal import valid_content


class ProposalStoreTest(unittest.TestCase):
    """내용 일치와 현재 계약 검사를 확인한다. 해시를 진본 인증으로 취급하지 않는다."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "proposals"
        self.result = validate_proposal(
            valid_content(), expected_input=dict(SOURCE), allowed_modules=ALLOWED,
        ).to_dict()
        self.result["status"] = "proposal_validated"

    def saved(self):
        store.save_proposal("example", self.result, root=self.root)
        return self.root / "example.json"

    def rewrite(self, value):
        # 파일 작성도 실제 데이터가 없는 전용 테스트 임시 폴더에 한정한다.
        (self.root / "example.json").write_text(json.dumps(value), encoding="utf-8")

    def test_roundtrip_has_no_authority_or_prompt(self):
        path = self.saved()
        restored = store.load_proposal("example", root=self.root)
        self.assertEqual(restored["proposal"], self.result["proposal"])
        self.assertEqual(restored["plan_signature"], self.result["plan_signature"])
        self.assertEqual(restored["status"], "proposal_restored")
        self.assertFalse(restored["model_called"])
        self.assertFalse(restored["execution_approved"])
        self.assertFalse(restored["worker_started"])
        self.assertEqual(set(json.loads(path.read_text(encoding="utf-8"))), {
            "schema_version", "proposal", "plan_signature", "proposal_sha256",
            "proposal_only", "execution_approved", "worker_started",
        })

    def test_list_is_empty_without_store_and_creates_nothing(self):
        self.assertEqual(store.list_saved_proposals(root=self.root), [])
        self.assertFalse(self.root.exists())

    def test_list_shows_names_and_times_newest_first_without_reading_content(self):
        self.saved()
        store.save_proposal("second", self.result, root=self.root)
        os.utime(self.root / "example.json", (1_700_000_000, 1_700_000_000))
        os.utime(self.root / "second.json", (1_700_000_100, 1_700_000_100))
        (self.root / "broken.json").write_text("not json", encoding="utf-8")
        os.utime(self.root / "broken.json", (1_600_000_000, 1_600_000_000))
        (self.root / "Bad-Name.json").write_text("{}", encoding="utf-8")
        (self.root / "notes.txt").write_text("x", encoding="utf-8")
        (self.root / "folder.json").mkdir()

        with patch("pathlib.Path.open", side_effect=AssertionError("내용 읽기 금지")):
            items = store.list_saved_proposals(root=self.root)

        self.assertEqual([item["name"] for item in items], ["second", "example", "broken"])
        self.assertEqual(set(items[0]), {"name", "saved_at"})
        self.assertTrue(items[0]["saved_at"] > items[1]["saved_at"])
        # 목록에 있어도 복원 검증은 별도다.
        with self.assertRaises(ValueError):
            store.load_proposal("broken", root=self.root)

    def test_list_rejects_relative_root_and_is_bounded(self):
        with self.assertRaises(ValueError):
            store.list_saved_proposals(root=Path("relative/proposals"))

        self.root.mkdir(parents=True)
        for index in range(store.MAX_LISTED + 5):
            (self.root / f"p{index}.json").write_text("{}", encoding="utf-8")

        self.assertEqual(len(store.list_saved_proposals(root=self.root)), store.MAX_LISTED)

    def test_duplicate_never_overwrites(self):
        path = self.saved()
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            store.save_proposal("example", self.result, root=self.root)
        self.assertEqual(path.read_bytes(), before)

    def test_unsafe_name_is_rejected_before_creation(self):
        for name in ("../escape", "bad/name", "C:escape", "Example", "", "a" * 65):
            with self.subTest(name=name), self.assertRaises(ValueError):
                store.save_proposal(name, self.result, root=self.root)
        self.assertFalse(self.root.exists())

    def test_changed_content_or_signature_rejected(self):
        path = self.saved()
        original = json.loads(path.read_text(encoding="utf-8"))
        for field in ("proposal_sha256", "plan_signature"):
            value = deepcopy(original)
            value[field] = "0" * 64
            self.rewrite(value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                store.load_proposal("example", root=self.root)

    def test_recomputed_checksum_cannot_allow_invalid_module_or_input(self):
        path = self.saved()
        original = json.loads(path.read_text(encoding="utf-8"))
        for change in ("module", "input"):
            value = deepcopy(original)
            if change == "module":
                value["proposal"]["pipeline"]["steps"][0]["version"] = "99.0.0"
            else:
                value["proposal"]["pipeline"]["input"]["partition"] = "test"
            value["proposal_sha256"] = store._digest(value["proposal"])
            self.rewrite(value)
            with self.subTest(change=change), self.assertRaises(ValueError):
                store.load_proposal("example", root=self.root)

    def test_authority_and_extra_fields_rejected(self):
        path = self.saved()
        original = json.loads(path.read_text(encoding="utf-8"))
        for field in ("execution_approved", "worker_started", "approval_id"):
            value = dict(original, **{field: True})
            self.rewrite(value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                store.load_proposal("example", root=self.root)

    def test_partial_duplicate_keys_and_oversize_rejected(self):
        path = self.saved()
        for raw in (b'{"partial":', b'{"x":1,"x":2}', b'x' * (store.MAX_SAVED_BYTES + 1)):
            path.write_bytes(raw)
            with self.subTest(raw=raw[:15]), self.assertRaises(ValueError):
                store.load_proposal("example", root=self.root)

    def test_clarification_and_forged_result_not_saved(self):
        for result in (dict(self.result, execution_approved=True),
                       dict(self.result, plan_signature="0" * 64),
                       dict(self.result, proposal={"schema_version": "llm_proposal_v1",
                            "status": "needs_clarification", "pipeline": None, "questions": ["목적은?"]})):
            with self.assertRaises(ValueError):
                store.save_proposal("example", result, root=self.root)
        self.assertFalse(self.root.exists())

    def test_cli_restore_never_loads_model_config(self):
        restored = {"status": "proposal_restored", "worker_started": False, "model_called": False}
        with patch("sys.argv", ["proposal", "--load-name", "example"]), redirect_stdout(StringIO()):
            with patch.object(llm_proposal_cli, "load_proposal", return_value=restored) as load:
                with patch.object(llm_proposal_cli, "load_local_config") as config:
                    self.assertEqual(llm_proposal_cli.main(), 0)
        load.assert_called_once_with("example")
        config.assert_not_called()

    def test_cli_save_requires_separate_confirmation(self):
        for answer in ("아니요", "SAVE"):
            with redirect_stdout(StringIO()), patch("builtins.input", return_value=answer):
                with patch.object(llm_proposal_cli, "save_proposal", return_value={"status": "proposal_saved"}) as save:
                    result = llm_proposal_cli._save_selected_result("example", deepcopy(self.result))
            if answer == "SAVE":
                save.assert_called_once()
                self.assertIn("saved_proposal", result)
            else:
                save.assert_not_called()
                self.assertEqual(result["save_status"], "declined")


if __name__ == "__main__":
    unittest.main()
