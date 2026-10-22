"""사용자 역할 지정 → 프로파일 계약 변환·배타 저장·재적재를 합성 컬럼으로 검사한다."""

import json
from pathlib import Path
import tempfile
import unittest

from src.agents.profile_store import build_profile, load_saved_profile, save_profile
from src.agents.split_evidence import read_metadata
from src.agents.worker_contract import _decode
from src.preprocessing.profiles import load_profile, profile_from_dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COLUMNS = ("lot_id", "x1", "x2", "CANARY_COLUMN", "label", "time")


def roles(**changes):
    value = dict.fromkeys(COLUMNS, "제외")
    value.update(changes)
    return value


class ProfileStoreTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(dir=PROJECT_ROOT)
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / "profiles"

    def eda(self, **changes):
        return build_profile(COLUMNS, roles(x1="수치형", x2="수치형", **changes), purpose="eda", profile_id="ui_eda01")

    def test_eda_profile_keeps_order_and_omits_excluded(self):
        profile = self.eda(lot_id="식별자", label="target", CANARY_COLUMN="범주형")
        value = profile.to_dict()
        self.assertEqual(value["workflow_id"], "eda_review_v1")
        self.assertEqual(value["adapter_id"], "tabular_csv_v1")
        self.assertEqual(value["roles"], {"numeric": ["x1", "x2"], "categorical": ["CANARY_COLUMN"], "identifiers": ["lot_id"],
                                          "target": "label", "group": None, "timestamp": None})
        self.assertNotIn('"time"', json.dumps(value))
        self.assertNotIn("split_rules", value)
        # 기본 품질 기준은 생략해 기존 식별값 규칙을 따른다.
        self.assertNotIn("quality_rules", value)

    def test_quality_and_split_rules(self):
        profile = build_profile(
            COLUMNS, roles(x1="수치형", label="target", time="timestamp"), purpose="split", profile_id="ui_split01",
            quality={"missing_ratio_threshold": 0.3, "high_missing": "block"},
            split={"strategy": "time", "train_ratio": 0.6, "validation_ratio": 0.2, "test_ratio": 0.2,
                   "random_state": 7, "stratify": False, "require_all_classes": False},
        )
        value = profile.to_dict()
        self.assertEqual(value["workflow_id"], "split_review_v1")
        self.assertEqual(value["quality_rules"]["missing_ratio_threshold"], 0.3)
        self.assertEqual(value["quality_rules"]["high_missing"], "block")
        self.assertEqual(value["split_rules"]["strategy"], "time")

    def test_rejections(self):
        split = {"strategy": "random", "train_ratio": 0.7, "validation_ratio": 0.15, "test_ratio": 0.15}
        cases = {
            "no_features": lambda: build_profile(COLUMNS, roles(label="target"), purpose="eda", profile_id="ui_x"),
            "two_targets": lambda: self.eda(label="target", time="target"),
            "unknown_role": lambda: self.eda(label="추천"),
            "missing_column": lambda: build_profile(COLUMNS, {"x1": "수치형"}, purpose="eda", profile_id="ui_x"),
            "split_without_rules": lambda: build_profile(COLUMNS, roles(x1="수치형"), purpose="split", profile_id="ui_x"),
            "eda_with_split": lambda: build_profile(COLUMNS, roles(x1="수치형"), purpose="eda", profile_id="ui_x", split=split),
            "bad_ratio": lambda: build_profile(COLUMNS, roles(x1="수치형"), purpose="split", profile_id="ui_x",
                                               split=dict(split, test_ratio=0.3)),
            "bad_id": lambda: build_profile(COLUMNS, roles(x1="수치형"), purpose="eda", profile_id="../x"),
            "quality_field": lambda: build_profile(COLUMNS, roles(x1="수치형"), purpose="eda", profile_id="ui_x",
                                                   quality={"allowed_labels": [0, 1]}),
            "stratify_group": lambda: build_profile(COLUMNS, roles(x1="수치형"), purpose="split", profile_id="ui_x",
                                                    split=dict(split, strategy="group", stratify=True)),
            "duplicate_columns": lambda: build_profile(("x1", "x1"), {"x1": "수치형"}, purpose="eda", profile_id="ui_x"),
        }

        for name, call in cases.items():
            with self.subTest(name):
                with self.assertRaises((ValueError, TypeError)) as caught:
                    call()

                self.assertNotIn("CANARY_COLUMN", str(caught.exception))

        self.assertFalse(self.root.exists())

    def test_save_reload_by_existing_readers_and_no_overwrite(self):
        profile = self.eda(label="target")
        saved = save_profile(profile, root=self.root)
        path = Path(saved["local_file"])
        self.assertEqual(path.name, "ui_eda01.json")
        self.assertIs(saved["execution_approved"], False)
        self.assertIs(saved["worker_started"], False)
        self.assertEqual(saved["workflow_id"], "eda_review_v1")

        # 기존 프로파일 소비 경로 세 가지가 같은 계약으로 읽는다.
        self.assertEqual(load_profile(path).signature, profile.signature)
        self.assertEqual(profile_from_dict(read_metadata(path)).signature, profile.signature)
        self.assertEqual(profile_from_dict(_decode(path.read_text(encoding="utf-8"))).signature, profile.signature)
        self.assertEqual(load_saved_profile(path, root=self.root).signature, profile.signature)

        before = path.read_bytes()

        with self.assertRaises(FileExistsError):
            save_profile(self.eda(label="target", lot_id="식별자"), root=self.root)

        self.assertEqual(path.read_bytes(), before)

    def test_load_rejects_outside_store_and_tampered_file(self):
        saved = save_profile(self.eda(), root=self.root)
        outside = self.root.parent / "outside.json"
        outside.write_text(Path(saved["local_file"]).read_text(encoding="utf-8"), encoding="utf-8")

        with self.assertRaises(ValueError):
            load_saved_profile(outside, root=self.root)

        tampered = self.root / "tampered.json"
        tampered.write_text('{"profile_id":"a","profile_id":"b"}', encoding="utf-8")

        with self.assertRaises(ValueError):
            load_saved_profile(tampered, root=self.root)

    def test_reserved_device_names_are_rejected_before_writing(self):
        for name in ("con", "prn", "aux", "nul", "com1", "lpt9"):
            with self.subTest(name):
                profile = build_profile(COLUMNS, roles(x1="수치형"), purpose="eda", profile_id=name)

                with self.assertRaises(ValueError):
                    save_profile(profile, root=self.root)

        self.assertEqual(list(self.root.iterdir()) if self.root.exists() else [], [])

    def test_relative_store_root_is_rejected(self):
        with self.assertRaises(ValueError):
            save_profile(self.eda(), root=Path("relative/profiles"))


if __name__ == "__main__":
    unittest.main()
