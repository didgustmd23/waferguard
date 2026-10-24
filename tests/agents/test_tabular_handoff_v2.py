"""typed 전달 묶음의 자료형 보존·파일 대조·위조 거부를 합성 v2 실행으로 검사한다.

v2 체인 생성 비용이 커서 클래스마다 한 번만 만들고, 각 검사는 원본 묶음의 복사본을 쓴다.
원본을 바꾸는 검사는 끝난 뒤 원본을 되돌린다.
"""

import csv
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.agents import csv_handoff_contract
from src.agents.artifacts import _fingerprint
from src.agents.csv_handoff_cli import export_bundle
from src.agents.csv_handoff_contract import load_handoff_partition, verify_bundle
from src.agents.csv_workflow_service import CsvWorkflowService
from src.agents.workflow_service_registry import workflow_service
from src.preprocessing.tabular_bundle import _encode
from tests.agents.test_tabular_workflow_v2 import PIPELINE, TabularWorkflowFixture


# Train(그룹 a)에 추가 반올림하면 달라지는 float64 값을 둔다. 라벨·그룹·식별자 결측은 분할 단계가 구조적으로 거부한다.
HANDOFF_CSV = ("x,flag,label,g,id\n0.10000000000000002,1,True,a,0012\n,0,False,a,0034\n9.0,1,True,a,0056\n"
               "100.0,0,False,b,0078\n,1,True,b,0090\n700.0,0,False,c,0102\n")


def _chain():
    """클래스 단위로 한 번 실행하는 v2 체인. 모듈 수준 TestCase로 두지 않아 테스트로 수집되지 않는다."""

    class Chain(TabularWorkflowFixture):
        csv_text = HANDOFF_CSV

        def runTest(self):
            pass

    return Chain()


def rewrite_manifest(root):
    """자기 manifest만 다시 계산한 위조를 만든다."""
    manifest = json.loads((root / "HANDOFF_COMPLETE.json").read_text(encoding="utf-8"))
    manifest["files"] = {name: _fingerprint(root / name) for name in manifest["files"]}
    manifest["payload_bytes"] = sum(item["size"] for item in manifest["files"].values())
    (root / "HANDOFF_COMPLETE.json").write_text(json.dumps(manifest), encoding="utf-8")


def rehash_row(root, partition, suffix, key):
    path = root / "prepared" / f"{partition}_{suffix}"
    report = json.loads((root / "prepared/application.json").read_text(encoding="utf-8"))
    digest = _fingerprint(path)
    report["partitions"][partition][key] = {"size_bytes": digest["size"], "sha256": digest["sha256"]}
    (root / "prepared/application.json").write_text(json.dumps(report), encoding="utf-8")
    rewrite_manifest(root)


def raw_features(root, partition):
    """묶음에 기록된 문자열을 Python float로 직접 읽은 기준값."""
    with (root / "prepared" / f"{partition}_features.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    return [[float(value) if value else math.nan for value in row] for row in rows[1:]]


class TabularHandoffV2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chain = _chain()
        cls.chain.setUp()
        try:
            cls.source_run, cls.source_id = cls.chain.run_chain()
            cls.bundle = cls.chain.bundles / "handoff01"
            cls.golden = cls.chain.root / "golden"
            shutil.copytree(cls.bundle, cls.golden)
        except BaseException:
            cls.chain.doCleanups()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.chain.doCleanups()

    def copy(self, name):
        """원본 묶음의 독립 복사본을 만든다."""
        target = self.chain.root / "copies" / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(self.golden, target)
        return target

    def restore_original(self):
        shutil.rmtree(self.bundle)
        shutil.copytree(self.golden, self.bundle)

    def forged(self, name, value):
        """Train 특징 첫 값을 바꾸고 자기 manifest까지 다시 계산한 묶음 B."""
        target = self.copy(name)
        path = target / "prepared/train_features.csv"
        lines = path.read_text(encoding="utf-8").split("\n")
        lines[1] = ",".join([value] + lines[1].split(",")[1:])
        path.write_text("\n".join(lines), encoding="utf-8")
        rehash_row(target, "train", "features.csv", "features")
        return target

    def test_typed_reader_round_trip_preserves_values_and_types(self):
        bundle = self.copy("types")
        self.assertEqual(verify_bundle(bundle)["schema_version"], "tabular_preparation_handoff_v2")
        features, rows = load_handoff_partition(bundle, "train")
        saved = json.loads((bundle / "preprocessor.json").read_text(encoding="utf-8"))
        self.assertEqual(list(features.columns), saved["output_columns"])

        # float64 값은 묶음의 %.17g 문자열과 비트 단위로 같아야 한다.
        self.assertEqual(features.iloc[0, 0].hex(), (0.10000000000000002).hex())
        self.assertEqual(features.iloc[2, 0], 9.0)
        for partition, frame in (("train", features), ("validation", load_handoff_partition(bundle, "validation")[0])):
            expected = raw_features(bundle, partition)
            observed = frame.to_numpy().tolist()
            self.assertEqual(len(observed), len(expected))
            for left, right in zip(observed, expected):
                self.assertEqual([value.hex() for value in left], [value.hex() for value in right])

        self.assertEqual(rows.index.tolist(), [0, 1, 2])
        self.assertEqual(rows["label"].tolist(), ["True", "False", "True"])
        self.assertTrue(all(type(value) is str for value in rows["label"]))
        self.assertEqual([type(value) for value in rows["flag"].tolist()], [int, int, int])
        self.assertTrue(pd.api.types.is_integer_dtype(rows["flag"]))
        self.assertEqual(rows["id"].tolist(), ["0012", "0034", "0056"])

        _, metadata = load_handoff_partition(bundle, "validation")
        self.assertEqual(metadata["flag"].tolist(), [0, 1])
        self.assertEqual(metadata["id"].tolist(), ["0078", "0090"])
        provenance = json.loads((bundle / "provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(provenance["source_contract_id"], self.source_id)
        readme = (bundle / "README.md").read_text(encoding="utf-8")
        self.assertIn("load_handoff_partition", readme)
        self.assertNotIn("문자열 값", readme)

    def test_reader_restores_bool_and_missing_metadata_distinct_from_string_and_integer(self):
        # CSV 입력 추론은 bool을 만들지 않고, 분할 단계는 라벨·그룹·식별자 결측을 거부한다. 그래서 reader 계층만 검사한다.
        # 행 메타데이터를 nullable 스키마로 바꾸고 자기 봉인한 사본이다. 원래 적용 결과 대조(results)의 거부는 이 테스트 범위 밖이다.
        bundle = self.copy("bool")
        path = bundle / "prepared/train_rows.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        changes = {"label": ("boolean", (True, False, pd.NA)), "flag": ("Int64", (1, pd.NA, 1)),
                   "id": ("string", ("0012", pd.NA, "0056"))}
        for name, (dtype, values) in changes.items():
            offset = value["metadata_columns"].index(name)
            value["metadata_schema"][offset]["dtype"] = {"kind": "plain", "dtype": dtype}
            for row, item in zip(value["metadata"], values):
                row[offset] = _encode(item)
        path.write_text(json.dumps(value), encoding="utf-8")
        rehash_row(bundle, "train", "rows.json", "row_contract")

        _, rows = load_handoff_partition(bundle, "train")
        labels, flags, ids = (rows[name].tolist() for name in ("label", "flag", "id"))
        self.assertIs(labels[0], True)
        self.assertIs(labels[1], False)
        self.assertIs(labels[2], pd.NA)
        self.assertEqual(str(rows["label"].dtype), "boolean")
        self.assertIs(type(flags[0]), int)
        self.assertEqual(flags[0], 1)
        self.assertIs(flags[1], pd.NA)
        self.assertEqual(str(rows["flag"].dtype), "Int64")
        self.assertEqual(ids[0::2], ["0012", "0056"])
        self.assertIs(ids[1], pd.NA)
        _, original = load_handoff_partition(self.golden, "train")
        self.assertEqual(original["label"].tolist()[0], "True")
        self.assertIsNot(type(original["label"].tolist()[0]), bool)

    def test_reader_returns_verified_bytes_when_original_is_swapped_after_verification(self):
        # 원래 결함 순서: 검사 시작 시점에 원본을 B(자기 봉인 위조)로 바꾸고 manifest까지 교체한다.
        # reader는 먼저 고정한 사본만 검사·해석하므로 성공하더라도 A의 값만 돌려줘야 한다.
        attacker = self.forged("after_b", "0.25")
        verify_bundle(attacker)
        bundle = self.copy("after_verify")
        real = csv_handoff_contract.verify_bundle
        calls = []

        def swapped(directory):
            calls.append(Path(directory))
            shutil.rmtree(bundle)
            shutil.copytree(attacker, bundle)
            return real(directory)

        with patch.object(csv_handoff_contract, "verify_bundle", side_effect=swapped):
            features, _ = load_handoff_partition(bundle, "train")
        self.assertEqual(len(calls), 1)
        self.assertNotEqual(calls[0], bundle)
        expected, _ = load_handoff_partition(self.golden, "train")
        self.assertEqual([value.hex() for value in features.to_numpy().ravel()],
                         [value.hex() for value in expected.to_numpy().ravel()])
        self.assertEqual(features.iloc[0, 0].hex(), (0.10000000000000002).hex())
        # 교체된 원본 B는 그 뒤 읽으면 B로 읽히며, 이는 묶음 자체 검사 범위의 정상 동작이다.
        self.assertEqual(load_handoff_partition(bundle, "train")[0].iloc[0, 0], 0.25)

    def test_reader_rejects_manifest_and_payload_swaps_during_copy(self):
        attacker = self.forged("attacker_b", "inf")
        # B는 자기 manifest가 맞아도 무한값이라 단독 검사에서 거부된다.
        with self.assertRaises(ValueError):
            verify_bundle(attacker)
        real = csv_handoff_contract._path

        for mode in ("manifest_and_payload", "payload_only"):
            bundle = self.copy("swap_" + mode)
            seen = []

            def hooked(value, *, directory, bundle=bundle, mode=mode):
                path = Path(value)
                if bundle in path.parents:
                    seen.append(path.relative_to(bundle).as_posix())
                    if seen[-1] == "prepared/train_features.csv" and seen.count(seen[-1]) == 1:
                        # 완료 기록 A를 읽은 뒤, 첫 파일을 복사하기 직전에 B로 교체한다.
                        items = ("HANDOFF_COMPLETE.json", "prepared/application.json", "prepared/train_features.csv")
                        for item in items if mode == "manifest_and_payload" else items[2:]:
                            shutil.copyfile(attacker / item, bundle / item)
                return real(value, directory=directory)

            with self.subTest(mode=mode), patch.object(csv_handoff_contract, "_path", side_effect=hooked):
                with self.assertRaisesRegex(ValueError, "완료 기록과 다릅니다"):
                    load_handoff_partition(bundle, "train")
                self.assertEqual(seen[:2], ["HANDOFF_COMPLETE.json", "prepared/train_features.csv"])

    def test_reader_rejects_size_hash_and_listing_mismatch(self):
        for change in ("size", "hash", "extra"):
            bundle = self.copy("mismatch_" + change)
            path = bundle / "prepared/train_features.csv"
            data = path.read_bytes()
            if change == "size":
                path.write_bytes(data + b"\n")
            elif change == "hash":
                index = data.rindex(b"9")
                path.write_bytes(data[:index] + b"8" + data[index + 1:])
            else:
                (bundle / "prepared/test_features.csv").write_text("x\n1\n", encoding="utf-8")
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    verify_bundle(bundle)
                with self.assertRaises(ValueError):
                    load_handoff_partition(bundle, "train")

    def test_copy_tampering_is_rejected_by_bundle_verification(self):
        for change in ("extra", "missing", "string_metadata"):
            bundle = self.copy("tamper_" + change)
            if change == "extra":
                (bundle / "prepared/test_features.csv").write_text("x\n1\n", encoding="utf-8")
            elif change == "missing":
                (bundle / "feature_mapping.json").unlink()
            else:
                path = bundle / "prepared/train_rows.json"
                value = json.loads(path.read_text(encoding="utf-8"))
                value["metadata"] = [[json.dumps(item) for item in row] for row in value["metadata"]]
                path.write_text(json.dumps(value), encoding="utf-8")
                rehash_row(bundle, "train", "rows.json", "row_contract")
            with self.subTest(change=change), self.assertRaises(ValueError):
                verify_bundle(bundle)

    def test_creation_rejects_version_source_and_overwrite(self):
        chain = self.chain
        with self.assertRaises(ValueError):
            CsvWorkflowService("csv_local_handoff").preview_form(
                {"name": "legacy01", "pipeline_name": "prep01", "max_bundle_bytes": 1024 ** 2}, chain.limits)
        service = workflow_service("tabular_local_handoff", "2.0.0")
        values = chain.values("tabular_local_handoff", self.source_run, self.source_id)
        with self.assertRaises(ValueError):
            service.preview_form(dict(values, name="handoff02", source_contract_id="tabular_source:" + "2" * 64), chain.limits)
        with self.assertRaises(ValueError):
            service.preview_form(values, chain.limits)
        with self.assertRaises(ValueError):
            export_bundle("prep01", "legacy02", 1024 ** 2, confirmed=True)

    def test_results_reject_self_sealed_forgery_and_other_attempt(self):
        chain = self.chain
        service = workflow_service("tabular_local_handoff", "2.0.0")
        self.addCleanup(self.restore_original)
        self.assertTrue(service.results("handoff01"))

        # 자기 manifest까지 다시 계산한 원본 위조: 묶음 자체 검사는 통과해도 원래 적용 결과 대조가 거부한다.
        path = self.bundle / "prepared/train_features.csv"
        lines = path.read_text(encoding="utf-8").split("\n")
        lines[1] = ",".join(["0.25"] + lines[1].split(",")[1:])
        path.write_text("\n".join(lines), encoding="utf-8")
        rehash_row(self.bundle, "train", "features.csv", "features")
        verify_bundle(self.bundle)
        with self.assertRaises(ValueError):
            service.results("handoff01")

        # 같은 분할·내보내기에서 다른 규칙으로 만든 두 번째 D 시도의 묶음으로 교체
        rules = json.loads(PIPELINE.read_text(encoding="utf-8"))
        rules["steps"][0]["settings"]["numeric_imputation"] = "mean"
        other = chain.root / "mean.json"
        other.write_text(json.dumps(rules), encoding="utf-8")
        candidate = workflow_service("tabular_prepare_candidate", "2.0.0")
        candidate.create(candidate.preview_form(dict(chain.values("tabular_prepare_candidate", self.source_run, self.source_id),
                                                     name="prep02", pipeline=str(other)), chain.limits), confirmed=True)
        chain.finish("tabular_prepare_candidate", "prep02")
        apply = workflow_service("tabular_prepare_apply", "2.0.0")
        apply.create(apply.preview_form({"pipeline_name": "prep02", "source_contract_id": self.source_id}, chain.limits), confirmed=True)
        chain.finish("tabular_prepare_apply", "prep02")
        export_bundle("prep02", "handoff02", 1024 ** 2, confirmed=True, source_contract_id=self.source_id)
        self.assertEqual(verify_bundle(chain.bundles / "handoff02")["schema_version"], "tabular_preparation_handoff_v2")
        shutil.rmtree(self.bundle)
        shutil.copytree(chain.bundles / "handoff02", self.bundle)
        with self.assertRaises(ValueError):
            service.results("handoff01")


if __name__ == "__main__":
    unittest.main()
