"""v2 표형 묶음의 자료형·행 계보·파일 계약을 합성 입력으로 검사한다."""

from dataclasses import asdict, replace
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.preprocessing import tabular_bundle as bundle


class TabularBundleTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / "table"

    def write(self, frame, **options):
        return bundle.write_bundle(frame, self.directory, provenance={"purpose": "synthetic"},
                                   max_output_bytes=4 * 1024 * 1024, **options)

    def reseal(self):
        """manifest를 재계산해도 의미 계약 위반을 통과시키지 않는다."""
        files = {}
        for name in ("schema.json", "data.jsonl", "provenance.json"):
            payload = (self.directory / name).read_bytes()
            files[name] = {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}
        manifest = {"version": "tabular_bundle_manifest_v2", "files": files}
        payload = json.dumps(manifest, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
        (self.directory / "manifest.json").write_bytes(payload)
        (self.directory / "SUCCESS.json").write_text(json.dumps({
            "version": "tabular_bundle_complete_v2", "manifest_sha256": hashlib.sha256(payload).hexdigest(),
        }), encoding="utf-8")

    def test_preserves_types_missing_categories_timezone_and_duplicate_index(self):
        frame = pd.DataFrame({
            "n": pd.Series([9007199254740993, pd.NA, -3], dtype="Int64"),
            "u": pd.Series([2**64 - 1, 2, 0], dtype="uint64"),
            "flag": pd.Series([True, pd.NA, False], dtype="boolean"),
            "text": pd.Series(["0012", "", pd.NA], dtype="string"),
            "mixed": pd.Series([None, pd.NA, float("nan")], dtype=object),
            "numbers": [float("inf"), -float("inf"), -0.0],
            "category": pd.Categorical(["b", None, "a"], categories=["b", "a", "unused"], ordered=True),
            "time": pd.date_range("2026-01-01", periods=3, tz="Asia/Seoul"),
        })
        frame.index = pd.Index([8, 8, 3], name="original")
        before = frame.copy(deep=True)
        reference = self.write(frame)
        restored = bundle.load_bundle(self.directory, expected=reference)
        pd.testing.assert_frame_equal(restored, frame)
        pd.testing.assert_frame_equal(frame, before)
        self.assertEqual(restored["n"].iloc[0], 9007199254740993)
        self.assertIsNone(restored["mixed"].iloc[0])
        self.assertIs(restored["mixed"].iloc[1], pd.NA)
        self.assertTrue(np.isnan(restored["mixed"].iloc[2]))
        self.assertTrue(np.signbit(restored["numbers"].iloc[2]))
        self.assertEqual([row[0] for row in bundle.iter_bundle_rows(self.directory, expected=reference)], [0, 1, 2])
        self.assertEqual(bundle.BundleRef.from_dict(asdict(reference)), reference)

    def test_subset_preserves_original_positions_and_selected_columns(self):
        frame = pd.DataFrame({"x": [3, 5, 7, 9], "label": [True, False, True, False]},
                             index=pd.Index([8, 8, 2, 1], name="source"))
        reference = self.write(frame)
        target = self.root / "train"
        selected = bundle.write_subset(self.directory, [1, 3], target, expected=reference,
                                       provenance={"partition": "train"}, max_output_bytes=65536)
        pd.testing.assert_frame_equal(bundle.load_bundle(target, expected=selected), frame.iloc[[1, 3]])
        self.assertEqual([row[0] for row in bundle.iter_bundle_rows(target, expected=selected)], [1, 3])
        pd.testing.assert_frame_equal(bundle.load_bundle(target, expected=selected, columns=["label"]),
                                      frame.iloc[[1, 3]][["label"]])

    def test_requires_external_reference_and_rejects_rehashed_changes(self):
        reference = self.write(pd.DataFrame({"x": [1, 2]}))
        path = self.directory / "data.jsonl"
        path.write_text(path.read_text(encoding="utf-8").replace('"values":[1]', '"values":[9]'), encoding="utf-8")
        self.reseal()
        with self.assertRaises(ValueError):
            bundle.load_bundle(self.directory, expected=reference)
        with self.assertRaises((ValueError, TypeError)):
            bundle.load_bundle(self.directory, expected=None)

    def test_rejects_invalid_scalar_even_when_manifest_matches(self):
        self.write(pd.DataFrame({"x": [1]}))
        (self.directory / "data.jsonl").write_text(
            '{"position":0,"index":0,"values":[{"$scalar":"import","value":"os"}]}\n', encoding="utf-8")
        self.reseal()
        with self.assertRaises(ValueError):
            bundle.inspect_bundle(self.directory, expected=None, max_bytes=65536)

    def test_rejects_schema_dtype_row_count_version_and_duplicate_keys(self):
        reference = self.write(pd.DataFrame({"x": [1, 2]}))
        original = (self.directory / "schema.json").read_text(encoding="utf-8")
        for key, value in (("rows", 3), ("version", "canonical_table_v1")):
            with self.subTest(key=key):
                schema = json.loads(original)
                schema[key] = value
                (self.directory / "schema.json").write_text(json.dumps(schema), encoding="utf-8")
                self.reseal()
                with self.assertRaises(ValueError):
                    bundle.inspect_bundle(self.directory, expected=None, max_bytes=65536)
        (self.directory / "schema.json").write_text('{"rows":1,"rows":2}', encoding="utf-8")
        self.reseal()
        with self.assertRaises(ValueError):
            bundle.inspect_bundle(self.directory, expected=None, max_bytes=65536)
        with self.assertRaises(ValueError):
            replace(reference, rows=True)

    def test_rejects_scalar_that_does_not_match_declared_dtype(self):
        self.write(pd.DataFrame({"x": [1]}))
        path = self.directory / "schema.json"
        schema = json.loads(path.read_text(encoding="utf-8"))
        schema["columns"][0]["dtype"] = {"kind": "plain", "dtype": "bool"}
        path.write_text(json.dumps(schema), encoding="utf-8")
        self.reseal()
        with self.assertRaises(ValueError):
            bundle.inspect_bundle(self.directory, expected=None, max_bytes=65536)

    def test_rejects_unsupported_objects_and_never_marks_partial_output_complete(self):
        with self.assertRaises(ValueError):
            self.write(pd.DataFrame({"x": [Decimal("1.2")]}))
        self.assertFalse((self.directory / "SUCCESS.json").exists())
        target = self.root / "limited"
        with self.assertRaises(ValueError):
            bundle.write_bundle(pd.DataFrame({"x": ["a" * 2000]}), target,
                                provenance={}, max_output_bytes=128)
        self.assertFalse((target / "SUCCESS.json").exists())

    def test_rejects_extra_files_overwrite_and_oversized_record(self):
        reference = self.write(pd.DataFrame({"x": [1]}))
        with self.assertRaises((ValueError, FileExistsError)):
            self.write(pd.DataFrame({"x": [2]}))
        self.assertEqual(bundle.inspect_bundle(self.directory, expected=reference, max_bytes=65536), reference)
        (self.directory / "extra.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            bundle.inspect_bundle(self.directory, expected=reference, max_bytes=65536)
        target = self.root / "oversized"
        with self.assertRaises(ValueError):
            bundle.write_bundle(pd.DataFrame({"x": ["a" * (1024 * 1024)]}), target,
                                provenance={}, max_output_bytes=4 * 1024 * 1024)
        self.assertFalse((target / "SUCCESS.json").exists())

    def test_rejects_invalid_positions_and_schema_types(self):
        for frame in (pd.DataFrame({"x": [1], "y": [2]}).set_axis(["x", "x"], axis=1),
                      pd.DataFrame({"x": [1 + 2j]}), pd.DataFrame({"x": [1]}, index=pd.MultiIndex.from_tuples([(1, 2)]))):
            with self.subTest(dtype=str(frame.dtypes)):
                with self.assertRaises(ValueError):
                    self.write(frame)
        reference = self.write(pd.DataFrame({"x": [1, 2]}))
        for positions in ([0, 0], [1, 0], [False], [7]):
            with self.subTest(positions=positions):
                with self.assertRaises(ValueError):
                    bundle.write_subset(self.directory, positions, self.root / ("bad" + str(len(positions)) + str(positions[0])),
                                        expected=reference, provenance={}, max_output_bytes=65536)

    def test_read_metadata_cannot_return_bytes_swapped_after_validation(self):
        reference = self.write(pd.DataFrame({"x": [1]}))
        original_inspect = bundle.inspect_bundle

        def verified_then_swap(*args, **kwargs):
            result = original_inspect(*args, **kwargs)
            path = self.directory / "schema.json"
            schema = json.loads(path.read_text(encoding="utf-8"))
            schema["rows"] = 999
            path.write_text(json.dumps(schema), encoding="utf-8")
            return result

        with patch.object(bundle, "inspect_bundle", side_effect=verified_then_swap):
            with self.assertRaises(ValueError):
                bundle.read_bundle_schema(self.directory, expected=reference)

    def test_named_column_axis_and_datetime_index_roundtrip(self):
        frame = pd.DataFrame({"x": [1, 2, 3]}, index=pd.date_range("2026-01-01", periods=3, tz="Asia/Seoul", name="time"))
        frame.columns.name = "fields"
        reference = self.write(frame)
        pd.testing.assert_frame_equal(bundle.load_bundle(self.directory, expected=reference), frame)

    def test_first_row_is_rejected_when_consumed_bytes_are_swapped_and_restored(self):
        reference = self.write(pd.DataFrame({"x": [1]}))
        data_path = self.directory / "data.jsonl"
        original = data_path.read_bytes()
        changed = original.replace(b'"values":[1]', b'"values":[9]')
        self.assertNotEqual(original, changed)
        real_inspect, real_open = bundle.inspect_bundle, Path.open
        armed = False

        class RestoringReader:
            def __init__(self, stream):
                self.stream, self.restored = stream, False

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.stream.close()

            def read(self, size=-1):
                return self.consume(self.stream.read, size)

            def readline(self, size=-1):
                return self.consume(self.stream.readline, size)

            def consume(self, reader, size):
                payload = reader(size)
                if not self.restored:
                    with real_open(data_path, "wb") as target:
                        target.write(original)
                    self.restored = True
                return payload

        def inspect_then_arm(*args, **kwargs):
            nonlocal armed
            result = real_inspect(*args, **kwargs)
            armed = True
            return result

        def swapped_open(path, mode="r", *args, **kwargs):
            nonlocal armed
            if armed and path == data_path and mode == "rb":
                armed = False
                with real_open(data_path, "wb") as target:
                    target.write(changed)
                return RestoringReader(real_open(path, mode, *args, **kwargs))
            return real_open(path, mode, *args, **kwargs)

        with patch.object(bundle, "inspect_bundle", side_effect=inspect_then_arm), patch.object(Path, "open", swapped_open):
            with self.assertRaises(ValueError):
                next(bundle.iter_bundle_rows(self.directory, expected=reference))
        self.assertEqual(data_path.read_bytes(), original)

    def test_final_marker_budget_failure_leaves_no_completion_marker(self):
        frame = pd.DataFrame({"x": [1]})
        reference = self.write(frame)
        target = self.root / "limited_marker"
        with self.assertRaises(ValueError):
            bundle.write_bundle(frame, target, provenance={"purpose": "synthetic"},
                                max_output_bytes=reference.size_bytes - 1)
        self.assertFalse((target / "SUCCESS.json").exists())

    def test_worker_boundary_checks_only_granted_folders(self):
        """worker 경계는 부모가 권한을 준 input·output/result·scratch/task 아래만 허용하고 그 위는 조회하지 않는다."""
        import shutil
        import subprocess
        root = self.root / "worker"
        for name in ("input", "output/result", "scratch/task", "tools"):
            (root / name).mkdir(parents=True)
        reference = bundle.write_bundle(pd.DataFrame({"x": [1, 2]}), root / "input/table",
                                        provenance={"purpose": "synthetic"}, max_output_bytes=65536)
        shutil.copytree(root / "input/table", root / "tools/table")
        junction = root / "output/result/link"
        if subprocess.run(["cmd", "/c", "mklink", "/J", str(junction), str(root / "input/table")],
                          capture_output=True).returncode:
            self.skipTest("이 환경에서는 junction을 만들 수 없습니다")
        self.addCleanup(lambda: junction.exists() and os.rmdir(junction))
        real_lstat = Path.lstat
        # 실제 AppContainer ACL처럼 작업 root·조상과 권한 없는 중간 폴더 output·scratch 조회를 거부한다.
        forbidden = {root, *root.parents, root / "output", root / "scratch"}

        def restricted(path, *args, **kwargs):
            if path in forbidden:
                raise PermissionError("모의 허용 범위 밖")
            return real_lstat(path, *args, **kwargs)

        with patch.object(Path, "lstat", restricted), bundle.worker_filesystem(root):
            subset = bundle.write_subset(root / "input/table", [1], root / "output/result/part", expected=reference,
                                         provenance={"purpose": "synthetic"}, max_output_bytes=65536)
            self.assertEqual(bundle.load_bundle(root / "output/result/part", expected=subset)["x"].tolist(), [2])
            # 권한 폴더 밖의 사본과 권한 폴더 안의 재분석 지점은 계속 거부한다.
            for directory in (root / "tools/table", junction):
                with self.subTest(directory=directory.name), self.assertRaises(ValueError):
                    bundle.inspect_bundle(directory, expected=reference, max_bytes=65536)

        # 권한 폴더 자체가 재분석 지점으로 바뀌어도 거부한다(권한 폴더까지는 검사한다).
        replaced = self.root / "replaced"
        (replaced / "output").mkdir(parents=True)
        subprocess.run(["cmd", "/c", "mklink", "/J", str(replaced / "output/result"), str(root / "tools")],
                       capture_output=True, check=True)
        self.addCleanup(lambda: os.rmdir(replaced / "output/result"))
        with bundle.worker_filesystem(replaced), self.assertRaises(ValueError):
            bundle.inspect_bundle(replaced / "output/result/table", expected=reference, max_bytes=65536)

    def test_rejects_hard_linked_payload(self):
        reference = self.write(pd.DataFrame({"x": [1]}))
        os.link(self.directory / "data.jsonl", self.root / "alias")
        with self.assertRaises(ValueError):
            bundle.inspect_bundle(self.directory, expected=reference, max_bytes=65536)
