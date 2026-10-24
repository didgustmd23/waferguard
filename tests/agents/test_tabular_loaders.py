"""합성 CSV의 확정 해석·타입 보존·snapshot과 검사 결과 결합을 확인한다."""

from dataclasses import asdict, replace
import csv
import hashlib
import json
from io import StringIO
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.agents.input_snapshot import Snapshot
from src.agents.worker_contract import WorkerLimits
from src.preprocessing import tabular_loaders as loader
from src.preprocessing.tabular_bundle import load_bundle, read_bundle_provenance


class TabularLoadersTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source, self.target = self.root / "input.bin", self.root / "table"
        self.limits = WorkerLimits(60, 2048, 2)
        self.spec = loader.CsvReadSpec()

    def prepare(self, payload, spec=None):
        self.source.write_bytes(payload)
        maximum = max(65536, len(payload))
        snapshot = Snapshot(len(payload), hashlib.sha256(payload).hexdigest(), maximum) if payload else None
        inspected = loader.inspect_csv(self.source, spec or self.spec, limits=self.limits,
                                       max_input_bytes=maximum)
        return snapshot, inspected

    def materialize(self, snapshot, inspection, spec=None, target=None, maximum=4 * 1024 * 1024):
        reference = loader.materialize_csv(self.source, spec or self.spec, target or self.target,
                                           expected_snapshot=snapshot, expected_inspection=inspection,
                                           max_output_bytes=maximum)
        return reference, load_bundle(target or self.target, expected=reference)

    def test_default_and_preserve_empty_fields(self):
        payload = b"a,b\n,NA\n1,0012\n"
        snapshot, inspection = self.prepare(payload)
        self.assertEqual(inspection["status"], "ready")
        _, frame = self.materialize(snapshot, inspection)
        self.assertEqual(str(frame.a.dtype), "Int64")
        self.assertIs(frame.a.iloc[0], pd.NA)
        self.assertEqual(frame.b.tolist(), ["NA", "0012"])
        spec = replace(self.spec, empty_field="preserve")
        snapshot, inspection = self.prepare(payload, spec)
        _, frame = self.materialize(snapshot, inspection, spec, self.root / "preserved")
        self.assertEqual(frame.a.tolist(), ["", "1"])
        self.assertEqual(str(frame.a.dtype), "string")

    def test_large_integer_and_lexical_strings_are_preserved(self):
        snapshot, inspection = self.prepare(
            b"integer,id,flag,date,huge\n9007199254740993,0012,True,2026-01-01,18446744073709551616\n"
            b"9007199254740994,0003,False,2026-01-02,18446744073709551617\n")
        _, frame = self.materialize(snapshot, inspection)
        self.assertEqual(frame.integer.tolist(), [9007199254740993, 9007199254740994])
        self.assertEqual(str(frame.integer.dtype), "int64")
        self.assertEqual(frame.id.tolist(), ["0012", "0003"])
        self.assertEqual(frame.flag.tolist(), ["True", "False"])
        self.assertEqual(frame.date.tolist(), ["2026-01-01", "2026-01-02"])
        self.assertEqual(frame.huge.iloc[0], "18446744073709551616")

    def test_uint64_and_nullable_float_keep_exact_types(self):
        snapshot, inspection = self.prepare(b"unsigned,f\n18446744073709551615,1.5\n0,\n")
        _, frame = self.materialize(snapshot, inspection)
        self.assertEqual(str(frame.unsigned.dtype), "uint64")
        self.assertEqual(frame.unsigned.iloc[0], 2**64 - 1)
        self.assertEqual(str(frame.f.dtype), "Float64")
        self.assertIs(frame.f.iloc[1], pd.NA)

    def test_long_decimal_and_mixed_float_integer_do_not_lose_precision(self):
        snapshot, inspection = self.prepare(
            b"precise,mixed\n0.123456789012345678901,9007199254740993\n0.5,1.5\n")
        _, frame = self.materialize(snapshot, inspection)
        self.assertEqual(frame.precise.tolist(), ["0.123456789012345678901", "0.5"])
        self.assertEqual(frame.mixed.tolist(), ["9007199254740993", "1.5"])
        self.assertEqual([column["precision_risk_count"] for column in inspection["columns"]], [1, 1])

    def test_all_missing_and_mixed_text_do_not_become_numeric(self):
        snapshot, inspection = self.prepare(b"missing,mixed\n,1\n,CANARY_SECRET\n")
        _, frame = self.materialize(snapshot, inspection)
        self.assertEqual(str(frame.missing.dtype), "object")
        self.assertTrue(frame.missing.isna().all())
        self.assertEqual(frame.mixed.tolist(), ["1", "CANARY_SECRET"])
        self.assertNotIn("CANARY_SECRET", json.dumps(inspection))
        column = inspection["columns"][1]
        self.assertEqual((column["numeric_success_count"], column["numeric_failure_count"]), (1, 1))

    def test_duplicate_repair_requires_exact_spec(self):
        payload = b"zero,zero,zero_1\n1,2,3\n"
        snapshot, blocked = self.prepare(payload)
        self.assertEqual(blocked["status"], "review_required")
        self.assertEqual(blocked["code"], "DUPLICATE_COLUMN_NAMES")
        with self.assertRaises(ValueError):
            self.materialize(snapshot, blocked)
        self.assertFalse(self.target.exists())
        spec = replace(self.spec, header_repair="number_duplicates")
        snapshot, ready = self.prepare(payload, spec)
        _, frame = self.materialize(snapshot, ready, spec)
        self.assertEqual(list(frame.columns), ["zero__1", "zero__2", "zero_1"])
        self.assertEqual(frame.iloc[0].tolist(), [1, 2, 3])
        self.assertEqual(ready["header_mapping"], [
            {"position": 0, "original": "zero", "renamed": "zero__1"},
            {"position": 1, "original": "zero", "renamed": "zero__2"},
        ])

    def test_quoted_newlines_escaped_quotes_bom_and_crlf(self):
        payload = b'\xef\xbb\xbfx,text\r\n1,"CANARY\r\nsecond,""quoted"""\r\n2,""\r\n'
        snapshot, inspection = self.prepare(payload, replace(self.spec, empty_field="preserve"))
        _, frame = self.materialize(snapshot, inspection, replace(self.spec, empty_field="preserve"))
        self.assertEqual(frame.text.tolist(), ['CANARY\r\nsecond,"quoted"', ""])
        self.assertEqual(frame.x.tolist(), [1, 2])
        self.assertEqual(self.source.read_bytes(), payload)

    def test_malformed_and_empty_csvs_have_fixed_codes_without_values(self):
        cases = ((b"", "CSV_EMPTY_INPUT"), (b"x\n", "CSV_EMPTY_DATA"),
                 (b"x,\n1,2\n", "EMPTY_COLUMN_NAME"), (b"x,y\n1\n", "CSV_ROW_WIDTH_MISMATCH"),
                 (b'x\n"CANARY', "CSV_MALFORMED"), (b'x\nCANARY"BAD\n', "CSV_MALFORMED"),
                 (b"x\n\xff\n", "CSV_ENCODING_INVALID"))
        for payload, code in cases:
            with self.subTest(code=code):
                _, result = self.prepare(payload)
                self.assertEqual(result["status"], "blocked_input")
                self.assertEqual(result["code"], code)
                self.assertNotIn("CANARY", json.dumps(result))

    def test_spec_rejects_extra_arguments_and_unsupported_interpretations(self):
        self.assertEqual(loader.CsvReadSpec.from_json(self.spec.to_json()), self.spec)
        for change in ({"encoding": "cp949"}, {"delimiter": "\t"}, {"header": None}, {"converters": "os.system"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                loader.CsvReadSpec.from_dict(dict(asdict(self.spec), **change))
        with self.assertRaises(ValueError):
            loader.CsvReadSpec.from_json('{"loader":"a","loader":"b"}')

    def test_original_is_unchanged_and_output_cannot_be_overwritten(self):
        payload = b"x\n1\n2\n"
        snapshot, inspection = self.prepare(payload)
        reference, _ = self.materialize(snapshot, inspection)
        before = {path.name: path.read_bytes() for path in self.target.iterdir()}
        with self.assertRaises((ValueError, FileExistsError)):
            self.materialize(snapshot, inspection)
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.target.iterdir()})
        self.assertEqual(self.source.read_bytes(), payload)
        provenance = read_bundle_provenance(self.target, expected=reference)
        self.assertEqual(provenance["inspection_id"], inspection["inspection_id"])
        self.assertEqual(provenance["read_spec"], asdict(self.spec))

    def test_source_changed_after_inspection_is_rejected_before_output(self):
        snapshot, inspection = self.prepare(b"x\n1\n")
        self.source.write_bytes(b"x\n9\n")
        with self.assertRaises(ValueError):
            self.materialize(snapshot, inspection)
        self.assertFalse(self.target.exists())

    def test_rehashed_forged_inspection_and_other_spec_are_rejected(self):
        snapshot, inspection = self.prepare(b"x\n1\n")
        forged = json.loads(json.dumps(inspection))
        forged["rows"] = 9
        payload = {key: value for key, value in forged.items() if key != "inspection_id"}
        forged["inspection_id"] = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
        with self.assertRaises(ValueError):
            self.materialize(snapshot, forged)
        with self.assertRaises(ValueError):
            self.materialize(snapshot, inspection, replace(self.spec, empty_field="preserve"))
        self.assertFalse(self.target.exists())

    def test_input_record_column_and_cell_limits_are_enforced(self):
        self.source.write_bytes(b"x\n1\n")
        result = loader.inspect_csv(self.source, self.spec, limits=self.limits, max_input_bytes=2)
        self.assertEqual(result["code"], "CSV_INPUT_TOO_LARGE")
        _, result = self.prepare(b"x\n" + b"z" * (1024 * 1024 + 1))
        self.assertEqual(result["code"], "CSV_RECORD_TOO_LARGE")
        with patch.object(loader, "MAX_CELLS", 3):
            _, result = self.prepare(b"x,y\n1,2\n3,4\n")
            self.assertEqual(result["code"], "CSV_TABLE_TOO_LARGE")
        header = ",".join(f"c{i}" for i in range(4097)).encode("ascii")
        _, result = self.prepare(header + b"\n" + b",".join([b"1"] * 4097) + b"\n")
        self.assertEqual(result["code"], "CSV_TOO_MANY_COLUMNS")

    def test_output_budget_failure_does_not_mark_completion(self):
        snapshot, inspection = self.prepare(b"x\n1\n")
        with self.assertRaises(ValueError):
            self.materialize(snapshot, inspection, maximum=16)
        self.assertFalse((self.target / "SUCCESS.json").exists())

    def test_limits_and_snapshot_types_do_not_accept_bool(self):
        self.source.write_bytes(b"x\n1\n")
        with self.assertRaises(ValueError):
            loader.inspect_csv(self.source, self.spec, limits=self.limits, max_input_bytes=True)
        snapshot, inspection = self.prepare(b"x\n1\n")
        value = asdict(snapshot)
        value["size_bytes"] = True
        with self.assertRaises(ValueError):
            loader.materialize_csv(self.source, self.spec, self.target, expected_snapshot=value,
                                   expected_inspection=inspection, max_output_bytes=65536)

    def test_hard_linked_input_is_rejected(self):
        self.source.write_bytes(b"x\n1\n")
        os.link(self.source, self.root / "alias")
        result = loader.inspect_csv(self.source, self.spec, limits=self.limits, max_input_bytes=65536)
        self.assertEqual(result["code"], "CSV_INPUT_UNSAFE")

    def test_valid_single_column_names_may_contain_other_separator_characters(self):
        snapshot, inspection = self.prepare(b'"a;b|c"\n1\n2\n')
        self.assertEqual(inspection["status"], "ready")
        _, frame = self.materialize(snapshot, inspection)
        self.assertEqual(list(frame.columns), ["a;b|c"])
        self.assertEqual(frame.iloc[:, 0].tolist(), [1, 2])

    def test_large_field_and_quoted_records_match_csv_writer_without_global_changes(self):
        rows = [["A,comma", 'B"quote', "C\r\nline"], ["D\rreturn", "E\nnext", ""], ["F" * 200000, "G한글", "H\ttext"]]
        stream = StringIO(newline="")
        writer = csv.writer(stream, lineterminator="\r\n")
        writer.writerow(["a", "b", "c"])
        writer.writerows(rows)
        before = csv.field_size_limit()
        spec = replace(self.spec, empty_field="preserve")
        snapshot, inspection = self.prepare(stream.getvalue().encode("utf-8"), spec)
        self.assertEqual(inspection["status"], "ready")
        _, frame = self.materialize(snapshot, inspection, spec)
        self.assertEqual(frame.values.tolist(), rows)
        self.assertEqual(csv.field_size_limit(), before)

    def test_expired_parser_deadline_is_reported_without_values(self):
        self.source.write_bytes(b"x\nCANARY_SECRET\n")
        with patch.object(loader.time, "monotonic", side_effect=[0, 61]):
            result = loader.inspect_csv(self.source, self.spec, limits=self.limits, max_input_bytes=65536)
        self.assertEqual(result["code"], "CSV_TIME_LIMIT")
        self.assertNotIn("CANARY_SECRET", json.dumps(result))

    def test_actual_consumed_snapshot_bytes_are_checked_after_aba_restore(self):
        snapshot, inspection = self.prepare(b"x\n1\n")
        original, real_open = self.source.read_bytes(), Path.open
        before = self.source.stat()
        armed = True

        class RestoringReader:
            def __init__(self, stream):
                self.stream, self.restored = stream, False

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.stream.close()

            def read(self, size=-1):
                payload = self.stream.read(size)
                if not self.restored:
                    with real_open(self_path, "wb") as target:
                        target.write(original)
                    os.utime(self_path, ns=(before.st_atime_ns, before.st_mtime_ns))
                    self.restored = True
                return payload

        self_path = self.source

        def swapped_open(path, mode="r", *args, **kwargs):
            nonlocal armed
            if armed and path == self_path and mode == "rb":
                armed = False
                with real_open(path, "wb") as target:
                    target.write(b"x\n9\n")
                return RestoringReader(real_open(path, mode, *args, **kwargs))
            return real_open(path, mode, *args, **kwargs)

        with patch.object(Path, "open", swapped_open), self.assertRaises(ValueError):
            self.materialize(snapshot, inspection)
        self.assertFalse(self.target.exists())
        self.assertEqual(self.source.read_bytes(), original)

    def test_duplicate_numbering_collision_returns_fixed_failure(self):
        spec = replace(self.spec, header_repair="number_duplicates")
        _, result = self.prepare(b"a,a,a_1,a__1,a___1,a____1\n1,2,3,4,5,6\n", spec)
        self.assertEqual(result["status"], "blocked_input")
        self.assertEqual(result["code"], "CSV_HEADER_REPAIR_CONFLICT")
        self.assertEqual(result["columns"], [])
        self.assertFalse(self.target.exists())

    def test_materialize_input_io_failure_does_not_expose_filename(self):
        snapshot, inspection = self.prepare(b"x\n1\n")
        self.source.unlink()
        with self.assertRaises(loader.CsvLoadError) as caught:
            self.materialize(snapshot, inspection)
        self.assertEqual(caught.exception.code, "CSV_INPUT_UNREADABLE")
        self.assertNotIn(str(self.root), str(caught.exception))
        self.assertIsNone(getattr(caught.exception, "filename", None))
        self.assertFalse(self.target.exists())

    def test_inspection_bool_float_and_integer_are_not_interchangeable(self):
        snapshot, inspection = self.prepare(b"x\n1\n")
        for index, (field, value) in enumerate((("rows", True), ("rows", 1.0), ("missing_count", False), ("numeric_success_count", True))):
            with self.subTest(field=field, value=value):
                forged = json.loads(json.dumps(inspection))
                if field == "rows":
                    forged[field] = value
                else:
                    forged["columns"][0][field] = value
                target = self.root / f"forged_{index}"
                with self.assertRaises(ValueError):
                    self.materialize(snapshot, forged, target=target)
                self.assertFalse(target.exists())

    def test_long_header_has_distinct_failure_code(self):
        _, report = self.prepare(b"a" * 257 + b"\n1\n")
        self.assertEqual(report["code"], "CSV_HEADER_TOO_LONG")
