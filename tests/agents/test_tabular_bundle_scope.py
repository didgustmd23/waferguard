"""공개 검증 호출 한 번 안에서만 같은 표형 묶음의 전체 행 검증을 재사용하는지 확인한다."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.preprocessing import tabular_bundle as bundle


class VerificationScopeTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / "table"
        self.frame = pd.DataFrame({"x": [1, 2, 3], "label": pd.Series(["0012", "a", pd.NA], dtype="string")})
        self.reference = bundle.write_bundle(self.frame, self.directory, provenance={"purpose": "synthetic"},
                                             max_output_bytes=65536)
        self.full = 0
        original = bundle._rows

        def counted(root, schema, *, verified_stream=None):
            if verified_stream is None:
                self.full += 1
            return original(root, schema, verified_stream=verified_stream)

        patcher = patch.object(bundle, "_rows", counted)
        patcher.start()
        self.addCleanup(patcher.stop)

    def reseal(self, directory):
        files = {}
        for name in ("schema.json", "data.jsonl", "provenance.json"):
            payload = (directory / name).read_bytes()
            files[name] = {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}
        payload = json.dumps({"version": "tabular_bundle_manifest_v2", "files": files}, sort_keys=True).encode("utf-8")
        (directory / "manifest.json").write_bytes(payload)
        (directory / "SUCCESS.json").write_text(json.dumps({"version": "tabular_bundle_complete_v2",
                                                            "manifest_sha256": hashlib.sha256(payload).hexdigest()}),
                                                encoding="utf-8")

    def reads(self):
        schema = bundle.read_bundle_schema(self.directory, expected=self.reference)
        provenance = bundle.read_bundle_provenance(self.directory, expected=self.reference)
        rows = list(bundle.iter_bundle_rows(self.directory, expected=self.reference))
        frame = bundle.load_bundle(self.directory, expected=self.reference)
        return schema, provenance, rows, frame

    def test_scope_validates_each_bundle_once_with_same_results(self):
        outside = self.reads()
        unscoped = self.full
        self.full = 0
        with bundle.verification_scope():
            inside = self.reads()
            with bundle.verification_scope():
                bundle.inspect_bundle(self.directory, expected=self.reference, max_bytes=self.reference.size_bytes)
        self.assertEqual(self.full, 1)
        self.assertGreater(unscoped, 1)
        self.assertEqual(outside[:3], inside[:3])
        pd.testing.assert_frame_equal(outside[3], inside[3])
        self.assertEqual(inside[2][0][2], [1, "0012"])

    def test_no_reuse_between_public_calls_or_after_errors(self):
        with bundle.verification_scope():
            bundle.read_bundle_schema(self.directory, expected=self.reference)
        with bundle.verification_scope():
            bundle.read_bundle_schema(self.directory, expected=self.reference)
        bundle.read_bundle_schema(self.directory, expected=self.reference)
        self.assertEqual(self.full, 3)
        with self.assertRaises(RuntimeError):
            with bundle.verification_scope():
                raise RuntimeError("stop")
        self.assertIsNone(bundle._VERIFIED.get())

    def test_changed_bytes_are_rejected_inside_scope(self):
        path = self.directory / "data.jsonl"
        original = path.read_bytes()
        for label, change in (("resealed", True), ("same_size", False)):
            with self.subTest(label=label), bundle.verification_scope():
                bundle.read_bundle_schema(self.directory, expected=self.reference)
                tampered = original.replace(b'"values":[1,', b'"values":[9,')
                self.assertNotEqual(tampered, original)
                self.assertEqual(len(tampered), len(original))
                path.write_bytes(tampered)
                if change:
                    self.reseal(self.directory)
                with self.assertRaises(ValueError):
                    bundle.inspect_bundle(self.directory, expected=self.reference, max_bytes=self.reference.size_bytes)
                with self.assertRaises(ValueError):
                    list(bundle.iter_bundle_rows(self.directory, expected=self.reference))
            shutil.rmtree(self.directory)
            self.reference = bundle.write_bundle(self.frame, self.directory, provenance={"purpose": "synthetic"},
                                                 max_output_bytes=65536)

    def test_other_reference_limit_path_or_production_check_are_not_reused(self):
        copy = self.root / "copy"
        shutil.copytree(self.directory, copy)
        with bundle.verification_scope():
            bundle.inspect_bundle(self.directory, expected=self.reference, max_bytes=self.reference.size_bytes)
            for other in (replace(self.reference, rows=2), replace(self.reference, completion_sha256="0" * 64),
                          replace(self.reference, size_bytes=self.reference.size_bytes + 1)):
                with self.subTest(other=other), self.assertRaises(ValueError):
                    bundle.inspect_bundle(self.directory, expected=other, max_bytes=self.reference.size_bytes + 1)
            # 다른 기대 지문은 행 검증 전에 거부되고 재사용 기록에도 남지 않는다.
            self.assertEqual(self.full, 1)
            bundle.inspect_bundle(self.directory, expected=self.reference, max_bytes=self.reference.size_bytes + 5)
            bundle.inspect_bundle(copy, expected=self.reference, max_bytes=self.reference.size_bytes)
            bundle.inspect_bundle(self.directory, expected=None, max_bytes=self.reference.size_bytes)
            bundle.inspect_bundle(self.directory, expected=None, max_bytes=self.reference.size_bytes)
            self.assertEqual(self.full, 5)
            with self.assertRaises(ValueError):
                bundle.inspect_bundle(self.directory, expected=self.reference, max_bytes=self.reference.size_bytes - 1)

    def test_scoped_result_is_bound_to_path_reference_condition_and_bytes(self):
        computed = []

        def result(directory=None, expected=None, condition=("positions",)):
            # 계산은 키와 같은 경로·기대 지문으로만 읽는다(scoped_result의 사용 계약).
            directory, expected = directory or self.directory, expected or self.reference

            def compute():
                computed.append(1)
                return tuple(row[0] for row in bundle.iter_bundle_rows(directory, expected=expected))

            return bundle.scoped_result(directory, expected=expected, condition=condition, compute=compute)

        copy = self.root / "copy"
        shutil.copytree(self.directory, copy)
        with bundle.verification_scope():
            self.assertEqual(result(), (0, 1, 2))
            self.assertEqual(result(), (0, 1, 2))
            self.assertEqual(len(computed), 1)
            # 검사 조건·경로가 다르면 공유하지 않는다.
            result(condition=("other",))
            result(directory=copy)
            self.assertEqual(len(computed), 3)
            # 다른 기대 지문은 그 지문으로 읽는 계산에서 거부되고 기록되지 않는다.
            with self.assertRaises(ValueError):
                result(expected=replace(self.reference, completion_sha256="0" * 64))
            self.assertEqual(len(computed), 4)
            # 재사용 때도 현재 파일 바이트가 전체 검증 때와 같아야 한다.
            path = self.directory / "data.jsonl"
            path.write_bytes(path.read_bytes().replace(b'"values":[1,', b'"values":[9,'))
            with self.assertRaises(ValueError):
                result()
            self.assertEqual(len(computed), 4)
        # 범위 밖에서는 매번 계산한다.
        result(directory=copy)
        result(directory=copy)
        self.assertEqual(len(computed), 6)

    def test_scoped_result_does_not_record_failures(self):
        calls = []

        def failing():
            calls.append(1)
            raise ValueError("bad value")

        with bundle.verification_scope():
            for _ in range(2):
                with self.assertRaises(ValueError):
                    bundle.scoped_result(self.directory, expected=self.reference, condition=("x",), compute=failing)
        self.assertEqual(len(calls), 2)
        with self.assertRaises(ValueError):
            bundle.scoped_result(self.directory, expected=None, condition=("x",), compute=lambda: True)

    def test_worker_filesystem_never_reuses(self):
        worker = self.root / "worker"
        target = worker / "input" / "table"
        target.parent.mkdir(parents=True)
        shutil.copytree(self.directory, target)
        with bundle.verification_scope(), bundle.worker_filesystem(worker):
            bundle.read_bundle_schema(target, expected=self.reference)
            bundle.read_bundle_schema(target, expected=self.reference)
            calls = []
            for _ in range(2):
                bundle.scoped_result(target, expected=self.reference, condition=("x",), compute=lambda: calls.append(1))
        self.assertEqual(self.full, 2)
        self.assertEqual(len(calls), 2)

    def test_public_entry_points_open_one_scope(self):
        from src.agents import csv_workflow_creation as creation, preparation_apply_contract as contract
        from src.agents import tabular_workflow_service as service
        seen = []
        with patch.object(creation, "_checked_pipeline", lambda *a, **k: seen.append(bundle._VERIFIED.get())), \
                patch.object(contract, "_read_candidate_evidence", lambda *a: seen.append(bundle._VERIFIED.get())), \
                patch.object(service, "_report_sources", lambda *a: seen.append(bundle._VERIFIED.get())):
            creation._pipeline("x", applied=True, typed=True)
            contract.read_candidate_evidence("x", "actor")
            service.report_sources({}, "source")
        self.assertEqual(seen, [{}, {}, {}])
        self.assertIsNot(seen[0], seen[1])
        self.assertIsNot(seen[1], seen[2])
        self.assertIsNone(bundle._VERIFIED.get())

    def test_service_open_shares_one_scope_per_call_and_releases_it(self):
        from src.agents.tabular_workflow_service import TabularWorkflowService
        # 표형 서비스는 CSV 서비스의 _open을 그대로 상속한다.
        service = TabularWorkflowService("tabular_prepare_apply")
        seen, opened = [], []

        def opened_body(name):
            # 실행 열기 안의 중첩 공개 검증(_pipeline·후보 증거 읽기)은 같은 범위를 쓴다.
            with bundle.verification_scope():
                seen.append(bundle._VERIFIED.get())
            bundle.inspect_bundle(self.directory, expected=self.reference, max_bytes=self.reference.size_bytes)
            bundle.read_bundle_schema(self.directory, expected=self.reference)
            seen.append(bundle._VERIFIED.get())
            if name == "tampered":
                path = self.directory / "data.jsonl"
                path.write_bytes(path.read_bytes().replace(b'"values":[1,', b'"values":[9,'))
                bundle.inspect_bundle(self.directory, expected=self.reference, max_bytes=self.reference.size_bytes)
            opened.append(name)
            return "cli", "tool", "manager"

        with patch.object(service, "_opened", opened_body):
            self.assertEqual(service._open("first"), ("cli", "tool", "manager"))
            # 반환 뒤의 승인 기록·실행·결과 검증은 범위 밖에서 돈다.
            self.assertIsNone(bundle._VERIFIED.get())
            service._open("second")
            # 범위 안에서 묶음이 바뀌면 재사용하지 않고 거부하며, 예외 뒤에도 범위가 남지 않는다.
            with self.assertRaises(ValueError):
                service._open("tampered")
            self.assertIsNone(bundle._VERIFIED.get())
        self.assertIs(seen[0], seen[1])
        self.assertIsNot(seen[1], seen[2])
        self.assertEqual(opened, ["first", "second"])
        # 호출마다 새 범위라 같은 묶음도 호출마다 한 번씩 전체 검증한다(호출 사이 재사용 없음).
        self.assertEqual(self.full, 3)

    def test_create_step_shares_one_scope_and_releases_it(self):
        from src.agents import tabular_v2_task as v2
        seen = []

        def body(*args):
            with bundle.verification_scope():
                seen.append(bundle._VERIFIED.get())
            seen.append(bundle._VERIFIED.get())
            if args[2] == "fail":
                raise ValueError("stop")
            return "created"

        with patch.object(v2, "_create_step", body):
            self.assertEqual(v2.create_step({}, None, "apply", "name"), "created")
            # 생성이 끝나면 승인 발급·실행(_drive)은 범위 밖에서 돈다.
            self.assertIsNone(bundle._VERIFIED.get())
            with self.assertRaises(ValueError):
                v2.create_step({}, None, "fail", "name")
        self.assertIs(seen[0], seen[1])
        self.assertIs(seen[2], seen[3])
        self.assertIsNot(seen[0], seen[2])
        self.assertIsNone(bundle._VERIFIED.get())

    def test_report_sources_shares_one_scope_and_releases_it_after_errors(self):
        from src.agents import tabular_workflow_service as service
        scopes = []

        def nested(*args):
            # 본문 안의 _pipeline 같은 중첩 공개 호출은 바깥(report_sources) 범위를 그대로 쓴다.
            with bundle.verification_scope():
                scopes.append(bundle._VERIFIED.get())
            bundle.inspect_bundle(self.directory, expected=self.reference, max_bytes=self.reference.size_bytes)
            bundle.read_bundle_schema(self.directory, expected=self.reference)
            scopes.append(bundle._VERIFIED.get())
            raise ValueError("stop")

        with patch.object(service, "_report_sources", nested), self.assertRaises(ValueError):
            service.report_sources({}, "source")
        self.assertIs(scopes[0], scopes[1])
        self.assertEqual(self.full, 1)
        self.assertIsNone(bundle._VERIFIED.get())
        # 예외 뒤 다음 공개 호출은 새로 전체 검증한다.
        bundle.read_bundle_schema(self.directory, expected=self.reference)
        self.assertEqual(self.full, 2)


if __name__ == "__main__":
    unittest.main()
