"""전달 묶음의 생성·독립 검사·변조·상한을 합성 입력으로 검증한다."""

from contextlib import redirect_stdout
import io
import json
import shutil
from pathlib import Path
import unittest
from unittest.mock import patch

from tests.agents import test_preparation_pipeline as fixtures
from src.agents import csv_handoff_cli as cli
from src.agents import preparation_pipeline_cli as pipeline
from src.agents.csv_handoff_contract import verify_bundle, PAYLOAD_FILES
from src.agents.preparation_apply_worker import execute_request as apply_worker
from src.agents.artifacts import _fingerprint
from src.preprocessing.transform import TabularPreprocessor


class CsvHandoffTest(unittest.TestCase):
    fixture_commit = fixtures.PreparationPipelineTest.fixture_commit
    complete = fixtures.PreparationPipelineTest.complete
    transport = fixtures.PreparationPipelineTest.transport
    create_run = fixtures.PreparationPipelineTest.create_run
    finish_candidate = fixtures.PreparationPipelineTest.finish_candidate
    approve = fixtures.PreparationPipelineTest.approve
    review = fixtures.PreparationPipelineTest.review

    def setUp(self):
        # 합성 승인 안내만 숨긴다. unittest 실패·결과는 그대로 출력한다.
        redirect = redirect_stdout(io.StringIO())
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)
        fixtures.PreparationPipelineTest.setUp(self)
        self.create_run()
        self.finish_candidate()
        self.review()
        pipeline.operate("example", "next")
        pipeline.operate("example", "next")
        approval = self.approve("APPLY")

        with patch("src.agents.preparation_apply.run_apply_worker", side_effect=lambda request, root, signal: self.transport(request, root, signal, apply_worker)):
            self.assertEqual(pipeline.operate("example", "next", approval["approval_id"])["status"], "completed")

        self.handoffs = self.root / "handoffs"
        self.handoffs.mkdir()
        context = patch("src.agents.csv_handoff_cli._directory", side_effect=lambda name, **kwargs: self.handoffs / name)
        context.start()
        self.addCleanup(context.stop)

    def make_bundle(self, name="example", maximum=1024 * 1024):
        return cli.export_bundle("example", name, maximum, confirmed=True)

    def rewrite_manifest(self, root):
        """해시 정합성이 있어도 내용 계약 위반을 차단하는지 확인하는 합성 보정."""
        path = root / "HANDOFF_COMPLETE.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["files"] = {name: _fingerprint(root / name) for name in PAYLOAD_FILES}
        manifest["payload_bytes"] = sum(record["size"] for record in manifest["files"].values())
        path.write_text(json.dumps(manifest), encoding="utf-8")

    def test_roundtrip_without_refit_or_original_logs(self):
        """재변환 없이 복사하고 독립 검사는 원래 로그를 읽지 않는다."""
        before = {str(path): _fingerprint(path) for path in self.apply_root.rglob("*.csv")}

        with patch.object(TabularPreprocessor, "fit", side_effect=AssertionError("재산출")), patch.object(TabularPreprocessor, "transform", side_effect=AssertionError("재변환")):
            self.assertEqual(self.make_bundle()["status"], "handoff_created")

        with patch("src.agents.preparation_pipeline_cli.open_run", side_effect=AssertionError("출처 로그 접근")):
            self.assertEqual(verify_bundle(self.handoffs / "example")["status"], "handoff_verified")

        for path, expected in before.items():
            self.assertEqual(_fingerprint(Path(path)), expected)

        with self.assertRaises(FileExistsError):
            self.make_bundle()

    def test_modified_extra_and_missing_files(self):
        """수정·추가·누락된 파일은 모두 거부한다."""
        self.make_bundle()

        for mode in ("modified", "extra", "missing"):
            root = self.handoffs / mode
            shutil.copytree(self.handoffs / "example", root)

            if mode == "modified":
                (root / "prepared/train_features.csv").write_text("changed", encoding="utf-8")
            elif mode == "extra":
                (root / "unexpected.txt").write_text("extra", encoding="utf-8")
            else:
                (root / "preprocessor.json").unlink()

            with self.subTest(mode=mode), self.assertRaises((ValueError, OSError)):
                verify_bundle(root)

    def test_structure_rejected_after_hash_update(self):
        """해시를 갱신해도 잘못된 특징·수치·원본 매핑·출처는 거부한다."""
        self.make_bundle()

        for mode in ("columns", "nan", "mapping", "positions", "provenance"):
            root = self.handoffs / mode
            shutil.copytree(self.handoffs / "example", root)

            if mode in ("columns", "nan"):
                path = root / "prepared/train_features.csv"
                lines = path.read_text(encoding="utf-8").splitlines()

                if mode == "columns":
                    lines[0] = "wrong" + lines[0]
                else:
                    fields = lines[1].split(",")
                    fields[0] = "nan"
                    lines[1] = ",".join(fields)

                path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                report_path = root / "prepared/application.json"
                report = json.loads(report_path.read_text(encoding="utf-8"))
                fingerprint = _fingerprint(path)
                report["partitions"]["train"]["features"] = {"size_bytes": fingerprint["size"], "sha256": fingerprint["sha256"]}
                report_path.write_text(json.dumps(report), encoding="utf-8")
            elif mode == "positions":
                path = root / "prepared/train_rows.json"
                value = json.loads(path.read_text(encoding="utf-8"))
                value["row_positions"][1] = value["row_positions"][0]
                path.write_text(json.dumps(value), encoding="utf-8")
                report_path = root / "prepared/application.json"
                report = json.loads(report_path.read_text(encoding="utf-8"))
                fingerprint = _fingerprint(path)
                report["partitions"]["train"]["row_contract"] = {"size_bytes": fingerprint["size"], "sha256": fingerprint["sha256"]}
                report_path.write_text(json.dumps(report), encoding="utf-8")
            elif mode == "mapping":
                path = root / "feature_mapping.json"
                value = json.loads(path.read_text(encoding="utf-8"))
                value["features"][0]["source_column"] = "wrong"
                path.write_text(json.dumps(value), encoding="utf-8")
            else:
                path = root / "provenance.json"
                value = json.loads(path.read_text(encoding="utf-8"))
                value["preprocessor_sha256"] = "0" * 64
                path.write_text(json.dumps(value), encoding="utf-8")

            self.rewrite_manifest(root)

            with self.subTest(mode=mode), self.assertRaises(ValueError):
                verify_bundle(root)

    def test_limits_confirmation_and_cancelled_source(self):
        """명시 선택·합계 상한·취소 상태를 확인하고 부분 결과를 확정하지 않는다."""
        with self.assertRaises(ValueError):
            cli.export_bundle("example", "unconfirmed", 1024 * 1024)

        with self.assertRaises(ValueError):
            self.make_bundle("too_small", 1)

        self.assertFalse((self.handoffs / "too_small").exists())
        self.assertTrue(list(self.handoffs.glob("too_small.partial-*")))
        (self.pipeline_root / "CANCEL_REQUESTED").touch()

        with self.assertRaises(ValueError):
            self.make_bundle("cancelled")

        self.assertFalse((self.handoffs / "cancelled").exists())

    def test_unfinished_source_rejected(self):
        """후보 검토만 있는 상태를 전달 완료로 취급하지 않는다."""
        from src.agents import preparation_apply_cli as application

        tool = application.open_run("example-apply")
        state = tool.states.load()
        state.status = "running"
        tool.states.save(state, expected_revision=state.revision)

        with self.assertRaises(ValueError):
            self.make_bundle("unfinished")

        self.assertFalse((self.handoffs / "unfinished").exists())


if __name__ == "__main__":
    unittest.main()
