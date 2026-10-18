"""임시 합성 파일로 형식 독립 복사·크기 제한·변경·기존 결과 보호를 검증한다."""

from io import BytesIO
from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch

from src.agents import input_snapshot as module


class InputSnapshotTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source.csv"
        self.source.write_bytes(b"value\n1\n")
        self.output = self.root / "snapshot"

    def test_exact_limit_and_tamper_detection(self):
        """상한과 같은 크기는 허용하고 복사본 변경은 거부한다."""
        before = self.source.read_bytes()
        snapshot = module.create_snapshot(self.source, self.output, max_input_bytes=len(before))
        self.assertTrue(module.verify_snapshot(self.output, snapshot))
        self.assertEqual(self.source.read_bytes(), before)
        (self.output / module.PAYLOAD_NAME).write_bytes(b"value\n2\n")
        self.assertFalse(module.verify_snapshot(self.output, snapshot))

    def test_oversize_creates_no_snapshot(self):
        """초과 입력은 작업 폴더를 만들기 전에 거부한다."""
        with self.assertRaises(module.InputTooLarge) as caught:
            module.create_snapshot(self.source, self.output, max_input_bytes=1)

        self.assertFalse(caught.exception.started)
        self.assertFalse(self.output.exists())

    def test_stream_growth_never_writes_excess(self):
        """복사 중에도 상한을 넘는 바이트를 저장하지 않는다."""
        target = BytesIO()

        with self.assertRaises(module.InputTooLarge) as caught:
            module._scan(BytesIO(b"12345"), 4, target)

        self.assertTrue(caught.exception.started)
        self.assertLessEqual(len(target.getvalue()), 4)

    def test_existing_directory_not_overwritten(self):
        """기존 결과 폴더를 덮어쓰지 않는다."""
        self.output.mkdir()

        with self.assertRaises(ValueError):
            module.create_snapshot(self.source, self.output, max_input_bytes=100)

    def test_changed_copy_has_no_completion(self):
        """복사와 재읽기의 해시가 다르면 완료 manifest를 만들지 않는다."""
        original = module._scan

        def changed(stream, maximum, target=None):
            size, digest = original(stream, maximum, target)
            return size, "0" * 64 if target is not None else digest

        with patch.object(module, "_scan", side_effect=changed), self.assertRaises(ValueError):
            module.create_snapshot(self.source, self.output, max_input_bytes=100)

        self.assertFalse((self.output / "snapshot.json").exists())
        self.assertTrue((self.output / (module.PAYLOAD_NAME + ".part")).exists())

    def test_formats_are_copied_as_uninterpreted_bytes(self):
        """확장자·유효 형식 여부와 무관하게 바이트만 보존하며 역직렬화하지 않는다."""
        payload = b"\x00\xff\x80not-a-valid-data-format\r\n"

        for index, name in enumerate(("sample.csv", "sample.parquet", "sample.npz", "sample.pkl", "no_extension")):
            with self.subTest(name=name):
                source = self.root / name
                source.write_bytes(payload)
                output = self.root / f"format_{index}"

                # pickle 복사도 역직렬화 허용을 뜻하지 않는다.
                with patch("pickle.load", side_effect=AssertionError("역직렬화 금지")), patch("pickle.loads", side_effect=AssertionError("역직렬화 금지")):
                    snapshot = module.create_snapshot(source, output, max_input_bytes=100)
                    self.assertTrue(module.verify_snapshot(output, snapshot))

                self.assertEqual((output / "input.bin").read_bytes(), payload)
                self.assertTrue(snapshot.source_id.startswith("file_snapshot:"))

    def test_legacy_csv_snapshot_not_silently_reused(self):
        """과거 CSV 전용 결과를 새 계약으로 오인하거나 자동 변경하지 않는다."""
        snapshot = module.create_snapshot(self.source, self.output, max_input_bytes=100)
        (self.output / module.PAYLOAD_NAME).rename(self.output / "input.csv")
        self.assertFalse(module.verify_snapshot(self.output, snapshot))

    def test_invalid_limits_and_hardlink(self):
        """잘못된 상한·계약·하드 링크 입력은 거부한다."""
        for maximum in (0, -1, True):
            with self.subTest(maximum=maximum), self.assertRaises(ValueError):
                module.create_snapshot(self.source, self.output, max_input_bytes=maximum)

        with self.assertRaises(ValueError):
            module.Snapshot(1, "invalid", 10)

        linked = self.root / "linked.csv"
        os.link(self.source, linked)

        with self.assertRaises(ValueError):
            module.create_snapshot(linked, self.output, max_input_bytes=100)


if __name__ == "__main__":
    unittest.main()
