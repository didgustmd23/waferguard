"""독립 합성 파일로 산출물 완료·손상·재사용 차단을 검증한다."""

from dataclasses import fields, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.agents.artifacts import (
    ArtifactContext,
    MARKER_NAME,
    check_artifacts,
    complete_artifacts,
)


class ArtifactsTest(unittest.TestCase):
    def setUp(self):
        """실제 원본과 무관한 임시 산출물 폴더를 준비한다."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)

        self.root = Path(temporary.name)
        self.output = self.root / "result.txt"
        self.output.write_text("synthetic_result", encoding="utf-8")

        self.context = ArtifactContext(
            source_id="source_v1",
            config_id="config_v1",
            plan_id="plan_v1",
            policy_version="policy_v1",
            tool_id="test_tool",
        )

    def test_complete_and_reuse_with_nested_files(self):
        """하위 파일까지 등록하고 기존 코어 완료 표식을 그대로 보존한다."""
        nested = self.root / "train"
        nested.mkdir()
        (nested / "shard.bin").write_bytes(b"synthetic_binary")

        core_marker = self.root / "SUCCESS.json"
        core_marker.write_text('{"status": "synthetic_complete"}', encoding="utf-8")
        original = core_marker.read_bytes()

        marker = complete_artifacts(self.root, self.context)
        result = check_artifacts(self.root, self.context)

        self.assertEqual(marker.name, MARKER_NAME)
        self.assertTrue(result.reusable)
        self.assertEqual(result.code, "ARTIFACT_REUSABLE")
        self.assertEqual(core_marker.read_bytes(), original)

        with self.assertRaises(FileExistsError):
            complete_artifacts(self.root, self.context)

    def test_incomplete_and_empty_outputs(self):
        """파일만 있는 부분 결과와 비어 있는 폴더는 완료로 인정하지 않는다."""
        result = check_artifacts(self.root, self.context)

        self.assertFalse(result.reusable)
        self.assertEqual(result.code, "ARTIFACT_INCOMPLETE")

        self.output.unlink()

        with self.assertRaises(ValueError):
            complete_artifacts(self.root, self.context)

    def test_each_context_change_rejects_reuse(self):
        """원본·설정·계획·정책·도구 변경을 각각 거부한다."""
        complete_artifacts(self.root, self.context)

        for field in fields(ArtifactContext):
            context = replace(self.context, **{field.name: "changed"})

            with self.subTest(field=field.name):
                result = check_artifacts(self.root, context)

                self.assertFalse(result.reusable)
                self.assertEqual(result.code, "ARTIFACT_CONTEXT_MISMATCH")

    def test_same_size_content_change_is_detected(self):
        """크기가 같아도 내용이 달라지면 해시 검사로 재사용을 막는다."""
        complete_artifacts(self.root, self.context)
        self.output.write_text("synthetic_resulX", encoding="utf-8")

        result = check_artifacts(self.root, self.context)

        self.assertFalse(result.reusable)
        self.assertEqual(result.code, "ARTIFACT_CONTENT_MISMATCH")

    def test_added_and_missing_files(self):
        """파일 추가와 누락 모두 등록된 파일 목록 불일치로 거부한다."""
        complete_artifacts(self.root, self.context)
        extra = self.root / "unexpected.bin"
        extra.write_bytes(b"extra")

        self.assertEqual(
            check_artifacts(self.root, self.context).code,
            "ARTIFACT_FILE_SET_MISMATCH",
        )

        extra.unlink()
        self.output.unlink()

        self.assertEqual(
            check_artifacts(self.root, self.context).code,
            "ARTIFACT_FILE_SET_MISMATCH",
        )

    def test_invalid_marker_and_parent_traversal(self):
        """잘못된 표식과 폴더 밖을 가리키는 파일 참조를 거부한다."""
        marker = complete_artifacts(self.root, self.context)
        record = json.loads(marker.read_text(encoding="utf-8"))
        fingerprint = record["files"].pop("result.txt")
        record["files"]["../outside.bin"] = fingerprint
        marker.write_text(json.dumps(record), encoding="utf-8")

        self.assertFalse(check_artifacts(self.root, self.context).reusable)

        marker.write_text("invalid_json", encoding="utf-8")

        self.assertFalse(check_artifacts(self.root, self.context).reusable)

    def test_duplicate_json_keys_and_invalid_size(self):
        """중복 키와 bool 형식의 파일 크기를 거부한다."""
        marker = complete_artifacts(self.root, self.context)
        record = json.loads(marker.read_text(encoding="utf-8"))
        record["files"]["result.txt"]["size"] = True
        marker.write_text(json.dumps(record), encoding="utf-8")

        self.assertEqual(
            check_artifacts(self.root, self.context).code,
            "ARTIFACT_INVALID_MARKER",
        )

        marker.write_text('{"version":1,"version":1}', encoding="utf-8")

        self.assertFalse(check_artifacts(self.root, self.context).reusable)

    def test_marker_only_does_not_pass(self):
        """완료 표식만 남아 있어도 실제 파일이 없으면 거부한다."""
        complete_artifacts(self.root, self.context)
        self.output.unlink()

        self.assertFalse(check_artifacts(self.root, self.context).reusable)

    def test_failed_publication_leaves_no_marker(self):
        """완료 표식 게시 실패 시 임시 파일을 정리하고 미완료로 유지한다."""
        with patch("src.agents.artifacts.os.link", side_effect=OSError("synthetic")):
            with self.assertRaises(OSError):
                complete_artifacts(self.root, self.context)

        self.assertEqual({path.name for path in self.root.iterdir()}, {"result.txt"})
        self.assertFalse(check_artifacts(self.root, self.context).reusable)

    def test_symbolic_link_is_rejected(self):
        """링크가 포함된 산출물을 등록하지 않는다. OS 권한이 없으면 건너뛴다."""
        link = self.root / "linked.txt"

        try:
            link.symlink_to(self.output)
        except OSError:
            self.skipTest("이 환경에서 심볼릭 링크 생성 권한이 없습니다.")

        with self.assertRaises(ValueError):
            complete_artifacts(self.root, self.context)


if __name__ == "__main__":
    unittest.main()
