"""합성 산출물만 사용하는 canonical 검증기 회귀 검사."""

import json
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.wm_sandbox.export_canonical import export_frame
from src.wafer_dl.validate_canonical import (
    CLASS_NAMES, ValidationError, decode_json, validate_bundle, write_validation_report,
    split_review_checks, print_split_review, main, read_json,
)


# ==========================================
# 학습 없이 정상·손상·중복 입력 검사
# - 실제 원본 pickle·사용자 산출물은 읽지 않음
# - 매 테스트마다 독립 임시 폴더의 작은 맵만 생성
# ==========================================
class CanonicalValidationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'canonical'
        rows = []
        for index in range(27):
            # 서로 다른 내용의 맵과 클래스별 독립 Lot 3개를 만든다.
            array = np.ones((2, 4), dtype=np.uint8)
            for bit in range(5):
                array.flat[bit] = 2 if index & (1 << bit) else 1
            rows.append({'waferMap': array, 'failureType': CLASS_NAMES[index // 3],
                         'lotName': f'lot_{index}', 'waferIndex': 1})
        self.frame = pd.DataFrame(rows)
        self.version = 'wm811k_' + 'a' * 64
        self.export(self.frame)

    def export(self, frame):
        # 변환기의 CLI 대신 순수 함수를 사용해 pickle 역직렬화를 배제한다.
        export_frame(frame, self.root, self.version, shard_rows=10)
        self.write_json('SUCCESS.json', {'status': 'canonical_export_complete', 'training_ready': False})
        self.write_json('provenance.json', {'source_sha256': 'a' * 64, 'source_bytes': 100})

    def write_json(self, name, value):
        (self.root / name).write_text(json.dumps(value), encoding='utf-8')

    def read_rows(self):
        return [json.loads(line) for line in (self.root / 'manifest.jsonl').read_text().splitlines()]

    def write_rows(self, rows):
        (self.root / 'manifest.jsonl').write_text(
            ''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')

    def test_valid_bundle_and_no_input_changes(self):
        """정상 산출물 통과와 읽기 전용 동작·진행률을 확인한다."""
        before = (self.root / 'manifest.jsonl').read_bytes()
        progress = []
        result = validate_bundle(self.root, progress=lambda done, total: progress.append((done, total)))
        self.assertTrue(result['integrity_passed'])
        self.assertFalse(result['training_ready'])
        self.assertEqual(result['split_assessment']['status'], 'preconditions_passed')
        self.assertIsNone(result['split_assessment']['split_feasible'])
        self.assertEqual(progress[-1], (3, 3))
        self.assertEqual(before, (self.root / 'manifest.jsonl').read_bytes())

    def test_caller_owned_validation_workspace(self):
        """명시한 빈 scratch를 사용해도 기존 검증 결과와 입력 보존이 동일하다."""
        expected = validate_bundle(self.root)
        workspace = Path(self.temporary.name) / "scratch"
        workspace.mkdir()
        actual = validate_bundle(self.root, workspace=workspace)
        self.assertEqual(actual, expected)
        self.assertTrue((workspace / "manifest.sqlite").is_file())

        with self.assertRaisesRegex(ValidationError, "INVALID_VALIDATION_WORKSPACE"):
            validate_bundle(self.root, workspace=workspace)

        inside = self.root / "scratch"
        inside.mkdir()

        with self.assertRaisesRegex(ValidationError, "INVALID_VALIDATION_WORKSPACE"):
            validate_bundle(self.root, workspace=inside)

    def test_completion_and_row_count(self):
        """완료 표식 누락과 행 누락을 거부한다."""
        (self.root / 'SUCCESS.json').unlink()
        with self.assertRaises(ValidationError):
            validate_bundle(self.root)
        self.write_json('SUCCESS.json', {'status': 'canonical_export_complete', 'training_ready': False})
        self.write_rows(self.read_rows()[:-1])
        with self.assertRaisesRegex(ValidationError, 'ROW_COUNT_MISMATCH'):
            validate_bundle(self.root)

    def test_path_and_label_contracts(self):
        """폴더 밖 참조와 클래스 번호 오연결을 거부한다."""
        rows = self.read_rows()
        original = rows[0]['map_ref']['shard']
        rows[0]['map_ref']['shard'] = '../outside.npz'
        self.write_rows(rows)
        with self.assertRaisesRegex(ValidationError, 'INVALID_MAP_REFERENCE'):
            validate_bundle(self.root)
        rows[0]['map_ref']['shard'] = original
        rows[0]['class_index'] = 1
        self.write_rows(rows)
        with self.assertRaisesRegex(ValidationError, 'LABEL_MAPPING_MISMATCH'):
            validate_bundle(self.root)

    def test_duplicate_id_and_summary(self):
        """원본 행 위치 중복과 변환 집계 변조를 거부한다."""
        rows = self.read_rows()
        rows[1]['source_row_position'] = 0
        self.write_rows(rows)
        with self.assertRaisesRegex(ValidationError, 'INVALID_ROW_POSITION'):
            validate_bundle(self.root)
        rows[1]['source_row_position'] = 1
        self.write_rows(rows)
        summary = json.loads((self.root / 'conversion.json').read_text())
        summary['counts']['accepted'] += 1
        self.write_json('conversion.json', summary)
        with self.assertRaisesRegex(ValidationError, 'SUMMARY_COUNT_MISMATCH'):
            validate_bundle(self.root)

    def test_hash_and_object_dtype(self):
        """내용 해시 변경과 object 배열을 검출하고 pickle 복원을 하지 않는다."""
        path = self.root / 'maps/shard_000000.npz'
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        arrays['map_000000000'][0, 0] = 2
        np.savez(path, **arrays)
        with self.assertRaisesRegex(ValidationError, 'MAP_HASH_MISMATCH'):
            validate_bundle(self.root)
        arrays['map_000000000'] = np.array([['unsafe']], dtype=object)
        np.savez(path, **arrays)
        with self.assertRaisesRegex(ValidationError, 'INVALID_ARRAY_DTYPE_OR_SIZE'):
            validate_bundle(self.root)

    def test_extra_array_and_map_statistics(self):
        """참조되지 않은 배열과 다이 집계 변조를 검출한다."""
        path = self.root / 'maps/shard_000000.npz'
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        np.savez(path, extra=np.ones((1, 1), dtype=np.uint8), **arrays)
        with self.assertRaisesRegex(ValidationError, 'DUPLICATE_OR_EXTRA_ARRAY'):
            validate_bundle(self.root)
        np.savez(path, **arrays)
        rows = self.read_rows()
        rows[0]['valid_die_count'] = 999
        self.write_rows(rows)
        with self.assertRaisesRegex(ValidationError, 'MAP_STATISTIC_MISMATCH'):
            validate_bundle(self.root)

    def test_duplicate_conflicts_need_review(self):
        """저장 정합성 통과와 라벨 충돌 검토 필요를 구분한다."""
        rows = self.read_rows()
        # 같은 맵에 다른 명시적 라벨을 붙이되 집계는 일관되게 변경한다.
        rows[0]['pattern_label'], rows[0]['class_index'] = 'Center', 1
        rows[1]['map_hash'] = rows[0]['map_hash']
        rows[1]['failed_die_count'] = rows[0]['failed_die_count']
        rows[1]['failed_die_ratio'] = rows[0]['failed_die_ratio']
        self.write_rows(rows)
        summary = json.loads((self.root / 'conversion.json').read_text())
        summary['class_counts']['None'] -= 1
        summary['class_counts']['Center'] += 1
        self.write_json('conversion.json', summary)
        path = self.root / 'maps/shard_000000.npz'
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        arrays['map_000000001'] = arrays['map_000000000'].copy()
        np.savez(path, **arrays)
        result = validate_bundle(self.root)
        self.assertTrue(result['integrity_passed'])
        self.assertEqual(result['duplicate_audit']['map_label_conflicts'], 1)
        self.assertEqual(result['split_assessment']['status'], 'review_required')

    def test_json_and_report_overwrite(self):
        """중복 JSON 키·비유한 숫자와 보고서 덮어쓰기를 거부한다."""
        for data in ('{"x": 1, "x": 2}', '{"x": NaN}'):
            with self.assertRaises(ValidationError):
                decode_json(data)
        output = Path(self.temporary.name) / 'report'
        write_validation_report(validate_bundle(self.root), output)
        self.assertTrue((output / 'validation.json').is_file())
        with self.assertRaises(FileExistsError):
            write_validation_report({}, output)

    def test_missing_input_stage(self):
        """잘못된 입력 폴더를 데이터 내용 없이 경로 단계 오류로 구분한다."""
        with self.assertRaisesRegex(ValidationError, 'FILE_OR_DIRECTORY_NOT_FOUND') as caught:
            validate_bundle(Path(self.temporary.name) / 'not_present')
        self.assertEqual(caught.exception.stage, 'input_directory')

    def test_metadata_error_roles(self):
        """관리 파일의 누락·폴더 오지정·과대 크기를 구분한다."""
        root = Path(self.temporary.name)
        with self.assertRaisesRegex(ValidationError, '^MISSING_SUCCESS_FILE$'):
            read_json(root / 'SUCCESS.json')
        folder = root / 'conversion.json'
        folder.mkdir()
        with self.assertRaisesRegex(ValidationError, '^NOT_A_FILE_CONVERSION$'):
            read_json(folder)
        large = root / 'provenance.json'
        with large.open('wb') as stream:
            stream.truncate(4_000_001)
        with self.assertRaisesRegex(ValidationError, '^OVERSIZED_PROVENANCE_FILE$'):
            read_json(large)

    def test_review_reasons_and_private_output(self):
        """여러 검토 사유를 동시에 구분하고 실제 집계·식별자를 출력하지 않는다."""
        report = validate_bundle(self.root)
        self.assertTrue(all(passed for _, passed, _ in split_review_checks(report)))
        report['duplicate_audit']['map_label_conflicts'] = 12345
        report['duplicate_audit']['wafer_id_conflicts'] = 23456
        report['split_assessment']['class_group_counts']['0'] = 2
        report['split_assessment']['largest_group_ratio'] = 0.9
        checks = {code: passed for code, passed, _ in split_review_checks(report)}
        for code in ('MAP_LABEL_CONFLICT', 'WAFER_ID_CONFLICT',
                     'INSUFFICIENT_CLASS_GROUPS', 'LARGE_CONNECTED_GROUP'):
            self.assertFalse(checks[code])
        capture = io.StringIO()
        with redirect_stdout(capture):
            print_split_review(report)
        for private in ('12345', '23456', 'lot_0', '0.9'):
            self.assertNotIn(private, capture.getvalue())
        self.assertIn('검토 필요', capture.getvalue())

    def test_saved_report_without_data_revalidation(self):
        """기존 보고서만 읽을 때 원본·NPZ를 재검사하지 않는다."""
        report = validate_bundle(self.root)
        output = Path(self.temporary.name) / 'saved_report'
        write_validation_report(report, output)
        with patch('sys.argv', ['validator', '--show-report', str(output / 'validation.json')]), \
                patch('src.wafer_dl.validate_canonical.validate_bundle',
                      side_effect=AssertionError('데이터 재검사 금지')), redirect_stdout(io.StringIO()):
            self.assertEqual(main(), 0)
        self.assertIn('분할 조건별 진단', (output / 'validation.md').read_text(encoding='utf-8'))

    def test_corrupt_archive_stage(self):
        """손상된 NPZ와 임시 DB 권한 문제를 서로 다른 단계로 구분한다."""
        (self.root / 'maps/shard_000000.npz').write_bytes(b'not a zip')
        with self.assertRaisesRegex(ValidationError, 'INVALID_NPZ_ARCHIVE') as caught:
            validate_bundle(self.root)
        self.assertEqual(caught.exception.stage, 'shards')
        with patch('src.wafer_dl.validate_canonical.sqlite3.connect',
                   side_effect=PermissionError('sensitive path must not leak')):
            with self.assertRaisesRegex(ValidationError, '^FILE_ACCESS_DENIED$') as caught:
                validate_bundle(self.root)
        self.assertEqual(caught.exception.stage, 'temporary_database')


if __name__ == '__main__':
    unittest.main()
