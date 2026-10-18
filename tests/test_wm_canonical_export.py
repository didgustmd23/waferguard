"""실제 pickle을 읽지 않는 합성 canonical 변환 검증."""
import json
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from scripts.wm_sandbox.export_canonical import export_bundle, export_frame, normalize_label, normalize_map, main


# ==========================================
# 합성 입력으로 라벨·맵·숫자 저장 계약 검증
# - 미라벨과 정상 클래스를 구분하고 잘못된 행은 격리
# - 가변 크기 맵을 pickle 없이 복원해 원본과 비교
# ==========================================
class CanonicalExportTest(unittest.TestCase):
    def test_nine_source_labels_and_aliases(self):
        """원본의 9개 표기·중첩 구조를 고정 클래스 순서로 변환한다."""
        labels = ('none', 'Center', 'Donut', 'Edge-Loc', 'Edge-Ring',
                  'Loc', 'Near-full', 'Random', 'Scratch')
        for index, label in enumerate(labels):
            with self.subTest(label=label):
                normalized, mapped_index, status = normalize_label(np.array([[label]]))
                self.assertEqual(mapped_index, index)
                self.assertEqual(status, 'labeled')
                self.assertEqual(normalized, 'Near-Full' if label == 'Near-full' else
                                 ('None' if label == 'none' else label))
        # 별칭을 무한 확대하지 않아 알 수 없는 표기는 여전히 격리한다.
        self.assertEqual(normalize_label('Near-Full'), ('Near-Full', 6, 'labeled'))
        self.assertEqual(normalize_label('near-FULL')[2], 'invalid')

    def test_near_full_export_and_progress(self):
        """Near-full이 격리되지 않으며 진행률과 규칙 버전을 기록한다."""
        labels = ('none', 'Center', 'Donut', 'Edge-Loc', 'Edge-Ring',
                  'Loc', 'Near-full', 'Random', 'Scratch')
        frame = pd.DataFrame({'waferMap': [np.array([[1, 2]]) for _ in labels],
                              'failureType': [np.array([[label]]) for label in labels],
                              'lotName': [f'lot_{index}' for index in range(9)]})
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'canonical'
            progress = []
            summary = export_frame(frame, output, 'synthetic', progress=progress.append)
            self.assertEqual(summary['class_counts']['Near-Full'], 1)
            self.assertEqual(summary['counts']['accepted'], 9)
            self.assertEqual(summary['label_coverage_status'], 'complete')
            self.assertEqual(summary['label_normalization_version'], 'wm_label_v2')
            self.assertEqual(progress[-1], 1.0)
            rows = [json.loads(line) for line in (output / 'manifest.jsonl').read_text().splitlines()]
            self.assertEqual(rows[6]['class_index'], 6)
            self.assertEqual(rows[6]['reason_codes'], [])

    def test_host_requires_both_permissions(self):
        """로컬 실행은 두 승인 옵션과 출력 경로 확인 전 원본을 읽지 않는다."""
        cases = ([], ['--trusted-pickle'], ['--allow-host'],
                 ['--allow-host', '--trusted-pickle'])
        for extra in cases:
            with self.subTest(options=extra), patch.dict('os.environ', {'USERNAME': 'local_user'}), \
                    patch('sys.argv', ['exporter', '--input', 'not_read.pkl'] + extra), \
                    patch('scripts.wm_sandbox.export_canonical.pd.read_pickle') as reader, \
                    redirect_stdout(io.StringIO()), patch('sys.stderr', io.StringIO()):
                with self.assertRaises(SystemExit):
                    main()
                reader.assert_not_called()

    def test_trusted_host_cli_with_mock_reader(self):
        """가짜 파일·mock 로더로 로컬 경로와 새 묶음 저장만 검증한다."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / 'raw'
            raw.mkdir()
            source = raw / 'synthetic.pkl'
            source.write_bytes(b'synthetic placeholder, never unpickled')
            output = root / 'processed'
            frame = pd.DataFrame({'waferMap': [np.array([[1, 2]])],
                                  'failureType': ['Near-full'], 'lotName': ['synthetic_lot']})
            with patch.dict('os.environ', {'USERNAME': 'local_user'}), \
                    patch('sys.argv', ['exporter', '--input', str(source), '--output-root', str(output),
                                       '--trusted-pickle', '--allow-host']), \
                    patch('scripts.wm_sandbox.export_canonical.pd.read_pickle', return_value=frame) as reader, \
                    redirect_stdout(io.StringIO()):
                main()
            reader.assert_called_once_with(source.resolve())
            bundles = list(output.glob('canonical_*'))
            self.assertEqual(len(bundles), 1)
            self.assertTrue((bundles[0] / 'SUCCESS.json').is_file())
            provenance = json.loads((bundles[0] / 'provenance.json').read_text())
            self.assertEqual(provenance['execution_mode'], 'trusted_host')

    def test_exact_frame_budget_and_one_byte_overflow(self):
        """Windows 줄바꿈과 ZIP 헤더 재기록까지 실제 파일 바이트로 제한한다."""
        frame = pd.DataFrame({
            "waferMap": [np.array([[1, 2]], dtype=np.uint8)],
            "failureType": ["Center"],
            "lotName": ["synthetic_lot"],
        })

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline"
            export_frame(frame, baseline, "synthetic")
            size = sum(path.stat().st_size for path in baseline.rglob("*") if path.is_file())
            exact = root / "exact"
            export_frame(frame, exact, "synthetic", max_output_bytes=size)
            self.assertEqual(sum(path.stat().st_size for path in exact.rglob("*") if path.is_file()), size)
            self.assertNotIn(b"\r\n", (exact / "manifest.jsonl").read_bytes())

            limited = root / "limited"

            with self.assertRaisesRegex(ValueError, "출력 크기 상한"):
                export_frame(frame, limited, "synthetic", max_output_bytes=size - 1)

            self.assertLessEqual(sum(path.stat().st_size for path in limited.rglob("*") if path.is_file()), size - 1)

    def test_bundle_budget_includes_provenance_and_success(self):
        """정상 성공 표식을 만드는 경로에서 마지막 한 바이트 부족도 미완료로 남긴다."""
        frame = pd.DataFrame({"waferMap": [np.array([[1, 2]])],
                              "failureType": ["Center"], "lotName": ["fixture_lot"]})

        with tempfile.TemporaryDirectory() as temporary, patch("scripts.wm_sandbox.export_canonical.datetime") as clock:
            clock.now.return_value.isoformat.return_value = "2026-10-17T00:00:00+00:00"
            root = Path(temporary)

            def bundle(name, maximum):
                export_bundle(frame, root / name, "a" * 64, 100,
                              execution_mode="appcontainer", max_output_bytes=maximum)

            bundle("baseline", 1024 * 1024)
            size = sum(path.stat().st_size for path in (root / "baseline").rglob("*") if path.is_file())
            bundle("exact", size)
            self.assertTrue((root / "exact/SUCCESS.json").is_file())
            self.assertEqual(sum(path.stat().st_size for path in (root / "exact").rglob("*") if path.is_file()), size)

            with self.assertRaisesRegex(ValueError, "출력 크기 상한"):
                bundle("limited", size - 1)

            self.assertTrue((root / "limited/provenance.json").is_file())
            self.assertFalse((root / "limited/SUCCESS.json").exists())
            self.assertLessEqual(sum(path.stat().st_size for path in (root / "limited").rglob("*") if path.is_file()), size - 1)

    def test_cli_output_budget_is_applied(self):
        """명령행 출력 상한이 실제 변환 경로에 전달된다."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            raw.mkdir()
            source = raw / "fixture.pkl"
            source.write_bytes(b"synthetic placeholder")
            frame = pd.DataFrame({"waferMap": [np.array([[1, 2]])],
                                  "failureType": ["Center"], "lotName": ["fixture_lot"]})

            with patch("sys.argv", ["exporter", "--input", str(source), "--output-root", str(root / "output"),
                                    "--trusted-pickle", "--allow-host", "--max-output-bytes", "1"]), \
                    patch("scripts.wm_sandbox.export_canonical.pd.read_pickle", return_value=frame), \
                    redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "출력 크기 상한"):
                    main()

            self.assertFalse(list((root / "output").rglob("SUCCESS.json")))

    def test_label_contract(self):
        # 단일 중첩 라벨, 빈 배열, 복수·미지 라벨을 각각 검사한다.
        self.assertEqual(normalize_label(np.array([['none']])), ('None', 0, 'labeled'))
        self.assertEqual(normalize_label(np.array([])), (None, None, 'unlabeled'))
        self.assertEqual(normalize_label(['Center', 'Scratch'])[2], 'invalid')
        self.assertEqual(normalize_label('unknown')[2], 'invalid')

    def test_invalid_maps(self):
        # 형 변환 전에 유효 다이·차원·유한값·허용값을 검사한다.
        for value in (np.zeros((2, 2)), np.array([[1.5]]),
                      np.array([[np.nan]]), np.array([1]), np.array([['1']])):
            self.assertIsNone(normalize_map(value)[0])

    def test_numeric_restore_and_row_accounting(self):
        # 모든 행을 보존하고 숫자 배열의 shard 참조와 격리 상태를 대조한다.
        maps = [np.array([[0, 1, 2]]), np.array([[2], [1]]), np.zeros((1, 1))]
        frame = pd.DataFrame({'waferMap': maps, 'failureType': ['none', [], 'bad'],
                              'lotName': ['lot_a', 'lot_b', None]})
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'canonical'
            summary = export_frame(frame, output, 'synthetic', shard_rows=1)
            rows = [json.loads(line) for line in (output / 'manifest.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 3)
            self.assertEqual([row['split'] for row in rows], ['pending', 'unlabeled', 'quarantined'])
            self.assertFalse(summary['training_ready'])
            self.assertEqual(summary['shard_count'], 2)
            for index in (0, 1):
                ref = rows[index]['map_ref']
                with np.load(output / ref['shard'], allow_pickle=False) as archive:
                    restored = archive[ref['key']]
                    self.assertEqual(restored.dtype, np.uint8)
                    np.testing.assert_array_equal(restored, maps[index])
            # 사용자 결과가 이미 있으면 덮어쓰지 않는다.
            with self.assertRaises(FileExistsError):
                export_frame(frame, output, 'synthetic')


if __name__ == '__main__':
    unittest.main()
