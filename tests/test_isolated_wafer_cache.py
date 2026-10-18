"""격리 캐시 코어의 기존 출력 동등성과 제한 NPY 판독 경계를 확인한다."""

from pathlib import Path
import unittest

import numpy as np

from tests import test_wafer_cache_dataset as fixtures
from src.wafer_dl.cache import build_cache, load_cache_config
from src.wafer_dl.isolated_cache import build_isolated_cache
from src.agents.wm_cache_results import check_numeric_shard


class IsolatedCacheTest(unittest.TestCase):
    setUp = fixtures.CacheDatasetTest.setUp

    def test_existing_cache_bytes_match(self):
        """기존 writer를 사용하므로 shard·색인·보고서 바이트가 모두 일치한다."""
        project = Path(__file__).resolve().parents[1]
        cache_path = project / 'configs/wm811k/cache.json'
        original, isolated = self.base / 'old_cache', self.base / 'isolated_cache'
        workspace = self.base / 'scratch'
        workspace.mkdir()
        build_cache(self.split, self.check, original, self.transform_path, cache_path,
                    approved=True, canonical_dir=self.fixture.root)
        build_isolated_cache(self.split, self.check, isolated, self.fixture.root,
                             self.transform, load_cache_config(cache_path),
                             workspace=workspace, max_output_bytes=32 * 1024 * 1024)
        files = {path.relative_to(original) for path in original.rglob('*') if path.is_file()}
        self.assertEqual(files, {path.relative_to(isolated) for path in isolated.rglob('*') if path.is_file()})

        for name in files:
            self.assertEqual((original / name).read_bytes(), (isolated / name).read_bytes(), str(name))

    def test_budget_fails_without_partial_cache(self):
        from src.wafer_dl.validate_canonical import ValidationError

        output, scratch = self.base / 'too_small', self.base / 'scratch'
        scratch.mkdir()
        config = load_cache_config(Path(__file__).resolve().parents[1] / 'configs/wm811k/cache.json')

        with self.assertRaises(ValidationError):
            build_isolated_cache(self.split, self.check, output, self.fixture.root, self.transform, config,
                                 workspace=scratch, max_output_bytes=1)

        self.assertFalse(output.exists())

    def test_numeric_reader_rejects_object_extra_and_empty(self):
        path = self.base / 'array.npy'
        np.save(path, np.array([[[1, 2], [0, 1]]], dtype=np.uint8), allow_pickle=False)
        check_numeric_shard(path, 1, [2, 2])

        with path.open('ab') as stream:
            stream.write(b'extra')

        with self.assertRaises(ValueError):
            check_numeric_shard(path, 1, [2, 2])

        np.save(path, np.zeros((1, 2, 2), dtype=np.uint8), allow_pickle=False)

        with self.assertRaises(ValueError):
            check_numeric_shard(path, 1, [2, 2])

        np.save(path, np.array([object()], dtype=object), allow_pickle=True)

        with self.assertRaises(ValueError):
            check_numeric_shard(path, 1, [2, 2])


if __name__ == '__main__':
    unittest.main()
