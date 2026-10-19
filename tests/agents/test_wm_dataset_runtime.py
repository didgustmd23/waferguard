"""설치 환경을 변경하지 않고 런타임 명세·경로·제한 복사 규칙을 검증한다."""

import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.agents.wm_canonical_contract import file_hash
from src.agents.wm_dataset_runtime import copy_runtime_file, decode_manifest, runtime_manifest, runtime_relative, pinned_torch_version


class DatasetRuntimeTest(unittest.TestCase):
    def test_runtime_manifest_pins_actual_build_for_creation_and_probe(self):
        """배포 버전이 local 표기를 생략해도 생성 계약에는 해시로 고정한 빌드를 담는다."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            torch = root / 'Lib/site-packages/torch'
            torch.mkdir(parents=True)
            (torch / '__init__.py').write_bytes(b'# synthetic')
            executable = root / 'python.exe'
            executable.write_bytes(b'synthetic')
            for build in ('cpu', 'cu130'):
                (torch / 'version.py').write_bytes(f"__version__ = '2.14.1+{build}'\n".encode())
                with patch('src.agents.appcontainer_probe.runtime_files', return_value=[executable]), \
                        patch('src.agents.wm_dataset_runtime.distribution', return_value=SimpleNamespace(version='2.14.1')):
                    records, version = runtime_manifest(root, 1024 * 1024)
                self.assertEqual(version, f'2.14.1+{build}')
                self.assertIn('Lib/site-packages/torch/version.py', records)

    def test_probe_and_dataset_share_one_version_validator(self):
        from src.agents.wm_cpu_probe import pinned_torch_version as probe_validator

        self.assertIs(probe_validator, pinned_torch_version)

    def test_library_names_and_unsafe_paths(self):
        for name in ('Lib/site-packages/torch/include/C++17.h', 'Lib/site-packages/setuptools/script (dev).tmpl'):
            runtime_relative(name)

        for name in ('../secret', 'Lib/../secret', '/absolute', 'C:/secret', 'Lib\\secret',
                     'Lib/CON', 'Lib/file. ', 'Lib//file', 'Lib/file:stream'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                runtime_relative(name)

    def test_manifest_types_duplicates_and_limits(self):
        records = {'Lib/site-packages/torch/__init__.py': {'size': 1, 'sha256': 'a' * 64}}
        self.assertEqual(decode_manifest(json.dumps(records).encode(), 1), records)

        for maximum in (-1, 0, True, 1.5):
            with self.subTest(maximum=maximum), self.assertRaises(ValueError):
                runtime_manifest(Path('missing'), maximum)

            with self.subTest(maximum=maximum), self.assertRaises(ValueError):
                decode_manifest(b'{}', maximum)

        for payload in (b'{}', b'{"x":1,"x":2}',
                        json.dumps({'../secret': {'size': 1, 'sha256': 'a' * 64}}).encode()):
            with self.assertRaises(ValueError):
                decode_manifest(payload, 1024)

    def test_bounded_copy_detects_growth_and_hash_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'source'
            source.write_bytes(b'original')
            record = file_hash(source, 100)
            copy_runtime_file(source, root / 'copy', record)
            self.assertEqual(source.read_bytes(), (root / 'copy').read_bytes())
            source.write_bytes(b'changed!')

            with self.assertRaises(ValueError):
                copy_runtime_file(source, root / 'bad_hash', record)

            source.write_bytes(b'longer than the approved bytes')

            with self.assertRaises(ValueError):
                copy_runtime_file(source, root / 'too_long', record)


if __name__ == '__main__':
    unittest.main()
