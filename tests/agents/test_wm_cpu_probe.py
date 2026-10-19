"""합성 CPU 검사 코어·결과 계약·복사 목록 검사. 실제 AppContainer를 시작하지 않는다."""

from copy import deepcopy
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.agents.wm_cpu_probe import COPY_FILES, validate_child, pinned_torch_version
from src.agents.wm_cpu_probe_payload import run_synthetic


class PinnedTorchVersionTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.prefix = Path(temporary.name)
        self.name = "Lib/site-packages/torch/version.py"
        self.path = self.prefix / self.name
        self.path.parent.mkdir(parents=True)

    def source(self, text):
        raw = text.encode("utf-8")
        self.path.write_bytes(raw)
        return {self.name: {"size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}}

    def test_cpu_build_literal_is_pinned_without_executing_file(self):
        files = self.source("raise RuntimeError('실행 금지')\n__version__ = '2.14.1+cpu'\n")
        self.assertEqual(pinned_torch_version(self.prefix, files, "2.14.1"), "2.14.1+cpu")

    def test_wrong_public_or_local_metadata_is_rejected(self):
        files = self.source("__version__ = '2.14.1+cpu'\n")
        for metadata in ("2.14.0", "2.14.1+cu128"):
            with self.assertRaises(ValueError):
                pinned_torch_version(self.prefix, files, metadata)

    def test_changed_file_is_rejected(self):
        files = self.source("__version__ = '2.14.1+cpu'\n")
        self.path.write_bytes(b"__version__ = '2.14.1+cu128'\n")
        with self.assertRaises(ValueError):
            pinned_torch_version(self.prefix, files, "2.14.1")

    def test_annotated_literal_and_invalid_definitions(self):
        files = self.source("__version__: str = '2.14.1+cpu'\n")
        self.assertEqual(pinned_torch_version(self.prefix, files, "2.14.1+cpu"), "2.14.1+cpu")
        for source in ("x = 1", "__version__ = str('2.14.1')", "__version__='2.14.1'\n__version__='2.14.1'"):
            with self.assertRaises(ValueError):
                pinned_torch_version(self.prefix, self.source(source), "2.14.1")


class WmCpuProbeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        # 실데이터·pickle 복원이 없어야 한다. PyTorch와 WM 코어는 실제 CPU에서 실행한다.
        # native worker처럼 상위 경로 검사가 거부돼도 staged 파일 검사는 동작해야 한다.
        with patch("pandas.read_pickle", side_effect=AssertionError("pickle 금지")), \
                patch("src.agents.wm_canonical_contract._root", side_effect=PermissionError(13, "synthetic")):
            cls.report = run_synthetic(cls.root)
        cls.report["status"] = "cpu_checks_completed"

    def test_existing_wm_cpu_core_and_fixed_expectations(self):
        self.assertTrue(validate_child(self.report), self.report)
        self.assertEqual(self.report["tensor_values"], [0, 2, 4, 6, 8, 10])
        self.assertFalse((self.root / "cache/test").exists())

    def test_different_probe_result_is_rejected(self):
        report = deepcopy(self.report)
        report["probe_id"] = "previous-probe"
        self.assertFalse(validate_child(report, "current-probe"))
        self.assertTrue(validate_child(report, "previous-probe"))

    def test_failed_checks_and_forged_contract_rejected(self):
        for field, value in (("device", "cuda"), ("samples", 53), ("model_trained", True),
                             ("real_data_read", True), ("tensor_values", [0]),
                             ("production_worker_enabled", True)):
            with self.subTest(field=field):
                report = deepcopy(self.report)
                report[field] = value
                self.assertFalse(validate_child(report))

        for name in self.report["checks"]:
            report = deepcopy(self.report)
            report["checks"][name] = False
            self.assertFalse(validate_child(report), name)

    def test_batch_shape_protocol_and_test_scope_rejected(self):
        for field, value in (("image_shape", [4, 3, 32, 32]), ("target_dtype", "torch.float32"),
                             ("test_maps_read", True), ("split_protocol_id", "another")):
            report = deepcopy(self.report)
            report["batches"]["validation"][field] = value
            self.assertFalse(validate_child(report), field)

    def test_workspace_must_be_new_and_empty(self):
        with self.assertRaises(ValueError):
            run_synthetic(self.root)

    def test_copy_list_contains_core_without_ui_or_test_imports(self):
        required = {"src/wafer_dl/split.py", "src/wafer_dl/dataset.py", "src/wafer_dl/eda.py",
                    "src/wafer_dl/isolated_cache.py", "src/wafer_dl/isolated_transform_check.py"}
        self.assertTrue(required <= set(COPY_FILES))
        self.assertFalse(any(name.startswith(("tests/", "src/ui/")) for name in COPY_FILES))

    def test_copied_code_imports_without_project_and_refuses_host_entry(self):
        from src.agents.wm_cpu_probe import ENTRY

        project = Path(__file__).resolve().parents[2]
        copied = self.root / "copied/tools"
        for name in (*COPY_FILES, ENTRY):
            target = copied / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project / name, target)

        # -I/-S와 별도 cwd에서 프로젝트 import 캐시 없이 복사 목록만 검사한다.
        code = "import sys; sys.path.insert(0, sys.argv[1]); from src.agents.wm_cpu_probe_payload import CHECKS; assert len(CHECKS)==11"
        result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", code, str(copied)],
                                cwd=copied, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        result = subprocess.run([sys.executable, "-I", "-S", "-B", str(copied / ENTRY), str(copied.parent)],
                                cwd=copied, capture_output=True, timeout=30)
        # 비격리 부모 환경에서 native 진입점을 호출하면 거부해야 한다.
        self.assertEqual(result.returncode, 2)

    def test_external_workspace_is_used_and_nonempty_rejected(self):
        from src.wafer_dl.split import run_split

        with patch("src.wafer_dl.split.tempfile.TemporaryDirectory", side_effect=AssertionError("임시 ACL 우회 금지")):
            with self.assertRaisesRegex(ValueError, "INVALID_SPLIT_WORKSPACE"):
                run_split(self.root / "canonical", self.root / "unused_split", self.root / "split_config.json",
                          approved=True, workspace=self.root / "split_workspace")


if __name__ == "__main__":
    unittest.main()
