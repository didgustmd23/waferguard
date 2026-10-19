"""합성 WM CPU 검사 전용 격리 진입점. 입력 파일이나 실행할 코드를 선택받지 않는다."""

import json
import os
from pathlib import Path
import re
import sys


def main():
    if len(sys.argv) != 2:
        return 2

    root = Path(sys.argv[1]).absolute()
    sys.path.insert(0, str(root / "tools"))
    from src.agents.pickle_import_worker import _isolated_process, _diagnostic

    # 실제 AppContainer 토큰·capability 부재·Job 소속을 검사한다.
    if not _isolated_process():
        return 2

    runtime = root / "runtime"
    sys.path.insert(1, str(runtime / "Lib/site-packages"))
    handles = [os.add_dll_directory(str(path)) for path in
               (runtime, runtime / "DLLs", runtime / "Library/bin", runtime / "Lib/site-packages/torch/lib")
               if path.is_dir()]
    stage = "IMPORT_PAYLOAD"
    result = {"schema_version": "wm_cpu_probe_v1", "status": "review_required",
              "real_data_read": False, "production_worker_enabled": False}

    def diagnostic(value):
        nonlocal stage
        stage = value
        _diagnostic(root, stage, "START")

    try:
        from src.agents.wm_cpu_probe_payload import run_synthetic

        workspace = root / "scratch/task/synthetic"
        workspace.mkdir()
        result = run_synthetic(workspace, diagnostic)
        result["status"] = "cpu_checks_completed"
        _diagnostic(root, "COMPLETE", "PASSED")
    except Exception as error:
        result.update(failure_stage=stage, failure_code="SYNTHETIC_CPU_CHECK_FAILED")
        if isinstance(error, PermissionError):
            result["failure_code"] = "SYNTHETIC_ACCESS_DENIED"

        number = getattr(error, "winerror", None) or getattr(error, "errno", None)
        if type(number) is int and 0 <= number <= 2 ** 32 - 1:
            result["os_error"] = number

        validation_code = getattr(error, "code", None)
        if type(validation_code) is str and re.fullmatch(r"[A-Z][A-Z0-9_]{0,127}", validation_code):
            result["validation_code"] = validation_code

        if isinstance(error, ModuleNotFoundError) and isinstance(error.name, str) and re.fullmatch(r"[A-Za-z0-9_.]{1,128}", error.name):
            result.update(failure_code="LIBRARY_DEPENDENCY_MISSING", missing_module=error.name)

        _diagnostic(root, stage, result["failure_code"])
    finally:
        result["probe_id"] = root.name
        (root / "output/child_result.json").write_text(json.dumps(result, allow_nan=False), encoding="utf-8")

    return 0 if result.get("status") == "cpu_checks_completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
