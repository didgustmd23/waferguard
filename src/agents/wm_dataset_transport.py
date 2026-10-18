"""기존 AppContainer·Job으로 WM 개발 구간 Dataset 검사 worker를 실행한다."""

import json
import os
from pathlib import Path
import shutil
import sys

from .appcontainer_native import AppContainer
from .appcontainer_probe import _acl, probe_environment, runtime_files
from .artifacts import _fingerprint, _root
from .wm_cache_transport import WORKER_FILES as BASE_FILES, CONTROL_FILES as BASE_CONTROLS
from .wm_dataset_runtime import decode_manifest, MAX_MANIFEST_BYTES, copy_runtime_file


ENTRY = "src/agents/wm_dataset_worker.py"
PREPARATION_STAGES = frozenset({
    "CODE_CHECK", "RUNTIME_INVENTORY", "PROFILE_CREATE", "INPUT_ACL", "OUTPUT_ACL",
    "RUNTIME_COPY", "TOOLS_COPY", "CODE_RECHECK", "ENVIRONMENT", "WORKER_LAUNCH", "PROFILE_CLEANUP",
})
PREPARATION_CODES = frozenset({
    "SETUP_ACCESS_DENIED", "SETUP_OS_ERROR", "SETUP_CONTRACT_REJECTED", "SETUP_FAILED",
})
WORKER_FILES = tuple(dict.fromkeys(BASE_FILES + (
    "src/agents/wm_dataset_results.py", "src/agents/wm_dataset_contract.py", "src/agents/wm_dataset_runtime.py",
    "src/wafer_dl/dataset.py",
    "src/wafer_dl/check_transform.py", "src/wafer_dl/transform.py",
    "scripts/wm_sandbox/export_canonical.py",
)))
CONTROL_FILES = tuple(dict.fromkeys(BASE_CONTROLS + WORKER_FILES + (
    ENTRY, "src/agents/wm_dataset.py", "src/agents/wm_dataset_cli.py",
    "src/agents/wm_dataset_transport.py",
)))


def code_manifest():
    """승인에 연결할 고정 코드의 해시를 계산한다."""
    project = Path(__file__).resolve().parents[2]

    return {name: _fingerprint(project / name)["sha256"] for name in CONTROL_FILES}


def run_dataset_worker(request, root, cancel_event):
    """현재 AppContainer·읽기 전용 입력 ACL·Job 한도와 종료 정리를 재사용한다."""
    root = _root(root)
    project = Path(__file__).resolve().parents[2]
    container = AppContainer("waferguard.wmdataset." + request.task_id)
    outcome = {"request_id": request.request_id, "profile_cleanup": False}
    stage = "CODE_CHECK"

    try:
        before = code_manifest()
        stage = "RUNTIME_INVENTORY"
        from .worker_contract import _decode
        from .wm_canonical_contract import digest
        from .wm_canonical_results import _read_small

        packet = _decode(_read_small(root / "input/analysis.json", 65536).decode())
        files = decode_manifest(_read_small(root / "input/runtime_files.json", MAX_MANIFEST_BYTES),
                                packet["max_runtime_bytes"])

        if digest(files) != packet["runtime_sha256"]:
            raise ValueError("Dataset 런타임 명세가 승인 범위와 다릅니다.")
        stage = "PROFILE_CREATE"
        sid = container.create()
        stage = "INPUT_ACL"

        for name in ("runtime", "input", "tools"):
            _acl(root / name, sid, "RX")

        stage = "OUTPUT_ACL"
        _acl(Path(request.output_directory), sid, "M", low=True)
        _acl(Path(request.scratch_directory), sid, "M", low=True)

        stage = "RUNTIME_COPY"
        for name, record in files.items():
            copy_runtime_file(Path(sys.prefix) / name, root / "runtime" / name, record)

        stage = "TOOLS_COPY"
        for name in WORKER_FILES + (ENTRY,):
            target = root / "tools" / ("worker.py" if name == ENTRY else name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project / name, target)

            if _fingerprint(target)["sha256"] != before[name]:
                raise ValueError("복사한 WM 개발 구간 Dataset 검사 worker 코드가 승인 대상과 다릅니다.")

        stage = "CODE_RECHECK"
        if before != code_manifest():
            raise ValueError("WM 개발 구간 Dataset 검사 worker 준비 중 코드가 변경되었습니다.")

        if cancel_event.is_set():
            outcome["launch"] = {"status": "cancelled"}
        else:
            stage = "ENVIRONMENT"
            environment = probe_environment(root / "runtime", Path(request.scratch_directory), os.environ)
            environment.update({"CUDA_VISIBLE_DEVICES": "-1", "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"})
            stage = "WORKER_LAUNCH"
            outcome["launch"] = container.launch(
                root / "runtime/python.exe",
                ["-I", "-S", "-B", str(root / "tools/worker.py"), str(root), request.request_id],
                environment, Path(request.scratch_directory),
                limits=request.limits, cancel_event=cancel_event,
            )
    except Exception as error:
        # 예외 원문에는 사용자 경로·환경 값이 포함될 수 있다. 고정 분류와
        # 숫자 OS 코드만 남기고 성공 응답·자동 재시도로 바꾸지 않는다.
        outcome["preparation_failure"] = preparation_failure(stage, error)
    finally:
        try:
            outcome["profile_cleanup"] = container.close()
        except Exception as error:
            outcome.setdefault("preparation_failure", preparation_failure("PROFILE_CLEANUP", error))

        (root / "transport.json").write_text(json.dumps(outcome), encoding="utf-8")

    return outcome


def preparation_failure(stage, error):
    """민감 정보 없는 준비 오류 분류. OS 코드는 권한 오류 등 로컬 진단에만 사용한다."""
    if isinstance(error, PermissionError):
        code = "SETUP_ACCESS_DENIED"
    elif isinstance(error, OSError):
        code = "SETUP_OS_ERROR"
    elif isinstance(error, ValueError):
        code = "SETUP_CONTRACT_REJECTED"
    else:
        code = "SETUP_FAILED"

    number = getattr(error, "winerror", None)

    if type(number) is not int:
        number = getattr(error, "errno", None)

    return {"stage": stage, "code": code,
            "os_error": number if type(number) is int and 0 <= number <= 2 ** 32 - 1 else None}
