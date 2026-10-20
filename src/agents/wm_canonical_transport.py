"""기존 AppContainer·Job으로 구조 판별형 WM canonical worker를 실행한다."""

import json
import os
from pathlib import Path
import shutil
import sys

from .operation_progress import report_progress
from .appcontainer_native import AppContainer
from .appcontainer_probe import _acl, probe_environment, runtime_files
from .artifacts import _fingerprint, _root
from .isolated_eda_transport import WORKER_FILES as BASE_FILES, CONTROL_FILES as BASE_CONTROLS


ENTRY = "src/agents/wm_canonical_worker.py"
WORKER_FILES = tuple(dict.fromkeys(BASE_FILES + (
    "src/agents/pickle_import_worker.py",
    "src/agents/wm_canonical_contract.py", "src/agents/wm_canonical_results.py",
    "src/preprocessing/canonical_data.py",
    "src/wafer_dl/validate_canonical.py", "src/wafer_dl/eda.py",
)))
CONTROL_FILES = tuple(dict.fromkeys(BASE_CONTROLS + WORKER_FILES + (
    ENTRY, "src/agents/wm_canonical.py", "src/agents/wm_canonical_cli.py",
    "src/agents/wm_canonical_transport.py",
    "src/agents/split_evidence.py",
    # 부모의 공통 승인·런타임·서비스 변경도 기존 승인에 반영한다.
    # worker에 이 관리 코드의 읽기 권한을 추가하는 목록은 아니다.
    "src/agents/workflow_approval.py", "src/agents/wm_runtime.py",
    "src/agents/wm_dataset_runtime.py",
    "src/agents/operation_progress.py",
    "src/agents/wm_workflow_service.py", "src/agents/wm_creation_schema.py",
    "src/agents/wm_workflow_execution.py", "src/agents/wm_module_catalogue.py",
    "src/agents/wm_run_creation.py", "src/agents/wm_run_evidence.py",
    "src/agents/wm_run_bindings.py", "src/agents/wm_workflow_guidance.py",
    "src/agents/wm_module_plan.py", "src/agents/agent_module_registry.py",
    "src/agents/module_specification.py", "configs/agents/modules/catalogue.json",
)))


def code_manifest():
    """승인에 연결할 고정 코드의 해시를 계산한다."""
    project = Path(__file__).resolve().parents[2]

    return {name: _fingerprint(project / name)["sha256"] for name in CONTROL_FILES}


def run_wm_worker(request, root, cancel_event):
    """현재 AppContainer·읽기 전용 입력 ACL·Job 한도와 종료 정리를 재사용한다."""
    root = _root(root)
    project = Path(__file__).resolve().parents[2]
    container = AppContainer("waferguard.wmcanonical." + request.task_id)
    outcome = {"request_id": request.request_id, "profile_cleanup": False}

    try:
        report_progress("contract_check")
        before = code_manifest()
        report_progress("runtime_inventory")
        files = runtime_files(Path(sys.prefix))
        report_progress("profile_create")
        sid = container.create()

        report_progress("permissions")
        for name in ("runtime", "input", "tools"):
            _acl(root / name, sid, "RX")

        _acl(Path(request.output_directory), sid, "M", low=True)
        _acl(Path(request.scratch_directory), sid, "M", low=True)

        report_progress("runtime_copy")
        copied_bytes = 0
        for index, source in enumerate(files, start=1):
            target = root / "runtime" / source.relative_to(sys.prefix)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            copied_bytes += target.stat().st_size
            report_progress("runtime_copy", completed=index, total=len(files), bytes_done=copied_bytes)

        report_progress("tools_copy")
        for name in WORKER_FILES + (ENTRY,):
            target = root / "tools" / ("worker.py" if name == ENTRY else name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project / name, target)

            if _fingerprint(target)["sha256"] != before[name]:
                raise ValueError("복사한 WM canonical worker 코드가 승인 대상과 다릅니다.")

        if before != code_manifest():
            raise ValueError("WM canonical worker 준비 중 코드가 변경되었습니다.")

        if cancel_event.is_set():
            outcome["launch"] = {"status": "cancelled"}
        else:
            environment = probe_environment(root / "runtime", Path(request.scratch_directory), os.environ)
            report_progress("worker_wait")
            outcome["launch"] = container.launch(
                root / "runtime/python.exe",
                ["-I", "-S", "-B", str(root / "tools/worker.py"), str(root), request.request_id],
                environment, Path(request.scratch_directory),
                limits=request.limits, cancel_event=cancel_event,
            )
    finally:
        report_progress("cleanup")
        outcome["profile_cleanup"] = container.close()
        (root / "transport.json").write_text(json.dumps(outcome), encoding="utf-8")

    return outcome
