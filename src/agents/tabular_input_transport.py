"""고정 표형 입력 worker의 기존 AppContainer·Job 운송 연결."""

import json
import os
from pathlib import Path
import shutil
import sys

from .appcontainer_native import AppContainer
from .appcontainer_probe import _acl, probe_environment, runtime_files
from .artifacts import _fingerprint, _root
from .module_pipeline_transport import WORKER_FILES as BASE_FILES, CONTROL_FILES as BASE_CONTROLS
from .writable_directory_acl import prepare_writable_directory


ENTRY = "src/agents/tabular_input_worker.py"
WORKER_FILES = tuple(dict.fromkeys(BASE_FILES + (
    "src/preprocessing/canonical_data.py", "src/preprocessing/tabular_bundle.py",
    "src/preprocessing/tabular_loaders.py", "src/agents/strict_json.py", ENTRY,
)))
CONTROL_FILES = tuple(dict.fromkeys(BASE_CONTROLS + WORKER_FILES + (
    "src/agents/tabular_input_transport.py", "src/agents/tabular_input_results.py",
    "src/agents/tabular_input_service.py", "src/agents/tabular_input_flow.py",
    "src/agents/tabular_source_contract.py", "src/agents/task_orchestrator.py", "src/agents/task_store.py",
    "src/agents/task_consent.py", "src/agents/delegated_approval.py", "src/agents/split_evidence.py",
    "src/agents/workflow_approval.py",
)))


def code_manifest():
    """입력 승인·계산·결과 검증에 관여하는 고정 파일의 지문을 반환한다."""
    project = Path(__file__).resolve().parents[2]
    return {name: _fingerprint(project / name)["sha256"] for name in CONTROL_FILES}


def run_input_worker(request, root, cancel_event):
    """격리 대체 없이 기존 런타임·ACL·Job 제한으로 입력 worker를 실행한다."""
    root = _root(root)
    project = Path(__file__).resolve().parents[2]
    container = AppContainer("waferguard.tabularinput." + request.task_id)
    outcome = {"request_id": request.request_id, "profile_cleanup": False}
    try:
        before = code_manifest()
        files = runtime_files(Path(sys.prefix))
        sid = container.create()
        for name in ("runtime", "input", "tools"):
            _acl(root / name, sid, "RX")
        for path in (request.output_directory, request.scratch_directory):
            prepare_writable_directory(root, Path(path), sid)
        for source in files:
            target = root / "runtime" / source.relative_to(sys.prefix)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        for name in WORKER_FILES:
            target = root / "tools" / ("worker.py" if name == ENTRY else name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project / name, target)
            if _fingerprint(target)["sha256"] != before[name]:
                raise ValueError("복사한 입력 worker 코드가 승인 범위와 다릅니다.")
        if before != code_manifest():
            raise ValueError("입력 worker 준비 중 코드가 바뀌었습니다.")
        if cancel_event.is_set():
            outcome["launch"] = {"status": "cancelled"}
        else:
            environment = probe_environment(root / "runtime", Path(request.scratch_directory), os.environ)
            outcome["launch"] = container.launch(
                root / "runtime/python.exe", ["-I", "-S", "-B", str(root / "tools/worker.py"), str(root), request.request_id],
                environment, Path(request.scratch_directory), limits=request.limits, cancel_event=cancel_event)
    finally:
        outcome["profile_cleanup"] = container.close()
        (root / "transport.json").write_text(json.dumps(outcome), encoding="utf-8")
    return outcome
