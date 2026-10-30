"""배열 통합 작업 worker 전용 코드 허용 목록과 AppContainer+Job 운송. 비격리 대체 경로는 없다."""

import json
import os
from pathlib import Path
import shutil
import sys

from .appcontainer_native import AppContainer
from .appcontainer_probe import _acl, probe_environment, runtime_files
from .artifacts import _fingerprint, _root
from .isolated_eda_transport import WORKER_FILES as BASE_FILES, CONTROL_FILES as BASE_CONTROLS


ENTRY = "src/agents/array_task_worker.py"
WORKER_FILES = tuple(dict.fromkeys(BASE_FILES + (
    "src/agents/input_snapshot.py", "src/agents/snapshot_eda_worker.py", "src/agents/strict_json.py",
    "src/preprocessing/split.py", "src/preprocessing/array_dataset.py",
)))
# 조정기·어댑터·결과 검증기는 위임 발급과 결과 등록을 맡으므로 같은 코드 지문에 묶는다.
CONTROL_FILES = tuple(dict.fromkeys(BASE_CONTROLS + WORKER_FILES + (
    ENTRY, "src/agents/array_task_transport.py", "src/agents/array_task_service.py", "src/agents/array_task_results.py",
    "src/agents/array_task.py", "src/agents/array_options.py", "src/agents/task_orchestrator.py",
    "src/agents/task_consent.py", "src/agents/task_store.py", "src/agents/protected_store.py", "src/agents/task_plan.py", "src/agents/task_recommend.py",
    "src/agents/task_adapters.py", "src/agents/delegated_approval.py", "src/agents/workflow_approval.py",
    "src/agents/snapshot_eda.py", "src/agents/audit.py", "src/agents/cli.py", "src/agents/task_report.py",
)))


def code_manifest():
    """승인 시점의 검토된 소스만 식별한다. 임의 모듈 탐색·다운로드는 하지 않는다."""
    project = Path(__file__).resolve().parents[2]

    return {name: _fingerprint(project / name)["sha256"] for name in CONTROL_FILES}


def run_array_worker(request, root, cancel_event):
    """새 AppContainer에 작업 복사본만 허용하고 종료 후 프로필을 정리한다."""
    root = _root(root)
    project = Path(__file__).resolve().parents[2]
    container = AppContainer("waferguard.arraytask." + request.task_id)
    outcome = {"request_id": request.request_id, "profile_cleanup": False}

    try:
        before = code_manifest()
        files = runtime_files(Path(sys.prefix))
        sid = container.create()

        for name in ("runtime", "input", "tools"):
            _acl(root / name, sid, "RX")

        _acl(Path(request.output_directory), sid, "M", low=True)
        _acl(Path(request.scratch_directory), sid, "M", low=True)

        for source in files:
            target = root / "runtime" / source.relative_to(sys.prefix)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)

        for name in WORKER_FILES + (ENTRY,):
            target = root / "tools" / ("worker.py" if name == ENTRY else name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project / name, target)

            if _fingerprint(target)["sha256"] != before[name]:
                raise ValueError("복사한 worker 코드가 승인 대상과 다릅니다.")

        if before != code_manifest():
            raise ValueError("worker 준비 중 코드가 변경되었습니다.")

        if cancel_event.is_set():
            outcome["launch"] = {"status": "cancelled"}
        else:
            environment = probe_environment(root / "runtime", Path(request.scratch_directory), os.environ)
            outcome["launch"] = container.launch(
                root / "runtime" / "python.exe",
                ["-I", "-S", "-B", str(root / "tools/worker.py"), str(root), request.request_id],
                environment, Path(request.scratch_directory), limits=request.limits, cancel_event=cancel_event,
            )
    finally:
        outcome["profile_cleanup"] = container.close()
        (root / "transport.json").write_text(json.dumps(outcome), encoding="utf-8")

    return outcome
