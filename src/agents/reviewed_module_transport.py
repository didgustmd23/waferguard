"""참조 모듈의 고정 소스·명세를 새 AppContainer+Job에 전달한다. 대체 실행 없음."""

import json
import os
from pathlib import Path
import shutil
import sys

from .appcontainer_native import AppContainer
from .appcontainer_probe import _acl, probe_environment, runtime_files
from .artifacts import _root, _fingerprint
from .module_catalogue import SOURCE_FILES, DESCRIPTOR
from .snapshot_eda_transport import WORKER_FILES as BASE_FILES, CONTROL_FILES as BASE_CONTROLS


ENTRY = "src/agents/reviewed_module_worker.py"
WORKER_FILES = tuple(dict.fromkeys(BASE_FILES + SOURCE_FILES + (DESCRIPTOR,
    "src/agents/module_catalogue.py", "src/agents/reviewed_module_contract.py",
    "src/agents/snapshot_eda_worker.py", "src/preprocessing/builtin_modules.py",
)))
CONTROL_FILES = tuple(dict.fromkeys(BASE_CONTROLS + WORKER_FILES + (
    ENTRY, "src/agents/reviewed_module_transport.py", "src/agents/reviewed_module.py",
    "src/agents/reviewed_module_cli.py", "src/agents/reviewed_module_results.py",
    "src/agents/quality_result_validation.py",
)))


def code_manifest():
    root = Path(__file__).resolve().parents[2]
    return {name: _fingerprint(root / name)["sha256"] for name in CONTROL_FILES}


def run_reviewed_worker(request, root, cancel_event):
    """원본·DB 권한을 추가하지 않고 승인된 고정 코드만 복사한다."""
    root = _root(root)
    project = Path(__file__).resolve().parents[2]
    container = AppContainer("waferguard.reviewedmodule." + request.task_id)
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

        for relative in WORKER_FILES + (ENTRY,):
            target = root / "tools" / ("worker.py" if relative == ENTRY else relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project / relative, target)

            if _fingerprint(target)["sha256"] != before[relative]:
                raise ValueError("복사한 참조 모듈·명세가 승인 대상과 다릅니다.")

        if code_manifest() != before:
            raise ValueError("worker 준비 중 코드가 변경되었습니다.")

        if cancel_event.is_set():
            outcome["launch"] = {"status": "cancelled"}
        else:
            environment = probe_environment(root / "runtime", Path(request.scratch_directory), os.environ)
            outcome["launch"] = container.launch(root / "runtime/python.exe",
                ["-I", "-S", "-B", str(root / "tools/worker.py"), str(root), request.request_id],
                environment, Path(request.scratch_directory), limits=request.limits, cancel_event=cancel_event)
    finally:
        outcome["profile_cleanup"] = container.close()
        (root / "transport.json").write_text(json.dumps(outcome), encoding="utf-8")

    return outcome
