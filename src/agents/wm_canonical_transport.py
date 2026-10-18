"""기존 AppContainer·Job으로 구조 판별형 WM canonical worker를 실행한다."""

import json
import os
from pathlib import Path
import shutil
import sys

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
                raise ValueError("복사한 WM canonical worker 코드가 승인 대상과 다릅니다.")

        if before != code_manifest():
            raise ValueError("WM canonical worker 준비 중 코드가 변경되었습니다.")

        if cancel_event.is_set():
            outcome["launch"] = {"status": "cancelled"}
        else:
            environment = probe_environment(root / "runtime", Path(request.scratch_directory), os.environ)
            outcome["launch"] = container.launch(
                root / "runtime/python.exe",
                ["-I", "-S", "-B", str(root / "tools/worker.py"), str(root), request.request_id],
                environment, Path(request.scratch_directory),
                limits=request.limits, cancel_event=cancel_event,
            )
    finally:
        outcome["profile_cleanup"] = container.close()
        (root / "transport.json").write_text(json.dumps(outcome), encoding="utf-8")

    return outcome
