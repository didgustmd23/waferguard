"""snapshot 품질 검사 전용 AppContainer+Job 운송 계층. 비격리 fallback은 없다."""

import json
import os
from pathlib import Path
import shutil
import sys

from .appcontainer_native import AppContainer
from .appcontainer_probe import _acl, probe_environment, runtime_files
from .artifacts import _fingerprint, _root
from .isolated_eda_transport import WORKER_FILES as BASE_FILES, CONTROL_FILES as BASE_CONTROLS


# 기존 합성 경로는 수정하지 않는다. 새 worker의 의존성을 명시적으로 추가한다.
WORKER_FILES = BASE_FILES + (
    "src/agents/snapshot_eda_worker.py",
    "src/agents/input_snapshot.py", "src/preprocessing/profiles.py",
    "src/preprocessing/quality.py", "src/preprocessing/split.py", "src/preprocessing/transform.py",
)
CONTROL_FILES = tuple(dict.fromkeys(BASE_CONTROLS + WORKER_FILES + (
    "src/agents/eda_input_contract.py", "src/agents/snapshot_eda.py",
    "src/agents/snapshot_eda_cli.py", "src/agents/snapshot_eda_worker.py",
    "src/agents/snapshot_eda_transport.py", "src/agents/eda_result_validation.py",
    "src/agents/snapshot_audit.py", "src/agents/snapshot_audit_cli.py",
    "src/agents/snapshot_audit_worker.py", "src/agents/snapshot_audit_transport.py",
    "src/agents/quality_result_validation.py",
    "src/agents/contracts.py", "src/agents/audit.py", "src/agents/cli.py",
    "src/agents/adapters/tabular.py",
)))


def code_manifest():
    """승인 대상인 검토된 코드의 식별값을 만든다. 인증 서명은 아니다."""
    project = Path(__file__).resolve().parents[2]
    return {name: _fingerprint(project / name)["sha256"] for name in CONTROL_FILES}


def run_snapshot_audit(request, root, cancel_event):
    """새 프로필에 작업 전용 권한만 부여한다. 원본 경로를 허용하지 않는다."""
    root = _root(root)
    project = Path(__file__).resolve().parents[2]
    container = AppContainer("waferguard.snapshotaudit." + request.task_id)
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

        for relative in WORKER_FILES:
            target = root / "tools" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project / relative, target)

            if _fingerprint(target)["sha256"] != before[relative]:
                raise ValueError("복사한 worker 코드가 일치하지 않습니다.")

        entry = root / "tools" / "worker.py"
        shutil.copyfile(project / "src/agents/snapshot_audit_worker.py", entry)

        if _fingerprint(entry)["sha256"] != before["src/agents/snapshot_audit_worker.py"] or before != code_manifest():
            raise ValueError("worker 준비 중 코드가 변경되었습니다.")

        # 런타임 준비 중 취소가 도착하면 프로세스를 시작하지 않는다.
        if cancel_event.is_set():
            outcome["launch"] = {"status": "cancelled"}
        else:
            environment = probe_environment(root / "runtime", Path(request.scratch_directory), os.environ)
            outcome["launch"] = container.launch(
                root / "runtime" / "python.exe",
                ["-I", "-S", "-B", str(entry), str(root), request.request_id],
                environment, Path(request.scratch_directory),
                limits=request.limits, cancel_event=cancel_event,
            )
    finally:
        outcome["profile_cleanup"] = container.close()
        (root / "transport.json").write_text(json.dumps(outcome, indent=2), encoding="utf-8")

    return outcome
