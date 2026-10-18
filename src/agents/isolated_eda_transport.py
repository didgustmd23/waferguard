"""고정 합성 EDA의 AppContainer+Job 실행. 테스트용 비격리 fallback은 제공하지 않는다."""

import json
from pathlib import Path
import shutil
import sys

from .appcontainer_native import AppContainer
from .appcontainer_probe import _acl, probe_environment, runtime_files
from .artifacts import _fingerprint, _root


# worker에는 공통 코어와 요청 계약에 필요한 고정 파일만 복사한다.
WORKER_FILES = (
    "src/__init__.py", "src/sensor_ml/__init__.py", "src/agents/__init__.py", "src/preprocessing/__init__.py",
    "src/preprocessing/contracts.py", "src/preprocessing/eda.py", "src/preprocessing/workflows.py",
    "src/preprocessing/eda_report.py",
    "src/agents/synthetic_eda_data.py", "src/agents/worker_contract.py",
    "src/agents/worker_paths.py", "src/agents/workflow.py", "src/agents/artifacts.py",
)
CONTROL_FILES = WORKER_FILES + (
    "src/agents/eda_worker.py", "src/agents/isolated_eda_transport.py",
    "src/agents/isolated_eda.py", "src/agents/job_object.py", "src/agents/appcontainer_native.py",
    "src/agents/appcontainer_probe.py", "src/agents/executor.py",
    "src/agents/approvals.py", "src/agents/approval_store.py",
    "src/agents/isolated_eda_cli.py", "src/agents/state.py", "src/agents/policy.py",
)


def code_manifest():
    """승인 후 관련 소스가 바뀌면 기존 계획으로 실행하지 않도록 식별한다."""
    project = Path(__file__).resolve().parents[2]

    return {name: _fingerprint(project / name)["sha256"] for name in CONTROL_FILES}


def run_isolated_eda(request, root, cancel_event):
    """새 프로필과 승인한 작업 폴더만 사용한다. 실제 데이터 경로 인자는 받지 않는다."""
    root = _root(root)
    project = Path(__file__).resolve().parents[2]
    container = AppContainer("waferguard.eda." + request.task_id)
    outcome = {"request_id": request.request_id, "status": "failed_runtime", "profile_cleanup": False}

    try:
        files = runtime_files(Path(sys.prefix))
        before = code_manifest()
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
                raise ValueError("복사한 worker 코드가 승인 대상과 일치하지 않습니다.")

        entry = root / "tools" / "worker.py"
        shutil.copyfile(project / "src/agents/eda_worker.py", entry)

        if _fingerprint(entry)["sha256"] != before["src/agents/eda_worker.py"] or before != code_manifest():
            raise ValueError("준비 중 worker 소스가 변경되었습니다.")

        import os

        environment = probe_environment(root / "runtime", Path(request.scratch_directory), os.environ)
        outcome["launch"] = container.launch(
            root / "runtime" / "python.exe",
            ["-I", "-S", "-B", str(entry), str(root), request.request_id],
            environment, Path(request.scratch_directory),
            limits=request.limits, cancel_event=cancel_event,
        )
        outcome["status"] = "returned"
    finally:
        outcome["profile_cleanup"] = container.close()
        (root / "transport.json").write_text(json.dumps(outcome, ensure_ascii=False, indent=2), encoding="utf-8")

    return outcome
