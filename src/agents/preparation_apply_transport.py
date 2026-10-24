"""저장 기준 적용 전용 코드 허용 목록과 AppContainer+Job 운송. 비격리 대체 경로는 없다."""

import json
import os
from pathlib import Path
import shutil
import sys

from .appcontainer_native import AppContainer
from .appcontainer_probe import _acl, probe_environment, runtime_files
from .artifacts import _fingerprint, _root
from .snapshot_eda_transport import WORKER_FILES as BASE_FILES, CONTROL_FILES as BASE_CONTROLS


ENTRY = "src/agents/preparation_apply_worker.py"
WORKER_FILES = tuple(dict.fromkeys(BASE_FILES + (
    "src/agents/snapshot_eda_worker.py", "src/agents/partition_csv.py",
    "src/agents/preparation_apply_io.py", "src/agents/partition_export_worker.py",
    "src/agents/tabular_preparation_worker.py", "src/agents/strict_json.py",
    "src/agents/tabular_worker_input.py", "src/preprocessing/tabular_bundle.py", "src/preprocessing/canonical_data.py",
)))
CONTROL_FILES = tuple(dict.fromkeys(BASE_CONTROLS + WORKER_FILES + (
    ENTRY, "src/agents/preparation_apply.py", "src/agents/preparation_apply_cli.py",
    "src/agents/preparation_apply_transport.py", "src/agents/preparation_apply_contract.py",
    "src/agents/preparation_apply_results.py", "src/agents/split_evidence.py",
    "src/agents/preparation_candidate_contract.py", "src/agents/preparation_candidate_results.py",
    "src/agents/preparation_candidate_report.py",
    "src/agents/partition_input_contract.py",
    "src/agents/tabular_preparation.py", "src/agents/tabular_source_contract.py",
    "src/agents/tabular_input_worker.py", "src/preprocessing/tabular_loaders.py",
)))


def code_manifest():
    """승인 시점의 검토된 소스만 식별한다. 임의 모듈 탐색·다운로드는 하지 않는다."""
    project = Path(__file__).resolve().parents[2]

    return {name: _fingerprint(project / name)["sha256"] for name in CONTROL_FILES}


def run_apply_worker(request, root, cancel_event):
    """새 AppContainer에 작업 복사본만 허용하고 종료 후 프로필을 정리한다."""
    root = _root(root)
    project = Path(__file__).resolve().parents[2]
    container = AppContainer("waferguard.preparationapply." + request.task_id)
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
                raise ValueError("복사한 내보내기 코드가 승인 대상과 다릅니다.")

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
