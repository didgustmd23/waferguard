"""모듈 파이프라인의 고정 코드 복사·AppContainer·Job 실행. 비격리 대체 경로는 없다."""

import json
import os
from pathlib import Path
import shutil
import sys

from .appcontainer_native import AppContainer
from .appcontainer_probe import _acl, probe_environment, runtime_files
from .artifacts import _fingerprint, _root
from .snapshot_eda_transport import WORKER_FILES as BASE_FILES, CONTROL_FILES as BASE_CONTROLS


ENTRY = "src/agents/module_pipeline_worker.py"
WORKER_FILES = tuple(dict.fromkeys(BASE_FILES + (
    "src/agents/snapshot_eda_worker.py", "src/agents/module_pipeline_contract.py",
    "src/preprocessing/module_contracts.py", "src/preprocessing/module_registry.py",
    "src/preprocessing/builtin_modules.py", "src/preprocessing/pipeline.py",
)))
CONTROL_FILES = tuple(dict.fromkeys(BASE_CONTROLS + WORKER_FILES + (
    ENTRY, "src/agents/module_pipeline.py", "src/agents/module_pipeline_cli.py",
    "src/agents/module_pipeline_transport.py", "src/agents/module_pipeline_results.py",
    "src/agents/quality_result_validation.py",
    # 부모의 계획 검토·설정 검증 코드도 승인 코드 지문에 포함한다.
    # worker에 이 파일들의 읽기 권한을 추가하는 목록은 아니다.
    "src/preprocessing/metadata_registry.py", "src/agents/module_catalogue.py",
    "src/agents/reviewed_csv_pipeline_contract.py",
    "configs/agents/modules/missing_row_histogram.json",
    "configs/agents/modules/missing_row_histogram_settings.json",
)))


def code_manifest():
    """승인 시점의 고정 코드 목록을 식별한다. 사용자 모듈을 자동 탐색하지 않는다."""
    project = Path(__file__).resolve().parents[2]

    return {name: _fingerprint(project / name)["sha256"] for name in CONTROL_FILES}


def run_pipeline_worker(request, root, cancel_event):
    """기존 격리 규칙을 유지하고 새 진입점만 고정한다. 승인 DB·원본 권한은 추가하지 않는다."""
    root = _root(root)
    project = Path(__file__).resolve().parents[2]
    container = AppContainer("waferguard.modulepipeline." + request.task_id)
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
                raise ValueError("복사한 모듈 실행 코드가 승인 대상과 다릅니다.")

        if before != code_manifest():
            raise ValueError("worker 준비 중 코드가 변경되었습니다.")

        if cancel_event.is_set():
            outcome["launch"] = {"status": "cancelled"}
        else:
            environment = probe_environment(root / "runtime", Path(request.scratch_directory), os.environ)
            outcome["launch"] = container.launch(
                root / "runtime" / "python.exe",
                ["-I", "-S", "-B", str(root / "tools" / "worker.py"), str(root), request.request_id],
                environment, Path(request.scratch_directory), limits=request.limits, cancel_event=cancel_event,
            )
    finally:
        outcome["profile_cleanup"] = container.close()
        (root / "transport.json").write_text(json.dumps(outcome), encoding="utf-8")

    return outcome
