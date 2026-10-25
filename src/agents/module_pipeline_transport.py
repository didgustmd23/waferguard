"""모듈 파이프라인의 고정 코드 복사·AppContainer·Job 실행. 비격리 대체 경로는 없다."""

import json
import os
from pathlib import Path
import shutil
import sys

from .appcontainer_native import AppContainer
from .operation_progress import report_progress
from .appcontainer_probe import _acl, probe_environment, runtime_files
from .artifacts import _fingerprint, _root
from .writable_directory_acl import prepare_writable_directory
from .snapshot_eda_transport import WORKER_FILES as BASE_FILES, CONTROL_FILES as BASE_CONTROLS


ENTRY = "src/agents/module_pipeline_worker.py"
WORKER_FILES = tuple(dict.fromkeys(BASE_FILES + (
    "src/agents/snapshot_eda_worker.py", "src/agents/module_pipeline_contract.py",
    "src/preprocessing/module_contracts.py", "src/preprocessing/module_registry.py",
    "src/preprocessing/builtin_modules.py", "src/preprocessing/pipeline.py",
    "src/preprocessing/input_profile.py",
    "src/preprocessing/canonical_data.py", "src/preprocessing/tabular_bundle.py", "src/preprocessing/tabular_loaders.py",
    "src/agents/tabular_worker_input.py", "src/agents/strict_json.py",
)))
CONTROL_FILES = tuple(dict.fromkeys(BASE_CONTROLS + WORKER_FILES + (
    ENTRY, "src/agents/module_pipeline.py", "src/agents/module_pipeline_cli.py",
    "src/agents/module_pipeline_transport.py", "src/agents/module_pipeline_results.py",
    "src/agents/quality_result_validation.py",
    "src/agents/writable_directory_acl.py",
    # 부모의 계획 검토·설정 검증 코드도 승인 코드 지문에 포함한다.
    # worker에 이 파일들의 읽기 권한을 추가하는 목록은 아니다.
    "src/preprocessing/metadata_registry.py", "src/agents/module_catalogue.py",
    "src/agents/reviewed_csv_pipeline_contract.py", "src/agents/input_profile_pipeline.py",
    # 승인 선택 재검증·바인더 경로 선택·승인 단계·실행 직전 바인딩 대조 코드다.
    # 승인 뒤 이 경로가 바뀌면 같은 승인으로 실행하지 않는다.
    "src/agents/pipeline_approval_cli.py", "src/agents/pipeline_binding_registry.py",
    "src/agents/pipeline_approval.py", "src/agents/pipeline_csv_execution.py",
    # 입력 진단 승인을 위임 발급하는 작업 조정기·동의 판정 코드와 조정기가 쓰는 초안·확인 검사
    "src/agents/task_orchestrator.py", "src/agents/task_consent.py", "src/agents/task_store.py",
    "src/agents/llm_task_assist.py", "src/agents/task_adapters.py", "src/agents/tabular_task.py",
    "src/agents/task_plan.py", "src/agents/task_recommend.py", "src/agents/delegated_approval.py",
    "src/agents/strict_json.py",
    # v2 표형 어댑터는 생성된 표의 진단 위임 범위(결정 A)와 대조값을 정한다.
    "src/agents/tabular_v2_task.py",
    "src/agents/tabular_pipeline.py", "src/agents/tabular_source_contract.py", "src/agents/split_evidence.py",
    "src/agents/tabular_input_worker.py", "src/agents/workflow_approval.py",
    "configs/agents/modules/missing_row_histogram.json",
    "configs/agents/modules/missing_row_histogram_settings.json",
)))


def code_manifest():
    """승인 시점의 고정 코드 목록을 식별한다. 사용자 모듈을 자동 탐색하지 않는다."""
    project = Path(__file__).resolve().parents[2]

    return {name: _fingerprint(project / name)["sha256"] for name in CONTROL_FILES}


def _failure_diagnostic(stage, error):
    """경로·원문·명령 출력을 남기지 않고 고정 단계와 숫자 오류만 기록한다."""
    categories = (PermissionError, FileNotFoundError, TimeoutError, OSError, ValueError)
    category = next((kind.__name__ for kind in categories if isinstance(error, kind)), "UNEXPECTED_EXCEPTION")
    numeric = {}

    for name in ("errno", "winerror"):
        value = getattr(error, name, None)
        numeric[name] = value if type(value) is int else None

    diagnostic = {"stage": stage, "error_type": category, **numeric}
    operation = getattr(error, "acl_operation", None)

    if operation in {
        "TARGET_CHECK", "OWNER_CHECK", "OWNER_MISMATCH", "CALLER_WRITE_OWNER",
        "WORKER_GRANT", "LOW_INTEGRITY", "WORKER_ACL",
    }:
        diagnostic["acl_operation"] = operation

    return diagnostic


def run_pipeline_worker(request, root, cancel_event):
    """기존 격리 규칙을 유지하고 새 진입점만 고정한다. 승인 DB·원본 권한은 추가하지 않는다."""
    root = _root(root)
    project = Path(__file__).resolve().parents[2]
    container = None
    outcome = {"request_id": request.request_id, "profile_cleanup": False}
    stage = "CONTAINER_INITIALIZE"

    try:
        container = AppContainer("waferguard.modulepipeline." + request.task_id)
        stage = "CODE_MANIFEST"
        report_progress("contract_check")
        before = code_manifest()
        stage = "RUNTIME_INVENTORY"
        report_progress("runtime_inventory")
        files = runtime_files(Path(sys.prefix))
        stage = "CONTAINER_CREATE"
        report_progress("profile_create")
        sid = container.create()

        report_progress("permissions")
        for name in ("runtime", "input", "tools"):
            stage = "ACL_" + name.upper()
            _acl(root / name, sid, "RX")

        stage = "ACL_OUTPUT"
        prepare_writable_directory(root, Path(request.output_directory), sid)
        stage = "ACL_SCRATCH"
        prepare_writable_directory(root, Path(request.scratch_directory), sid)

        stage = "RUNTIME_COPY"
        report_progress("runtime_copy")
        copied_bytes = 0
        for index, source in enumerate(files, start=1):
            target = root / "runtime" / source.relative_to(sys.prefix)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            copied_bytes += target.stat().st_size
            report_progress("runtime_copy", completed=index, total=len(files), bytes_done=copied_bytes)

        stage = "CODE_COPY"
        report_progress("tools_copy")
        for name in WORKER_FILES + (ENTRY,):
            target = root / "tools" / ("worker.py" if name == ENTRY else name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project / name, target)

            if _fingerprint(target)["sha256"] != before[name]:
                raise ValueError("복사한 모듈 실행 코드가 승인 대상과 다릅니다.")

        stage = "CODE_RECHECK"
        if before != code_manifest():
            raise ValueError("worker 준비 중 코드가 변경되었습니다.")

        stage = "CANCEL_CHECK"
        if cancel_event.is_set():
            outcome["launch"] = {"status": "cancelled"}
        else:
            stage = "ENVIRONMENT"
            environment = probe_environment(root / "runtime", Path(request.scratch_directory), os.environ)
            stage = "LAUNCH"
            report_progress("worker_wait")
            outcome["launch"] = container.launch(
                root / "runtime" / "python.exe",
                ["-I", "-S", "-B", str(root / "tools" / "worker.py"), str(root), request.request_id],
                environment, Path(request.scratch_directory), limits=request.limits, cancel_event=cancel_event,
            )
    except Exception as error:
        outcome["diagnostic"] = _failure_diagnostic(stage, error)
        raise
    finally:
        try:
            report_progress("cleanup")
            outcome["profile_cleanup"] = container.close() if container is not None else True
        except Exception as error:
            # 정리 예외로 원래 실패 진단을 덮어쓰지 않는다. 정리 확인이 없으면
            # 상위 실행기는 결과를 성공으로 커밋할 수 없다.
            outcome["cleanup_diagnostic"] = _failure_diagnostic("CONTAINER_CLEANUP", error)
            outcome["profile_cleanup"] = False

        (root / "transport.json").write_text(json.dumps(outcome), encoding="utf-8")

    return outcome
