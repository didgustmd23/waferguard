"""고정 표형 입력 worker의 기존 AppContainer·Job 운송 연결."""

from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil

from .appcontainer_native import AppContainer
from .appcontainer_probe import _acl, probe_environment
from .artifacts import _fingerprint, _root
from .module_pipeline_transport import WORKER_FILES as BASE_FILES, CONTROL_FILES as BASE_CONTROLS
from .protected_store import READ_EXECUTE, check_object, check_tree, trusted_principals
from .worker_paths import WorkerPathPolicy
from .worker_runtime import RuntimeUnavailable, runtime_lease
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
    "src/agents/tabular_v2_task.py",
    "src/agents/task_consent.py", "src/agents/delegated_approval.py", "src/agents/split_evidence.py",
    "src/agents/workflow_approval.py", "src/agents/worker_runtime.py", "src/agents/protected_store.py",
)))


def code_manifest():
    """입력 승인·계산·결과 검증에 관여하는 고정 파일의 지문을 반환한다."""
    project = Path(__file__).resolve().parents[2]
    return {name: _fingerprint(project / name)["sha256"] for name in CONTROL_FILES}


# icacls M(Modify) = 0x1301BF. 작업 SID는 output/result·scratch/task에서 이 범위 안의 권한만 가진다.
MODIFY = 0x1301BF


def _job_permissions_ok(root, request, sid):
    """실행 직전 작업 폴더의 의미상 권한을 확인한다.

    루트·output·scratch 상위·transport 기록은 신뢰 주체만, input·tools(코드·입력)는 작업 SID 읽기·실행 이하,
    output/result·scratch/task는 작업 SID Modify 이하와 Low 레이블만 허용한다. 다른 주체의 ACE는 모두 거부한다.
    """
    trusted = trusted_principals()
    return (all(check_object(path, trusted) for path in (root, root / "output", root / "scratch"))
            and all(check_tree(root / name, trusted, extra={sid: READ_EXECUTE}) for name in ("input", "tools"))
            and all(check_tree(Path(path), trusted, extra={sid: MODIFY}, low_allowed=True)
                    for path in (request.output_directory, request.scratch_directory)))


def run_input_worker(request, root, cancel_event, *, runtime_id, runtime_base=None, entry=ENTRY, worker_files=WORKER_FILES,
                     manifest=code_manifest, container_prefix="waferguard.tabularinput."):
    """격리 대체 없이 공유 런타임·ACL·Job 제한으로 입력 worker를 실행한다.

    runtime_id는 승인한 packet의 런타임 ID이며 위치는 코드 상수의 저장소에서만 정해진다. 검증부터 worker 종료까지
    버전 폴더를 보유한다(lease). 준비·검증에 실패하면 원본 환경으로 대신 실행하지 않고 transport.json에 고정 코드만 남긴다.
    entry·worker_files·manifest·container_prefix는 같은 운송을 쓰는 다른 표형 worker(시간 복원)의 고정 코드 목록이다. 기본값은 입력 worker다.
    """
    root = _root(root)
    project = Path(__file__).resolve().parents[2]
    container = AppContainer(container_prefix + request.task_id)
    outcome = {"request_id": request.request_id, "profile_cleanup": False}
    try:
        before = manifest()
        with ExitStack() as lease:
            try:
                runtime = lease.enter_context(runtime_lease(runtime_id, base=runtime_base, cancel_event=cancel_event))
            except RuntimeUnavailable as error:
                if error.code == "RUNTIME_CANCELLED":
                    outcome["launch"] = {"status": "cancelled"}
                else:
                    outcome["runtime_error"] = error.code
                return outcome
            outcome["runtime"] = {"id": runtime.runtime_id, "prepared": runtime.prepared, "timings": runtime.timings}
            # 공유 런타임과 작업 폴더가 서로 겹치거나 링크를 지나지 않는지 기존 경로 정책으로 확인한다.
            WorkerPathPolicy((runtime.directory, root / "input", root / "tools"), root / "output", root / "scratch")
            if runtime.directory in root.parents or root in runtime.directory.parents:
                raise ValueError("공유 런타임과 작업 폴더가 겹칩니다.")
            sid = container.create()
            for name in ("input", "tools"):
                _acl(root / name, sid, "RX")
            for path in (request.output_directory, request.scratch_directory):
                prepare_writable_directory(root, Path(path), sid)
            for name in worker_files:
                target = root / "tools" / ("worker.py" if name == entry else name)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(project / name, target)
                if _fingerprint(target)["sha256"] != before[name]:
                    raise ValueError("복사한 입력 worker 코드가 승인 범위와 다릅니다.")
            if before != manifest():
                raise ValueError("입력 worker 준비 중 코드가 바뀌었습니다.")
            if not _job_permissions_ok(root, request, sid):
                # 작업 폴더에 다른 주체·넓은 권한이 있으면 실행하지 않는다. 권한을 고치지 않는다.
                outcome["runtime_error"] = "INPUT_STORE_PERMISSION_INVALID"
            elif cancel_event.is_set():
                outcome["launch"] = {"status": "cancelled"}
            else:
                environment = probe_environment(runtime.python.parent, Path(request.scratch_directory), os.environ)
                outcome["launch"] = container.launch(
                    runtime.python, ["-I", "-S", "-B", str(root / "tools/worker.py"), str(root), request.request_id, runtime.runtime_id],
                    environment, Path(request.scratch_directory), limits=request.limits, cancel_event=cancel_event,
                    capabilities=(runtime.capability,))
    finally:
        outcome["profile_cleanup"] = container.close()
        (root / "transport.json").write_text(json.dumps(outcome), encoding="utf-8")
    return outcome
