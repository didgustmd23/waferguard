"""Python 3.12 CSV 합성 격리 검사. 사용자 데이터·기존 승인·지원 정책을 변경하지 않는다."""

import argparse
from dataclasses import asdict
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

from .appcontainer_native import AppContainer
from .appcontainer_probe import _acl, probe_environment, runtime_files
from .artifacts import _fingerprint, _root
from .csv_runtime_probe_tasks import STAGES, fixture, prepare_task
from .csv_runtime_probe_checks import validate_reply, verify_outputs
from .split_evidence import read_metadata
from .worker_contract import WorkerLimits
from .wm_cpu_probe import ProbeProgress  # 출력만 재사용한다. WM 처리·PyTorch 검사는 호출하지 않는다.


FAMILIES = {"split": "snapshot_split", "export": "partition_export",
            "candidate": "preparation_candidate", "apply": "preparation_apply"}
DRIVER = "src/agents/csv_runtime_probe_worker.py"
TOKEN_CHECK = "src/agents/pickle_import_worker.py"
MAX_RUNTIME_BYTES = 512 * 1024 ** 2


def check_environment():
    if os.name != "nt" or sys.version_info[:2] != (3, 12):
        raise ValueError("Windows/Python 3.12 고정 합성 검사만 지원합니다.")
    import ctypes

    if not ctypes.windll.shell32.IsUserAnAdmin():
        raise ValueError("CSV_PROBE_ADMIN_REQUIRED: 같은 환경의 관리자 PowerShell에서 실행하세요.")


def copy_families(root, project):
    """각 worker의 기존 복사 목록을 분리한다. 다른 worker의 추가 파일로 누락을 숨기지 않는다."""
    manifests = {}
    for family, identifier in FAMILIES.items():
        transport = importlib.import_module("src.agents." + identifier + "_transport")
        files = tuple(dict.fromkeys((*transport.WORKER_FILES, transport.ENTRY, TOKEN_CHECK, DRIVER)))
        manifests[family] = {name: _fingerprint(project / name) for name in files}
        for name, record in manifests[family].items():
            target = root / "tools" / family / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project / name, target)
            if _fingerprint(target) != record:
                raise ValueError("합성 검사 코드 복사본이 일치하지 않습니다.")
    return manifests


def control_manifest(project):
    names = {DRIVER, TOKEN_CHECK, "src/agents/csv_runtime_probe.py", "src/agents/csv_runtime_probe_tasks.py",
             "src/agents/csv_runtime_probe_checks.py"}
    for identifier in FAMILIES.values():
        transport = importlib.import_module("src.agents." + identifier + "_transport")
        names.update(transport.CONTROL_FILES)
    return {name: _fingerprint(project / name) for name in sorted(names)}


def _new_root():
    project = _root(Path(__file__).resolve().parents[2])
    parent = project
    for part in ("logs", "agents", "csv_runtime_probes"):
        parent = parent / part
        parent.mkdir(exist_ok=True)
        _root(parent)
    root = parent / uuid.uuid4().hex
    root.mkdir()
    return root


def run_probe(limits, max_runtime_bytes=MAX_RUNTIME_BYTES):
    check_environment()
    if not isinstance(limits, WorkerLimits) or type(max_runtime_bytes) is not int or not 1 <= max_runtime_bytes <= MAX_RUNTIME_BYTES:
        raise ValueError("자원 계약과 512 MiB 이하 런타임 복사 예산이 필요합니다.")
    project = _root(Path(__file__).resolve().parents[2])
    root = _new_root()
    for name in ("runtime", "tools", "tasks"):
        (root / name).mkdir()
    result = {"schema_version": "csv_runtime_probe_v1", "probe_id": root.name, "status": "review_required",
        "python": list(sys.version_info[:3]), "limits": asdict(limits), "max_runtime_bytes": max_runtime_bytes,
        "real_data_read": False, "production_worker_enabled": False, "model_trained": False,
        "network_evidence_retested": False, "scope": "fixed_synthetic_original_worker_calls_and_copy_manifests",
        "jobs": [], "profile_cleanup": True}
    progress = ProbeProgress()
    progress.start()
    outputs = {}
    stage = "RUNTIME_INVENTORY"
    try:
        controls = control_manifest(project)
        result["control_manifest"] = controls
        progress.stage("CSV 런타임 목록·크기 확인")
        files = runtime_files(Path(sys.prefix))
        total = sum(path.stat().st_size for path in files)
        if total > max_runtime_bytes:
            raise ValueError("런타임 복사 상한을 초과했습니다.")
        manifest = {path.relative_to(sys.prefix).as_posix(): _fingerprint(path) for path in files}
        result["runtime_bytes"] = total
        (root / "runtime_files.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        stage = "RUNTIME_COPY"
        progress.stage("CSV 격리 런타임 복사·해시 검사 (한 번만 복사)")
        copied = 0
        for index, source in enumerate(files, start=1):
            relative = source.relative_to(sys.prefix)
            target = root / "runtime" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if _fingerprint(target) != manifest[relative.as_posix()]:
                raise ValueError("런타임 복사본이 일치하지 않습니다.")
            copied += manifest[relative.as_posix()]["size"]
            progress.update(index, len(files), copied, total)
        stage = "CODE_COPY"
        progress.stage("각 worker의 분리된 복사 코드 확인")
        code = copy_families(root, project)
        result["code_manifests"] = code
        source = fixture(root)
        source_before = _fingerprint(source)

        for selected in STAGES:
            stage = selected.upper()
            progress.stage("CSV 합성 단계: " + selected)
            task, family, request = prepare_task(root, selected, outputs, limits)
            # 한 단계에 한 AppContainer·한 Job을 사용한다. 이전 단계 권한을 재사용하지 않는다.
            container = AppContainer("waferguard.csvprobe." + task.name)
            job = {"stage": selected, "family": family, "request_id": request.request_id, "profile_cleanup": False, "passed": False}
            result["jobs"].append(job)
            try:
                sid = container.create()
                for path in (root / "runtime", root / "tools" / family, task / "input"):
                    _acl(path, sid, "RX")
                # receipt와 worker 결과가 함께 기록되는 작업 전용 출력만 허용한다.
                _acl(task / "output", sid, "M", low=True)
                _acl(task / "output/result", sid, "M", low=True)
                _acl(task / "scratch/task", sid, "M", low=True)
                for name, expected in code[family].items():
                    if _fingerprint(project / name) != expected or _fingerprint(root / "tools" / family / name) != expected:
                        raise ValueError("실행 전에 검사 코드가 변경됐습니다.")
                environment = probe_environment(root / "runtime", task / "scratch/task", os.environ)
                environment["CUDA_VISIBLE_DEVICES"] = "-1"
                environment["MPLBACKEND"] = "Agg"
                job["launch"] = container.launch(root / "runtime/python.exe",
                    ["-I", "-S", "-B", str(root / "tools" / family / DRIVER), str(task), request.request_id, family],
                    environment, task / "scratch/task", limits=limits)
                job["passed"] = validate_reply(root, task, family, request, job["launch"])
                if not job["passed"]:
                    raise ValueError("합성 worker의 응답·종료·자원 검사에 실패했습니다.")
                outputs[selected] = task / "output/result"
            finally:
                try:
                    job["profile_cleanup"] = container.close()
                except Exception:
                    result["profile_cleanup"] = False
                    raise
                if not job["profile_cleanup"]:
                    result["profile_cleanup"] = False
            if not job["profile_cleanup"]:
                raise ValueError("AppContainer 정리를 확인하지 못했습니다.")

        stage = "PARENT_RESULT_CHECK"
        progress.stage("합성 결과·Train 기준·변환 수치·행 연결 검증")
        result["checks"] = verify_outputs(root, outputs)
        result["checks"]["synthetic_source_unchanged"] = _fingerprint(source) == source_before
        progress.stage("검사 코드·런타임 복사본 최종 해시 확인")
        result["checks"]["control_code_unchanged"] = control_manifest(project) == controls
        result["checks"]["runtime_copies_unchanged"] = all(_fingerprint(root / "runtime" / name) == record for name, record in manifest.items())
        if all(result["checks"].values()) and len(result["jobs"]) == len(STAGES) and result["profile_cleanup"]:
            result["status"] = "csv_compatibility_passed"
    except Exception as error:
        result.update(failure_stage=stage, failure_code="CSV_SYNTHETIC_CHECK_FAILED")
        number = getattr(error, "winerror", None) or getattr(error, "errno", None)
        if type(number) is int:
            result["os_error"] = number
    finally:
        progress.close()
        (root / "result.json").write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    return root, result


def main():
    parser = argparse.ArgumentParser(description="고정 합성 CSV worker 4종의 Python 3.12 격리 검사. 실데이터·자동 지원 활성화 없음")
    parser.add_argument("--confirm-synthetic-csv-probe", action="store_true")
    parser.add_argument("--memory-mib", type=int, default=2048)
    parser.add_argument("--processes", type=int, default=2)
    parser.add_argument("--wall-seconds", type=int, default=300, help="각 자식 실행의 시간 상한이며 전체 검사 시간 추정이 아닙니다.")
    parser.add_argument("--max-runtime-bytes", type=int, default=MAX_RUNTIME_BYTES)
    args = parser.parse_args()
    if not args.confirm_synthetic_csv_probe:
        parser.error("고정 합성 입력·런타임 복사·일회성 AppContainer 생성 확인이 필요합니다.")
    try:
        root, result = run_probe(WorkerLimits(args.wall_seconds, args.memory_mib, args.processes), args.max_runtime_bytes)
    except ValueError as error:
        parser.error(str(error))
    print(result["status"])
    if result.get("failure_stage"):
        print("실패 단계:", result["failure_stage"], "/ 코드:", result["failure_code"])
    print("로컬 검사 결과:", root / "result.json")
    print("합성 검사입니다. 실데이터 실행·CSV 3.12 지원·네트워크 보안 재인증을 활성화하지 않았습니다.")
    return 0 if result["status"] == "csv_compatibility_passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
