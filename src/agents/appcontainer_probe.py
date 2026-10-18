"""합성 AppContainer 호환성 검사 CLI. 실데이터 worker를 활성화하지 않는다."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

from .appcontainer_native import AppContainer
from .artifacts import _is_link, _root
from .loopback_probe import LoopbackObserver, assess_loopback, query_loopback_exemption
from .worker_contract import WorkerLimits
from .probe_progress import read_progress


PACKAGES = (
    "numpy", "numpy.libs", "pandas", "pandas.libs", "scipy", "scipy.libs", "sklearn",
    "matplotlib", "mpl_toolkits", "contourpy", "cycler", "dateutil", "six.py", "pytz",
    "tzdata", "packaging", "pyparsing", "PIL", "kiwisolver", "joblib", "threadpoolctl.py",
    "threadpoolctl", "fontTools", "cloudpickle",
)
METADATA = (
    "numpy-", "pandas-", "scipy-", "scikit_learn-", "matplotlib-", "contourpy-", "cycler-",
    "python_dateutil-", "six-", "pytz-", "tzdata-", "packaging-", "pyparsing-", "pillow-",
    "kiwisolver-", "joblib-", "threadpoolctl-", "fonttools-", "cloudpickle-",
)
REQUIRED = {
    "appcontainer_token", "allowed_read", "allowed_write", "import_numpy", "import_pandas",
    "import_sklearn", "import_matplotlib", "numeric_pipeline", "csv_roundtrip", "agg_plot",
    "network_capabilities_absent",
}
DENIALS = {"protected_read", "input_write", "loopback_network", "external_network"}


def probe_environment(runtime, output, parent):
    """Windows 프로필 조회에 필요한 변수만 허용한다. 부모 환경 전체는 상속하지 않는다."""
    required = ("SystemRoot", "LOCALAPPDATA", "USERPROFILE")

    for name in required:
        value = parent.get(name)

        if not isinstance(value, str) or not value or "\0" in value:
            raise ValueError("검사 프로세스 시작에 필요한 Windows 환경 변수가 없습니다.")

    environment = {name: parent[name] for name in required}
    environment.update({
        "WINDIR": parent["SystemRoot"], "PATH": str(runtime),
        "TEMP": str(output), "TMP": str(output), "MPLCONFIGDIR": str(output / "mpl"),
        "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    })

    if isinstance(parent.get("APPDATA"), str) and parent["APPDATA"] and "\0" not in parent["APPDATA"]:
        environment["APPDATA"] = parent["APPDATA"]

    # 경로를 알려주는 것은 파일 접근 권한을 주는 것이 아니다.
    # 실제 접근 제한은 AppContainer token과 ACL이 계속 담당한다.
    return environment


def runtime_files(prefix):
    """기존 환경은 읽기만 하고, 고정 라이브러리와 표준 라이브러리만 복사 대상으로 선택한다."""
    prefix = _root(prefix)
    candidates = [prefix / "python.exe", *prefix.glob("*.dll")]
    standard = prefix / "Lib"

    for directory, folders, names in os.walk(standard, followlinks=False):
        folders[:] = [name for name in folders if name not in {"site-packages", "__pycache__", "test", "tests", "idlelib", "ensurepip", "turtledemo"}]
        candidates.extend(Path(directory) / name for name in names if not name.endswith((".pyc", ".pyo")))

    for root in (prefix / "DLLs",):
        if root.exists():
            candidates.extend(path for path in root.rglob("*") if path.is_file())

    candidates.extend((prefix / "Library" / "bin").glob("*.dll"))
    site = prefix / "Lib" / "site-packages"
    selected = [site / name for name in PACKAGES]
    selected.extend(path for path in site.glob("*.dist-info") if path.name.lower().startswith(METADATA))

    for item in selected:
        if item.is_dir():
            candidates.extend(path for path in item.rglob("*") if path.is_file() and "__pycache__" not in path.parts and not path.name.endswith((".pyc", ".pyo")))
        elif item.is_file():
            candidates.append(item)

    files = sorted(set(candidates))

    for path in files:
        if not path.is_file() or any(_is_link(component) for component in (path, *path.parents) if component != prefix.parent):
            raise ValueError("검사 런타임 원본에 링크 또는 누락 파일이 있습니다.")

    if sum(path.stat().st_size for path in files) > 512 * 1024**2:
        raise ValueError("검사 런타임 복사 상한 512MiB를 초과했습니다.")

    return files


def evaluate_child(value):
    """연결 시간 초과·파일 없음·잘못된 결과를 격리 통과로 처리하지 않는다."""
    if not isinstance(value, dict) or value.get("schema_version") != "appcontainer_child_v1" or value.get("real_data_read") is not False or "runtime_error" in value:
        return False

    checks = value.get("checks")

    if not isinstance(checks, dict) or any(checks.get(name) is not True for name in REQUIRED):
        return False

    for name in DENIALS:
        check = checks.get(name)

        if not isinstance(check, dict) or check.get("denied") is not True or type(check.get("error_code")) is not int or check["error_code"] not in {1, 5, 13, 10013}:
            return False

    return True


def launch_succeeded(launch):
    """자원 통계를 포함해도 정상 종료를 판정하되 정리 실패는 통과시키지 않는다."""
    if not isinstance(launch, dict) or launch.get("status") != "exited" or launch.get("exit_code") != 0:
        return False

    resources = launch.get("resources")

    return resources is None or (
        isinstance(resources, dict)
        and resources.get("status") == "succeeded"
        and resources.get("cleanup_confirmed") is True
    )


def summarize_components(child, launch, loopback):
    """영역별 결과를 분리한다. 일부 통과를 전체 격리 성공으로 표현하지 않는다."""
    checks = child.get("checks", {}) if isinstance(child, dict) else {}

    if not isinstance(checks, dict):
        checks = {}

    process_ok = launch_succeeded(launch)
    process_ok = process_ok and isinstance(child, dict) and "runtime_error" not in child
    runtime_names = (
        "import_numpy", "import_pandas", "import_sklearn", "import_matplotlib",
        "numeric_pipeline", "csv_roundtrip", "agg_plot",
    )

    def status(required, denied_names=()):
        # 허용 동작과 명시적 접근 거부를 함께 확인한다. timeout은 통과가 아니다.
        passed = process_ok and all(checks.get(name) is True for name in required)

        for name in denied_names:
            denial = checks.get(name, {})
            passed = passed and isinstance(denial, dict) and denial.get("denied") is True
            passed = passed and type(denial.get("error_code")) is int
            passed = passed and denial.get("error_code") in {1, 5, 13, 10013}

        return "passed" if passed else "review_required"

    return {
        "runtime": status(runtime_names),
        "file_access": status(("allowed_read", "allowed_write"), ("protected_read", "input_write")),
        "network_token": status(("appcontainer_token", "network_capabilities_absent")),
        "external_network": status((), ("external_network",)),
        "loopback": loopback["status"],
    }


def _acl(path, sid, permission, *, low=False):
    """새 검사 하위 폴더에만 SID 권한을 부여한다. 기존 Python·프로젝트 ACL은 바꾸지 않는다."""
    executable = Path(os.environ["SystemRoot"]) / "System32" / "icacls.exe"
    command = [str(executable), str(path), "/grant:r", f"*{sid}:(OI)(CI){permission}", "/Q"]
    result = subprocess.run(command, capture_output=True, timeout=30, check=False)

    if result.returncode:
        raise OSError(result.returncode, "PROBE_ACL_SETUP_FAILED")

    if low:
        result = subprocess.run([str(executable), str(path), "/setintegritylevel", "(OI)(CI)L", "/Q"], capture_output=True, timeout=30, check=False)

        if result.returncode:
            raise OSError(result.returncode, "PROBE_INTEGRITY_SETUP_FAILED")


def run_probe(*, wfp_capture=None, limits=None, cancel_event=None):
    """전용 새 폴더와 일회성 프로필에서만 실험한다. 기존 결과·환경을 덮어쓰지 않는다."""
    if os.name != "nt" or sys.version_info[:2] != (3, 10):
        raise ValueError("현재 probe는 Windows의 Python 3.10 환경만 지원합니다.")

    project = _root(Path(__file__).resolve().parents[2])
    parent = project

    for part in ("logs", "agents", "appcontainer_probes"):
        parent = parent / part
        parent.mkdir(exist_ok=True)
        _root(parent)

    identifier = uuid.uuid4().hex
    root = parent / identifier
    root.mkdir(exist_ok=False)
    folders = {name: root / name for name in ("runtime", "input", "output", "tools", "protected")}

    for directory in folders.values():
        directory.mkdir()

    result = {
        "schema_version": "appcontainer_probe_v1", "status": "review_required",
        "runtime_copied": False, "profile_cleanup": False, "real_data_read": False,
        "production_worker_enabled": False,
    }
    container = AppContainer("waferguard.probe." + identifier)
    observer = LoopbackObserver()
    stage = "runtime_inventory"

    try:
        # 복사량을 먼저 확인한다. 기존 환경의 파일·설치·ACL은 수정하지 않는다.
        files = runtime_files(Path(sys.prefix))
        result["runtime_bytes"] = sum(path.stat().st_size for path in files)
        stage = "profile_create"
        sid = container.create()
        stage = "acl_setup"

        for name in ("runtime", "input", "tools"):
            _acl(folders[name], sid, "RX")

        _acl(folders["output"], sid, "M", low=True)
        stage = "runtime_copy"

        for source in files:
            target = folders["runtime"] / source.relative_to(sys.prefix)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)

        shutil.copyfile(Path(__file__).with_name("appcontainer_child.py"), folders["tools"] / "probe.py")
        shutil.copyfile(Path(__file__).with_name("probe_progress.py"), folders["tools"] / "probe_progress.py")
        result["runtime_copied"] = True
        stage = "synthetic_input"
        (folders["input"] / "synthetic.txt").write_text("synthetic_only", encoding="utf-8")
        (folders["protected"] / "secret.txt").write_text("synthetic_secret_only", encoding="utf-8")
        observer.start()
        observer.control("before")
        result["loopback_exception_before"] = query_loopback_exemption(sid)
        (folders["input"] / "probe.json").write_text(json.dumps({
            "port": observer.port, "child_token": observer.tokens["child"],
        }), encoding="utf-8")
        stage = "environment_setup"
        environment = probe_environment(folders["runtime"], folders["output"], os.environ)

        # 별도 동의 CLI에서만 수집기가 전달된다. 기본 probe는 ETW를 시작하지 않는다.
        launch_options = {}

        if limits is not None:
            launch_options.update({"limits": limits, "cancel_event": cancel_event})

        if wfp_capture is not None:
            stage = "wfp_capture_start"
            wfp_capture.start(root, sid, observer.port)
            launch_options["on_started"] = wfp_capture.process_started

        stage = "process_launch"

        try:
            result["launch"] = container.launch(
                folders["runtime"] / "python.exe",
                ["-I", "-S", "-B", str(folders["tools"] / "probe.py"), str(root)],
                environment, folders["output"], seconds=60, **launch_options,
            )
        finally:
            # 정상 종료·프로세스 오류 모두 수집 종료를 시도한다.
            if wfp_capture is not None:
                wfp_capture.stop()
        observer.control("after")
        result["loopback_exception_after"] = query_loopback_exemption(sid)
        result["loopback_observation"] = observer.snapshot()
        child_path = folders["output"] / "child_result.json"
        stage = "result_check"
        progress_path = folders["output"] / "progress.jsonl"

        try:
            if _is_link(progress_path):
                result["progress"] = {"status": "invalid"}
            else:
                result["progress"] = read_progress(progress_path)
        except OSError:
            result["progress"] = {"status": "unavailable"}

        if child_path.is_file() and child_path.stat().st_size <= 64 * 1024:
            result["child"] = json.loads(child_path.read_text(encoding="utf-8"))
            result["loopback_assessment"] = assess_loopback(
                result["child"], result["loopback_observation"],
                result["loopback_exception_before"], result["loopback_exception_after"],
            )
            result["component_status"] = summarize_components(
                result["child"], result["launch"], result["loopback_assessment"],
            )

            if launch_succeeded(result["launch"]) and evaluate_child(result["child"]) and result["loopback_assessment"]["status"] == "passed":
                result["status"] = "compatibility_passed"
            else:
                result["failure_code"] = "COMPATIBILITY_CHECKS_NOT_PASSED"
        else:
            # 최종 파일 미생성보다 실제 종료 원인을 우선 기록한다.
            resource_status = result.get("launch", {}).get("status")
            result["failure_code"] = {
                "resource_exceeded": "PROBE_RESOURCE_EXCEEDED",
                "timed_out": "PROBE_TIMED_OUT", "cancelled": "PROBE_CANCELLED",
            }.get(resource_status, "PROBE_RESULT_MISSING")
    except Exception as error:
        # 라이브러리 예외·설치 경로를 감사 로그나 외부 응답에 복사하지 않는다.
        result["failure_code"] = "APPCONTAINER_PROBE_NOT_COMPLETE"
        result["failure_stage"] = stage

        if isinstance(error, OSError) and type(error.errno) is int:
            result["native_error_code"] = error.errno

        known_codes = {
            "PROBE_ACL_SETUP_FAILED", "PROBE_INTEGRITY_SETUP_FAILED",
            "APPCONTAINER_PROFILE_CREATE_FAILED", "APPCONTAINER_SID_FAILED",
            "APPCONTAINER_PROCESS_CREATE_FAILED",
        }

        if isinstance(error, OSError) and error.strerror in known_codes:
            result["failure_code"] = error.strerror
    finally:
        if wfp_capture is not None:
            wfp_capture.stop()
            result["wfp_capture"] = {
                "status": wfp_capture.record["status"],
                "stop_confirmed": wfp_capture.record["stop_confirmed"],
                "cleanup_confirmed": wfp_capture.record["cleanup_confirmed"],
                "drop_evidence_verified": False,
            }

            # 최초 시작 실패와 후속 정리 문제를 별도 기록한다. 원인을 덮어쓰지 않는다.
            if result.get("failure_stage") == "wfp_capture_start" and "failure_code" in wfp_capture.record:
                result["failure_code"] = wfp_capture.record["failure_code"]

            if wfp_capture._attempted and not wfp_capture.record["cleanup_confirmed"]:
                result["status"] = "review_required"
                result["cleanup_warning"] = "WFP_CAPTURE_STOP_NOT_CONFIRMED"
                result.setdefault("failure_code", "WFP_CAPTURE_STOP_NOT_CONFIRMED")

        observer.close()
        result["profile_cleanup"] = container.close()

        if not result["profile_cleanup"]:
            result["status"] = "review_required"
            result["profile_name"] = container.name

        (root / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    return root, result


def main():
    parser = argparse.ArgumentParser(description="합성 AppContainer 호환성 검사. 기존 환경·실데이터는 변경하지 않습니다.")
    parser.add_argument("--confirm-isolation-probe", action="store_true", help="전용 런타임 복사·일회성 AppContainer 생성·새 폴더 ACL 설정에 동의")
    parser.add_argument("--memory-mib", type=int, default=2048, help="Job 전체 커밋 메모리 상한(MiB)")
    parser.add_argument("--processes", type=int, default=1, help="루트 worker를 포함한 활성 프로세스 수 상한")
    parser.add_argument("--wall-seconds", type=int, default=60, help="worker 실행 경과 시간 상한(초)")
    args = parser.parse_args()

    if not args.confirm_isolation_probe:
        parser.error("실행하려면 --confirm-isolation-probe로 검사 범위를 확인하세요.")

    limits = WorkerLimits(args.wall_seconds, args.memory_mib, args.processes)
    root, result = run_probe(limits=limits)
    print(result["status"])

    if "failure_code" in result:
        print("검사 종료 코드:", result["failure_code"])

    progress = result.get("progress", {})

    if progress.get("last_checkpoint"):
        checkpoint = progress["last_checkpoint"]
        print("마지막 합성 검사 단계:", checkpoint["stage"], "/", checkpoint["state"])

    for attempt in progress.get("process_attempts", []):
        print("프로세스 생성 시도:", attempt["stage"], attempt["program"], attempt["package"], attempt["site"], attempt["line"])

    for name, status in result.get("component_status", {}).items():
        print(f"검사 영역 {name}: {status}")

    if "loopback_assessment" in result:
        print("로컬 통신 검사:", result["loopback_assessment"]["code"])

    print("로컬 검사 결과:", root / "result.json")
    print("실데이터 worker를 활성화하지 않았습니다.")
    raise SystemExit(0 if result["status"] == "compatibility_passed" else 2)


if __name__ == "__main__":
    main()
