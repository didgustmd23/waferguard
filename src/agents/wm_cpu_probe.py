"""Python 3.12용 고정 합성 WM/PyTorch CPU 검사. 실데이터 worker를 활성화하지 않는다."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import shutil
import sys
import uuid

from .appcontainer_native import AppContainer
from .appcontainer_probe import _acl, probe_environment
from .artifacts import _fingerprint, _root
from .isolated_eda import _read_file
from .wm_cpu_probe_payload import CHECKS, SCHEMA
from .wm_dataset_runtime import copy_runtime_file, runtime_manifest, pinned_torch_version
from .wm_dataset_transport import WORKER_FILES as WM_FILES, preparation_failure
from .worker_contract import WorkerLimits, _decode
from .worker_paths import _path


ENTRY = "src/agents/wm_cpu_probe_worker.py"
PAYLOAD = "src/agents/wm_cpu_probe_payload.py"
COPY_FILES = tuple(dict.fromkeys((*WM_FILES, PAYLOAD)))


def assess_result(child, launch, limits, torch_version, probe_id):
    """부모 판정 항목을 분리해 성공한 검사와 남은 불일치를 명확히 기록한다."""
    resources = launch.get("resources", {})
    return {
        "worker_exit": launch.get("status") == "exited" and launch.get("exit_code") == 0,
        "resources": (resources.get("status") == "succeeded" and resources.get("cleanup_confirmed") is True and
                      resources.get("active_processes") == 0 and
                      resources.get("configured_job_memory_bytes") == limits.memory_mib * 1024 ** 2 and
                      resources.get("configured_process_limit") == limits.processes),
        "child_contract": validate_child(child, probe_id),
        "pinned_torch_build": child.get("torch_version") == torch_version,
    }


def validate_child(value, probe_id=None):
    """성공 문자열뿐 아니라 고정 기대값·구간·shape·dtype를 재검사한다."""
    if (type(value) is not dict or (probe_id is not None and value.get("probe_id") != probe_id) or
            value.get("schema_version") != SCHEMA or
            value.get("status") != "cpu_checks_completed" or value.get("device") != "cpu" or
            value.get("model_trained") is not False or value.get("real_data_read") is not False or
            value.get("production_worker_enabled") is not False or value.get("gpu_verified") is not False or
            value.get("samples") != 54 or value.get("class_count") != 9 or
            value.get("tensor_values") != [0, 2, 4, 6, 8, 10]):
        return False

    checks, counts, batches = (value.get(key) for key in ("checks", "split_counts", "batches"))
    if (type(checks) is not dict or set(checks) != CHECKS or any(v is not True for v in checks.values()) or
            type(counts) is not dict or set(counts) != {"train", "validation", "test"} or
            any(type(n) is not int or n < 9 for n in counts.values()) or sum(counts.values()) != 54 or
            type(batches) is not dict or set(batches) != {"train", "validation"}):
        return False

    protocols = []
    for name, batch in batches.items():
        if (type(batch) is not dict or batch.get("status") != "dataset_batch_passed" or
                batch.get("split") != name or batch.get("image_shape") != [4, 2, 32, 32] or
                batch.get("image_dtype") != "torch.float32" or batch.get("target_dtype") != "torch.int64" or
                batch.get("test_maps_read") is not False or batch.get("model_trained") is not False):
            return False
        protocols.append(batch.get("split_protocol_id"))

    return (isinstance(protocols[0], str) and re.fullmatch(r"wm_split_[a-f0-9]{64}", protocols[0]) is not None and
            protocols[0] == protocols[1])


def run_probe(limits, max_runtime_bytes):
    """기존 AppContainer+Job에 고정 검사만 전달한다. 사용자 입력 데이터는 받지 않는다."""
    if os.name != "nt" or sys.version_info[:2] != (3, 12):
        raise ValueError("현재 합성 CPU 검사는 Windows/Python 3.12 전용입니다.")
    if not isinstance(limits, WorkerLimits) or type(max_runtime_bytes) is not int or not 1 <= max_runtime_bytes <= 32 * 1024 ** 3:
        raise ValueError("자원 제한과 32 GiB 이하 런타임 복사 예산이 필요합니다.")

    project = _root(Path(__file__).resolve().parents[2])
    parent = project
    for part in ("logs", "agents", "wm_cpu_probes"):
        parent /= part
        parent.mkdir(exist_ok=True)
        _root(parent)

    root = parent / uuid.uuid4().hex
    root.mkdir()
    for name in ("runtime", "tools", "output", "scratch"):
        (root / name).mkdir()
    (root / "scratch/task").mkdir()

    result = {"schema_version": "wm_cpu_probe_parent_v1", "status": "review_required",
              "python": list(sys.version_info[:3]), "limits": asdict(limits),
              "max_runtime_bytes": max_runtime_bytes, "real_data_read": False,
              "production_worker_enabled": False, "gpu_verified": False,
              "network_evidence_retested": False, "worker_started": False,
              "worker_launch_attempted": False, "profile_cleanup": False}
    container = AppContainer("waferguard.wmcpu." + root.name)
    stage = "PROFILE_CREATE"
    try:
        sid = container.create()
        stage = "OUTPUT_ACL"
        for name in ("runtime", "tools"):
            _acl(root / name, sid, "RX")
        for name in ("output", "scratch/task"):
            _acl(root / name, sid, "M", low=True)

        stage = "RUNTIME_INVENTORY"
        files, torch_version = runtime_manifest(Path(sys.prefix), max_runtime_bytes)
        result.update(runtime_bytes=sum(item["size"] for item in files.values()), torch_version=torch_version)
        (root / "runtime_files.json").write_text(json.dumps(files, sort_keys=True), encoding="utf-8")
        result["runtime_manifest"] = _fingerprint(root / "runtime_files.json")
        before = {name: _fingerprint(project / name) for name in (*COPY_FILES, ENTRY)}
        result["code_manifest"] = before
        stage = "RUNTIME_COPY"
        for name, record in files.items():
            copy_runtime_file(Path(sys.prefix) / name, root / "runtime" / name, record)

        stage = "TOOLS_COPY"
        for name, record in before.items():
            target = root / "tools" / ("worker.py" if name == ENTRY else name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project / name, target)
            if _fingerprint(target) != record:
                raise ValueError("검사 코드 복사본이 일치하지 않습니다.")
        if any(_fingerprint(project / name) != record for name, record in before.items()):
            raise ValueError("검사 준비 중 코드가 변경되었습니다.")

        stage = "WORKER_LAUNCH"
        environment = probe_environment(root / "runtime", root / "scratch/task", os.environ)
        environment.update({"CUDA_VISIBLE_DEVICES": "-1", "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"})
        result["worker_launch_attempted"] = True
        result["worker_started"] = None
        result["launch"] = container.launch(root / "runtime/python.exe",
            ["-I", "-S", "-B", str(root / "tools/worker.py"), str(root)],
            environment, root / "scratch/task", limits=limits)
        result["worker_started"] = True

        stage = "RESULT_CHECK"
        child = _decode(_read_file(_path(root / "output/child_result.json", directory=False), 65536).decode("utf-8"))
        result["child"] = child
        result["result_checks"] = assess_result(child, result["launch"], limits, torch_version, root.name)
        if all(result["result_checks"].values()):
            result["status"] = "cpu_compatibility_passed"
        else:
            result["failure_code"] = "SYNTHETIC_CPU_CHECKS_NOT_PASSED"
    except Exception as error:
        result["preparation_failure"] = preparation_failure(stage, error)
    finally:
        try:
            result["profile_cleanup"] = container.close()
        except Exception as error:
            result["cleanup_failure"] = preparation_failure("PROFILE_CLEANUP", error)
        finally:
            if not result["profile_cleanup"]:
                result["status"] = "review_required"
            (root / "result.json").write_text(json.dumps(result, allow_nan=False, indent=2), encoding="utf-8")

    return root, result


def main():
    parser = argparse.ArgumentParser(description="고정 합성 WM/PyTorch CPU 검사. 실데이터·GPU·학습·지원 정책 변경 없음")
    parser.add_argument("--confirm-synthetic-wm-cpu-probe", action="store_true")
    parser.add_argument("--memory-mib", type=int, default=8192)
    parser.add_argument("--processes", type=int, default=2)
    parser.add_argument("--wall-seconds", type=int, default=300)
    parser.add_argument("--max-runtime-bytes", type=int, default=16 * 1024 ** 3)
    args = parser.parse_args()
    if not args.confirm_synthetic_wm_cpu_probe:
        parser.error("고정 합성 입력·런타임 복사·일회성 AppContainer 생성에 대한 확인이 필요합니다.")

    root, result = run_probe(WorkerLimits(args.wall_seconds, args.memory_mib, args.processes), args.max_runtime_bytes)
    print(result["status"])
    if result.get("failure_code"):
        print("부모 판정 코드:", result["failure_code"])
        print("부모 판정 항목:", result.get("result_checks", {}))
    if result.get("preparation_failure"):
        print("준비 실패:", result["preparation_failure"])
    if result.get("child", {}).get("failure_code"):
        print("검사 실패:", result["child"]["failure_stage"], result["child"]["failure_code"])
    print("로컬 검사 결과:", root / "result.json")
    print("CPU 합성 검사입니다. 실데이터·GPU·모델 학습·WM 지원 정책을 활성화하지 않았습니다.")
    return 0 if result["status"] == "cpu_compatibility_passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
