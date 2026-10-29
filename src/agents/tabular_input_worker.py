"""고정 snapshot 검사·생성만 수행하는 입력 worker. 부모 승인 코드와 원본 폴더는 읽지 않는다."""

from pathlib import Path
import sys


POLICY_VERSION = "tabular_input_v2"
MAX_OUTPUT_BYTES = 64 * 1024 * 1024


def build_packet(action, snapshot, spec, limits, code_id, *, runtime_id, inspection_id=None, inspection_sha256=None,
                 max_output_bytes=MAX_OUTPUT_BYTES):
    """부모와 worker가 함께 검증할 고정 IPC 설정을 만든다.

    runtime_id는 승인 전에 고정한 공유 런타임 구성 ID다. config_id·승인 범위·요청 식별에 함께 결합된다.
    """
    import re
    from dataclasses import asdict
    from src.agents.input_snapshot import Snapshot
    from src.agents.worker_contract import WorkerLimits, _decode
    from src.preprocessing.tabular_loaders import CsvReadSpec

    if action not in ("inspect", "materialize") or type(snapshot) is not dict or set(snapshot) != set(Snapshot.__dataclass_fields__):
        raise ValueError("입력 도구의 고정 action·snapshot 계약이 필요합니다.")
    expected, resource = Snapshot(**snapshot), WorkerLimits(**limits)
    parsed = CsvReadSpec.from_dict(spec)
    hashes = (code_id, runtime_id) if action == "inspect" else (code_id, runtime_id, inspection_id, inspection_sha256)
    if any(type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None for value in hashes):
        raise ValueError("입력 도구의 코드·선행 결과 지문이 필요합니다.")
    if action == "inspect" and (inspection_id is not None or inspection_sha256 is not None):
        raise ValueError("검사 단계에는 생성 의존성을 넣을 수 없습니다.")
    if type(max_output_bytes) is not int or not 1 <= max_output_bytes <= 2**63 - 1:
        raise ValueError("입력 도구 출력 상한이 올바르지 않습니다.")
    packet = {"version": POLICY_VERSION, "loader_version": "1.0.0", "action": action,
              "snapshot": asdict(expected), "read_spec": parsed.to_dict(), "limits": asdict(resource), "code_id": code_id,
              "runtime_id": runtime_id,
              "inspection_id": inspection_id, "inspection_sha256": inspection_sha256, "max_output_bytes": max_output_bytes}
    import json
    _decode(json.dumps(packet, ensure_ascii=False, allow_nan=False))
    return packet


def execute_request(request, root, runtime_id):
    """경로·전체 설정 지문·선행 검사 파일·실행 런타임 ID를 대조한 뒤 고정 작업만 실행한다.

    runtime_id는 main이 실제 실행 파일 위치·완료 기록에서 확인한 값이다. 승인한 packet의 ID와 다르면 실행하지 않는다.
    """
    from dataclasses import asdict
    import hashlib
    import json
    from src.agents.input_snapshot import Snapshot
    from src.agents.snapshot_eda_worker import _verify_staged_snapshot
    from src.agents.worker_contract import WorkerResult, _decode
    from src.preprocessing.canonical_data import decode_json
    from src.preprocessing.tabular_bundle import worker_filesystem
    from src.preprocessing.tabular_loaders import CsvReadSpec, inspect_csv, materialize_csv, CsvLoadError

    def dump(value):
        return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")

    def read_json(path, maximum):
        with path.open("rb") as stream:
            raw = stream.read(maximum + 1)
        if len(raw) > maximum:
            raise ValueError("입력 메타데이터 크기 상한을 초과했습니다.")
        if maximum <= 1024 * 1024:
            return decode_json(raw), raw
        # 검사 본문은 IPC가 아닌 제한된 sidecar이며 중복 키·비표준 숫자는 같은 방식으로 거부한다.
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError("검사 JSON의 중복 키입니다.")
                result[key] = value
            return result
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("비표준 숫자")))
        if type(value) is not dict:
            raise ValueError("검사 객체가 필요합니다.")
        return value, raw

    root = Path(root).absolute()
    source, settings = root / "input/snapshot/input.bin", root / "input/settings.json"
    output, stage = root / "output/result", "REQUEST"
    try:
        with worker_filesystem(root):
            packet, _ = read_json(settings, 65536)
            action = packet.get("action")
            expected_packet = build_packet(action, packet["snapshot"], packet["read_spec"], packet["limits"], packet["code_id"],
                                           runtime_id=packet["runtime_id"], inspection_id=packet["inspection_id"], inspection_sha256=packet["inspection_sha256"],
                                           max_output_bytes=packet["max_output_bytes"])
            if dump(expected_packet) != dump(packet) or packet["runtime_id"] != runtime_id:
                raise ValueError("검토된 입력 설정 또는 실행 런타임과 다릅니다.")
            files = (str(source), str(source.parent / "snapshot.json"), str(settings))
            if action == "materialize":
                files += (str(root / "input/inspection.json"),)
            expected = Snapshot(**packet["snapshot"])
            if (request.task_id != root.name or request.stage != "ingest" or request.action_id != "run_ingest"
                    or request.policy_version != POLICY_VERSION or request.dependency_directories
                    or request.input_files != files or request.output_directory != str(output)
                    or request.scratch_directory != str(root / "scratch/task") or request.source_id != expected.source_id
                    or request.config_id != hashlib.sha256(dump(packet)).hexdigest() or asdict(request.limits) != packet["limits"]
                    or not _verify_staged_snapshot(root, expected)):
                raise ValueError("입력 worker 요청과 설정이 다릅니다.")
            stage = "COMPUTE"
            operation = output / "operation"
            operation.mkdir(exist_ok=False)
            spec = CsvReadSpec.from_dict(packet["read_spec"])
            reference = None
            if action == "inspect":
                report = inspect_csv(source, spec, limits=request.limits, max_input_bytes=expected.max_input_bytes)
                inspection_bytes = dump(report)
            else:
                report, inspection_bytes = read_json(root / "input/inspection.json", 16 * 1024 * 1024)
                if (hashlib.sha256(inspection_bytes).hexdigest() != packet["inspection_sha256"]
                        or report.get("inspection_id") != packet["inspection_id"]):
                    raise ValueError("선행 검사 파일이 승인 지문과 다릅니다.")
                reference = materialize_csv(source, spec, operation / "table", expected_snapshot=expected,
                                            expected_inspection=report, max_output_bytes=packet["max_output_bytes"])
            (operation / "inspection.json").write_bytes(inspection_bytes)
            binding = {"version": "tabular_input_result_v1", "request_id": request.request_id,
                       "packet": packet, "bundle": None if reference is None else asdict(reference),
                       "inspection_sha256": hashlib.sha256(inspection_bytes).hexdigest()}
            (operation / "binding.json").write_bytes(dump(binding))
            size = sum(path.stat().st_size for path in operation.rglob("*") if path.is_file())
            if size > packet["max_output_bytes"] or not _verify_staged_snapshot(root, expected):
                raise ValueError("출력 상한 또는 입력 복사본이 일치하지 않습니다.")
        return WorkerResult(request.task_id, request.request_id, "succeeded", "WORKER_EXITED_OK", ("operation",))
    except Exception as error:
        code = error.code if isinstance(error, CsvLoadError) else "INPUT_INVALID"
        try:
            (root / "scratch/task/diagnostic.json").write_bytes(dump({"stage": stage, "code": code}))
        except OSError:
            pass
        return WorkerResult(request.task_id, request.request_id, "blocked_input", "WORKER_INPUT_BLOCKED")


def shared_runtime(executable, runtime_id):
    """실행 중인 Python이 부모가 고른 공유 런타임 버전인지 위치·폴더 이름·완료 기록의 ID로 대조한다.

    보안 경계는 부모의 선택·해시 검증이다. 이 확인은 인자와 실제 실행 파일이 어긋난 실행을 멈추는 일관성 검사다.
    """
    import json
    import re
    runtime = Path(executable).absolute().parent
    if (type(runtime_id) is not str or re.fullmatch(r"[0-9a-f]{64}", runtime_id) is None
            or runtime.name != "runtime" or runtime.parent.name != runtime_id):
        return None
    try:
        with (runtime.parent / "manifest.json").open("rb") as stream:
            record = json.loads(stream.read(64 * 1024 * 1024 + 1))
    except (OSError, ValueError):
        return None
    return runtime if type(record) is dict and record.get("runtime_id") == runtime_id else None


def main():
    """공유 런타임 경로만 초기화하고 부모가 고정한 요청·런타임 ID를 대조한다."""
    import os
    if len(sys.argv) != 4:
        return 2
    root = Path(sys.argv[1]).absolute()
    runtime = shared_runtime(sys.executable, sys.argv[3])
    if runtime is None:
        return 2
    sys.path[:0] = [str(root / "tools"), str(runtime / "Lib/site-packages")]
    handles = [os.add_dll_directory(str(path)) for path in (runtime, runtime / "DLLs", runtime / "Library/bin") if path.is_dir()]
    from src.agents.worker_contract import WorkerRequest
    try:
        with (root / "input/request.json").open("rb") as stream:
            request = WorkerRequest.from_json(stream.read(65537).decode("utf-8"))
        if request.request_id != sys.argv[2]:
            return 2
        result = execute_request(request, root, sys.argv[3])
        (root / "output/result/worker_result.json").write_text(result.to_json(), encoding="utf-8")
        return 0 if result.status == "succeeded" else 2
    except Exception:
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
