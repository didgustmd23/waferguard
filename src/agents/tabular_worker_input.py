"""승인 DB 없이 v2 고정 packet과 격리 폴더 안의 typed 표만 읽는다."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from src.preprocessing.tabular_bundle import (BundleRef, FILES, inspect_bundle, load_bundle,
                                               read_bundle_provenance, worker_filesystem)
from src.preprocessing.tabular_loaders import CsvReadSpec
from .input_snapshot import Snapshot


SPLIT_POLICY = "tabular_split_v2"
EXPORT_POLICY = "partition_export_v2"
DIAGNOSTIC_POLICY = "tabular_diagnostic_worker_v2"
MAX_SPLIT_ROWS = 1_000_000


def digest(value):
    """기존 계약과 같은 정규 JSON 내용 지문을 계산한다."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def table_files(root, settings, *, extra=()):
    """worker가 읽을 파일을 고정 목록으로 만든다. 경로 탐색 결과로 권한을 넓히지 않는다."""
    return tuple(str(Path(root) / "input" / name) for name in (*("table/" + name for name in sorted(FILES)), settings, *extra))


def validate_source(packet):
    """생성 증거의 경로 없는 내용 참조와 typed 묶음 기대값만 복원한다."""
    keys = {"schema_version", "snapshot", "bundle", "read_spec", "inspection_id", "inspection_sha256", "max_output_bytes",
            "loader_id", "loader_version", "policy_version", "materialization_config_id"}
    if (type(packet) is not dict or set(packet) != keys or packet["schema_version"] != "tabular_source_v2"
            or packet["loader_id"] != "delimited_v1" or packet["loader_version"] != "1.0.0"
            or packet["policy_version"] != "tabular_input_v2"):
        raise ValueError("검증된 v2 표형 출처 packet이 필요합니다.")
    CsvReadSpec.from_dict(packet["read_spec"])
    Snapshot(**packet["snapshot"])
    for key in ("inspection_id", "inspection_sha256", "materialization_config_id"):
        value = packet[key]
        if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("표형 출처 지문 형식이 다릅니다.")
    reference = BundleRef.from_dict(packet["bundle"])
    if reference.rows > MAX_SPLIT_ROWS:
        raise ValueError("현재 분할 실행은 1백만 행 이하만 지원합니다.")
    if type(packet["max_output_bytes"]) is not int or reference.size_bytes > packet["max_output_bytes"]:
        raise ValueError("표형 출처 크기 상한이 다릅니다.")
    return reference


def source_id(packet):
    """경로 없는 생성 참조 전체를 입력 식별값으로 고정한다."""
    validate_source(packet)
    return "tabular_source:" + digest(packet)


def check_staged(root, packet):
    """격리 경계 아래의 파일·자료형·provenance와 승인된 참조를 대조한다."""
    reference = validate_source(packet)
    with worker_filesystem(root):
        inspect_bundle(Path(root) / "input/table", expected=reference, max_bytes=reference.size_bytes)
        provenance = read_bundle_provenance(Path(root) / "input/table", expected=reference)
    expected = {"source_snapshot_sha256": packet["snapshot"]["sha256"],
                "source_snapshot_size_bytes": packet["snapshot"]["size_bytes"],
                "read_spec": packet["read_spec"], "inspection_id": packet["inspection_id"],
                "inspection_sha256": packet["inspection_sha256"], "loader_id": packet["loader_id"],
                "loader_version": packet["loader_version"], "policy_version": packet["policy_version"]}
    if any(digest(provenance.get(key)) != digest(value) for key, value in expected.items()):
        raise ValueError("typed 입력의 생성 계보가 다릅니다.")
    if reference.rows > MAX_SPLIT_ROWS:
        raise ValueError("현재 분할 실행은 1백만 행 이하만 지원합니다.")
    return reference


def load_staged(root, packet):
    """worker 전용 typed reader. 원래 인덱스와 자료형을 바꾸지 않는다."""
    reference = check_staged(root, packet)
    with worker_filesystem(root):
        return load_bundle(Path(root) / "input/table", expected=reference)


def execute_diagnostic(request, root):
    """기존 모듈 worker의 v2 분기. 단일 계획 전체와 서명을 여기서 다시 검사한다."""
    from src.preprocessing.input_profile import input_profile_pipeline, bind_input_profile_plan, profile_input, write_input_profile
    from .snapshot_eda_worker import _diagnostic
    from .worker_contract import WorkerResult, _decode
    root = Path(root).absolute()
    try:
        if (request.policy_version != DIAGNOSTIC_POLICY or request.stage != "run_eda" or request.action_id != "run_eda"
                or request.task_id != root.name or request.input_files != table_files(root, "eda.json")
                or request.dependency_directories or request.output_directory != str(root / "output/result")
                or request.scratch_directory != str(root / "scratch/task")):
            raise ValueError("진단 worker 고정 경로·작업이 다릅니다.")
        with (root / "input/eda.json").open("rb") as stream:
            packet = _decode(stream.read(65537).decode("utf-8"))
        if (set(packet) != {"version", "source", "profile", "limits", "code_id", "pipeline", "pipeline_signature"}
                or packet["version"] != DIAGNOSTIC_POLICY or packet["profile"] is not None
                or digest(packet) != request.config_id or source_id(packet["source"]) != request.source_id
                or packet["limits"] != asdict(request.limits) or packet["pipeline"] != input_profile_pipeline()):
            raise ValueError("v2 고정 진단 설정이 승인 범위와 다릅니다.")
        _, plan = bind_input_profile_plan(packet["pipeline"])
        if plan.signature != packet["pipeline_signature"]:
            raise ValueError("고정 진단 서명이 다릅니다.")
        frame = load_staged(root, packet["source"])
        destination = root / "output/result/pipeline"
        destination.mkdir()
        write_input_profile(profile_input(frame), destination / plan.steps[0].step_id)
        check_staged(root, packet["source"])
        (destination / "pipeline_result.json").write_text(json.dumps({"version": "module_pipeline_result_v1",
            "pipeline_signature": plan.signature, "row_count": len(frame), "status": "passed", "completed_steps": [plan.steps[0].step_id]},
            ensure_ascii=False, allow_nan=False), encoding="utf-8")
        return WorkerResult(request.task_id, request.request_id, "succeeded", "WORKER_EXITED_OK", ("pipeline",))
    except (ValueError, TypeError, KeyError, OSError, UnicodeError):
        _diagnostic(root, "INPUT_VALIDATION", "INPUT_INVALID")
        return WorkerResult(request.task_id, request.request_id, "blocked_input", "WORKER_INPUT_BLOCKED")
    except Exception:
        _diagnostic(root, "MODULE_EXECUTION", "RUNTIME_FAILED")
        return WorkerResult(request.task_id, request.request_id, "failed_runtime", "WORKER_FAILED")


def execute_export(request, root):
    """확정 위치만 typed 구간으로 저장한다. Test 묶음·값 파일은 만들지 않는다."""
    from src.preprocessing.contracts import TabularInput
    from src.preprocessing.profiles import profile_from_dict
    from src.preprocessing.split import _check_boundaries
    from src.preprocessing.tabular_bundle import write_subset
    from .strict_json import loads
    from .snapshot_eda_worker import _diagnostic
    from .worker_contract import WorkerResult, _decode
    root = Path(root).absolute()
    try:
        if (request.policy_version != EXPORT_POLICY or request.stage != "ingest" or request.action_id != "run_ingest"
                or request.task_id != root.name or request.input_files != table_files(root, "export.json", extra=("split.json",))
                or request.dependency_directories or request.output_directory != str(root / "output/result")
                or request.scratch_directory != str(root / "scratch/task")):
            raise ValueError("내보내기 고정 경로·작업이 다릅니다.")
        with (root / "input/export.json").open("rb") as stream:
            packet = _decode(stream.read(65537).decode("utf-8"))
        keys = {"version", "source", "profile", "limits", "split_evidence_id", "split_file_sha256", "max_partition_bytes", "code_id"}
        if set(packet) != keys or packet["version"] != EXPORT_POLICY or digest(packet) != request.config_id:
            raise ValueError("내보내기 설정이 다릅니다.")
        profile = profile_from_dict(packet["profile"])
        maximum = packet["max_partition_bytes"]
        if (profile.adapter_id != "tabular_canonical_v2" or profile.split_rules is None
                or request.source_id != source_id(packet["source"]) or packet["limits"] != asdict(request.limits)
                or type(maximum) is not int or not 1 <= maximum <= 2**63 - 1):
            raise ValueError("내보내기 원본·프로파일·자원이 다릅니다.")
        with (root / "input/split.json").open("rb") as stream:
            raw = stream.read(8 * 1024 * 1024 + 1)
        if len(raw) > 8 * 1024 * 1024 or hashlib.sha256(raw).hexdigest() != packet["split_file_sha256"]:
            raise ValueError("확정 분할 파일이 다릅니다.")
        proposal = loads(raw.decode("utf-8"), 8 * 1024 * 1024)
        if (set(proposal) != {"schema_version", "source_id", "config_id", "rules", "partitions", "summary", "status", "warnings", "proposal_id"}
                or proposal["schema_version"] != "split_proposal_v1" or proposal["status"] != "passed"
                or proposal["source_id"] != request.source_id or proposal["config_id"] != profile.signature
                or digest(proposal["rules"]) != digest(asdict(profile.split_rules))
                or digest({k: v for k, v in proposal.items() if k != "proposal_id"}) != proposal["proposal_id"]):
            raise ValueError("확정 분할의 출처·규칙·지문이 다릅니다.")
        partitions = proposal["partitions"]
        if type(partitions) is not dict or set(partitions) != {"train", "validation", "test"}:
            raise ValueError("확정 구간 목록이 다릅니다.")
        for positions in partitions.values():
            if (type(positions) is not list or not positions or len(positions) > MAX_SPLIT_ROWS
                    or any(type(v) is not int or v < 0 for v in positions) or positions != sorted(set(positions))):
                raise ValueError("확정 행 위치 형식이 다릅니다.")
        frame = load_staged(root, packet["source"])
        data = TabularInput(frame, profile.roles)
        _check_boundaries(data, profile.split_rules, partitions)
        destination = root / "output/result/exports"
        destination.mkdir()
        records = {}
        reference = validate_source(packet["source"])
        with worker_filesystem(root):
            for name in ("train", "validation"):
                positions = partitions[name]
                provenance = {"parent_source_id": request.source_id, "split_signature": packet["split_evidence_id"],
                              "row_positions_sha256": digest(positions)}
                result = write_subset(root / "input/table", positions, destination / name, expected=reference,
                                      provenance=provenance, max_output_bytes=maximum)
                records[name] = {"bundle": asdict(result), "row_positions_sha256": digest(positions)}
        (destination / "partition_exports.json").write_text(json.dumps({"schema_version": "partition_exports_v2",
            "source_id": request.source_id, "split_evidence_id": packet["split_evidence_id"],
            "roles": asdict(profile.roles), "partitions": records}, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        check_staged(root, packet["source"])
        return WorkerResult(request.task_id, request.request_id, "succeeded", "WORKER_EXITED_OK", ("exports",))
    except (ValueError, TypeError, KeyError, OSError, UnicodeError):
        _diagnostic(root, "INPUT_VALIDATION", "INPUT_INVALID")
        return WorkerResult(request.task_id, request.request_id, "blocked_input", "WORKER_INPUT_BLOCKED")
    except Exception:
        _diagnostic(root, "INPUT_VALIDATION", "RUNTIME_FAILED")
        return WorkerResult(request.task_id, request.request_id, "failed_runtime", "WORKER_FAILED")
