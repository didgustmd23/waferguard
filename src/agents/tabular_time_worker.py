"""시간 결측 복원 미리보기·적용만 수행하는 격리 worker. 부모 승인 코드·원본 파일·작업 기록은 읽지 않는다.

입력은 작업 폴더에 복사한 생성 표 묶음과 고정 설정(settings.json), 적용이면 사용자가 확인한 미리보기(preview.json)뿐이다.
미리보기는 find_restorations 판정 결과(행 번호·복원 시각)만 남기고, 적용은 같은 판정이 미리보기와 완전히 같을 때만 새 묶음을 만든다.
자동 보간·외삽·행 삭제·재시도는 없다. 결과·진단 파일에는 원본 시간값을 넣지 않는다.
"""

from pathlib import Path
import sys


POLICY_VERSION = "tabular_time_restore_v1"
PREVIEW_VERSION = "tabular_time_preview_v1"
RESULT_VERSION = "tabular_time_result_v1"
PROVENANCE_VERSION = "tabular_time_restored_v1"
MAX_OUTPUT_BYTES = 64 * 1024 * 1024
MAX_PREVIEW_BYTES = 16 * 1024 * 1024
ACTIONS = ("preview", "apply")
TIME_FIELDS = ("column", "timestamp_format", "timezone", "frequency", "interval_minutes", "row_order_confirmed")
BUNDLE_FILES = ("SUCCESS.json", "data.jsonl", "manifest.json", "provenance.json", "schema.json")


def dump(value):
    """부모·worker가 같은 바이트로 대조하는 정규 JSON."""
    import json
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")


def build_packet(action, source_id, bundle, rules, limits, code_id, *, runtime_id, max_output_bytes=MAX_OUTPUT_BYTES,
                 preview_sha256=None):
    """부모와 worker가 함께 검증할 고정 IPC 설정을 만든다. 규칙 서명은 계획의 time_signature와 같은 직렬화다."""
    from dataclasses import asdict
    import hashlib
    import json
    import re
    from src.agents.worker_contract import WorkerLimits, _decode
    from src.preprocessing.tabular_bundle import BundleRef
    from src.preprocessing.time_restore import TimeRestoreRules

    if (action not in ACTIONS or type(source_id) is not str
            or re.fullmatch(r"tabular_source:[0-9a-f]{64}", source_id) is None):
        raise ValueError("시간 복원 도구의 고정 action·입력 출처가 필요합니다.")
    if type(rules) is not dict or set(rules) != set(TIME_FIELDS):
        raise ValueError("시간 복원 규칙 필드가 계약과 다릅니다.")
    TimeRestoreRules(**rules)
    reference, resource = BundleRef.from_dict(bundle), WorkerLimits(**limits)
    hashes = (code_id, runtime_id) + ((preview_sha256,) if action == "apply" else ())
    if any(type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None for value in hashes):
        raise ValueError("시간 복원 도구의 코드·런타임·미리보기 지문이 필요합니다.")
    if action == "preview" and preview_sha256 is not None:
        raise ValueError("미리보기 단계에는 미리보기 지문을 넣을 수 없습니다.")
    if type(max_output_bytes) is not int or not 1 <= max_output_bytes <= 2**63 - 1:
        raise ValueError("시간 복원 도구 출력 상한이 올바르지 않습니다.")
    ordered = {name: rules[name] for name in TIME_FIELDS}
    packet = {"version": POLICY_VERSION, "action": action, "source_id": source_id, "bundle": asdict(reference),
              "rules": ordered, "rules_signature": hashlib.sha256(json.dumps(ordered, ensure_ascii=False, sort_keys=True,
                                                                             allow_nan=False).encode("utf-8")).hexdigest(),
              "limits": asdict(resource), "code_id": code_id, "runtime_id": runtime_id, "max_output_bytes": max_output_bytes,
              "preview_sha256": preview_sha256}
    _decode(json.dumps(packet, ensure_ascii=False, allow_nan=False))
    return packet


def rebuild_packet(packet):
    """저장·전달된 packet을 같은 검증으로 다시 만든다. 필드가 다르면 ValueError."""
    if type(packet) is not dict:
        raise ValueError("시간 복원 설정 객체가 필요합니다.")
    return build_packet(packet.get("action"), packet.get("source_id"), packet.get("bundle"), packet.get("rules"),
                        packet.get("limits"), packet.get("code_id"), runtime_id=packet.get("runtime_id"),
                        max_output_bytes=packet.get("max_output_bytes"), preview_sha256=packet.get("preview_sha256"))


# ==========================================
# 판정·적용 공통 규칙 (부모 결과 검증도 같은 함수를 쓴다)
# - 문자열 컬럼은 형식이 있어야 하고, 시각 자료형 컬럼은 형식 없이 시각 값으로 읽는다
# - 시각 자료형에 시간대가 있으면 규칙 시간대를 함께 지정할 수 없고, 없으면 규칙 시간대가 필요하다
# - 미리보기 문서에는 행 번호(0부터)·복원 시각(ISO)·건수·지문만 둔다(원본 시간값 없음)
# ==========================================

def check_column(dtype, rules):
    """시간 컬럼의 묶음 자료형이 규칙과 맞는지 확인한다. 맞지 않으면 TimeRestoreError."""
    from src.preprocessing.time_restore import TimeRestoreError

    kind = dtype.get("kind") if type(dtype) is dict else None
    if kind == "plain" and dtype.get("dtype") in ("string", "object"):
        if rules.timestamp_format is None:
            raise TimeRestoreError("TIME_PARSE_FAILED")
        return
    if kind != "timestamp" or rules.timestamp_format is not None:
        raise TimeRestoreError("TIME_PARSE_FAILED")
    if dtype["timezone"] is not None and rules.timezone is not None:
        raise TimeRestoreError("TIME_TIMEZONE_CONFLICT")
    if dtype["timezone"] is None and rules.timezone is None:
        raise TimeRestoreError("TIME_TIMEZONE_REQUIRED")


def restored_cell(time, rules, dtype):
    """복원 시각을 그 컬럼 자료형의 셀 값으로 바꾼다. 문자열 컬럼은 규칙 형식 표기, 시각 컬럼은 같은 단위·시간대의 시각이다."""
    from src.preprocessing.time_restore import TimeRestoreError, format_time

    check_column(dtype, rules)
    value = format_time(time, rules)
    if dtype["kind"] == "plain":
        return value
    import pandas as pd
    stamp = pd.Timestamp(value)
    stamp = stamp.tz_convert(dtype["timezone"]) if dtype["timezone"] is not None else stamp.tz_convert(rules.timezone).tz_localize(None)
    try:
        return stamp.as_unit(dtype["unit"], round_ok=False)
    except ValueError as error:
        raise TimeRestoreError("TIME_AMBIGUOUS") from error


def preview_document(packet, row_count, restoration, code):
    """미리보기 결과 문서. 멈춘 판정은 고정 코드만 담고 복원 대상은 비운다."""
    from src.preprocessing.time_restore import restoration_digest

    return {"version": PREVIEW_VERSION, "source_id": packet["source_id"], "rules_signature": packet["rules_signature"],
            "row_count": row_count, "status": "ready" if code is None else "stopped", "code": code,
            "restored_count": len(restoration.rows), "rows": list(restoration.rows),
            "times": [item.isoformat() for item in restoration.times], "restoration_digest": restoration_digest(restoration)}


def provenance_document(packet, restoration, preview_sha256):
    """파생 표의 출처 정보. 부모 묶음 지문·규칙·확인한 미리보기 지문에 결합한다."""
    from src.preprocessing.time_restore import restoration_digest

    bundle = packet["bundle"]
    return {"version": PROVENANCE_VERSION, "policy_version": POLICY_VERSION, "parent_source_id": packet["source_id"],
            "parent_bundle": {key: bundle[key] for key in ("manifest_sha256", "completion_sha256", "size_bytes", "rows", "columns")},
            "rules": packet["rules"], "rules_signature": packet["rules_signature"], "preview_sha256": preview_sha256,
            "restoration_digest": restoration_digest(restoration), "restored_count": len(restoration.rows),
            "row_identity": "parent_positions"}


class PreviewChanged(ValueError):
    """적용 단계의 재판정이 사용자가 확인한 미리보기와 다르다."""


def execute_request(request, root, runtime_id):
    """경로·전체 설정 지문·입력 묶음·실행 런타임 ID를 대조한 뒤 미리보기 또는 적용만 실행한다."""
    from dataclasses import asdict
    import hashlib
    from src.agents.worker_contract import WorkerResult
    from src.preprocessing.canonical_data import decode_json
    from src.preprocessing.tabular_bundle import (BundleRef, _write, inspect_bundle, iter_bundle_rows, read_bundle_schema,
                                                  worker_filesystem)
    from src.preprocessing.time_restore import (Restoration, TimeRestoreError, TimeRestoreRules, apply_restorations,
                                                find_restorations)

    def read(path, maximum):
        with path.open("rb") as stream:
            raw = stream.read(maximum + 1)
        if len(raw) > maximum:
            raise ValueError("입력 파일 크기 상한을 초과했습니다.")
        return raw

    root = Path(root).absolute()
    settings, bundle, preview_input = root / "input/settings.json", root / "input/bundle", root / "input/preview.json"
    output, stage = root / "output/result", "REQUEST"
    try:
        with worker_filesystem(root):
            packet = decode_json(read(settings, 65536))
            if dump(rebuild_packet(packet)) != dump(packet) or packet["runtime_id"] != runtime_id:
                raise ValueError("검토된 시간 복원 설정 또는 실행 런타임과 다릅니다.")
            action = packet["action"]
            files = tuple(str(bundle / name) for name in BUNDLE_FILES) + (str(settings),)
            if action == "apply":
                files += (str(preview_input),)
            if (request.task_id != root.name or request.stage != "ingest" or request.action_id != "run_ingest"
                    or request.policy_version != POLICY_VERSION or request.dependency_directories
                    or request.input_files != files or request.output_directory != str(output)
                    or request.scratch_directory != str(root / "scratch/task") or request.source_id != packet["source_id"]
                    or request.config_id != hashlib.sha256(dump(packet)).hexdigest() or asdict(request.limits) != packet["limits"]):
                raise ValueError("시간 복원 worker 요청과 설정이 다릅니다.")
            reference = BundleRef.from_dict(packet["bundle"])
            inspect_bundle(bundle, expected=reference, max_bytes=reference.size_bytes)
            stage = "COMPUTE"
            operation = output / "operation"
            operation.mkdir(exist_ok=False)
            schema = read_bundle_schema(bundle, expected=reference)
            rules = TimeRestoreRules(**packet["rules"])
            index = next((number for number, column in enumerate(schema["columns"]) if column["name"] == rules.column), None)
            dtype = None if index is None else schema["columns"][index]["dtype"]
            try:
                if index is None:
                    raise TimeRestoreError("TIME_RULES_INVALID")
                check_column(dtype, rules)
                values = [row[index] for _, _, row in iter_bundle_rows(bundle, expected=reference)]
                restoration, code = find_restorations(values, rules), None
            except TimeRestoreError as error:
                # 멈춘 판정도 검증 가능한 결과로 남긴다. 고정 코드만 담고 원본 값은 담지 않는다.
                restoration, code = Restoration((), ()), error.code
            preview_bytes = dump(preview_document(packet, reference.rows, restoration, code))
            if len(preview_bytes) > MAX_PREVIEW_BYTES:
                raise ValueError("미리보기 크기 상한을 초과했습니다.")
            preview_sha256 = hashlib.sha256(preview_bytes).hexdigest()
            result = None
            if action == "apply":
                confirmed = read(preview_input, MAX_PREVIEW_BYTES)
                # 사용자가 확인한 미리보기와 바이트까지 같은 재판정일 때만 적용한다(규칙·입력·코드가 같아야 같은 결과).
                if (hashlib.sha256(confirmed).hexdigest() != packet["preview_sha256"] or confirmed != preview_bytes
                        or code is not None or not restoration.rows):
                    raise PreviewChanged("확인한 미리보기와 재판정이 다릅니다.")
                apply_restorations(values, restoration, rules)
                changes = {row: restored_cell(time, rules, dtype) for row, time in zip(restoration.rows, restoration.times)}

                def rows():
                    for ordinal, (position, label, row) in enumerate(iter_bundle_rows(bundle, expected=reference)):
                        if ordinal in changes:
                            row = list(row)
                            row[index] = changes[ordinal]
                        yield position, label, row

                result = _write(operation / "table", schema, rows(), provenance_document(packet, restoration, preview_sha256),
                                packet["max_output_bytes"])
            (operation / "preview.json").write_bytes(preview_bytes)
            binding = {"version": RESULT_VERSION, "request_id": request.request_id, "packet": packet,
                       "preview_sha256": preview_sha256, "bundle": None if result is None else asdict(result)}
            (operation / "binding.json").write_bytes(dump(binding))
            size = sum(path.stat().st_size for path in operation.rglob("*") if path.is_file())
            if size > packet["max_output_bytes"]:
                raise ValueError("출력 상한을 초과했습니다.")
        return WorkerResult(request.task_id, request.request_id, "succeeded", "WORKER_EXITED_OK", ("operation",))
    except Exception as error:
        code = "TIME_PREVIEW_CHANGED" if isinstance(error, PreviewChanged) else "INPUT_INVALID"
        try:
            (root / "scratch/task/diagnostic.json").write_bytes(dump({"stage": stage, "code": code}))
        except OSError:
            pass
        return WorkerResult(request.task_id, request.request_id, "blocked_input", "WORKER_INPUT_BLOCKED")


def main():
    """공유 런타임 경로만 초기화하고 부모가 고정한 요청·런타임 ID를 대조한다(입력 worker와 같은 확인)."""
    import os
    if len(sys.argv) != 4:
        return 2
    root = Path(sys.argv[1]).absolute()
    sys.path[:0] = [str(root / "tools")]
    from src.agents.tabular_input_worker import shared_runtime
    runtime = shared_runtime(sys.executable, sys.argv[3])
    if runtime is None:
        return 2
    sys.path.insert(1, str(runtime / "Lib/site-packages"))
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
