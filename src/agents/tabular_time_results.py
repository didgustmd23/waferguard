"""시간 복원 worker 결과(미리보기·파생 표)를 부모가 다시 검증한다. 파생 표에서 원래 표와 다른 셀은 미리보기 행의 시간 셀뿐이어야 한다."""

from dataclasses import asdict
from datetime import datetime
import hashlib
from itertools import zip_longest
import json

from src.preprocessing.tabular_bundle import (BundleRef, _encode, _safe, inspect_bundle, iter_bundle_rows,
                                              read_bundle_provenance, read_bundle_schema)
from src.preprocessing.time_restore import (CODES, Restoration, TimeRestoreError, TimeRestoreRules, apply_restorations,
                                            find_restorations, restoration_digest)
from . import strict_json
from .tabular_time_worker import (MAX_PREVIEW_BYTES, POLICY_VERSION, PREVIEW_VERSION, RESULT_VERSION, check_column,
                                  preview_document, provenance_document, rebuild_packet, restored_cell)


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _require(value):
    if not value:
        raise ValueError("시간 복원 worker 결과의 형식·지문 또는 계보가 일치하지 않습니다.")


def _read(root, name, maximum):
    path = _safe(root / name, directory=False)
    with path.open("rb") as stream:
        payload = stream.read(maximum + 1)
    _require(len(payload) <= maximum)
    parsed = strict_json.loads(payload.decode("utf-8"), maximum)
    _require(payload == _json(parsed).encode("utf-8"))
    return parsed, hashlib.sha256(payload).hexdigest()


def preview_restoration(preview, packet):
    """미리보기 문서의 키·건수·행 번호·ISO 시각·지문을 대조하고 복원 판정(Restoration)으로 되돌린다."""
    keys = {"version", "source_id", "rules_signature", "row_count", "status", "code", "restored_count", "rows", "times",
            "restoration_digest"}
    _require(type(preview) is dict and set(preview) == keys and preview["version"] == PREVIEW_VERSION)
    _require(preview["source_id"] == packet["source_id"] and preview["rules_signature"] == packet["rules_signature"])
    count = preview["row_count"]
    _require(type(count) is int and count == packet["bundle"]["rows"])
    rows, texts = preview["rows"], preview["times"]
    _require(type(rows) is list and type(texts) is list and len(rows) == len(texts)
             and type(preview["restored_count"]) is int and preview["restored_count"] == len(rows))
    if preview["status"] == "stopped":
        _require(preview["code"] in CODES and not rows)
    else:
        _require(preview["status"] == "ready" and preview["code"] is None)
    previous, times = 0, []
    for row, text in zip(rows, texts):
        # 앞뒤 끝은 복원 대상이 될 수 없다(외삽 없음). 행 번호는 엄격히 증가한다.
        _require(type(row) is int and previous < row < count - 1 and type(text) is str and len(text) <= 64)
        value = datetime.fromisoformat(text)
        _require(value.tzinfo is not None and value.isoformat() == text)
        previous = row
        times.append(value)
    restoration = Restoration(tuple(rows), tuple(times))
    _require(preview["restoration_digest"] == restoration_digest(restoration))
    return restoration


def _rejudged(preview, packet, parent_directory):
    """원래 묶음(packet의 검증된 지문)의 시간 컬럼을 같은 규칙으로 다시 판정해 미리보기와 정확히 같은지 확인한다.

    worker가 손상돼 형식만 자기 일관된 미리보기(규칙과 다른 행·시각, 위조한 성공·멈춤)를 내도 등록하지 않는다.
    부모 재계산의 판정 중단은 고정 코드로만 비교하고, 그 밖의 오류는 고정 문구의 검증 실패다(원본 값을 담지 않는다).
    """
    parent = BundleRef.from_dict(packet["bundle"])
    rules = TimeRestoreRules(**packet["rules"])
    schema = read_bundle_schema(parent_directory, expected=parent)
    index = next((number for number, column in enumerate(schema["columns"]) if column["name"] == rules.column), None)
    try:
        if index is None:
            raise TimeRestoreError("TIME_RULES_INVALID")
        check_column(schema["columns"][index]["dtype"], rules)
        values = [row[index] for _, _, row in iter_bundle_rows(parent_directory, expected=parent)]
        restoration, code = find_restorations(values, rules), None
    except TimeRestoreError as error:
        restoration, code = Restoration((), ()), error.code
    _require(_json(preview) == _json(preview_document(packet, parent.rows, restoration, code)))


def _binding(directory, request, action, committed, parent_directory):
    root = _safe(directory, directory=True)
    names = {"preview.json", "binding.json"} | ({"table"} if action == "apply" else set())
    if committed:
        names.add("AGENT_COMPLETE.json")
    _require({path.name for path in root.iterdir()} == names)
    binding, _ = _read(root, "binding.json", 65536)
    _require(type(binding) is dict and set(binding) == {"version", "request_id", "packet", "preview_sha256", "bundle"})
    _require(binding["version"] == RESULT_VERSION and binding["request_id"] == request.request_id)
    packet = binding["packet"]
    _require(_json(rebuild_packet(packet)) == _json(packet) and packet["action"] == action)
    _require(hashlib.sha256(_json(packet).encode("utf-8")).hexdigest() == request.config_id
             and request.policy_version == POLICY_VERSION and request.stage == "ingest" and request.action_id == "run_ingest"
             and packet["source_id"] == request.source_id and asdict(request.limits) == packet["limits"])
    total = sum(_safe(path, directory=False).stat().st_size for path in root.rglob("*")
                if path.is_file() and path.name != "AGENT_COMPLETE.json")
    _require(total <= packet["max_output_bytes"])
    preview, digest = _read(root, "preview.json", MAX_PREVIEW_BYTES)
    _require(digest == binding["preview_sha256"])
    restoration = preview_restoration(preview, packet)
    _rejudged(preview, packet, parent_directory)
    return root, binding, packet, preview, digest, restoration


def read_preview(directory, expected_request, *, parent_directory, committed=False):
    """고정 요청에 묶인 미리보기 결과만 반환한다. 파생 표는 없어야 하고, 원래 묶음의 재판정과 같아야 한다."""
    _, binding, packet, preview, digest, _ = _binding(directory, expected_request, "preview", committed, parent_directory)
    _require(binding["bundle"] is None and packet["preview_sha256"] is None)
    return {"preview": preview, "preview_sha256": digest}


def _cell(value):
    return _json(_encode(value))


def read_application(directory, expected_request, *, parent_directory, committed=False):
    """파생 묶음을 확인한 미리보기·원래 묶음과 행 단위로 대조한다.

    스키마·행 수·행 위치·인덱스가 같고, 미리보기 행의 시간 셀만 원래 결측에서 미리보기 시각의 규칙 표기로 바뀌며 나머지 셀은 같아야 한다.
    """
    root, binding, packet, preview, digest, restoration = _binding(directory, expected_request, "apply", committed, parent_directory)
    _require(packet["preview_sha256"] == digest and preview["status"] == "ready" and restoration.rows)
    parent = BundleRef.from_dict(packet["bundle"])
    reference = BundleRef.from_dict(binding["bundle"])
    table = root / "table"
    inspect_bundle(table, expected=reference, max_bytes=packet["max_output_bytes"])
    schema = read_bundle_schema(table, expected=reference)
    _require(schema == read_bundle_schema(parent_directory, expected=parent)
             and (reference.rows, reference.columns) == (parent.rows, parent.columns))
    provenance = read_bundle_provenance(table, expected=reference)
    _require(_json(provenance) == _json(provenance_document(packet, restoration, digest)))
    rules = TimeRestoreRules(**packet["rules"])
    index = next(number for number, column in enumerate(schema["columns"]) if column["name"] == rules.column)
    dtype = schema["columns"][index]["dtype"]
    # 원래 묶음의 복원 대상 셀이 결측인지 같은 적용 함수로 확인한 뒤, 기대 셀을 같은 변환으로 만든다.
    apply_restorations([row[index] for _, _, row in iter_bundle_rows(parent_directory, expected=parent)], restoration, rules)
    expected = {row: _cell(restored_cell(time, rules, dtype)) for row, time in zip(restoration.rows, restoration.times)}
    for ordinal, (left, right) in enumerate(zip_longest(iter_bundle_rows(parent_directory, expected=parent),
                                                        iter_bundle_rows(table, expected=reference))):
        _require(left is not None and right is not None and left[0] == right[0] and _cell(left[1]) == _cell(right[1]))
        for number, (before, after) in enumerate(zip(left[2], right[2])):
            if number == index and ordinal in expected:
                _require(_cell(after) == expected[ordinal])
            else:
                _require(_cell(before) == _cell(after))
    return {"preview": preview, "preview_sha256": digest, "bundle": asdict(reference)}
