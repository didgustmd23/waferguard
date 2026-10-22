"""역할 지정 전 입력 진단 작업의 생성·결과 읽기·프로파일 연결. Streamlit에 의존하지 않는다.

진단 결과는 로컬 화면 표시와 사용자 역할 지정에만 쓴다. LLM 대화·해설 자료로 보내지 않는다.
"""

from collections import Counter
from dataclasses import asdict
import csv
import json
import os
from pathlib import Path
from uuid import uuid4

from src.agents import pipeline_approval_cli as backend
from src.agents.artifacts import _root
from src.agents.input_profile_pipeline import ADAPTER_ID, committed_diagnostic, read_profile_table
from src.agents.input_snapshot import PAYLOAD_NAME, create_snapshot
from src.agents.isolated_eda import _read_file
from src.agents.profile_store import load_saved_profile
from src.agents.worker_contract import WorkerLimits, _decode
from src.preprocessing.input_profile import DEDUPE_INPUT, HEADER_DEDUPE_ID
from .dialogue_service import fingerprint
from .execution_service import UiExecution
from .source_selection import SNAPSHOT_ROOT


MAX_HEADER_CHARS = 1024 * 1024
DIAGNOSTIC_NOTICE = ("기초 통계와 이상치 후보는 유한값만으로 계산했습니다. 무한대는 별도 건수로 표시합니다. "
                     "전체 입력 기준의 진단용 값이며 전처리 기준으로 사용하지 않습니다. LLM에는 보내지 않습니다.")


class UiInputProfiling(UiExecution):
    """기존 승인 실행 제어기를 그대로 쓴다. 해설(LLM) 경로를 열지 않는 진단 전용 표식만 더한다."""

    diagnostic_only = True

    def __init__(self, receipt, selection_id, *, source_selection_id, snapshot_directory, purpose="input_profile"):
        super().__init__(receipt, selection_id)
        self.source_selection_id = source_selection_id
        self.snapshot_directory = snapshot_directory
        self.purpose = purpose


def create_profiling(source_selection_id, snapshot_directory, limits, *, confirmed=False, header_dedupe=False):
    """새 이름의 진단(또는 중복 컬럼명 정비) run만 준비한다. 승인 발급·worker 실행은 하지 않는다."""
    if confirmed is not True:
        raise ValueError("입력 진단 작업 준비에 동의해야 합니다.")

    name = ("input-dedupe-" if header_dedupe else "input-profile-") + uuid4().hex[:16]
    receipt = backend.create_input_profile(name, snapshot_directory, WorkerLimits(**limits), confirmed=True,
                                           header_dedupe=header_dedupe)
    directory = _root(Path(receipt["run_dir"]))
    selection = backend._read_json(directory / "selection.json", backend.MAX_SELECTION_BYTES)
    expected = "input_header_dedupe" if header_dedupe else "input_profile_diagnostic"

    if selection["profile"] is not None or selection["pipeline"]["pipeline_id"] != expected:
        raise ValueError("입력 진단 작업 기록이 아닙니다.")

    return UiInputProfiling({"name": name, "run_dir": receipt["run_dir"], "binding_id": receipt["binding_id"]},
                            fingerprint(selection), source_selection_id=source_selection_id,
                            snapshot_directory=selection["snapshot_directory"],
                            purpose="header_dedupe" if header_dedupe else "input_profile")


def _committed(execution, module_id):
    """완료·커밋된 run을 현재 코드·승인 기록·부모 검증기로 다시 확인하고 (run, tool, 출력 폴더)를 반환한다."""
    if not isinstance(execution, UiInputProfiling):
        raise TypeError("입력 진단 작업이 필요합니다.")

    state = execution.read_state()

    if state.status != "completed":
        raise ValueError("완료된 입력 진단 결과만 열 수 있습니다.")

    run = execution._check_current()
    tool, output = committed_diagnostic(run, module_id)
    return run, tool, output


def load_profile_result(execution):
    """완료·커밋된 진단 run의 결과를 현재 코드·승인 기록·부모 검증기로 다시 확인한 뒤 읽는다."""
    run, tool, output = _committed(execution, "input_profile")
    summary, table = read_profile_table(tool, output)
    result = {
        "diagnostic_only": True, "llm_transfer": False, "run_name": execution.name,
        "source_selection_id": execution.source_selection_id,
        "snapshot_directory": tool.contract.snapshot_directory, "snapshot_sha256": tool.contract.snapshot.sha256,
        "summary": summary, "columns": table,
    }
    result["result_id"] = fingerprint(result)
    return result


def read_header(path):
    """CSV 첫 레코드만 문자 상한 안에서 읽는다. 상한을 넘거나 형식이 깨지면 ValueError."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        def lines():
            budget = MAX_HEADER_CHARS

            while True:
                line = stream.readline(budget + 1)

                if not line:
                    return

                budget -= len(line)

                if budget < 0:
                    raise ValueError("헤더 읽기 상한을 초과했습니다.")

                yield line

        try:
            return next(csv.reader(lines()), [])
        except (UnicodeError, csv.Error) as error:
            raise ValueError("CSV 헤더를 읽지 못했습니다.") from error


def blocked_reason(execution):
    """차단된 진단 run의 고정 사유를 판별한다. 차단 기준은 바꾸지 않고 컬럼명을 고치지 않는다.

    승인된 snapshot을 다시 검증한 뒤 헤더 첫 행만 읽는다. 1단계 결과 검증의 컬럼 순서 대조와 같은 범위다.
    반환값에는 고정 코드와 개수만 넣고 컬럼명·값·경로는 넣지 않는다.
    """
    if not isinstance(execution, UiInputProfiling) or execution.read_state().status != "blocked_input":
        return None

    run = execution._check_current()

    if run.binding.adapter_id != ADAPTER_ID:
        raise ValueError("입력 진단 바인딩이 아닙니다.")

    contract = run.binding.input_contract
    contract.check_input()

    try:
        header = read_header(Path(contract.snapshot_directory) / PAYLOAD_NAME)
    except ValueError:
        return {"code": "HEADER_UNREADABLE"}

    if not header or any(not name.strip() for name in header):
        return {"code": "EMPTY_COLUMN_NAME"}

    repeated = [count for count in Counter(header).values() if count > 1]

    if repeated:
        return {"code": "DUPLICATE_COLUMN_NAMES", "duplicated_names": len(repeated), "affected_columns": sum(repeated)}

    return {"code": "INPUT_BLOCKED_OTHER"}


def missing_over(result, threshold):
    """결과 표만으로 결측률 기준 초과 컬럼 수를 센다. 원본을 다시 읽거나 컬럼을 제외하지 않는다."""
    return sum(1 for row in result["columns"] if row["missing_ratio"] > threshold)


def link_record(result, saved, *, derived=False):
    """저장한 프로파일을 같은 입력 선택의 기본값으로만 연결한다. 승인·실행 의미는 없다."""
    return {"path": saved["local_file"], "profile_signature": saved["profile_signature"],
            "workflow_id": saved["workflow_id"], "source_selection_id": result["source_selection_id"],
            "result_id": result["result_id"], "derived_input": derived is True, "execution_approved": False}


def dedupe_message(mapping):
    """바뀐 이름을 원래 이름별로 묶어 고정 형식으로 안내한다. 로컬 화면 전용이며 LLM에 보내지 않는다."""
    groups = {}

    for column in mapping["columns"]:
        if column["original"] != column["renamed"]:
            groups.setdefault(column["original"], []).append(column["renamed"])

    parts = [f"{name}라는 컬럼명이 {len(new)}번 사용되어 {new[0]}~{new[-1]}로 변경했습니다." for name, new in groups.items()]
    return " ".join(parts) + " 데이터 값과 컬럼 순서는 유지했고, 원본은 변경하지 않았습니다."


def register_derived_input(execution, *, root=None):
    """정비 run의 검증된 결과로 새 snapshot과 등록 기록을 만든다. 등록 기록이 생기기 전에는 정상 입력이 아니다.

    파생 입력 파일은 격리 worker가 만들었고 부모 검증을 통과한 것이다. 부모는 바이트 복사만 하며 CSV를 다시 쓰지 않는다.
    새 snapshot의 해시가 검증된 파생 입력 해시와 다르면 등록하지 않는다. 기존 파일은 덮어쓰지 않는다.
    """
    run, tool, output = _committed(execution, HEADER_DEDUPE_ID)
    validated = tool._read_result(output / "pipeline")

    if validated is None:
        raise ValueError("정비 결과 검증에 실패했습니다.")

    contract = tool.contract
    base = {"source_selection_id": execution.source_selection_id, "original_directory": contract.snapshot_directory,
            "run_name": execution.name}

    if not validated["changed"]:
        return {**base, "changed": False, "directory": contract.snapshot_directory,
                "message": "중복 컬럼명이 없어 원본 snapshot을 그대로 사용합니다. 파생 입력을 만들지 않았습니다."}

    root = Path(SNAPSHOT_ROOT if root is None else root).absolute()

    # prepare_source_snapshot과 같은 방식으로 링크 없는 저장 경로만 준비한다.
    for component in reversed((root, *root.parents)):
        component.mkdir(exist_ok=True)
        _root(component)

    name = "derived_" + uuid4().hex
    destination = root / name
    source = output / "pipeline" / tool.module_plan.steps[0].step_id / DEDUPE_INPUT
    snapshot = create_snapshot(source, destination,
                               max_input_bytes=max(contract.snapshot.max_input_bytes, validated["derived_size"]))

    if snapshot.sha256 != validated["derived_sha256"] or snapshot.size_bytes != validated["derived_size"]:
        raise ValueError("등록할 파생 입력이 검증된 정비 결과와 다릅니다.")

    record = {
        "version": "derived_input_record_v1", "rule": validated["mapping"]["rule"], "run_name": execution.name,
        "source_selection_id": execution.source_selection_id,
        "original": {"snapshot_directory": contract.snapshot_directory, "snapshot": asdict(contract.snapshot)},
        "derived": {"snapshot_directory": str(destination), "snapshot": asdict(snapshot)},
        "mapping": validated["mapping"],
    }
    record_path = root / (name + ".json")

    # 등록 기록은 배타 생성한다. 이 파일이 있어야 파생 입력을 정상 입력으로 사용한다.
    with record_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
        stream.flush()
        os.fsync(stream.fileno())

    return {**base, "changed": True, "directory": str(destination), "record_path": str(record_path),
            "message": dedupe_message(validated["mapping"])}


def _check_record(derived):
    """등록 기록을 다시 읽어 파생 snapshot manifest·원본 연결이 그대로인지 확인한다. 내용 해시는 이후 바인더가 다시 검사한다."""
    record = _decode(_read_file(Path(derived["record_path"]), 4 * 1024 * 1024).decode("utf-8"))
    manifest = _decode(_read_file(Path(derived["directory"]) / "snapshot.json", 4096).decode("utf-8"))

    if (record.get("version") != "derived_input_record_v1"
            or record["source_selection_id"] != derived["source_selection_id"]
            or record["original"]["snapshot_directory"] != derived["original_directory"]
            or record["derived"]["snapshot_directory"] != derived["directory"]
            or manifest != {"version": "file_snapshot_v1", **record["derived"]["snapshot"]}):
        raise ValueError("파생 입력 등록 기록이 현재 파일과 맞지 않습니다.")


def effective_snapshot(state, source_selection_id, base_directory):
    """같은 입력 선택에서 등록한 파생 입력이 있으면 그 snapshot을, 없으면 원래 snapshot을 반환한다.

    등록 기록·manifest가 맞지 않으면 원래 snapshot으로 조용히 바꾸지 않고 ValueError를 낸다.
    """
    derived = state.get("ui_derived_input")

    if (isinstance(derived, dict) and derived.get("changed") is True
            and derived.get("source_selection_id") == source_selection_id
            and derived.get("original_directory") == base_directory):
        try:
            _check_record(derived)
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise ValueError("파생 입력 등록 기록을 확인하지 못했습니다.") from error

        return derived["directory"]

    return base_directory


def linked_profile(state, source_selection_id, workflow_id):
    """현재 입력 선택과 목적이 같고 저장 파일이 저장 당시와 같을 때만 연결 경로를 반환한다.

    지문 비교는 저장 후 우발적 교체를 감지하는 용도다. 무결성 인증이 아니며 최종 검사는 이후 바인더가 한다.
    """
    linked = state.get("ui_linked_profile")

    if (not isinstance(linked, dict) or linked.get("source_selection_id") != source_selection_id
            or linked.get("workflow_id") != workflow_id):
        return ""

    try:
        if load_saved_profile(linked["path"]).signature != linked["profile_signature"]:
            return ""
    except (OSError, ValueError, TypeError, KeyError):
        return ""

    return linked["path"]
