"""CSV workflow 제안의 정적 계약. 데이터 읽기·모델 호출·작업 생성·승인을 수행하지 않는다."""

from dataclasses import dataclass
import hashlib
import json
import math
import re

from src.preprocessing.module_contracts import check_id
from src.preprocessing.split import SplitRules
from src.preprocessing.transform import PreparationRules
from .csv_workflow_proposal_catalogue import workflow_proposal_specifications
from .llm_proposal_contract import _decode_response
from .workflow_service_registry import CSV_IDS


SCHEMA_VERSION = "llm_workflow_proposal_v1"


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _object(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError("workflow 계약의 정확한 필드가 필요합니다.")


def _settings(value, spec):
    schema = spec["settings_schema"]
    _object(value, schema["required"])
    kinds = {"null": lambda item: item is None, "string": lambda item: type(item) is str,
             "boolean": lambda item: type(item) is bool, "integer": lambda item: type(item) is int,
             "number": lambda item: type(item) in (int, float) and math.isfinite(item)}
    for key, item in value.items():
        rule = schema["properties"][key]
        types = rule.get("type", [])
        types = [types] if type(types) is str else types
        if (types and not any(kinds[kind](item) for kind in types)
                or "enum" in rule and item not in rule["enum"]
                or "const" in rule and item != rule["const"]):
            raise ValueError("공개 workflow 설정의 타입·선택 값이 올바르지 않습니다.")
        if type(item) in (int, float):
            if ("minimum" in rule and item < rule["minimum"]
                    or "maximum" in rule and item > rule["maximum"]
                    or "exclusiveMinimum" in rule and item <= rule["exclusiveMinimum"]
                    or "exclusiveMaximum" in rule and item >= rule["exclusiveMaximum"]):
                raise ValueError("공개 workflow 설정 범위를 벗어났습니다.")
        if type(item) is str and len(item) > rule.get("maxLength", 256):
            raise ValueError("공개 workflow 설정 문자열이 너무 깁니다.")

    module_id = spec["module_id"]
    if module_id == "csv_split_review":
        SplitRules(**value)
        timestamp = value["timestamp_format"]
        if timestamp is not None and re.fullmatch(r"(?:%[aAbBcdHIjmMpSUwWxXyYZfzGVu%]|[0-9 ./:_+\-TZ])+", timestamp) is None:
            raise ValueError("시간 형식에는 지원하는 날짜 지시자와 구분자만 허용합니다.")
    elif module_id == "csv_prepare_candidate":
        PreparationRules(**value)
    return json.loads(_canonical(value))


@dataclass(frozen=True)
class ValidatedWorkflowProposal:
    payload_json: str
    plan_signature: str | None
    contracts_json: str

    def to_dict(self):
        return {"proposal": json.loads(self.payload_json), "plan_signature": self.plan_signature,
                "contracts": json.loads(self.contracts_json), "proposal_only": True,
                "execution_approved": False, "worker_started": False, "requires_local_binding": True}


def validate_workflow_proposal(response, *, allowed_modules, selected_settings, existing_inputs=()):
    """호출자가 허용한 작업·선택 설정·논리 선행 참조만 사용한다.

    selected_settings는 사용자 명시 값 또는 사용자가 선택한 기본 규칙의 전체 유효값이다.
    이 함수는 자연어 의도·설정 출처·실제 완료 증거를 인증하지 않는다. 로컬 바인딩·검토가 별도로 필요하다.
    """
    catalogue = workflow_proposal_specifications()
    specs = {(spec["module_id"], spec["version"]): spec for spec in catalogue}
    if (type(allowed_modules) is not tuple or not allowed_modules
            or any(type(item) is not tuple or len(item) != 2
                   or any(type(part) is not str for part in item) for item in allowed_modules)
            or len(set(allowed_modules)) != len(allowed_modules)
            or any(item not in specs for item in allowed_modules)):
        raise ValueError("이번 요청에 허용할 공개 CSV 작업 ID·정확한 버전을 지정하세요.")
    if type(selected_settings) is not dict or set(selected_settings) - {item[0] for item in allowed_modules}:
        raise ValueError("허용 작업에 대해 사용자가 선택한 설정이 필요합니다.")
    selected = {key: _settings(value, specs[next(item for item in allowed_modules if item[0] == key)])
                for key, value in selected_settings.items()}
    outputs = {spec["output"]: spec["module_id"] for spec in catalogue}
    if (type(existing_inputs) is not tuple or any(type(item) is not str or item not in outputs for item in existing_inputs)
            or len(set(existing_inputs)) != len(existing_inputs)):
        raise ValueError("기존 선행 입력은 검토된 고정 논리 참조만 허용합니다.")

    value = _decode_response(response)
    _object(value, ("schema_version", "status", "workflow", "questions"))
    if value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("CSV workflow 제안 버전이 필요합니다.")
    questions = value["questions"]
    if (type(questions) is not list or len(questions) > 5
            or any(type(item) is not str or not item.strip() or len(item) > 500
                   or any(ord(char) < 32 for char in item) for item in questions)):
        raise ValueError("확인 질문은 500자 이하의 비어 있지 않은 문장으로 최대 5개까지 허용합니다.")
    if value["status"] == "needs_clarification":
        if value["workflow"] is not None or not questions:
            raise ValueError("미결정 상태에는 workflow 없이 질문만 있어야 합니다.")
        return ValidatedWorkflowProposal(_canonical(value), None, "[]")
    if value["status"] != "proposed" or questions:
        raise ValueError("확정 제안은 proposed와 빈 질문 목록이 필요합니다.")

    workflow = value["workflow"]
    _object(workflow, ("workflow_id", "steps"))
    check_id(workflow["workflow_id"])
    steps = workflow["steps"]
    if type(steps) is not list or not 1 <= len(steps) <= 5:
        raise ValueError("CSV workflow는 1~5개 순차 작업만 허용합니다.")

    available = {"source.selected_input", *existing_inputs}
    seen, contracts, first_rank = [], [], None
    for offset, step in enumerate(steps):
        _object(step, ("step_id", "module_id", "version", "inputs", "settings", "depends_on"))
        check_id(step["step_id"])
        key = (step["module_id"], step["version"])
        if any(type(part) is not str for part in key) or key not in allowed_modules:
            raise ValueError("미허용 CSV 작업 또는 버전입니다.")
        rank = CSV_IDS.index(step["module_id"])
        if first_rank is None:
            first_rank = rank
            if any(CSV_IDS.index(outputs[item]) >= first_rank for item in existing_inputs):
                raise ValueError("새로 생성할 결과와 기존 선행 입력을 혼합할 수 없습니다.")
        if (rank != first_rank + offset or step["step_id"] in seen or step["step_id"] == "source"
                or type(step["depends_on"]) is not list or step["depends_on"] != seen[-1:]):
            raise ValueError("등록 순서를 따르는 연속 선형 작업과 바로 앞 단계 의존성이 필요합니다.")

        spec = specs[key]
        if step["inputs"] != spec["inputs"] or not set(spec["inputs"].values()) <= available:
            raise ValueError("필수 선행 입력 또는 고정 논리 참조가 다릅니다.")
        settings = _settings(step["settings"], spec)
        if step["module_id"] not in selected or _canonical(settings) != _canonical(selected[step["module_id"]]):
            raise ValueError("제안 설정이 사용자가 선택한 유효 설정과 다릅니다.")
        seen.append(step["step_id"])
        available.add(spec["output"])
        contracts.append(spec)

    signed = {"workflow": workflow, "contracts": contracts, "existing_inputs": sorted(existing_inputs)}
    signature = hashlib.sha256(_canonical(signed).encode("utf-8")).hexdigest()
    return ValidatedWorkflowProposal(_canonical(value), signature, _canonical(contracts))
