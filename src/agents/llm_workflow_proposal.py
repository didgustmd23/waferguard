"""공개 CSV 작업 명세와 사용자 선택 규칙만으로 실행안을 제안한다. 실행 권한은 없다."""

import json

from .csv_workflow_proposal_catalogue import workflow_proposal_specifications
from .llm_config import load_local_config
from .llm_proposal import validate_request_text
from .llm_workflow_contract import SCHEMA_VERSION, validate_workflow_proposal
from .local_ollama import LocalProposalError, request_local_proposal


SYSTEM_INSTRUCTIONS = (
    "CSV 전처리 workflow 제안만 한국어 JSON으로 반환하세요. 실행·승인·파일 접근 권한은 없습니다. "
    "역할은 요청한 등록 작업의 순서·의존성 제안이며, 일반적인 전처리 상담이나 모델 학습 조언이 아닙니다. "
    "selected_settings는 사용자가 검토하고 확정한 전체 유효 규칙입니다. 미정인 초안이나 추천 후보가 아닙니다. "
    "요청 작업이 명확하고 그 작업의 규칙이 selected_settings에 있으면 proposed와 questions:[]를 반환하세요. "
    "none, false, null도 명시적으로 선택한 값입니다. 선택 여부를 다시 묻거나 다른 값으로 변경을 권하지 마세요. "
    "사용자가 요청하지 않은 최적화·스케일링·이상치·구간화 필요성을 먼저 질문하지 마세요. "
    "확인 질문은 원하는 작업 종류가 미정이거나 요청 작업의 전체 규칙이 없거나, "
    "사용자의 명시적 변경 요청이 고정 규칙과 충돌하거나 미지원 작업을 요청한 경우에만 하세요. "
    "그때는 needs_clarification, workflow:null과 현재 결정에 필요한 한국어 질문만 반환하세요. "
    "명시적 규칙 변경은 UI에서 규칙을 다시 선택하고 새 대화를 시작해야 합니다. 답변만으로 변경하지 마세요. "
    "입력 컬럼 존재·시간 형식·결측·클래스 분포·실제 분할 가능성·선행 완료 증거는 이후 로컬 검증기의 책임입니다. "
    "그 정보를 제안 전에 요구하지 말고, 검증 완료를 주장하지도 마세요. 조건 불충족 시 로컬 실행기가 중단합니다. "
    "예를 들어 time 규칙이 있으면 시간 컬럼이 있는지 재질문하거나 random 변경을 권하지 마세요. "
    "stratify는 random에만 지원됩니다. time에서 stratify=true나 require_all_classes=false를 권하지 마세요. "
    "scaling=none, outlier=none, binning=none은 유지합니다. binning은 수치형 구간화이며 "
    "범주형 인코딩과 다릅니다. iqr_clip은 클리핑이지 행 삭제가 아닙니다. "
    "등록 순서의 연속 작업만 사용하고 바로 앞 단계에 depends_on으로 의존하세요. "
    "inputs는 공개 명세의 논리 참조를 그대로 사용하세요. 실제 경로·기록을 요구하지 마세요. "
    "settings에는 선택한 전체 값을 그대로 복사하고 누락된 규칙을 추측하거나 기본값으로 채우지 마세요. "
    "품질 검토·분할 확정·후보 검토·적용 승인은 이후 로컬 실행기의 별도 단계입니다. "
    "Train 기준 산출과 저장 기준 적용만 허용하며 Test 적용·기준 재산출·모델 학습은 금지합니다. "
    "기존 입력 참조는 실제 완료 증명이 아닙니다. JSON 외의 문장은 쓰지 마세요.\n"
)


def checked_context(*, allowed_modules, selected_settings, existing_inputs=()):
    """모델 호출 전에도 호출자의 선택 계약을 검사하고 독립 복사본으로 고정한다."""
    context = dict(allowed_modules=allowed_modules, selected_settings=selected_settings,
                   existing_inputs=existing_inputs)
    clarification = {"schema_version": SCHEMA_VERSION, "status": "needs_clarification",
                     "workflow": None, "questions": ["원하는 작업을 확인해 주세요."]}
    validate_workflow_proposal(json.dumps(clarification), **context)

    return json.dumps(context, ensure_ascii=False, allow_nan=False)


def restore_context(context_json):
    context = json.loads(context_json)
    context["allowed_modules"] = tuple(tuple(item) for item in context["allowed_modules"])
    context["existing_inputs"] = tuple(context["existing_inputs"])
    return context


def response_schema(specifications, selected_settings):
    """형식 제한은 생성 보조다. 순서·의존성·선택값의 최종 판정은 정적 검증기가 한다."""
    identifier = {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,63}$"}
    steps = []

    for spec in specifications:
        module_id = spec["module_id"]

        if module_id not in selected_settings:
            continue

        properties = {
            "step_id": identifier, "module_id": {"const": module_id},
            "version": {"const": spec["version"]}, "inputs": {"const": spec["inputs"]},
            "settings": {"const": selected_settings[module_id]},
            "depends_on": {"type": "array", "maxItems": 1, "items": identifier},
        }
        steps.append({"type": "object", "additionalProperties": False,
                      "required": list(properties), "properties": properties})

    workflow = {"type": "object", "additionalProperties": False,
                "required": ["workflow_id", "steps"], "properties": {
                    "workflow_id": identifier,
                    "steps": {"type": "array", "minItems": 1, "maxItems": 5,
                              "items": {"anyOf": steps}},
                }}
    return {
        "type": "object", "additionalProperties": False,
        "required": ["schema_version", "status", "workflow", "questions"],
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "status": {"enum": ["proposed", "needs_clarification"]},
            "workflow": {"anyOf": [workflow, {"type": "null"}]} if steps else {"type": "null"},
            "questions": {"type": "array", "maxItems": 5,
                          "items": {"type": "string", "minLength": 1, "maxLength": 500}},
        },
    }


def build_request_payload(request, config=None, *, allowed_modules, selected_settings, existing_inputs=()):
    """실제 경로·역할·완료 증거를 읽거나 첨부하지 않는다. 자유 문장은 자동 익명화하지 않는다."""
    validate_request_text(request)
    context = restore_context(checked_context(allowed_modules=allowed_modules,
                                             selected_settings=selected_settings, existing_inputs=existing_inputs))
    config = load_local_config() if config is None else config
    specs = [item for item in workflow_proposal_specifications()
             if (item["module_id"], item["version"]) in context["allowed_modules"]]
    schema = response_schema(specs, context["selected_settings"])
    public_context = {
        "modules": specs,
        "selected_settings": context["selected_settings"],
        "settings_status": "fixed_user_selection",
        "input_validation_owner": "local_validator_after_proposal",
        "existing_inputs": context["existing_inputs"],
    }
    system = SYSTEM_INSTRUCTIONS + json.dumps(public_context, ensure_ascii=False)
    payload = {"model": config.model,
               "messages": [{"role": "system", "content": system}, {"role": "user", "content": request}],
               "stream": False, "think": config.think, "format": schema,
               "options": {"temperature": config.temperature, "num_ctx": config.num_ctx,
                           "num_predict": config.num_predict}}

    if len(json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")) > 65536:
        raise ValueError("공개 요청과 누적 대화가 64 KiB 전송 상한을 초과했습니다.")

    return payload


def propose_request(request, config=None, *, backend="http", **context):
    """통신은 한 번만 한다. 계약 위반에 대한 자동 수정·재시도·작업 생성은 없다."""
    config = load_local_config() if config is None else config
    frozen = restore_context(checked_context(**context))
    payload = build_request_payload(request, config, **frozen)

    if backend == "http":
        response = request_local_proposal(payload, config)
    elif backend == "langchain":
        try:
            from .langchain_local import request_langchain_proposal
        except ImportError as error:
            raise LocalProposalError("LANGCHAIN_DEPENDENCY_MISSING") from error

        response = request_langchain_proposal(payload, config)
    else:
        raise LocalProposalError("MODEL_BACKEND_REJECTED")

    try:
        return validate_workflow_proposal(response, **frozen)
    except (ValueError, TypeError, KeyError) as error:
        raise LocalProposalError("WORKFLOW_PROPOSAL_CONTRACT_REJECTED") from error
