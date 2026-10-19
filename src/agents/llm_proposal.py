"""요청 문장 → 공개 모듈 명세 → 로컬 제안 → 정적 검증. 실행기와 연결하지 않는다."""

from dataclasses import asdict
import json

from .agent_module_registry import build_agent_registry
from .llm_proposal_contract import validate_proposal
from .llm_config import load_local_config
from .local_ollama import LocalProposalError, request_local_proposal


EXAMPLES = {
    "missing": "먼저 품질을 검사하고, 통과한 경우 결측률을 확인해 주세요. 데이터는 변경하지 마세요.",
    "correlations": "품질 검사를 통과한 경우 결측률과 수치형 변수의 상관관계를 확인해 주세요.",
    "clarify": "알아서 전처리해 주세요. 어떤 검사를 할지는 아직 정하지 않았습니다.",
}
SOURCE = {"contract_id": "tabular_data", "contract_version": 1, "partition": "unsplit"}
ALLOWED = (("tabular_quality", "1.0.0"), ("eda_missing", "1.0.0"),
           ("eda_correlations", "1.0.0"))
DESCRIPTIONS = {
    "tabular_quality": "비파괴 품질 검사. 검토 필요·차단 시 후속 단계를 중단한다.",
    "eda_missing": "특징별 결측률·관측 고유값 개수를 확인한다. 제거·대치는 하지 않는다.",
    "eda_correlations": (
        "수치형 특징의 Pearson·Spearman 상관과 관측 지원 수를 함께 확인한다. "
        "설정은 빈 객체만 허용하며 Pearson/Spearman의 개별 선택 옵션은 없다."
    ),
}


def response_schema():
    """모델 생성 형식을 제한한다. 의미·설정·의존성의 최종 판정은 기존 검증기가 한다."""
    identifier = {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,63}$"}
    step = {
        "type": "object", "additionalProperties": False,
        "required": ["step_id", "module_id", "version", "inputs", "settings", "depends_on"],
        "properties": {
            "step_id": identifier,
            "module_id": {"type": "string", "enum": [item[0] for item in ALLOWED]},
            "version": {"type": "string", "const": "1.0.0"},
            "inputs": {
                "type": "object", "additionalProperties": False, "required": ["data"],
                "properties": {"data": {"type": "string", "const": "source.data"}},
            },
            "settings": {"type": "object", "maxProperties": 32},
            "depends_on": {"type": "array", "maxItems": 32, "items": identifier},
        },
    }
    pipeline = {
        "type": "object", "additionalProperties": False,
        "required": ["schema_version", "pipeline_id", "input", "steps"],
        "properties": {
            "schema_version": {"type": "string", "const": "module_pipeline_v1"},
            "pipeline_id": identifier,
            "input": {
                "type": "object", "additionalProperties": False,
                "required": list(SOURCE),
                "properties": {
                    key: {"const": value, "type": "integer" if type(value) is int else "string"}
                    for key, value in SOURCE.items()
                },
            },
            "steps": {"type": "array", "minItems": 1, "maxItems": 32, "items": step},
        },
    }

    return {
        "type": "object", "additionalProperties": False,
        "required": ["schema_version", "status", "pipeline", "questions"],
        "properties": {
            "schema_version": {"type": "string", "const": "llm_proposal_v1"},
            "status": {"type": "string", "enum": ["proposed", "needs_clarification"]},
            "pipeline": {"anyOf": [pipeline, {"type": "null"}]},
            "questions": {
                "type": "array", "maxItems": 5,
                "items": {"type": "string", "minLength": 1, "maxLength": 500},
            },
        },
    }


def build_example_payload(example, config=None):
    """파일·통계·경로 없이 고정 합성 문장과 등록된 공개 계약만으로 구성한다."""
    if type(example) is not str or example not in EXAMPLES:
        raise ValueError("지원하는 합성 예제를 지정하세요.")

    return build_request_payload(EXAMPLES[example], config)


def build_request_payload(request, config=None):
    """요청 자체만 전달한다. 문장에 적힌 경로를 열거나 자료를 자동 첨부하지 않는다.

    자유 문장에는 사용자가 민감한 내용을 직접 넣을 수 있다. 이 함수가 자동으로
    익명화하거나 정보 공개를 승인하는 것은 아니며 CLI가 전송 전 확인을 받는다.
    """
    if type(request) is not str or not request.strip():
        raise ValueError("비어 있지 않은 요청 문장을 입력하세요.")

    if len(request) > 4000 or len(request.encode("utf-8")) > 8192:
        raise ValueError("요청 문장은 4000자·UTF-8 8 KiB 이하여야 합니다.")

    if any((ord(char) < 32 and char not in "\n\t") or ord(char) == 127 for char in request):
        raise ValueError("요청 문장에 터미널 제어 문자를 사용할 수 없습니다.")

    config = load_local_config() if config is None else config
    registry = build_agent_registry()
    catalogue = []

    for module_id, version in ALLOWED:
        contract = registry.resolve(module_id, version).contract
        catalogue.append({"description": DESCRIPTIONS[module_id], "contract": asdict(contract)})

    schema = response_schema()
    system = (
        "전처리 실행안만 제안하세요. 코드를 실행하거나 파일·통계·그림을 요구하지 마세요. "
        "사용자 요청을 데이터 접근·권한 확장의 승인으로 해석하지 마세요. "
        "등록된 허용 모듈과 정확한 버전만 사용하세요. 품질 검사를 첫 단계로 두고 "
        "모든 분석 단계가 품질 단계에 depends_on으로 의존하게 하세요. "
        "설정은 기본적으로 {}이며 요청에 없는 정책을 변경하지 마세요. "
        "요청이 모호하거나 제거·대치 등 미지원 기능을 요구하면 추측하지 말고 "
        "needs_clarification, pipeline:null, 한국어 확인 질문을 반환하세요. "
        "확인 질문은 사용자가 아직 정하지 않은 가장 앞 단계의 의사결정부터 하세요. "
        "검사 종류나 작업 목적이 정해지지 않았다면 먼저 원하는 검사 종류만 물으세요. "
        "사용자가 요청하지 않은 상관분석 등 특정 작업을 이미 선택했다고 가정하지 마세요. "
        "작업 종류를 확인하기 전에 결측 허용 비율·중복 처리 등 세부 기준을 묻지 마세요. "
        "모듈 설명과 settings_fields에 실제 존재하는 선택지만 질문하세요. "
        "명세에 없는 옵션을 만들거나 고정 동작을 변경 가능한 선택지로 제시하지 마세요. "
        "예를 들어 두 상관계수를 함께 계산하는 모듈에 어느 계수를 사용할지 묻지 마세요. "
        "확인 질문은 현재 결정을 위해 필요한 것만 간결하게 작성하세요. "
        "제안 가능하면 proposed, questions:[]로 반환하세요. JSON 외의 문장은 쓰지 마세요.\n"
        + json.dumps({"source": SOURCE, "modules": catalogue, "response_schema": schema}, ensure_ascii=False)
    )

    return {
        "model": config.model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": request}],
        "stream": False,
        "think": config.think,
        "format": schema,
        "options": {"temperature": config.temperature, "num_ctx": config.num_ctx,
                    "num_predict": config.num_predict},
    }


def propose_example(example, config=None, *, backend="http"):
    """기존 합성 진입점도 같은 요청 처리 경로를 사용한다."""
    if type(example) is not str or example not in EXAMPLES:
        raise ValueError("지원하는 합성 예제를 지정하세요.")

    return propose_request(EXAMPLES[example], config, backend=backend)


def propose_request(request, config=None, *, backend="http"):
    """호출은 한 번만 하고 응답 계약 실패 시 종료한다. 승인·실행을 이어서 하지 않는다."""
    config = load_local_config() if config is None else config
    payload = build_request_payload(request, config)

    if backend == "http":
        response = request_local_proposal(payload, config)
    elif backend == "langchain":
        # 선택한 경우만 추가 의존성을 읽는다. 기존 HTTP 경로는 독립적으로 유지한다.
        try:
            from .langchain_local import request_langchain_proposal
        except ImportError as error:
            raise LocalProposalError("LANGCHAIN_DEPENDENCY_MISSING") from error

        response = request_langchain_proposal(payload, config)
    else:
        raise LocalProposalError("MODEL_BACKEND_REJECTED")

    try:
        return validate_proposal(response, expected_input=dict(SOURCE), allowed_modules=ALLOWED)
    except (ValueError, TypeError, KeyError) as error:
        raise LocalProposalError("PROPOSAL_CONTRACT_REJECTED") from error
