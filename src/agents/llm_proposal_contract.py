"""LLM이 제안한 JSON만 검사한다. 파일 읽기·승인·모델 호출·실행 권한은 없다."""

from dataclasses import dataclass
import json

from src.preprocessing.pipeline import validate_pipeline
from .agent_module_registry import build_agent_registry
from .wm_module_plan import validate_wm_modules


MAX_RESPONSE_BYTES = 64 * 1024
SCHEMA_VERSION = "llm_proposal_v1"


@dataclass(frozen=True)
class ValidatedProposal:
    """검토 전 제안을 불변 JSON으로 보관한다. 실행 승인이나 입력 정합성 증거가 아니다."""

    payload_json: str
    plan_signature: str | None

    def to_dict(self):
        # 복사본을 반환하므로 화면 표시용 수정이 검증된 제안 자체를 바꾸지 않는다.
        return {
            "proposal": json.loads(self.payload_json),
            "plan_signature": self.plan_signature,
            "proposal_only": True,
            "execution_approved": False,
            "worker_started": False,
        }


def _unique_object(pairs):
    """같은 키를 두 번 적어 앞의 보안 설정을 숨기는 응답을 거부한다."""
    result = {}

    for key, value in pairs:
        if key in result:
            raise ValueError("제안 JSON에 중복 키가 있습니다.")

        result[key] = value

    return result


def _invalid_number(value):
    raise ValueError("제안 JSON에 NaN·Infinity를 사용할 수 없습니다.")


def _decode_response(response):
    """설명문·Markdown 울타리를 추출하지 않고 응답 전체를 엄격한 JSON으로 해석한다."""
    if type(response) is not str:
        raise ValueError("모델 응답은 JSON 문자열이어야 합니다.")

    try:
        if len(response.encode("utf-8")) > MAX_RESPONSE_BYTES:
            raise ValueError("모델 응답은 64 KiB 이하여야 합니다.")

        return json.loads(
            response,
            object_pairs_hook=_unique_object,
            parse_constant=_invalid_number,
        )
    except (UnicodeError, RecursionError, json.JSONDecodeError) as error:
        # 원문은 데이터가 섞일 수 있어 예외 메시지에 포함하지 않는다.
        raise ValueError("모델 응답의 JSON 형식·인코딩·중첩이 올바르지 않습니다.") from error


def validate_proposal(response, *, expected_input, allowed_modules):
    """신뢰된 호출자가 정한 입력 계약·모듈 허용 목록 안에서만 정적 제안을 허용한다.

    expected_input은 모델 응답에서 가져오지 않는다. 로컬에서 선택한 논리 입력
    계약이며 경로나 데이터를 담지 않는다. allowed_modules도 해당 요청에 허용한
    (모듈 ID, 정확한 버전) 목록이다. 전체 등록 목록을 자동 허용하지 않는다.
    """
    if (type(expected_input) is not dict
            or set(expected_input) != {"contract_id", "contract_version", "partition"}
            or type(expected_input["contract_id"]) is not str
            or type(expected_input["contract_version"]) is not int
            or expected_input["contract_version"] != 1
            or type(expected_input["partition"]) is not str):
        raise ValueError("로컬에서 선택한 논리 입력 계약 v1이 필요합니다.")

    if (type(allowed_modules) is not tuple or not allowed_modules
            or any(type(item) is not tuple or len(item) != 2
                   or any(type(part) is not str for part in item)
                   for item in allowed_modules)):
        raise ValueError("요청에 허용할 모듈 ID·버전 tuple을 명시하세요.")

    registry = build_agent_registry()

    for module_id, version in allowed_modules:
        registry.resolve(module_id, version)

    value = _decode_response(response)

    if (type(value) is not dict
            or set(value) != {"schema_version", "status", "pipeline", "questions"}
            or value["schema_version"] != SCHEMA_VERSION):
        raise ValueError("지원하는 제안 버전과 정확한 필드만 사용할 수 있습니다.")

    questions = value["questions"]

    if (type(questions) is not list or len(questions) > 5
            or any(type(item) is not str or not item.strip() or len(item) > 500
                   for item in questions)):
        raise ValueError("확인 질문은 비어 있지 않은 500자 이하 문장으로 최대 5개까지 허용합니다.")

    if value["status"] == "needs_clarification":
        if value["pipeline"] is not None or not questions:
            raise ValueError("추가 질문 상태에는 실행안 없이 확인 질문만 있어야 합니다.")

        signature = None

    elif value["status"] == "proposed":
        if questions:
            raise ValueError("실행안 제안과 미해결 질문을 함께 확정할 수 없습니다.")

        pipeline = value["pipeline"]

        if type(pipeline) is not dict or pipeline.get("input") != expected_input:
            raise ValueError("제안 입력은 로컬에서 선택한 논리 입력 계약과 같아야 합니다.")

        if pipeline.get("schema_version") == "module_pipeline_v1":
            plan = validate_pipeline(pipeline, registry)
            _require_quality_gate(plan)

        elif pipeline.get("schema_version") == "wm_module_pipeline_v1":
            plan = validate_wm_modules(pipeline)

        else:
            raise ValueError("지원하지 않는 파이프라인 제안 종류입니다.")

        if any((step.module_id, step.version) not in allowed_modules for step in plan.steps):
            raise ValueError("이번 요청에 허용하지 않은 모듈을 제안했습니다.")

        signature = plan.signature

    else:
        raise ValueError("제안 상태는 proposed 또는 needs_clarification이어야 합니다.")

    return ValidatedProposal(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False),
        signature,
    )


def _require_quality_gate(plan):
    """표형 분석은 선행 품질 검사에 의존해야 한다. 검토 중단을 우회하지 않는다."""
    quality_ancestors = set()

    for step in plan.steps:
        contract = json.loads(step.contract_json)
        has_gate = bool(set(step.depends_on) & quality_ancestors)

        if step.module_id == "tabular_quality" or has_gate:
            quality_ancestors.add(step.step_id)

        if contract["kind"] in ("analysis", "transformation", "reporting") and not has_gate:
            raise ValueError("분석·변환·보고 단계는 선행 품질 검사 통과에 의존해야 합니다.")
