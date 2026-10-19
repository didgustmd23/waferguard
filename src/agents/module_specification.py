"""공통 UI·서비스용 선언 명세. 조회는 데이터 읽기·코드 실행·승인 발급이 아니다."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from src.preprocessing.module_contracts import check_id
from .isolated_eda import _read_file
from .worker_contract import _decode
from .worker_paths import _path


CATALOGUE_PATH = Path(__file__).resolve().parents[2] / "configs/agents/modules/catalogue.json"
CATALOGUE_VERSION = "agent_module_catalogue_v1"
MAX_CATALOGUE_BYTES = 128 * 1024

# 식별자는 향후 서비스 연결용 선언이다. import 경로·셸 명령으로 해석하지 않는다.
EXECUTION_PROFILES = {
    "tabular_analysis": ("calculation", "tabular_module_pipeline_v1", "module_pipeline_code_v1", "run_pipeline"),
    "tabular_candidate": ("calculation", "tabular_preparation_candidate_v1", "preparation_candidate_code_v1", "derive_preparation"),
    "tabular_apply": ("calculation", "tabular_preparation_apply_v1", "preparation_apply_code_v1", "apply_preparation"),
    "wm_workflow": ("workflow", "wm_workflow_v1", "existing_wm_worker_code_v1", None),
}
RESULT_VIEWS = frozenset({"table", "chart", "image", "report", "artifact_list"})
FIELDS = frozenset({"module_id", "version", "title", "description", "execution_profile",
                    "settings_schema", "result_views", "ui_visible", "llm_proposable"})


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _settings_schema(identifier, module):
    """기존 규칙 객체에서 기본값을 읽는다. 실행 시 판정은 기존 검증기가 최종 담당한다."""
    if type(identifier) is not str or identifier not in {"empty", "quality_rules", "preparation_rules", "workflow_creation"}:
        raise ValueError("등록되지 않은 설정 스키마 ID입니다.")

    defaults = {}
    constraints = {}

    if identifier in {"quality_rules", "preparation_rules"}:
        validated = module.validate_settings()
        rules = validated.get("rules")

        if rules is None:
            raise ValueError("규칙 객체와 설정 스키마가 일치하지 않습니다.")

        from src.preprocessing.quality import QualityRules
        from src.preprocessing.transform import PreparationRules

        expected = QualityRules if identifier == "quality_rules" else PreparationRules

        if not isinstance(rules, expected):
            raise ValueError("다른 규칙 객체의 설정 스키마를 사용할 수 없습니다.")

        defaults = json.loads(_json(asdict(rules)))
        constraints["missing_ratio_threshold"] = {"minimum": 0, "maximum": 1}

        if identifier == "quality_rules":
            constraints["allowed_labels"] = {"type": ["array", "null"], "minItems": 1,
                                               "items": {"type": ["string", "number", "boolean"]}}
            constraints["timestamp_format"] = {"type": ["string", "null"], "minLength": 1}

            for name, value in defaults.items():
                if value == "review":
                    constraints[name] = {"enum": ["ignore", "review", "block"]}
        else:
            constraints["missing_ratio_threshold"]["type"] = ["number", "null"]
            choices = {
                "numeric_imputation": ["median", "mean", "none"],
                "scaling": ["none", "standard", "minmax", "robust"],
                "outlier": ["none", "iqr_clip"], "binning": ["none", "quantile", "uniform"],
                "bin_encoding": ["ordinal", "onehot"],
                "categorical_imputation": ["missing_category", "most_frequent"],
                "unknown_category": ["error", "ignore"], "algorithm_version": [defaults["algorithm_version"]],
            }
            constraints.update({name: {"enum": values} for name, values in choices.items()})
            constraints["iqr_multiplier"] = {"exclusiveMinimum": 0}

            for name, minimum, maximum in (("n_bins", 2, 100), ("max_categories", 1, 4096),
                                            ("max_output_features", 1, 100000),
                                            ("max_output_cells", 1, 100000000)):
                constraints[name] = {"minimum": minimum, "maximum": maximum}

    elif module.contract.settings_fields:
        raise ValueError("설정이 있는 모듈에 빈 스키마를 연결할 수 없습니다.")

    if set(defaults) != set(module.contract.settings_fields):
        raise ValueError("설정 스키마와 구현의 허용 필드가 다릅니다.")

    properties = {}
    types = {str: "string", bool: "boolean", int: "integer", float: "number", type(None): "null"}

    for name, value in defaults.items():
        properties[name] = {"type": types.get(type(value), "array"), "default": value,
                            **constraints.get(name, {})}

    return {"schema_id": identifier, "type": "object", "properties": properties, "required": [],
            "additionalProperties": False,
            "scope": "workflow_creation" if identifier == "workflow_creation" else "module",
            "form_ready": identifier != "workflow_creation",
            "authoritative_validation": "registered_settings_validator",
            "notice": "WM 계획 설정은 빈 객체입니다. 작업 생성 설정 스키마는 서비스 분리 단계에서 연결합니다."
                      if identifier == "workflow_creation" else "설정 조합은 기존 검증기로 재검사해야 합니다."}


@dataclass(frozen=True)
class ModuleSpecification:
    """직렬화 가능한 불변 명세. 함수·실데이터·로컬 경로·승인 ID를 담지 않는다."""

    payload_json: str

    @property
    def signature(self):
        return hashlib.sha256(self.payload_json.encode("utf-8")).hexdigest()

    def to_dict(self):
        """호출자가 수정해도 등록 정보가 바뀌지 않도록 새 객체를 반환한다."""
        return json.loads(self.payload_json)


class ModuleCatalogue:
    """검토된 구현과 별도 명세를 대조한다. 목록 누락·중복·잘못된 연결은 거부한다."""

    def __init__(self, document, registry):
        if (type(document) is not dict or set(document) != {"schema_version", "modules"}
                or document["schema_version"] != CATALOGUE_VERSION
                or type(document["modules"]) is not list or not 1 <= len(document["modules"]) <= 256):
            raise ValueError("모듈 목록 명세 형식이 올바르지 않습니다.")

        expected = {(item.module_id, item.version) for item in registry.contracts()}
        self._specifications = {}

        for record in document["modules"]:
            if type(record) is not dict or set(record) != FIELDS:
                raise ValueError("모듈 명세 필드가 올바르지 않습니다.")

            check_id(record["module_id"])
            key = (record["module_id"], record["version"])

            if type(key[1]) is not str or key not in expected or key in self._specifications:
                raise ValueError("모듈 명세의 ID·버전이 미등록이거나 중복되었습니다.")

            for field, maximum in (("title", 100), ("description", 2000)):
                value = record[field]

                if (type(value) is not str or not value.strip() or len(value) > maximum
                        or any(ord(char) < 32 or ord(char) == 127 for char in value)):
                    raise ValueError("모듈 이름·설명에 빈 값·제어 문자·과도한 길이를 사용할 수 없습니다.")

            if any(type(record[name]) is not bool for name in ("ui_visible", "llm_proposable")):
                raise ValueError("모듈 노출 정책은 bool이어야 합니다.")

            views = record["result_views"]
            if (type(views) is not list or not views or any(type(view) is not str for view in views)
                    or len(views) != len(set(views)) or set(views) - RESULT_VIEWS):
                raise ValueError("허용된 공통 결과 표시 유형을 중복 없이 지정하세요.")

            profile = record["execution_profile"]
            if type(profile) is not str or profile not in EXECUTION_PROFILES:
                raise ValueError("등록되지 않은 실행 프로파일입니다.")

            module = registry.resolve(*key)
            kind, service, runtime, stage = EXECUTION_PROFILES[profile]
            from .wm_module_catalogue import WmWorkflowModule

            is_workflow = isinstance(module, WmWorkflowModule)
            if is_workflow != (kind == "workflow"):
                raise ValueError("계산 모듈과 다단계 승인 작업의 실행 방식을 혼용할 수 없습니다.")

            expected_profile = ("wm_workflow" if is_workflow else
                                "tabular_candidate" if module.contract.criteria_policy == "derive_on_train" else
                                "tabular_apply" if module.contract.criteria_policy == "apply_saved" else "tabular_analysis")
            if (profile != expected_profile
                    or is_workflow != (record["settings_schema"] == "workflow_creation")):
                raise ValueError("모듈의 기준 처리 방식·설정과 실행 프로파일이 일치하지 않습니다.")

            settings = _settings_schema(record["settings_schema"], module)
            value = {
                "schema_version": "agent_module_specification_v1", "contract": asdict(module.contract),
                "title": record["title"], "description": record["description"], "settings": settings,
                "execution": {"kind": kind, "service_id": service, "runtime_manifest_id": runtime,
                              "worker_stage": module.worker_stage if is_workflow else stage,
                              "review_stages": list(module.review_stages) if is_workflow else [],
                              "connection_status": "service_registered" if is_workflow else "declaration_only"},
                "result_views": views,
                "exposure": {"ui_visible": record["ui_visible"], "llm_proposable": record["llm_proposable"]},
            }
            self._specifications[key] = ModuleSpecification(_json(value))

        if set(self._specifications) != expected:
            raise ValueError("등록된 구현 중 설명 명세가 누락된 모듈이 있습니다.")

    def resolve(self, module_id, version):
        check_id(module_id)
        if type(version) is not str or (module_id, version) not in self._specifications:
            raise ValueError("등록되지 않은 명세 ID 또는 버전입니다.")

        return self._specifications[(module_id, version)]

    def list(self, *, audience="all"):
        """노출 정책에 맞는 새 JSON 목록만 반환한다. 실행 가능 여부 판단은 아니다."""
        if type(audience) is not str or audience not in {"all", "ui", "llm"}:
            raise ValueError("명세 조회 대상은 all·ui·llm 중 하나여야 합니다.")

        values = [item.to_dict() for _, item in sorted(self._specifications.items())]
        field = "ui_visible" if audience == "ui" else "llm_proposable"
        return values if audience == "all" else [item for item in values if item["exposure"][field]]

    @property
    def signature(self):
        """명세 전체의 변경 식별값이며 실행 승인이나 인증 서명이 아니다."""
        return hashlib.sha256(_json(self.list()).encode("utf-8")).hexdigest()


def load_catalogue(registry):
    """고정된 명세 파일만 크기·중복 키·비표준 숫자 검사를 거쳐 읽는다."""
    path = _path(CATALOGUE_PATH, directory=False)
    document = _decode(_read_file(path, MAX_CATALOGUE_BYTES).decode("utf-8"))
    return ModuleCatalogue(document, registry)
