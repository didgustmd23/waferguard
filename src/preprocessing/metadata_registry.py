"""고정 모듈 명세와 JSON 설정 스키마만 읽는 계획 검증용 등록부."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from .builtin_modules import build_builtin_registry
from .module_contracts import ModuleContract


SCHEMA_PATH = Path("configs/agents/modules/missing_row_histogram_settings.json")
SCHEMA_SHA256 = "1e8bc8d4888b1ada4a6ff67127dbdddad61797329c3e899a3d9345ca3116b1bd"
MAX_SCHEMA_BYTES = 16 * 1024
MAX_STRING_LENGTH = 4096


def _load_schema(project_root, contract):
    """해시가 고정된 작은 JSON 스키마를 읽고 구조·필드 제한을 검증한다."""
    path = Path(project_root) / SCHEMA_PATH
    payload = path.read_bytes()

    if len(payload) > MAX_SCHEMA_BYTES:
        raise ValueError("모듈 설정 스키마가 허용 크기를 초과했습니다.")

    normalized = payload.replace(b"\r\n", b"\n")

    if hashlib.sha256(normalized).hexdigest() != SCHEMA_SHA256:
        raise ValueError("모듈 설정 스키마가 검토된 해시와 다릅니다.")

    def unique_pairs(items):
        result = {}

        for key, value in items:
            if key in result:
                raise ValueError("모듈 설정 스키마에 중복 키가 있습니다.")

            result[key] = value

        return result

    def invalid_number(_value):
        raise ValueError("모듈 설정 스키마에 비표준 숫자가 있습니다.")

    schema = json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=unique_pairs,
        parse_constant=invalid_number,
    )
    expected = {"schema_version", "module_id", "module_version", "fields"}

    if type(schema) is not dict or set(schema) != expected:
        raise ValueError("모듈 설정 스키마의 필드가 올바르지 않습니다.")

    if (schema["schema_version"] != "module_settings_schema_v1"
            or schema["module_id"] != contract.module_id
            or schema["module_version"] != contract.version
            or type(schema["fields"]) is not dict
            or len(schema["fields"]) > 64):
        raise ValueError("모듈 설정 스키마의 ID 또는 버전이 올바르지 않습니다.")

    for name, rule in schema["fields"].items():
        if type(name) is not str or not name.isidentifier() or type(rule) is not dict:
            raise ValueError("모듈 설정 필드 정의가 올바르지 않습니다.")

        allowed = {"type", "required", "default", "minimum", "maximum", "min_length", "max_length", "values"}

        if set(rule) - allowed or "type" not in rule:
            raise ValueError("모듈 설정 규칙에 허용되지 않은 항목이 있습니다.")

        if type(rule["type"]) is not str or rule["type"] not in {"boolean", "integer", "number", "string", "enum"}:
            raise ValueError("지원하지 않는 모듈 설정 자료형입니다.")

        allowed_by_type = {
            "boolean": {"type", "required", "default"},
            "integer": {"type", "required", "default", "minimum", "maximum"},
            "number": {"type", "required", "default", "minimum", "maximum"},
            "string": {"type", "required", "default", "min_length", "max_length"},
            "enum": {"type", "required", "default", "values"},
        }

        if set(rule) - allowed_by_type[rule["type"]]:
            raise ValueError("설정 자료형에 맞지 않는 스키마 항목이 있습니다.")

        if "required" in rule and type(rule["required"]) is not bool:
            raise ValueError("설정 필드 required는 bool이어야 합니다.")

        if rule["type"] == "enum":
            values = rule.get("values")

            if type(values) is not list or not values or len(values) > 128:
                raise ValueError("enum 설정은 1~128개 허용값을 명시해야 합니다.")

            if any(type(value) not in (str, int, float, bool, type(None)) for value in values):
                raise ValueError("enum 허용값은 단순 JSON 값이어야 합니다.")

            if len({(type(value).__name__, repr(value)) for value in values}) != len(values):
                raise ValueError("enum 허용값은 중복될 수 없습니다.")

        if rule["type"] == "string":
            for key in ("min_length", "max_length"):
                if key in rule and (type(rule[key]) is not int or not 0 <= rule[key] <= MAX_STRING_LENGTH):
                    raise ValueError("문자열 길이 제한이 올바르지 않습니다.")

        if rule["type"] in {"integer", "number"}:
            for key in ("minimum", "maximum"):
                if key in rule:
                    value = rule[key]

                    if type(value) not in (int, float) or not math.isfinite(value):
                        raise ValueError("숫자 범위 제한은 유한한 숫자여야 합니다.")

            if ("minimum" in rule and "maximum" in rule
                    and rule["minimum"] > rule["maximum"]):
                raise ValueError("숫자 범위의 최소값이 최대값보다 큽니다.")

        if "default" in rule:
            _validate_value(name, rule, rule["default"])

        if rule.get("required", False) and "default" in rule:
            raise ValueError("required 필드에는 기본값을 함께 지정할 수 없습니다.")

    return schema


def _validate_value(name, rule, value):
    """JSON 설정값 하나를 선언된 기본 자료형·범위에 맞춰 검사한다."""
    kind = rule["type"]

    if kind == "boolean":
        valid = type(value) is bool
    elif kind == "integer":
        valid = type(value) is int
    elif kind == "number":
        valid = type(value) in (int, float) and math.isfinite(value)
    elif kind == "string":
        valid = type(value) is str and len(value) <= MAX_STRING_LENGTH
    else:
        valid = any(type(value) is type(item) and value == item for item in rule["values"])

    if not valid:
        raise ValueError(f"모듈 설정 '{name}'의 자료형 또는 허용값이 올바르지 않습니다.")

    if kind in {"integer", "number"}:
        if "minimum" in rule and value < rule["minimum"]:
            raise ValueError(f"모듈 설정 '{name}'이 최솟값보다 작습니다.")

        if "maximum" in rule and value > rule["maximum"]:
            raise ValueError(f"모듈 설정 '{name}'이 최댓값보다 큽니다.")

    if kind == "string":
        if len(value) < rule.get("min_length", 0) or len(value) > rule.get("max_length", MAX_STRING_LENGTH):
            raise ValueError(f"모듈 설정 '{name}'의 문자열 길이가 허용 범위를 벗어났습니다.")


@dataclass(frozen=True)
class MetadataModule:
    """계약과 JSON 스키마만 가진 계획 검증 항목으로 계산 함수를 포함하지 않는다."""

    contract: object
    settings_schema: dict

    def validate_settings(self, settings=None):
        """기본값을 적용하고 잘못되거나 알 수 없는 설정은 거부한다."""
        settings = {} if settings is None else settings

        if type(settings) is not dict or any(type(key) is not str for key in settings):
            raise ValueError("모듈 설정은 문자열 키를 가진 JSON 객체여야 합니다.")

        fields = self.settings_schema["fields"]

        if set(settings) - set(fields):
            raise ValueError("등록된 모듈에 허용되지 않은 설정이 있습니다.")

        effective = {}

        for name, rule in fields.items():
            if name in settings:
                value = settings[name]
            elif "default" in rule:
                value = rule["default"]
            elif rule.get("required", False):
                raise ValueError(f"필수 모듈 설정 '{name}'이 없습니다.")
            else:
                continue

            _validate_value(name, rule, value)
            effective[name] = value

        return effective


class MetadataRegistry:
    """고정 내장 모듈과 사용자 모듈의 선언 메타데이터를 버전별로 조회한다."""

    def __init__(self, modules):
        self._modules = dict(modules)

    def resolve(self, module_id, version):
        """정확히 등록된 ID·버전만 반환한다."""
        module = self._modules.get((module_id, version))

        if module is None:
            raise ValueError("등록되지 않은 모듈 ID 또는 지원하지 않는 버전입니다.")

        return module


def build_metadata_registry(project_root=None, *, reviewed_contract=None):
    """내장 등록부에 호출자가 검토한 계약의 메타데이터만 추가한다.

    파일·코드 해시와 실행 권한 검사는 에이전트 조립 계층의 책임이다.
    공통 코어는 에이전트를 import하거나 사용자 계산 코드를 로딩하지 않는다.
    계약을 전달하지 않으면 내장 모듈만 반환하며 미검토 모듈을 자동 등록하지 않는다.
    """
    root = Path(__file__).resolve().parents[2] if project_root is None else Path(project_root)
    builtin = build_builtin_registry()
    modules = {
        (contract.module_id, contract.version): builtin.resolve(contract.module_id, contract.version)
        for contract in builtin.contracts()
    }

    if reviewed_contract is None:
        return MetadataRegistry(modules)

    if not isinstance(reviewed_contract, ModuleContract):
        raise TypeError("검토된 ModuleContract가 필요합니다.")

    contract = reviewed_contract
    schema = _load_schema(root, contract)

    if set(schema["fields"]) != set(contract.settings_fields):
        raise ValueError("설정 스키마 필드가 고정 모듈 계약과 일치하지 않습니다.")

    key = (contract.module_id, contract.version)

    if key in modules:
        raise ValueError("검토 계약으로 내장 모듈을 덮어쓸 수 없습니다.")

    modules[key] = MetadataModule(contract, schema)
    return MetadataRegistry(modules)
