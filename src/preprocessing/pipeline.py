"""사용자 모듈 설정의 정적 검증과 순차 계획 생성. 데이터를 읽거나 실행하지 않는다."""

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
from pathlib import Path

from .builtin_modules import build_builtin_registry
from .module_contracts import check_id


MAX_CONFIG_BYTES = 64 * 1024
MAX_STEPS = 32
RUNTIME_CHECKS = frozenset({"snapshot_verified", "tabular_roles_validated", "safe_loader_completed"})


def _canonical(value):
    """키 순서에 독립적인 JSON을 생성하며 비표준 숫자는 거부한다."""
    def convert(item):
        if is_dataclass(item) and not isinstance(item, type):
            return asdict(item)

        raise TypeError("계획에는 JSON 값과 검토된 설정 객체만 사용할 수 있습니다.")

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False, default=convert)


def _object(value, required, optional=()):
    """필수·허용 필드 외에는 거부한다. 임의 실행 옵션이 섞이지 않게 한다."""
    if type(value) is not dict or set(required) - set(value) or set(value) - set(required) - set(optional):
        raise ValueError("파이프라인 설정의 필수 필드 또는 허용 필드가 올바르지 않습니다.")


@dataclass(frozen=True)
class PlannedStep:
    """정규화된 한 단계. JSON 문자열로 보관해 설정의 사후 변경을 막는다."""

    step_id: str
    module_id: str
    version: str
    inputs: tuple[tuple[str, str], ...]
    depends_on: tuple[str, ...]
    settings_json: str
    effective_settings_json: str
    contract_json: str


@dataclass(frozen=True)
class PipelinePlan:
    """설정 검증 결과. source/code/승인에 결합된 실행 계획을 대신하지 않는다."""

    pipeline_id: str
    partition: str
    steps: tuple[PlannedStep, ...]

    def to_dict(self):
        return {
            "schema_version": "module_pipeline_plan_v1",
            "pipeline_id": self.pipeline_id,
            "input": {"contract_id": "tabular_data", "contract_version": 1, "partition": self.partition},
            "stop_on_review_or_block": True,
            "steps": [asdict(step) for step in self.steps],
        }

    @property
    def signature(self):
        """모듈 버전·유효 설정·순서·연결·계약 변경을 식별한다. 보안 서명은 아니다."""
        return hashlib.sha256(_canonical(self.to_dict()).encode("utf-8")).hexdigest()


def _bindings(raw, contract, available):
    """이전 출력만 참조하고 입출력 계약 ID·버전을 모두 대조한다."""
    if type(raw) is not dict:
        raise ValueError("단계 입력 연결은 dict여야 합니다.")

    ports = {port.name: port for port in contract.inputs}

    if set(raw) - set(ports) or any(port.required and port.name not in raw for port in contract.inputs):
        raise ValueError("단계의 필수 입력이 없거나 미등록 입력 포트를 지정했습니다.")

    for name, reference in raw.items():
        if type(reference) is not str or reference not in available:
            raise ValueError("입력은 source.data 또는 앞 단계의 출력 포트를 참조해야 합니다.")

        expected = (ports[name].contract_id, ports[name].contract_version)

        if available[reference] != expected:
            raise ValueError("연결한 입출력 계약의 ID 또는 버전이 일치하지 않습니다.")

    return tuple(sorted(raw.items()))


def validate_pipeline(config, registry=None):
    """JSON 설정을 검증하며 원본 읽기·모듈 계산·승인 소비·worker 시작은 하지 않는다."""
    _object(config, ("schema_version", "pipeline_id", "input", "steps"))
    check_id(config["pipeline_id"])

    if config["schema_version"] != "module_pipeline_v1":
        raise ValueError("지원하지 않는 파이프라인 설정 버전입니다.")

    _object(config["input"], ("contract_id", "contract_version", "partition"))
    source = config["input"]

    if source["contract_id"] != "tabular_data" or type(source["contract_version"]) is not int or source["contract_version"] != 1:
        raise ValueError("초기 파이프라인은 tabular_data v1 입력만 지원합니다.")

    if source["partition"] not in ("unsplit", "train", "validation", "test"):
        raise ValueError("입력 구간은 unsplit·train·validation·test 중 하나여야 합니다.")

    steps = config["steps"]

    if type(steps) is not list or not 1 <= len(steps) <= MAX_STEPS:
        raise ValueError("파이프라인은 1~32개 순차 단계로 구성해야 합니다.")

    # 직접 전달한 dict도 파일 입력과 동일한 크기·JSON 값 제한을 적용한다.
    if len(_canonical(config).encode("utf-8")) > MAX_CONFIG_BYTES:
        raise ValueError("파이프라인 설정 크기는 64 KiB 이하여야 합니다.")

    registry = build_builtin_registry() if registry is None else registry
    available = {"source.data": ("tabular_data", 1)}
    seen = set()
    planned = []

    for step in steps:
        _object(step, ("step_id", "module_id", "version", "inputs"), ("settings", "depends_on"))
        step_id = step["step_id"]
        check_id(step_id)

        if step_id == "source" or step_id in seen:
            raise ValueError("단계 ID는 중복되거나 예약어 source일 수 없습니다.")

        module = registry.resolve(step["module_id"], step["version"])
        contract = module.contract

        # 알 수 없는 선행 증거를 '충족'으로 간주하지 않는다.
        if set(contract.preconditions) - RUNTIME_CHECKS:
            raise ValueError("현재 계획기가 지원하지 않는 선행 검사가 있습니다.")

        if contract.criteria_policy == "derive_on_train" and source["partition"] != "train":
            raise ValueError("전처리 기준은 Train 구간에서만 산출할 수 있습니다.")

        dependencies = step.get("depends_on", [])

        if type(dependencies) is not list or any(type(name) is not str for name in dependencies):
            raise ValueError("선행 단계 목록은 문자열 list여야 합니다.")

        if len(set(dependencies)) != len(dependencies) or not set(dependencies).issubset(seen):
            raise ValueError("선행 단계가 중복되거나 아직 정의되지 않았습니다.")

        bindings = _bindings(step["inputs"], contract, available)
        settings = step.get("settings", {})
        effective = module.validate_settings(settings)

        # 출력 연결도 실행 의존성이다. 명시 depends_on에 없어도 검증된 계획에 포함한다.
        inferred = {reference.split(".", 1)[0] for _, reference in bindings} - {"source"}
        dependencies = tuple(sorted(set(dependencies) | inferred))
        planned.append(PlannedStep(
            step_id, contract.module_id, contract.version, bindings, dependencies,
            _canonical(settings), _canonical(effective), _canonical(asdict(contract)),
        ))
        seen.add(step_id)

        for port in contract.outputs:
            available[f"{step_id}.{port.name}"] = (port.contract_id, port.contract_version)

    return PipelinePlan(config["pipeline_id"], source["partition"], tuple(planned))


def load_pipeline(path, registry=None):
    """크기·중복 키·비표준 숫자를 거부하고 설정 파일만 읽는다. 데이터 경로는 받지 않는다."""
    def unique_pairs(items):
        value = {}

        for key, item in items:
            if key in value:
                raise ValueError("파이프라인 JSON에 중복 키가 있습니다.")

            value[key] = item

        return value

    def invalid_number(value):
        raise ValueError("파이프라인 JSON에 비표준 숫자를 사용할 수 없습니다.")

    with Path(path).open("rb") as file:
        payload = file.read(MAX_CONFIG_BYTES + 1)

    if len(payload) > MAX_CONFIG_BYTES:
        raise ValueError("파이프라인 설정 크기는 64 KiB 이하여야 합니다.")

    try:
        config = json.loads(payload.decode("utf-8"), object_pairs_hook=unique_pairs, parse_constant=invalid_number)
    except (UnicodeError, RecursionError) as error:
        raise ValueError("파이프라인 JSON의 인코딩 또는 중첩이 올바르지 않습니다.") from error

    return validate_pipeline(config, registry)
