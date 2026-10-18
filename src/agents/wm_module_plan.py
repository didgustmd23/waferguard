"""WM 승인 작업의 포트·순서·버전을 검증한다. 실제 데이터나 승인 저장소는 읽지 않는다."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from src.preprocessing.module_contracts import check_id
from src.preprocessing.pipeline import PlannedStep, _bindings, _canonical, _object
from .agent_module_registry import build_agent_registry
from .wm_module_catalogue import WmWorkflowModule
from .worker_contract import _decode


MAX_CONFIG_BYTES = 64 * 1024
SOURCE_PARTITIONS = {
    "wm_canonical": "unsplit",
    "wm_quality_review": "unsplit",
    "wm_conflict_candidate": "unsplit",
    "wm_eda_evidence": "unsplit",
    "wm_split_proposal": "unsplit",
    "wm_confirmed_split": "development",
    "wm_transform_evidence": "development",
    "wm_development_cache": "development",
}


@dataclass(frozen=True)
class WmModulePlan:
    """정적 계획만 보관한다. 계획 서명은 데이터 검증·승인·실행 완료의 증거가 아니다."""

    pipeline_id: str
    input_json: str
    steps: tuple[PlannedStep, ...]
    routes_json: str

    def to_dict(self):
        return {
            "schema_version": "wm_module_plan_v1",
            "pipeline_id": self.pipeline_id,
            "input": json.loads(self.input_json),
            "steps": [asdict(step) for step in self.steps],
            "execution_routes": json.loads(self.routes_json),
            "plan_only": True,
            "automatic_execution": False,
            "stop_on_review_or_block": True,
            "test_maps_allowed": False,
            "model_training_allowed": False,
        }

    @property
    def signature(self):
        return hashlib.sha256(_canonical(self.to_dict()).encode("utf-8")).hexdigest()


def validate_wm_modules(config):
    """기존 입력 연결 검증을 재사용하고 WM 전용 증거 계약과 승인 경계를 확인한다."""
    _object(config, ("schema_version", "pipeline_id", "input", "steps"))
    check_id(config["pipeline_id"])

    if config["schema_version"] != "wm_module_pipeline_v1":
        raise ValueError("지원하지 않는 WM 모듈 계획 버전입니다.")

    if len(_canonical(config).encode("utf-8")) > MAX_CONFIG_BYTES:
        raise ValueError("WM 모듈 계획은 64 KiB 이하여야 합니다.")

    source = config["input"]
    _object(source, ("contract_id", "contract_version", "partition"))

    if (type(source["contract_id"]) is not str or source["contract_id"] not in SOURCE_PARTITIONS
            or type(source["contract_version"]) is not int or source["contract_version"] != 1
            or source["partition"] != SOURCE_PARTITIONS[source["contract_id"]]):
        raise ValueError("지원하는 WM 증거 계약 v1과 해당 구간을 지정하세요. Test 맵은 지원하지 않습니다.")

    steps = config["steps"]

    if type(steps) is not list or not 1 <= len(steps) <= 32:
        raise ValueError("WM 계획은 1~32개 순차 단계로 구성해야 합니다.")

    registry = build_agent_registry()
    available = {"source.evidence": (source["contract_id"], 1)}
    seen, planned, routes = set(), [], []

    for step in steps:
        _object(step, ("step_id", "module_id", "version", "inputs"), ("settings", "depends_on"))
        identifier = step["step_id"]
        check_id(identifier)

        if identifier == "source" or identifier in seen:
            raise ValueError("단계 ID는 중복되거나 예약어 source일 수 없습니다.")

        module = registry.resolve(step["module_id"], step["version"])

        if not isinstance(module, WmWorkflowModule):
            raise ValueError("표형 계산 모듈은 WM 증거 계획에 연결할 수 없습니다.")

        dependencies = step.get("depends_on", [])

        if (type(dependencies) is not list or any(type(item) is not str for item in dependencies)
                or len(set(dependencies)) != len(dependencies) or not set(dependencies).issubset(seen)):
            raise ValueError("선행 단계가 중복되거나 아직 정의되지 않았습니다.")

        # 공통 포트 검사로 누락·미래 참조·다른 계약 연결을 거부한다.
        bindings = _bindings(step["inputs"], module.contract, available)
        settings = step.get("settings", {})
        effective = module.validate_settings(settings)
        inferred = {reference.split(".", 1)[0] for _, reference in bindings} - {"source"}
        planned.append(PlannedStep(
            identifier, module.contract.module_id, module.contract.version,
            bindings, tuple(sorted(set(dependencies) | inferred)),
            _canonical(settings), _canonical(effective), _canonical(asdict(module.contract)),
        ))
        routes.append({
            "step_id": identifier, "cli": "src.agents." + module.cli_id,
            "worker_stage": module.worker_stage, "review_stages": list(module.review_stages),
            "approval": "existing_cli_new_stage_approval",
            "runtime_evidence_required": True,
        })
        seen.add(identifier)

        for port in module.contract.outputs:
            available[f"{identifier}.{port.name}"] = (port.contract_id, port.contract_version)

    return WmModulePlan(config["pipeline_id"], _canonical(source), tuple(planned), _canonical(routes))


def load_wm_modules(path):
    """설정만 크기 상한 안에서 읽는다. 중복 키·비표준 숫자는 기존 디코더로 거부한다."""
    with Path(path).open("rb") as stream:
        payload = stream.read(MAX_CONFIG_BYTES + 1)

    if len(payload) > MAX_CONFIG_BYTES:
        raise ValueError("WM 모듈 계획은 64 KiB 이하여야 합니다.")

    try:
        return validate_wm_modules(_decode(payload.decode("utf-8")))
    except (UnicodeError, RecursionError) as error:
        raise ValueError("WM 계획의 UTF-8 인코딩 또는 중첩이 올바르지 않습니다.") from error
