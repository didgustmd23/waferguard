"""변환 모듈을 두 승인 작업으로 계획한다. 데이터 읽기·승인·worker 실행은 하지 않는다."""

from dataclasses import asdict, dataclass
import hashlib
import json

from src.preprocessing.module_contracts import check_id
from src.preprocessing.pipeline import _canonical, _object
from src.preprocessing.preparation_modules import build_preparation_registry


@dataclass(frozen=True)
class PreparationModulePlan:
    """설정만 담는 불변 계획. 실행 시 분할·후보·코드·자원 증거 결합은 별도로 필요하다."""

    pipeline_id: str
    rules_json: str
    contracts_json: str

    def to_dict(self):
        return {
            "schema_version": "preparation_module_plan_v1",
            "pipeline_id": self.pipeline_id,
            "rules": json.loads(self.rules_json),
            "contracts": json.loads(self.contracts_json),
            "steps": [
                {"module_id": "tabular_prepare_propose", "version": "1.0.0",
                 "stage": "derive_preparation", "cli": "src.agents.preparation_candidate_cli",
                 "partitions": ["train"], "approval": "new_explicit_approval"},
                {"module_id": "tabular_prepare_apply", "version": "1.0.0",
                 "stage": "apply_preparation", "cli": "src.agents.preparation_apply_cli",
                 "partitions": ["train", "validation"], "approval": "new_explicit_approval",
                 "requires": "committed_candidate_and_user_review"},
            ],
            "automatic_execution": False,
            "test_allowed": False,
        }

    @property
    def signature(self):
        return hashlib.sha256(_canonical(self.to_dict()).encode("utf-8")).hexdigest()


def validate_preparation_modules(config):
    """고정된 2단계·명시 버전·후보 검토 경계를 검증한다. 경로·임의 실행 코드는 받지 않는다."""
    _object(config, ("schema_version", "pipeline_id", "steps"))
    check_id(config["pipeline_id"])

    if config["schema_version"] != "preparation_module_pipeline_v1":
        raise ValueError("지원하지 않는 변환 모듈 계획 버전입니다.")

    if len(_canonical(config).encode("utf-8")) > 64 * 1024:
        raise ValueError("변환 모듈 계획은 64 KiB 이하여야 합니다.")

    steps = config["steps"]

    if type(steps) is not list or len(steps) != 2:
        raise ValueError("후보 산출과 별도 승인 적용의 두 단계가 필요합니다.")

    registry = build_preparation_registry()
    normalized = []
    rules = None

    for step, identifier in zip(steps, ("tabular_prepare_propose", "tabular_prepare_apply")):
        _object(step, ("module_id", "version", "approval"), ("settings",))

        if step["module_id"] != identifier or step["approval"] != "new_explicit_approval":
            raise ValueError("변환 순서와 단계별 새 승인 경계를 유지해야 합니다.")

        module = registry.resolve(identifier, step["version"])
        effective = module.validate_settings(step.get("settings", {}))
        normalized.append(asdict(module.contract))

        if identifier == "tabular_prepare_propose":
            rules = asdict(effective["rules"])

    return PreparationModulePlan(config["pipeline_id"], _canonical(rules), _canonical(normalized))
