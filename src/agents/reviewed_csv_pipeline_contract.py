"""검토된 CSV 모듈 계획을 snapshot·profile 계약에 결합하는 정적 바인더."""

from dataclasses import asdict, dataclass
import hashlib
import json
from dataclasses import replace

from src.preprocessing.metadata_registry import (
    SCHEMA_SHA256,
    build_metadata_registry,
)
from src.preprocessing.pipeline import PipelinePlan, validate_pipeline
from src.preprocessing.profiles import DatasetProfile
from src.preprocessing.quality import QualityRules
from .module_pipeline_contract import MODULE_FILES
from .eda_input_contract import EdaInputContract, bind_eda_input
from .executor import ExecutionPlan, ExecutionStep, _digest
from .input_snapshot import Snapshot
from .approvals import ApprovalScope
from .worker_contract import WorkerLimits
from .module_catalogue import MODULE_ID, VERSION, inspect_module


POLICY_VERSION = "reviewed_pipeline_binding_v1"
QUALITY_MODULE_ID = "tabular_quality"
QUALITY_MODULE_VERSION = "1.0.0"


def _canonical(value):
    """계약 식별자에 사용할 정렬된 JSON을 만든다."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


@dataclass(frozen=True)
class ReviewedCsvPipelineBinding:
    """입력·프로파일·정규화 계획을 함께 고정한 실행 전 계약."""

    input_contract: EdaInputContract
    plan: PipelinePlan
    pipeline_config_json: str
    execution_code_id: str
    settings_schema_sha256: str
    policy_version: str = POLICY_VERSION

    @property
    def adapter_id(self):
        """이 바인더가 처리하는 등록 입력 어댑터 ID를 반환한다."""
        return self.input_contract.profile.adapter_id

    @property
    def source_id(self):
        """공통 승인 계층에서 사용하는 불투명 입력 출처 식별자."""
        return self.input_contract.snapshot.source_id

    def approval_summary(self):
        """사용자 승인 화면에 필요한 형식별 세부값을 JSON 호환 형태로 반환한다."""
        return {
            "adapter_id": self.adapter_id,
            "source_id": self.source_id,
            "snapshot_bytes": self.input_contract.snapshot.size_bytes,
            "snapshot_sha256": self.input_contract.snapshot.sha256,
            "roles": self.input_contract.profile.to_dict()["roles"],
            "quality_rules": asdict(self.input_contract.profile.quality_rules),
            "modules": [
                {
                    "step_id": step.step_id,
                    "module_id": step.module_id,
                    "version": step.version,
                    "settings": json.loads(step.effective_settings_json),
                }
                for step in self.plan.steps
            ],
            "settings_schema_sha256": self.settings_schema_sha256,
            "pipeline_signature": self.plan.signature,
            "execution_code_id": self.execution_code_id,
            "limits": asdict(self.input_contract.limits),
        }

    def to_dict(self):
        """저장·승인 단계가 사용할 데이터 전용 계약 표현을 반환한다."""
        return {
            "schema_version": self.policy_version,
            "input_contract": self.input_contract.to_dict(),
            "pipeline_plan": self.plan.to_dict(),
            "pipeline_config": json.loads(self.pipeline_config_json),
            "pipeline_signature": self.plan.signature,
            "execution_code_id": self.execution_code_id,
            "settings_schema_sha256": self.settings_schema_sha256,
        }

    @property
    def binding_id(self):
        """스냅샷·프로파일·계획·검토 메타데이터의 조합 식별자."""
        return hashlib.sha256(_canonical(self.to_dict()).encode("utf-8")).hexdigest()

    def execution_plan(self):
        """RulesExecutor가 승인 범위를 만들 때 사용할 고정 계획을 구성한다."""
        arguments = {
            "binding_id": self.binding_id,
            "binding": self.to_dict(),
        }
        return ExecutionPlan(
            source_id=self.input_contract.snapshot.source_id,
            config_id=self.binding_id,
            policy_version=self.policy_version,
            steps=(
                ExecutionStep("preflight", "run_preflight", _canonical(arguments)),
                ExecutionStep("run_pipeline", "run_pipeline", _canonical(arguments)),
            ),
        )

    def approval_scope(self, actor_id, run_id):
        """실행 계획의 run_pipeline 단계에 대한 1회 승인 범위를 계산한다."""
        plan = self.execution_plan()
        step = next(item for item in plan.steps if item.stage == "run_pipeline")

        return ApprovalScope(
            actor_id=actor_id,
            run_id=run_id,
            stage=step.stage,
            tool_id=step.action_id,
            target_id=f"{run_id}:{step.stage}",
            arguments_id=_digest(step.arguments_json),
            source_id=plan.source_id,
            config_id=plan.config_id,
            plan_id=plan.plan_id,
            policy_version=plan.policy_version,
        )


def _effective_quality_profile(config, profile):
    """프로파일 규칙을 기본으로 두고 설정에 명시된 품질 규칙만 덮어쓴다."""
    steps = config.get("steps")

    if type(steps) is list and any(
        type(step) is not dict or step.get("module_id") not in MODULE_FILES
        for step in steps
    ):
        raise ValueError("현재 격리 worker에는 등록된 기본 분석 모듈만 연결할 수 있습니다.")

    if type(steps) is not list or not steps:
        raise ValueError("CSV 모듈 계획에 품질 검사 단계가 필요합니다.")

    quality = steps[0]

    if (type(quality) is not dict
            or quality.get("module_id") != QUALITY_MODULE_ID
            or quality.get("version") != QUALITY_MODULE_VERSION):
        raise ValueError("첫 단계는 고정 버전 tabular_quality 검사여야 합니다.")

    if type(quality.get("step_id")) is not str:
        raise ValueError("품질 검사 단계 ID가 필요합니다.")

    settings = quality.get("settings", {})

    if type(settings) is not dict:
        raise ValueError("품질 단계는 rules 설정만 받을 수 있습니다.")

    overrides = settings.get("rules", settings)

    if type(overrides) is not dict:
        raise ValueError("품질 규칙 설정은 JSON 객체여야 합니다.")

    effective = asdict(profile.quality_rules)

    if set(overrides) - set(effective):
        raise ValueError("프로파일에 정의되지 않은 품질 규칙이 포함되어 있습니다.")

    effective.update(overrides)

    if isinstance(effective.get("allowed_labels"), list):
        effective["allowed_labels"] = tuple(effective["allowed_labels"])

    return replace(profile, quality_rules=QualityRules(**effective))


def _bind_quality_settings(config, effective_profile):
    """계획에도 실제 적용할 완전한 품질 규칙을 기록한다."""
    bound = json.loads(_canonical(config))
    # tabular_quality 모듈은 QualityRules 필드를 최상위 설정으로 받는다.
    # 사용자 계획 입력의 rules 래퍼는 바인딩 단계에서 해제한다.
    bound["steps"][0]["settings"] = asdict(effective_profile.quality_rules)
    return json.loads(_canonical(bound))


def bind_reviewed_csv_pipeline(
    config,
    snapshot_directory,
    snapshot: Snapshot,
    profile: DatasetProfile,
    limits: WorkerLimits,
    *,
    project_root=None,
):
    """검토된 CSV 분석 계획을 입력 계약과 결합하되 worker는 시작하지 않는다."""
    if not isinstance(profile, DatasetProfile):
        raise TypeError("검증된 DatasetProfile이 필요합니다.")

    if profile.adapter_id != "tabular_csv_v1":
        raise ValueError("현재 사용자 분석 계약은 tabular_csv_v1만 지원합니다.")

    if type(config) is not dict or type(config.get("input")) is not dict:
        raise ValueError("검증된 파이프라인 JSON 객체가 필요합니다.")

    # partition 문자열만으로 Train·Validation·Test 이력을 증명할 수 없어
    # 신뢰된 분할 계약이 연결되기 전까지는 미분할 입력만 결합한다.
    if config["input"].get("partition") != "unsplit":
        raise ValueError("분할 증거 계약이 연결되지 않아 unsplit 입력만 사용할 수 있습니다.")

    steps = config.get("steps")

    if type(steps) is not list or not steps:
        raise ValueError("CSV 모듈 계획에 단계가 필요합니다.")

    quality_positions = [
        index for index, step in enumerate(steps)
        if type(step) is dict and step.get("module_id") == QUALITY_MODULE_ID
    ]

    if quality_positions != [0]:
        raise ValueError("품질 검사는 계획의 첫 단계에 정확히 한 번 있어야 합니다.")

    quality_step_id = steps[0]["step_id"]

    for index, step in enumerate(steps):
        if type(step) is not dict or step.get("inputs") != {"data": "source.data"}:
            raise ValueError("각 단계는 검증된 원본 CSV의 source.data만 입력으로 사용해야 합니다.")

        if index > 0:
            dependencies = step.get("depends_on", [])

            if type(dependencies) is not list or quality_step_id not in dependencies:
                raise ValueError("모든 분석 단계는 품질 검사 단계의 통과에 의존해야 합니다.")

    effective_profile = _effective_quality_profile(config, profile)
    bound_config = _bind_quality_settings(config, effective_profile)
    # 사용자 명세·코드 해시 검사는 권한을 소유한 에이전트 조립 계층에서 수행한다.
    # 공통 등록부에는 검토된 불변 계약만 주입하여 코어→에이전트 역참조를 막는다.
    contract = inspect_module(MODULE_ID, VERSION, project_root=project_root)
    registry = build_metadata_registry(project_root, reviewed_contract=contract)
    plan = validate_pipeline(bound_config, registry)

    input_contract = bind_eda_input(
        snapshot_directory,
        snapshot,
        effective_profile,
        limits,
    )

    # 승인 대상 코드와 격리 worker 실행 코드를 함께 고정한다.
    from .module_pipeline_transport import code_manifest
    from .snapshot_eda import _digest
    code_id = _digest(code_manifest())

    return ReviewedCsvPipelineBinding(
        input_contract=input_contract,
        plan=plan,
        pipeline_config_json=_canonical(bound_config),
        execution_code_id=code_id,
        settings_schema_sha256=SCHEMA_SHA256,
    )
