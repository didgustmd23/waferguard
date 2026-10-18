"""Train 증거·설정·자원 상한을 후보 기준 산출 요청에 결합한다."""

from dataclasses import asdict, dataclass
import hashlib
import json

from src.preprocessing.transform import PreparationRules
from .approvals import ApprovalScope
from .executor import ExecutionPlan, ExecutionStep
from .partition_input_contract import PartitionInputContract
from .snapshot_eda import _digest
from .split_evidence import read_metadata
from .state import LocalStateStore
from .worker_contract import WorkerLimits
from .artifacts import _root


POLICY_VERSION = "preparation_candidate_v1"


def saved_export_scope(directory):
    """완료 실행의 저장 계획에서 ingest 범위를 복원한다. DB 승인 발급·갱신은 하지 않는다."""
    root = _root(directory)
    saved = read_metadata(root / "execution_plan.json")
    selection = read_metadata(root / "selection.json")
    state = LocalStateStore(root).load()

    if set(saved) != {"run_id", "actor_id", "plan"}:
        raise ValueError("저장된 내보내기 실행 계획이 필요합니다.")

    value = dict(saved["plan"])
    value["steps"] = tuple(ExecutionStep(**step) for step in value["steps"])
    plan = ExecutionPlan(**value)

    if (state.status != "completed" or saved["run_id"] != state.run_id or saved["actor_id"] != state.actor_id
            or plan.policy_version != "partition_export_v1" or selection["plan_id"] != plan.plan_id
            or tuple((step.stage, step.action_id) for step in plan.steps) != (("preflight", "run_preflight"), ("ingest", "run_ingest"))):
        raise ValueError("완료·고정된 내보내기 실행 계획과 상태가 다릅니다.")

    step = plan.steps[-1]

    if _digest(json.loads(step.arguments_json)) != plan.config_id:
        raise ValueError("저장된 내보내기 인자와 설정 식별값이 다릅니다.")

    return ApprovalScope(state.actor_id, state.run_id, "ingest", "run_ingest", f"{state.run_id}:ingest",
                         hashlib.sha256(step.arguments_json.encode("utf-8")).hexdigest(),
                         plan.source_id, plan.config_id, plan.plan_id, plan.policy_version)


@dataclass(frozen=True)
class PreparationCandidateContract:
    """후보 산출만 허용한다. Validation 적용·전처리 완료·모델 학습 승인은 아니다."""

    train: PartitionInputContract
    rules: PreparationRules
    limits: WorkerLimits
    max_candidate_bytes: int = 4 * 1024 * 1024

    def __post_init__(self):
        if not isinstance(self.train, PartitionInputContract) or self.train.partition != "train":
            raise ValueError("승인·커밋된 Train 구간 계약만 후보 산출에 사용할 수 있습니다.")

        if not isinstance(self.rules, PreparationRules) or not isinstance(self.limits, WorkerLimits):
            raise TypeError("전처리 규칙과 자원 상한 계약이 필요합니다.")

        if type(self.max_candidate_bytes) is not int or not 1 <= self.max_candidate_bytes <= 8 * 1024 * 1024:
            raise ValueError("후보 파일 상한은 1~8 MiB 범위의 양의 정수 바이트여야 합니다.")

    def check(self):
        return self.train.check()

    def packet(self, code_id):
        if type(code_id) is not str or len(code_id) != 64 or any(char not in "0123456789abcdef" for char in code_id):
            raise ValueError("후보 산출 코드의 SHA-256 식별값이 필요합니다.")

        self.check()
        return {
            "version": POLICY_VERSION, "train_contract_id": self.train.contract_id,
            "split_evidence_id": self.train.split.signature,
            "export_marker_sha256": self.train.export.marker_sha256,
            "snapshot": asdict(self.train.snapshot), "roles": asdict(self.train.roles),
            "train_rows": len(self.train.row_positions), "rules": asdict(self.rules),
            "limits": asdict(self.limits), "max_candidate_bytes": self.max_candidate_bytes,
            "code_id": code_id,
        }
