"""완료 후보·같은 분할의 두 구간·정확한 상태 해시를 적용 승인에 결합한다."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from src.preprocessing.profiles import profile_from_dict
from src.preprocessing.transform import PreparationRules
from .approvals import ApprovalScope
from .artifacts import _root, _fingerprint
from .executor import ExecutionPlan, ExecutionStep
from .partition_input_contract import bind_partition_input
from .preparation_candidate_contract import PreparationCandidateContract
from .preparation_candidate_results import read_candidate
from .snapshot_eda import _digest
from .split_evidence import CommittedArtifact, read_confirmed_split, read_committed_artifact, read_metadata
from .state import LocalStateStore
from .worker_contract import WorkerLimits


POLICY_VERSION = "preparation_apply_v1"


@dataclass(frozen=True)
class CandidateEvidence:
    """과거 승인·커밋된 후보 상태의 정확한 식별값. 현재 코드로 후보를 재산출하지 않는다."""

    artifact: CommittedArtifact
    derivation: PreparationCandidateContract
    code_id: str
    state_sha256: str
    output_columns_json: str

    def __post_init__(self):
        if not isinstance(self.artifact, CommittedArtifact) or not isinstance(self.derivation, PreparationCandidateContract):
            raise TypeError("완료 후보 산출물과 기준 산출 계약이 필요합니다.")

    @property
    def directory(self):
        return self.artifact.directory / "candidate"

    @property
    def output_columns(self):
        return json.loads(self.output_columns_json)

    @property
    def signature(self):
        return _digest({"marker": self.artifact.marker_sha256, "state": self.state_sha256,
                        "train": self.derivation.train.contract_id, "columns": self.output_columns,
                        "code_id": self.code_id})

    def check(self):
        if read_candidate_evidence(self.artifact.run_directory, self.artifact.scope.actor_id) != self:
            raise ValueError("완료된 후보 또는 선행 구간 증거가 변경되었습니다.")

        return True


def read_candidate_evidence(run_directory, actor):
    """저장 계획·선택·승인 소비·커밋·후보 파일을 대조한다. 승인 DB는 읽기 전용이다."""
    root = _root(run_directory)
    saved = read_metadata(root / "execution_plan.json")
    record = read_metadata(root / "selection.json")
    state = LocalStateStore(root).load()
    fields = {"split_run", "split_scope", "export_run", "export_scope", "train_contract_id", "rules", "limits", "max_candidate_bytes", "code_id", "plan_id"}

    if set(record) != fields or set(saved) != {"run_id", "actor_id", "plan"} or state.actor_id != actor or state.status != "completed":
        raise ValueError("같은 OS 계정이 완료한 후보 실행 기록이 필요합니다.")

    value = dict(saved["plan"])
    value["steps"] = tuple(ExecutionStep(**step) for step in value["steps"])
    plan = ExecutionPlan(**value)
    split_scope = ApprovalScope(**record["split_scope"])
    export_scope = ApprovalScope(**record["export_scope"])

    if split_scope.actor_id != actor or export_scope.actor_id != actor:
        raise ValueError("후보의 선행 분할·내보내기 승인 주체가 다릅니다.")

    split = read_confirmed_split(record["split_run"], split_scope)
    export = read_committed_artifact(record["export_run"], export_scope)
    profile = profile_from_dict(read_metadata(Path(record["split_run"]) / "profile.json"))

    if profile.signature != split_scope.config_id:
        raise ValueError("선행 프로파일이 확정 분할과 다릅니다.")

    train = bind_partition_input(split, export, "train", profile.roles)
    contract = PreparationCandidateContract(train, PreparationRules(**record["rules"]), WorkerLimits(**record["limits"]), record["max_candidate_bytes"])
    packet = json.loads(json.dumps(contract.packet(record["code_id"])))

    if (saved["run_id"] != state.run_id or saved["actor_id"] != actor or record["train_contract_id"] != train.contract_id
            or plan.policy_version != "preparation_candidate_v1" or state.policy_version != plan.policy_version
            or state.current_stage != "derive_preparation" or record["plan_id"] != plan.plan_id
            or plan.source_id != train.snapshot.source_id or plan.config_id != _digest(packet)
            or tuple((step.stage, step.action_id) for step in plan.steps) != (("preflight", "run_preflight"), ("derive_preparation", "derive_preparation"))
            or json.loads(plan.steps[-1].arguments_json) != packet):
        raise ValueError("후보의 저장 계획·설정·Train 계약이 서로 다릅니다.")

    scope = ApprovalScope(actor, state.run_id, "derive_preparation", "derive_preparation", f"{state.run_id}:derive_preparation",
                          hashlib.sha256(plan.steps[-1].arguments_json.encode("utf-8")).hexdigest(),
                          plan.source_id, plan.config_id, plan.plan_id, plan.policy_version)
    artifact = read_committed_artifact(root, scope)

    if {path.name for path in artifact.directory.iterdir()} != {"candidate", "AGENT_COMPLETE.json"} or read_candidate(artifact.directory / "candidate", contract, record["code_id"]) is None:
        raise ValueError("승인·커밋된 후보 파일의 정합성이 일치하지 않습니다.")

    payload = read_metadata(artifact.directory / "candidate/preprocessor.json")
    artifact.check()
    return CandidateEvidence(artifact, contract, record["code_id"],
                             _fingerprint(artifact.directory / "candidate/preprocessor.json")["sha256"],
                             json.dumps(payload["output_columns"], ensure_ascii=False))


@dataclass(frozen=True)
class PreparationApplyContract:
    """동일 후보를 Train·Validation에만 적용한다. 설정 변경·재산출·Test 적용은 허용하지 않는다."""

    candidate: CandidateEvidence
    limits: WorkerLimits
    max_partition_bytes: int = 32 * 1024 * 1024

    def __post_init__(self):
        if not isinstance(self.candidate, CandidateEvidence) or not isinstance(self.limits, WorkerLimits):
            raise TypeError("완료된 후보 증거와 자원 계약이 필요합니다.")

        if type(self.max_partition_bytes) is not int or not 1 <= self.max_partition_bytes <= 1024 * 1024 * 1024:
            raise ValueError("적용 출력 파일 상한은 1 바이트~1 GiB 범위여야 합니다.")

    @property
    def train(self):
        return self.candidate.derivation.train

    @property
    def validation(self):
        return bind_partition_input(self.train.split, self.train.export, "validation", self.train.roles)

    def check(self):
        self.candidate.check()
        self.validation.check()
        return True

    def positions(self):
        return {name: list(part.row_positions) for name, part in (("train", self.train), ("validation", self.validation))}

    def packet(self, code_id):
        if type(code_id) is not str or len(code_id) != 64 or any(char not in "0123456789abcdef" for char in code_id):
            raise ValueError("적용 코드의 SHA-256 식별값이 필요합니다.")

        self.check()
        partitions = {name: {"contract_id": part.contract_id, "snapshot": asdict(part.snapshot),
                             "rows": len(part.row_positions), "positions_sha256": _digest(list(part.row_positions))}
                      for name, part in (("train", self.train), ("validation", self.validation))}
        return {"version": POLICY_VERSION, "candidate_id": self.candidate.signature,
                "preprocessor_sha256": self.candidate.state_sha256,
                "max_state_bytes": self.candidate.derivation.max_candidate_bytes,
                "roles": asdict(self.train.roles), "rules": asdict(self.candidate.derivation.rules),
                "output_columns_sha256": _digest(self.candidate.output_columns),
                "output_feature_count": len(self.candidate.output_columns), "partitions": partitions,
                "limits": asdict(self.limits), "max_partition_bytes": self.max_partition_bytes, "code_id": code_id}
