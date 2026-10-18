"""snapshot·프로파일·자원 요청의 로컬 EDA 계약. 실행·승인 발급은 하지 않는다."""

from dataclasses import asdict, dataclass
import hashlib
import json

from src.preprocessing.profiles import DatasetProfile, profile_from_dict
from .artifacts import _root
from .executor import ExecutionPlan, ExecutionStep
from .input_snapshot import Snapshot, verify_snapshot
from .worker_contract import WorkerLimits, _decode


CONTRACT_VERSION = "snapshot_eda_contract_v1"
REGISTERED_EDA_ADAPTERS = frozenset({"tabular_csv_v1"})


def _canonical(value):
    """같은 내용이 같은 식별값이 되도록 JSON을 정규화한다."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


@dataclass(frozen=True)
class EdaInputContract:
    """로컬 전용 계약. 데이터 읽기 승인이나 LLM 전달 객체가 아니다."""

    snapshot_directory: str
    snapshot: Snapshot
    profile_json: str
    limits: WorkerLimits
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self):
        if self.schema_version != CONTRACT_VERSION:
            raise ValueError("지원하지 않는 EDA 입력 계약 버전입니다.")

        if not isinstance(self.snapshot, Snapshot) or not isinstance(self.limits, WorkerLimits):
            raise TypeError("snapshot·worker 자원 계약이 필요합니다.")

        if not isinstance(self.snapshot_directory, str) or not self.snapshot_directory:
            raise ValueError("로컬 snapshot 경로가 필요합니다.")

        # 프로파일의 경로·모듈·임의 인자를 허용하지 않는 기존 파서를 재사용한다.
        profile = profile_from_dict(_decode(self.profile_json))

        if profile.adapter_id not in REGISTERED_EDA_ADAPTERS:
            raise ValueError("격리 EDA 연결 대상으로 등록되지 않은 어댑터입니다.")

        object.__setattr__(self, "profile_json", _canonical(profile.to_dict()))

    @property
    def profile(self):
        """검증된 설정을 새 객체로 반환하여 외부 변경을 계약에 반영하지 않는다."""
        return profile_from_dict(_decode(self.profile_json))

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "snapshot_directory": self.snapshot_directory,
            "snapshot": asdict(self.snapshot),
            "profile": json.loads(self.profile_json),
            "limits": asdict(self.limits),
        }

    def to_json(self):
        """컬럼 역할·경로를 포함하므로 로컬 저장/IPC에만 사용한다."""
        payload = _canonical(self.to_dict())
        _decode(payload)
        return payload

    @property
    def contract_id(self):
        """입력 크기·상한·경로·프로파일·자원 변경을 식별한다. 인증 서명이 아니다."""
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_json(cls, payload):
        value = _decode(payload)

        if set(value) != {"schema_version", "snapshot_directory", "snapshot", "profile", "limits"}:
            raise ValueError("EDA 입력 계약 필드가 올바르지 않습니다.")

        for key, contract_type in (("snapshot", Snapshot), ("limits", WorkerLimits)):
            if not isinstance(value[key], dict) or set(value[key]) != set(contract_type.__dataclass_fields__):
                raise ValueError("EDA 입력의 크기·자원 계약 필드가 올바르지 않습니다.")

        return cls(
            value["snapshot_directory"], Snapshot(**value["snapshot"]),
            _canonical(value["profile"]), WorkerLimits(**value["limits"]), value["schema_version"],
        )

    def check_input(self):
        """실행 직전 재호출할 복사본 검사. CSV 내용 해석이나 ACL 부여는 하지 않는다."""
        root = _root(self.snapshot_directory)

        if str(root) != self.snapshot_directory or not verify_snapshot(root, self.snapshot):
            raise ValueError("EDA 입력 복사본이 계약과 일치하지 않습니다.")

        return True

    def execution_plan(self):
        """기존 승인 scope가 입력·설정 변경을 구분하도록 EDA 전용 계획을 만든다."""
        self.check_input()
        return ExecutionPlan(
            source_id=self.snapshot.source_id, config_id=self.contract_id,
            policy_version=CONTRACT_VERSION,
            steps=(
                ExecutionStep("preflight", "run_preflight"),
                ExecutionStep("run_eda", "run_eda", self.to_json()),
            ),
        )


def bind_eda_input(directory, snapshot, profile, limits):
    """호출자가 보존한 기대 snapshot과 검증된 프로파일을 결합한다."""
    if not isinstance(profile, DatasetProfile):
        raise TypeError("검증된 DatasetProfile이 필요합니다.")

    contract = EdaInputContract(str(_root(directory)), snapshot, _canonical(profile.to_dict()), limits)
    contract.check_input()
    return contract
