"""v2 표형 출처·프로파일·구간을 기존 승인 소비와 커밋 증거에 결합한다."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re

from src.preprocessing.contracts import ColumnRoles
from src.preprocessing.profiles import profile_from_dict
from src.preprocessing.tabular_bundle import (
    BundleRef, inspect_bundle, iter_bundle_rows, read_bundle_schema, read_bundle_provenance,
)
from src.preprocessing.tabular_loaders import CsvReadSpec
from .artifacts import _root
from .input_snapshot import Snapshot, verify_snapshot
from .split_evidence import CommittedArtifact, ConfirmedSplit, read_metadata
from .strict_json import loads
from .worker_contract import WorkerLimits


POLICY_VERSION = "tabular_input_v2"
ADAPTER_ID = "tabular_canonical_v2"
MAX_CONTRACT_BYTES = 1024 * 1024


def _require(condition):
    if not condition:
        raise ValueError("표형 출처·승인 또는 계보 계약이 일치하지 않습니다.")


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _digest(value):
    # 기존 split/evidence와 같은 JSON 직렬화 규칙으로 단계 간 해시를 대조한다.
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _hash(value):
    _require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None)


def _dict(value, cls):
    _require(type(value) is dict and set(value) == set(cls.__dataclass_fields__))
    return value


def _spec(payload):
    return CsvReadSpec.from_json(payload).to_dict()


def materialization_config_id(snapshot, read_spec_json, inspection_id, limits, execution_code_id, *, inspection_sha256, max_output_bytes):
    """생성 승인이 고정해야 할 입력·해석·검사 증거·코드·자원 서명을 산출한다."""
    _require(type(snapshot) is Snapshot and type(limits) is WorkerLimits)
    _hash(inspection_id)
    _hash(execution_code_id)
    from .tabular_input_worker import build_packet
    return _digest(build_packet("materialize", asdict(snapshot), _spec(read_spec_json), asdict(limits), execution_code_id,
                                inspection_id=inspection_id, inspection_sha256=inspection_sha256, max_output_bytes=max_output_bytes))


@dataclass(frozen=True)
class TabularSourceContract:
    """파일 위치가 아닌 승인한 해석과 내용·생성 증거를 기준으로 표를 식별한다."""

    snapshot_directory: str
    snapshot: Snapshot
    bundle_directory: str
    bundle: BundleRef
    read_spec_json: str
    inspection_id: str
    materialization: CommittedArtifact
    limits: WorkerLimits
    execution_code_id: str
    inspection_sha256: str
    max_output_bytes: int
    loader_id: str = "delimited_v1"
    loader_version: str = "1.0.0"
    policy_version: str = POLICY_VERSION
    schema_version: str = "tabular_source_v2"

    def __post_init__(self):
        _require(type(self.snapshot) is Snapshot and type(self.bundle) is BundleRef)
        _require(type(self.materialization) is CommittedArtifact and type(self.limits) is WorkerLimits)
        for value in (self.snapshot_directory, self.bundle_directory):
            _require(type(value) is str and Path(value).is_absolute())
        _require((self.loader_id, self.loader_version, self.policy_version, self.schema_version)
                 == ("delimited_v1", "1.0.0", POLICY_VERSION, "tabular_source_v2"))
        self.materialization_config_id

    @property
    def materialization_config_id(self):
        """묶음 생성 전 승인 인자와 대조할 설정 식별값이다."""
        return materialization_config_id(self.snapshot, self.read_spec_json, self.inspection_id,
                                         self.limits, self.execution_code_id, inspection_sha256=self.inspection_sha256,
                                         max_output_bytes=self.max_output_bytes)

    @property
    def source_id(self):
        """동일 바이트라도 해석/생성 결과가 다르면 다른 입력으로 식별한다."""
        return "tabular_source:" + _digest(self.packet())

    def packet(self):
        """경로·승인 DB를 제외한 worker용 고정 내용 식별 정보를 반환한다."""
        return {"schema_version": self.schema_version, "snapshot": asdict(self.snapshot), "bundle": asdict(self.bundle),
                "read_spec": _spec(self.read_spec_json), "inspection_id": self.inspection_id,
                "inspection_sha256": self.inspection_sha256, "max_output_bytes": self.max_output_bytes,
                "loader_id": self.loader_id, "loader_version": self.loader_version, "policy_version": self.policy_version,
                "materialization_config_id": self.materialization_config_id}

    def check_input(self):
        """생성 승인 소비·커밋과 snapshot·묶음·해석·출처 연결을 매번 다시 확인한다."""
        self.materialization.check()
        scope = self.materialization.scope
        _require((scope.stage, scope.tool_id, scope.source_id, scope.config_id, scope.policy_version)
                 == ("ingest", "run_ingest", self.snapshot.source_id, self.materialization_config_id, POLICY_VERSION))
        root = _root(self.snapshot_directory)
        _require(str(root) == self.snapshot_directory and verify_snapshot(root, self.snapshot))
        directory = _root(self.bundle_directory)
        _require(directory == self.materialization.directory / "table")
        _require(self.bundle.size_bytes <= self.max_output_bytes)
        inspect_bundle(directory, expected=self.bundle, max_bytes=self.bundle.size_bytes)
        provenance = read_bundle_provenance(directory, expected=self.bundle)
        expected = {"source_snapshot_sha256": self.snapshot.sha256, "source_snapshot_size_bytes": self.snapshot.size_bytes,
                    "loader_id": self.loader_id, "loader_version": self.loader_version,
                    "read_spec": _spec(self.read_spec_json), "inspection_id": self.inspection_id,
                    "inspection_sha256": self.inspection_sha256, "policy_version": self.policy_version}
        _require(all(provenance.get(key) == value for key, value in expected.items()))
        return True

    def to_dict(self):
        """로컬 저장용 계약이며 반환값은 모델 전송 자료가 아니다."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        """정적 계약 타입만 복원하며 복원 자체는 승인·실행을 하지 않는다."""
        data = dict(_dict(value, cls))
        for key, item_type in (("snapshot", Snapshot), ("limits", WorkerLimits), ("materialization", CommittedArtifact)):
            data[key] = item_type(**_dict(data[key], item_type))
        data["bundle"] = BundleRef.from_dict(data["bundle"])
        return cls(**data)

    def to_json(self):
        """중복 키를 만들지 않는 정규 JSON으로 로컬 계약을 직렬화한다."""
        return _json(self.to_dict())

    @classmethod
    def from_json(cls, payload):
        """중복 키·비표준 숫자·크기 초과를 거부한 뒤 계약을 복원한다."""
        return cls.from_dict(loads(payload, MAX_CONTRACT_BYTES))


@dataclass(frozen=True)
class TabularEdaInputContract:
    """일반 EDA/분할의 필수 프로파일을 v2 출처와 결합한다."""

    source: TabularSourceContract
    profile_json: str
    limits: WorkerLimits
    schema_version: str = "tabular_eda_input_v2"

    def __post_init__(self):
        _require(type(self.source) is TabularSourceContract and type(self.limits) is WorkerLimits)
        _require(self.schema_version == "tabular_eda_input_v2" and self.profile.adapter_id == ADAPTER_ID)
        _require(all(getattr(self.limits, key) <= getattr(self.source.limits, key) for key in WorkerLimits.__dataclass_fields__))

    @property
    def profile(self):
        """제한된 JSON에서 역할·품질·분할 설정을 복원한다."""
        return profile_from_dict(loads(self.profile_json, MAX_CONTRACT_BYTES))

    @property
    def source_id(self):
        return self.source.source_id

    @property
    def contract_id(self):
        return _digest({"source": self.source.packet(), "profile": self.profile.to_dict(), "limits": asdict(self.limits),
                        "schema_version": self.schema_version})

    def check_input(self):
        """필수 컬럼이 실제 승인된 스키마에 있는지 검증한다. 역할을 자동 변경하지 않는다."""
        self.source.check_input()
        schema = read_bundle_schema(self.source.bundle_directory, expected=self.source.bundle)
        _require(set(self.profile.roles.required_columns).issubset(column["name"] for column in schema["columns"]))
        return True

    def to_dict(self):
        """로컬 계약 저장용 값을 반환한다."""
        return {"source": self.source.to_dict(), "profile_json": self.profile_json,
                "limits": asdict(self.limits), "schema_version": self.schema_version}

    @classmethod
    def from_dict(cls, value):
        """허용된 v2 출처와 자원 계약만 복원한다."""
        data = dict(_dict(value, cls))
        data["source"] = TabularSourceContract.from_dict(data["source"])
        data["limits"] = WorkerLimits(**_dict(data["limits"], WorkerLimits))
        return cls(**data)


@dataclass(frozen=True)
class TabularPartitionInputContract:
    """확정 분할·커밋된 export·원래 행 위치에 묶인 Train/Validation 표다."""

    split: ConfirmedSplit
    export: CommittedArtifact
    partition: str
    bundle: BundleRef
    roles_json: str
    parent_source_id: str
    schema_version: str = "tabular_partition_input_v2"

    def __post_init__(self):
        _require(type(self.split) is ConfirmedSplit and type(self.export) is CommittedArtifact and type(self.bundle) is BundleRef)
        _require(self.partition in ("train", "validation") and self.schema_version == "tabular_partition_input_v2")
        _require(type(self.parent_source_id) is str and re.fullmatch(r"tabular_source:[0-9a-f]{64}", self.parent_source_id) is not None)
        self.roles

    @property
    def roles(self):
        value = loads(self.roles_json, MAX_CONTRACT_BYTES)
        _require(set(value) == set(ColumnRoles.__dataclass_fields__))
        for key in ("numeric", "categorical", "identifiers"):
            _require(type(value[key]) is list)
            value[key] = tuple(value[key])
        return ColumnRoles(**value)

    @property
    def bundle_directory(self):
        return self.export.directory / self.partition

    @property
    def row_positions(self):
        return self.split.positions(self.partition)

    @property
    def contract_id(self):
        return _digest(asdict(self))

    @property
    def source_id(self):
        return "tabular_partition:" + self.contract_id

    def check(self):
        """자기 선언이 아닌 승인·커밋·분할·내보내기 영수증을 재대조한다."""
        self.split.check()
        self.export.check()
        _require(self.split.artifact.scope.source_id == self.export.scope.source_id == self.parent_source_id)
        _require((self.export.scope.stage, self.export.scope.tool_id) == ("ingest", "run_ingest"))
        root = _root(self.export.directory)
        _require({path.name for path in root.iterdir()} == {"partition_exports.json", "train", "validation", "AGENT_COMPLETE.json"})
        receipt = read_metadata(root / "partition_exports.json")
        _require(set(receipt) == {"schema_version", "split_evidence_id", "source_id", "roles", "partitions"})
        _require(receipt["schema_version"] == "partition_exports_v2" and receipt["source_id"] == self.parent_source_id)
        _require(receipt["split_evidence_id"] == self.split.signature and receipt["roles"] == json.loads(_json(asdict(self.roles))))
        _require(type(receipt["partitions"]) is dict and set(receipt["partitions"]) == {"train", "validation"})
        record = receipt["partitions"][self.partition]
        _require(type(record) is dict and set(record) == {"bundle", "row_positions_sha256"})
        _require(BundleRef.from_dict(record["bundle"]) == self.bundle)
        _require(record["row_positions_sha256"] == _digest(list(self.row_positions)) and self.bundle.rows == len(self.row_positions))
        inspect_bundle(self.bundle_directory, expected=self.bundle, max_bytes=self.bundle.size_bytes)
        _require(tuple(row[0] for row in iter_bundle_rows(self.bundle_directory, expected=self.bundle)) == self.row_positions)
        schema = read_bundle_schema(self.bundle_directory, expected=self.bundle)
        _require(set(self.roles.required_columns).issubset(column["name"] for column in schema["columns"]))
        provenance = read_bundle_provenance(self.bundle_directory, expected=self.bundle)
        _require(provenance.get("parent_source_id") == self.parent_source_id
                 and provenance.get("split_signature") == self.split.signature
                 and provenance.get("row_positions_sha256") == record["row_positions_sha256"])
        return True

    def to_dict(self):
        """로컬 구간 계약을 JSON 호환 값으로 반환한다."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        """임의 객체나 버전 downgrade 없이 정적 증거 타입만 복원한다."""
        data = dict(_dict(value, cls))
        split = dict(_dict(data["split"], ConfirmedSplit))
        split["artifact"] = CommittedArtifact(**_dict(split["artifact"], CommittedArtifact))
        data["split"] = ConfirmedSplit(**split)
        data["export"] = CommittedArtifact(**_dict(data["export"], CommittedArtifact))
        data["bundle"] = BundleRef.from_dict(data["bundle"])
        return cls(**data)
