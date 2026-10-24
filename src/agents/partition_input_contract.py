"""확정 분할과 커밋된 구간 내보내기 기록에 묶인 입력 계약. 임의 CSV를 Train으로 인정하지 않는다."""

from dataclasses import asdict, dataclass
import json

from src.preprocessing.contracts import ColumnRoles
from src.preprocessing.split import _digest
from .input_snapshot import SNAPSHOT_VERSION, Snapshot, verify_snapshot
from .split_evidence import CommittedArtifact, ConfirmedSplit, read_metadata
from .artifacts import _root


def bind_typed_partition_input(split, export, partition, roles):
    """v2 내보내기만 자료형 보존 구간 계약으로 복원한다. 임의 파일은 받지 않는다."""
    from src.preprocessing.tabular_bundle import BundleRef
    from .tabular_source_contract import TabularPartitionInputContract
    # 분할·내보내기 증거 재검사는 아래 contract.check()가 수행한다. 영수증은 크기 제한 JSON으로만 먼저 읽는다.
    receipt = read_metadata(export.directory / "partition_exports.json")
    if (partition not in ("train", "validation") or receipt.get("schema_version") != "partition_exports_v2"
            or split.artifact.scope.policy_version != "tabular_split_v2" or export.scope.policy_version != "partition_export_v2"
            or receipt.get("roles") != json.loads(json.dumps(asdict(roles)))):
        raise ValueError("승인된 v2 분할·내보내기·역할이 필요합니다.")
    try:
        contract = TabularPartitionInputContract(split, export, partition, BundleRef.from_dict(receipt["partitions"][partition]["bundle"]),
                                                 json.dumps(asdict(roles), ensure_ascii=False), receipt["source_id"])
    except (KeyError, TypeError) as error:
        raise ValueError("내보내기 영수증의 구간 기록이 올바르지 않습니다.") from error
    contract.check()
    return contract


@dataclass(frozen=True)
class PartitionInputContract:
    """로컬 전용 구간 계약. 데이터·라벨·대치 통계를 포함하지 않고 변환도 실행하지 않는다."""

    split: ConfirmedSplit
    export: CommittedArtifact
    partition: str
    snapshot: Snapshot
    roles_json: str
    schema_version: str = "partition_input_v1"

    def __post_init__(self):
        if not isinstance(self.split, ConfirmedSplit) or not isinstance(self.export, CommittedArtifact):
            raise TypeError("확정 분할과 커밋 내보내기 증거가 필요합니다.")

        if self.partition not in ("train", "validation") or self.schema_version != "partition_input_v1":
            raise ValueError("초기 입력 계약은 Train·Validation만 지원합니다.")

        if not isinstance(self.snapshot, Snapshot):
            raise TypeError("구간의 Snapshot 계약이 필요합니다.")

        self.roles

    @property
    def roles(self):
        value = json.loads(self.roles_json)

        if type(value) is not dict or set(value) != set(ColumnRoles.__dataclass_fields__):
            raise ValueError("구간 입력의 컬럼 역할 계약이 올바르지 않습니다.")

        for name in ("numeric", "categorical", "identifiers"):
            if type(value[name]) is not list:
                raise ValueError("구간 컬럼 역할 목록은 JSON 배열이어야 합니다.")

            value[name] = tuple(value[name])

        return ColumnRoles(**value)

    @property
    def snapshot_directory(self):
        # 입력 JSON에서 임의 경로를 받지 않고 검증된 산출물 아래의 고정 구간명만 사용한다.
        return self.export.directory / self.partition

    @property
    def row_positions(self):
        return self.split.positions(self.partition)

    @property
    def contract_id(self):
        return _digest(asdict(self))

    def check(self):
        """저장된 객체의 선언만 믿지 않고 승인·커밋·구간·바이트 결합을 다시 확인한다."""
        actual = bind_partition_input(self.split, self.export, self.partition, self.roles)

        if actual != self:
            raise ValueError("구간 입력 계약이 현재 증거와 다릅니다.")

        return True


def bind_partition_input(split, export, partition, roles):
    """이미 내보낸 결과만 결합한다. CSV 추출·새 snapshot 생성·분할·fit은 수행하지 않는다."""
    if not isinstance(split, ConfirmedSplit) or not isinstance(export, CommittedArtifact) or not isinstance(roles, ColumnRoles):
        raise TypeError("확정 분할·내보내기 증거·ColumnRoles가 필요합니다.")

    if partition not in ("train", "validation"):
        raise ValueError("Test는 초기 변환 입력 계약에 자동 포함하지 않습니다.")

    split.check()
    export.check()

    if export.scope.tool_id != "run_ingest" or export.scope.source_id != split.artifact.scope.source_id:
        raise ValueError("동일 원본의 승인된 구간 내보내기 작업이 필요합니다.")

    root = _root(export.directory)

    if {path.name for path in root.iterdir()} != {"partition_exports.json", "train", "validation", "AGENT_COMPLETE.json"}:
        raise ValueError("Train·Validation 내보내기 파일 계약이 올바르지 않습니다.")

    receipt = read_metadata(root / "partition_exports.json")
    expected_roles = json.loads(json.dumps(asdict(roles)))
    required = {"schema_version", "split_evidence_id", "source_id", "proposal_id", "roles", "partitions"}

    if set(receipt) != required or receipt["schema_version"] != "partition_exports_v1":
        raise ValueError("검증된 구간 내보내기 기록이 필요합니다.")

    if (
        receipt["split_evidence_id"] != split.signature
        or receipt["source_id"] != split.artifact.scope.source_id
        or receipt["proposal_id"] != split.proposal["proposal_id"]
        or receipt["roles"] != expected_roles
    ):
        raise ValueError("내보내기 출처·분할·컬럼 역할이 기대 계약과 다릅니다.")

    partitions = receipt["partitions"]

    if type(partitions) is not dict or set(partitions) != {"train", "validation"}:
        raise ValueError("승인된 Train·Validation 내보내기 목록이 필요합니다.")

    snapshots = {}

    for name in ("train", "validation"):
        record = partitions[name]
        positions = split.positions(name)

        if type(record) is not dict or set(record) != {"row_count", "row_positions_sha256", "snapshot"}:
            raise ValueError("구간별 내보내기 필드가 올바르지 않습니다.")

        if type(record["row_count"]) is not int or record["row_count"] != len(positions) or record["row_positions_sha256"] != _digest(list(positions)):
            raise ValueError("구간의 행 수·순서 식별값이 확정 분할과 다릅니다.")

        if type(record["snapshot"]) is not dict or set(record["snapshot"]) != set(Snapshot.__dataclass_fields__):
            raise ValueError("구간 복사본의 크기·해시 계약이 올바르지 않습니다.")

        snapshots[name] = Snapshot(**record["snapshot"])

    selected = snapshots[partition]
    marker = read_metadata(root / partition / "snapshot.json")

    if marker != {"version": SNAPSHOT_VERSION, **asdict(selected)} or not verify_snapshot(root / partition, selected):
        raise ValueError("선택한 구간의 실제 복사본이 내보내기 기록과 다릅니다.")

    # 전체 원본·Test CSV를 파싱하지 않는다. 승인된 내보내기의 파생 관계를 신뢰 경계로 둔다.
    export.check()

    return PartitionInputContract(
        split, export, partition, selected,
        json.dumps(expected_roles, ensure_ascii=False, sort_keys=True, allow_nan=False),
    )
