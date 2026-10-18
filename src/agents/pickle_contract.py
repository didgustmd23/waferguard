"""구조 판별형 pickle 변환의 입력·자동 선택 규칙 계약."""

from dataclasses import asdict, dataclass
import hashlib
import json

from .artifacts import _root
from .input_snapshot import Snapshot, verify_snapshot
from .worker_contract import WorkerLimits


POLICY_VERSION = "pickle_import_v2"
ADAPTER_ID = "pickle_auto_v1"
CONVERTERS = frozenset({"auto", "tabular_v1", "numeric_array_v1", "wm811k_v1"})


def digest(value):
    """승인에 묶을 JSON 입력의 내용 식별값을 계산한다."""
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def check_options(max_output_bytes, shard_rows):
    """출력 상한과 shard 크기는 요청을 만들기 전에 검증한다."""
    if type(max_output_bytes) is not int or not 1 <= max_output_bytes <= 2 ** 63 - 1:
        raise ValueError("출력 바이트 상한은 유효한 양의 정수여야 합니다.")

    if type(shard_rows) is not int or not 1 <= shard_rows <= 1000:
        raise ValueError("shard 행 수는 1~1000 사이의 정수여야 합니다.")


@dataclass(frozen=True)
class PickleInput:
    """원본 내용·경로 대신 고정된 복사본을 승인 범위에 결합한다."""

    snapshot_directory: str
    snapshot: Snapshot
    limits: WorkerLimits
    max_output_bytes: int
    shard_rows: int = 1000
    converter_id: str = "auto"
    select_path: tuple = ()

    def __post_init__(self):
        if not isinstance(self.snapshot, Snapshot) or not isinstance(self.limits, WorkerLimits):
            raise TypeError("복사본과 worker 자원 계약이 필요합니다.")

        if type(self.snapshot_directory) is not str or not self.snapshot_directory:
            raise ValueError("복사본 폴더의 절대 경로가 필요합니다.")

        check_options(self.max_output_bytes, self.shard_rows)

        if self.converter_id not in CONVERTERS:
            raise ValueError("등록되지 않은 pickle 변환기입니다.")

        if not isinstance(self.select_path, tuple) or len(self.select_path) > 4:
            raise ValueError("선택 경로는 깊이 4 이하의 키·인덱스 목록이어야 합니다.")

        for value in self.select_path:
            if not ((type(value) is str and 0 < len(value) <= 256) or
                    (type(value) is int and 0 <= value <= 2_000_000)):
                raise ValueError("선택 경로에는 문자열 키 또는 음수가 아닌 인덱스만 허용합니다.")

    def check(self):
        """부모에서만 전체 경로와 복사본 해시를 재확인한다."""
        root = _root(self.snapshot_directory)

        if str(root) != self.snapshot_directory or not verify_snapshot(root, self.snapshot):
            raise ValueError("pickle 복사본이 저장된 입력 계약과 다릅니다.")

        return True

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("pickle 입력 계약 필드가 올바르지 않습니다.")

        value = dict(value)
        value["snapshot"] = Snapshot(**value["snapshot"])
        value["limits"] = WorkerLimits(**value["limits"])
        if type(value["select_path"]) is not list:
            raise ValueError("저장된 선택 경로는 JSON 배열이어야 합니다.")

        value["select_path"] = tuple(value["select_path"])

        return cls(**value)

    def packet(self, code_id):
        """worker에는 원본·승인 DB 경로 없이 처리에 필요한 고정 설정만 보낸다."""
        return {
            "version": POLICY_VERSION, "adapter_id": ADAPTER_ID,
            "snapshot": asdict(self.snapshot), "limits": asdict(self.limits),
            "max_output_bytes": self.max_output_bytes, "shard_rows": self.shard_rows,
            "code_id": code_id, "converter_id": self.converter_id,
            "select_path": list(self.select_path),
        }
