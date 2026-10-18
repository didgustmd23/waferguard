"""WM canonical 묶음의 바이트 복사·파일 지문·승인 입력 계약."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re

from src.preprocessing.canonical_data import decode_json
from .artifacts import _is_link, _root
from .worker_contract import WorkerLimits


POLICY_VERSION = "wm_canonical_eda_v1"
MAX_FILES = 4096
MANAGEMENT = frozenset({"manifest.jsonl", "conversion.json", "provenance.json", "SUCCESS.json"})


def digest(value):
    """파일·설정을 정렬된 JSON으로 결합한다. 전자 서명이나 안전성 인증은 아니다."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path, maximum):
    """파일이 복사·검사 중 커져도 상한보다 오래 읽지 않는다."""
    hasher = hashlib.sha256()
    size = 0

    with path.open("rb") as stream:
        while True:
            chunk = stream.read(min(1024 * 1024, maximum - size + 1))

            if not chunk:
                break

            size += len(chunk)

            if size > maximum:
                raise ValueError("승인된 파일 크기 상한을 초과했습니다.")

            hasher.update(chunk)

    return {"size": size, "sha256": hasher.hexdigest()}


def inventory(directory, maximum, *, staged=False):
    """NPZ를 파싱하지 않고 허용된 일반 파일 목록·크기·해시만 고정한다."""
    root = Path(directory) if staged else _root(directory)

    if _is_link(root) or not root.is_dir() or {p.name for p in root.iterdir()} != MANAGEMENT | {"maps"}:
        raise ValueError("지원되는 canonical 파일 목록이 필요합니다.")

    maps = root / "maps"

    if _is_link(maps) or not maps.is_dir():
        raise ValueError("canonical maps는 링크가 아닌 폴더여야 합니다.")

    paths = [root / name for name in sorted(MANAGEMENT)]

    for path in maps.iterdir():
        if re.fullmatch(r"shard_[0-9]{6,7}\.npz", path.name) is None or len(paths) >= MAX_FILES:
            raise ValueError("canonical shard 이름 또는 파일 개수 상한이 올바르지 않습니다.")

        paths.append(path)

    files = {}
    total = 0

    for path in sorted(paths):
        if _is_link(path) or not path.is_file() or path.stat().st_nlink != 1:
            raise ValueError("canonical 입력에는 일반 파일만 허용합니다.")

        record = file_hash(path, maximum - total)
        total += record["size"]
        files[path.relative_to(root).as_posix()] = record

    return files


def check_files(files, maximum):
    """저장된 파일 명세도 경로·타입·크기·SHA-256을 엄격히 검사한다."""
    if type(files) is not dict or not 4 <= len(files) <= MAX_FILES or not MANAGEMENT <= set(files):
        raise ValueError("canonical 파일 명세가 올바르지 않습니다.")

    total = 0

    for name, record in files.items():
        if name not in MANAGEMENT and re.fullmatch(r"maps/shard_[0-9]{6,7}\.npz", name) is None:
            raise ValueError("canonical 명세에 허용되지 않은 경로가 있습니다.")

        if (type(record) is not dict or set(record) != {"size", "sha256"}
            or type(record["size"]) is not int or record["size"] < 0
            or type(record["sha256"]) is not str or re.fullmatch(r"[a-f0-9]{64}", record["sha256"]) is None):
            raise ValueError("canonical 파일 크기 또는 지문이 올바르지 않습니다.")

        total += record["size"]

    if total > maximum:
        raise ValueError("canonical 파일 합계가 입력 상한을 초과했습니다.")


def copy_snapshot(source, target, maximum, expected=None):
    """원본은 보존하고 새 묶음에만 복사한 뒤 양쪽 지문을 다시 대조한다."""
    source = _root(source)
    before = inventory(source, maximum)

    if expected is not None and before != expected:
        raise ValueError("canonical 입력이 승인된 파일 명세와 다릅니다.")

    target = Path(target)
    target.mkdir(exist_ok=False)
    (target / "maps").mkdir()

    for name, record in before.items():
        remaining = record["size"]

        with (source / name).open("rb") as reader, (target / name).open("xb") as writer:
            while True:
                chunk = reader.read(min(1024 * 1024, remaining + 1))

                if not chunk:
                    break

                remaining -= len(chunk)

                if remaining < 0:
                    raise ValueError("복사 중 canonical 입력이 커졌습니다.")

                writer.write(chunk)

    if inventory(target, maximum) != before or inventory(source, maximum) != before:
        raise ValueError("복사 중 canonical 입력이 변경되었습니다.")

    return before


@dataclass(frozen=True)
class WmCanonicalInput:
    """파일 바이트와 실행 설정을 함께 고정하며 데이터 값은 LLM에 전달하지 않는다."""

    snapshot_directory: str
    files_json: str
    limits: WorkerLimits
    max_input_bytes: int
    max_output_bytes: int
    include_examples: bool = False
    max_group_ratio: float = 0.2

    def __post_init__(self):
        if type(self.snapshot_directory) is not str or not Path(self.snapshot_directory).is_absolute():
            raise ValueError("canonical 복사본의 절대 경로가 필요합니다.")

        if not isinstance(self.limits, WorkerLimits):
            raise TypeError("worker 자원 계약이 필요합니다.")

        for value in (self.max_input_bytes, self.max_output_bytes):
            if type(value) is not int or not 1 <= value <= 2 ** 63 - 1:
                raise ValueError("입력·출력 상한은 유효한 양의 정수여야 합니다.")

        if (type(self.include_examples) is not bool or type(self.max_group_ratio) not in (int, float)
            or not 0 < self.max_group_ratio <= 1):
            raise ValueError("예시 이미지 여부 또는 연결 그룹 검토 상한이 올바르지 않습니다.")

        check_files(self.files, self.max_input_bytes)

    @property
    def files(self):
        return decode_json(self.files_json.encode("utf-8"))

    @property
    def source_id(self):
        return "wm_canonical:" + digest(self.files)

    def check(self):
        if inventory(self.snapshot_directory, self.max_input_bytes) != self.files:
            raise ValueError("canonical 복사본이 승인된 파일 명세와 다릅니다.")

    def packet(self, code_id):
        return {"version": POLICY_VERSION, "source_id": self.source_id, "limits": asdict(self.limits),
                "max_input_bytes": self.max_input_bytes, "max_output_bytes": self.max_output_bytes,
                "include_examples": self.include_examples, "max_group_ratio": self.max_group_ratio,
                "code_id": code_id}

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("canonical 입력 계약 필드가 올바르지 않습니다.")

        value = dict(value)
        value["limits"] = WorkerLimits(**value["limits"])
        return cls(**value)
