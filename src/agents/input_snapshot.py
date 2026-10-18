"""형식 독립적인 로컬 파일 복사본. 내용 해석·EDA·승인 발급·LLM 전송은 하지 않는다."""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re

from .artifacts import _root
from .worker_paths import _path


MIB = 1024 * 1024
CHUNK_BYTES = MIB
SNAPSHOT_VERSION = "file_snapshot_v1"
PAYLOAD_NAME = "input.bin"


class InputTooLarge(ValueError):
    """원문 경로·데이터 없이 사용자에게 크기와 상한을 안내한다."""

    def __init__(self, size, maximum, *, started=False):
        super().__init__("입력 파일이 설정한 크기 상한을 초과했습니다.")
        self.size = size
        self.maximum = maximum
        self.started = started


@dataclass(frozen=True)
class Snapshot:
    """승인 결합에 사용할 복사본 내용 식별값. 자체 승인을 의미하지 않는다."""

    size_bytes: int
    sha256: str
    max_input_bytes: int

    def __post_init__(self):
        if (
            type(self.size_bytes) is not int or type(self.max_input_bytes) is not int
            or not 0 < self.size_bytes <= self.max_input_bytes
            or not isinstance(self.sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", self.sha256)
        ):
            raise ValueError("복사본 크기·해시 계약이 올바르지 않습니다.")

    @property
    def source_id(self):
        return "file_snapshot:" + self.sha256


def _identity(info):
    """파일 교체·크기·수정 시각 변경을 탐지한다. 악의적 동시 변경의 완전한 방어는 아니다."""
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _scan(stream, maximum, target=None):
    """전체 파일을 메모리에 올리지 않고 제한된 청크로 해시·복사를 수행한다."""
    digest = hashlib.sha256()
    size = 0

    while True:
        # 상한 직후 한 바이트까지만 읽어 증가를 탐지한다.
        chunk = stream.read(min(CHUNK_BYTES, maximum - size + 1))

        if not chunk:
            break

        size += len(chunk)

        if size > maximum:
            raise InputTooLarge(size, maximum, started=target is not None)

        digest.update(chunk)

        if target is not None:
            target.write(chunk)

    return size, digest.hexdigest()


def create_snapshot(source, destination, *, max_input_bytes):
    """새 폴더에만 복사한다. 실패한 부분 파일은 보존하되 완료 manifest는 만들지 않는다."""
    if type(max_input_bytes) is not int or max_input_bytes <= 0:
        raise ValueError("입력 크기 상한은 양의 정수 바이트여야 합니다.")

    source = _path(source, directory=False)

    initial = source.stat()

    # 초과 입력은 출력 폴더 생성이나 복사 전에 거부한다.
    if initial.st_size > max_input_bytes:
        raise InputTooLarge(initial.st_size, max_input_bytes)

    if initial.st_size == 0:
        raise ValueError("비어 있지 않은 일반 입력 파일이 필요합니다.")

    destination = Path(destination).absolute()
    _root(destination.parent)

    # 원본/기존 폴더를 덮어쓰지 않는다. 경로 구성 요소도 검사한다.
    if destination.exists() or destination.is_symlink():
        raise ValueError("존재하지 않는 새 snapshot 폴더가 필요합니다.")

    destination.mkdir(exist_ok=False)
    _root(destination)
    # 확장자로 파일을 실행·해석하지 않는다. 어댑터는 별도 프로파일에서 선택한다.
    partial = destination / (PAYLOAD_NAME + ".part")

    with source.open("rb") as stream:
        if _identity(os.fstat(stream.fileno())) != _identity(initial):
            raise ValueError("복사 전에 입력 파일이 변경되었습니다.")

        with partial.open("xb") as target:
            size, digest = _scan(stream, max_input_bytes, target)
            target.flush()
            os.fsync(target.fileno())

        # 같은 열린 파일을 다시 읽어 복사 바이트와 비교한다.
        stream.seek(0)
        try:
            checked_size, checked_digest = _scan(stream, max_input_bytes)
        except InputTooLarge as error:
            error.started = True
            raise
        final = os.fstat(stream.fileno())

    current = _path(source, directory=False).stat()

    if (
        _identity(initial) != _identity(final)
        or _identity(initial) != _identity(current)
        or (size, digest) != (checked_size, checked_digest)
    ):
        raise ValueError("복사 중 원본이 변경되었습니다. 부분 결과를 사용하지 마세요.")

    snapshot = Snapshot(size, digest, max_input_bytes)

    # 복사본 자체도 다시 확인한 뒤에만 완료 manifest를 기록한다.
    with partial.open("rb") as stream:
        if _scan(stream, max_input_bytes) != (size, digest):
            raise ValueError("복사본 정합성 검증에 실패했습니다.")

    partial.rename(destination / PAYLOAD_NAME)
    manifest = {"version": SNAPSHOT_VERSION, **asdict(snapshot)}

    with (destination / "snapshot.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())

    return snapshot


def verify_snapshot(directory, expected):
    """외부에서 보존한 기대 계약과 대조한다. manifest만으로 승인·안전성을 판단하지 않는다."""
    if not isinstance(expected, Snapshot):
        raise TypeError("기대 Snapshot 계약이 필요합니다.")

    root = _root(directory)

    if {item.name for item in root.iterdir()} != {PAYLOAD_NAME, "snapshot.json"}:
        return False

    source = _path(root / PAYLOAD_NAME, directory=False)
    marker = _path(root / "snapshot.json", directory=False)

    if marker.stat().st_size > 4096:
        return False

    manifest = json.loads(marker.read_text(encoding="utf-8"))

    if manifest != {"version": SNAPSHOT_VERSION, **asdict(expected)}:
        return False

    with source.open("rb") as stream:
        return _scan(stream, expected.max_input_bytes) == (expected.size_bytes, expected.sha256)


def main():
    parser = argparse.ArgumentParser(description="로컬 파일 크기 제한 복사본 준비 (내용 해석·EDA는 하지 않음)")
    parser.add_argument("--input", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--max-input-mib", required=True, type=int)
    parser.add_argument("--confirm-local-copy", action="store_true")
    args = parser.parse_args()

    try:
        if not args.confirm_local_copy:
            raise ValueError("로컬 복사에 명시적으로 동의해야 합니다.")

        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,47}", args.name):
            raise ValueError("새 snapshot 이름 형식이 올바르지 않습니다.")

        if args.max_input_mib <= 0:
            raise ValueError("입력 크기 상한은 양의 정수 MiB여야 합니다.")

        root = Path(__file__).resolve().parents[2]

        for part in ("logs", "agents", "input_snapshots"):
            root = root / part
            root.mkdir(exist_ok=True)
            _root(root)

        destination = root / args.name
        snapshot = create_snapshot(
            Path(args.input).absolute(), destination,
            max_input_bytes=args.max_input_mib * MIB,
        )
        print("입력 복사본 준비 완료")
        print(f"로컬 저장 경로: {destination}")
        print(f"복사 크기: {snapshot.size_bytes / MIB:.2f} MiB")
        print("내용 해석·EDA·worker 실행·승인 발급은 하지 않았습니다. LLM으로 전달하지 마세요.")
    except InputTooLarge as error:
        print(str(error))
        print(f"확인한 파일 크기: {error.size / MIB:.2f} MiB")
        print(f"허용 상한: {error.maximum / MIB:.2f} MiB")
        print("복사 중 상한을 초과해 중단했습니다. 부분 결과를 사용하지 마세요." if error.started else "파일을 복사하거나 EDA를 실행하지 않았습니다.")
        print("더 큰 파일이 필요하면 --max-input-mib 값을 조정한 뒤 새 작업으로 요청하세요.")
        raise SystemExit(2)
    except Exception:
        print("복사본 준비에 실패했습니다. 동의·크기 상한·로컬 일반 파일·새 이름을 확인하세요.")
        print("실패한 폴더가 있으면 미완료 결과입니다. EDA는 실행하지 않았습니다.")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
