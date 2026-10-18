"""로컬 산출물의 완료 표식과 파일 정합성을 검사한다."""

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile


MARKER_NAME = "AGENT_COMPLETE.json"
MAX_MARKER_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class ArtifactContext:
    """재사용 시 일치해야 하는 로컬 계획·입력 식별값."""

    source_id: str
    config_id: str
    plan_id: str
    policy_version: str
    tool_id: str

    def __post_init__(self):
        values = asdict(self).values()

        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError("산출물 계약의 필수 식별값이 올바르지 않습니다.")


@dataclass(frozen=True)
class ArtifactCheck:
    """상세 파일 내용 없이 전달하는 재사용 검사 결과."""

    reusable: bool
    code: str


def _is_link(path):
    """심볼릭 링크와 Windows junction 등 재분석 지점을 거부한다."""
    info = path.lstat()
    attributes = getattr(info, "st_file_attributes", 0)

    return stat.S_ISLNK(info.st_mode) or bool(attributes & 0x400)


def _root(directory):
    """경로 구성 요소에 링크가 없는 실제 폴더를 확인한다."""
    root = Path(os.path.abspath(directory))

    for component in (root, *root.parents):
        if _is_link(component):
            raise ValueError("산출물 경로에 링크를 사용할 수 없습니다.")

    if not root.is_dir():
        raise ValueError("산출물 폴더가 필요합니다.")

    return root


def _safe_relative(value):
    """절대 경로·상위 이동·Windows 특수 이름을 허용하지 않는다."""
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_./-]+", value):
        raise ValueError("산출물 상대 경로가 올바르지 않습니다.")

    parts = value.split("/")
    reserved = {"CON", "PRN", "AUX", "NUL"}
    reserved.update(f"{prefix}{index}" for prefix in ("COM", "LPT") for index in range(1, 10))

    for part in parts:
        if part in ("", ".", "..") or part.endswith("."):
            raise ValueError("산출물 상대 경로가 올바르지 않습니다.")

        if part.split(".")[0].upper() in reserved:
            raise ValueError("산출물 경로에 예약 이름을 사용할 수 없습니다.")

    return PurePosixPath(value)


def _inventory(root):
    """완료 표식을 제외한 파일을 수집하며 링크·특수 파일을 차단한다."""
    files = {}

    # followlinks=False와 별도 재분석 지점 검사로 폴더 밖의 대상을 읽지 않는다.
    for directory, folders, names in os.walk(root, followlinks=False, onerror=_walk_error):
        for name in folders + names:
            path = Path(directory) / name

            if _is_link(path):
                raise ValueError("산출물에 링크를 사용할 수 없습니다.")

            if name in folders:
                continue

            relative = path.relative_to(root).as_posix()

            if relative == MARKER_NAME:
                continue

            _safe_relative(relative)

            if not stat.S_ISREG(path.stat().st_mode):
                raise ValueError("산출물은 일반 파일이어야 합니다.")

            files[relative] = path

    return files


def _walk_error(error):
    """읽지 못한 하위 폴더를 조용히 건너뛰지 않는다."""
    raise error


def _fingerprint(path):
    """파일을 작은 청크로 읽어 크기와 SHA-256을 자동 계산한다."""
    digest = hashlib.sha256()
    size = 0

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)

    return {"size": size, "sha256": digest.hexdigest()}


# ==========================================
# 완료 산출물 등록
# - 실행기는 도구별 내용 검증을 통과한 새 attempt 폴더만 전달한다.
# - 기존 전처리 코어의 SUCCESS.json은 변경하지 않는다.
# - 파일 목록·크기·해시를 계산한 뒤 별도 완료 표식을 생성한다.
# - 동일 attempt에 대한 동시 쓰기는 상위 실행기가 차단해야 한다.
# ==========================================
def complete_artifacts(directory, context: ArtifactContext):
    """검증된 산출물 폴더에 완료 표식을 추가한다. 내용 검증은 호출자 책임이다."""
    if not isinstance(context, ArtifactContext):
        raise TypeError("산출물 계약은 ArtifactContext여야 합니다.")

    root = _root(directory)
    marker = root / MARKER_NAME

    if marker.exists() or marker.is_symlink():
        raise FileExistsError("이미 완료 표식이 있는 산출물입니다.")

    files = _inventory(root)

    if not files:
        raise ValueError("빈 산출물은 완료 처리할 수 없습니다.")

    record = {
        "version": 1,
        "context": asdict(context),
        "files": {name: _fingerprint(path) for name, path in sorted(files.items())},
    }
    payload = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")

    if len(payload) > MAX_MARKER_BYTES:
        raise ValueError("산출물 완료 표식 크기 상한을 초과했습니다.")

    temporary = None

    try:
        with tempfile.NamedTemporaryFile(dir=root, prefix="agent_", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

        # 완성된 임시 파일을 연결한다. 기존 완료 표식은 덮어쓰지 않는다.
        os.link(temporary, marker)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    return marker


# ==========================================
# 재사용 검사
# - 완료 표식·입력 계약·실제 파일 목록이 모두 일치해야 한다.
# - 누락·추가·손상 파일과 부분 생성 결과는 재사용하지 않는다.
# - SHA-256은 손상 검사이며 악의적인 표식 재작성의 인증 수단이 아니다.
# ==========================================
def check_artifacts(directory, context: ArtifactContext) -> ArtifactCheck:
    """저장된 산출물의 현재 정합성을 확인하고 고정 결과 코드를 반환한다."""
    if not isinstance(context, ArtifactContext):
        raise TypeError("산출물 계약은 ArtifactContext여야 합니다.")

    try:
        root = _root(directory)
        marker = root / MARKER_NAME

        if not marker.exists():
            return ArtifactCheck(False, "ARTIFACT_INCOMPLETE")

        if _is_link(marker) or not marker.is_file() or marker.stat().st_size > MAX_MARKER_BYTES:
            return ArtifactCheck(False, "ARTIFACT_INVALID_MARKER")

        record = json.loads(marker.read_text(encoding="utf-8"), object_pairs_hook=_unique_keys)

        if not isinstance(record, dict) or set(record) != {"version", "context", "files"}:
            return ArtifactCheck(False, "ARTIFACT_INVALID_MARKER")

        if type(record["version"]) is not int or record["version"] != 1:
            return ArtifactCheck(False, "ARTIFACT_INVALID_MARKER")

        if record["context"] != asdict(context):
            return ArtifactCheck(False, "ARTIFACT_CONTEXT_MISMATCH")

        expected = record["files"]

        if not isinstance(expected, dict) or not expected:
            return ArtifactCheck(False, "ARTIFACT_INVALID_MARKER")

        actual = _inventory(root)

        for name, fingerprint in expected.items():
            _safe_relative(name)

            if not _valid_fingerprint(fingerprint):
                return ArtifactCheck(False, "ARTIFACT_INVALID_MARKER")

        if set(expected) != set(actual):
            return ArtifactCheck(False, "ARTIFACT_FILE_SET_MISMATCH")

        for name, path in actual.items():
            if _fingerprint(path) != expected[name]:
                return ArtifactCheck(False, "ARTIFACT_CONTENT_MISMATCH")

        return ArtifactCheck(True, "ARTIFACT_REUSABLE")
    except (OSError, ValueError, RecursionError):
        return ArtifactCheck(False, "ARTIFACT_UNREADABLE_OR_INVALID")


def _unique_keys(pairs):
    """JSON의 중복 키를 거부하여 해석 차이를 막는다."""
    result = {}

    for key, value in pairs:
        if key in result:
            raise ValueError("완료 표식에 중복 키가 있습니다.")

        result[key] = value

    return result


def _valid_fingerprint(value):
    """완료 표식의 파일 크기·해시 형식을 엄격하게 검사한다."""
    if not isinstance(value, dict) or set(value) != {"size", "sha256"}:
        return False

    return (
        type(value["size"]) is int
        and value["size"] >= 0
        and isinstance(value["sha256"], str)
        and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is not None
    )
