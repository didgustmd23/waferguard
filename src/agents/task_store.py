"""지속 전처리 작업의 로컬 기록. 대화 원문·모델 응답 원문·승인 ID·API 키는 저장하지 않는다.

기록은 진행 상태의 표시·재개 근거일 뿐 실행 권한이 아니다. 실제 완료 여부는 각 작업의 기존 검증기가 다시 판정한다.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from uuid import uuid4

from .artifacts import _is_link, _root
from .cli import _current_actor
from .executor import _run_lock
from .llm_proposal_store import _store_root
from .protected_store import _create_protected, _hold, _permission_ok, _security, check_object, check_tree, trusted_principals
from . import strict_json


STORE_ROOT = Path(__file__).resolve().parents[2] / "logs/agents/tasks"
SCHEMA_VERSION = "task_v1"
# 2026-10-21 이전 CSV 전용 기록. 읽을 때 표형 CSV 유형으로 간주하며 완료 결과 조회·보고서 재조립을 계속 지원한다.
LEGACY_VERSIONS = {"csv_task_v1": "tabular_csv"}
KIND_PATTERN = r"[a-z][a-z0-9_]{0,39}"
MAX_TASK_BYTES = 512 * 1024
# 진행 제어 정보(단계·동의·확정 계획·문제)는 기록 본문에 둔다. 넓은 표에서 커지는 내용 구역만 분리 저장한다.
PART_SECTIONS = ("diagnostic", "draft", "candidate", "report", "attempts", "input_selection")
PART_INLINE_BYTES = 32 * 1024
MAX_PART_BYTES = 16 * 1024 * 1024
PART_NAME = r"(diagnostic|draft|candidate|report|attempts|input_selection)-[0-9a-f]{64}\.json"
# 승인 판단에 쓰는 하위 기록(분리 저장·저장 실행안·입력 복사본). 작업 잠금 때 의미상 권한을 검사한다.
RECORD_FOLDERS = ("parts", "plans", "snapshot", "derived_snapshot")
PHASES = ("created", "diagnosing", "awaiting_input", "awaiting_review", "processing", "awaiting_apply", "applying", "completed",
          "needs_attention")


class TaskRecordError(ValueError):
    """작업 기록을 읽지 못한 확인된 원인. 화면이 원인별로 다음 행동을 안내한다."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _directory(task_id, root):
    if type(task_id) is not str or re.fullmatch(r"[a-f0-9]{32}", task_id) is None:
        raise ValueError("작업 ID 형식이 올바르지 않습니다.")

    return _root(_store_root(root, create=False) / task_id)


def create_task(goal, source, kind, *, root=None):
    """새 작업 폴더와 초기 기록만 만든다. 파일 복사·worker·모델 호출은 하지 않는다."""
    if type(kind) is not str or re.fullmatch(KIND_PATTERN, kind) is None:
        raise ValueError("작업 유형 형식이 올바르지 않습니다.")

    root = STORE_ROOT if root is None else Path(root)
    task_id = uuid4().hex
    directory = _store_root(root, create=True) / task_id
    # 생성 순간부터 신뢰 주체(현재 사용자·SYSTEM·Administrators)만 쓸 수 있는 보호 DACL. 하위 기록은 이 ACE만 상속한다.
    try:
        _create_protected(directory, trusted_principals()[0])
        _root(directory)
    except (OSError, ValueError) as error:
        raise TaskRecordError("RECORD_PROTECT_FAILED", "작업 폴더를 보호 권한으로 만들지 못했습니다.") from error
    record = {
        "schema_version": SCHEMA_VERSION, "kind": kind, "task_id": task_id, "actor_id": _current_actor(),
        "created_at": _now(), "updated_at": _now(), "revision": 0, "phase": "created", "problem": None,
        "goal": goal, "source": source, "snapshot": None, "consents": {}, "diagnostic": None, "draft": None,
        "confirmed": None, "attempt": 0, "steps": {}, "candidate": None, "report": None,
    }
    _write(directory / "task.json", record, exclusive=True)
    return record


def _encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")


def _replace(path, payload):
    # 완성된 임시 파일을 같은 폴더에서 교체한다. 중단돼도 이전 파일이 남는다.
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")

    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())

    os.replace(temporary, path)


def _unsafe(path):
    """링크(심볼릭·junction)이거나 하드 링크가 여러 개인 파일이면 참."""
    return _is_link(path) or (path.is_file() and path.stat().st_nlink > 1)


def _is_part(value):
    return type(value) is dict and set(value) == {"$part"}


def _store_parts(directory, record):
    """큰 내용 구역을 parts/<구역>-<sha256>.json으로 먼저 쓰고, 기록 본문에는 파일 이름·해시·크기만 남긴다.

    같은 내용은 같은 파일 이름이므로 다시 쓰지 않는다. 기존 파일이 다르면 같은 이름으로 원자적으로 고쳐 쓴다.
    이전 내용의 파일은 지우지 않는다(자동 삭제 없음).
    """
    stored = dict(record)

    for section in PART_SECTIONS:
        value = record.get(section)

        if value is None or _is_part(value):
            continue

        payload = _encode({"section": section, "value": value})

        if len(payload) <= PART_INLINE_BYTES:
            continue

        if len(payload) > MAX_PART_BYTES:
            raise ValueError("작업 기록 구역이 크기 상한을 초과했습니다.")

        digest = hashlib.sha256(payload).hexdigest()
        folder = directory / "parts"

        folder.mkdir(exist_ok=True)
        _root(folder)
        path = folder / f"{section}-{digest}.json"

        if path.exists() and _unsafe(path):
            raise ValueError("분리 저장 파일을 안전하게 열 수 없습니다.")

        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            _replace(path, payload)

        stored[section] = {"$part": {"file": path.name, "sha256": digest, "bytes": len(payload)}}

    return stored


def _load_parts(directory, record):
    """참조된 분리 저장 파일을 해시·크기·구역 이름으로 대조해 기록에 되돌린다. 하나라도 다르면 열지 않는다."""
    loaded = dict(record)

    for section in PART_SECTIONS:
        value = record.get(section)

        if not _is_part(value):
            continue

        ref = value["$part"]

        try:
            if (type(ref) is not dict or set(ref) != {"file", "sha256", "bytes"} or type(ref["file"]) is not str
                    or re.fullmatch(PART_NAME, ref["file"]) is None or not ref["file"].startswith(section + "-")
                    or ref["file"] != f"{section}-{ref['sha256']}.json" or type(ref["bytes"]) is not int
                    or not 0 < ref["bytes"] <= MAX_PART_BYTES):
                raise ValueError("분리 저장 참조 형식이 올바르지 않습니다.")

            path = _root(directory / "parts") / ref["file"]

            if not path.exists() or _unsafe(path) or not path.is_file() or path.stat().st_size != ref["bytes"]:
                raise ValueError("분리 저장 파일을 안전하게 열 수 없습니다.")

            payload = path.read_bytes()

            if len(payload) != ref["bytes"] or hashlib.sha256(payload).hexdigest() != ref["sha256"]:
                raise ValueError("분리 저장 파일 내용이 기록의 해시와 다릅니다.")

            item = strict_json.loads(payload.decode("utf-8"), MAX_PART_BYTES)

            if set(item) != {"section", "value"} or item["section"] != section:
                raise ValueError("분리 저장 파일의 구역이 기록과 다릅니다.")
        except (OSError, ValueError, UnicodeError) as error:
            raise TaskRecordError("RECORD_PART_INVALID", "작업 기록의 분리 저장 파일이 없거나 기록과 다릅니다.") from error

        loaded[section] = item["value"]

    return loaded


def _write(path, value, *, exclusive=False):
    payload = _encode(value)

    if len(payload) > MAX_TASK_BYTES:
        raise ValueError("작업 기록이 크기 상한을 초과했습니다.")

    if exclusive:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        return

    _replace(path, payload)


def load_task(task_id, *, root=None):
    """기록 형식·소유자를 확인해 읽는다. 다른 OS 사용자의 작업은 열지 않는다."""
    try:
        directory = _directory(task_id, STORE_ROOT if root is None else Path(root))
    except (OSError, ValueError) as error:
        raise TaskRecordError("RECORD_NOT_FOUND", "작업 기록을 찾지 못했습니다.") from error

    path = directory / "task.json"

    if not path.is_file() or path.is_symlink():
        raise TaskRecordError("RECORD_NOT_FOUND" if not path.is_symlink() else "RECORD_UNREADABLE",
                              "작업 기록 파일을 안전하게 열 수 없습니다.")

    if path.stat().st_size > MAX_TASK_BYTES:
        raise TaskRecordError("RECORD_TOO_LARGE", "작업 기록이 크기 상한을 초과했습니다.")

    # 작업 저장소 상한(512 KiB)으로 읽는다. worker IPC 상한(64 KiB)과 별개다. 중복 키·비표준 숫자는 거부한다.
    try:
        record = strict_json.loads(path.read_text(encoding="utf-8"), MAX_TASK_BYTES)
    except strict_json.JsonTooLarge as error:
        raise TaskRecordError("RECORD_TOO_LARGE", "작업 기록이 크기 상한을 초과했습니다.") from error
    except (OSError, ValueError, UnicodeError) as error:
        raise TaskRecordError("RECORD_UNREADABLE", "작업 기록 형식을 읽지 못했습니다.") from error

    if type(record) is dict and record.get("schema_version") in LEGACY_VERSIONS and "kind" not in record:
        record = dict(record, kind=LEGACY_VERSIONS[record["schema_version"]])

    if (record.get("schema_version") not in {SCHEMA_VERSION, *LEGACY_VERSIONS}
            or type(record.get("kind")) is not str or re.fullmatch(KIND_PATTERN, record["kind"]) is None
            or record.get("task_id") != task_id or record.get("phase") not in PHASES):
        raise TaskRecordError("RECORD_INVALID", "작업 기록 형식이 올바르지 않습니다.")

    if record.get("actor_id") != _current_actor():
        raise TaskRecordError("RECORD_OWNER_MISMATCH", "다른 OS 사용자의 작업 기록입니다.")

    return _load_parts(directory, record)


def _protected(directory, handle):
    """보유한 작업 폴더가 보호 경계 안인지 본다. 하나라도 다르면 False이며 아무것도 고치지 않는다.

    작업 폴더 자체(보유 핸들): 소유자·상속 차단·신뢰 주체 ACE. 직속 파일(task.json·잠금 등)과 승인 판단 하위 기록
    (parts·plans·snapshot·derived_snapshot): 소유자·신뢰 주체 외 허용 ACE·레이블·링크·하드 링크.
    workflow·profiles·report·input_runs는 보호 DACL 상속에 맡기고 여기서 의미상 검사하지 않는다(input_runs는 입력 서비스가 검사).
    """
    trusted = trusted_principals()

    try:
        _root(directory)

        if not _permission_ok(_security(handle=handle), trusted, None, root=True):
            return False

        for item in directory.iterdir():
            if item.is_dir() and not _is_link(item):
                if item.name in RECORD_FOLDERS and not check_tree(item, trusted):
                    return False
            elif not check_object(item, trusted):
                return False

        return True
    except (OSError, ValueError):
        return False


@contextmanager
def task_lock(task_id, *, root=None, protected=True):
    """같은 작업의 동시 진행을 막고, 잠금 동안 작업 폴더를 보유한다. 화면 재표시·중복 클릭은 잠금에서 거부된다.

    보유 중에는 작업 폴더와 상위 경로의 이름 변경·삭제가 막힌다. protected이면 작업 기록이 보호 경계 안일 때만 진행한다
    (위임 발급·진행·복구). 보호되지 않은 이전 작업은 RECORD_UNPROTECTED로 멈추며 권한을 고치지 않는다.
    protected=False는 승인을 발급하지 않는 보고서 재조립에만 쓴다.
    """
    directory = _directory(task_id, STORE_ROOT if root is None else Path(root))

    with _run_lock(directory) as acquired:
        if not acquired:
            raise ValueError("이 작업은 다른 요청이 진행 중입니다.")

        try:
            handle, release = _hold(directory)
        except OSError as error:
            raise TaskRecordError("RECORD_UNREADABLE", "작업 폴더를 안전하게 열 수 없습니다.") from error

        try:
            if protected and not _protected(directory, handle):
                raise TaskRecordError("RECORD_UNPROTECTED", "작업 기록이 보호 경계 밖에 있어 승인·진행에 쓰지 않습니다.")

            if not protected:
                # 보고서 재조립도 보유 뒤 경로가 그대로인지 확인한다(링크로 다른 작업 폴더를 열지 않음).
                try:
                    _root(directory)
                except ValueError as error:
                    raise TaskRecordError("RECORD_UNREADABLE", "작업 폴더를 안전하게 열 수 없습니다.") from error

            yield directory
        finally:
            release()


def save_task(record, *, expected_revision, root=None):
    """revision이 기대값과 같을 때만 저장한다. 중간 변경을 덮어쓰지 않는다."""
    current = load_task(record["task_id"], root=root)

    if current["revision"] != expected_revision:
        raise ValueError("작업 기록이 다른 요청으로 변경되었습니다.")

    updated = dict(record, revision=expected_revision + 1, updated_at=_now())

    if updated["phase"] not in PHASES:
        raise ValueError("지원하지 않는 작업 단계입니다.")

    directory = _directory(record["task_id"], STORE_ROOT if root is None else Path(root))
    # 분리 저장 파일을 먼저 쓴다. 본문 저장이 실패해도 이전 본문과 그 참조 파일은 그대로다.
    _write(directory / "task.json", _store_parts(directory, updated))
    return updated


def task_directory(task_id, *, root=None):
    return _directory(task_id, STORE_ROOT if root is None else Path(root))
