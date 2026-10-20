"""제안의 로컬 보관·복원. 해시는 변경 감지용이며 실행 승인·진본 인증이 아니다."""

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import stat

from src.preprocessing.module_contracts import check_id
from .artifacts import _is_link
from .llm_proposal import ALLOWED, SOURCE
from .llm_proposal_contract import _decode_response, validate_proposal
from .worker_paths import _path


STORE_ROOT = Path(__file__).resolve().parents[2] / "logs/agents/llm_proposals"
MAX_SAVED_BYTES = 64 * 1024
MAX_LISTED = 200


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _store_path(name, root, *, create):
    """모델이 아닌 사용자의 짧은 이름으로만 경로를 정하고 링크·덮어쓰기를 거부한다."""
    check_id(name)
    return _store_root(root, create=create) / f"{name}.json"


def _store_root(root, *, create):
    root = Path(root)

    if not root.is_absolute():
        raise ValueError("제안 보관 위치는 절대 로컬 경로여야 합니다.")

    # 존재하는 조상부터 링크 여부를 확인한다. 개인 사용자의 동시 경로 변조까지
    # 방어하는 OS 보관소가 아니며 승인 DB나 실행기의 파일 권한을 바꾸지 않는다.
    for component in (root, *root.parents):
        try:
            if _is_link(component):
                raise ValueError("제안 보관 경로에 링크를 사용할 수 없습니다.")
        except FileNotFoundError:
            continue

    if create:
        root.mkdir(parents=True, exist_ok=True)

    return _path(root, directory=True)


def list_saved_proposals(*, root=STORE_ROOT):
    """저장 파일의 이름·수정 시각만 최근 순으로 나열한다.

    파일 내용은 읽지 않는다. 목록에 있다는 것은 검증 통과가 아니며 복원 시 load_proposal이 검사한다.
    수정 시각은 파일 시스템 값이며 저장 시점의 진본 기록이 아니다.
    """
    if not Path(root).is_absolute():
        raise ValueError("제안 보관 위치는 절대 로컬 경로여야 합니다.")

    if not Path(root).exists():
        return []

    items = []

    for entry in _store_root(root, create=False).iterdir():
        if entry.suffix != ".json":
            continue

        try:
            check_id(entry.stem)
        except ValueError:
            continue

        info = entry.lstat()

        if _is_link(entry) or not stat.S_ISREG(info.st_mode):
            continue

        saved_at = datetime.fromtimestamp(info.st_mtime).astimezone()
        items.append({"name": entry.stem, "saved_at": saved_at.isoformat(timespec="seconds")})

    items.sort(key=lambda item: (item["saved_at"], item["name"]), reverse=True)
    return items[:MAX_LISTED]


def _validated(value):
    """저장 파일의 허용 목록을 믿지 않고 현재 프로그램의 계약으로 다시 검사한다."""
    proposal = validate_proposal(
        _canonical(value), expected_input=dict(SOURCE), allowed_modules=ALLOWED,
    )

    if value["status"] != "proposed":
        raise ValueError("미해결 확인 질문은 실행안으로 보관하지 않습니다.")

    return proposal


def save_proposal(name, result, *, root=STORE_ROOT):
    """확인된 실행안만 새 파일에 저장한다. 요청 원문·모델 원문·승인 토큰은 저장하지 않는다."""
    if (type(result) is not dict or result.get("status") != "proposal_validated"
            or result.get("proposal_only") is not True
            or result.get("execution_approved") is not False
            or result.get("worker_started") is not False):
        raise ValueError("실행 권한이 없는 검증된 제안만 저장할 수 있습니다.")

    value = result.get("proposal")
    proposal = _validated(value)

    if proposal.plan_signature != result.get("plan_signature"):
        raise ValueError("표시한 제안과 계획 지문이 일치하지 않습니다.")

    envelope = {
        "schema_version": "llm_saved_proposal_v1", "proposal": value,
        "proposal_sha256": _digest(value), "plan_signature": proposal.plan_signature,
        "proposal_only": True, "execution_approved": False, "worker_started": False,
    }
    payload = _canonical(envelope).encode("utf-8")

    if len(payload) > MAX_SAVED_BYTES:
        raise ValueError("저장할 제안은 64 KiB 이하여야 합니다.")

    target = _store_path(name, root, create=True)

    # 단일 파일을 배타 생성한다. 중단으로 생긴 불완전 파일도 덮어쓰지 않고
    # 복원 검사에서 거부한다. 승인·완료 표식을 생성하는 작업이 아니다.
    with target.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())

    restored = load_proposal(name, root=root)
    return {"status": "proposal_saved", "name": name, "local_file": str(target),
            "plan_signature": restored["plan_signature"], "execution_approved": False,
            "worker_started": False}


def load_proposal(name, *, root=STORE_ROOT):
    """크기·링크·체크섬·현재 계약·계획 지문을 재검사하고 승인 없는 제안으로만 복원한다."""
    target = _path(_store_path(name, root, create=False), directory=False)

    with target.open("rb") as stream:
        raw = stream.read(MAX_SAVED_BYTES + 1)

    if len(raw) > MAX_SAVED_BYTES:
        raise ValueError("저장 제안의 크기 상한을 초과했습니다.")

    envelope = _decode_response(raw.decode("utf-8"))
    expected = {"schema_version", "proposal", "proposal_sha256", "plan_signature",
                "proposal_only", "execution_approved", "worker_started"}

    if (type(envelope) is not dict or set(envelope) != expected
            or envelope["schema_version"] != "llm_saved_proposal_v1"
            or envelope["proposal_only"] is not True
            or envelope["execution_approved"] is not False
            or envelope["worker_started"] is not False):
        raise ValueError("저장 제안의 버전·필드·권한 선언이 올바르지 않습니다.")

    value = envelope["proposal"]

    if _digest(value) != envelope["proposal_sha256"]:
        raise ValueError("저장한 제안 내용이 변경되었습니다.")

    proposal = _validated(value)

    if proposal.plan_signature != envelope["plan_signature"]:
        raise ValueError("현재 등록 모듈·설정·계획 지문과 저장 제안이 일치하지 않습니다.")

    result = proposal.to_dict()
    result.update(status="proposal_restored", name=name, model_called=False)
    return result
