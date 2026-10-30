"""작업 결정(A·C·D·T·R) 동의 기록과 위임 승인 발급 조건. 일치 판정은 계약 코드가 하며 LLM이 정하지 않는다.

# ==========================================
# 위임 근거 (docs/agents/design/CSV_TASK_AGENT.md 2.5절·7.6절)
# - A: 입력 진단(허용한 정비 포함)
# - C: 확정한 계획의 처리 단계(결정 D 전까지)
# - D: 사용자가 본 적용 대상의 적용과 결과 전달
# - T: 확정 전 계획의 시간 결측 복원 미리보기(계획 지문·시간 규칙·입력 출처 결합, 표형 v2 time 계획)
# - R: 사용자가 본 복원 미리보기 결과(미리보기 지문)의 적용과 파생 작업용 표 생성
# - 결정별 허용 (모듈, 단계)는 확정 계획에서 계산해 동의 내용(allowed)에 고정한다. 고정 표를 두지 않는다.
# - B(역할·기준 확인)만으로는 실행 권한을 주지 않는다.
# ==========================================
"""

from datetime import datetime, timedelta, timezone
import hashlib
import json
from uuid import uuid4

from .cli import _current_actor


CONSENT_VERSION = "task_consent_v2"
LIFETIME = timedelta(hours=12)
DECISIONS = frozenset({"A", "C", "D", "T", "R"})


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()


def _allowed(content):
    """동의 내용의 허용 (모듈, 단계) 목록. 비어 있거나 형식이 다르면 None."""
    allowed = content.get("allowed")

    if (type(allowed) is not list or not allowed
            or any(type(item) is not list or len(item) != 2 or any(type(part) is not str or not part for part in item)
                   for item in allowed)):
        return None

    return {tuple(item) for item in allowed}


def allowed_list(pairs):
    """(모듈, 단계) 집합을 동의 내용에 고정할 정렬 목록으로 바꾼다."""
    return sorted([module, stage] for module, stage in set(pairs))


def grant(decision, task_id, content, *, now=None):
    """사용자가 화면에서 확인한 내용만 고정한다. 승인 기록을 발급하지 않는다."""
    if decision not in DECISIONS or type(content) is not dict or not _allowed(content):
        raise ValueError("지원하지 않는 결정 종류 또는 내용입니다.")

    granted = (now or datetime.now(timezone.utc)).replace(microsecond=0)
    return {"version": CONSENT_VERSION, "consent_id": uuid4().hex, "decision": decision, "task_id": task_id,
            "actor_id": _current_actor(), "granted_at": granted.isoformat(),
            "expires_at": (granted + LIFETIME).isoformat(), "content": content, "content_sha256": _digest(content)}


def check(consent, decision, task_id, *, now=None):
    """동의 형식·만료·소유자·내용 해시를 확인한다. 하나라도 다르면 ValueError."""
    now = now or datetime.now(timezone.utc)

    if (type(consent) is not dict or consent.get("version") != CONSENT_VERSION or consent.get("decision") != decision
            or consent.get("task_id") != task_id or consent.get("actor_id") != _current_actor()
            or consent.get("content_sha256") != _digest(consent.get("content"))):
        raise ValueError("유효한 작업 동의가 아닙니다.")

    if not datetime.fromisoformat(consent["granted_at"]) <= now < datetime.fromisoformat(consent["expires_at"]):
        raise ValueError("작업 동의가 만료되었습니다. 다시 확인해 주세요.")

    return consent["content"]


def authorizer(task, decision, module_id, stage, *, expected, bound=None, summary_stage=True, now=None):
    """위임 발급 직전 잠금 안에서 호출할 판정 함수를 만든다.

    expected는 동의 내용과 같아야 하는 현재 값(코드 식별값·계획 지문·자원·후보 ID 등)이다.
    bound는 잠금 안의 최신 승인 미리보기 요약이 가져야 하는 값(동의 후 확보한 snapshot 해시 등)이다.
    """
    bound = {} if bound is None else bound
    content = check(task["consents"].get(decision), decision, task["task_id"], now=now)

    if (module_id, stage) not in (_allowed(content) or set()):
        raise ValueError("이 단계는 해당 결정의 위임 범위가 아닙니다.")

    for key, value in expected.items():
        if content.get(key) != value:
            raise ValueError("실제 실행 범위가 동의한 계획과 다릅니다: " + key)

    def authorize(preview):
        summary = preview["summary"]

        # 진단 승인 요약에는 단계 키가 없다. 그 경우는 호출자가 고정 단계와 계획 서명을 bound로 결합한다.
        if (summary_stage and summary.get("stage") != stage) or summary.get("limits") != content["limits"]:
            raise ValueError("승인 범위의 단계·자원이 동의와 다릅니다.")

        if any(summary.get(key) != value for key, value in bound.items()):
            raise ValueError("승인 범위의 입력이 동의 후 확보한 값과 다릅니다.")

        consent = task["consents"].get(decision)
        check(consent, decision, task["task_id"], now=now)
        # 위임 기록에 남길 사용 근거(추적용). 서명이 아니므로 동의의 진본 인증으로 쓰지 않는다.
        return {"decision": decision, "consent_id": consent["consent_id"], "consent_sha256": consent["content_sha256"],
                "task_id": task["task_id"], "attempt": task.get("attempt", 0)}

    return authorize
