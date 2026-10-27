"""지속 전처리 작업 조정기. 유형 어댑터가 정한 진단·계획·단계를 기존 서비스 API로 진행하고 사용자 결정 지점에서 멈춘다.

데이터 형태·도메인·입력 형식을 모른다. 유형별 기능은 task_adapters 등록부의 어댑터로만 찾는다.
새 worker·운송·계산 경로는 없다. 사용자가 동의한 범위(A·C·D)와 정확히 맞는 단계에만 기존 1회 승인을
위임 발급한다. 실패한 실행은 자동으로 다시 시도하지 않는다.
설계 기준: docs/agents/design/CSV_TASK_AGENT.md 7절
"""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from . import task_adapters, task_consent, task_recommend, task_store
from .input_snapshot import Snapshot, create_snapshot, verify_snapshot
from .llm_config import LocalModelConfig
from .llm_proposal import validate_request_text
from .perf_timing import timed
from .task_plan import (PlanError, TaskProblem, answer_conflicts, apply_answers, changes_between, origins, signature,
                        unanswered_required)
from .worker_contract import WorkerLimits
from .worker_paths import _path


__all__ = ["TaskProblem", "PlanError", "start_task", "diagnose", "resolve_problem", "resume_input", "review_state", "check_plan",
           "preview_plan", "apply_revisable", "confirm_plan", "approve_apply"]
TERMINAL = {"completed", "blocked_input", "failed_runtime", "security_locked", "cancelled"}


def _short(task):
    return task["task_id"][:10]


def _adapter(record):
    return task_adapters.adapter(record["kind"])


def _save(record, root):
    return task_store.save_task(record, expected_revision=record["revision"], root=root)


def _stop(record, problem, root):
    # 내부 단계가 이미 저장한 최신 기록에 이어서 정지 상태만 남긴다.
    record = task_store.load_task(record["task_id"], root=root)
    record.update(phase="needs_attention", problem={"code": problem.code, "message": problem.message,
                                                    "retryable": problem.retryable, "diagnostic": problem.diagnostic,
                                                    "hint": problem.hint, "actions": problem.actions,
                                                    "detail": problem.detail, "error": problem.error,
                                                    "findings": problem.findings})
    return _save(record, root)


def _latest(task_id, root, record):
    """실패 진단용 최신 기록. 읽지 못하면 손에 있는 기록을 쓴다."""
    try:
        return task_store.load_task(task_id, root=root)
    except Exception:
        return record


def _error_info(error, record):
    """처리 실패의 예외 종류와 발생 단계. 예외 원문·경로·값은 담지 않는다. 원인 추적에만 쓴다."""
    import re

    def name(value):
        text = type(value).__name__
        return text if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", text) else "Unknown"

    pending = [step for step, item in record.get("steps", {}).items() if item.get("status") != "completed"]
    cause = error.__cause__ or error.__context__
    return {"type": name(error), "cause_type": None if cause is None else name(cause),
            "stage": pending[-1] if pending else record.get("phase"),
            "step_status": record["steps"][pending[-1]]["status"] if pending else None}


def _check_snapshot(record):
    """진단에 쓴 작업 복사본이 그대로인지 내용 해시로 확인한다. 바뀌면 이후 생성·위임을 하지 않는다."""
    snapshot = record["snapshot"]

    if not verify_snapshot(Path(snapshot["directory"]), Snapshot(**snapshot["snapshot"])):
        raise TaskProblem("SNAPSHOT_CHANGED", "작업에 사용 중인 복사본이 변경되어 중단했습니다. 새 작업으로 시작하세요.")
    checker = getattr(_adapter(record), "check_source", None)
    if checker is not None:
        checker(record)


def _run_name(record, step_id, *, attempt=None):
    attempt = record["attempt"] if attempt is None else attempt
    step_id = _adapter(record).RUN_ALIAS.get(step_id, step_id)
    # 내부 작업 이름은 32자 이하로 고정한다.
    return "t" + _short(record) + "a" + str(attempt) + "-" + step_id[:4]


def _check_names(record):
    """작업 기록의 내부 작업 이름·모듈을 작업 ID와 어댑터 단계표로부터 다시 계산해 대조한다."""
    steps = {step_id: module_id for step_id, module_id, _ in _adapter(record).STEPS}

    for step_id, item in record["steps"].items():
        if (step_id not in steps or item["module_id"] != steps[step_id] or item["attempt"] != record["attempt"]
                or item["run_name"] != _run_name(record, step_id)):
            raise TaskProblem("STATE_UNCLEAR", "작업 기록의 내부 작업 연결이 맞지 않습니다. 결과 확인이 필요합니다.")

    candidate = record.get("candidate")
    review_step = _adapter(record).REVIEW_STEP

    if candidate is not None and candidate["run_name"] != record["steps"].get(review_step, {}).get("run_name"):
        raise TaskProblem("STATE_UNCLEAR", "작업 기록의 적용 대상 연결이 맞지 않습니다. 결과 확인이 필요합니다.")


def _allowed(record, decision):
    """확정 계획의 단계와 카탈로그 승인 단계로 결정별 허용 (모듈, 단계)를 계산한다. 고정 표를 쓰지 않는다."""
    adapter = _adapter(record)
    return task_consent.allowed_list((module_id, stage) for _, module_id, item in adapter.STEPS if item == decision
                                     for stage in adapter.approval_stages(module_id))


# ==========================================
# 작업 시작과 결정 A (입력 진단)
# ==========================================

def start_task(goal, source_path, limits, model, *, root=None, input_generation="legacy"):
    """작업 기록만 만든다. 파일 복사·worker·모델 호출은 결정 A 이후에 한다.

    input_generation은 새 작업의 유형 후보 집합(legacy/current)이다. 유형은 선택한 파일의 형식으로 고정 등록부에서만 고른다.
    """
    validate_request_text(goal)
    WorkerLimits(**limits)
    LocalModelConfig(**model)
    source = _path(Path(source_path).absolute(), directory=False)
    kind = task_adapters.select(source, generation=input_generation)
    record = task_store.create_task(goal, {"path": str(source), "size_bytes": source.stat().st_size}, kind, root=root)
    record.update(limits=dict(limits), model=dict(model))
    return task_store.save_task(record, expected_revision=record["revision"], root=root)


def consent_a_scope(record, action=None):
    """결정 A 화면에 보여 주고 동의로 고정할 내용. 해결 동작을 고르면 그 동작을 포함한 새 동의 내용이 된다."""
    adapter = _adapter(record)
    content = adapter.consent_a_content(record) if action is None else adapter.consent_a_content(record, action)
    return {"source_path": record["source"]["path"], "source_size_bytes": record["source"]["size_bytes"],
            "limits": record["limits"], "model": record["model"]["model"], "model_host": record["model"]["host"], **content}


def diagnose(task_id, *, root=None, requester=None):
    """결정 A를 기록하고 snapshot → 유형별 진단 → 로컬 모델 권장안까지 진행한 뒤 계획 확인에서 멈춘다."""
    with task_store.task_lock(task_id, root=root) as directory:
        record = task_store.load_task(task_id, root=root)
        adapter = _adapter(record)

        if record["phase"] != "created":
            raise ValueError("이미 진단을 시작한 작업입니다.")

        source = _path(Path(record["source"]["path"]), directory=False)

        if source.stat().st_size != record["source"]["size_bytes"]:
            return _stop(record, TaskProblem("SOURCE_CHANGED", "선택 후 원본 크기가 바뀌었습니다. 새 작업으로 다시 시작하세요."), root)

        record["consents"]["A"] = task_consent.grant("A", task_id, consent_a_scope(record))
        record["phase"] = "diagnosing"
        record = _save(record, root)
        context = SimpleNamespace(save=lambda value: _save(value, root), authorizer=task_consent.authorizer)

        try:
            snapshot = create_snapshot(source, directory / "snapshot", max_input_bytes=record["source"]["size_bytes"])
            record["snapshot"] = {"directory": str(directory / "snapshot"), "snapshot": asdict(snapshot)}
            record = _save(record, root)
            record = adapter.diagnose(record, directory, context)
        except TaskProblem as problem:
            return _stop(record, problem, root)
        except Exception:
            return _stop(record, TaskProblem("DIAGNOSTIC_FAILED", "입력 진단을 완료하지 못했습니다. 자동 재시도하지 않습니다."), root)

        if record["phase"] == "awaiting_input":
            return record
        return _recommend(record, adapter, requester, root)


def resolve_problem(task_id, action, *, root=None, requester=None):
    """차단 화면에서 사용자가 고른 어댑터의 해결 동작을 새 결정 A 동의로 진행하고 진단·권장안까지 이어 간다.

    입력은 검증된 작업 복사본으로 고정한다. 복사본을 만든 뒤 원본이 이동·삭제·변경돼도 복사본이 온전하면 진행하고,
    원본 경로는 출처 정보로만 둔다. 해결할 수 없는 요청은 새 동의·실행 기록을 만들기 전에 거부한다.
    이전 진단·동의 연결은 history에 남기며 기존 실행 기록과 소비된 승인은 지우거나 다시 쓰지 않는다.
    실패하면 자동으로 다시 시도하지 않는다.
    """
    with task_store.task_lock(task_id, root=root) as directory:
        record = task_store.load_task(task_id, root=root)
        adapter = _adapter(record)
        problem = record.get("problem") or {}

        # 동의 발급 전 검사: 작업 단계, 기록에 제시된 동작, 어댑터의 해결 가능 여부(이미 정비한 입력 등).
        if record["phase"] != "needs_attention" or action not in (problem.get("actions") or []):
            raise ValueError("이 작업에서 고를 수 있는 해결 동작이 아닙니다.")

        adapter.check_resolution(record, action)

        try:
            _check_snapshot(record)
        except TaskProblem as stopped:
            return _stop(record, stopped, root)

        consent = task_consent.grant("A", task_id, consent_a_scope(record, action))
        entry = {"action": action, "status": "started",
                 "previous_diagnostic_run": (record.get("diagnostic") or {}).get("run_name"),
                 "previous_consent_id": record["consents"]["A"]["consent_id"], "consent_id": consent["consent_id"],
                 "snapshot_before_sha256": record["snapshot"]["snapshot"]["sha256"]}
        record["consents"]["A"] = consent
        record["history"] = [*record.get("history", []), entry]
        record.update(phase="diagnosing", problem=None)
        record = _save(record, root)
        context = SimpleNamespace(save=lambda value: _save(value, root), authorizer=task_consent.authorizer)

        try:
            record = adapter.resolve(record, directory, context, action)
        except TaskProblem as stopped:
            return _stop(record, stopped, root)
        except Exception:
            return _stop(record, TaskProblem("DIAGNOSTIC_FAILED", "해결 동작을 완료하지 못했습니다. 자동 재시도하지 않습니다."), root)

        record["history"][-1].update(status="completed", **adapter.resolution_facts(record, action))
        if record["phase"] == "awaiting_input":
            return _save(record, root)
        return _recommend(record, adapter, requester, root)


def resume_input(task_id, selection, *, expected_revision, expected_inspection_id, root=None, requester=None):
    """검증된 입력 해석 대기를 새 A 개정으로 재개한다. 오래된 제출은 기록 생성 전에 거부한다."""
    with task_store.task_lock(task_id, root=root) as directory:
        record = task_store.load_task(task_id, root=root)
        current = record.get("input_selection") or {}
        if (record["phase"] != "awaiting_input" or type(expected_revision) is not int
                or expected_revision != record["revision"] or expected_inspection_id != current.get("inspection_id")):
            raise ValueError("대기 중인 입력 해석 또는 revision이 바뀌었습니다.")
        adapter = _adapter(record)
        adapter.check_input_selection(record, selection, directory)
        task_consent.check(record["consents"].get("A"), "A", task_id)
        content = {"source_path": record["source"]["path"], "source_size_bytes": record["source"]["size_bytes"],
                   "limits": record["limits"], "model": record["model"]["model"], "model_host": record["model"]["host"],
                   **adapter.input_consent_content(record, selection)}
        record["consents"]["A"] = task_consent.grant("A", task_id, content)
        record.update(phase="diagnosing", problem=None)
        record = _save(record, root)
        context = SimpleNamespace(save=lambda value: _save(value, root), authorizer=task_consent.authorizer)
        try:
            record = adapter.resume_input(record, directory, context, selection)
        except TaskProblem as problem:
            return _stop(record, problem, root)
        except Exception:
            return _stop(record, TaskProblem("INPUT_PREPARATION_FAILED", "입력 준비를 완료하지 못했습니다. 자동 재시도하지 않습니다."), root)
        if record["phase"] == "awaiting_input":
            return record
        return _recommend(record, adapter, requester, root)


def _recommend(record, adapter, requester, root):
    """검증된 진단 결과로 로컬 모델 권장안을 한 번 받고 계획 확인에서 멈춘다."""
    options = {} if requester is None else {"requester": requester}

    try:
        record["draft"] = task_recommend.request(
            record["goal"], adapter.recommendation_input(record), adapter.base_plan(record), adapter.plan_fields(record),
            lambda plan: adapter.validate_plan(record, plan), LocalModelConfig(**record["model"]),
            on_rejected=lambda draft: adapter.on_rejected(record, draft), **options)
    except Exception:
        # 권장안 준비 중 예상하지 못한 오류는 다시 요청하지 않고 기본 계획으로 이어 간다.
        try:
            record["draft"] = {"summary": [], "notes": [], "changes": [], "questions": [], "dropped_questions": 0,
                               "plan": adapter.base_plan(record), "source": "default", "code": "MODEL_REQUEST_FAILED"}
        except Exception:
            return _stop(record, TaskProblem("DIAGNOSTIC_FAILED", "진단 결과로 기본 계획을 만들지 못했습니다. 새 작업으로 시작하세요."), root)

    record["phase"] = "awaiting_review"
    return _save(record, root)


# ==========================================
# 계획 확인 (권장 변경·질문 답변·전체 검증) → 결정 C 단계
# ==========================================

def _shown_plan(record):
    """확인 화면에 보여 준 계획. 재시도면 직전에 확정한 계획, 아니면 권장안을 합친 계획."""
    confirmed = record.get("confirmed")
    return confirmed["plan"] if confirmed and "plan" in confirmed else record["draft"]["plan"]


def review_state(record):
    """확인 화면이 그릴 내용. 질문은 표시한 계획에서 어댑터가 다시 계산하고 모델 질문(안내용)을 덧붙인다."""
    adapter, plan = _adapter(record), _shown_plan(record)
    asked = adapter.questions(record, plan)
    owned = {item["path"] for item in asked if item["path"] is not None}
    # 프로그램 질문이 이미 다루는 항목의 모델 질문은 보이지 않는다(같은 항목에 출처가 다른 두 질문이 생기지 않게).
    questions = asked + [item for item in record["draft"].get("questions", []) if item["path"] is None or item["path"] not in owned]
    # 권장 변경마다 출처와 어댑터의 확인 안내를 붙인다. 형식 검사 통과는 내용의 타당성을 뜻하지 않는다.
    recommended = [dict(item, check=adapter.change_check(record, item) if hasattr(adapter, "change_check") else None)
                   for item in record["draft"].get("changes", [])]
    # 옵션 설명은 표시한 계획 기준이다. 양식·답변을 반영한 최종 계획의 설명은 preview_plan이 다시 만든다.
    guide = (adapter.option_guide(record, plan, origins(adapter.base_plan(record), recommended, plan, plan, []))
             if hasattr(adapter, "option_guide") else [])
    return {"plan": plan, "fields": adapter.plan_fields(record), "questions": questions, "findings": _findings(record),
            "problem": problem_view(record),
            "changes": changes_between(adapter.base_plan(record), record["draft"]["plan"]),
            "recommended": recommended, "guide": guide}


def revisable(record):
    """계획을 고쳐 다시 확인할 수 있는 정지인지.

    재시도 가능으로 기록된 정지에 더해, 결정 C 자동 단계가 입력 분류(INPUT_INVALID)로 끝난 실패도 포함한다. 결정 D 이후·
    상태 불명·환경 실패(권한·입출력·런타임)는 포함하지 않는다. 같은 계획을 자동 재실행하지 않으며 새 동의(결정 C)가 필요하다.
    """
    problem = record.get("problem") or {}

    if record["phase"] != "needs_attention":
        return False

    if problem.get("retryable"):
        return True

    pending = [item for item in record.get("steps", {}).values() if item.get("status") != "completed"]
    return (problem.get("code") == "STEP_FAILED" and (problem.get("diagnostic") or {}).get("code") == "INPUT_INVALID"
            and len(pending) == 1 and pending[0].get("decision") == "C" and pending[0].get("status") == "created"
            and "D" not in record.get("consents", {}))


def apply_revisable(record):
    """적용 확인(결정 D 전)에서 계획을 고쳐 새 시도로 진행할 수 있는지. 후보만 계산했고 적용 동의가 없을 때만 참이다.

    처리 영향 경고를 본 사용자가 설정을 직접 고치는 경로다. 설정을 자동으로 바꾸지 않으며 새 동의(결정 C)가 필요하다.
    """
    return (record["phase"] == "awaiting_apply" and record.get("candidate") is not None and "D" not in record.get("consents", {})
            and all(item.get("decision") == "C" and item.get("status") == "completed" for item in record.get("steps", {}).values()))


def problem_view(record):
    """화면에 보일 문제. 이전 형식 기록에 없는 지적·원인 안내는 확인된 진단으로 다시 계산한다. 기록은 바꾸지 않는다."""
    problem = record.get("problem")

    if not problem:
        return None

    view = dict(problem, findings=_findings(record), revisable=revisable(record))

    # 이전 형식 기록은 분할의 모든 입력 차단을 품질 정지로 저장했다. 확인된 다른 원인이면 그 원인·안내를 보인다.
    if problem.get("code") == "QUALITY_REVIEW" and problem.get("retryable") and not problem.get("findings"):
        refreshed = _refreshed(record)

        if refreshed is not None and refreshed.code != "QUALITY_REVIEW":
            view.update(code=refreshed.code, message=refreshed.message, detail=refreshed.detail, findings=refreshed.findings)

    # 기록에 두지 않는 표시용 세부(예: 구간별 빠진 라벨)는 어댑터가 검증된 근거에서 다시 읽는다. 기록은 바꾸지 않는다.
    display = getattr(_adapter(record), "display_findings", None)

    if display is not None and view["findings"]:
        try:
            view["findings"] = display(record, view["findings"])
        except Exception:
            pass

    if view.get("hint") is None and view.get("diagnostic"):
        pending = [step for step, item in record.get("steps", {}).items() if item.get("status") != "completed"]

        try:
            view["hint"] = _adapter(record).failure_hint(record, pending[-1], view["diagnostic"]) if pending else None
        except Exception:
            view["hint"] = None

    return view


def _findings(record):
    """재시도 가능한 정지의 확인된 지적. 기록에 없으면(이전 형식) 어댑터가 검증된 결과로 다시 읽는다. 읽기만 한다."""
    problem = record.get("problem") or {}

    if not problem.get("retryable"):
        return []

    if problem.get("findings"):
        return problem["findings"]

    refreshed = _refreshed(record)
    return [] if refreshed is None else refreshed.findings


def _refreshed(record):
    """재시도 가능한 정지를 어댑터가 검증된 결과로 다시 분류한 문제. 읽기만 하며 확인하지 못하면 None이다."""
    pending = [step for step, item in record.get("steps", {}).items() if item.get("status") != "completed"]

    try:
        return _adapter(record).blocked_problem(record, pending[-1]) if pending else None
    except Exception:
        return None


def check_plan(record, plan, answers):
    """사용자 수정 계획에 답변을 반영하고 필수 질문·계획 전체를 검증한다. 화면 선검사와 서버 확인이 같이 쓴다."""
    adapter = _adapter(record)
    fields = adapter.plan_fields(record)
    shown = review_state(record)["questions"]
    conflicts = answer_conflicts(_shown_plan(record), plan, shown, answers or {})

    # 양식에서 직접 바꾼 값과 답변이 다르면 어느 쪽도 자동으로 고르지 않는다. 사용자가 한쪽을 골라 다시 확인한다.
    if conflicts:
        raise PlanError("ANSWER_FORM_CONFLICT", "계획 양식에서 바꾼 값과 질문 답변이 같은 항목에 서로 다른 값을 정했습니다.",
                        subjects=[item["path"] for item in conflicts])

    final, applied = apply_answers(plan, shown, answers or {}, fields)
    required = _unresolved(adapter, record, final, applied)

    if required:
        raise PlanError("REQUIRED_QUESTION_UNANSWERED", "답해야 하는 질문이 남아 있습니다.",
                        subjects=[item["id"] for item in required])

    adapter.validate_plan(record, final)
    return final, applied


def _unresolved(adapter, record, final, applied):
    """최종 계획에서 다시 계산한 필수 질문 중 해결되지 않은 것. 확정 판정(check_plan)과 화면 안내가 같은 계산을 쓴다."""
    return unanswered_required(adapter.questions(record, final), applied)


def unresolved_questions(record, plan, answers):
    """제출한 계획·답변 기준으로 남은 필수 질문(문장·관련 항목·선택지). 화면 안내용이며 판정·저장·실행은 하지 않는다.

    표시한 계획에서 계산한 질문 목록에 없던 질문(제출한 계획에서 새로 생긴 질문)도 포함한다. 각 항목의 shown은 화면 위
    질문 목록에서 답할 수 있는지를 뜻한다. 충돌·답변 적용 오류는 check_plan과 같은 PlanError로 올린다.
    """
    adapter = _adapter(record)
    shown = review_state(record)["questions"]
    final, applied = apply_answers(plan, shown, answers or {}, adapter.plan_fields(record))
    ids = {item["id"] for item in shown}
    return [dict(item, shown=item["id"] in ids) for item in _unresolved(adapter, record, final, applied)]


def preview_plan(record, plan, answers):
    """확정 전에 보여 줄 최종 계획. check_plan과 같은 검증을 거친 값이며, digest를 confirm_plan에 넘겨 같은 계획만 확정한다.

    검증 실패는 PlanError로 그대로 올린다. 실행·저장·모델 호출은 없다. 옵션 설명은 어댑터가 있으면 붙인다.
    """
    adapter = _adapter(record)
    final, applied = check_plan(record, plan, answers)
    state = review_state(record)
    source = origins(adapter.base_plan(record), state["recommended"], state["plan"], plan, applied)
    guide = adapter.option_guide(record, final, source) if hasattr(adapter, "option_guide") else []
    return {"plan": final, "answers": applied, "origins": source, "digest": _preview_digest(record, final),
            "plan_digest": signature(final), "changes": changes_between(state["plan"], final), "guide": guide}


def _preview_digest(record, final):
    """미리보기 결합값. 최종 계획과 미리 본 시점의 시도 번호·기록 revision을 묶는다.

    확정이 한 번 진행되면 revision이 바뀌므로 같은 요청을 다시 보내도(다른 탭·재전송) 새 시도를 만들지 않는다.
    """
    return signature({"plan": final, "attempt": record["attempt"], "revision": record["revision"]})


def confirm_plan(task_id, plan, answers=None, *, root=None, expected_digest=None):
    """확인한 계획을 고정하고 결정 C 단계까지 자동 진행한 뒤 결정 D에서 멈춘다. 검증에 실패하면 아무것도 만들지 않는다.

    expected_digest가 있으면 화면에서 미리 본 최종 계획·시도·기록 revision이 서버 값과 같을 때만 진행한다.
    적용 확인 단계(결정 D 전)에서 계획을 고칠 때는 expected_digest가 반드시 있어야 한다.
    """
    with task_store.task_lock(task_id, root=root) as directory:
        record = task_store.load_task(task_id, root=root)
        adapter = _adapter(record)
        problem = record.get("problem") or {}

        if record["phase"] != "awaiting_review" and not revisable(record) and not apply_revisable(record):
            raise ValueError("계획을 확인할 단계가 아닙니다.")

        try:
            _check_snapshot(record)
        except TaskProblem as problem:
            return _stop(record, problem, root)

        final, applied = check_plan(record, plan, answers)

        if (expected_digest is None and record["phase"] == "awaiting_apply") or (
                expected_digest is not None and expected_digest != _preview_digest(record, final)):
            raise PlanError("PLAN_PREVIEW_CHANGED", "미리 본 최종 계획과 확정하려는 계획 또는 작업 상태가 다릅니다. 최종 계획을 다시 확인하세요.")

        if record["steps"]:
            # 이전 시도의 단계·위임 기록·결정 C 동의는 지우지 않고 보존한다. 새 시도는 새 이름·새 동의로만 진행한다.
            # 보존과 현재 단계 비우기를 시도 번호와 같은 저장에 묶는다. 준비 중 실패해도 옛 단계·후보가 현재 시도로 남지 않는다.
            record.setdefault("attempts", []).append({
                "attempt": record["attempt"], "steps": record["steps"], "consent_c": record["consents"].get("C"),
                "problem_code": problem.get("code"), "plan_signature": (record.get("confirmed") or {}).get("plan_signature")})
            record.update(steps={}, candidate=None, problem=None, phase="awaiting_review")

        attempt = record["attempt"] + 1
        label = "task_" + _short(record) + "_" + str(attempt)
        # 파일을 만들기 전에 시도 번호를 저장한다. 중간 실패 뒤 다시 확인해도 같은 이름과 충돌하지 않는다.
        record["attempt"] = attempt
        record = _save(record, root)
        prepared = adapter.prepare(record, directory, final, label)
        advisory = [item["id"] for item in adapter.questions(record, final)]
        record.update(problem=None, steps={}, candidate=None,
                      confirmed={"plan": final, "answers": applied, "open_questions": advisory, **prepared})
        record["consents"]["C"] = task_consent.grant("C", task_id, {
            "plan_signature": prepared["plan_signature"], "plan_digest": signature(final), "answers": applied,
            "limits": record["limits"], "attempt": attempt, "snapshot_sha256": record["snapshot"]["snapshot"]["sha256"],
            "auto_steps": [step for step, _, decision in adapter.STEPS if decision == "C"],
            "allowed": _allowed(record, "C"), **adapter.consent_c_content(record)})
        record["phase"] = "processing"
        record = _save(record, root)

        try:
            for step_id, module_id, decision in adapter.STEPS:
                if decision == "C":
                    record = _run_step(record, directory, step_id, module_id, decision, root)

            record["candidate"] = adapter.review(record)
        except TaskProblem as problem:
            return _stop(record, problem, root)
        except Exception as error:
            latest = _latest(task_id, root, record)
            return _stop(record, TaskProblem("PROCESSING_FAILED", "처리 단계를 완료하지 못했습니다. 자동 재시도하지 않습니다.",
                                             error=_error_info(error, latest)), root)

        record["phase"] = "awaiting_apply"
        return _save(record, root)


def _run_step(record, directory, step_id, module_id, decision, root):
    """어댑터가 만든 단계를 기존 승인·실행 API로 끝까지 진행한다. 이름은 부작용 전에 예약한다."""
    adapter = _adapter(record)

    if step_id in record["steps"]:
        raise TaskProblem("STATE_UNCLEAR", "이 단계는 이미 준비된 기록이 있습니다. 결과 확인이 필요합니다.")

    _check_names(record)
    _check_snapshot(record)
    name = _run_name(record, step_id)
    record["steps"][step_id] = {"module_id": module_id, "run_name": name, "status": "reserved",
                                "attempt": record["attempt"], "decision": decision, "delegations": {}}
    record = _save(record, root)

    with timed("create_step", step_id):
        adapter.create_step(record, directory, step_id, name)

    record["steps"][step_id]["status"] = "created"
    record = _save(record, root)
    adapter.check_lineage(record)
    record = _drive(record, step_id, module_id, decision, name, root)
    record["steps"][step_id]["status"] = "completed"
    return _save(record, root)


def _drive(record, step_id, module_id, decision, name, root):
    adapter = _adapter(record)
    service = adapter.service(module_id)

    for _ in range(40):
        with timed("status", step_id):
            state = service.status(name)

        if state["status"] in TERMINAL:
            if state["status"] == "completed":
                return record

            blocked = adapter.blocked_problem(record, step_id) if state["status"] == "blocked_input" else None

            if blocked is not None:
                raise blocked

            raise _step_failure(record, adapter, service, step_id, name)

        actions = {item["id"] for item in state["actions"]}

        if "approve" in actions:
            _check_snapshot(record)
            expected = {"plan_signature": record["confirmed"]["plan_signature"], "limits": record["limits"],
                        "allowed": _allowed(record, decision), **adapter.expected(record, decision)}

            if decision == "C":
                # 이전 시도의 동의·계획으로 되돌린 기록을 새 시도 이름으로 다시 실행하지 않는다.
                expected["snapshot_sha256"] = record["snapshot"]["snapshot"]["sha256"]
                expected["attempt"] = record["attempt"]

            authorize = task_consent.authorizer(record, decision, module_id, state["stage"], expected=expected)
            # 측정 기록은 호출 순서를 바꾸지 않는다(미리보기를 먼저 계산한 뒤 위임 기록).
            with timed("preview_approval", step_id):
                preview = service.preview_approval(name)
            with timed("record_delegated_approval", step_id):
                result = service.record_delegated_approval(name, preview, authorize=authorize)
            record["steps"][step_id]["delegations"][state["stage"]] = {
                "consent_id": record["consents"][decision]["consent_id"], "scope_sha256": result["scope_sha256"]}
            record = _save(record, root)
            with timed("status", step_id):
                current = service.status(name)
            with timed("next", step_id):
                service.next(name, expected_stage=current["stage"], expected_revision=current["revision"],
                             approval_id=result["approval_id"])
        elif "next" in actions:
            with timed("next", step_id):
                service.next(name, expected_stage=state["stage"], expected_revision=state["revision"])
        else:
            raise TaskProblem("STATE_UNCLEAR", "처리 단계 상태가 진행되지 않았습니다. 결과 확인이 필요합니다.")

    raise TaskProblem("STATE_UNCLEAR", "처리 단계 상태가 진행되지 않았습니다. 결과 확인이 필요합니다.")


class RecoveryError(ValueError):
    """완료 기록 복구를 거부한 확인된 원인. 기록은 바꾸지 않는다."""

    def __init__(self, code, step=None):
        super().__init__(code)
        self.code, self.step = code, step


def recoverable(record):
    """모든 처리 단계가 완료로 기록됐지만 완료 저장 전에 멈춘 적용인지. 판정만 하며 실행 근거는 recover_completion이 다시 검증한다."""
    if record["phase"] != "applying" or record.get("report") is not None or record.get("problem"):
        return False

    steps = record.get("steps", {})
    expected = {step_id for step_id, _, _ in _adapter(record).STEPS}
    return (set(steps) == expected and all(item.get("status") == "completed" for item in steps.values())
            and {"C", "D"} <= set(record.get("consents", {})))


def recover_completion(task_id, *, root=None):
    """worker 재실행·새 승인 없이 완료 기록을 복구한다. 기존 산출물과 승인 소비 증거를 먼저 모두 검증한다.

    - 기록의 작업 이름·입력 snapshot·단계 연결을 다시 확인한다.
    - 결정 C·D 동의의 형식·소유자·내용 해시를 발급 시점 기준으로 확인한다. 새 실행에는 쓰지 않는다.
    - 단계마다 모듈·결정·시도 번호가 계획과 같고, 위임 기록이 해당 결정의 동의에서 나왔으며 실행 상태가 완료인지 본다.
    - 확인한 적용 대상 서명이 그대로인지 본다.
    - 보고서는 검증된 결과(커밋된 승인 작업·결과 파일 해시·도메인 검증)로만 다시 조립한다. 같은 내용의 기존 파일은 그대로 참조한다.
    하나라도 다르면 기록을 바꾸지 않고 거부한다.
    """
    from datetime import datetime
    from .task_report import write_report

    with task_store.task_lock(task_id, root=root) as directory:
        record = task_store.load_task(task_id, root=root)

        if not recoverable(record):
            raise RecoveryError("RECOVERY_NOT_APPLICABLE")

        adapter = _adapter(record)

        try:
            _check_names(record)
            _check_snapshot(record)
            adapter.check_lineage(record)
        except (TaskProblem, ValueError) as error:
            raise RecoveryError("RECOVERY_RECORD_MISMATCH") from error

        for decision in ("C", "D"):
            consent = record["consents"][decision]

            try:
                content = task_consent.check(consent, decision, task_id, now=datetime.fromisoformat(consent["granted_at"]))
            except (ValueError, TypeError, KeyError) as error:
                raise RecoveryError("RECOVERY_CONSENT_INVALID") from error

            if content.get("plan_signature") != record["confirmed"]["plan_signature"]:
                raise RecoveryError("RECOVERY_CONSENT_INVALID")

            # C는 입력 snapshot·시도 번호에, D는 사용자가 본 적용 대상 서명에 결합되어 있어야 한다. 서명 키 이름은 어댑터마다 다르다.
            if decision == "C" and (content.get("snapshot_sha256") != record["snapshot"]["snapshot"]["sha256"]
                                    or content.get("attempt") != record["attempt"]):
                raise RecoveryError("RECOVERY_CONSENT_INVALID")

            if decision == "D" and record["candidate"]["signature"] not in content.values():
                raise RecoveryError("RECOVERY_CONSENT_INVALID")

        verified = {}

        for step_id, module_id, decision in adapter.STEPS:
            item = record["steps"][step_id]
            consent_id = record["consents"][decision]["consent_id"]

            if (item["module_id"] != module_id or item["decision"] != decision or item["attempt"] != record["attempt"]
                    or not item["delegations"] or any(value["consent_id"] != consent_id for value in item["delegations"].values())):
                raise RecoveryError("RECOVERY_RECORD_MISMATCH", step_id)

            try:
                # 어댑터가 결과 검증을 제공하면 그것으로 확인한다(실행 뒤 코드가 바뀐 단계는 읽기 전용 검증). 아니면 실행 상태를 본다.
                if hasattr(adapter, "step_completed"):
                    verified[step_id] = adapter.step_completed(record, step_id)
                elif adapter.service(module_id).status(item["run_name"])["status"] == "completed":
                    verified[step_id] = "current_code"
                else:
                    raise ValueError("완료되지 않은 단계입니다.")
            except ValueError as error:
                raise RecoveryError("RECOVERY_STEP_UNVERIFIED", step_id) from error

        try:
            if adapter.review_signature(record) != record["candidate"]["signature"]:
                raise RecoveryError("RECOVERY_CANDIDATE_CHANGED")

            record["report"] = write_report(record, directory, reuse_identical=True)
        except RecoveryError:
            raise
        except (TaskProblem, ValueError, OSError) as error:
            raise RecoveryError("RECOVERY_RESULT_UNVERIFIED") from error

        record["recovered"] = {"at": task_store._now(), "reason": "COMPLETION_NOT_SAVED", "worker_runs": 0, "new_approvals": 0,
                               "read_only_steps": sorted(step for step, how in verified.items() if how == "recorded_code")}
        record["phase"] = "completed"
        return _save(record, root)


def _step_failure(record, adapter, service, step_id, name):
    """확인된 worker 분류와 어댑터의 추정 원인을 나눠 담는다. 읽지 못하면 비워 두고 추정하지 않는다."""
    try:
        diagnostic = service.worker_diagnostic(name) if hasattr(service, "worker_diagnostic") else None
    except Exception:
        diagnostic = None

    try:
        hint = adapter.failure_hint(record, step_id, diagnostic)
    except Exception:
        hint = None

    return TaskProblem("STEP_FAILED", "처리 단계가 완료되지 않았습니다(" + adapter.STEP_LABELS.get(step_id, step_id) + "). "
                       "자동 재시도하지 않습니다.", diagnostic=diagnostic, hint=hint)


# ==========================================
# 결정 D (적용) → 적용·전달 → 통합 결과
# ==========================================

def approve_apply(task_id, *, root=None, expected_signature=None, expected_attempt=None):
    """사용자가 본 적용 대상에 결합한 D 동의로 남은 단계를 진행하고 통합 결과 보고서를 만든다.

    화면이 본 후보 서명·시도 번호를 넘기면 서버 값과 같을 때만 진행한다(다른 탭에서 계획을 고친 뒤의 오래된 화면 거부).
    """
    from .task_report import write_report

    with task_store.task_lock(task_id, root=root) as directory:
        record = task_store.load_task(task_id, root=root)
        adapter = _adapter(record)

        if record["phase"] != "awaiting_apply" or record["candidate"] is None:
            raise ValueError("적용 대상을 확인한 뒤에만 적용할 수 있습니다.")

        if ((expected_signature is not None and expected_signature != record["candidate"]["signature"])
                or (expected_attempt is not None and expected_attempt != record["attempt"])):
            raise ValueError("화면에서 확인한 적용 대상이 현재 작업과 다릅니다. 화면을 새로 고쳐 다시 확인하세요.")

        try:
            _check_names(record)
            _check_snapshot(record)
            adapter.check_lineage(record)
        except TaskProblem as problem:
            return _stop(record, problem, root)

        if adapter.review_signature(record) != record["candidate"]["signature"]:
            return _stop(record, TaskProblem("CANDIDATE_CHANGED", "확인한 적용 대상이 바뀌었습니다. 적용하지 않습니다."), root)

        # 저장 실행안을 열 수 없으면(코드 변경·파일 변경) 결정 D를 발급하지 않고 멈춘다. 계획 수정으로 새 시도·새 동의를 준비한다.
        checker = getattr(adapter, "check_saved_plan", None)

        try:
            if checker is not None:
                checker(record, directory)
        except (ValueError, OSError):
            return _stop(record, TaskProblem("PLAN_STALE", "저장한 실행안을 현재 코드로 열 수 없거나 확정한 계획과 달라 적용하지 "
                                             "않았습니다. '계획 수정'으로 새 시도를 준비하세요(새 동의 필요).", retryable=True), root)

        record["consents"]["D"] = task_consent.grant("D", task_id, {
            "plan_signature": record["confirmed"]["plan_signature"], "limits": record["limits"],
            "auto_steps": [step for step, _, decision in adapter.STEPS if decision == "D"],
            "allowed": _allowed(record, "D"), **adapter.consent_d_content(record)})
        record["phase"] = "applying"
        record = _save(record, root)

        try:
            for step_id, module_id, decision in adapter.STEPS:
                if decision == "D":
                    record = _run_step(record, directory, step_id, module_id, decision, root)

            record["report"] = write_report(record, directory)
        except TaskProblem as problem:
            return _stop(record, problem, root)
        except Exception as error:
            latest = _latest(task_id, root, record)
            return _stop(record, TaskProblem("APPLY_FAILED", "적용·전달을 완료하지 못했습니다. 자동 재시도하지 않습니다.",
                                             error=_error_info(error, latest)), root)

        record["phase"] = "completed"
        return _save(record, root)
