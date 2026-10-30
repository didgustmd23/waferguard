"""표형 v2 time 계획의 시간 결측 복원 정책. 미리보기는 결정 T, 적용은 미리보기 결과 지문에 결합한 결정 R로만 실행한다.

공통 조정기는 계획 동작 hook(plan_actions·plan_action_consent·plan_action)만 호출한다. 원본 파일과 기존 생성 표는 바꾸지 않고,
적용은 격리 실행이 새 작업용 표(파생 표)를 만든다. 자동 보간·외삽·행 삭제·재시도는 없다.
작업 기록·오류 문구에는 원본 시간값을 넣지 않는다(복원 시각·행 번호·건수·지문만 둔다).
"""

from datetime import datetime
from pathlib import Path

from src.preprocessing.tabular_bundle import verification_scope
from src.preprocessing.time_restore import CODES as TIME_CODES, TimeRestoreError, TimeRestoreRules, format_time
from . import tabular_input_flow as input_flow
from .protected_store import StoreUnavailable
from .tabular_source_contract import TabularSourceContract
from .tabular_time_service import STORE_MESSAGES, TabularTimeService, code_id
from .tabular_time_worker import MAX_OUTPUT_BYTES, POLICY_VERSION
from .task_plan import TaskProblem, signature
from .worker_runtime import MESSAGES as RUNTIME_MESSAGES, NOTICE as RUNTIME_NOTICE, RuntimeUnavailable, current_runtime_id


PREVIEW, APPLY = "time_preview", "time_apply"
# 동작 → (서비스 action, 위임 도구 ID, 결정). 도구 ID·단계는 동의 내용 allowed와 같은 문자열이다.
ACTIONS = {PREVIEW: ("preview", "tabular_time_preview", "T"), APPLY: ("apply", "tabular_time_apply", "R")}
STAGE = "ingest"
MAX_SHOWN = 200
TITLE = "시간 결측 복원"
STOP_NEXT = {
    "TIME_EDGE_MISSING": "첫 행·마지막 행의 시간값을 원본에서 채우거나 해당 행을 뺀 파일로 새 작업을 시작하세요.",
    "TIME_DUPLICATE": "같은 시각이 두 번 이상 나오는 행을 원본에서 확인하세요. 행마다 주기상 한 시점이어야 합니다.",
    "TIME_FREQUENCY_MISMATCH": "복원 주기(매일·평일·고정 간격)와 간격이 실제 기록과 맞는지, 행 순서가 시간 순서인지 확인해 계획 양식에서 고치세요.",
    "TIME_PARSE_FAILED": "시간 형식이 실제 값의 표기와 같은지 확인해 계획 양식에서 고치세요(예: %Y-%m-%d %H:%M).",
    "TIME_AMBIGUOUS": "서머타임 전환 구간처럼 현지 시각이 하나로 정해지지 않습니다. 시간대를 확인하거나 오프셋(%z)이 있는 값으로 기록된 파일을 쓰세요.",
    "TIME_TIMEZONE_CONFLICT": "시간 컬럼 값에 이미 시간대가 있습니다. 계획 양식의 시간대를 비우세요.",
    "TIME_TIMEZONE_REQUIRED": "시간 컬럼 값에 시간대가 없습니다. 계획 양식에 IANA 시간대를 지정하세요.",
}
STOP_DEFAULT = "계획 양식의 시간 설정(형식·시간대·주기)과 원본의 시간 컬럼을 확인하세요."
FAILURES = {
    "TIME_PREVIEW_FAILED": "복원 미리보기 실행이 완료되지 않았습니다. 작업용 표는 바뀌지 않았고 자동 재시도하지 않습니다. 다시 미리보기할 수 있습니다.",
    "TIME_APPLY_FAILED": "복원 적용 실행이 완료되지 않았습니다. 작업용 표는 바뀌지 않았고 자동 재시도하지 않습니다. 새 미리보기부터 다시 진행하세요.",
    "TIME_PREVIEW_CHANGED": "적용 시점에 다시 판정한 결과가 확인한 미리보기와 달라 적용하지 않았습니다. 작업용 표는 바뀌지 않았습니다. 새 미리보기부터 다시 진행하세요.",
}


def _source(record):
    return TabularSourceContract.from_dict(record["input_selection"]["source"])


def _rules(final):
    return TimeRestoreRules(**final["time"])


def _current(record, final):
    """현재 확정 대상 계획·입력 출처에 결합된 시간 복원 기록. 계획·출처가 다르면 None(이전 미리보기·확인은 쓰지 않음)."""
    current = record.get("time_restore")
    if (not current or current.get("plan_digest") != signature(final)
            or current.get("source_contract_id") != _source(record).source_id):
        return None
    return current


def _status(record, final):
    current = _current(record, final)
    if current is not None:
        return current["status"]
    return "stale" if record.get("time_restore") else "none"


def _allowed(record, final):
    status = _status(record, final)
    if status in ("none", "stale", "failed", "previewing", "applying"):
        return PREVIEW
    if status == "previewed" and _current(record, final)["preview"]["restored_count"] > 0:
        return APPLY
    # 적용 완료·복원 0행이어도 새 미리보기(새 시도·새 결정 T)를 허용한다. 확정 직전 증거 재확인이 실패했을 때
    # 안내한 다음 행동(새 미리보기)을 실제로 할 수 있게 한다. 이전 실행 폴더·기록은 history로 보존한다.
    if status in ("previewed", "applied"):
        return PREVIEW
    return None


def _ready_state(record, final):
    """현재 계획에 결합된 결과로 바로 확정할 수 있는 상태(적용 완료 또는 복원 0행)인지."""
    status = _status(record, final)
    return status == "applied" or (status == "previewed" and _current(record, final)["preview"]["restored_count"] == 0)


def _rules_text(time):
    labels = {"daily": "매일 같은 현지 시각", "weekdays": "평일(월~금) 같은 현지 시각", "fixed": "고정 간격"}
    frequency = labels.get(time["frequency"], time["frequency"])
    if time["frequency"] == "fixed":
        frequency += f" {time['interval_minutes']}분"
    return (f"규칙: 시간 컬럼 {time['column']} · 형식 {time['timestamp_format'] or '(시각 자료형)'} · 시간대 "
            f"{time['timezone'] or '(값의 오프셋)'} · 주기 {frequency} · 행 순서 확인함")


def _shown_rows(preview, rules):
    rows = []
    for row, text in zip(preview["rows"], preview["times"]):
        try:
            written = str(format_time(datetime.fromisoformat(text), rules))
        except (TimeRestoreError, ValueError):
            written = "(표기 확인 불가)"
        rows.append({"데이터 행(1부터)": row + 1, "복원 시각": text, "표에 기록할 값": written})
    return rows


def view(record, final):
    """최종 계획 미리보기 아래에 보일 시간 복원 상태·표·동의 문구·실행 동작. time 계획이 아니면 None. 실행·저장은 하지 않는다."""
    if final["split"]["strategy"] != "time":
        return None
    status, current, rules = _status(record, final), _current(record, final), _rules(final)
    allowed = _allowed(record, final)
    lines, table, consent, action, expected = [], None, None, None, None
    if status == "stale":
        lines.append(("warning", "계획(시간 규칙 등) 또는 입력이 이전 미리보기 뒤에 바뀌어 이전 미리보기·확인은 쓰지 않습니다. 다시 미리보기하세요."))
    if status == "stopped":
        lines.append(("warning", "복원 미리보기가 멈췄습니다: " + TIME_CODES.get(current["code"], current["code"])
                      + ". 작업용 표는 바뀌지 않았고 자동 보간·외삽·행 삭제는 하지 않습니다."))
        lines.append(("text", "다음 행동: " + STOP_NEXT.get(current["code"], STOP_DEFAULT) + " 규칙을 고치면 다시 미리보기할 수 있습니다."))
    if status == "failed":
        lines.append(("error", current.get("message") or FAILURES["TIME_PREVIEW_FAILED"]))
    if status in ("previewing", "applying"):
        lines.append(("warning", "이전 시간 복원 실행의 완료를 확인하지 못했습니다. 자동으로 다시 실행하지 않습니다. 다시 미리보기하면 새 실행으로 "
                                 "진행합니다(이전 실행 기록은 보존하고 쓰지 않습니다)."))
    if status in ("previewed", "applied"):
        preview = current["preview"]
        lines.append(("text", f"복원 대상: 전체 {preview['row_count']}행 중 {preview['restored_count']}행 (내부 결측만, 앞뒤 끝 결측 없음)"))
        if preview["restored_count"]:
            table = {"title": "복원 대상 행과 복원 시각", "rows": _shown_rows(preview, rules),
                     "columns": ["데이터 행(1부터)", "복원 시각", "표에 기록할 값"]}
            if preview["truncated"]:
                lines.append(("warning", f"복원 대상이 많아 앞의 {len(preview['rows'])}행만 표시합니다. 적용은 미리보기의 전체 "
                                         f"{preview['restored_count']}행에 대해 같은 판정으로만 진행합니다."))
        else:
            lines.append(("info", "시간 컬럼에 결측이 없어 복원할 행이 없습니다. 작업용 표를 바꿀 필요가 없어 결정 R(적용)은 필요하지 않습니다."))
            lines.append(("text", "확정 시 입력: 기존 작업용 표(복원 대상 0행)로 time 분할합니다. " + _rules_text(final["time"])))
            lines.append(("text", "다음 행동: 아래 결정 C 동의 후 '확인하고 진행'을 누르세요."))
        if preview["restored_count"] and status == "previewed":
            lines.append(("text", "다음 행동: 아래에서 복원을 적용(결정 R)해야 time 분할로 확정할 수 있습니다."))
    if status == "applied":
        lines.append(("success", f"복원을 적용했습니다: {current['preview']['restored_count']}행의 시간 셀을 채운 새 작업용 표(파생 표)를 "
                                 "만들었습니다. 원본 파일과 기존 작업용 표는 그대로입니다."))
        lines.append(("text", f"확정 시 입력: 파생 표(복원 {current['preview']['restored_count']}행)로 time 분할합니다. "
                              + _rules_text(final["time"])))
        lines.append(("text", "다음 행동: 아래 결정 C 동의 후 '확인하고 진행'을 누르세요."))
    if allowed == PREVIEW and _ready_state(record, final):
        # 지금 결과로 확정할 수 있지만, 확정 직전 재확인이 실패했다면 새 미리보기로 다시 시작할 수 있다.
        consent = ["다시 미리보기(선택): 확정이 '복원 결과 또는 작업용 표를 다시 확인하지 못했습니다'로 거부되면 여기서 새 미리보기를 "
                   "실행하세요. 새 시도 번호와 새 결정 T로 같은 규칙을 다시 판정하며, 이전 미리보기·적용 결과와 실행 기록은 보존하고 "
                   "확정에는 새 결과를 씁니다.",
                   _rules_text(final["time"]),
                   "원본 파일과 작업용 표는 바뀌지 않으며, 1회 승인은 이 동의 범위에서 위임 발급합니다. 복원 행이 있으면 다시 적용(결정 R)이 필요합니다."]
        action = {"id": PREVIEW, "label": "다시 미리보기 (새 시도)", "agree": "위 규칙으로 새 복원 미리보기 실행에 동의합니다 (결정 T)",
                  "warning": "새 복원 미리보기 범위를 확인하고 체크해 주세요.", "progress": "시간 결측 복원 대상을 격리 실행으로 다시 판정하고 있습니다."}
    elif allowed == PREVIEW:
        consent = ["동의 범위(결정 T): 아래 규칙으로 작업용 표의 시간 컬럼만 격리 실행에서 읽어, 복원할 행과 시각을 판정합니다.",
                   _rules_text(final["time"]),
                   "앞뒤 끝 결측·중복 시각·주기/순서 불일치·시간대 모호성·형식 불일치가 있으면 복원하지 않고 멈춥니다. 자동 재시도하지 않습니다.",
                   "이 단계는 미리보기만 합니다. 원본 파일과 작업용 표는 바뀌지 않으며, 1회 승인은 이 동의 범위에서 위임 발급합니다."]
        action = {"id": PREVIEW, "label": "복원 미리보기 실행", "agree": "위 규칙으로 복원 미리보기 실행에 동의합니다 (결정 T)",
                  "warning": "복원 미리보기 범위를 확인하고 체크해 주세요.", "progress": "시간 결측 복원 대상을 격리 실행으로 판정하고 있습니다."}
    elif allowed == APPLY:
        expected = current["preview"]["preview_sha256"]
        consent = [f"동의 범위(결정 R): 위 미리보기의 {current['preview']['restored_count']}행에만 복원 시각을 기록한 새 작업용 표(파생 표)를 "
                   "격리 실행으로 만듭니다.",
                   "적용 직전 같은 규칙으로 다시 판정해 미리보기와 완전히 같을 때만 기록합니다. 다르면 적용하지 않습니다.",
                   "원본 파일과 기존 작업용 표는 바꾸지 않고, 다른 셀은 그대로 둡니다. 적용 뒤 부모 검증이 행 단위로 다시 확인합니다."]
        action = {"id": APPLY, "label": "복원 적용", "agree": "위 미리보기 결과로 복원 적용에 동의합니다 (결정 R)",
                  "warning": "복원 적용 범위를 확인하고 체크해 주세요.", "progress": "확인한 미리보기로 복원을 적용해 새 작업용 표를 만들고 있습니다."}
    return {"title": TITLE, "status": status, "lines": lines, "table": table, "consent": consent, "action": action,
            "expected": expected}


def consent_content(record, final, action, expected):
    """결정 T·R 동의 내용. 발급 전 조건(상태·규칙·입력·미리보기 지문)을 확인하고 맞지 않으면 ValueError."""
    if action not in ACTIONS or final["split"]["strategy"] != "time":
        raise ValueError("이 계획에서 실행할 수 있는 시간 복원 동작이 아닙니다.")
    _rules(final)
    if _allowed(record, final) != action:
        raise ValueError("현재 시간 복원 상태에서 고를 수 있는 동작이 아닙니다. 화면을 새로 고쳐 다시 확인하세요.")
    content = {"plan_digest": signature(final), "time": final["time"], "time_signature": signature(final["time"]),
               "source_contract_id": _source(record).source_id, "time_code_id": code_id(), "time_policy": POLICY_VERSION,
               "max_output_bytes": MAX_OUTPUT_BYTES, "limits": record["limits"]}
    previous = record.get("time_restore") or {}
    if action == PREVIEW:
        if expected is not None:
            raise ValueError("미리보기에는 확인할 미리보기 지문이 없습니다.")
        return "T", dict(content, time_attempt=previous.get("attempt", 0) + 1, allowed=[["tabular_time_preview", STAGE]])
    preview = previous["preview"]
    if expected != preview["preview_sha256"]:
        raise ValueError("확인한 미리보기와 현재 미리보기가 다릅니다. 화면을 새로 고쳐 다시 확인하세요.")
    return "R", dict(content, time_attempt=previous["attempt"], preview_run=previous["preview_run"],
                     preview_sha256=preview["preview_sha256"], restoration_digest=preview["restoration_digest"],
                     restored_count=preview["restored_count"], allowed=[["tabular_time_apply", STAGE]])


def _drive(service, name, record, context, action, scope, bound):
    """서비스의 공개 상태·승인·진행 API로 완료까지 간다. 승인은 해당 결정의 동의 범위에서만 위임 발급한다.

    위임 근거(동의 ID·승인 범위 해시)는 기록의 시간 복원 항목에 남기고 저장은 호출자가 한 번에 한다(실행 기록에도 위임 이벤트가 남는다).
    """
    kind, tool, decision = ACTIONS[action]
    for _ in range(6):
        state = service.status(name)
        if state["status"] == "completed":
            return service.verified(name)
        if state["status"] in {"blocked_input", "failed_runtime", "security_locked", "cancelled"}:
            runtime = service.runtime_failure(name) if state["status"] == "failed_runtime" else None
            if runtime is not None:
                raise TaskProblem(runtime, {**RUNTIME_MESSAGES, **STORE_MESSAGES}[runtime] + " " + RUNTIME_NOTICE)
            diagnostic = service.worker_diagnostic(name) if state["status"] == "blocked_input" else None
            if diagnostic is not None and diagnostic["code"] == "TIME_PREVIEW_CHANGED":
                raise TaskProblem("TIME_PREVIEW_CHANGED", FAILURES["TIME_PREVIEW_CHANGED"])
            code = "TIME_PREVIEW_FAILED" if kind == "preview" else "TIME_APPLY_FAILED"
            raise TaskProblem(code, FAILURES[code])
        approval_id = None
        if state["status"] == "awaiting_approval":
            preview = service.preview_approval(name)
            authorize = context.authorizer(record, decision, tool, STAGE, expected=scope, bound=bound)
            issued = service.record_delegated_approval(name, preview, authorize=authorize)
            record["time_restore"].setdefault("delegations", {})[kind] = {
                "consent_id": record["consents"][decision]["consent_id"], "scope_sha256": issued["scope_sha256"]}
            approval_id = issued["approval_id"]
        service.next(name, expected_stage=state["stage"], expected_revision=state["revision"], approval_id=approval_id)
    raise TaskProblem("TIME_STATE_UNCLEAR", "시간 복원 실행 상태를 확인해야 합니다. 자동 재시도하지 않습니다.")


def _store_problem(error):
    """저장소 보호 실패(시간 복원 실행 저장소 또는 입력 실행 저장소)를 데이터 오류와 구분한 고정 문제로 바꾼다."""
    messages = {**input_flow.STORE_MESSAGES, **STORE_MESSAGES}
    code = error.code if error.code in messages else "TIME_STORE_PERMISSION_INVALID"
    return TaskProblem(code, messages[code] + " " + RUNTIME_NOTICE)


def run(record, directory, context, action, final):
    """동의를 발급한 뒤 호출한다. 미리보기 또는 적용을 한 번 실행하고 결과·실패를 작업 기록에 남긴다. 단계(phase)는 바꾸지 않는다."""
    kind, _, decision = ACTIONS[action]
    source = _source(record)
    service = TabularTimeService(kind, root=directory / "time_runs")
    prefix = "t" + record["task_id"][:10]
    previous = record.get("time_restore")
    # 위임 발급 때 동의 내용과 대조할 값은 동의에서 복사하지 않고 현재 계획·기록·코드에서 다시 계산한다.
    scope = {"plan_digest": signature(final), "time_signature": signature(final["time"]), "limits": record["limits"],
             "time_attempt": (previous or {}).get("attempt", 0) + 1 if action == PREVIEW else previous["attempt"],
             "source_contract_id": source.source_id, "time_code_id": code_id(), "time_policy": POLICY_VERSION,
             "max_output_bytes": MAX_OUTPUT_BYTES}
    if action == APPLY:
        preview = previous["preview"]
        scope.update(preview_run=previous["preview_run"], preview_sha256=preview["preview_sha256"],
                     restoration_digest=preview["restoration_digest"], restored_count=preview["restored_count"])
    content = record["consents"][decision]["content"]
    if any(content.get(key) != value for key, value in scope.items()):
        raise ValueError("시간 복원 동의가 현재 계획·입력·미리보기와 다릅니다.")
    if action == PREVIEW:
        history = list((previous or {}).get("history", []))
        if previous:
            history.append({key: previous.get(key) for key in ("attempt", "status", "code", "plan_digest", "preview_run",
                                                               "apply_run", "consent_id", "apply_consent_id")})
        entry = {"status": "previewing", "attempt": scope["time_attempt"], "plan_digest": scope["plan_digest"],
                 "time_signature": scope["time_signature"], "source_contract_id": source.source_id,
                 "preview_run": f"{prefix}-tp{scope['time_attempt']}", "apply_run": None, "runtime_id": None,
                 "consent_id": record["consents"]["T"]["consent_id"], "history": history}
    else:
        entry = previous
        if entry["plan_digest"] != scope["plan_digest"] or entry["source_contract_id"] != source.source_id:
            raise ValueError("적용 동의가 확인한 미리보기와 다릅니다.")
        entry.update(status="applying", apply_run=f"{prefix}-ta{entry['attempt']}",
                     apply_consent_id=record["consents"]["R"]["consent_id"])
    record["time_restore"] = entry
    record = context.save(record)
    entry = record["time_restore"]
    try:
        with verification_scope():
            _execute(record, context, service, action, final, source, scope)
    except TaskProblem as problem:
        entry.update(status="failed", code=problem.code, message=problem.message)
    except StoreUnavailable as error:
        problem = _store_problem(error)
        entry.update(status="failed", code=problem.code, message=problem.message)
    except Exception:
        code = "TIME_PREVIEW_FAILED" if action == PREVIEW else "TIME_APPLY_FAILED"
        entry.update(status="failed", code=code, message=FAILURES[code])
    return context.save(record)


def _execute(record, context, service, action, final, source, scope):
    """실행 기록 생성 → 위임 승인 → 실행 → 부모 재검증. 결과는 기록의 시간 복원 항목(같은 객체)에 남기고 저장은 호출자가 한다."""
    kind = ACTIONS[action][0]
    entry = record["time_restore"]
    if action == PREVIEW:
        # 승인 미리보기·위임 발급 전에 실행 런타임 구성 ID를 고정한다. 적용도 같은 ID로만 실행한다.
        try:
            entry["runtime_id"] = current_runtime_id()
        except RuntimeUnavailable as error:
            raise TaskProblem(error.code, RUNTIME_MESSAGES[error.code] + " " + RUNTIME_NOTICE) from None
        name = entry["preview_run"]
        service.create(name, source=source, rules=final["time"], limits=record["limits"], runtime_id=entry["runtime_id"])
    else:
        name = entry["apply_run"]
        service.create(name, source=source, rules=final["time"], limits=record["limits"], runtime_id=entry["runtime_id"],
                       preview_run=entry["preview_run"])
    bound = {"run_name": name, "action": kind, "source_id": source.source_id, "rules_signature": scope["time_signature"],
             "time_code_id": scope["time_code_id"], "runtime_id": entry["runtime_id"],
             "max_output_bytes": MAX_OUTPUT_BYTES, "preview_sha256": scope.get("preview_sha256")}
    _, result = _drive(service, name, record, context, action, scope, bound)
    preview = result["preview"]
    if action == PREVIEW:
        entry.update(status="previewed" if preview["status"] == "ready" else "stopped", code=preview["code"],
                     preview={"row_count": preview["row_count"], "restored_count": preview["restored_count"],
                              "rows": preview["rows"][:MAX_SHOWN], "times": preview["times"][:MAX_SHOWN],
                              "truncated": preview["restored_count"] > MAX_SHOWN,
                              "preview_sha256": result["preview_sha256"], "restoration_digest": preview["restoration_digest"]})
        return
    if result["preview_sha256"] != entry["preview"]["preview_sha256"]:
        raise ValueError("적용 결과의 미리보기 지문이 확인한 값과 다릅니다.")
    derived = service.derived_source(name)
    entry.update(status="applied", source=derived.to_dict(), source_id=derived.source_id)


# ==========================================
# time 분할 입력 준비(확정 전 판정과 확정 직전 증거 재검증)
# - 적용 완료면 검증된 파생 표, 복원 대상 0행이면 커밋된 미리보기를 재검증한 뒤 기존 생성 표를 입력으로 쓴다
# - 그 밖의 상태(미리보기 없음·이전 계획·멈춤·실패·복원 N행 미적용)는 준비되지 않음이다
# - 확정 대상 계획(지문)·규칙·원래 출처·미리보기 지문이 기록과 모두 같아야 한다
# ==========================================

NOT_READY = {
    "none": "먼저 아래 '시간 결측 복원'에서 복원 미리보기를 실행하세요.",
    "stale": "계획 또는 입력이 이전 미리보기 뒤에 바뀌었습니다. 아래 '시간 결측 복원'에서 다시 미리보기하세요.",
    "stopped": "복원 미리보기가 멈췄습니다. 아래 원인과 다음 행동을 확인하세요.",
    "failed": "복원 실행이 완료되지 않았습니다. 아래 '시간 결측 복원'에서 새 미리보기부터 진행하세요.",
    "previewing": "이전 시간 복원 실행의 완료를 확인하지 못했습니다. 새 미리보기부터 진행하세요.",
    "applying": "이전 시간 복원 실행의 완료를 확인하지 못했습니다. 새 미리보기부터 진행하세요.",
    "previewed": "복원 대상 행이 있습니다. 아래에서 복원을 적용(결정 R)해야 time 분할로 확정할 수 있습니다.",
}


def readiness(record, final):
    """기록만으로 판정한 time 분할 입력 준비 상태. (준비 여부, 다음 행동 문구). 증거 재검증은 ready_source가 한다."""
    if _ready_state(record, final):
        return True, None
    return False, NOT_READY.get(_status(record, final), NOT_READY["none"])


def ready_source(record, final, directory):
    """확정 직전 time 분할 입력 출처와 결합 사실(time_source)을 증거로 다시 확인해 돌려준다. 맞지 않으면 ValueError.

    적용 완료: 파생 표 출처(부모·적용 커밋·묶음·provenance)를 다시 확인하고 규칙·미리보기 지문을 대조한다.
    복원 0행: 커밋된 미리보기 실행을 다시 검증(부모 재판정 포함)하고 기존 생성 표를 입력으로 쓴다. 결정 R은 없다.
    """
    if not readiness(record, final)[0]:
        raise ValueError("time 분할 입력이 준비되지 않았습니다.")
    current = _current(record, final)
    preview = current["preview"]
    original = _source(record)
    facts = {"parent_source_id": original.source_id, "preview_run": current["preview_run"],
             "preview_sha256": preview["preview_sha256"], "restoration_digest": preview["restoration_digest"],
             "restored_count": preview["restored_count"], "time_signature": signature(final["time"])}
    if current["status"] == "applied":
        source = restored_source(record)
        if (source.rules != final["time"] or source.preview_sha256 != preview["preview_sha256"]
                or source.restoration_digest != preview["restoration_digest"] or source.restored_count != preview["restored_count"]):
            raise ValueError("파생 표가 확정 대상 규칙·미리보기와 다릅니다.")
        return source, dict(facts, kind=source.kind, apply_run=current["apply_run"])
    original.check_input()
    _, result = TabularTimeService("preview", root=Path(directory) / "time_runs").verified(current["preview_run"])
    verified = result["preview"]
    if (result["preview_sha256"] != preview["preview_sha256"] or verified["status"] != "ready" or verified["restored_count"] != 0
            or verified["source_id"] != original.source_id or verified["rules_signature"] != signature(final["time"])):
        raise ValueError("복원 0행 미리보기가 확정 대상 규칙·입력과 다릅니다.")
    return original, dict(facts, kind=original.kind, apply_run=None)


def restored_source(record):
    """적용을 마친 파생 표 출처를 다시 확인해 돌려준다. 다음 단계(time 분할)의 입력이다. 없으면 None."""
    from .tabular_time_service import TimeRestoredSource
    current = record.get("time_restore") or {}
    if current.get("status") != "applied":
        return None
    source = TimeRestoredSource.from_dict(current["source"])
    original = _source(record)
    # 이 작업 폴더의 적용 실행(작업 ID로 정해지는 이름)에서 나온 출처여야 한다. 다른 작업의 온전한 출처로 바꾼 기록은 거부한다.
    run = Path(original.snapshot_directory).parent / "time_runs" / "apply" / str(current.get("apply_run"))
    if (source.parent.source_id != original.source_id or source.source_id != current["source_id"]
            or current.get("apply_run") != f"t{record['task_id'][:10]}-ta{current['attempt']}"
            or Path(source.application.run_directory) != run):
        raise ValueError("작업 기록의 파생 표 출처가 이 작업의 작업용 표와 다릅니다.")
    source.check_input()
    return source
