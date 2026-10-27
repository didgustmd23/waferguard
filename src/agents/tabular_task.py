"""표형(CSV 로더) 작업 어댑터. 공통 조정기가 부르는 진단·계획·단계 생성·결과 요약을 기존 CSV 작업 5종에 연결한다.

계산·격리·결과 검증은 기존 실행기와 결과 검증기가 한다. 이 모듈은 이름·경로·선행 결과를 채우고,
검증된 결과 파일에서 사실만 읽는다. 수치를 다시 계산하거나 추정하지 않으며 없으면 '미집계'로 둔다.
설계 기준: docs/agents/design/CSV_TASK_AGENT.md 7절
"""

from collections import Counter
from dataclasses import asdict, replace
import json
from pathlib import Path

from src.preprocessing.input_profile import (DEDUPE_INPUT, HEADER_DEDUPE_ID, bind_input_profile_plan, header_dedupe_pipeline,
                                             input_profile_pipeline, parse_header, split_header)
from . import llm_task_assist as rules
from . import llm_workflow_store as plans
from . import pipeline_approval_cli as diagnostics
from .csv_workflow_creation import _pipeline as prepared_pipeline
from .csv_workflow_creation import code_id as csv_code_id
from .csv_workflow_proposal_catalogue import workflow_proposal_specifications
from .input_profile_pipeline import committed_diagnostic, read_profile_table
from .input_snapshot import PAYLOAD_NAME, create_snapshot
from .isolated_eda import _read_file, _valid_png
from .llm_workflow_binding import binding_code_id, bind_workflow_step, create_bound_step
from .llm_workflow_contract import validate_workflow_proposal
from .module_pipeline_transport import code_manifest
from .preparation_module_plan import validate_preparation_modules
from .profile_store import save_profile
from . import tabular_options
from .snapshot_eda import _digest
from .task_plan import PlanError, TaskProblem
from .workflow_service_registry import CSV_IDS, workflow_service
from .worker_contract import WorkerLimits
from . import strict_json
from .worker_paths import _path


KIND = "tabular_csv"
INPUT = {"label": "CSV 파일", "picker": "csv_file",
         "help": "원본은 변경하지 않고 작업용 복사본만 사용합니다. 경로는 모델에 보내지 않습니다."}
MISSING = "미집계"
TERMINAL = {"completed", "blocked_input", "failed_runtime", "security_locked", "cancelled"}
STEPS = (("split", "csv_split_review", "C"), ("export", "csv_partition_export", "C"),
         ("candidate", "csv_prepare_candidate", "C"), ("apply", "csv_prepare_apply", "D"),
         ("handoff", "csv_local_handoff", "D"))
# 전처리 적용은 후보 작업과 같은 내부 작업 이름을 쓴다.
RUN_ALIAS = {"apply": "candidate"}
REVIEW_STEP = "candidate"
STEP_LABELS = {"split": "분할 검토·확정", "export": "개발 구간 내보내기", "candidate": "Train 기준 전처리 후보",
               "apply": "전처리 적용", "handoff": "로컬 전달 묶음"}
ALLOWED = tuple((module_id, "1.0.0") for module_id in CSV_IDS)
TRANSFER_SCOPE = "goal_text_column_names_and_aggregates_without_rows_or_individual_values"
MAX_REPORT_TEXT = 64 * 1024
HEADER_LIMIT = 1024 * 1024
ACTIONS = {"header_dedupe": {
    "label": "번호 붙인 복사본으로 진행",
    "consent": "원본 파일은 그대로 두고, 중복된 컬럼명에만 이름_1, 이름_2처럼 번호를 붙인 작업용 복사본을 격리 실행으로 만듭니다. "
               "데이터 값과 컬럼 순서는 바꾸지 않습니다. 이후 진단·처리는 이 복사본으로 이어 가며, 정비 단계의 1회 승인은 "
               "이 동의(결정 A) 범위에서 위임 발급합니다."}}
FINDING_LABELS = {"CONSTANT_FEATURE": "값이 하나뿐인 특징", "HIGH_MISSING_RATIO": "결측률 기준 초과",
                  "ALL_MISSING_FEATURE": "전체 결측 특징", "TARGET_MISSING": "target 결측",
                  "DUPLICATE_ID": "식별자 중복", "REQUIRED_COLUMN_MISSING": "필수 컬럼 없음",
                  "TIMESTAMP_FORMAT_REQUIRED": "시간 형식 필요(통합 작업에서 지정 불가)", "TIMESTAMP_MISSING": "시간 결측",
                  "INVALID_TIMESTAMP": "시간 형식 불일치"}
HINTS = {
    "CATEGORY_LIMIT_EXCEEDED": {
        "confirmed": True,
        "cause": "범주형 특징의 원핫 범주 수가 상한을 넘어 Train 기준 전처리 후보를 만들 수 없었습니다(worker 세부 코드로 확인).",
        "next": "계획 확인 화면의 질문에 따라 관련 항목의 역할을 제외(값이 행마다 다르면 식별자)로 바꾼 뒤 다시 확인하세요. 새 시도로 이어서 진행합니다."},
    "CATEGORY_LIMIT_LIKELY": {
        "cause": "범주형 특징의 고유값이 원핫 범주 상한을 넘어 후보 계산이 실패했을 가능성이 큽니다(추정, 이전 기록은 세부 코드가 없습니다).",
        "next": "계획 확인 화면의 질문에 따라 관련 항목의 역할을 제외(값이 행마다 다르면 식별자)로 바꾼 뒤 다시 확인하세요."},
    "UNSEEN_CATEGORY_LIKELY": {
        "cause": "범주형 특징의 값이 행마다 달라 Validation에 Train에 없던 범주가 있었을 가능성이 큽니다(추정).",
        "next": "새 작업을 시작해 해당 컬럼을 식별자로 지정하세요. 같은 설정으로 다시 진행하면 같은 실패가 납니다."},
}
CODE_LABELS = dict(rules.CODE_LABELS)
# 권장안 거부 중 하위 진단(draft.rejection)이 생기는 코드. 진단이 없는 이전 기록은 상세 원인 미확인으로 안내한다.
REJECTION_DETAIL_CODES = ("SPLIT_INVALID", "SPLIT_ROLE_MISMATCH")
# 저장된 하위 진단을 표시하기 직전의 허용 명세 검사. 진단을 만든 표형 계획 규칙의 명세를 그대로 쓴다.
rejection_valid = rules.rejection_valid
REMOVED_LABELS = {"HIGH_MISSING_RATIO": "Train 결측 비율이 기준 초과", "CONSTANT_OR_ALL_MISSING": "Train에서 값이 한 종류뿐이거나 전부 결측"}
OLD_CANDIDATE = "미산출(영향 집계 이전 버전으로 계산한 후보)"
OLD_APPLICATION = "미산출(영향 집계 이전 버전의 적용 기록)"


def accepts(path):
    return Path(path).suffix.lower() == ".csv"


def _short(record):
    return record["task_id"][:10]


def _diagnostic_code():
    return _digest(code_manifest())


def _plan(record):
    """확정 계획. 2026-10-21 이전 기록은 역할·분할·전처리를 confirmed 최상위에 두었다."""
    confirmed = record["confirmed"]
    return confirmed["plan"] if "plan" in confirmed else {key: confirmed[key] for key in rules.SECTIONS}


def service(module_id):
    return workflow_service(module_id, "1.0.0")


# ==========================================
# 결정 A와 입력 진단
# ==========================================

def consent_a_content(record, action=None):
    """결정 A 내용. 중복 컬럼명 정비는 사용자가 차단 화면에서 고를 때만 새 동의로 허용한다."""
    dedupe = action == "header_dedupe"
    allowed = [["header_dedupe", "run_pipeline"], ["input_profile", "run_pipeline"]] if dedupe else [["input_profile", "run_pipeline"]]
    steps = ["header_dedupe", "derived_snapshot", "input_profile"] if dedupe else ["input_snapshot", "input_profile"]
    return {"transfer": TRANSFER_SCOPE, "header_dedupe": dedupe, "auto_steps": steps,
            "module_pipeline_code_id": _diagnostic_code(), "allowed": allowed}


def consent_a_text(model):
    return ["읽는 것: 선택한 CSV의 복사본만 사용하며 원본은 변경하지 않습니다.",
            f"모델 전달: 목표 문장과 컬럼명·컬럼별 집계(결측률·고유값 수·평균·표준편차 등)를 로컬 모델 {model}에만 보냅니다. "
            "원본 행과 최솟값·최댓값 같은 개별 값은 보내지 않습니다.",
            "자동 진행: 입력 복사와 격리 진단. 진단 단계의 1회 승인은 이 동의 범위에서 위임 발급합니다."]


def header_reason(snapshot_directory):
    """차단된 진단의 헤더 사유를 고정 코드와 개수로만 판별한다. 승인된 작업 복사본의 첫 레코드만 읽는다.

    차단 기준은 바꾸지 않고 컬럼명을 고치지 않는다. 반환값에 컬럼명·값·경로는 넣지 않는다. 사유가 헤더가 아니면 None.
    """
    path = _path(Path(snapshot_directory) / PAYLOAD_NAME, directory=False)

    with path.open("rb") as stream:
        prefix = stream.read(HEADER_LIMIT)

    try:
        names = parse_header(split_header(prefix)[1])
    except ValueError:
        return {"code": "HEADER_UNREADABLE"}

    if any(not name.strip() for name in names):
        return {"code": "EMPTY_COLUMN_NAME"}

    repeated = [count for count in Counter(names).values() if count > 1]

    if repeated:
        return {"code": "DUPLICATE_COLUMN_NAMES", "duplicated_names": len(repeated), "affected_columns": sum(repeated)}

    return None


def _blocked(record, dedupe_done):
    reason = header_reason(record["snapshot"]["directory"])
    code = (reason or {}).get("code")

    if code == "DUPLICATE_COLUMN_NAMES" and not dedupe_done:
        detail = {key: value for key, value in reason.items() if key != "code"}
        return TaskProblem(code, "같은 컬럼명이 여러 번 쓰여 입력 진단을 진행할 수 없습니다. 원본은 그대로 두고 중복된 이름에만 "
                                 "번호를 붙인 작업용 복사본으로 이어서 진행할 수 있습니다.", actions=("header_dedupe",), detail=detail)

    if code == "EMPTY_COLUMN_NAME":
        return TaskProblem(code, "이름이 비어 있는 컬럼이 있습니다. 자동으로 고치지 않습니다. 파일에서 컬럼명을 채운 뒤 새 작업으로 시작하세요.")

    if code == "HEADER_UNREADABLE":
        return TaskProblem(code, "CSV 헤더를 읽지 못했습니다(UTF-8 인코딩·따옴표·줄바꿈 확인). 파일을 확인한 뒤 새 작업으로 시작하세요.")

    return TaskProblem("DIAGNOSTIC_BLOCKED", "입력 진단이 차단되었습니다. CSV 형식 오류일 수 있습니다. 파일을 확인한 뒤 새 작업으로 시작하세요.")


def _run_pipeline(record, context, name, label, header_dedupe, slot):
    """격리 입력 진단(또는 중복 컬럼명 정비) run을 진행한다. 승인은 결정 A의 허용 범위에서만 위임 발급한다."""
    snapshot = record["snapshot"]
    pipeline = header_dedupe_pipeline() if header_dedupe else input_profile_pipeline()
    record[slot] = {"run_name": name, "status": "reserved"}
    record = context.save(record)
    diagnostics.create_input_profile(name, Path(snapshot["directory"]), WorkerLimits(**record["limits"]), confirmed=True,
                                     header_dedupe=header_dedupe)

    for _ in range(8):
        state = diagnostics.status(name)

        if state["status"] in TERMINAL:
            if state["status"] == "completed":
                return record

            if state["status"] == "blocked_input" and not header_dedupe:
                raise _blocked(record, "derived" in record["source"])

            raise TaskProblem("DIAGNOSTIC_FAILED", "입력 진단 실행에 실패했습니다. 자동 재시도하지 않습니다.")

        if state["status"] == "awaiting_approval":
            preview = diagnostics.preview_approval(name)
            authorize = context.authorizer(
                record, "A", label, "run_pipeline",
                expected={"module_pipeline_code_id": _diagnostic_code(), "limits": record["limits"],
                          "source_path": record["source"]["path"], "header_dedupe": header_dedupe or "derived" in record["source"]},
                bound={"adapter_id": "tabular_csv_diagnostic_v1", "snapshot_sha256": snapshot["snapshot"]["sha256"],
                       "pipeline_signature": bind_input_profile_plan(pipeline)[1].signature},
                summary_stage=False)
            result = diagnostics.record_delegated_approval(name, preview, authorize=authorize)
            record[slot]["delegation"] = {"consent_id": record["consents"]["A"]["consent_id"],
                                          "scope_sha256": _digest(preview["scope"])}
            record = context.save(record)
            diagnostics.next_step(name, result["approval_id"])
        else:
            diagnostics.next_step(name)

    raise TaskProblem("STATE_UNCLEAR", "진단 작업 상태가 진행되지 않았습니다. 결과 확인이 필요합니다.")


def diagnose(record, directory, context, *, name=None):
    """격리 입력 진단을 진행하고 검증된 진단 결과만 돌려준다. 승인은 결정 A로만 위임 발급한다."""
    name = name or "t" + _short(record) + "-diag"
    record = _run_pipeline(record, context, name, "input_profile", False, "diagnostic")
    _, _, run = diagnostics._open(name)
    tool, output = committed_diagnostic(run, "input_profile")
    summary, columns = read_profile_table(tool, output)
    record["diagnostic"].update(status="completed", summary=summary, columns=columns)
    return record


def _groups(mapping):
    """바뀐 이름을 원래 이름별로 묶는다. 로컬 화면·보고서 전용이며 컬럼명은 일반 텍스트로만 표시한다."""
    groups = {}

    for column in mapping["columns"]:
        if column["original"] != column["renamed"]:
            groups.setdefault(column["original"], []).append(column["renamed"])

    return [{"original": name, "count": len(names), "first": names[0], "last": names[-1]} for name, names in groups.items()]


def check_resolution(record, action):
    """동의 발급 전 해결 가능 여부. 제시한 사유와 동작이 맞고 아직 정비하지 않은 입력일 때만 통과한다."""
    if action != "header_dedupe" or (record.get("problem") or {}).get("code") != "DUPLICATE_COLUMN_NAMES":
        raise ValueError("이 작업에서 고를 수 없는 해결 동작입니다.")

    if "derived" in record["source"] or record.get("dedupe") is not None:
        raise ValueError("이미 정비한 입력입니다. 다시 정비하지 않습니다.")

    if not (Path(record["snapshot"]["directory"]) / PAYLOAD_NAME).is_file():
        raise ValueError("작업 복사본이 없어 정비할 수 없습니다. 새 작업으로 시작하세요.")


def resolution_facts(record, action):
    """해결 동작 이력에 남길 연결 정보. 실행 이름과 해시만 담고 원본 값은 담지 않는다."""
    return {"resolution_run": record["source"]["derived"]["run_name"],
            "snapshot_after_sha256": record["snapshot"]["snapshot"]["sha256"],
            "rediagnostic_run": record["diagnostic"]["run_name"]}


def resolve(record, directory, context, action):
    """차단 화면에서 사용자가 고른 해결 동작. 중복 컬럼명만 기존 격리 정비로 번호를 붙인 작업용 복사본을 만들고 진단을 이어 간다.

    정비 결과는 부모 검증기가 원본 snapshot과 바이트 단위로 대조한다. 파생 복사본은 검증된 해시와 같을 때만 쓴다.
    원본 파일·기존 snapshot은 바꾸지 않고, 새 파일은 덮어쓰지 않는다. 일반 분석의 중복 헤더 거부 기준은 그대로다.
    """
    # 실행 직전 재검사. 동의 발급 전 검사와 같은 조건이며, 정비 입력은 검증된 작업 복사본으로 고정한다.
    if action != "header_dedupe" or "derived" in record["source"]:
        raise ValueError("이 작업에서 고를 수 없는 해결 동작입니다.")

    name = "t" + _short(record) + "-dedu"
    record = _run_pipeline(record, context, name, "header_dedupe", True, "dedupe")
    _, _, run = diagnostics._open(name)
    tool, output = committed_diagnostic(run, HEADER_DEDUPE_ID)
    validated = tool._read_result(output / "pipeline")

    if validated is None or not validated["changed"]:
        raise TaskProblem("DEDUPE_FAILED", "중복 컬럼명 정비 결과를 확인하지 못했습니다. 자동 재시도하지 않습니다. 새 작업으로 시작하세요.")

    original = record["snapshot"]
    source = output / "pipeline" / tool.module_plan.steps[0].step_id / DEDUPE_INPUT
    snapshot = create_snapshot(source, directory / "derived_snapshot",
                               max_input_bytes=max(original["snapshot"]["max_input_bytes"], validated["derived_size"]))

    if snapshot.sha256 != validated["derived_sha256"] or snapshot.size_bytes != validated["derived_size"]:
        raise TaskProblem("DEDUPE_FAILED", "정비한 복사본이 검증된 정비 결과와 다릅니다. 사용하지 않습니다.")

    record["source"]["derived"] = {"rule": validated["mapping"]["rule"], "run_name": name, "original_snapshot": original,
                                   "renamed": _groups(validated["mapping"])}
    record["snapshot"] = {"directory": str(directory / "derived_snapshot"), "snapshot": asdict(snapshot)}
    record = context.save(record)
    return diagnose(record, directory, context, name="t" + _short(record) + "-dia2")


def input_notices(record):
    """정비한 복사본을 쓰는 작업의 안내. 컬럼명이 들어가므로 일반 텍스트로만 표시한다."""
    derived = record["source"].get("derived")

    if derived is None:
        return []

    lines = [f"{item['original']}라는 컬럼명이 {item['count']}번 쓰여 {item['first']}~{item['last']}로 번호를 붙였습니다."
             for item in derived["renamed"]]
    return lines + ["데이터 값과 컬럼 순서는 그대로이고 원본 파일은 변경하지 않았습니다. 이후 단계는 이 복사본을 씁니다."]


def diagnosis_view(record):
    """결정 화면의 진단 요약. 컬럼 이름은 표 값으로만 쓴다."""
    summary = record["diagnostic"]["summary"]
    shown = ("column", "dtype_group", "missing_ratio", "observed_unique", "finite_count")
    return {"lines": [f"행 {summary['row_count']}개 · 컬럼 {summary['column_count']}개 · 중복 행 {summary['repeated_rows']}개"],
            "rows": [{name: row[name] for name in shown} for row in record["diagnostic"]["columns"]]}


# ==========================================
# 권장안·계획·질문 (공통 task_recommend·task_plan이 호출)
# ==========================================

def plan_fields(record):
    return rules.plan_fields(record["diagnostic"]["columns"])


def base_plan(record):
    return rules.base_plan(record["diagnostic"]["columns"])


def recommendation_input(record):
    """권장안 입력. 전역 집계, 지시문, 컬럼별 판단 대상을 준다. 요청 예산과 제공 범위는 공통 계층이 정한다."""
    diagnostic = record["diagnostic"]
    summary = diagnostic["summary"]
    return {"data": {"row_count": summary["row_count"], "repeated_rows": summary["repeated_rows"]},
            "instructions": rules.instructions(),
            "items": rules.recommendation_items(record["goal"], summary, diagnostic["columns"], base_plan(record))}


def validate_plan(record, plan):
    """계획 전체 검증. 기존 프로파일·분할·전처리 계약, 역할×분할 교차 검사, 작업 카탈로그 검증을 모두 거친다."""
    profile, preparation = rules.validate_plan(record["diagnostic"]["columns"], plan)

    try:
        _proposal("task_check", profile, preparation)
    except (ValueError, TypeError) as error:
        raise PlanError("CATALOGUE_PLAN_INVALID", CODE_LABELS["CATALOGUE_PLAN_INVALID"]) from error

    return profile, preparation


def questions(record, plan):
    diagnostic = record["diagnostic"]
    return rules.plan_questions(record["goal"], diagnostic["summary"], diagnostic["columns"], plan)


def on_rejected(record, draft):
    return rules.on_rejected(draft)


def option_guide(record, plan, origins):
    """확정 전 옵션 설명과 진단 기준 사전 경고. 표시 전용이며 계획을 바꾸지 않는다."""
    return tabular_options.option_guide(record, plan, origins)


def _selected(profile, preparation):
    return {"csv_split_review": asdict(profile.split_rules), "csv_partition_export": {},
            "csv_prepare_candidate": asdict(preparation), "csv_prepare_apply": {}, "csv_local_handoff": {}}


def _proposal(label, profile, preparation):
    selected = _selected(profile, preparation)
    context = {"allowed_modules": ALLOWED, "selected_settings": selected, "existing_inputs": ()}
    specs = {item["module_id"]: item for item in workflow_proposal_specifications()}
    steps, previous = [], []

    for step_id, module_id, _ in STEPS:
        steps.append({"step_id": step_id, "module_id": module_id, "version": "1.0.0",
                      "inputs": specs[module_id]["inputs"], "settings": selected[module_id], "depends_on": previous})
        previous = [step_id]

    response = {"schema_version": "llm_workflow_proposal_v1", "status": "proposed", "questions": [],
                "workflow": {"workflow_id": label, "steps": steps}}
    return validate_workflow_proposal(json.dumps(response, ensure_ascii=False), **context), context


def approval_stages(module_id):
    """카탈로그 명세의 승인 단계. 결정 C·D의 허용 범위는 확정 계획에서 이 값으로 계산한다."""
    return next(item["approval_boundaries"] for item in workflow_proposal_specifications() if item["module_id"] == module_id)


def prepare(record, directory, plan, label):
    """확정 계획을 기존 프로파일·전처리 계획·작업 계획 파일로 새로 만든다. 기존 파일은 덮어쓰지 않는다."""
    profile, preparation = validate_plan(record, plan)
    profile = replace(profile, profile_id=label)
    saved_profile = save_profile(profile, root=directory / "profiles")
    plan_path = _write_preparation_plan(directory, label, plan["preparation"], preparation)
    proposal, context = _proposal(label, profile, preparation)
    plans.save_workflow(label, proposal, context, root=_plan_root(directory))
    return {"profile_path": saved_profile["local_file"], "profile_signature": saved_profile["profile_signature"],
            "preparation_plan": str(plan_path), "plan_name": label, "plan_signature": proposal.plan_signature}


def _plan_root(directory):
    root = directory / "workflow"
    root.mkdir(exist_ok=True)
    return root


def _write_preparation_plan(directory, label, preparation, rules_value):
    """기존 변환 모듈 계획 형식으로 새 파일만 쓴다. 검증된 규칙이 확인한 값과 다르면 거부한다."""
    plan = {"schema_version": "preparation_module_pipeline_v1", "pipeline_id": label, "steps": [
        {"module_id": "tabular_prepare_propose", "version": "1.0.0", "approval": "new_explicit_approval",
         "settings": dict(preparation)},
        {"module_id": "tabular_prepare_apply", "version": "1.0.0", "approval": "new_explicit_approval"}]}

    if json.loads(validate_preparation_modules(plan).rules_json) != asdict(rules_value):
        raise ValueError("전처리 모듈 계획 규칙이 확인한 값과 다릅니다.")

    folder = directory / "plans"
    folder.mkdir(exist_ok=True)
    path = folder / (label + ".json")

    with path.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, ensure_ascii=False, sort_keys=True)

    return path


# ==========================================
# 동의 내용·실행 대조값
# ==========================================

def consent_c_content(record):
    plan = _plan(record)
    return {"profile_signature": record["confirmed"]["profile_signature"], "preparation": plan["preparation"],
            "split": plan["split"], "csv_code_id": csv_code_id(), "binding_code_id": binding_code_id(),
            # 분할 EDA의 사람 검토 단계도 이 동의로 승인한다(화면 동의 문구에 명시).
            "delegated_reviews": ["review_eda"]}


def consent_c_text():
    return ("동의 범위(결정 C): 확인한 역할·처리안으로 품질 검사 → 분할 EDA → EDA 검토 → 분할 제안·확정 → 개발 구간 내보내기 → "
            "Train 기준 전처리 후보 산출까지 자동 진행합니다. 분할 EDA의 사람 검토 단계를 포함해 각 단계의 1회 승인은 이 범위에서 "
            "위임 발급합니다. EDA 보고서는 최종 결과에서 볼 수 있습니다. 실제 적용은 후보를 본 뒤 따로 확인합니다.")


def consent_d_content(record):
    return {"candidate_signature": review_signature(record), "csv_code_id": csv_code_id(), "binding_code_id": binding_code_id()}


def consent_d_text():
    return ("Train 구간에서 산출한 전처리 후보입니다. 적용하면 같은 기준으로 Train·Validation 구간의 새 특징 파일을 만듭니다. "
            "Test 구간에는 적용하지 않고 기준을 다시 산출하지 않습니다. 원본·기존 snapshot은 변경하지 않습니다.")


def expected(record, decision):
    """위임 발급 직전 동의 내용과 같아야 하는 현재 코드 지문·후보 서명."""
    values = {"csv_code_id": csv_code_id(), "binding_code_id": binding_code_id()}

    if decision == "D":
        values["candidate_signature"] = review_signature(record)

    return values


# ==========================================
# 단계 생성·입력 계보 대조
# ==========================================

def _values(record, step_id, name):
    """기존 생성 폼 값을 조정기가 채운다. 사용자는 이름·경로를 입력하지 않는다."""
    steps = record["steps"]
    if step_id == "split":
        snapshot = record["snapshot"]
        return {"name": name, "source": str(Path(snapshot["directory"]) / "input.bin"),
                "profile": record["confirmed"]["profile_path"],
                "max_input_bytes": snapshot["snapshot"]["size_bytes"], "allow_feature_findings": False}
    if step_id == "export":
        return {"name": name, "split_name": steps["split"]["run_name"], "max_partition_bytes": 32 * 1024 ** 2}
    if step_id == "candidate":
        return {"name": name, "split_name": steps["split"]["run_name"], "export_name": steps["export"]["run_name"],
                "pipeline": record["confirmed"]["preparation_plan"], "max_candidate_bytes": 4 * 1024 ** 2,
                "max_partition_bytes": 32 * 1024 ** 2}
    if step_id == "apply":
        return {"pipeline_name": steps["candidate"]["run_name"]}
    return {"name": name, "pipeline_name": steps["candidate"]["run_name"], "max_bundle_bytes": 128 * 1024 ** 2}


def check_saved_plan(record, directory):
    """저장 실행안을 현재 연결 코드로 열고 동의 C·D에 결합한 확정 계획 서명과 같은지 확인한다. 저장 실행안은 자기 해시만 갖는다.

    공통 조정기는 결정 D 발급 전에도 불러, 코드 변경 등으로 열 수 없는 실행안이면 D 없이 멈추고 계획 수정으로 이어 가게 한다.
    """
    saved = plans.load_workflow(record["confirmed"]["plan_name"], root=_plan_root(directory))

    if saved["proposal"].plan_signature != record["confirmed"]["plan_signature"]:
        raise ValueError("저장 실행안이 확정한 계획과 다릅니다. 자동 재개하지 않습니다.")

    return saved


def create_step(record, directory, step_id, name):
    """기존 결합·생성 API로 한 단계를 만든다. 승인·실행은 공통 조정기가 한다."""
    saved = check_saved_plan(record, directory)
    completed = {step: item["run_name"] for step, item in record["steps"].items() if item["status"] == "completed"}
    binding = bind_workflow_step(saved["proposal"], **saved["context"], step_id=step_id,
                                 values=_values(record, step_id, name), limits=WorkerLimits(**record["limits"]),
                                 completed_runs=completed)
    plans.begin_creation(saved, binding, root=_plan_root(directory))
    result = create_bound_step(binding, confirmed=True)
    plans.finish_creation(saved, binding, result, root=_plan_root(directory))


def check_lineage(record):
    """분할 작업이 작업 복사본과 같은 내용으로 만들어졌는지 확인한다."""
    from . import snapshot_split_cli

    if "split" not in record["steps"]:
        return

    split = snapshot_split_cli.open_run(record["steps"]["split"]["run_name"])

    if split.contract.snapshot.sha256 != record["snapshot"]["snapshot"]["sha256"]:
        raise TaskProblem("SNAPSHOT_CHANGED", "분할 작업의 입력이 진단한 복사본과 다릅니다. 새 작업으로 시작하세요.")

    check_profile(record, split.contract.profile)


def check_profile(record, profile):
    """분할 실행 계약에 고정된 프로파일(역할·품질·분할 설정)이 동의 C에 결합한 프로파일 서명과 같은지 확인한다.

    저장 경로·파일만 바꾼 기록은 분할 규칙이 같아도 다른 역할로 실행될 수 있으므로 단계 생성 직후와 결정 D 전에 거부한다.
    """
    consent = (record["consents"].get("C") or {}).get("content") or {}

    if profile.signature != consent.get("profile_signature"):
        raise TaskProblem("PROFILE_CHANGED", "분할 작업의 역할·품질·분할 설정이 동의한 계획과 다릅니다. 새 작업으로 시작하세요.")


def quality_findings(record):
    """분할 전 품질 검사가 멈춘 경우의 확인된 지적. 실행 계약의 프로파일로 저장된 보고서를 다시 검증한 것만 쓴다.

    보고서를 검증하지 못하면 None을 돌려주고 지적을 추정하지 않는다. 품질 기준은 바꾸지 않는다.
    """
    from . import snapshot_split_cli
    from .quality_result_validation import read_quality_report

    from .task_orchestrator import _check_names

    split = record["steps"].get("split")

    if split is None:
        return None

    # 작업 기록의 내부 작업 이름을 먼저 대조한다. 다른 작업의 분할 결과를 이 작업 화면에 보이지 않는다.
    _check_names(record)
    tool = snapshot_split_cli.open_run(split["run_name"])
    manager = snapshot_split_cli.executor(tool)
    task = manager.approvals.get_task(manager.scope_for("audit"))

    if task is None:
        return None

    report = read_quality_report(manager._output(task) / "quality", tool.contract.profile)
    return None if report is None else [{"code": item["code"], "subject": item.get("column")} for item in report["findings"]]


def blocked_problem(record, step_id):
    """단계가 입력 검토로 멈췄을 때 그 자리에서 고쳐 이어 갈 수 있는 경우. 해당 없으면 None."""
    if step_id == "split":
        findings = _safe(lambda: quality_findings(record)) or []
        counts = {}

        for item in findings:
            counts[item["code"]] = counts.get(item["code"], 0) + 1

        message = ("분할 전 품질 검사가 검토 필요 또는 차단으로 중단했습니다. 아래 확인된 지적 컬럼을 제외하거나 역할·기준을 고친 뒤 "
                   "다시 확인하면 새 분할 작업으로 이어서 진행합니다." if findings else
                   "분할 전 품질 검사가 검토 필요 또는 차단으로 중단했습니다. 지적 목록을 확인하지 못했습니다. 값이 하나뿐인 컬럼·높은 결측 "
                   "컬럼을 제외하거나 역할·기준을 고친 뒤 다시 확인하세요.")
        return TaskProblem("QUALITY_REVIEW", message, retryable=True, detail=counts, findings=findings)

    return None


def failure_hint(record, step_id, diagnostic):
    """확인된 worker 분류와 확정 계획으로 추정 원인을 고른다. 근거가 없으면 None(추정하지 않음)."""
    if record.get("confirmed") is None or diagnostic is None or diagnostic["code"] != "INPUT_INVALID":
        return None

    plan = _plan(record)

    if step_id == "candidate":
        summary, columns = record["diagnostic"]["summary"], record["diagnostic"]["columns"]
        names = [item["column"] for item in rules.category_limit_risk(summary, columns, plan)]

        if diagnostic.get("detail") == "CATEGORY_LIMIT_EXCEEDED":
            return {"code": "CATEGORY_LIMIT_EXCEEDED", "subjects": names}

        if diagnostic.get("detail") is None and names:
            return {"code": "CATEGORY_LIMIT_LIKELY", "subjects": names}

    names = rules.identifier_like(record["diagnostic"]["summary"], record["diagnostic"]["columns"], plan["roles"])

    if step_id == "apply" and names and plan["preparation"]["unknown_category"] == "error":
        return {"code": "UNSEEN_CATEGORY_LIKELY", "subjects": names}

    return None


# ==========================================
# 결정 D 대상과 결과 요약 (검증된 결과 파일만 읽는다)
# ==========================================

def _json(path, limit=4 * 1024 * 1024):
    # 검증된 결과 파일은 넓은 표에서 64 KiB를 넘을 수 있다. worker IPC 상한 대신 파일 읽기 상한으로 엄격하게 읽는다.
    return strict_json.loads(_read_file(Path(path), limit).decode("utf-8"), limit)


def _safe(function):
    """검증된 파일에서 값을 꺼내지 못하면 None을 돌려 '미집계'로 표시한다."""
    try:
        return function()
    except (OSError, ValueError, KeyError, TypeError, IndexError, StopIteration):
        return None


def _number(value, digits=4):
    if value is None:
        return MISSING

    return f"{value:.{digits}g}" if isinstance(value, float) else str(value)


def _recorded_code(record, step):
    """실행 당시 연결 코드 지문(해당 결정 동의에 기록된 값). 현재 코드와 같으면 None — 읽기 전용 검증을 쓰지 않는다."""
    from datetime import datetime
    from . import task_consent

    consent = (record.get("consents") or {}).get(step["decision"])

    try:
        content = task_consent.check(consent, step["decision"], record["task_id"], now=datetime.fromisoformat(consent["granted_at"]))
    except (ValueError, TypeError, KeyError):
        return None

    recorded = content.get("csv_code_id")
    return recorded if type(recorded) is str and recorded != csv_code_id() else None


def _verified_step(record, step_id):
    step = record["steps"][step_id]
    module = service(step["module_id"])

    try:
        result = module.results(step["run_name"])
    except ValueError:
        recorded = _recorded_code(record, step) if step["module_id"] == "csv_local_handoff" else None

        # 실행 뒤 코드가 바뀐 경우에만 읽기 전용 검증으로 확인한다. 실행 승인 검사는 그대로이며 이 경로는 실행하지 않는다.
        if recorded is None:
            raise

        result = module.recorded_results(step["run_name"], recorded)

    if result["status"] != "completed":
        raise ValueError("완료되지 않은 단계가 있어 결과를 만들지 않습니다.")

    return {"items": result["items"], "recorded_code_id": result.get("recorded_code_id"),
            "outputs": {item["label"]: item["path"] for item in result["items"] if item["kind"] == "artifact_list"}}


def step_completed(record, step_id):
    """완료 기록 복구용 단계 확인. 결과 검증(필요하면 읽기 전용 검증)을 통과하면 검증 방식을 돌려준다."""
    return "recorded_code" if _verified_step(record, step_id)["recorded_code_id"] else "current_code"


def review_signature(record):
    return prepared_pipeline(record["candidate"]["run_name"])[3].signature


def _origin(record, path):
    """확정 계획 항목의 결정 출처. 모델 권장 이유는 모델 의견으로 따로 돌려준다."""
    plan = _plan(record)
    section, name = path.split(".", 1)
    answered = {item["path"] for item in (record["confirmed"] or {}).get("answers", []) if item.get("path")}
    drafted = record["draft"]["plan"][section].get(name)
    reason = next((item.get("reason") for item in record["draft"].get("changes", []) if item["path"] == path), None)

    if path in answered:
        return "질문 답변", None

    if plan[section].get(name) != drafted:
        return "사용자 변경", None

    if drafted != base_plan(record)[section].get(name):
        return "모델 권장", reason

    return "기본 계획", None


def _excluded_reason(record, column, role, removed):
    """특징에서 빠진 이유. 역할 결정(출처 포함)과 Train 기준 제거 사유를 구분한다."""
    if role in ("target", "식별자", "group", "timestamp"):
        return role + " 역할로 보존(특징 아님)"

    if column["column"] in removed:
        return "Train 기준 제거: " + REMOVED_LABELS.get(removed[column["column"]], removed[column["column"]])

    if role != "제외":
        return ""

    source, reason = _origin(record, "roles." + column["column"])

    if source == "기본 계획":
        unique = column["observed_unique"]
        return f"관측값 {unique}종(상수·전체 결측)이라 기본 계획에서 제외" if unique is not None and unique <= 1 else "기본 계획에서 제외"

    if source == "모델 권장":
        return "모델 권장으로 제외(모델 의견: " + str(reason or "이유 없음") + ")"

    return source + "으로 제외"


def _impact_by_column(candidate):
    impact = (candidate or {}).get("impact")
    return None if impact is None else {item["column"]: item for item in impact}


def _category_display(token):
    """검증된 내부 범주 표현을 표시용 값으로 바꾼다. 처리 기준은 변경하지 않는다."""
    if token == "missing":
        return "별도 결측 범주"
    try:
        kind, value = json.loads(token)
    except (TypeError, ValueError):
        return MISSING
    if kind == "str" and isinstance(value, str):
        return value
    if kind == "bool" and type(value) is bool:
        return "true" if value else "false"
    if kind == "number" and type(value) in (int, float):
        return _number(value)
    return MISSING


def _processing(record, preprocessor, candidate):
    """컬럼별 역할·결정 출처·처리·제외 사유. 처리 내용은 검증된 후보 기준과 Train 영향 집계에서만 가져온다."""
    plan = _plan(record)
    roles, rules_value = plan["roles"], plan["preparation"]
    fitted = {item["column"]: item for item in ((preprocessor or {}).get("state", {}).get("records", []))}
    removed = {item["column"]: item.get("reason") for item in (candidate or {}).get("removed", []) if isinstance(item, dict)}
    impact = _impact_by_column(candidate)
    rows = []

    for column in record["diagnostic"]["columns"]:
        name, role = column["column"], roles.get(column["column"], "제외")
        item = fitted.get(name)
        effect = (impact or {}).get(name)

        if role not in ("수치형", "범주형"):
            action = "사용하지 않음" if role == "제외" else "행 메타데이터(rows.json)로 보존"
        elif name in removed:
            action = "Train 기준 제거"
        elif item is None:
            action = MISSING
        elif item["kind"] == "numeric":
            # 대치값(Train 기준 값)과 대치 건수(후보 worker 집계)를 다른 항목으로 쓴다. 값을 건수처럼 보이게 하지 않는다.
            if item.get("impute") is None:
                parts = ["대치 없음"]
            else:
                parts = ["대치값 " + _number(item["impute"]) + " (" + tabular_options.value_text("numeric_imputation", rules_value["numeric_imputation"]) + ")"]

                if effect is not None:
                    parts.append(f"대치 건수 Train {effect['imputed']}칸(후보 집계)" if effect["imputed"]
                                 else "Train 결측 0칸이라 대치 대상 없음(후보 집계)")

            parts.append("clip 경계 [" + _number(item["clip"][0]) + ", " + _number(item["clip"][1]) + "]" if item.get("clip") else "clip 없음")

            if item.get("edges") is not None:
                parts.append(f"구간 {len(item['edges']) + 1}개 ({rules_value['bin_encoding']}, 출력 {item['width']}열)")
            elif rules_value["scaling"] != "none":
                parts.append(rules_value["scaling"] + " 스케일 center " + _number(item.get("center")) + " · scale " + _number(item.get("scale")))

            action = ", ".join(parts)
        else:
            missing_note = "(별도 결측 범주 포함)" if "missing" in item.get("categories", []) else ""
            # 범주 결측은 숫자 대치가 아니다. 별도 결측 범주로 두는지 Train 최빈값으로 채우는지만 구분한다.
            filling = ("결측은 별도 결측 범주로 표시" if item.get("fill") == "missing"
                       else "결측은 Train 최빈값 '" + _category_display(item.get("fill")) + "'로 대치")
            action = f"범주 {len(item.get('categories', []))}개{missing_note} → 출력 {item.get('width', MISSING)}열, {filling}"

            if effect is not None:
                action += f" · Train 결측 {effect['missing']}칸(후보 집계)"

        if effect is not None:
            if effect["clipped"]:
                action += f" · Train 값 {effect['clipped']}개 clip"
            elif effect["clipped"] == 0:
                action += " · clip 설정됨, Train 변경 0개(후보 집계)"

            if effect["constant_after"] and effect["distinct_before"] > 1:
                action += f" · 변환 후 상수(변환 전 {effect['distinct_before']}종, 정보 손실)"

        source, _ = _origin(record, "roles." + name)
        rows.append({"컬럼": name, "역할": role, "결정 출처": source, "처리": action,
                     "제외 사유": _excluded_reason(record, column, role, removed)})

    return rows


def _expected_impact(candidate, preprocessor):
    """결정 D와 보고서에 쓸 Train 후보 기준 예상 영향. 이전 버전 후보면 미산출 사유를 돌려준다."""
    impact = (candidate or {}).get("impact")

    if candidate is None:
        return {"missing": MISSING}

    if impact is None:
        return {"missing": OLD_CANDIDATE}

    rows = candidate["binding"]["train_rows"]
    records = {item["column"]: item for item in (preprocessor or {}).get("state", {}).get("records", [])}
    lost = [item for item in impact if item["constant_after"] and item["distinct_before"] > 1]
    clipped = [item for item in impact if item["clipped"]]
    observed = sum(rows - item["missing"] for item in impact if item["clipped"] is not None)
    table = []

    for item in impact:
        record = records.get(item["column"], {})
        binned = record.get("edges") is not None

        if not (item["clipped"] or item["constant_after"] or binned or item["width"] > 1):
            continue

        table.append({"특징": item["column"], "변환 전 고유값": item["distinct_before"],
                      "clipping 변경 값": MISSING if item["clipped"] is None else f"{item['clipped']} ({item['clipped'] / max(rows - item['missing'], 1):.1%})",
                      "구간 수": len(record["edges"]) + 1 if binned else "-", "출력 열": item["width"],
                      "변환 후 상수": "예" if item["constant_after"] else "아니오",
                      "정보 손실": "예" if item in lost else "-"})

    widths = []

    for label, selected in (("수치형(구간화 없음)", lambda record: record["kind"] == "numeric" and record.get("edges") is None),
                            ("수치형 구간화", lambda record: record["kind"] == "numeric" and record.get("edges") is not None),
                            ("범주형 원핫", lambda record: record["kind"] == "categorical")):
        chosen = [record for record in records.values() if selected(record)]

        if chosen:
            widths.append({"구분": label, "입력 특징": len(chosen), "출력 열": sum(record["width"] for record in chosen)})

    return {"rows": rows, "lost": lost, "clipped_values": sum(item["clipped"] for item in clipped), "clipped_features": len(clipped),
            "clip_observed": observed, "constant": sum(1 for item in impact if item["constant_after"]),
            "missing_after": sum(item["missing_after"] for item in impact), "table": table, "widths": widths}


def _impact_warnings(expected, removed_count):
    """정보 손실 사전 경고. 설정은 바꾸지 않고 사실과 다음 행동만 적는다."""
    if "missing" in expected:
        return ["처리 영향(clipping 변경 값·변환 후 상수)은 " + expected["missing"] + "입니다."]

    warnings = []

    if expected["lost"]:
        names = ", ".join(item["column"] for item in expected["lost"][:10]) + (" 외" if len(expected["lost"]) > 10 else "")
        warnings.append(f"변환 전에는 값이 여러 종류였지만 변환 후 모든 값이 같아지는 특징 {len(expected['lost'])}개(Train 후보 계산 집계): {names}. "
                        "이 특징은 학습에 정보를 주지 못합니다(정보 손실). clipping 경계가 한 값으로 모였거나 구간이 하나로 합쳐진 경우입니다.")

    if expected["clipped_values"]:
        warnings.append(f"clipping으로 Train 값 {expected['clipped_values']}개(특징 {expected['clipped_features']}개, clip 대상 관측값의 "
                        f"{expected['clipped_values'] / max(expected['clip_observed'], 1):.1%})가 경계값으로 바뀝니다.")

    if removed_count:
        warnings.append(f"Train 기준으로 제거되는 특징 {removed_count}개가 있습니다. 사유는 제거 후보 표를 확인하세요.")

    if warnings:
        warnings.append("경고 때문에 설정을 자동으로 바꾸지 않았습니다. 바꾸려면 적용하지 말고 '계획 수정'에서 해당 옵션(예: 극단값 자르기 none) "
                        "이나 특징 역할을 고친 뒤 다시 확인하세요(새 시도, 새 동의).")

    return warnings


def review(record):
    """결정 D에서 보여 줄 후보. 결과 검증을 통과한 후보 파일에서 구조화된 표와 예상 영향(Train 후보 계산)을 만든다."""
    name = record["steps"][REVIEW_STEP]["run_name"]
    _, _, _, evidence, _ = prepared_pipeline(name)
    verified = _verified_step(record, REVIEW_STEP)
    folder = Path(verified["outputs"]["derive_preparation"]) / "candidate"
    candidate = _safe(lambda: _json(folder / "candidate.json"))
    preprocessor = _safe(lambda: _json(folder / "preprocessor.json"))
    report = next((item["path"] for item in verified["items"] if item["kind"] == "report" and item["label"].endswith("candidate/candidate.md")), None)
    raw = None if report is None else _path(Path(report), directory=False).read_bytes()
    return {"run_name": name, "signature": evidence.signature, "facts": review_facts(record, candidate, preprocessor, raw)}


def review_facts(record, candidate, preprocessor, raw):
    """후보·전처리 기준(검증된 JSON)과 원문 보고서 바이트로 결정 D 화면의 표·예상 영향을 만든다. 파일을 읽지 않는다."""
    text = None if raw is None else raw[:MAX_REPORT_TEXT].decode("utf-8", errors="replace")
    binding = (candidate or {}).get("binding", {})
    expected = _expected_impact(candidate, preprocessor)
    unknown = expected.get("missing")
    metrics = [{"label": "Train 행 수", "value": binding.get("train_rows", MISSING)},
               {"label": "제거 후보", "value": len(candidate["removed"]) if candidate else MISSING},
               {"label": "유지 입력 특징", "value": len(candidate["retained"]) if candidate else MISSING},
               {"label": "적용 후 예상 출력 특징", "value": (candidate or {}).get("output_feature_count", MISSING)},
               {"label": "clipping으로 바뀌는 값 (예상)", "value": unknown or expected["clipped_values"]},
               {"label": "변환 후 상수 특징 (예상)", "value": unknown or expected["constant"]},
               {"label": "정보 손실 특징 (예상)", "value": unknown or len(expected["lost"])}]
    removed = [{"특징": item["column"] if isinstance(item, dict) else item,
                "사유": REMOVED_LABELS.get(item.get("reason"), item.get("reason", MISSING)) if isinstance(item, dict) else MISSING}
               for item in (candidate or {}).get("removed", [])]
    settings = [{"항목": tabular_options.GUIDE[key]["label"] if key in tabular_options.GUIDE else key, "설정 이름": key,
                 "값": tabular_options.value_text(key, value)} for key, value in sorted((binding.get("rules") or {}).items())]
    tables = [{"title": "특징별 처리", "rows": _processing(record, preprocessor, candidate)},
              {"title": "제거 후보", "rows": removed}]

    if unknown is None:
        tables += [{"title": "예상 처리 영향 (Train 후보 계산, 영향이 있는 특징만)", "rows": expected["table"]},
                   {"title": "출력 폭 (구간화·원핫)", "rows": expected["widths"]}]

    tables.append({"title": "실제 실행 설정 (후보 계산에 쓴 값)", "rows": settings})
    facts = {"metrics": metrics, "tables": tables, "warnings": _impact_warnings(expected, len(removed)),
             "notes": ["표의 영향 수치는 Train 후보 계산에서 격리 worker가 센 예상치입니다. 부모 검증은 형식·범위와 저장 기준으로 "
                       "정해지는 값(대치 건수·남는 결측, clip 경계가 한 값이거나 구간이 하나면 상수)만 확인했습니다. "
                       "적용하면 Train 합계를 적용 결과와 다시 대조하고, Validation 실측치는 결과 보고서에 따로 표시합니다."],
             "text": text, "text_truncated": text is not None and len(text.encode("utf-8")) >= MAX_REPORT_TEXT - 4}
    return facts


def _class_rows(partitions, plan):
    """분할 확정 결과의 구간별 클래스에서 확인한 target 클래스 누락. 클래스 조건을 해제하고 확정한 분할에서만 생긴다."""
    seen = {name: {str(entry["label"])[:80] for entry in item["classes"]} for name, item in partitions.items() if item and item.get("classes")}
    labels = sorted(set().union(*seen.values())) if seen else []
    target = next((name for name, role in plan["roles"].items() if role == "target"), "전체")
    handled = ("사용자가 '모든 구간에 모든 클래스 요구'를 해제해 진행함" if not plan["split"].get("require_all_classes")
               else "분할 설정 확인 필요")
    rows = []

    for name in ("train", "validation", "test"):
        absent = [label for label in labels if name in seen and label not in seen[name]]

        if absent:
            rows.append({"구분": "확인된 클래스 누락", "컬럼": target,
                         "내용": f"{name} 구간에 클래스 {', '.join(absent)} 없음(분할 확정 결과). 이 구간으로 해당 클래스를 평가할 수 없습니다.",
                         "처리": handled})

    return rows


def _problems(record, quality, renamed=None):
    """진단(전체 입력 기준)과 품질 검사에서 확인된 항목. 처리 방식은 확정한 역할·규칙에서만 가져온다.

    renamed는 중복 컬럼명 번호 부여 기록이다. 없으면 v1 작업 기록의 정비 이력을 쓴다.
    """
    plan = _plan(record)
    roles, preparation = plan["roles"], plan["preparation"]
    rows = []

    for column in record["diagnostic"]["columns"]:
        name, role = column["column"], roles.get(column["column"], "제외")
        feature = role in ("수치형", "범주형")

        if column["missing_count"]:
            action = ("수치 결측을 " + tabular_options.value_text("numeric_imputation", preparation["numeric_imputation"]) + "으로 대치"
                      if role == "수치형" and preparation["numeric_imputation"] != "none"
                      else "범주 결측: " + tabular_options.value_text("categorical_imputation", preparation["categorical_imputation"]) if role == "범주형"
                      else "특징으로 쓰지 않음" if not feature else "대치하지 않음")
            rows.append({"구분": "결측", "컬럼": name, "내용": f"{column['missing_count']}행 ({column['missing_ratio']:.1%})", "처리": action})

        if column["observed_unique"] is not None and column["observed_unique"] <= 1:
            rows.append({"구분": "상수·전체 결측", "컬럼": name, "내용": f"관측 고유값 {column['observed_unique']}개",
                         "처리": "특징에서 제외" if not feature else "Train 기준 상수 제거" if preparation["drop_constant"] else "특징 유지"})

        for key, label in (("blank_string_count", "공백 문자열"), ("pos_inf_count", "+무한대"), ("neg_inf_count", "-무한대")):
            if column.get(key):
                rows.append({"구분": label, "컬럼": name, "내용": f"{column[key]}개", "처리": "자동 수정하지 않음"})

        if column.get("outlier_candidate_count"):
            rows.append({"구분": "이상치 후보(IQR 1.5, 진단용)", "컬럼": name, "내용": f"{column['outlier_candidate_count']}개 · 통계적 후보이며 오류 판정이 아닙니다. 정상 0/1 값도 포함될 수 있습니다.",
                         "처리": "Train 기준 IQR clip" if preparation["outlier"] == "iqr_clip" and role == "수치형" else "변경하지 않음(이상치 처리 미선택)"})

    if record["diagnostic"]["summary"].get("repeated_rows"):
        rows.append({"구분": "중복 행", "컬럼": "전체", "내용": f"{record['diagnostic']['summary']['repeated_rows']}행", "처리": "삭제하지 않음"})

    for finding in (quality or {}).get("findings", []):
        rows.append({"구분": "품질 검사: " + FINDING_LABELS.get(finding["code"], finding["code"]), "컬럼": finding.get("column") or "전체",
                     "내용": f"{finding['level']} · 건수 {_number(finding.get('count'))}", "처리": "확정 전 검토"})

    for name in rules.identifier_like(record["diagnostic"]["summary"], record["diagnostic"]["columns"], roles):
        rows.append({"구분": "식별자 의심", "컬럼": name, "내용": "관측 고유값 수가 행 수와 같음", "처리": "범주형 특징으로 유지(사용자 확정)"})

    for item in record["source"].get("derived", {}).get("renamed", []) if renamed is None else renamed:
        rows.append({"구분": "중복 컬럼명", "컬럼": item["original"], "내용": f"{item['count']}개 컬럼이 같은 이름",
                     "처리": f"작업용 복사본에서 {item['first']}~{item['last']}로 번호 부여(값·순서 유지)"})

    return rows


def images(record):
    """화면 표시용 그래프. 저장된 경로를 쓰지 않고 매번 결과 검증을 거친 PNG만 돌려준다."""
    found = []

    for item in _verified_step(record, "split")["items"]:
        if item["kind"] == "image":
            data = _safe(lambda: _read_file(Path(item["path"]), 8 * 1024 * 1024))

            if data and _valid_png(data):
                found.append({"label": Path(item["label"]).name, "path": item["path"], "data": data})

    return found


FILE_USES = {
    "prepared/train_features.csv": "학습용 입력 특징(Train). 모델 학습에 사용",
    "prepared/validation_features.csv": "검증용 입력 특징(Validation). 설정 비교·조기 종료 등에 사용",
    "prepared/train_rows.json": "Train 행 위치와 target·식별자 등 보존 컬럼. 특징 파일과 같은 행 순서",
    "prepared/validation_rows.json": "Validation 행 위치와 보존 컬럼. 특징 파일과 같은 행 순서",
    "prepared/application.json": "적용 기록(구간별 행 수·파일 해시·실제 영향 합계)",
    "prepared/application.md": "적용 기록 안내",
    "preprocessor.json": "Train 기준 전처리 기준(대치값·경계·범주). Test·새 데이터에 같은 기준을 적용할 때 사용",
    "feature_mapping.json": "출력 열 → 원본 컬럼 연결표",
    "provenance.json": "출처·승인 실행 연결 기록",
    "README.md": "전달 묶음 안내",
}


def _measured(application, name, key):
    impact = (application or {}).get("impact")

    if application is None:
        return MISSING

    return OLD_APPLICATION if impact is None else impact[name][key]


MEASURED = "적용 결과(적용 검증기가 출력 파일로 대조)"
CANDIDATE_BASIS = "후보 worker 집계(부모는 형식·범위·결정적 관계만 확인했으며 원본 건수를 다시 센 것이 아님)"
COMPARISON_NOTE = ("같은 대상 구간·컬럼 범위끼리만 처리 전/후를 같은 행에 둡니다. 입력 결측은 후보가 유지한 입력 특징 기준이고 출력 결측은 "
                   "출력 열 기준이라, 제거·원핫·구간화로 열 구성이 달라질 수 있어 서로 차감하지 않습니다. Test 구간은 변환하지 않았으며 "
                   "Test의 결측·분포는 이 보고서에서 새로 집계하지 않았습니다. 미산출은 괄호에 사유를 적었습니다.")


def _comparison(record, features, candidate, application, partitions, expected, output_width):
    """처리 전후 비교 행. 각 행에 대상 구간·컬럼 범위·단위·근거를 두고, 범위가 다른 값은 같은 행에 놓지 않는다.

    후보 Train 집계(worker)와 적용 출력 대조 값, 진단(전체 입력) 값을 섞어 하나의 전후 변화로 쓰지 않는다. 새로 계산하지 않는다.
    """
    unknown = expected.get("missing")
    applied = (application or {}).get("partitions", {})
    removed = (candidate or {}).get("removed")
    impact = (candidate or {}).get("impact")
    retained = f"후보가 유지한 입력 특징 {len(impact)}개" if impact is not None else "후보가 유지한 입력 특징"
    outputs = f"출력 열 {output_width}개"
    # 후보·적용이 모두 영향을 기록했으면 적용 검증기가 후보 Train 합계와 적용 Train 합계의 일치를 확인했다.
    crossed = unknown is None and (application or {}).get("impact") is not None

    def row(item, scope, columns, unit, before, after, basis):
        return {"항목": item, "대상 구간": scope, "컬럼 범위": columns, "단위": unit, "처리 전": before, "처리 후": after, "근거": basis}

    def estimate(key):
        if unknown:
            return " / 후보 예상: " + unknown
        return f" / 후보 예상 {expected[key]}" + (", 적용 Train 합계와 일치(적용 검증기 교차 대조)" if crossed else "(형식·범위만 검사)")

    rows = []

    for name in ("train", "validation", "test"):
        item, state = partitions.get(name), applied.get(name)
        rows.append(row("구간 행 수", name, "-", "행", f"{item['rows']}행 (분할 확정)" if item else MISSING,
                        f"{state['rows']}행 변환" if state else "변환하지 않음",
                        "전: 분할 확정 결과 / 후: " + MEASURED if state
                        else "전: 분할 확정 결과 / 이 구간은 변환하지 않았고 결측·분포를 새로 집계하지 않음"))

    rows += [
        row("특징 → 출력 열", "train·validation", "특징 역할 컬럼 → 출력 열", "개 → 열", features, output_width,
            "후: " + MEASURED + f" / 후보 예상 {(candidate or {}).get('output_feature_count', MISSING)}열"),
        row("Train 기준 제거 특징", "train", "특징 역할 컬럼", "개", "-", len(removed) if removed is not None else MISSING,
            "후보 기준(적용은 같은 기준을 그대로 사용, 재산출 없음)"),
        row("결측 칸 (진단)", "전체 입력(분할 전)", f"모든 컬럼 {len(record['diagnostic']['columns'])}개", "칸",
            sum(column["missing_count"] for column in record["diagnostic"]["columns"]),
            "비교 대상 없음(이 범위 전체를 변환하지 않음: Test 구간·특징이 아닌 컬럼 포함)", "진단 집계(격리 worker, 전체 입력 기준)"),
        row("입력 결측", "train", retained, "칸", OLD_CANDIDATE if impact is None else sum(item["missing"] for item in impact),
            "-", CANDIDATE_BASIS),
        row("입력 결측", "validation", retained, "칸",
            "미집계(현재 결과에 Validation 처리 전 특징별 결측 집계가 없음)", "-", "-"),
        row("대치 건수", "train", retained, "칸", "-", OLD_CANDIDATE if impact is None else sum(item["imputed"] for item in impact),
            CANDIDATE_BASIS),
    ]

    for name in ("train", "validation"):
        tail = estimate("missing_after") if name == "train" else ""
        rows.append(row("출력 결측", name, outputs, "칸", "-", _measured(application, name, "missing_values"), MEASURED + tail))

    for name in ("train", "validation"):
        tail = estimate("clipped_values") if name == "train" else ""
        rows.append(row("clipping으로 바뀐 값", name, "clip 설정 수치 특징", "값", "-", _measured(application, name, "clipped_values"),
                        "적용 worker 집계(부모는 범위·일관성만 검사)" + tail))

    for name in ("train", "validation"):
        tail = estimate("constant") if name == "train" else ""
        rows.append(row("변환 후 상수 특징", name, "출력 특징(특징별 출력 열 묶음)", "특징", "-",
                        _measured(application, name, "constant_features"), MEASURED + tail))

    for name in ("train", "validation"):
        rows.append(row("미등록 범주 발생", name, "범주 특징(Train에 없던 값)", "건(특징별 발생 합계)", "-",
                        _measured(application, name, "unknown_categories"), MEASURED + " / 고유 행 수가 아님"))

    return rows


IMAGE_NOTES = {
    "missing.png": ("결측률 그래프 · 대상: 전체 입력(분할·대치 전), 특징 {features}개 · 선정: 결측률 내림차순 {shown} · "
                    "결측이 없는 특징도 포함될 수 있음"),
    "distributions.png": ("수치 특징 분포(히스토그램 30구간, 결측 제외) · 대상: 전체 입력(분할·대치·스케일 전), 수치 특징 {numeric}개 · "
                          "선정: 입력 순서 앞쪽 {numeric_shown}"),
    "correlations.png": ("Pearson 상관행렬 · 대상: 전체 입력(분할·대치 전), 쌍별 관측값 · 선정: 수치 특징 입력 순서 앞쪽 {numeric_shown} · "
                         "수치 특징이 1개면 자기 상관만 표시되어 해석 대상이 아님"),
}


def _image_notes(plan, limit):
    """분할 EDA 그래프 3종의 대상·시점·선정 기준. 개수는 확정 계획의 역할 수, 상한은 승인된 분할 설정값만 쓴다."""
    roles = plan["roles"].values()
    features = sum(1 for role in roles if role in ("수치형", "범주형"))
    numeric = sum(1 for role in roles if role == "수치형")

    def shown(total):
        if type(limit) is not int:
            return "일부(표시 상한: 이 보고서에서 미확인)"
        return f"상위 {min(limit, total)}개 (상한 {limit})" if total else "해당 없음"

    values = {"features": features, "numeric": numeric, "shown": shown(features),
              "numeric_shown": shown(numeric).replace("상위 ", "") if type(limit) is int else shown(numeric)}
    return {name: text.format(**values) for name, text in IMAGE_NOTES.items()}


def report_facts(record):
    """검증된 결과 파일에서 보고서 사실만 모은다. 계산·추정은 하지 않는다.

    예상치는 Train 후보 계산(worker 집계, 후보 검증기는 형식·범위·결정적 값만 확인), 실측치는 적용 결과(적용 검증기가
    출력 파일로 대조)에서 가져온다. 두 기록이 모두 영향을 담으면 적용 검증기가 Train 합계 일치도 확인했다.
    """
    if set(record["steps"]) != set(STEP_LABELS):
        raise ValueError("작업 기록의 단계 구성이 완료 작업과 다릅니다.")

    check_lineage(record)
    verified = {step_id: _verified_step(record, step_id) for step_id in record["steps"]}
    split = verified["split"]["outputs"]
    quality = _safe(lambda: _json(Path(split["audit"]) / "quality" / "quality.json"))
    proposal = _safe(lambda: _json(Path(split["propose_split"]) / "proposal.json"))
    candidate_dir = Path(verified["candidate"]["outputs"]["derive_preparation"]) / "candidate"
    candidate = _safe(lambda: _json(candidate_dir / "candidate.json"))
    preprocessor = _safe(lambda: _json(candidate_dir / "preprocessor.json"))
    application = _safe(lambda: _json(Path(verified["apply"]["outputs"]["apply_preparation"]) / "prepared" / "application.json"))
    bundle = _safe(lambda: next(item["path"] for item in verified["handoff"]["items"] if item["label"] == "handoff"))
    manifest = _safe(lambda: _json(Path(bundle) / "HANDOFF_COMPLETE.json"))
    return report_from_sources(record, {
        "quality": quality, "proposal": proposal, "candidate": candidate, "preprocessor": preprocessor,
        "application": application, "files": (manifest or {}).get("files", {}), "bundle": bundle,
        "reports": [item["path"] for values in verified.values() for item in values["items"] if item["kind"] == "report"],
        "image_labels": [item["label"] for item in images(record)], "renamed": None, "notices": input_notices(record),
        "input": {"source_path": record["source"]["path"], "snapshot_sha256": record["snapshot"]["snapshot"]["sha256"]}})


def report_from_sources(record, sources):
    """검증된 결과의 사실 묶음으로 보고서를 조립한다. 파일을 읽거나 계산·추정하지 않는다.

    sources는 유형별 어댑터가 검증한 quality·proposal·candidate·preprocessor·application(JSON)과 전달 파일 목록·경로,
    보고서 경로, 그래프 이름, 헤더 정비 기록(renamed), 입력 안내(notices), 입력 식별값(input)이다.
    """
    quality, proposal, candidate = sources["quality"], sources["proposal"], sources["candidate"]
    preprocessor, application, bundle = sources["preprocessor"], sources["application"], sources["bundle"]
    renamed = sources["renamed"]
    summary = record["diagnostic"]["summary"]
    plan = _plan(record)
    features = sum(1 for role in plan["roles"].values() if role in ("수치형", "범주형"))
    partitions = {item["split"]: item for item in (proposal or {}).get("summary", [])}
    applied = (application or {}).get("partitions", {})
    expected = _expected_impact(candidate, preprocessor)
    unknown = expected.get("missing")
    removed = (candidate or {}).get("removed")
    output_width = ((application or {}).get("binding") or {}).get("output_feature_count", MISSING)
    comparison = _comparison(record, features, candidate, application, partitions, expected, output_width)
    split_rules = plan["split"]
    split_rows = []

    for name in ("train", "validation", "test"):
        item = partitions.get(name)
        classes = ", ".join(f"{entry['label']}: {entry['count']}" for entry in item["classes"]) if item and item.get("classes") else "-"
        split_rows.append({"구간": name, "요청 비율": split_rules.get(name + "_ratio", MISSING),
                           "확정 비율": f"{item['ratio']:.3f}" if item and isinstance(item.get("ratio"), (int, float)) else MISSING,
                           "행 수": item["rows"] if item else MISSING,
                           "그룹 수": item["group_count"] if item and item.get("group_count") is not None else "-",
                           "클래스": classes, "변환": "적용" if name in applied else "적용 안 함"})
    no_target = [] if any(role == "target" for role in plan["roles"].values()) else [
        "target(예측 대상) 없이 처리했습니다. 확정 때 사용자가 '예측 대상 없음'을 골랐습니다. 예측용이라면 이 결과를 쓰지 마세요."]
    warnings = []

    for text in _impact_warnings(expected, len(removed or [])):
        if not text.startswith("경고 때문에"):
            warnings.append({"구분": "정보 손실" if "미산출" not in text else "미검증", "내용": text})

    excluded = [row["컬럼"] for row in _processing(record, preprocessor, candidate) if row["역할"] == "제외"]

    if excluded:
        warnings.append({"구분": "정보 손실", "내용": f"역할이 제외인 컬럼 {len(excluded)}개는 특징으로 쓰지 않았습니다. 사유는 컬럼별 처리 표에 있습니다."})

    problems = _problems(record, quality, renamed) + _class_rows(partitions, plan)

    for row in _problems(record, quality, renamed):
        if row["처리"] in ("자동 수정하지 않음", "삭제하지 않음") or row["처리"].startswith("범주형 특징으로 유지"):
            warnings.append({"구분": "미처리", "내용": f"{row['구분']} — {row['컬럼']}: {row['내용']} ({row['처리']})"})

    warnings += [
        {"구분": "미처리", "내용": "Test 구간에는 변환을 적용하지 않았습니다." if application and not application.get("test_used")
                                  else "Test 구간 적용 여부: " + MISSING},
        {"구분": "미검증", "내용": "모델 학습·성능 평가는 하지 않았습니다. 전처리 설정이 예측 성능에 주는 영향은 확인하지 않았습니다."},
        {"구분": "미검증", "내용": "clipping 변경 값 수는 격리 worker가 센 값입니다. 부모 검증은 범위·일관성과 후보·적용 Train 합계의 "
                                  "일치만 확인했습니다."},
        {"구분": "미검증", "내용": "특징별 영향(변환 전 고유값 수, 특징별 clipping 수, 특징별 상수 여부)은 후보 worker 집계입니다. 부모는 "
                                  "형식·범위와 저장 기준으로 정해지는 값만 확인했고, 합계만 적용 결과와 대조했습니다."},
    ]
    cautions = [
        *no_target,
        *sources["notices"],
        "진단 집계(결측률·이상치 후보 등)는 전체 입력 기준입니다. 전처리 기준(대치값 등)은 Train 구간에서만 산출했습니다.",
        "분할 EDA의 사람 검토 단계는 결정 C 동의로 위임 승인했습니다. EDA 결과는 이 보고서의 '그래프' 절과 기술 상세의 "
        "보고서 경로에서 확인하세요.",
    ]
    files = [{"파일": name, "용도": FILE_USES.get(name, "설명 없음"), "크기(byte)": info["size"]}
             for name, info in sorted(sources["files"].items())]
    lost = MISSING if unknown else len(expected["lost"])
    highlights = [
        f"입력 {summary['row_count']}행·{summary['column_count']}컬럼에서 특징 {features}개를 골라 Train 기준으로 전처리했고, "
        f"출력 {output_width}열을 만들었습니다.",
        "적용 구간: " + (", ".join(f"{name} {applied[name]['rows']}행" for name in ("train", "validation") if name in applied) or MISSING)
        + ". Test는 변환하지 않았습니다.",
        f"주요 문제: 진단·품질 검사·분할 확인 지적 {len(problems)}건, 정보 손실 예상 특징 {lost}개, "
        f"Train 기준 제거 {len(removed) if removed is not None else MISSING}개.",
    ]
    usage = [
        "Train 특징 파일로 모델을 학습하고 Validation 특징 파일로 설정을 비교하세요. 행 순서는 같은 이름의 rows.json과 일치합니다.",
        "target·식별자 등 보존 컬럼은 rows.json에 있습니다. 특징 파일에는 들어 있지 않습니다.",
        "Test·새 데이터에는 preprocessor.json의 Train 기준을 그대로 적용해야 합니다. 이 작업은 Test에 적용하지 않았습니다.",
        "결과 파일은 로컬 전달 묶음 안에 있습니다. 경로는 기술 상세에서 확인하세요.",
    ]
    return {
        "title": "표형 전처리 결과 보고서",
        "highlights": highlights,
        "lines": [f"분할: {split_rules['strategy']} · 요청 비율 {split_rules['train_ratio']}/{split_rules['validation_ratio']}/"
                  f"{split_rules['test_ratio']} → 확정 비율 "
                  + "/".join(row["확정 비율"] for row in split_rows) + " (train/validation/test, 행 기준, 시드 "
                  f"{split_rules['random_state']}). group·시간 경계를 지키면 확정 비율이 요청과 다를 수 있습니다."],
        "split_table": {"title": "분할 결과 (요청 설정과 확정 결과)", "rows": split_rows},
        "metrics": [{"label": "입력 행", "value": summary["row_count"]}, {"label": "입력 컬럼", "value": summary["column_count"]},
                    {"label": "특징 수 (전→후)", "value": f"{features} → {output_width}"},
                    {"label": "품질 검사", "value": (quality or {}).get("status", MISSING)}],
        "problems": problems, "processing": _processing(record, preprocessor, candidate),
        "comparison": comparison, "comparison_note": COMPARISON_NOTE,
        "warnings": warnings, "usage": usage,
        "images": [{"label": label} for label in sources["image_labels"]],
        "image_notes": _image_notes(plan, sources.get("plot_limit")),
        "cautions": cautions, "files": files, "files_title": "결과 파일 (로컬 전달 묶음)",
        "technical": {**sources["input"], "bundle_path": bundle, "preprocessor_sha256": (candidate or {}).get("preprocessor_sha256"),
                      "candidate_schema": (candidate or {}).get("schema_version"), "application_schema": (application or {}).get("schema_version"),
                      "rules": (candidate or {}).get("binding", {}).get("rules"),
                      "reports": sources["reports"]},
        "outputs": {"bundle": bundle},
    }
