"""표형 v2(생성된 canonical 표) 작업 어댑터. 공통 조정기가 부르는 입력 해석·진단·계획·단계 생성·결과 요약을 v2 서비스에 연결한다.

입력은 결정 A로 검사·생성(tabular_input_flow)한 뒤 생성된 표에 고정 진단을 붙인다. 권장안·옵션 설명·역할 검사·보고서 조립은
CSV v1 어댑터(tabular_task)와 같은 코드를 쓰고, 결과 파일은 v2 서비스의 결합 읽기(완료 표식·manifest 해시)로만 연다.
계산·격리·결과 검증은 기존 실행기와 검증기가 한다. 수치를 다시 계산하거나 추정하지 않는다.
"""

from dataclasses import asdict, replace
import json

from src.preprocessing.input_profile import bind_input_profile_plan, input_profile_pipeline
from src.preprocessing.tabular_bundle import verification_scope
from . import llm_task_assist as rules
from . import llm_workflow_store as plans
from . import pipeline_approval_cli as diagnostics
from . import tabular_input_flow as flow
from . import tabular_report_evidence as evidence
from . import tabular_task as shared
from .csv_workflow_creation import _pipeline
from .csv_workflow_proposal_catalogue import workflow_proposal_specifications
from .input_profile_pipeline import committed_diagnostic, parse_profile_table, profile_table_files
from .llm_workflow_binding import binding_code_id, bind_workflow_step, create_bound_step
from .llm_workflow_contract import validate_workflow_proposal
from .module_pipeline_transport import code_manifest
from .profile_store import save_profile
from .snapshot_eda import _digest
from .tabular_pipeline import ADAPTER_ID as DIAGNOSTIC_ADAPTER
from .tabular_source_contract import ADAPTER_ID as PROFILE_ADAPTER, TabularSourceContract
from .tabular_workflow_service import (_committed_bytes, candidate_sources, code_id as tabular_code_id,
                                      report_sources, source_of, verified_images)
from .task_plan import PlanError, TaskProblem
from .workflow_service_registry import TABULAR_IDS, TABULAR_VERSION, workflow_service
from .worker_contract import WorkerLimits


KIND = "tabular_v2"
INPUT = dict(shared.INPUT)
MISSING = shared.MISSING
STEPS = tuple((step_id, module_id, decision) for (step_id, _, decision), module_id in zip(shared.STEPS, TABULAR_IDS))
RUN_ALIAS = shared.RUN_ALIAS
REVIEW_STEP = shared.REVIEW_STEP
STEP_LABELS = shared.STEP_LABELS
ALLOWED = tuple((module_id, TABULAR_VERSION) for module_id in TABULAR_IDS)
# 입력 해석은 대기 화면의 선택(resume_input)으로 다시 정한다. 차단 화면의 해결 동작은 없다.
ACTIONS = {}
FINDING_LABELS = dict(shared.FINDING_LABELS, CLASS_MISSING_IN_SPLIT="구간에 없는 target 클래스")
SPLIT_NAMES = {"train": "Train", "validation": "Validation", "test": "Test"}
HINTS = shared.HINTS
REJECTION_DETAIL_CODES = shared.REJECTION_DETAIL_CODES
rejection_valid = shared.rejection_valid
CODE_LABELS = dict(shared.CODE_LABELS, NUMERIC_ROLE_TYPE_UNSUPPORTED=(
    "수치형 역할에는 생성된 표에서 수치 자료형(정수·실수)으로 읽힌 컬럼만 지정할 수 있습니다. 숫자처럼 보이는 문자열(예: 앞자리 0)을 "
    "수치로 바꾸는 정비는 아직 지원하지 않습니다. 해당 컬럼은 범주형·식별자·제외 중에서 고르세요."))
INPUT_CHOICES = {
    "header_repair": {"label": "중복 컬럼명 처리", "options": {"number_duplicates": "같은 이름에 번호를 붙여 진행(이름_1, 이름_2)"}},
    "empty_field": {"label": "빈 칸 처리", "options": {"missing": "결측으로 읽기 (기본)",
                                                    "preserve": "빈 문자열로 보존 (해당 컬럼은 수치 역할 불가)"}},
}
INPUT_BLOCKED = {
    "EMPTY_COLUMN_NAME": "이름이 비어 있는 컬럼이 있습니다. 자동으로 고치지 않습니다. 파일에서 컬럼명을 채운 뒤 새 작업으로 시작하세요.",
    "CSV_ENCODING_INVALID": "UTF-8로 읽을 수 없는 바이트가 있습니다. 파일을 UTF-8(BOM 허용)로 저장한 뒤 새 작업으로 시작하세요.",
    "CSV_MALFORMED": "따옴표·줄바꿈 형식이 올바르지 않은 행이 있습니다. 파일을 확인한 뒤 새 작업으로 시작하세요.",
    "CSV_ROW_WIDTH_MISMATCH": "헤더와 열 수가 다른 행이 있습니다. 쉼표 구분·따옴표를 확인한 뒤 새 작업으로 시작하세요.",
    "CSV_EMPTY_INPUT": "빈 파일입니다. 헤더와 데이터가 있는 CSV로 새 작업을 시작하세요.",
    "CSV_EMPTY_DATA": "헤더만 있고 데이터 행이 없습니다. 데이터가 있는 CSV로 새 작업을 시작하세요.",
    "CSV_HEADER_REPAIR_CONFLICT": "번호를 붙인 컬럼명이 기존 이름과 겹쳐 자동 정비할 수 없습니다. 파일의 컬럼명을 고친 뒤 새 작업으로 시작하세요.",
    "CSV_HEADER_TOO_LONG": "256자를 넘는 컬럼명이 있습니다. 컬럼명을 줄인 뒤 새 작업으로 시작하세요.",
}
INPUT_BLOCKED_DEFAULT = "입력 검사를 통과하지 못했습니다(지원 범위: UTF-8·쉼표 구분·첫 행 헤더 CSV, 행 수·크기 상한). 파일을 확인한 뒤 새 작업으로 시작하세요."


def accepts(path):
    return shared.accepts(path)


def _short(record):
    return record["task_id"][:10]


def _diagnostic_code():
    return _digest(code_manifest())


def _source(record):
    return TabularSourceContract.from_dict(record["input_selection"]["source"])


def service(module_id):
    return workflow_service(module_id, TABULAR_VERSION)


# ==========================================
# 결정 A: 입력 검사·생성과 생성된 표의 고정 진단
# ==========================================

def _a_content(content):
    """입력 검사·생성 범위에 생성된 표의 고정 진단 위임을 더한다."""
    return dict(content, transfer=shared.TRANSFER_SCOPE, diagnostic_code_id=_diagnostic_code(),
                allowed=[*content["allowed"], ["input_profile", "run_pipeline"]])


def consent_a_content(record, action=None):
    if action is not None:
        raise ValueError("이 작업 유형에는 차단 화면의 해결 동작이 없습니다.")
    return _a_content(flow.consent_content(record))


def input_consent_content(record, selection):
    return _a_content(flow.consent_content(record, selection))


def consent_a_text(model):
    return ["읽는 것: 선택한 CSV의 복사본만 사용하며 원본은 변경하지 않습니다.",
            "입력 해석: UTF-8(BOM 허용)·쉼표 구분·첫 행 헤더로 읽고, 빈 칸은 결측으로 읽습니다. 컬럼 자료형은 값으로 추론하며 앞자리 0이 있는 "
            "숫자는 문자열로 보존합니다. 같은 컬럼명이 여러 번 있으면 멈추고 처리 방법을 묻습니다.",
            f"모델 전달: 목표 문장과 컬럼명·컬럼별 집계(결측률·고유값 수·평균·표준편차 등)를 로컬 모델 {model}에만 보냅니다. "
            "원본 행과 최솟값·최댓값 같은 개별 값은 보내지 않습니다.",
            "자동 진행: 입력 복사 → 격리 입력 검사 → 자료형을 보존한 작업용 표 생성 → 격리 진단. 각 단계의 1회 승인은 이 동의 범위에서 "
            "위임 발급합니다."]


def check_input_selection(record, selection, directory):
    return flow.check_selection(record, selection, directory)


def _prepare(record, directory, context, selection=None):
    try:
        return flow.prepare_input(record, directory, context, selection=selection)
    except TaskProblem as problem:
        if problem.code != "INPUT_BLOCKED":
            raise
        code = (problem.diagnostic or {}).get("code")
        raise TaskProblem("INPUT_BLOCKED", INPUT_BLOCKED.get(code, INPUT_BLOCKED_DEFAULT), diagnostic=problem.diagnostic) from None


def diagnose(record, directory, context):
    """입력 검사·생성 뒤 생성된 표를 진단한다. 해석 선택이 필요하면 진단·모델 단계로 가지 않고 대기한다."""
    return _after_input(_prepare(record, directory, context), context)


def resume_input(record, directory, context, selection):
    return _after_input(_prepare(record, directory, context, selection), context)


def check_source(record):
    """후속 단계 전에 원시 복사본과 생성 묶음·승인 연결을 다시 확인한다. 생성 전에는 원시 복사본 검사만 공통 조정기가 한다."""
    if (record.get("input_selection") or {}).get("status") == "materialized":
        flow.check_source(record)


def _after_input(record, context):
    if record["phase"] == "awaiting_input":
        return record
    flow.check_source(record)
    return _diagnose_table(record, context)


def _diagnose_table(record, context):
    """생성된 표에 고정 입력 진단을 격리 실행한다. 승인은 결정 A의 허용 범위에서만 위임 발급한다."""
    source = _source(record)
    name = "t" + _short(record) + "-dv" + str(record["input_selection"]["attempt"])
    record["diagnostic"] = {"run_name": name, "status": "reserved"}
    record = context.save(record)
    diagnostics.create_tabular_diagnostic(name, source, WorkerLimits(**record["limits"]), confirmed=True)

    for _ in range(8):
        state = diagnostics.status(name)

        if state["status"] in shared.TERMINAL:
            if state["status"] != "completed":
                raise TaskProblem("DIAGNOSTIC_FAILED" if state["status"] != "blocked_input" else "DIAGNOSTIC_BLOCKED",
                                  "생성된 표의 입력 진단을 완료하지 못했습니다. 자동 재시도하지 않습니다. 새 작업으로 시작하세요.")
            break

        if state["status"] == "awaiting_approval":
            preview = diagnostics.preview_approval(name)
            authorize = context.authorizer(
                record, "A", "input_profile", "run_pipeline",
                expected={"diagnostic_code_id": _diagnostic_code(), "limits": record["limits"]},
                bound={"adapter_id": DIAGNOSTIC_ADAPTER, "source_id": source.source_id,
                       "pipeline_signature": bind_input_profile_plan(input_profile_pipeline())[1].signature},
                summary_stage=False)
            result = diagnostics.record_delegated_approval(name, preview, authorize=authorize)
            record["diagnostic"]["delegation"] = {"consent_id": record["consents"]["A"]["consent_id"],
                                                  "scope_sha256": _digest(preview["scope"])}
            record = context.save(record)
            diagnostics.next_step(name, result["approval_id"])
        else:
            diagnostics.next_step(name)
    else:
        raise TaskProblem("STATE_UNCLEAR", "진단 작업 상태가 진행되지 않았습니다. 결과 확인이 필요합니다.")

    summary, columns = diagnostic_table(name, source)
    record["diagnostic"].update(status="completed", summary=summary, columns=columns, source_id=source.source_id)
    return record


def diagnostic_table(name, source):
    """커밋·검증된 진단 결과의 요약·컬럼 표. 화면·모델 입력·계획 검사가 쓰므로 완료 표식에 묶인 바이트만 해석한다.

    진단 완료 표식은 승인 체인에 묶인 기대 해시가 없어 표식 재작성 위조는 손상 검사 수준까지만 막는다.
    """
    _, _, run = diagnostics._open(name)

    if run.binding.adapter_id != DIAGNOSTIC_ADAPTER or run.binding.source_id != source.source_id:
        raise ValueError("진단 결과가 이 작업의 생성된 표와 다릅니다.")

    tool, output = committed_diagnostic(run, "input_profile")
    table, summary = profile_table_files(tool)
    return parse_profile_table(_committed_bytes(output, table), _committed_bytes(output, summary))


def input_review(record):
    """해석 대기 화면의 내용. 검사 사유는 고정 코드와 개수만 쓰고 컬럼명·값은 넣지 않는다."""
    current = record["input_selection"]
    inspection = current["inspection"]
    return {"code": inspection["code"], "detail": dict(inspection.get("detail") or {}),
            "message": "같은 컬럼명이 여러 번 쓰여 입력 해석을 멈췄습니다. 원본은 그대로 두고, 아래에서 처리 방법을 고르면 새 검사와 "
                       "작업용 표 생성으로 이어 갑니다.",
            "choices": {key: dict(value, current=current["spec"][key]) for key, value in INPUT_CHOICES.items()},
            "consent": "동의 범위(결정 A 갱신): 고른 해석으로 작업 복사본을 다시 검사하고 자료형을 보존한 작업용 표를 만든 뒤 진단합니다. "
                       "데이터 값과 컬럼 순서는 바꾸지 않습니다. 각 단계의 1회 승인은 이 동의 범위에서 위임 발급합니다. 진단 뒤에는 처음 "
                       "동의와 같은 범위(목표 문장·컬럼명·컬럼별 집계, 원본 행·개별 값 제외)를 로컬 모델 " + record["model"]["model"]
                       + "에 보내 권장안을 받습니다.",
            "inspection_id": current["inspection_id"]}


def _renamed(record):
    mapping = ((record.get("input_selection") or {}).get("inspection") or {}).get("header_mapping") or []
    return shared._groups({"columns": mapping})


def input_notices(record):
    """입력 해석 안내. 컬럼명이 들어가므로 일반 텍스트로만 표시한다."""
    selection = record.get("input_selection") or {}
    if selection.get("status") != "materialized":
        return []
    lines = [f"{item['original']}라는 컬럼명이 {item['count']}번 쓰여 {item['first']}~{item['last']}로 번호를 붙였습니다."
             for item in _renamed(record)]
    if lines:
        lines.append("데이터 값과 컬럼 순서는 그대로이고 원본 파일은 변경하지 않았습니다. 이후 단계는 생성된 작업용 표를 씁니다.")
    empty = selection["spec"]["empty_field"]
    lines.append("입력 해석: UTF-8·쉼표 구분·첫 행 헤더, 빈 칸은 " + ("결측" if empty == "missing" else "빈 문자열") + "으로 읽었습니다. "
                 "자료형은 생성된 표에 고정했고 이후 단계와 결과 파일이 같은 자료형을 씁니다.")
    return lines


def diagnosis_view(record):
    return shared.diagnosis_view(record)


# ==========================================
# 권장안·계획·질문 (CSV v1 어댑터와 같은 규칙)
# ==========================================

def plan_fields(record):
    return shared.plan_fields(record)


def base_plan(record):
    return shared.base_plan(record)


def recommendation_input(record):
    return shared.recommendation_input(record)


def questions(record, plan):
    return shared.questions(record, plan)


def on_rejected(record, draft):
    return shared.on_rejected(record, draft)


def option_guide(record, plan, origins):
    return shared.option_guide(record, plan, origins)


def _numeric_types(record, plan):
    """수치형 역할은 생성된 표에서 수치 자료형으로 읽힌 컬럼만 허용한다. 자동으로 다른 역할로 바꾸지 않는다."""
    groups = {column["column"]: column["dtype_group"] for column in record["diagnostic"]["columns"]}
    wrong = [name for name, role in plan["roles"].items() if role == "수치형" and groups.get(name) != "numeric"]
    if wrong:
        raise PlanError("NUMERIC_ROLE_TYPE_UNSUPPORTED", CODE_LABELS["NUMERIC_ROLE_TYPE_UNSUPPORTED"],
                        subjects=["roles." + name for name in wrong])


def validate_plan(record, plan):
    """CSV v1과 같은 계획 검증에 생성된 표의 자료형 검사와 v2 작업 카탈로그 검증을 더한다."""
    profile, preparation = rules.validate_plan(record["diagnostic"]["columns"], plan)
    _numeric_types(record, plan)

    try:
        _proposal("task_check", profile, preparation)
    except (ValueError, TypeError) as error:
        raise PlanError("CATALOGUE_PLAN_INVALID", CODE_LABELS["CATALOGUE_PLAN_INVALID"]) from error

    return profile, preparation


def _selected(profile, preparation):
    return {"tabular_split_review": asdict(profile.split_rules), "tabular_partition_export": {},
            "tabular_prepare_candidate": asdict(preparation), "tabular_prepare_apply": {}, "tabular_local_handoff": {}}


def _specifications():
    return {item["module_id"]: item for item in workflow_proposal_specifications(include_tabular=True) if item["module_id"] in TABULAR_IDS}


def _proposal(label, profile, preparation):
    selected = _selected(profile, preparation)
    context = {"allowed_modules": ALLOWED, "selected_settings": selected, "existing_inputs": ()}
    specs = _specifications()
    steps, previous = [], []

    for step_id, module_id, _ in STEPS:
        steps.append({"step_id": step_id, "module_id": module_id, "version": TABULAR_VERSION,
                      "inputs": specs[module_id]["inputs"], "settings": selected[module_id], "depends_on": previous})
        previous = [step_id]

    response = {"schema_version": "llm_workflow_proposal_v1", "status": "proposed", "questions": [],
                "workflow": {"workflow_id": label, "steps": steps}}
    return validate_workflow_proposal(json.dumps(response, ensure_ascii=False), **context), context


def approval_stages(module_id):
    return _specifications()[module_id]["approval_boundaries"]


def prepare(record, directory, plan, label):
    """확정 계획을 v2 프로파일·전처리 계획·작업 계획 파일로 새로 만든다. 기존 파일은 덮어쓰지 않는다."""
    profile, preparation = validate_plan(record, plan)
    # 같은 역할 검사를 거친 프로파일에 v2 입력 계약의 어댑터 ID만 명시한다. CSV v1 기본값은 바꾸지 않는다.
    profile = replace(profile, profile_id=label, adapter_id=PROFILE_ADAPTER)
    saved_profile = save_profile(profile, root=directory / "profiles")
    plan_path = shared._write_preparation_plan(directory, label, plan["preparation"], preparation)
    proposal, context = _proposal(label, profile, preparation)
    plans.save_workflow(label, proposal, context, root=shared._plan_root(directory))
    return {"profile_path": saved_profile["local_file"], "profile_signature": saved_profile["profile_signature"],
            "preparation_plan": str(plan_path), "plan_name": label, "plan_signature": proposal.plan_signature}


# ==========================================
# 동의 내용·실행 대조값
# ==========================================

def consent_c_content(record):
    plan = shared._plan(record)
    return {"profile_signature": record["confirmed"]["profile_signature"], "preparation": plan["preparation"],
            "split": plan["split"], "source_contract_id": _source(record).source_id,
            "tabular_code_id": tabular_code_id(), "binding_code_id": binding_code_id(), "delegated_reviews": ["review_eda"]}


def consent_c_text():
    return shared.consent_c_text()


def consent_d_content(record):
    return {"candidate_signature": review_signature(record), "source_contract_id": _source(record).source_id,
            "tabular_code_id": tabular_code_id(), "binding_code_id": binding_code_id()}


def consent_d_text():
    return shared.consent_d_text()


def expected(record, decision):
    """위임 발급 직전 동의 내용과 같아야 하는 현재 코드 지문·출처·후보 서명."""
    values = {"tabular_code_id": tabular_code_id(), "binding_code_id": binding_code_id(),
              "source_contract_id": _source(record).source_id}

    if decision == "D":
        values["candidate_signature"] = review_signature(record)

    return values


# ==========================================
# 단계 생성·입력 계보 대조
# ==========================================

def _values(record, step_id, name):
    """v2 생성 폼 값을 조정기가 채운다. 사용자는 이름·경로를 입력하지 않는다."""
    steps, source = record["steps"], _source(record)
    source_id = source.source_id
    if step_id == "split":
        return {"name": name, "source_run": source.materialization.run_directory, "source_contract_id": source_id,
                "profile": record["confirmed"]["profile_path"]}
    if step_id == "export":
        return {"name": name, "split_name": steps["split"]["run_name"], "source_contract_id": source_id,
                "max_partition_bytes": 32 * 1024 ** 2}
    if step_id == "candidate":
        return {"name": name, "split_name": steps["split"]["run_name"], "export_name": steps["export"]["run_name"],
                "source_contract_id": source_id, "pipeline": record["confirmed"]["preparation_plan"],
                "max_candidate_bytes": 4 * 1024 ** 2, "max_partition_bytes": 32 * 1024 ** 2}
    if step_id == "apply":
        return {"pipeline_name": steps["candidate"]["run_name"], "source_contract_id": source_id}
    return {"name": name, "pipeline_name": steps["candidate"]["run_name"], "source_contract_id": source_id,
            "max_bundle_bytes": 128 * 1024 ** 2}


def check_saved_plan(record, directory):
    return shared.check_saved_plan(record, directory)


def create_step(record, directory, step_id, name):
    """기존 결합·생성 API로 한 단계를 만든다. 승인·실행은 공통 조정기가 한다."""
    # 결합과 생성 직전 재결합이 완료 단계 증거를 다시 검증한다. 이 호출 안에서만 같은 표형 묶음 검증을 공유한다.
    with verification_scope():
        return _create_step(record, directory, step_id, name)


def _create_step(record, directory, step_id, name):
    saved = check_saved_plan(record, directory)
    completed = {step: item["run_name"] for step, item in record["steps"].items() if item["status"] == "completed"}
    binding = bind_workflow_step(saved["proposal"], **saved["context"], step_id=step_id,
                                 values=_values(record, step_id, name), limits=WorkerLimits(**record["limits"]),
                                 completed_runs=completed)
    plans.begin_creation(saved, binding, root=shared._plan_root(directory))
    result = create_bound_step(binding, confirmed=True)
    plans.finish_creation(saved, binding, result, root=shared._plan_root(directory))


def check_lineage(record):
    """분할 작업이 이 작업의 생성 출처로 만들어졌는지 확인한다."""
    if "split" not in record["steps"]:
        return

    _, tool, _ = service("tabular_split_review")._open(record["steps"]["split"]["run_name"])

    if source_of("tabular_split_review", tool) != _source(record).source_id:
        raise TaskProblem("SNAPSHOT_CHANGED", "분할 작업의 입력이 진단한 작업용 표와 다릅니다. 새 작업으로 시작하세요.")

    shared.check_profile(record, tool.contract.profile)


def quality_findings(record):
    """분할 전 품질 검사가 멈춘 경우의 확인된 지적. 실행 계약의 프로파일로 저장된 보고서를 다시 검증한 것만 쓴다."""
    from .quality_result_validation import read_quality_report
    from .task_orchestrator import _check_names

    split = record["steps"].get("split")

    if split is None:
        return None

    _check_names(record)
    _, tool, manager = service("tabular_split_review")._open(split["run_name"])
    task = manager.approvals.get_task(manager.scope_for("audit"))

    if task is None:
        return None

    import tempfile
    from pathlib import Path
    from .artifacts import _root
    from .isolated_eda import _read_file
    from .worker_paths import _path

    # 품질 검토로 멈춘 실행은 커밋되지 않아 완료 표식이 없다. 두 파일을 한 번만 읽어 비공개 사본에 두고 그 사본을 검증·해석해,
    # 보이는 지적이 검증한 바이트와 같게 한다(검증 뒤 교체 방지).
    folder = manager._output(task) / "quality"
    if {path.name for path in _root(folder).iterdir()} != {"quality.json", "quality.md"}:
        return None
    with tempfile.TemporaryDirectory() as scratch:
        copy = Path(scratch)
        for name in ("quality.json", "quality.md"):
            (copy / name).write_bytes(_read_file(_path(folder / name, directory=False), 1024 * 1024))
        report = read_quality_report(copy, tool.contract.profile)

    return None if report is None else [{"code": item["code"], "subject": item.get("column")} for item in report["findings"]]


def split_stop(record):
    """입력 차단으로 멈춘 분할 실행의 단계와, 분할 제안에서 멈췄으면 검증한 제안의 구간별 빠진 target 클래스.

    제안은 커밋되지 않아 완료 표식이 없다. 품질 지적과 같이 두 파일을 한 번만 읽은 비공개 사본을 기존 제안 검사로 검증하고,
    현재 계약의 분할 규칙과 같고 클래스 누락 경고만 있는 제안일 때만 지적을 돌려준다. 확인하지 못하면 지적은 None이다.
    """
    from .task_orchestrator import _check_names

    split = record["steps"].get("split")

    if split is None:
        return None, None

    _check_names(record)
    _, tool, manager = service("tabular_split_review")._open(split["run_name"])
    state = tool.states.load()

    if state.status != "blocked_input" or state.current_stage != "propose_split":
        return state.current_stage if state.status == "blocked_input" else None, None

    task = manager.approvals.get_task(manager.scope_for("propose_split"))

    if task is None:
        return state.current_stage, None

    import tempfile
    from pathlib import Path
    from .artifacts import _root
    from .isolated_eda import _read_file
    from .worker_paths import _path

    folder = manager._output(task)
    names = {"proposal.json": 8 * 1024 * 1024, "proposal.md": 65536}

    # 근거를 검증하지 못해도 멈춘 단계는 확인된 값이다. 단계는 유지하고 지적만 보이지 않는다.
    try:
        if {path.name for path in _root(folder).iterdir()} != set(names):
            return state.current_stage, None
        with tempfile.TemporaryDirectory() as scratch:
            copy = Path(scratch)
            for name, maximum in names.items():
                (copy / name).write_bytes(_read_file(_path(folder / name, directory=False), maximum))
            proposal = tool._check_rows(copy, "propose_split")
    except (OSError, ValueError, KeyError, TypeError):
        return state.current_stage, None

    if (proposal["rules"] != asdict(tool.contract.profile.split_rules) or proposal["status"] != "review_required"
            or proposal["warnings"] != ["CLASS_MISSING_IN_SPLIT"]):
        return state.current_stage, None

    classes = {row["split"]: {str(item["label"])[:80] for item in row["classes"]} for row in proposal["summary"]}
    every = sorted(set().union(*classes.values()))
    missing = {name: [label for label in every if label not in classes[name]] for name in SPLIT_NAMES}
    return state.current_stage, {name: labels for name, labels in missing.items() if labels} or None


CLASS_MISSING_MESSAGE = ("분할 전 품질 검사는 통과했지만, 분할 제안에서 일부 구간에 target 클래스가 없어 멈췄습니다(현재 설정: 모든 구간에 "
                         "모든 클래스 요구). 아래 구간·클래스를 확인하세요. 분할 비율·시드·group 기준을 바꾸거나, 클래스가 빠진 구간을 "
                         "허용하려면 계획 양식의 '모든 구간에 모든 클래스 요구'를 끈 뒤 다시 확인하면 새 동의로 이어서 진행합니다.")
SPLIT_REVIEW_MESSAGE = ("분할 제안 단계에서 멈췄습니다. 제안 근거를 확인하지 못했습니다. 분할 방식·비율·시드와 target·group 역할, "
                        "'모든 구간에 모든 클래스 요구' 설정을 확인한 뒤 다시 확인하세요.")
SPLIT_BLOCKED_MESSAGE = ("분할 단계에서 입력 차단으로 멈췄습니다(품질 검사 지적 아님). 역할·분할 기준을 확인한 뒤 다시 확인하면 새 동의로 "
                         "이어서 진행합니다.")


def blocked_problem(record, step_id):
    """분할 실행이 입력 차단으로 멈췄을 때 멈춘 단계에 맞는 원인을 보이고 계획을 고쳐 이어 갈 수 있게 한다.

    품질 검사(audit)나 단계를 확인하지 못한 경우는 CSV v1과 같은 품질 문구를 쓴다. 분할 제안의 클래스 누락은 검증한 제안 근거로만 보인다.
    """
    if step_id != "split":
        return None

    stage, missing = shared._safe(lambda: split_stop(record)) or (None, None)

    if stage == "propose_split":
        if missing:
            # 작업 기록에는 구간 이름만 둔다. 라벨 값은 화면에 보일 때 검증한 제안 근거에서 다시 읽는다(display_findings).
            return TaskProblem("SPLIT_CLASS_MISSING", CLASS_MISSING_MESSAGE, retryable=True,
                               detail={"CLASS_MISSING_IN_SPLIT": len(missing)},
                               findings=[{"code": "CLASS_MISSING_IN_SPLIT", "subject": SPLIT_NAMES[name]} for name in missing])
        return TaskProblem("SPLIT_REVIEW", SPLIT_REVIEW_MESSAGE, retryable=True)

    if stage not in (None, "audit"):
        return TaskProblem("SPLIT_BLOCKED", SPLIT_BLOCKED_MESSAGE, retryable=True)

    findings = shared._safe(lambda: quality_findings(record)) or []
    counts = {}

    for item in findings:
        counts[item["code"]] = counts.get(item["code"], 0) + 1

    message = ("분할 전 품질 검사가 검토 필요 또는 차단으로 중단했습니다. 아래 확인된 지적 컬럼을 제외하거나 역할·기준을 고친 뒤 "
               "다시 확인하면 새 분할 작업으로 이어서 진행합니다." if findings else
               "분할 전 품질 검사가 검토 필요 또는 차단으로 중단했습니다. 지적 목록을 확인하지 못했습니다. 값이 하나뿐인 컬럼·높은 결측 "
               "컬럼을 제외하거나 역할·기준을 고친 뒤 다시 확인하세요.")
    return TaskProblem("QUALITY_REVIEW", message, retryable=True, detail=counts, findings=findings)


def display_findings(record, findings):
    """화면 표시용 지적. 클래스 누락은 검증한 제안 근거에서 구간별 빠진 라벨을 다시 읽어 붙이고 기록에는 저장하지 않는다.

    다시 확인하지 못하면 저장된 구간 이름만 보인다(라벨을 추정해 채우지 않음).
    """
    if not any(item["code"] == "CLASS_MISSING_IN_SPLIT" for item in findings):
        return findings

    stage, missing = shared._safe(lambda: split_stop(record)) or (None, None)

    if stage != "propose_split" or not missing:
        return findings

    return [{"code": "CLASS_MISSING_IN_SPLIT", "subject": SPLIT_NAMES[name] + " · " + label}
            for name, labels in missing.items() for label in labels]


def failure_hint(record, step_id, diagnostic):
    return shared.failure_hint(record, step_id, diagnostic)


# ==========================================
# 결정 D 대상과 결과 요약 (결합 읽기만 쓴다)
# ==========================================

def _verified_step(record, step_id):
    step = record["steps"][step_id]
    result = service(step["module_id"]).results(step["run_name"])

    if result["status"] != "completed":
        raise ValueError("완료되지 않은 단계가 있어 결과를 만들지 않습니다.")

    return result


def step_completed(record, step_id):
    """완료 기록 복구용 단계 확인. v2는 현재 코드의 결과 검증만 허용한다(읽기 전용 과거 결과 검증 없음)."""
    _verified_step(record, step_id)
    return "current_code"


def review_signature(record):
    return _pipeline(record["candidate"]["run_name"], typed=True)[3].signature


def review(record):
    """결정 D에서 보여 줄 후보. 단계 결과 검증 뒤 후보 증거의 표식 해시에 묶인 바이트로 표와 예상 영향을 만든다."""
    name = record["steps"][REVIEW_STEP]["run_name"]
    _verified_step(record, REVIEW_STEP)
    sources = candidate_sources(name)
    return {"run_name": name, "signature": sources["signature"],
            "facts": shared.review_facts(record, sources["candidate"], sources["preprocessor"], sources["text"])}


def images(record):
    """화면 표시용 그래프. 분할 단계 결과를 검증한 뒤 완료 표식에 묶인 PNG 바이트만 돌려준다."""
    return [item for item in verified_images(_verified_step(record, "split")) if shared._valid_png(item["data"])]


FILE_USES = dict(shared.FILE_USES, **{
    "prepared/train_rows.json": "Train 행 위치와 target·식별자 등 보존 컬럼(원래 자료형 보존). load_handoff_partition으로 읽기",
    "prepared/validation_rows.json": "Validation 행 위치와 보존 컬럼(원래 자료형 보존). load_handoff_partition으로 읽기",
    "README.md": "전달 묶음 안내와 자료형 보존 읽기 예제",
})


def report_facts(record):
    """v2 보고 근거(report_sources: 단계 재검증·단계 간 연결·해시 결합 읽기)로 보고서를 조립한다. 계산·추정은 하지 않는다."""
    if set(record["steps"]) != set(STEP_LABELS):
        raise ValueError("작업 기록의 단계 구성이 완료 작업과 다릅니다.")

    check_lineage(record)
    source = _source(record)
    runs = {step_id: item["run_name"] for step_id, item in record["steps"].items()}
    facts = report_sources(runs, source.source_id)
    handoff = facts["handoff"]
    report = shared.report_from_sources(record, {
        "quality": facts["quality"], "proposal": facts["proposal"], "candidate": facts["candidate"],
        "preprocessor": facts["preprocessor"], "application": facts["application"], "files": handoff["files"],
        "bundle": handoff["directory"], "reports": facts["reports"],
        "image_labels": [item["label"] for item in images(record)], "renamed": _renamed(record),
        "notices": input_notices(record), "plot_limit": facts["plot_limit"],
        "input": {"source_path": record["source"]["path"], "snapshot_sha256": record["snapshot"]["snapshot"]["sha256"],
                  "source_contract_id": source.source_id, "read_spec": facts["source"]["read_spec"],
                  "inspection_id": facts["source"]["inspection_id"], "manifest_sha256": handoff["manifest_sha256"]}})
    report["files"] = [dict(row, 용도=FILE_USES.get(row["파일"], row["용도"])) for row in report["files"]]
    report["usage"].insert(2, "rows.json은 target·식별자의 원래 자료형(예: 문자열 '0012', 정수 1)을 보존합니다. 전달 묶음의 "
                              "load_handoff_partition으로 읽으면 특징 표와 같은 행 순서·원래 자료형으로 받습니다(README 예제).")
    # 입력 해석 설정의 사용자용 설명은 참고 사항(input_notices)에 있고 원래 키(read_spec)는 기술 상세에 있다. 본문에는 생성 결과만 둔다.
    report["lines"].append(f"생성된 표: {facts['source']['rows']}행·{len(facts['source']['columns'])}열 (입력 해석 설정은 참고 사항 참조)")
    # 처리 전 EDA·계획과 결정 절은 v2 보고 근거(검증한 같은 EDA 바이트·승인된 실행 값)로만 만든다. v1·배열 보고서에는 없다.
    plan = shared._plan(record)
    report["eda_tables"] = evidence.eda_tables(facts["eda"], record["diagnostic"], plan)
    report["eda_note"] = evidence.EDA_SCOPE_NOTE
    report["decision_tables"] = evidence.decision_tables(
        record, plan, shared._origin, _recommendation(record), facts["candidate"], facts["application"],
        shared._expected_impact(facts["candidate"], facts["preprocessor"]), facts["split_rules"])
    # 보고서 그래프는 위 분할 결과 표·target 표와 같은 검증된 값으로만 만든다. 그림 파일을 새로 쓰지 않는다.
    report["charts"] = [evidence.split_chart(plan["split"], facts["proposal"]), evidence.target_chart(facts["eda"])]
    report["cautions"] = [EDA_CAUTION if text.startswith("분할 EDA의 사람 검토 단계") else text for text in report["cautions"]]
    return report


EDA_CAUTION = ("분할 EDA의 사람 검토 단계는 결정 C 동의로 위임 승인했습니다. 처리 전 EDA 요약은 이 보고서의 '처리 전 데이터와 기본 EDA' 절에, "
               "그래프는 '그래프' 절에, 전체 EDA 표는 기술 상세의 보고서 경로와 같은 폴더에 있습니다.")


def _recommendation(record):
    """권장안 사용 여부 한 줄. 거부 사유의 상세는 참고 사항(recommendation_notice)에 있다."""
    from .task_report import code_label

    draft = record["draft"]

    if draft["source"] == "model":
        return (f"사용함: 모델 변경 제안 {len(draft.get('changes') or [])}개. 확정 계획에 남은 항목은 결정 출처 '모델 권장'으로 표시합니다.")

    return "사용 못 함: 기본 계획에서 시작(" + (code_label(record, draft.get("code")) or "사유 코드 확인 불가") + "). 상세 사유는 참고 사항에 있습니다."
