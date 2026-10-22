"""테스트 전용 합성 배열 작업 어댑터. 도메인 프로파일 없는 배열 표본 모음을 공통 조정기·질문·승인·보고 경로로 진행한다.

제품 등록부에는 넣지 않고 테스트에서 task_adapters.ADAPTERS에 등록한다. 계산은 테스트 프로세스 안에서 하지만
실행 상태·1회 승인·승인 소비·결과 무결성은 실제 LocalStateStore·ApprovalStore·RulesExecutor가 맡고,
위임 발급은 공통 delegated_approval.record_delegated를 쓴다. 격리 worker는 이 어댑터의 검증 대상이 아니다.
"""

import hashlib
import json
from pathlib import Path
import threading

from src.agents.approval_store import ApprovalStore
from src.agents.cli import _current_actor
from src.agents.contracts import ToolResult
from src.agents.delegated_approval import record_delegated
from src.agents.executor import ExecutionPlan, ExecutionStep, RegisteredTool, RulesExecutor
from src.agents.state import LocalStateStore
from src.agents.task_plan import PlanError, TaskProblem, question, signature
from src.agents.workflow_approval import approval_progress, preview_approval


KIND = "array_synthetic"
INPUT = {"label": "배열 표본 파일", "picker": "csv_file", "help": "테스트 전용"}
STEPS = (("review", "array_review", "C"), ("apply", "array_apply", "D"))
RUN_ALIAS = {}
REVIEW_STEP = "review"
STEP_LABELS = {"review": "표본 정리 계획", "apply": "표본 정리 적용"}
CODE_LABELS = {"PLAN_FIELDS_INVALID": "배열 계획 항목 구성이 계약과 다릅니다",
               "GROUP_MISSING": "그룹 기준 분할에는 그룹 값이 필요합니다"}
HINTS = {}
ACTIONS = {}
STAGES = {"array_profile": "audit", "array_review": "derive_preparation", "array_apply": "apply_preparation"}
ROOT = None
TERMINAL = {"completed", "blocked_input", "failed_runtime", "security_locked", "cancelled"}


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _samples(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ==========================================
# 실제 실행기·승인 저장소 위의 합성 서비스 (공통 서비스 API)
# ==========================================

class _RunTool:
    """workflow_approval.preview_approval이 요구하는 실행 도구 형태."""

    def __init__(self, states, arguments):
        self.states, self.arguments, self.cancel = states, arguments, threading.Event()

    def _check_arguments(self, arguments):
        if arguments != self.arguments:
            raise ValueError("인자가 바뀌었습니다.")


class Service:
    def __init__(self, module_id):
        if module_id not in STAGES:
            raise ValueError("등록되지 않은 합성 모듈입니다.")

        self.module_id, self.stage = module_id, STAGES[module_id]

    def _directory(self, name):
        return ROOT / self.module_id / name

    def create(self, name, *, source, settings, limits, parent=None):
        directory = self._directory(name)
        directory.mkdir(parents=True, exist_ok=False)
        config = {"source": str(source), "source_sha256": _digest(source), "settings": settings, "limits": limits,
                  "parent": None if parent is None else str(parent)}
        (directory / "config.json").write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
        LocalStateStore(directory / "run").create(_current_actor(), policy_version="array_synthetic_v1")

    def _open(self, name):
        directory = self._directory(name)
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        states = LocalStateStore(directory / "run")
        arguments = {"settings": config["settings"]}
        plan = ExecutionPlan(source_id="array:" + config["source_sha256"], config_id=signature(config),
                             policy_version="array_synthetic_v1",
                             steps=(ExecutionStep("preflight", "run_preflight"),
                                    ExecutionStep(self.stage, _ACTIONS[self.stage], json.dumps(arguments))))
        tools = (RegisteredTool("run_preflight", _preflight, lambda output, arguments: (output / "preflight.json").is_file()),
                 RegisteredTool(_ACTIONS[self.stage], lambda output, arguments: self._execute(config, output, arguments),
                                lambda output, arguments: (output / "result.json").is_file(), requires_approval=True))
        manager = RulesExecutor(states, ApprovalStore(directory / "run" / "approvals.sqlite3"), plan, tools)
        return config, _RunTool(states, arguments), manager

    def _summary(self, config):
        def factory(tool):
            # 승인 요약은 단계·자원과 입력 해시를 담는다. 위임 판정이 동의 후 확보한 입력과 대조한다.
            return {"stage": tool.states.load().current_stage, "limits": config["limits"],
                    "source_sha256": _digest(config["source"])}

        return factory

    def _execute(self, config, output, arguments):
        samples = _samples(config["source"])
        settings = arguments["settings"]

        if self.stage == "audit":
            labels = [item["label"] for item in samples if item["label"] is not None]
            arrays = [json.dumps(item["array"]) for item in samples]
            result = {"samples": len(samples), "shapes": sorted({f"{len(item['array'])}x{len(item['array'][0])}" for item in samples}),
                      "label_counts": {label: labels.count(label) for label in sorted(set(labels))},
                      "label_missing": len(samples) - len(labels), "groups": len({item["group"] for item in samples}),
                      "duplicates": len(arrays) - len(set(arrays)),
                      "value_set": sorted({value for item in samples for row in item["array"] for value in row})}
        else:
            kept, seen, dropped_duplicates, dropped_unlabeled = [], set(), 0, 0

            for item in samples:
                key = json.dumps(item["array"])

                if settings["drop_duplicates"] and key in seen:
                    dropped_duplicates += 1
                    continue

                if item["label"] is None and settings["missing_labels"] == "drop":
                    dropped_unlabeled += 1
                    continue

                seen.add(key)
                kept.append(item["sample_id"])

            result = {"kept": kept, "dropped_duplicates": dropped_duplicates, "dropped_unlabeled": dropped_unlabeled}

            if self.stage == "apply_preparation":
                parent = json.loads((Path(config["parent"]) / "result.json").read_text(encoding="utf-8"))

                if parent["kept"] != kept:
                    return ToolResult("blocked_input", errors=("REVIEW_MISMATCH",))

        (output / "result.json").write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
        return ToolResult("passed")

    def status(self, name):
        config, tool, manager = self._open(name)
        state = tool.states.load()
        actions = []

        if not state.blockers and state.status in {"created", "running", "awaiting_approval"}:
            progress = approval_progress(manager, state.current_stage)

            if not progress["task_exists"]:
                actions = [{"id": ("execute" if progress["valid_approval_ids"] else "approve")
                            if state.status == "awaiting_approval" else "next"}]

        return {"status": state.status, "stage": state.current_stage, "revision": state.revision, "actions": actions}

    def preview_approval(self, name):
        config, tool, manager = self._open(name)
        return preview_approval(tool, manager, self._summary(config))

    def record_delegated_approval(self, name, preview, *, authorize):
        config, tool, manager = self._open(name)
        return record_delegated(tool, manager, self._summary(config), preview, authorize=authorize)

    def next(self, name, *, expected_stage, expected_revision, approval_id=None):
        config, tool, manager = self._open(name)
        state = tool.states.load()

        if state.current_stage != expected_stage or state.revision != expected_revision:
            raise ValueError("선택한 단계·revision이 변경됐습니다.")

        return manager.run_next(approval_id=approval_id)

    def output(self, name):
        config, tool, manager = self._open(name)
        task = manager.approvals.get_task(manager.scope_for(self.stage))

        if task is None or task["status"] != "committed":
            raise ValueError("커밋된 결과가 없습니다.")

        return manager._output(task)

    def results(self, name):
        config, tool, manager = self._open(name)

        if tool.states.load().status != "completed":
            raise ValueError("완료된 작업만 조회합니다.")

        output = self.output(name)
        return {"status": "completed", "items": [{"kind": "artifact_list", "label": self.stage, "path": str(output)}]}

    def worker_diagnostic(self, name):
        return None


def _preflight(output, arguments):
    (output / "preflight.json").write_text("{}", encoding="utf-8")
    return ToolResult("passed")


_ACTIONS = {"audit": "run_audit", "derive_preparation": "derive_preparation", "apply_preparation": "apply_preparation"}


def service(module_id):
    return Service(module_id)


def _result(record, step_id):
    step = record["steps"][step_id]
    items = service(step["module_id"]).results(step["run_name"])["items"]
    return json.loads((Path(items[0]["path"]) / "result.json").read_text(encoding="utf-8"))


# ==========================================
# 어댑터 계약 (공통 조정기가 부르는 이름)
# ==========================================

def accepts(path):
    return str(path).endswith(".arrays.json")


def consent_a_content(record):
    return {"transfer": "goal_and_array_aggregates_without_values", "auto_steps": ["input_snapshot", "array_profile"],
            "allowed": [["array_profile", "audit"]]}


def consent_a_text(model):
    return ["읽는 것: 배열 표본 파일의 복사본", f"모델 전달: 목표와 표본 집계만 {model}에 보냅니다."]


def diagnose(record, directory, context):
    name = "t" + record["task_id"][:10] + "-diag"
    record["diagnostic"] = {"run_name": name, "status": "reserved"}
    record = context.save(record)
    profile = service("array_profile")
    profile.create(name, source=Path(record["snapshot"]["directory"]) / "input.bin", settings={}, limits=record["limits"])

    for _ in range(8):
        state = profile.status(name)

        if state["status"] in TERMINAL:
            if state["status"] != "completed":
                raise TaskProblem("DIAGNOSTIC_FAILED", "배열 진단을 완료하지 못했습니다.")

            break

        if state["actions"][0]["id"] == "approve":
            authorize = context.authorizer(record, "A", "array_profile", "audit", expected={"limits": record["limits"]},
                                           bound={"source_sha256": record["snapshot"]["snapshot"]["sha256"]})
            result = profile.record_delegated_approval(name, profile.preview_approval(name), authorize=authorize)
            record["diagnostic"]["delegation"] = {"consent_id": record["consents"]["A"]["consent_id"],
                                                  "scope_sha256": result["scope_sha256"]}
            record = context.save(record)
            current = profile.status(name)
            profile.next(name, expected_stage=current["stage"], expected_revision=current["revision"],
                         approval_id=result["approval_id"])
        else:
            profile.next(name, expected_stage=state["stage"], expected_revision=state["revision"])

    output = profile.output(name)
    record["diagnostic"].update(status="completed", summary=json.loads((output / "result.json").read_text(encoding="utf-8")))
    return record


def input_notices(record):
    return []


def diagnosis_view(record):
    summary = record["diagnostic"]["summary"]
    return {"lines": [f"표본 {summary['samples']}개 · 형상 {', '.join(summary['shapes'])} · 그룹 {summary['groups']}개"],
            "rows": [{"항목": "라벨 결측", "값": summary["label_missing"]}, {"항목": "중복 표본", "값": summary["duplicates"]}]}


def plan_fields(record):
    return [{"path": "options.drop_duplicates", "type": "boolean"}, {"path": "options.group_split", "type": "boolean"},
            {"path": "options.missing_labels", "type": "enum", "options": ["drop", "keep"], "nullable": True}]


def base_plan(record):
    return {"options": {"drop_duplicates": False, "group_split": False, "missing_labels": None}}


def recommendation_input(record):
    # 판단 대상을 라벨별 표본 묶음으로 둔다. 배치·예산은 표형과 같은 공통 계층이 정한다.
    summary = record["diagnostic"]["summary"]
    items = [{"id": "label:" + label, "priority": 1, "reasons": [], "bucket": "label", "paths": [],
              "detail": {"samples": count}} for label, count in sorted(summary["label_counts"].items())]
    return {"data": {"summary": summary}, "instructions": "이 작업은 배열 표본 모음입니다. path는 options.<항목> 형식입니다.",
            "items": items}


def validate_plan(record, plan):
    if type(plan) is not dict or set(plan) != {"options"} or set(plan["options"]) != {"drop_duplicates", "group_split", "missing_labels"}:
        raise PlanError("PLAN_FIELDS_INVALID", CODE_LABELS["PLAN_FIELDS_INVALID"])

    if plan["options"]["group_split"] and record["diagnostic"]["summary"]["groups"] < 2:
        raise PlanError("GROUP_MISSING", CODE_LABELS["GROUP_MISSING"])


def questions(record, plan):
    summary, options, result = record["diagnostic"]["summary"], plan["options"], []

    if summary["label_missing"] and options["missing_labels"] is None:
        result.append(question("options.missing_labels", "required", "라벨이 없는 표본을 어떻게 할지 고르세요.",
                               path="options.missing_labels",
                               options=[{"value": "drop", "label": "제외"}, {"value": "keep", "label": "유지"}]))

    if summary["duplicates"]:
        result.append(question("options.drop_duplicates", "advisory", "같은 배열의 중복 표본이 있습니다. 제외할까요?",
                               path="options.drop_duplicates", recommended=True,
                               options=[{"value": True, "label": "제외"}, {"value": False, "label": "유지"}]))

    return result


def check_resolution(record, action):
    raise ValueError("이 유형은 해결 동작이 없습니다.")


def resolution_facts(record, action):
    return {}


def on_rejected(record, draft):
    return draft


def approval_stages(module_id):
    return [STAGES[module_id]]


def prepare(record, directory, plan, label):
    validate_plan(record, plan)
    path = directory / "plans" / (label + ".json")
    path.parent.mkdir(exist_ok=True)

    with path.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, sort_keys=True)

    return {"plan_name": label, "plan_signature": signature(plan)}


def consent_c_content(record):
    return {"options": record["confirmed"]["plan"]["options"]}


def consent_c_text():
    return "동의 범위(결정 C): 확인한 계획으로 표본 정리 계획을 만듭니다. 실제 적용은 따로 확인합니다."


def consent_d_content(record):
    return {"review_signature": review_signature(record)}


def consent_d_text():
    return "확인한 정리 계획을 적용해 정리된 표본 목록을 만듭니다. 원본은 변경하지 않습니다."


def expected(record, decision):
    return {"review_signature": review_signature(record)} if decision == "D" else {}


def create_step(record, directory, step_id, name):
    parent = service("array_review").output(record["steps"]["review"]["run_name"]) if step_id == "apply" else None
    service(dict((step, module) for step, module, _ in STEPS)[step_id]).create(
        name, source=Path(record["snapshot"]["directory"]) / "input.bin",
        settings=record["confirmed"]["plan"]["options"], limits=record["limits"], parent=parent)


def check_lineage(record):
    for step_id, item in record["steps"].items():
        config = json.loads((ROOT / item["module_id"] / item["run_name"] / "config.json").read_text(encoding="utf-8"))

        if config["source_sha256"] != record["snapshot"]["snapshot"]["sha256"]:
            raise TaskProblem("SNAPSHOT_CHANGED", "작업 입력이 진단한 복사본과 다릅니다.")


def blocked_problem(record, step_id):
    return None


def failure_hint(record, step_id, diagnostic):
    return None


def review_signature(record):
    return hashlib.sha256(json.dumps(_result(record, "review"), sort_keys=True).encode()).hexdigest()


def review(record):
    result = _result(record, "review")
    facts = {"metrics": [{"label": "유지 표본", "value": len(result["kept"])},
                         {"label": "중복 제외", "value": result["dropped_duplicates"]},
                         {"label": "라벨 없음 제외", "value": result["dropped_unlabeled"]}],
             "tables": [{"title": "정리 계획", "rows": [{"항목": key, "값": value} for key, value in sorted(record["confirmed"]["plan"]["options"].items())]}],
             "text": None, "text_truncated": False}
    return {"run_name": record["steps"]["review"]["run_name"], "signature": review_signature(record), "facts": facts}


def images(record):
    return []


def report_facts(record):
    if set(record["steps"]) != set(STEP_LABELS):
        raise ValueError("작업 기록의 단계 구성이 완료 작업과 다릅니다.")

    check_lineage(record)
    summary, applied = record["diagnostic"]["summary"], _result(record, "apply")
    return {"title": "배열 표본 정리 결과 보고서", "lines": [],
            "metrics": [{"label": "입력 표본", "value": summary["samples"]}, {"label": "정리 후 표본", "value": len(applied["kept"])}],
            "problems": [{"구분": "라벨 결측", "내용": summary["label_missing"]}, {"구분": "중복 표본", "내용": summary["duplicates"]}],
            "processing": [{"항목": key, "값": value} for key, value in sorted(record["confirmed"]["plan"]["options"].items())],
            "comparison": [{"항목": "표본 수", "전": summary["samples"], "후": len(applied["kept"])},
                           {"항목": "형상별 분포", "전": "미집계", "후": "미집계"}],
            "images": [], "cautions": ["합성 배열 표본입니다."], "files": [{"파일": "result.json"}], "files_title": "결과 파일",
            "technical": {"source_sha256": record["snapshot"]["snapshot"]["sha256"]}, "outputs": {"bundle": None}}
