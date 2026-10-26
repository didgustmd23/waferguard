"""LLM workflow를 현재 한 CSV 생성 범위에 결합한다. 승인 발급·worker 실행은 하지 않는다."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re

from src.preprocessing.profiles import profile_from_dict
from .csv_workflow_creation import CsvCreationPreview, code_id as csv_code_id
from .csv_workflow_binding_evidence import completed_workflow
from .csv_workflow_proposal_catalogue import workflow_proposal_specifications
from .llm_workflow_contract import ValidatedWorkflowProposal, validate_workflow_proposal
from .llm_workflow_proposal import checked_context, restore_context
from .preparation_module_plan import validate_preparation_modules
from .split_evidence import read_metadata
from .worker_contract import WorkerLimits
from .workflow_service_registry import service_version, workflow_service


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def binding_code_id():
    """바인더·정적 제안·기존 서비스 연결 변경도 생성 전 재검사한다.

    저장 실행안 검사·생성 기록(store)과 결합 검사가 쓰는 전처리 모듈 계획 해석도 단계 생성 판정에 관여하므로 포함한다.
    """
    files = ("llm_workflow_binding.py", "csv_workflow_binding_evidence.py", "llm_workflow_contract.py",
             "llm_workflow_proposal.py", "csv_workflow_proposal_catalogue.py", "llm_workflow_store.py",
             "preparation_module_plan.py")
    from .tabular_workflow_service import code_id as tabular_code_id

    return _digest({"csv": csv_code_id(), "tabular": tabular_code_id(), "files": {
        name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() for name in files}})


@dataclass(frozen=True)
class WorkflowStepBinding:
    """로컬 전용 미리보기. 경로·증거를 포함하므로 LLM payload로 사용하지 않는다."""

    payload_json: str

    @property
    def packet(self):
        return json.loads(self.payload_json)

    @property
    def binding_id(self):
        return _digest(self.packet)


def _run_names(value, keys):
    if (type(value) is not dict or set(value) != set(keys)
            or any(type(name) is not str or re.fullmatch(r"[a-z][a-z0-9_-]{0,47}", name) is None
                   for name in value.values())):
        raise ValueError("필수 선행 참조마다 정확한 기존 작업 이름을 지정하세요.")


def bind_workflow_step(proposal, *, allowed_modules, selected_settings, step_id, values, limits,
                       completed_runs=None, existing_inputs=(), existing_runs=None):
    """전체 제안을 재검증하고 지금 준비할 한 단계만 결합한다. 파일을 생성하지 않는다.

    completed_runs는 이번 제안의 앞 단계 ID→완료 작업 이름이다. existing_runs는
    중간 시작을 위해 명시한 논리 참조→완료 작업 이름이다. 둘 모두 실제 로컬 검증을 통과해야 한다.
    """
    if not isinstance(proposal, ValidatedWorkflowProposal) or not isinstance(limits, WorkerLimits):
        raise TypeError("검증된 workflow 제안과 WorkerLimits가 필요합니다.")

    context_json = checked_context(allowed_modules=allowed_modules, selected_settings=selected_settings,
                                   existing_inputs=existing_inputs)
    context = restore_context(context_json)
    checked = validate_workflow_proposal(proposal.payload_json, **context)

    if checked != proposal or checked.plan_signature is None:
        raise ValueError("동일한 확정 제안이 필요합니다. 확인 질문은 실행에 연결하지 않습니다.")

    steps = checked.to_dict()["proposal"]["workflow"]["steps"]
    positions = [index for index, step in enumerate(steps) if step["step_id"] == step_id]

    if len(positions) != 1:
        raise ValueError("고정 제안에 존재하는 단계 ID가 필요합니다.")

    index = positions[0]
    step = steps[index]
    completed = {} if completed_runs is None else completed_runs
    existing = {} if existing_runs is None else existing_runs
    _run_names(completed, [item["step_id"] for item in steps[:index]])
    _run_names(existing, existing_inputs)
    specs = {item["output"]: item for item in workflow_proposal_specifications(include_tabular=True)}
    references, evidence, lineage = {}, {}, {}

    def add(reference, module_id, name, settings=None):
        proof = completed_workflow(module_id, name, settings)

        for key, identity in proof["lineage"].items():
            if key in lineage and lineage[key] != identity:
                raise ValueError("선행 작업의 출처·분할·내보내기·후보 연결이 서로 다릅니다.")

            lineage[key] = identity

        references[reference] = name
        evidence[reference] = proof

    for reference, name in existing.items():
        add(reference, specs[reference]["module_id"], name)

    for prior in steps[:index]:
        reference = next(key for key, spec in specs.items() if spec["module_id"] == prior["module_id"])
        add(reference, prior["module_id"], completed[prior["step_id"]], prior["settings"])

    service = workflow_service(step["module_id"], step["version"])
    native = service.preview_form(values, limits)
    selected = native.packet["values"]
    module = step["module_id"]
    # CSV v1·표형 v2는 같은 역할의 연결 규칙을 쓰고, v2 논리 참조에는 접두사가 붙는다.
    role = module.split("_", 1)[1]
    prefix = "tabular_" if module.startswith("tabular_") else ""

    if role == "split_review":
        profile = profile_from_dict(read_metadata(Path(selected["profile"])))

        if profile.split_rules is None or asdict(profile.split_rules) != step["settings"]:
            raise ValueError("분할 프로파일 규칙이 검토한 제안과 다릅니다.")
    else:
        connections = {
            "partition_export": {"split_name": "split.confirmed"},
            "prepare_candidate": {"split_name": "split.confirmed", "export_name": "export.development"},
            "prepare_apply": {"pipeline_name": "candidate.saved"},
            "local_handoff": {"pipeline_name": "prepared.saved"},
        }

        for field, reference in connections[role].items():
            if selected[field] != references.get(prefix + reference):
                raise ValueError("생성 입력이 제안에 결합한 정확한 선행 작업과 다릅니다.")

        parents = native.packet["parents"]
        expected = {
            "partition_export": {"split": lineage.get("split")},
            "prepare_candidate": {"split": lineage.get("split"), "train": lineage.get("train")},
            "prepare_apply": {"candidate": lineage.get("candidate")},
            "local_handoff": {"candidate": lineage.get("candidate")},
        }[role]
        if prefix:
            # v2는 모든 선행 결과와 새 생성 입력이 같은 표형 출처 ID에 묶여야 한다.
            expected["source"] = lineage.get("source")

        if any(parents.get(key) != identity or identity is None for key, identity in expected.items()):
            raise ValueError("생성 미리보기의 실제 선행 증거가 결합한 완료 기록과 다릅니다.")

        if role == "local_handoff" and parents["application"]["sha256"] != lineage.get("prepared"):
            raise ValueError("전달 대상의 적용 완료 증거가 변경됐습니다.")

        if role == "prepare_candidate":
            plan = validate_preparation_modules(read_metadata(Path(selected["pipeline"])))

            if json.loads(plan.rules_json) != step["settings"]:
                raise ValueError("전처리 모듈 계획 규칙이 검토한 제안과 다릅니다.")

    return WorkflowStepBinding(_json({
        "schema_version": "llm_workflow_binding_v1", "proposal": asdict(checked),
        "context": json.loads(context_json), "plan_signature": checked.plan_signature,
        "step_id": step_id, "module_id": module, "values": selected, "limits": asdict(limits),
        "completed_runs": completed, "existing_runs": existing, "evidence": evidence,
        "creation": native.packet, "binding_code_id": binding_code_id(),
        "worker_started": False, "execution_approval_issued": False, "local_only": True,
    }))


def create_bound_step(binding, *, confirmed=False):
    """정확한 미리보기를 다시 결합한 뒤 기존 서비스의 생성만 요청한다."""
    if confirmed is not True or not isinstance(binding, WorkflowStepBinding):
        raise ValueError("현재 로컬 생성 범위를 명시적으로 확인해야 합니다.")

    packet = binding.packet
    proposal = ValidatedWorkflowProposal(**packet["proposal"])
    current = bind_workflow_step(proposal, **restore_context(_json(packet["context"])),
        step_id=packet["step_id"], values=packet["values"], limits=WorkerLimits(**packet["limits"]),
        completed_runs=packet["completed_runs"], existing_runs=packet["existing_runs"])

    if current != binding:
        raise ValueError("제안·입력·완료 증거·후보·코드·자원이 미리보기 이후 변경됐습니다.")

    service = workflow_service(packet["module_id"], service_version(packet["module_id"]))
    result = service.create(CsvCreationPreview(_json(packet["creation"])), confirmed=True)
    return {**result, "binding_id": binding.binding_id, "plan_signature": packet["plan_signature"],
            "step_id": packet["step_id"], "execution_approval_issued": False, "local_only": True}
