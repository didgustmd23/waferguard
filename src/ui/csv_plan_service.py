"""공통 5단계 UI의 CSV 실행안·단계별 작업 연결. 렌더링은 실행하지 않는다."""

import json
from pathlib import Path
from threading import Lock

from src.agents.csv_workflow_binding_evidence import completed_workflow
from src.agents.csv_workflow_creation import CsvCreationPreview
from src.agents.llm_workflow_binding import bind_workflow_step, create_bound_step
from src.agents.llm_workflow_contract import validate_workflow_proposal
from src.agents.llm_workflow_proposal import checked_context, restore_context
from src.agents import llm_workflow_store as store
from src.agents.worker_contract import WorkerLimits
from src.agents.isolated_eda import _read_file
from .dialogue_service import UiDialogue, fingerprint
from .workflow_service import UiWorkflow


SCOPE = "CSV 분할·전처리"


def selected(session):
    return session.get("ui_workflow_mode", "LLM 제안") == "LLM 제안" and session.get("ui_proposal_scope") == SCOPE


class CsvPlanDialogue(UiDialogue):
    def __init__(self, settings, context, *, factory=None, ledger=None):
        self.context_json = checked_context(**context)

        if factory is None:
            from src.agents.llm_workflow_dialogue import WorkflowProposalDialogue

            factory = lambda config: WorkflowProposalDialogue(config, **restore_context(self.context_json))

        super().__init__(settings, factory=factory, ledger=ledger)

    def _validate(self, proposal):
        return validate_workflow_proposal(json.dumps(proposal, ensure_ascii=False),
                                          **restore_context(self.context_json)).to_dict()


class BoundCsvWorkflow(UiWorkflow):
    """승인·결과 UI를 재사용하고 생성 시에만 새 바인더를 통과한다."""

    def __init__(self, owner, step):
        super().__init__(step["module_id"], step["version"])
        self.owner, self.step_id = owner, step["step_id"]
        self.reviewed = True
        self.binding = None
        self.candidate_report = None
        self.creation_widget_prefix = "csv_plan_" + owner.saved["selection_id"][:12] + "_" + self.step_id + "_"
        self.creation_defaults = {"name": owner.saved["name"][:16] + "-" + str(owner.index + 1)}
        by_module = {item["module_id"]: owner.runs.get(item["step_id"]) for item in owner.steps}
        self.creation_defaults.update({key: by_module[module] for key, module in (
            ("split_name", "csv_split_review"), ("export_name", "csv_partition_export"),
            ("pipeline_name", "csv_prepare_candidate")) if by_module.get(module)})
        self.creation_confirmation_label = (
            "후보 보고서를 검토했고 저장 기준 적용 작업 준비에 동의합니다"
            if self.module_id == "csv_prepare_apply" else "위 설정으로 새 작업 기록만 생성합니다"
        )

    def preview_creation(self, values, limits):
        self.creation_preview = None
        self.creation_selection = None
        self.binding = None
        self.candidate_report = None
        saved = self.owner.recheck()
        binding = bind_workflow_step(saved["proposal"], **saved["context"], step_id=self.step_id,
            values=values, limits=WorkerLimits(**limits), completed_runs=self.owner.previous_runs())
        proof = binding.packet["evidence"].get("candidate.saved")

        if proof and proof["local_reports"].get("candidate"):
            text = _read_file(Path(proof["local_reports"]["candidate"]), 65536).decode("utf-8")

            if completed_workflow("csv_prepare_candidate", proof["run_name"]) != proof:
                raise ValueError("보고서 표시 복사본을 준비하는 중 후보가 변경됐습니다.")

            self.candidate_report = text

        self.binding = binding
        self.creation_preview = CsvCreationPreview(json.dumps(binding.packet["creation"], ensure_ascii=False, sort_keys=True))
        self.creation_selection = fingerprint({"values": values, "limits": limits})
        return self.creation_preview.packet

    def create(self, values, limits, *, confirmed=False):
        if (confirmed is not True or self.binding is None or self.creation_preview is None
                or self.creation_receipt is not None
                or self.creation_selection != fingerprint({"values": values, "limits": limits})):
            raise ValueError("현재 입력의 바인딩과 새 생성 확인이 필요합니다.")

        saved = self.owner.recheck()
        binding = self.binding
        self.creation_preview = None
        self.binding = None
        # 중단되면 intent를 남긴다. 다음 렌더링·복원에서 생성 요청을 반복하지 않는다.
        store.begin_creation(saved, binding, root=self.owner.root)
        result = create_bound_step(binding, confirmed=True)
        store.finish_creation(saved, binding, result, root=self.owner.root)
        self.owner.runs = store.created_runs(saved, root=self.owner.root)
        self.creation_receipt = result
        self.name = result["run_name"]
        self.state = None
        return result


class CsvPlanSession:
    def __init__(self, saved, *, root=None):
        steps = saved["proposal"].to_dict()["proposal"]["workflow"]["steps"]

        if saved["context"]["existing_inputs"] or steps[0]["module_id"] != "csv_split_review":
            raise ValueError("초기 UI는 새 분할부터 시작하는 연속 실행안을 지원합니다.")

        self.root = Path(saved["directory"]).parent if root is None else Path(root)
        self.saved = saved
        self.steps = steps
        self.index = 0
        self.runs = {}
        self.active = None
        self._lock = Lock()

    def recheck(self):
        current = store.load_workflow(self.saved["name"], root=self.root)

        if current["selection_id"] != self.saved["selection_id"]:
            raise ValueError("저장 실행안이 변경됐습니다.")

        return current

    def previous_runs(self):
        return {step["step_id"]: self.runs[step["step_id"]] for step in self.steps[:self.index]}

    def restore(self):
        """명시적 불러오기만 수행한다. 승인 ID·모델 명령·실행 pending은 복원하지 않는다."""
        self.runs = store.created_runs(self.recheck(), root=self.root)
        self.index = 0

        for index, step in enumerate(self.steps):
            self.index = index
            name = self.runs.get(step["step_id"])

            if name is None:
                break

            workflow = BoundCsvWorkflow(self, step)
            workflow.connect(name)
            self.active = workflow

            if workflow.state["status"] != "completed":
                if any(later["step_id"] in self.runs for later in self.steps[index + 1:]):
                    raise ValueError("미완료 작업 뒤의 생성 기록을 자동 연결하지 않습니다.")

                return

            if step["module_id"] == "csv_local_handoff":
                workflow.service().results(name)
            else:
                completed_workflow(step["module_id"], name, step["settings"])

        self.active = BoundCsvWorkflow(self, self.steps[self.index])
        name = self.runs.get(self.steps[self.index]["step_id"])

        if name:
            self.active.connect(name)

    def advance(self):
        """사용자 요청으로 완료 증거를 확인하고 다음 빈 폼만 선택한다."""
        if not self._lock.acquire(blocking=False):
            raise ValueError("다른 단계 선택 요청을 처리 중입니다.")

        try:
            return self._advance()
        finally:
            self._lock.release()

    def _advance(self):
        self.runs = store.created_runs(self.recheck(), root=self.root)

        if self.active is None or self.active.name is None or self.index >= len(self.steps) - 1:
            raise ValueError("다음 단계로 진행할 완료 작업이 없습니다.")

        step = self.steps[self.index]

        if self.runs.get(step["step_id"]) != self.active.name:
            raise ValueError("현재 화면 작업이 저장 실행안의 연결과 다릅니다.")

        completed_workflow(step["module_id"], self.active.name, step["settings"])
        self.index += 1
        self.active = BoundCsvWorkflow(self, self.steps[self.index])
