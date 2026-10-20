"""기존 메모리 전용 LangGraph의 승인 없는 CSV workflow 대화 확장."""

import json

from .llm_dialogue import ProposalDialogue
from .llm_workflow_contract import validate_workflow_proposal
from .llm_workflow_proposal import build_request_payload, checked_context, propose_request, restore_context


class WorkflowProposalDialogue(ProposalDialogue):
    """규칙은 세션 동안 고정한다. 변경하려면 새 선택값으로 새 대화를 시작한다."""

    def __init__(self, config=None, *, allowed_modules, selected_settings, existing_inputs=()):
        self._context_json = checked_context(allowed_modules=allowed_modules,
                                             selected_settings=selected_settings, existing_inputs=existing_inputs)
        super().__init__(config)

    def _build_payload(self, text, config):
        return build_request_payload(text, config, **restore_context(self._context_json))

    def _propose(self, text):
        return propose_request(text, self.config, backend="langchain",
                               **restore_context(self._context_json)).to_dict()

    def _validate(self, proposal):
        return validate_workflow_proposal(json.dumps(proposal, ensure_ascii=False),
                                          **restore_context(self._context_json)).to_dict()
