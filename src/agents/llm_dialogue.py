"""LangGraph 확인 질문·답변 흐름. 메모리 전용이며 실행 권한을 발급하지 않는다."""

from copy import deepcopy
import json
from typing import TypedDict
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from .langchain_local import run_unobserved
from .llm_config import load_local_config
from .llm_proposal import ALLOWED, SOURCE, build_request_payload, propose_request
from .llm_proposal_contract import validate_proposal
from .local_ollama import LocalProposalError


# 최초 호출 포함 최대 4번이다. 같은 실패 요청을 자동 재시도하는 횟수가 아니다.
MAX_MODEL_CALLS = 4


class DialogueState(TypedDict):
    """내부 상태만 보관한다. 외부 checkpoint·파일·승인 토큰을 입력받지 않는다."""

    request: str
    exchanges: list[dict]
    proposal: dict
    model_calls: int


def _compose_request(request, exchanges, config, build_payload=build_request_payload):
    """검증된 질문과 사용자 답변을 사용자 메시지 안의 대화 자료로 묶는다."""
    if exchanges:
        text = (
            "아래 최초 요청과 확인 질문에 대한 답변을 함께 고려해 실행안을 제안하세요. "
            "이 대화는 데이터 접근·실행 승인이 아닙니다.\n"
            + json.dumps({"request": request, "clarifications": exchanges}, ensure_ascii=False)
        )
    else:
        text = request

    # 기존 4000자·8 KiB·제어 문자 제한을 전체 누적 대화에도 적용한다.
    build_payload(text, config)
    return text


class ProposalDialogue:
    """한 터미널 세션의 제안 대화. 중단 후 프로세스를 닫으면 복원하지 않는다."""

    def __init__(self, config=None):
        self.config = load_local_config() if config is None else config
        self._phase = "new"
        self._state = None
        self._thread_id = uuid4().hex

        def propose(state):
            text = _compose_request(state["request"], state["exchanges"], self.config, self._build_payload)
            result = self._propose(text)

            # 그래프 경계에서도 선택한 계약으로 검사한다. 상태의 권한 선언은 신뢰하지 않는다.
            checked = self._validate(result["proposal"])
            return {"proposal": checked, "model_calls": state["model_calls"] + 1}

        def route(state):
            status = state["proposal"]["proposal"]["status"]

            if status == "proposed" or state["model_calls"] >= MAX_MODEL_CALLS:
                return END

            return "clarify"

        def clarify(state):
            questions = state["proposal"]["proposal"]["questions"]
            # 재개 시 이 노드는 처음부터 실행된다. interrupt 앞에서 모델을 호출하거나
            # 파일을 저장하지 않아 이전 모델 요청이 반복되지 않는다.
            answer = interrupt({"questions": questions})
            self._build_payload(answer, self.config)
            return {"exchanges": state["exchanges"] + [{"questions": questions, "answer": answer}]}

        builder = StateGraph(DialogueState)
        builder.add_node("propose", propose)
        builder.add_node("clarify", clarify)
        builder.add_edge(START, "propose")
        builder.add_conditional_edges("propose", route, {END: END, "clarify": "clarify"})
        builder.add_edge("clarify", "propose")
        # 코드가 만든 기본 자료형만 보관하며 외부 저장소·checkpoint 역직렬화는 받지 않는다.
        self._graph = builder.compile(checkpointer=InMemorySaver())

    def _build_payload(self, text, config):
        return build_request_payload(text, config)

    def _propose(self, text):
        return propose_request(text, self.config, backend="langchain").to_dict()

    def _validate(self, proposal):
        return validate_proposal(
            json.dumps(proposal, ensure_ascii=False),
            expected_input=dict(SOURCE), allowed_modules=ALLOWED,
        ).to_dict()

    def preview(self, request_or_answer):
        """다음 호출에 전달할 실제 누적 문장을 만든다. 모델 호출·상태 변경은 없다."""
        if self._phase == "new":
            request, exchanges = request_or_answer, []
        elif self._phase == "awaiting_answer":
            self._build_payload(request_or_answer, self.config)
            request = self._state["request"]
            exchanges = self._state["exchanges"] + [{
                "questions": self._state["proposal"]["proposal"]["questions"],
                "answer": request_or_answer,
            }]
        else:
            raise LocalProposalError("DIALOGUE_NOT_RESUMABLE")

        text = _compose_request(request, exchanges, self.config, self._build_payload)
        return {"status": "dialogue_preview", "endpoint": self.config.endpoint,
                "payload": self._build_payload(text, self.config), "model_called": False,
                "worker_started": False, "execution_approved": False}

    def send(self, request_or_answer, *, confirmed=False):
        """미리보기와 같은 문장을 명시적 전송 확인 후 한 번만 호출한다."""
        if confirmed is not True:
            raise LocalProposalError("DIALOGUE_SEND_CONFIRMATION_REQUIRED")

        # 잘못된 입력은 그래프를 재개하기 전에 거부하므로 수정 후 다시 입력할 수 있다.
        self.preview(request_or_answer)

        if self._phase == "new":
            incoming = {"request": request_or_answer, "exchanges": [], "proposal": {}, "model_calls": 0}
        else:
            incoming = Command(resume=request_or_answer)

        def advance(manager):
            return self._graph.invoke(incoming, config={
                "configurable": {"thread_id": self._thread_id}, "recursion_limit": 16,
                "callbacks": manager, "tags": [], "metadata": {},
            })

        try:
            state = run_unobserved(advance)
        except BaseException as error:
            # 통신 중단·오류 이후에는 같은 checkpoint를 자동 재개하지 않는다.
            self.close()

            if isinstance(error, (LocalProposalError, KeyboardInterrupt)):
                raise

            if not isinstance(error, Exception):
                raise

            raise LocalProposalError("DIALOGUE_CALL_FAILED") from error

        self._state = state
        proposal = deepcopy(state["proposal"])

        if proposal["proposal"]["status"] == "proposed":
            self._phase = "completed"
            status = "proposal_validated"
        elif state["model_calls"] >= MAX_MODEL_CALLS:
            self._phase = "limit_reached"
            status = "clarification_limit_reached"
        else:
            self._phase = "awaiting_answer"
            status = "awaiting_answer"

        proposal.update(status=status, model_calls=state["model_calls"], backend="langchain",
                        dialogue_engine="langgraph", model_called=True, automatic_retry=False)
        return proposal

    def close(self):
        """보관 객체 참조를 해제한다. 운영체제 메모리의 보안 소거를 보증하지 않는다."""
        self._graph = None
        self._state = None
        self._phase = "closed"
