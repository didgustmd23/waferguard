"""LangChain 모델 인터페이스와 기존 제한형 로컬 HTTP 운송을 연결한다."""

from contextvars import Context
import json

from langchain_core.callbacks import CallbackManager
from langchain_core.globals import get_debug, get_verbose
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langsmith import tracing_context
from pydantic import PrivateAttr

from .llm_config import LocalModelConfig
from .local_ollama import LocalProposalError, request_local_proposal


class LocalProposalChatModel(BaseChatModel):
    """도구·임의 옵션 없이 미리 구성한 요청 한 건만 처리하는 채팅 모델.

    ChatOllama의 별도 SDK 운송 대신 기존 HTTP 운송을 재사용한다. 따라서
    loopback 고정·프록시 무시·응답 크기 제한·완료 응답 검사를 유지한다.
    이 클래스 자체가 승인·데이터 읽기·전처리 실행 권한을 갖지는 않는다.
    """

    _payload_json: str = PrivateAttr(default="")
    _local_config: LocalModelConfig | None = PrivateAttr(default=None)

    @property
    def _llm_type(self):
        """모델명이나 사용자 문장 대신 고정된 어댑터 종류만 식별한다."""
        return "waferguard-local-proposal"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """LangChain 메시지가 검토한 요청과 같을 때 한 번만 HTTP로 전달한다."""
        if stop is not None or kwargs or self._local_config is None:
            raise LocalProposalError("LANGCHAIN_OPTIONS_REJECTED")

        payload = json.loads(self._payload_json)
        converted = []

        for message in messages:
            # 텍스트 2개 외에 이미지·도구 응답·추가 메시지를 허용하지 않는다.
            if type(message) not in (SystemMessage, HumanMessage) or type(message.content) is not str:
                raise LocalProposalError("LANGCHAIN_MESSAGES_REJECTED")

            if message.additional_kwargs:
                raise LocalProposalError("LANGCHAIN_MESSAGES_REJECTED")

            role = "system" if isinstance(message, SystemMessage) else "user"
            converted.append({"role": role, "content": message.content})

        if converted != payload["messages"]:
            raise LocalProposalError("LANGCHAIN_MESSAGES_REJECTED")

        content = request_local_proposal(payload, self._local_config)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])


def run_unobserved(operation):
    """모델·그래프 호출 모두에 같은 추적·콜백 비상속 규칙을 적용한다."""
    def isolated():
        with tracing_context(enabled=False, parent=False):
            if get_debug() or get_verbose():
                raise LocalProposalError("LANGCHAIN_DIAGNOSTICS_ENABLED")

            manager = CallbackManager.configure([], [], verbose=False)

            if manager.handlers or manager.inheritable_handlers:
                raise LocalProposalError("LANGCHAIN_CALLBACKS_REJECTED")

            return operation(manager)

    return Context().run(isolated)


def request_langchain_proposal(payload, config):
    """신선한 대화 문맥에서 추적·캐시 없이 모델 인터페이스를 호출한다.

    외부 코드의 콜백·태그·대화 문맥은 상속하지 않는다. LangSmith 추적은
    호출 범위에서 끄며 환경 변수 자체는 바꾸지 않는다. 전역 디버그 출력이나
    추가 콜백이 활성화되어 있으면 조용히 우회하지 않고 호출 전에 거부한다.
    """
    if (not isinstance(config, LocalModelConfig) or type(payload) is not dict
            or payload.get("model") != config.model):
        raise LocalProposalError("MODEL_CONFIG_MISMATCH")

    if (set(payload) != {"model", "messages", "stream", "think", "format", "options"}
            or payload["stream"] is not False or payload["think"] is not config.think
            or type(payload["format"]) is not dict
            or payload["options"] != {"temperature": config.temperature, "num_ctx": config.num_ctx,
                                      "num_predict": config.num_predict}):
        raise LocalProposalError("LANGCHAIN_OPTIONS_REJECTED")

    serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False)

    if len(serialized.encode("utf-8")) > 64 * 1024:
        raise LocalProposalError("REQUEST_TOO_LARGE")

    messages = payload.get("messages")

    if (type(messages) is not list or len(messages) != 2
            or [item.get("role") for item in messages if type(item) is dict] != ["system", "user"]
            or any(set(item) != {"role", "content"} or type(item["content"]) is not str
                   for item in messages)):
        raise LocalProposalError("LANGCHAIN_MESSAGES_REJECTED")

    def invoke_once(manager):
        model = LocalProposalChatModel(cache=False, verbose=False, callbacks=[])
        model._payload_json = serialized
        model._local_config = config
        result = model.invoke(
            [SystemMessage(content=messages[0]["content"]), HumanMessage(content=messages[1]["content"])],
            config={"callbacks": manager, "tags": [], "metadata": {}},
        )

        if type(result.content) is not str or result.tool_calls or result.invalid_tool_calls:
            raise LocalProposalError("LANGCHAIN_RESPONSE_REJECTED")

        return result.content

    try:
        return run_unobserved(invoke_once)
    except LocalProposalError:
        raise
    except Exception as error:
        # 외부 라이브러리 예외에 문장·응답이 포함될 수 있어 원문을 CLI에 내보내지 않는다.
        raise LocalProposalError("LANGCHAIN_CALL_FAILED") from error
