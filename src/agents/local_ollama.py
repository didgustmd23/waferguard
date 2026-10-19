"""로컬 Ollama 제안 응답 운송. SDK 추적·프록시·재시도·외부 fallback을 사용하지 않는다."""

import http.client
import json

from .llm_proposal_contract import _decode_response


MAX_HTTP_BYTES = 256 * 1024


class LocalProposalError(Exception):
    """프롬프트·응답·서버 오류 원문을 포함하지 않는 고정 오류 코드."""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


def request_local_proposal(payload, config):
    """검증한 loopback 설정으로 한 번만 요청하고 제한된 크기의 완료 응답만 반환한다.

    이 함수는 격리 worker 밖의 제안 계층용이다. HTTPConnection은 환경변수의
    프록시나 OLLAMA_HOST를 사용하지 않으며 HTTP 리다이렉트도 따라가지 않는다.
    socket timeout은 각 대기 한도이지 OS가 강제하는 전체 실행 시간 제한은 아니다.
    서버 자체의 외부 연결까지 차단하는 기능은 아니므로 신뢰된 로컬 Ollama와
    이미 설치한 로컬 모델을 사용해야 한다.
    """
    from .llm_config import LocalModelConfig

    if not isinstance(config, LocalModelConfig) or payload.get("model") != config.model:
        raise LocalProposalError("MODEL_CONFIG_MISMATCH")

    body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")

    if len(body) > 64 * 1024:
        raise LocalProposalError("REQUEST_TOO_LARGE")

    connection = http.client.HTTPConnection(
        config.host, config.port, timeout=config.timeout_seconds,
    )

    try:
        connection.request(
            "POST", "/api/chat", body=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        response = connection.getresponse()

        if response.status != 200:
            # 모델 부재·서버 오류·리다이렉트 모두 본문이나 Location을 출력하지 않는다.
            raise LocalProposalError("OLLAMA_HTTP_FAILED")

        if response.getheader("Content-Encoding") not in (None, "identity"):
            raise LocalProposalError("OLLAMA_ENCODING_REJECTED")

        raw = response.read(MAX_HTTP_BYTES + 1)

        if len(raw) > MAX_HTTP_BYTES:
            raise LocalProposalError("OLLAMA_RESPONSE_TOO_LARGE")

        # HTTP 포장도 중복 키·NaN을 거부한다. 제안 본문은 뒤에서 별도로 검사한다.
        envelope = _decode_response(raw.decode("utf-8"))

        if (type(envelope) is not dict or envelope.get("done") is not True
                or envelope.get("done_reason") != "stop" or envelope.get("model") != config.model):
            raise LocalProposalError("OLLAMA_INCOMPLETE_RESPONSE")

        message = envelope.get("message")

        if (type(message) is not dict or message.get("role") != "assistant"
                or type(message.get("content")) is not str or not message["content"].strip()
                or message.get("tool_calls")):
            raise LocalProposalError("OLLAMA_MESSAGE_REJECTED")

        return message["content"]

    except LocalProposalError:
        raise
    except (OSError, http.client.HTTPException) as error:
        raise LocalProposalError("OLLAMA_CONNECTION_FAILED") from error
    except (ValueError, UnicodeError, RecursionError) as error:
        raise LocalProposalError("OLLAMA_RESPONSE_INVALID") from error
    finally:
        connection.close()
