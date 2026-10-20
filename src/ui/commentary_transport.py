"""그림 해설의 제한된 loopback 운송. 프록시·재시도·추적·외부 전환 없음."""

import http.client
import json
import re

from src.agents.llm_config import LocalModelConfig
from src.agents.llm_proposal_contract import _decode_response


class CommentaryHTTPError(ValueError):
    """응답 원문 없이 HTTP 상태와 고정 분류만 보관한다."""

    def __init__(self, status, reason, *, requested_tokens=None, context_tokens=None):
        super().__init__("COMMENTARY_HTTP_REJECTED")
        self.http_status = status
        self.reason = reason
        self.requested_tokens = requested_tokens
        self.context_tokens = context_tokens


def _http_reason(response, details=None):
    """서버 오류는 로컬에서 제한적으로 분류한다. 본문·경로·프롬프트는 반환하지 않는다."""
    text = ""
    if response.getheader("Content-Encoding") in (None, "identity"):
        try:
            raw = response.read(8193)
            if len(raw) <= 8192:
                value = _decode_response(raw.decode("utf-8"))
                if type(value) is dict and type(value.get("error")) is str:
                    text = value["error"].lower()
        except (ValueError, UnicodeError, OSError, http.client.HTTPException):
            pass

    if details is not None:
        match = re.search(r"\brequest\s+\(([0-9]{1,9}) tokens\)\s+exceeds the available context size\s+\(([0-9]{1,9}) tokens\)", text)
        if match:
            details.update(requested_tokens=int(match[1]), context_tokens=int(match[2]))

    if any(term in text for term in ("out of memory", "not enough memory", "insufficient memory", "cuda error: out of memory")):
        return "MEMORY_EXHAUSTED"
    if any(term in text for term in ("context length", "context window", "context size", "input length", "too many tokens")):
        return "CONTEXT_LIMIT"
    if any(term in text for term in ("does not support images", "doesn't support images", "does not support vision")):
        return "IMAGES_UNSUPPORTED"
    if response.status == 404:
        return "MODEL_OR_ENDPOINT_NOT_FOUND"
    if response.status in (401, 403):
        return "ACCESS_REJECTED"
    if response.status == 429:
        return "SERVER_BUSY"
    if 300 <= response.status < 400:
        return "REDIRECT_REJECTED"
    if response.status == 400:
        return "INVALID_REQUEST"
    if response.status >= 500:
        return "SERVER_ERROR"
    return "UNCLASSIFIED"


def request_images(payload, config):
    if not isinstance(config, LocalModelConfig) or payload.get("model") != config.model:
        raise ValueError("COMMENTARY_MODEL_MISMATCH")
    body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(body) > 12 * 1024 ** 2:
        raise ValueError("COMMENTARY_REQUEST_TOO_LARGE")

    connection = http.client.HTTPConnection(config.host, config.port, timeout=config.timeout_seconds)
    try:
        connection.request("POST", "/api/chat", body=body, headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            details = {}
            reason = _http_reason(response, details)
            raise CommentaryHTTPError(response.status, reason, **details)
        if response.getheader("Content-Encoding") not in (None, "identity"):
            raise ValueError("COMMENTARY_ENCODING_REJECTED")
        raw = response.read(65537)
        if len(raw) > 65536:
            raise ValueError("COMMENTARY_RESPONSE_TOO_LARGE")
        value = _decode_response(raw.decode("utf-8"))
        message = value.get("message", {})
        if value.get("done_reason") == "length":
            raise ValueError("COMMENTARY_TOKEN_LIMIT")
        if (value.get("done") is not True or value.get("done_reason") != "stop"
                or value.get("model") != config.model or message.get("role") != "assistant"
                or message.get("tool_calls") or type(message.get("content")) is not str):
            raise ValueError("COMMENTARY_RESPONSE_REJECTED")
        return message["content"]
    finally:
        connection.close()
