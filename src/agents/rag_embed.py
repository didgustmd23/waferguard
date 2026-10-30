"""BGE-M3 임베딩 운송. 기존 작업 설정의 loopback Ollama만 쓰고 프록시·리다이렉트·재시도·잘림·다른 제공자 전환이 없다.

local_ollama와 같은 방식(http.client, 환경 변수 프록시 미사용)으로 설치 목록 조회(데이터 없음) 1회와 임베딩 1회만 한다.
오류는 고정 코드이며 서버 응답·입력 원문을 담지 않는다.
"""

import http.client
import json
import math
import re

from .llm_config import LocalModelConfig
from .llm_proposal_contract import _decode_response


MAX_INPUT_CHARS = 1500
MAX_INPUT_TOKENS = 1024
MAX_REQUEST_BYTES = 16 * 1024
MAX_TAGS_BYTES = 64 * 1024
MAX_RESPONSE_BYTES = 1024 * 1024


class EmbedError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def check_input(text):
    if type(text) is not str or not text.strip() or any(ord(char) < 32 and char not in "\n\t" for char in text):
        raise EmbedError("RAG_EMBED_INPUT_INVALID")

    if len(text) > MAX_INPUT_CHARS:
        raise EmbedError("RAG_EMBED_INPUT_TOO_LONG")


def check_vector(response, model, dimensions):
    """벡터 1개·모델 이름·차원·유한값·영벡터·토큰 수를 확인한다. 하나라도 다르면 쓰지 않는다."""
    if type(response) is not dict or "error" in response or response.get("model") != model:
        raise EmbedError("RAG_EMBED_RESPONSE_INVALID")

    vectors, tokens = response.get("embeddings"), response.get("prompt_eval_count")

    if type(vectors) is not list or len(vectors) != 1 or type(vectors[0]) is not list or len(vectors[0]) != dimensions:
        raise EmbedError("RAG_EMBED_RESPONSE_INVALID")

    vector = vectors[0]

    if (any(type(value) not in (int, float) or not math.isfinite(value) for value in vector)
            or math.sqrt(sum(value * value for value in vector)) < 1e-6):
        raise EmbedError("RAG_EMBED_RESPONSE_INVALID")

    if type(tokens) is not int or not 0 < tokens <= MAX_INPUT_TOKENS:
        raise EmbedError("RAG_EMBED_RESPONSE_INVALID")

    return [float(value) for value in vector]


class OllamaEmbedder:
    """한 객체로 임베딩을 limit회까지만 보낸다(작업 질의는 1회). 넘는 호출은 전송 없이 거부한다(자동 재시도 방지).

    색인 구축 관리 명령만 문서 수만큼 limit을 준다. 모델 식별값은 객체마다 처음 한 번 확인한다.
    """

    def __init__(self, config, *, limit=1):
        if not isinstance(config, LocalModelConfig) or type(limit) is not int or not 1 <= limit <= 1000:
            raise EmbedError("RAG_EMBED_CONFIG_INVALID")

        self.config, self.limit, self.calls, self.verified = config, limit, {"tags": 0, "embed": 0}, None

    def _request(self, method, path, payload, limit):
        body = None if payload is None else json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")

        if body is not None and len(body) > MAX_REQUEST_BYTES:
            raise EmbedError("RAG_EMBED_INPUT_TOO_LONG")

        connection = http.client.HTTPConnection(self.config.host, self.config.port, timeout=self.config.timeout_seconds)

        try:
            headers = {"Accept": "application/json"}

            if body is not None:
                headers["Content-Type"] = "application/json"

            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()

            if response.status != 200:
                raise EmbedError("RAG_EMBED_HTTP_FAILED")

            if response.getheader("Content-Encoding") not in (None, "identity"):
                raise EmbedError("RAG_EMBED_RESPONSE_INVALID")

            raw = response.read(limit + 1)

            if len(raw) > limit:
                raise EmbedError("RAG_EMBED_RESPONSE_INVALID")

            return _decode_response(raw.decode("utf-8"))
        except EmbedError:
            raise
        except (OSError, http.client.HTTPException) as error:
            raise EmbedError("RAG_EMBED_CONNECTION_FAILED") from error
        except (ValueError, UnicodeError, RecursionError) as error:
            raise EmbedError("RAG_EMBED_RESPONSE_INVALID") from error
        finally:
            connection.close()

    def identity(self, model, digest):
        """설치 목록에서 모델 이름·digest를 확인한다. 데이터를 보내지 않으며 모델을 실행하지 않는다."""
        self.calls["tags"] += 1
        tags = self._request("GET", "/api/tags", None, MAX_TAGS_BYTES)
        models = tags.get("models") if type(tags) is dict else None

        if type(models) is not list:
            raise EmbedError("RAG_EMBED_RESPONSE_INVALID")

        found = [item for item in models if type(item) is dict and item.get("name") == model]

        if len(found) != 1:
            raise EmbedError("RAG_EMBED_MODEL_MISSING")

        if type(found[0].get("digest")) is not str or found[0]["digest"] != digest:
            raise EmbedError("RAG_EMBED_MODEL_MISMATCH")

    def embed(self, text, *, model, digest, dimensions):
        check_input(text)

        if type(model) is not str or re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.:/-]{0,127}", model) is None:
            raise EmbedError("RAG_EMBED_CONFIG_INVALID")

        if self.calls["embed"] >= self.limit:
            raise EmbedError("RAG_EMBED_ALREADY_USED")

        if self.verified != (model, digest):
            self.identity(model, digest)
            self.verified = (model, digest)

        self.calls["embed"] += 1
        response = self._request("POST", "/api/embed", {"model": model, "input": text, "truncate": False}, MAX_RESPONSE_BYTES)
        return check_vector(response, model, dimensions)
