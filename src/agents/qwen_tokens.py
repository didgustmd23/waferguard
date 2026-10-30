"""RAG ON 근거 예산용 Qwen3.5 입력 토큰 로컬 계산. 생성 요청을 보내지 않는다.

검증한 계산 구성(독립 시험 28개 요청에서 서버 prompt_eval_count와 일치)에서만 정확 계산을 쓴다.
- Ollama 0.40.7-rc0 + qwen3.5:9b(digest 고정) + 서버 렌더러 qwen3.5 + think=false + system·user 각 1개, 도구·이미지 없음
- 토크나이저: 공식 Qwen/Qwen3.5-9B tokenizer.json(리비전·sha256 고정), tokenizers 0.23.3
- 렌더링: Ollama v0.40.7-rc0 model/renderers/qwen35.go의 system+user 경로(앞뒤 공백 제거,
  역할·종료 특수 토큰, think=false의 빈 thinking 블록)를 고정 규칙으로 옮겼다. 응답 형식(format JSON schema)·options는
  프롬프트에 들어가지 않으므로 세지 않는다.
하나라도 다르거나 미지원 입력(NFC로 바뀌는 문자열, GGUF에만 있는 특수 토큰 문자열)이면 고정 사유와 함께
TokenCountUnavailable을 내고, 호출자는 기존 바이트 추정 예산을 쓴다. 파일·패키지를 내려받지 않는다.
모델 확인은 데이터 없는 상태 조회(/api/version·/api/tags)만 한다. 프롬프트·토큰 ID·오류 원문을 남기지 않는다.
"""

import hashlib
import http.client
import json
from pathlib import Path
import re
import unicodedata

from .llm_config import LocalModelConfig
from .llm_proposal_contract import _decode_response
from .worker_paths import _path


OLLAMA_VERSION = "0.40.7-rc0"
MODEL = "qwen3.5:9b"
MODEL_DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
TOKENIZER_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
TOKENIZER_SHA256 = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
TOKENIZERS_VERSION = "0.23.3"
TOKENIZER_PATH = Path(__file__).resolve().parents[2] / "models" / "tokenizers" / "qwen3.5-9b" / "tokenizer.json"
RENDER_RULE = "ollama-0.40.7-rc0/qwen3.5/system-user/think-false/v1"
MAX_TOKENIZER_BYTES = 16 * 1024 * 1024
MAX_STATUS_BYTES = 64 * 1024
PAYLOAD_KEYS = {"model", "messages", "stream", "think", "format", "options"}
# 설치 GGUF에만 있는 특수 토큰(공식 tokenizer.json에 없음). 서버는 1토큰으로 세지만 로컬 계산은 여러 토큰으로 센다.
GGUF_ONLY_SPECIAL = ("<|audio_start|>", "<|audio_end|>", "<|audio_pad|>", "<tts_pad>", "<tts_text_bos>", "<tts_text_eod>",
                     "<tts_text_bos_single>")
GGUF_PAD = re.compile(r"\[PAD\d+\]")
# 계산 구성 식별값(기록·동의용). 구성 요소가 하나라도 바뀌면 값이 바뀐다.
COUNT_ID = hashlib.sha256(json.dumps([OLLAMA_VERSION, MODEL, MODEL_DIGEST, TOKENIZER_REVISION, TOKENIZER_SHA256,
                                      TOKENIZERS_VERSION, RENDER_RULE]).encode("utf-8")).hexdigest()

_loaded = {}


class TokenCountUnavailable(Exception):
    """정확 계산을 쓰지 않는 고정 사유. 원문을 담지 않는다."""

    def __init__(self, code):
        super().__init__(code)
        self.code = code


def _tokenizer():
    """검증한 바이트로만 토크나이저를 만들고 같은 프로세스에서 재사용한다(읽은 파일 = 검사한 파일)."""
    if TOKENIZER_SHA256 in _loaded:
        return _loaded[TOKENIZER_SHA256]

    try:
        import tokenizers
    except ImportError as error:
        raise TokenCountUnavailable("RAG_TOKENS_LIBRARY_UNAVAILABLE") from error

    if getattr(tokenizers, "__version__", None) != TOKENIZERS_VERSION:
        raise TokenCountUnavailable("RAG_TOKENS_LIBRARY_MISMATCH")

    try:
        path = _path(TOKENIZER_PATH.absolute(), directory=False)
        with path.open("rb") as stream:
            raw = stream.read(MAX_TOKENIZER_BYTES + 1)
    except (OSError, ValueError, TypeError) as error:
        raise TokenCountUnavailable("RAG_TOKENS_FILE_UNAVAILABLE") from error

    if len(raw) > MAX_TOKENIZER_BYTES or hashlib.sha256(raw).hexdigest() != TOKENIZER_SHA256:
        raise TokenCountUnavailable("RAG_TOKENS_FILE_MISMATCH")

    try:
        tokenizer = tokenizers.Tokenizer.from_str(raw.decode("utf-8"))
    except Exception as error:
        raise TokenCountUnavailable("RAG_TOKENS_FILE_MISMATCH") from error

    _loaded[TOKENIZER_SHA256] = tokenizer
    return tokenizer


def _status(config, path):
    connection = http.client.HTTPConnection(config.host, config.port, timeout=config.timeout_seconds)

    try:
        connection.request("GET", path, headers={"Accept": "application/json"})
        response = connection.getresponse()

        if response.status != 200 or response.getheader("Content-Encoding") not in (None, "identity"):
            raise TokenCountUnavailable("RAG_TOKENS_STATUS_FAILED")

        raw = response.read(MAX_STATUS_BYTES + 1)

        if len(raw) > MAX_STATUS_BYTES:
            raise TokenCountUnavailable("RAG_TOKENS_STATUS_FAILED")

        return _decode_response(raw.decode("utf-8"))
    except TokenCountUnavailable:
        raise
    except (OSError, http.client.HTTPException, ValueError, UnicodeError, RecursionError) as error:
        raise TokenCountUnavailable("RAG_TOKENS_STATUS_FAILED") from error
    finally:
        connection.close()


def check_server(config):
    """서버 버전·설치 모델 digest를 데이터 없이 조회해 검증한 구성과 대조한다(렌더러는 모델 digest에 묶인다)."""
    version = _status(config, "/api/version")

    if type(version) is not dict or version.get("version") != OLLAMA_VERSION:
        raise TokenCountUnavailable("RAG_TOKENS_SERVER_MISMATCH")

    tags = _status(config, "/api/tags")
    models = tags.get("models") if type(tags) is dict else None
    found = [item for item in models if type(item) is dict and item.get("name") == MODEL] if type(models) is list else []

    if len(found) != 1 or found[0].get("digest") != MODEL_DIGEST:
        raise TokenCountUnavailable("RAG_TOKENS_MODEL_MISMATCH")


def render(messages):
    """Ollama qwen3.5 렌더러의 system+user·think=false 경로."""
    system, user = messages
    return ("<|im_start|>system\n" + system["content"].strip() + "<|im_end|>\n"
            "<|im_start|>user\n" + user["content"].strip() + "<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def supported(payload):
    """검증한 요청 형식만 받는다. 아니면 RAG_TOKENS_INPUT_UNSUPPORTED."""
    messages = payload.get("messages") if type(payload) is dict else None

    if (type(payload) is not dict or set(payload) != PAYLOAD_KEYS or payload["model"] != MODEL or payload["think"] is not False
            or payload["stream"] is not False or type(messages) is not list or len(messages) != 2
            or [message.get("role") if type(message) is dict else None for message in messages] != ["system", "user"]
            or any(set(message) != {"role", "content"} or type(message["content"]) is not str for message in messages)):
        raise TokenCountUnavailable("RAG_TOKENS_INPUT_UNSUPPORTED")

    for message in messages:
        text = message["content"]

        # 앞뒤 공백 처리(Go TrimSpace·Python strip 차이)·NFC 정규화(서버 미적용)·GGUF 전용 특수 토큰은 검증하지 않았다.
        if (text != text.strip() or not text or unicodedata.normalize("NFC", text) != text
                or any(token in text for token in GGUF_ONLY_SPECIAL) or GGUF_PAD.search(text)):
            raise TokenCountUnavailable("RAG_TOKENS_INPUT_UNSUPPORTED")

    return messages


class Counter:
    """검증한 토크나이저·서버 구성에 묶인 계산기. 같은 작업의 근거 후보마다 다시 로딩하지 않는다."""

    def __init__(self, config, tokenizer):
        self.config, self._tokenizer = config, tokenizer

    def count(self, payload):
        messages = supported(payload)
        options = payload["options"]

        # 예산식(num_ctx − num_predict − 여유)이 이 요청의 실제 설정과 같아야 한다.
        if (type(options) is not dict or options.get("num_ctx") != self.config.num_ctx
                or options.get("num_predict") != self.config.num_predict):
            raise TokenCountUnavailable("RAG_TOKENS_INPUT_UNSUPPORTED")

        return len(self._tokenizer.encode(render(messages), add_special_tokens=False).ids)

    def fits(self, payload, limit, max_request_bytes):
        """입력 토큰 ≤ limit(num_ctx − num_predict − 여유)이고 전체 HTTP 본문이 상한 이하인지."""
        body = len(json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"))
        return body <= max_request_bytes and self.count(payload) <= limit


def prepare(config):
    """정확 계산기를 준비한다. 파일·라이브러리를 먼저 확인한 뒤 서버 상태를 조회한다(실패 시 고정 사유)."""
    if (not isinstance(config, LocalModelConfig) or config.model != MODEL or config.think is not False):
        raise TokenCountUnavailable("RAG_TOKENS_CONFIG_MISMATCH")

    tokenizer = _tokenizer()
    check_server(config)
    return Counter(config, tokenizer)
