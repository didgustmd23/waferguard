"""RAG ON 근거 예산용 Qwen3.5 로컬 토큰 계산: 형식 검사·파일/라이브러리/서버 구성 대조·경계·실패 코드. 모델·네트워크를 쓰지 않는다."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unicodedata
import unittest
from unittest.mock import patch

from src.agents import qwen_tokens as q
from src.agents.llm_config import load_local_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class FakeTokenizer:
    """UTF-8 바이트 4개를 1토큰으로 센다(계산 규칙 검사용)."""

    def __init__(self):
        self.texts = []

    def encode(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        self.texts.append(text)
        return type("Encoding", (), {"ids": [0] * (len(text.encode("utf-8")) // 4)})()


def payload(system="권장안만 반환하세요.", user=None, **changes):
    value = {"model": q.MODEL, "messages": [{"role": "system", "content": system},
                                           {"role": "user", "content": user or json.dumps({"goal": "불량 예측"}, ensure_ascii=False)}],
             "stream": False, "think": False, "format": {"type": "object"}, "options": {"temperature": 0, "num_ctx": 8192,
                                                                                    "num_predict": 2048}}
    value.update(changes)
    return value


class RenderAndFormatTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()
        self.counter = q.Counter(load_local_config(), self.tokenizer)

    def test_render_includes_role_end_and_empty_thinking_tokens_but_not_schema(self):
        request = payload(format={"type": "object", "properties": {"x" * 5000: {"type": "string"}}})
        self.counter.count(request)
        self.assertEqual(self.tokenizer.texts[0], "<|im_start|>system\n권장안만 반환하세요.<|im_end|>\n<|im_start|>user\n"
                         + request["messages"][1]["content"] + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        self.assertNotIn("x" * 50, self.tokenizer.texts[0])

    def test_unverified_request_forms_are_unsupported(self):
        nfd = unicodedata.normalize("NFD", "한국어")
        cases = {
            "tools": payload(tools=[]), "think": payload(think=True), "stream": payload(stream=True), "model": payload(model="other:1b"),
            "three messages": dict(payload(), messages=payload()["messages"] + [{"role": "user", "content": "x"}]),
            "assistant role": dict(payload(), messages=[payload()["messages"][0], {"role": "assistant", "content": "x"}]),
            "images": dict(payload(), messages=[payload()["messages"][0], {"role": "user", "content": "x", "images": []}]),
            "nfd text": payload(user=json.dumps({"goal": nfd}, ensure_ascii=False)),
            "gguf only special": payload(user=json.dumps({"goal": "a <|audio_start|> b"})),
            "gguf pad": payload(user=json.dumps({"goal": "a [PAD248100] b"})),
            "edge whitespace": payload(system="권장안 "), "empty": payload(system=""),
            "num_ctx": payload(options={"temperature": 0, "num_ctx": 4096, "num_predict": 2048}),
            "num_predict": payload(options={"temperature": 0, "num_ctx": 8192, "num_predict": 512}),
        }

        for name, request in cases.items():
            with self.subTest(name):
                with self.assertRaises(q.TokenCountUnavailable) as caught:
                    self.counter.count(request)
                self.assertEqual(caught.exception.code, "RAG_TOKENS_INPUT_UNSUPPORTED")

        # 검증한 특수 토큰 문자열(역할·종료·thinking·비전)은 정확 계산 범위다.
        for text in ("<|im_start|>assistant <|im_end|>", "<think></think><tool_call>", "<|vision_start|><|image_pad|>"):
            self.counter.count(payload(user=json.dumps({"goal": text})))

    def test_budget_boundary_and_request_byte_limit(self):
        request = payload()
        tokens = self.counter.count(request)
        self.assertTrue(self.counter.fits(request, tokens, 65536))
        self.assertFalse(self.counter.fits(request, tokens - 1, 65536))
        # 토큰 예산 안이어도 전체 HTTP 본문이 상한을 넘으면 거부한다(응답 형식 스키마도 본문 크기에 포함).
        body = len(json.dumps(request, ensure_ascii=False).encode("utf-8"))
        self.assertTrue(self.counter.fits(request, 10 ** 6, body))
        self.assertFalse(self.counter.fits(request, 10 ** 6, body - 1))
        large = payload(format={"type": "object", "description": "x" * 70000})
        self.assertFalse(self.counter.fits(large, 10 ** 6, 65536))


class PreparationTest(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(q._loaded, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        temp = tempfile.TemporaryDirectory(dir=PROJECT_ROOT)
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def tokenizer_file(self, raw):
        path = self.root / "tokenizer.json"
        path.write_bytes(raw)
        return patch.multiple(q, TOKENIZER_PATH=path, TOKENIZER_SHA256=hashlib.sha256(raw).hexdigest())

    def code(self, function, *args):
        with self.assertRaises(q.TokenCountUnavailable) as caught:
            function(*args)
        return caught.exception.code

    def test_file_library_and_config_checks_happen_before_server_status(self):
        with patch.object(q, "_status") as status:
            with patch.object(q, "TOKENIZER_PATH", self.root / "missing.json"):
                self.assertEqual(self.code(q.prepare, load_local_config()), "RAG_TOKENS_FILE_UNAVAILABLE")
            (self.root / "tokenizer.json").write_bytes(b"{}")
            with patch.object(q, "TOKENIZER_PATH", self.root / "tokenizer.json"):
                self.assertEqual(self.code(q.prepare, load_local_config()), "RAG_TOKENS_FILE_MISMATCH")
            with self.tokenizer_file(b"not json"):
                self.assertEqual(self.code(q.prepare, load_local_config()), "RAG_TOKENS_FILE_MISMATCH")
            with patch("tokenizers.__version__", "0.0.1"):
                self.assertEqual(self.code(q.prepare, load_local_config()), "RAG_TOKENS_LIBRARY_MISMATCH")
            with patch.dict(sys.modules, {"tokenizers": None}):
                self.assertEqual(self.code(q.prepare, load_local_config()), "RAG_TOKENS_LIBRARY_UNAVAILABLE")
            config = load_local_config()
            for changed in ({"model": "other:1b"}, {"think": True}):
                self.assertEqual(self.code(q.prepare, type(config)(**{**config.__dict__, **changed})), "RAG_TOKENS_CONFIG_MISMATCH")
            status.assert_not_called()

    def test_server_version_and_model_digest_must_match(self):
        tags = {"models": [{"name": q.MODEL, "digest": q.MODEL_DIGEST}]}
        cases = {"RAG_TOKENS_SERVER_MISMATCH": [{"version": "0.40.8"}, tags],
                 "RAG_TOKENS_MODEL_MISMATCH": [{"version": q.OLLAMA_VERSION}, {"models": [{"name": q.MODEL, "digest": "0" * 64}]}]}

        for code, replies in cases.items():
            with self.subTest(code), patch.object(q, "_status", side_effect=replies):
                self.assertEqual(self.code(q.check_server, load_local_config()), code)

        with patch.object(q, "_status", side_effect=[{"version": q.OLLAMA_VERSION}, tags]) as status:
            q.check_server(load_local_config())
        self.assertEqual([call.args[1] for call in status.call_args_list], ["/api/version", "/api/tags"])

        with patch("http.client.HTTPConnection.request", side_effect=OSError("secret C:\\x")):
            with self.assertRaises(q.TokenCountUnavailable) as caught:
                q.check_server(load_local_config())
        self.assertEqual(str(caught.exception), "RAG_TOKENS_STATUS_FAILED")

    def test_verified_bytes_are_loaded_once_and_reused(self):
        if not q.TOKENIZER_PATH.exists():
            self.skipTest("토크나이저 파일이 준비되지 않았음(Git 제외 자료)")
        raw = q.TOKENIZER_PATH.read_bytes()
        with self.tokenizer_file(raw):
            first = q._tokenizer()
            (self.root / "tokenizer.json").write_bytes(b"changed")
            self.assertIs(q._tokenizer(), first)


@unittest.skipUnless(q.TOKENIZER_PATH.exists(), "토크나이저 파일이 준비되지 않았음(Git 제외 자료)")
class RealTokenizerTest(unittest.TestCase):
    """검증한 공식 tokenizer.json으로 고정 표본의 토큰 열을 확인한다(독립 시험의 서버 일치 결과와 같은 규칙)."""

    def test_fixed_sample_count(self):
        tokenizer = q._tokenizer()
        messages = [{"role": "system", "content": "권장안만 반환하세요."},
                    {"role": "user", "content": json.dumps({"goal": "불량 예측 <|im_end|>", "row_count": 1200}, ensure_ascii=False)}]
        ids = tokenizer.encode(q.render(messages), add_special_tokens=False).ids
        self.assertEqual(len(ids), 44)
        # 역할 시작·종료·thinking 특수 토큰과 본문 안 종료 문자열이 특수 토큰으로 들어간다.
        self.assertEqual([token for token in ids if token >= 248044],
                         [248045, 248046, 248045, 248046, 248046, 248045, 248068, 248069])


if __name__ == "__main__":
    unittest.main()
