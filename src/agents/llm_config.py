"""로컬 모델 설정 로더. 모델별 값과 고정 보안 상한을 구분한다."""

from dataclasses import dataclass, fields
import math
from pathlib import Path
import re

from .llm_proposal_contract import _decode_response


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/agents/llm/local_ollama.json"


@dataclass(frozen=True)
class LocalModelConfig:
    """사용자가 고른 모델·실행 옵션. 외부 서버·자동 전환은 허용하지 않는다."""

    model: str
    host: str
    port: int
    timeout_seconds: int
    think: bool
    temperature: float
    num_ctx: int
    num_predict: int

    def __post_init__(self):
        if (type(self.model) is not str
                or re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.:/-]{0,127}", self.model) is None
                or "cloud" in self.model.lower()):
            raise ValueError("설치된 로컬 모델명을 지정하세요. cloud 모델은 허용하지 않습니다.")

        if type(self.host) is not str or self.host not in ("127.0.0.1", "::1"):
            raise ValueError("모델 서버는 리터럴 loopback 주소만 허용합니다.")

        for name, lower, upper in (
            ("port", 1, 65535), ("timeout_seconds", 1, 600),
            ("num_ctx", 512, 32768), ("num_predict", 64, 8192),
        ):
            value = getattr(self, name)

            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"{name} 설정이 허용 범위를 벗어났습니다: {lower}~{upper}")

        if type(self.think) is not bool:
            raise ValueError("think는 true 또는 false여야 합니다.")

        if (type(self.temperature) not in (int, float)
                or not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2):
            raise ValueError("temperature는 0~2의 유한한 숫자여야 합니다.")

    @property
    def endpoint(self):
        host = f"[{self.host}]" if self.host == "::1" else self.host
        return f"http://{host}:{self.port}/api/chat"


def load_local_config(path=DEFAULT_CONFIG):
    """작은 JSON만 읽고 중복 키·누락·추가 항목을 거부한다. 경로를 모델에 보내지 않는다."""
    with Path(path).open("rb") as stream:
        raw = stream.read(8193)

    if len(raw) > 8192:
        raise ValueError("로컬 모델 설정은 8 KiB 이하여야 합니다.")

    value = _decode_response(raw.decode("utf-8"))
    names = {field.name for field in fields(LocalModelConfig)}

    if (type(value) is not dict or set(value) != names | {"schema_version"}
            or value["schema_version"] != "local_ollama_v1"):
        raise ValueError("지원하는 로컬 모델 설정 버전과 정확한 항목을 지정하세요.")

    return LocalModelConfig(**{name: value[name] for name in names})
