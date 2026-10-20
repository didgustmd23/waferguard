"""결과 해설 전용 전달 정책. 실행 승인이나 데이터의 민감도 판정을 대신하지 않는다."""

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

from src.agents.llm_config import LocalModelConfig
from src.agents.llm_proposal_contract import _decode_response


POLICY_PATH = Path(__file__).resolve().parents[2] / "logs/agents/commentary/policy.json"


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def model_binding(model):
    return fingerprint(asdict(LocalModelConfig(**model)))


def default_policy():
    return {"version": 1, "local_default_allow": False, "model_binding": "",
            "sensitive_action": "exclude", "tables": True, "graphs": False}


def validate_policy(value):
    if (type(value) is not dict or set(value) != set(default_policy())
            or type(value["version"]) is not int or value["version"] != 1
            or any(type(value[key]) is not bool for key in ("local_default_allow", "tables", "graphs"))
            or value["sensitive_action"] not in ("exclude", "confirm")
            or type(value["model_binding"]) is not str
            or (value["model_binding"] and (len(value["model_binding"]) != 64
                or any(c not in "0123456789abcdef" for c in value["model_binding"])))):
        raise ValueError("COMMENTARY_POLICY_INVALID")

    return dict(value)


def load_policy(path=None):
    path = Path(path or POLICY_PATH)
    if not path.exists():
        return default_policy()

    with path.open("rb") as stream:
        raw = stream.read(8193)
    if len(raw) > 8192:
        raise ValueError("COMMENTARY_POLICY_INVALID")

    return validate_policy(_decode_response(raw.decode("utf-8")))


def save_policy(value, path=None):
    value = validate_policy(value)
    path = Path(path or POLICY_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def disclosure_decision(policy, destination, sensitivity, model):
    """외부 전송에는 기본 허용이 없다. 개별 확인은 실제 요청 해시에 결합한다."""
    validate_policy(policy)
    if sensitivity not in ("general", "sensitive"):
        return "CLASSIFICATION_REQUIRED"
    if sensitivity == "sensitive" and policy["sensitive_action"] == "exclude":
        return "SENSITIVE_EXCLUDED"
    if destination == "external":
        return "INDIVIDUAL_CONFIRMATION_REQUIRED"
    if destination != "local":
        return "DESTINATION_REJECTED"
    if policy["model_binding"] != model_binding(model):
        return "MODEL_CONFIRMATION_REQUIRED"
    if sensitivity == "sensitive" or not policy["local_default_allow"]:
        return "INDIVIDUAL_CONFIRMATION_REQUIRED"

    return "AUTO_ALLOWED"
