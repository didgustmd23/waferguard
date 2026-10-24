"""CSV 실행안·작업 연결의 로컬 보관. 승인 토큰·대화·모델 응답 원문은 저장하지 않는다."""

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

from .artifacts import _fingerprint
from .isolated_eda import _read_file
from .llm_proposal_store import _store_path
from .llm_workflow_binding import WorkflowStepBinding, binding_code_id
from .llm_workflow_contract import validate_workflow_proposal
from .llm_workflow_proposal import checked_context, restore_context
from .worker_contract import _decode
from .worker_paths import _path
from .workflow_service_registry import service_version, workflow_service


STORE_ROOT = Path(__file__).resolve().parents[2] / "logs/agents/llm_workflows"


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _directory(name, root, *, create=False):
    # 공통 보관 경로 검증을 재사용한다. root/name은 단일 사용자 지정 ID만 허용한다.
    location = _store_path(name, root, create=create).with_suffix("")

    if create:
        location.mkdir()  # 불완전 기록도 덮어쓰지 않는다.

    return _path(location, directory=True)


def _write(path, value):
    payload = _json(value).encode("utf-8")

    if len(payload) > 65536:
        raise ValueError("실행안 연결 기록은 파일당 64 KiB 이하여야 합니다.")

    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _read(path):
    return _decode(_read_file(_path(path, directory=False), 65536).decode("utf-8"))


def save_workflow(name, proposal, context, *, root=STORE_ROOT):
    frozen = restore_context(checked_context(**context))
    checked = validate_workflow_proposal(proposal.payload_json, **frozen)

    if checked != proposal or checked.plan_signature is None:
        raise ValueError("동일한 확정 실행안만 저장할 수 있습니다.")

    selection = {"proposal": asdict(checked), "context": json.loads(checked_context(**frozen)),
                 "binding_code_id": binding_code_id()}
    packet = {"schema_version": "llm_saved_workflow_v1", "selection": selection,
              "selection_id": _hash(selection)}
    directory = _directory(name, root, create=True)
    _write(directory / "plan.json", packet)
    return load_workflow(name, root=root)


def load_workflow(name, *, root=STORE_ROOT):
    directory = _directory(name, root)
    packet = _read(directory / "plan.json")

    if set(packet) != {"schema_version", "selection", "selection_id"} or packet["schema_version"] != "llm_saved_workflow_v1":
        raise ValueError("지원하는 저장 실행안의 정확한 필드가 필요합니다.")

    selection = packet["selection"]

    if (set(selection) != {"proposal", "context", "binding_code_id"}
            or packet["selection_id"] != _hash(selection) or selection["binding_code_id"] != binding_code_id()):
        raise ValueError("저장 실행안·연결 코드가 변경됐습니다. 자동 재개하지 않습니다.")

    context = restore_context(_json(selection["context"]))
    checked_context(**context)
    record = selection["proposal"]
    checked = validate_workflow_proposal(record["payload_json"], **context)

    if asdict(checked) != record or checked.plan_signature is None:
        raise ValueError("저장 실행안이 현재 정적 계약과 다릅니다.")

    return {"name": name, "directory": str(directory), "selection_id": packet["selection_id"],
            "proposal": checked, "context": context}


def run_identity(module_id, name):
    service = workflow_service(module_id, service_version(module_id))
    _, tool, _ = service._open(name)
    state = tool.states.load()
    return {"run_id": state.run_id, "actor_id": state.actor_id, "plan_id": tool.plan.plan_id,
            "selection": _fingerprint(tool.states.directory / "selection.json")}


def begin_creation(saved, binding, *, root=STORE_ROOT):
    """부작용 전에 배타 intent를 보존한다. 실패·중단 intent는 재시도 근거가 아니다."""
    current = load_workflow(saved["name"], root=root)
    packet = binding.packet

    if (current["selection_id"] != saved["selection_id"]
            or packet["plan_signature"] != current["proposal"].plan_signature):
        raise ValueError("저장 실행안과 현재 생성 범위가 다릅니다.")

    _write(Path(current["directory"]) / (packet["step_id"] + ".intent.json"),
           {"binding": packet, "binding_id": binding.binding_id})


def finish_creation(saved, binding, result, *, root=STORE_ROOT):
    directory = _directory(saved["name"], root)
    step = binding.packet["step_id"]
    intent = _read(directory / (step + ".intent.json"))

    if intent != {"binding": binding.packet, "binding_id": binding.binding_id}:
        raise ValueError("생성 전 기록과 완료 연결 범위가 다릅니다.")

    _write(directory / (step + ".created.json"), {
        "binding_id": binding.binding_id, "module_id": binding.packet["module_id"],
        "run_name": result["run_name"], "identity": run_identity(binding.packet["module_id"], result["run_name"]),
    })


def created_runs(saved, *, root=STORE_ROOT):
    current = load_workflow(saved["name"], root=root)

    if current["selection_id"] != saved["selection_id"]:
        raise ValueError("선택한 저장 실행안이 변경됐습니다.")

    runs = {}
    directory = Path(current["directory"])
    missing = False

    for step in current["proposal"].to_dict()["proposal"]["workflow"]["steps"]:
        intent_path = directory / (step["step_id"] + ".intent.json")
        receipt_path = directory / (step["step_id"] + ".created.json")

        if not intent_path.exists() and not receipt_path.exists():
            missing = True
            continue

        if missing or not intent_path.exists() or not receipt_path.exists():
            raise ValueError("생성 결과가 불명확하거나 단계 연결이 끊겼습니다. 자동 재생성하지 않습니다.")

        intent, receipt = _read(intent_path), _read(receipt_path)
        binding = WorkflowStepBinding(_json(intent["binding"]))
        packet = binding.packet
        name = packet["creation"]["run_name"]
        expected = {"binding_id": binding.binding_id, "module_id": step["module_id"],
                    "run_name": name, "identity": run_identity(step["module_id"], name)}

        if (set(intent) != {"binding", "binding_id"} or intent["binding_id"] != binding.binding_id
                or packet["step_id"] != step["step_id"]
                or packet["plan_signature"] != current["proposal"].plan_signature or receipt != expected):
            raise ValueError("생성 연결·실행 식별값이 저장 기록과 다릅니다.")

        runs[step["step_id"]] = name

    return runs
