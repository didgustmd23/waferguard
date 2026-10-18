"""저장 제안을 로컬 입력에 결합하고 기존 승인 작업을 준비한다. LLM·worker는 호출하지 않는다."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from src.preprocessing.module_contracts import check_id
from src.preprocessing.profiles import profile_from_dict
from .input_snapshot import SNAPSHOT_VERSION, Snapshot
from .llm_proposal_store import _store_path, load_proposal
from .pipeline_approval_cli import MAX_SELECTION_BYTES, _read_json, create
from .pipeline_binding_registry import get_pipeline_binder
from .worker_contract import WorkerLimits
from .worker_paths import _path


CANDIDATE_ROOT = Path(__file__).resolve().parents[2] / "logs/agents/llm_execution_candidates"


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _fingerprint(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def preview_candidate(proposal_name, snapshot_directory, profile_path, limits):
    """기존 바인더로 입력·설정·코드를 검사하고 로컬 미리보기만 반환한다."""
    if not isinstance(limits, WorkerLimits):
        raise TypeError("검증된 자원 설정이 필요합니다.")

    restored = load_proposal(proposal_name)
    pipeline = restored["proposal"]["pipeline"]
    snapshot_dir = _path(Path(snapshot_directory).absolute(), directory=True)
    profile_file = _path(Path(profile_path).absolute(), directory=False)
    info = _read_json(snapshot_dir / "snapshot.json", 4096)

    if (type(info) is not dict or set(info) != {"version", "size_bytes", "sha256", "max_input_bytes"}
            or info["version"] != SNAPSHOT_VERSION):
        raise ValueError("지원되는 snapshot 계약이 필요합니다.")

    snapshot = Snapshot(info["size_bytes"], info["sha256"], info["max_input_bytes"])
    profile_value = _read_json(profile_file, 64 * 1024)
    profile = profile_from_dict(profile_value)
    binder = get_pipeline_binder(profile.adapter_id)
    binding = binder(pipeline, snapshot_dir, snapshot, profile, limits)

    # source.data의 수치 값을 부모에서 분석하지 않는다. 바인더는 snapshot의
    # 바이트 정합성과 프로파일·코드·계획을 검사하고 유효 품질 규칙을 결합한다.
    selection = {
        "schema_version": "llm_execution_candidate_v1", "proposal_name": proposal_name,
        "proposal_plan_signature": restored["plan_signature"],
        "snapshot_directory": str(snapshot_dir), "profile_path": str(profile_file),
        "profile": profile_value, "pipeline": pipeline, "limits": asdict(limits),
        "binding": json.loads(_canonical(binding.to_dict())),
    }

    if len(_canonical(selection).encode("utf-8")) > MAX_SELECTION_BYTES:
        raise ValueError("실행 후보의 로컬 설정 크기 상한을 초과했습니다.")

    return {
        "status": "candidate_preview", "selection_id": _fingerprint(selection),
        "selection": selection, "binding_id": binding.binding_id,
        "effective_execution": binding.approval_summary(),
        "worker_started": False, "model_called": False, "execution_approval_issued": False,
        "notice": "로컬 검토용입니다. 입력·설정·코드를 결합하지만 실행 승인·worker 시작은 하지 않습니다.",
    }


def _write_new(path, value):
    """작업 준비 파일은 배타 생성하고 기존 파일을 덮어쓰지 않는다."""
    payload = _canonical(value).encode("utf-8")

    if len(payload) > MAX_SELECTION_BYTES:
        raise ValueError("작업 준비 파일의 크기 상한을 초과했습니다.")

    with path.open("xb") as stream:
        stream.write(payload)


def create_candidate(name, proposal_name, snapshot_directory, profile_path, limits,
                     *, expected_selection_id, confirmed=False, root=CANDIDATE_ROOT):
    """확인한 내용이 그대로인 경우만 기존 승인 작업 생성 함수를 호출한다.

    새 승인 DB에는 작업 초기 상태만 준비된다. 실행 승인 발급·사전 검사 실행·
    승인 소비·worker 시작은 기존 CLI에서 사용자가 별도로 진행한다.
    """
    if not confirmed:
        raise ValueError("로컬 실행 후보 내용을 사용자가 확인해야 합니다.")

    # 기존 승인 CLI의 작업 이름 제한과 공통 ID 제한을 모두 만족시킨다.
    check_id(name)

    if len(name) > 48:
        raise ValueError("실행 후보 이름은 48자 이하여야 합니다.")

    preview = preview_candidate(proposal_name, snapshot_directory, profile_path, limits)

    if preview["selection_id"] != expected_selection_id:
        raise ValueError("검토 후 제안·입력·프로파일·코드·자원 설정이 변경되었습니다.")

    selection = preview["selection"]
    target = _store_path(name, root, create=True).with_suffix("")
    target.mkdir(exist_ok=False)
    _path(target, directory=True)
    _write_new(target / "candidate.json", selection)
    _write_new(target / "pipeline.json", selection["pipeline"])
    _write_new(target / "profile.json", selection["profile"])

    # 신뢰된 복사 설정을 전달한다. 임의 CLI·import 경로나 LLM 문자열을 실행하지 않는다.
    result = create(
        name, selection["snapshot_directory"], target / "profile.json", target / "pipeline.json",
        limits, confirmed=True,
    )

    # 기존 생성기 내부의 최종 바인딩도 미리보기와 같은지 확인한다. 차이가 있으면
    # 생성된 미승인 작업은 그대로 남기되 완료 영수증을 발급하거나 자동 재시도하지 않는다.
    run_dir = _path(Path(result["run_dir"]), directory=True)
    actual = _read_json(run_dir / "selection.json", MAX_SELECTION_BYTES)

    if (actual["binding"] != selection["binding"]
            or actual["pipeline"] != selection["pipeline"]
            or actual["profile"] != selection["profile"]
            or actual["limits"] != selection["limits"]):
        raise ValueError("생성된 승인 작업과 확인한 실행 후보가 일치하지 않습니다.")

    receipt = {
        "status": "execution_candidate_created", "name": name,
        "proposal_name": proposal_name, "selection_id": preview["selection_id"],
        "binding_id": preview["binding_id"], "run_dir": str(run_dir),
        "candidate_directory": str(target), "worker_started": False,
        "model_called": False, "execution_approval_issued": False,
        "notice": "작업 준비만 완료했습니다. 기존 pipeline_approval_cli에서 사전 검사·승인을 별도로 진행하세요.",
    }
    _write_new(target / "CREATED.json", receipt)
    return receipt
