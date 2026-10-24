"""기존 CSV·표형 서비스의 커밋·도메인 검증을 재사용하는 로컬 읽기 전용 증거 어댑터."""

from dataclasses import asdict

from .artifacts import _fingerprint
from .csv_workflow_creation import _pipeline, _split
from .split_evidence import read_committed_artifact
from .workflow_service_registry import service_version, workflow_service


def completed_workflow(module_id, name, expected_settings=None):
    """상태 문자열만 믿지 않고 기존 결과 검증기로 파일·승인 소비·커밋을 검사한다."""
    version = service_version(module_id)
    if version is None:
        raise ValueError("선행 입력에는 분할·내보내기·후보·적용 완료 작업만 허용합니다.")
    typed = version != "1.0.0"
    service = workflow_service(module_id, version)
    _, tool, manager = service._open(name)
    before = _fingerprint(tool.states.directory / "state.json")
    result = service.results(name)
    state = tool.states.load()

    if (result["status"] != "completed" or state.status != "completed"
            or state.cancel_requested or tool.cancel.is_set() or state.blockers):
        raise ValueError("차단·취소되지 않은 검증된 완료 작업이 필요합니다.")

    settings, lineage, reports = {}, {}, {}
    role = module_id.split("_", 1)[1]

    if role == "split_review":
        _, _, evidence = _split(name, typed=typed)
        settings = asdict(tool.contract.profile.split_rules)
        lineage["split"] = evidence.signature
    elif role == "partition_export":
        from .partition_input_contract import bind_partition_input, bind_typed_partition_input

        evidence = read_committed_artifact(tool.states.directory, manager.scope_for("ingest"))
        bind = bind_typed_partition_input if typed else bind_partition_input
        train = bind(tool.contract.split, evidence, "train", tool.contract.source.profile.roles)
        lineage = {"split": tool.contract.split.signature, "export": evidence.marker_sha256,
                   "train": train.contract_id}
    elif role in {"prepare_candidate", "prepare_apply"}:
        _, _, candidate, evidence, application = _pipeline(name, applied=role == "prepare_apply", typed=typed)
        train = evidence.derivation.train
        lineage = {"split": train.split.signature, "export": train.export.marker_sha256,
                   "train": train.contract_id, "candidate": evidence.signature}
        settings = asdict(candidate.contract.rules) if role == "prepare_candidate" else {}
        reports["candidate"] = str(evidence.directory / "candidate.md")

        if role == "prepare_apply":
            lineage["prepared"] = _fingerprint(application.states.directory / "state.json")["sha256"]
    elif typed and role == "local_handoff":
        # v2 전달은 같은 출처 계보 확인에만 쓴다. 후속 단계의 선행 입력으로 쓰이지 않는다.
        lineage["candidate"] = _pipeline(tool.arguments["pipeline_name"], applied=True, typed=True)[3].signature
    else:
        raise ValueError("선행 입력에는 분할·내보내기·후보·적용 완료 작업만 허용합니다.")

    if typed:
        from .tabular_workflow_service import source_of

        lineage["source"] = source_of(module_id, tool)

    if expected_settings is not None and settings != expected_settings:
        raise ValueError("선행 작업 규칙이 고정한 LLM 제안과 다릅니다.")

    if _fingerprint(tool.states.directory / "state.json") != before:
        raise ValueError("증거 조회 중 선행 상태가 변경됐습니다.")

    return {"module_id": module_id, "run_name": name, "run_id": state.run_id,
            "state": before, "plan_id": tool.plan.plan_id, "source_id": tool.plan.source_id,
            "config_id": tool.plan.config_id, "lineage": lineage, "local_reports": reports}
