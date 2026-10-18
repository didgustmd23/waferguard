"""완료된 충돌 후보의 커밋·승인·선택 파일 지문을 실제 제외 작업에 결합한다."""

from .isolated_eda import _read_file
from .split_evidence import read_committed_artifact
from .wm_canonical_contract import digest
from .wm_conflict_cli import executor, open_run
from .worker_contract import _decode


def read_candidate_evidence(name):
    """기존 후보를 바꾸지 않고 검증된 완료 결과와 원래 입력만 참조한다."""
    tool = open_run(name)
    state = tool.states.load()

    if state.status != "completed" or tool.cancel.is_set() or state.blockers:
        raise ValueError("완료되고 취소되지 않은 충돌 제외 후보가 필요합니다.")

    scope = executor(tool).scope_for("derive_preparation")
    attempt = read_committed_artifact(tool.states.directory, scope).directory
    result = tool.inspect_attempt(attempt)
    report = _decode(_read_file(attempt / "candidate/diagnosis/diagnosis.json", 65536).decode("utf-8"))
    hypothetical = report["hypothetical_exclusion"]

    # 현재 첫 적용 범위는 모든 가상 사전조건을 통과한 후보로 제한한다.
    if hypothetical["split_assessment"]["status"] != "preconditions_passed":
        raise ValueError("가상 제외 후 검토가 남아 있는 후보는 실제 적용하지 않습니다.")

    evidence = {"run_id": state.run_id, "plan_id": tool.plan.plan_id, "config_id": tool.config_id,
                "source_id": tool.contract.source_id, "candidate_files": result["files"],
                "selection_files": {name: result["files"][name] for name in ("conflict_hashes.jsonl", "excluded_rows.csv")},
                "excluded_rows": report["candidate_excluded_manifest_rows"], "hypothetical": hypothetical}
    evidence["candidate_evidence_id"] = digest(evidence)
    return tool.contract, evidence


def check_candidate_evidence(name, expected):
    contract, actual = read_candidate_evidence(name)

    if actual != expected:
        raise ValueError("제외 후보가 승인 범위에 결합된 기록과 다릅니다.")

    return contract
