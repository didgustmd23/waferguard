"""완료된 WM 분할 제안과 고정 그룹 목록을 별도 확정 승인에 결합한다."""

from .isolated_eda import _read_file
from .split_evidence import read_committed_artifact
from .wm_canonical_contract import digest
from .wm_split_cli import executor, open_run
from .worker_contract import _decode


def read_proposal_evidence(name):
    tool = open_run(name)
    state = tool.states.load()

    if state.status != "completed" or tool.cancel.is_set() or state.blockers:
        raise ValueError("취소되지 않은 완료 WM 분할 제안이 필요합니다.")

    artifact = read_committed_artifact(tool.states.directory, executor(tool).scope_for("derive_preparation"))
    checked = tool.inspect_attempt(artifact.directory)
    directory = artifact.directory / "proposal"
    report = _decode(_read_file(directory / "proposal.json", 65536).decode("utf-8"))

    if checked["status"] != "proposal_ready" or report["split_complete"] is not False:
        raise ValueError("검토할 수 있는 분할안이 필요합니다.")

    evidence = {"run_id": state.run_id, "plan_id": tool.plan.plan_id, "source_id": tool.contract.source_id,
                "proposal_directory": str(directory), "proposal_files": checked["files"], "proposal": report,
                "canonical_root": tool.contract.snapshot_directory}
    evidence["evidence_id"] = digest(evidence)
    return tool.contract, evidence


def check_proposal_evidence(name, expected):
    contract, actual = read_proposal_evidence(name)

    if actual != expected:
        raise ValueError("확정할 제안·그룹 배정·기존 증거가 변경되었습니다.")

    return contract
