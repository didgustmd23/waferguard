"""완료된 WM 확정 분할·canonical·변환 설정을 검사 승인에 결합한다."""

from .isolated_eda import _read_file
from .split_evidence import read_committed_artifact
from .wm_canonical_contract import digest
from .wm_split_commit_cli import executor, open_run
from .worker_contract import _decode


def read_split_evidence(name, config):
    tool = open_run(name)
    state = tool.states.load()

    if state.status != "completed" or tool.cancel.is_set() or state.blockers:
        raise ValueError("취소되지 않은 완료 WM 확정 분할이 필요합니다.")

    artifact = read_committed_artifact(tool.states.directory, executor(tool).scope_for("commit_split"))
    checked = tool.inspect_attempt(artifact.directory)
    directory = artifact.directory / "split"
    report = _decode(_read_file(directory / "split.json", 65536).decode("utf-8"))

    if checked["split_complete"] is not True or report["canonical_root"] != tool.contract.snapshot_directory:
        raise ValueError("같은 canonical에 연결된 확정 분할이 필요합니다.")

    evidence = {"run_id": state.run_id, "plan_id": tool.plan.plan_id, "source_id": tool.contract.source_id,
                "split_directory": str(directory), "split_files": checked["files"], "split_report": report,
                "canonical_root": tool.contract.snapshot_directory, "transform_config": config}
    evidence["evidence_id"] = digest(evidence)
    return tool.contract, evidence


def check_split_evidence(name, expected):
    contract, actual = read_split_evidence(name, expected["transform_config"])

    if actual != expected:
        raise ValueError("확정 분할·canonical·변환 설정이 승인 범위와 다릅니다.")

    return contract
