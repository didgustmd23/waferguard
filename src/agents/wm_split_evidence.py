"""완료된 WM 품질·EDA 결과와 분할 설정을 제안 작업에 결합한다."""

from .isolated_eda import _read_file
from .split_evidence import read_committed_artifact
from .wm_canonical_cli import executor, open_run
from .wm_canonical_contract import digest
from .worker_contract import _decode


def read_eda_evidence(name, config):
    """새 분석이나 승인을 발급하지 않고 검증된 완료 EDA만 참조한다."""
    tool = open_run(name)
    state = tool.states.load()

    if state.status != "completed" or tool.cancel.is_set() or state.blockers:
        raise ValueError("취소되지 않은 완료 WM EDA가 필요합니다.")

    scope = executor(tool).scope_for("run_eda")
    attempt = read_committed_artifact(tool.states.directory, scope).directory
    checked = tool.inspect_attempt(attempt)
    quality = _decode(_read_file(attempt / "analysis/quality/validation.json", 65536).decode("utf-8"))

    if (checked["quality_status"] != "passed" or checked["eda_generated"] is not True
        or quality["split_assessment"]["status"] != "preconditions_passed"
        or quality["duplicate_audit"]["map_label_conflicts"] != 0
        or quality["duplicate_audit"]["wafer_id_conflicts"] != 0
        or config["max_group_ratio"] != tool.contract.max_group_ratio):
        raise ValueError("동일 검토 상한에서 충돌 없이 통과한 WM 품질·EDA가 필요합니다.")

    evidence = {"run_id": state.run_id, "plan_id": tool.plan.plan_id, "source_id": tool.contract.source_id,
                "report_files": checked["files"], "quality": quality, "split_config": config}
    evidence["evidence_id"] = digest(evidence)
    return tool.contract, evidence


def check_eda_evidence(name, expected):
    contract, actual = read_eda_evidence(name, expected["split_config"])

    if actual != expected:
        raise ValueError("분할안에 결합된 EDA·품질·설정 증거가 변경되었습니다.")

    return contract
