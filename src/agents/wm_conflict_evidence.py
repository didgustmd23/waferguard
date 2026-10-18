"""검토 중단된 WM 품질 실행의 입력·승인 소비·보고서를 후보 산출에 결합한다."""

from dataclasses import asdict
import json
import sqlite3

from .isolated_eda import _read_file
from .wm_canonical_cli import executor, open_run
from .wm_canonical_contract import digest, file_hash
from .worker_contract import _decode
from .wm_conflict_results import POLICY_VERSION


def read_quality_evidence(name):
    """이전 결과를 재실행·완료 승격하지 않고 동일 묶음의 검토 보고서만 확인한다."""
    tool = open_run(name)
    state = tool.states.load()

    if (state.status != "blocked_input" or state.current_stage != "run_eda"
        or tool.cancel.is_set() or "SECURITY_LOCKED" in state.blockers):
        raise ValueError("취소되지 않은 WM 품질 검토 중단 실행이 필요합니다.")

    manager = executor(tool)
    scope = manager.scope_for("run_eda")
    task = manager.approvals.get_task(scope)

    if task is None or task["status"] != "failed":
        raise ValueError("WM 품질 검사 시도의 종료 기록이 필요합니다.")

    # 기존 DB는 읽기 전용으로 열어 정확한 범위의 승인 소비와 예약을 함께 확인한다.
    scope_json = json.dumps(asdict(scope), ensure_ascii=False, sort_keys=True)
    database = tool.states.directory / "approvals.sqlite3"
    connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)

    try:
        records = connection.execute(
            "SELECT r.reservation_id,a.status FROM reservations r JOIN approvals a ON a.approval_id=r.approval_id "
            "WHERE r.scope_json=? AND a.scope_json=r.scope_json", (scope_json,),
        ).fetchall()
    finally:
        connection.close()

    if records != [(task["task_id"], "consumed")]:
        raise ValueError("기존 품질 검사의 승인 소비 증거가 일치하지 않습니다.")

    # 작업 ID는 경로로 사용하기 전에 기존 worker 요청 생성기의 검사를 통과시킨다.
    request = tool._request(task["task_id"])
    attempt = tool.states.directory / "attempts" / task["task_id"]
    checked = tool.inspect_attempt(attempt)
    quality = _decode(_read_file(attempt / "analysis/quality/validation.json", 65536).decode("utf-8"))

    if (checked["quality_status"] != "review_required" or checked["eda_generated"] is not False
        or quality["duplicate_audit"]["map_label_conflicts"] <= 0):
        raise ValueError("동일 맵 라벨 충돌이 확인된 품질 보고서가 필요합니다.")

    evidence = {
        "run_id": state.run_id, "task_id": task["task_id"], "request_id": request.request_id,
        "source_id": tool.contract.source_id, "config_id": tool.config_id, "plan_id": tool.plan.plan_id,
        "receipt_sha256": file_hash(attempt / "receipt.json", 65536)["sha256"],
        "quality_sha256": file_hash(attempt / "analysis/quality/validation.json", 65536)["sha256"],
        "row_count": quality["row_count"], "duplicate_audit": quality["duplicate_audit"],
        "split_assessment": quality["split_assessment"],
    }
    return tool.contract, evidence


def check_quality_evidence(name, expected):
    """후보 생성·승인·실행·결과 조회에서도 이전 보고서 변경을 탐지한다."""
    contract, actual = read_quality_evidence(name)

    if actual != expected:
        raise ValueError("기존 WM 품질 결과가 후보에 결합된 증거와 다릅니다.")

    return contract
