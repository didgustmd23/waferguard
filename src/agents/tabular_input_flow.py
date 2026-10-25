"""표형 입력 준비 정책. 공통 조정기는 해석 대기·재개 hook만 호출한다."""

from pathlib import Path

from src.preprocessing.tabular_loaders import CsvReadSpec
from . import task_consent
from .input_snapshot import Snapshot, verify_snapshot
from .tabular_input_service import TabularInputService, code_id
from .tabular_input_worker import MAX_OUTPUT_BYTES, POLICY_VERSION
from .tabular_source_contract import TabularSourceContract
from .task_plan import TaskProblem


def _spec(record, selection=None):
    previous = (record.get("input_selection") or {}).get("spec", CsvReadSpec().to_dict())
    return CsvReadSpec.from_dict(dict(previous, **(selection or {})))


def consent_content(record, selection=None):
    """화면에서 확인할 해석·코드·자원과 두 입력 도구의 위임 범위를 고정한다."""
    current = record.get("input_selection") or {}
    return {"allowed": [["tabular_inspect", "ingest"], ["tabular_materialize", "ingest"]],
            "input_policy": POLICY_VERSION, "input_code_id": code_id(), "read_spec": _spec(record, selection).to_dict(),
            "input_attempt": current.get("attempt", 0) + 1,
            "max_output_bytes": MAX_OUTPUT_BYTES,
            "previous_inspection_id": current.get("inspection_id"),
            "previous_inspection_sha256": current.get("inspection_sha256")}


def _verify_snapshot(record):
    snapshot = record["snapshot"]
    if not verify_snapshot(Path(snapshot["directory"]), Snapshot(**snapshot["snapshot"])):
        raise ValueError("입력 해석 대상 복사본이 변경됐습니다.")


def check_selection(record, selection, directory):
    """새 동의·실행 생성 전에 선택과 기존 검사 증거를 확인한다."""
    current = record["input_selection"]
    if (current["status"] != "waiting" or type(selection) is not dict or not selection
            or set(selection) - {"header_repair", "empty_field"}):
        raise ValueError("대기 중인 입력 해석 선택이 필요합니다.")
    parsed = _spec(record, selection)
    if parsed.to_dict() == current["spec"]:
        raise ValueError("입력 해석이 바뀌지 않았습니다.")
    _verify_snapshot(record)
    content = task_consent.check(record["consents"].get("A"), "A", record["task_id"])
    if content["input_code_id"] != code_id():
        raise ValueError("입력 실행 코드가 바뀌었습니다. 새 작업이 필요합니다.")
    service = TabularInputService("inspect", root=directory / "input_runs")
    _, result = service.verified(current["inspect_run"])
    if (result["inspection"]["inspection_id"] != current["inspection_id"]
            or result["inspection_sha256"] != current["inspection_sha256"]
            or result["inspection"] != current["inspection"]
            or result["inspection"]["status"] != "review_required"
            or result["inspection"]["code"] != "DUPLICATE_COLUMN_NAMES"
            or parsed.header_repair != "number_duplicates"):
        raise ValueError("선택한 해석이 검증된 검사·해결 범위와 다릅니다.")
    return parsed.to_dict()


def _drive(service, name, record, context, scope):
    for _ in range(6):
        state = service.status(name)
        if state["status"] == "completed":
            return service.verified(name)
        if state["status"] in {"blocked_input", "failed_runtime", "security_locked", "cancelled"}:
            raise TaskProblem("INPUT_PREPARATION_FAILED", "입력 준비 실행이 중단됐습니다. 자동 재시도하지 않습니다.")
        approval_id = None
        if state["status"] == "awaiting_approval":
            preview = service.preview_approval(name)
            summary = preview.to_dict()["summary"]
            dependency = record["input_selection"] if service.action == "materialize" else {}
            authorize = context.authorizer(record, "A", "tabular_" + service.action, "ingest", expected=scope,
                                            bound={"run_name": name, "action": service.action,
                                                   "snapshot_sha256": record["snapshot"]["snapshot"]["sha256"],
                                                   "input_code_id": scope["input_code_id"], "read_spec": scope["read_spec"],
                                                   "max_output_bytes": scope["max_output_bytes"],
                                                   "inspection_id": dependency.get("inspection_id"),
                                                   "inspection_sha256": dependency.get("inspection_sha256")})
            # 선행 검사 지문은 미리보기에서 추출하는 데 그치지 않고 작업 기록의 검증된 값과 대조한다.
            if service.action == "materialize" and (
                    summary["inspection_id"] != record["input_selection"]["inspection_id"]
                    or summary["inspection_sha256"] != record["input_selection"]["inspection_sha256"]):
                raise ValueError("생성 위임 대상이 이번 검사 결과와 다릅니다.")
            issued = service.record_delegated_approval(name, preview, authorize=authorize)
            approval_id = issued["approval_id"]
        service.next(name, expected_stage=state["stage"], expected_revision=state["revision"], approval_id=approval_id)
    raise TaskProblem("INPUT_STATE_UNCLEAR", "입력 준비 상태를 확인해야 합니다. 자동 재시도하지 않습니다.")


def prepare_input(record, directory, context, *, selection=None):
    """검사 대기에서는 모델·생성을 막고 준비된 해석만 별도 생성 승인으로 이어 간다."""
    _verify_snapshot(record)
    content = task_consent.check(record["consents"].get("A"), "A", record["task_id"])
    spec = _spec(record, selection)
    previous = record.get("input_selection") or {}
    expected_attempt = previous.get("attempt", 0) + 1
    expected = {"input_code_id": code_id(), "input_policy": POLICY_VERSION, "read_spec": spec.to_dict(),
                "input_attempt": expected_attempt, "max_output_bytes": MAX_OUTPUT_BYTES,
                "previous_inspection_id": previous.get("inspection_id"),
                "previous_inspection_sha256": previous.get("inspection_sha256"), "limits": record["limits"]}
    if any(content.get(key) != value for key, value in expected.items()):
        raise ValueError("입력 준비 내용이 결정 A와 다릅니다.")
    history = list(previous.get("history", []))
    if previous:
        history.append({"inspect_run": previous["inspect_run"], "inspection_id": previous["inspection_id"],
                        "inspection_sha256": previous["inspection_sha256"], "attempt": previous["attempt"],
                        "previous_consent_id": previous["consent_id"], "consent_id": record["consents"]["A"]["consent_id"]})
    prefix = "t" + record["task_id"][:10]
    current = {"status": "inspecting", "attempt": expected_attempt, "spec": spec.to_dict(), "history": history,
               "consent_id": record["consents"]["A"]["consent_id"], "inspect_run": f"{prefix}-i{expected_attempt}",
               "materialize_run": None}
    record["input_selection"] = current
    record = context.save(record)
    inspect = TabularInputService("inspect", root=directory / "input_runs")
    arguments = {"snapshot_directory": record["snapshot"]["directory"], "snapshot": record["snapshot"]["snapshot"],
                 "spec": spec.to_dict(), "limits": record["limits"], "max_output_bytes": MAX_OUTPUT_BYTES}
    inspect.create(current["inspect_run"], **arguments)
    _, result = _drive(inspect, current["inspect_run"], record, context, expected)
    current.update(inspection=result["inspection"], inspection_id=result["inspection"]["inspection_id"],
                   inspection_sha256=result["inspection_sha256"])
    record["input_selection"] = current
    if result["inspection"]["status"] == "review_required":
        current["status"] = "waiting"
        record["phase"] = "awaiting_input"
        return context.save(record)
    if result["inspection"]["status"] != "ready":
        context.save(record)
        raise TaskProblem("INPUT_BLOCKED", "입력 검사를 통과하지 못했습니다. 검사 사유를 확인하고 새 입력으로 시작하세요.",
                          diagnostic={"stage": "INPUT_INSPECTION", "code": result["inspection"]["code"]})
    current.update(status="materializing", materialize_run=f"{prefix}-m{expected_attempt}")
    record = context.save(record)
    materialize = TabularInputService("materialize", root=directory / "input_runs")
    materialize.create(current["materialize_run"], inspection_run=current["inspect_run"], **arguments)
    _drive(materialize, current["materialize_run"], record, context, expected)
    source = materialize.source_contract(current["materialize_run"])
    current.update(status="materialized", source=source.to_dict())
    record["input_selection"] = current
    return context.save(record)


def check_source(record):
    """후속 단계 전에 원시 복사본과 생성 묶음·승인 연결을 다시 확인한다.

    기록의 출처가 이 작업의 생성 실행(작업 ID로 정해지는 이름·작업 폴더 위치)과 이 작업의 snapshot에서 나왔는지도 대조한다.
    다른 작업의 온전한 출처로 바꾼 기록은 출처 자체 검사를 통과해도 거부한다.
    """
    _verify_snapshot(record)
    current = record["input_selection"]
    source = TabularSourceContract.from_dict(current["source"])
    snapshot = record["snapshot"]
    name = f"t{record['task_id'][:10]}-m{current['attempt']}"
    run = Path(snapshot["directory"]).parent / "input_runs" / "materialize" / name
    if (current.get("materialize_run") != name or Path(source.materialization.run_directory) != run
            or source.snapshot != Snapshot(**snapshot["snapshot"]) or source.snapshot_directory != snapshot["directory"]):
        raise ValueError("작업 기록의 표형 출처가 이 작업의 생성 실행·복사본과 다릅니다.")
    source.check_input()
