"""WM 모듈 계획의 단계와 기존 작업을 연결한다. 조회 결과로 승인·완료 상태를 쓰지 않는다."""

from pathlib import Path

from src.preprocessing.pipeline import _object
from .worker_contract import _decode
from .wm_module_catalogue import wm_modules
from .wm_run_evidence import RunNotComplete, SPECS, WmRunReader, check_run_name


def load_bindings(path, plan):
    """실제 작업 배정은 로컬 파일에서 읽는다. 계획 서명 불일치·미등록 단계는 거부한다."""
    with Path(path).open("rb") as stream:
        raw = stream.read(65537)

    value = _decode(raw.decode("utf-8"))
    _object(value, ("schema_version", "pipeline_signature", "runs"))

    if value["schema_version"] != "wm_run_bindings_v1" or value["pipeline_signature"] != plan.signature:
        raise ValueError("작업 배정이 현재 모듈 계획의 버전·서명과 다릅니다.")

    runs = value["runs"]
    identifiers = {step.step_id for step in plan.steps}

    if type(runs) is not dict or set(runs) - identifiers:
        raise ValueError("계획에 없는 단계에 작업을 배정할 수 없습니다.")

    for name in runs.values():
        check_run_name(name)

    if any(step.module_id not in SPECS for step in plan.steps):
        raise ValueError("현재 실행 연결 조회는 WM 정상 개발 6단계만 지원합니다.")

    return dict(runs)


def inspect_bindings(plan, runs):
    """순서대로 검증하고 첫 미배정·미완료·불일치에서 멈춘다. 후속 작업을 실행하지 않는다."""
    if type(runs) is not dict or set(runs) - {step.step_id for step in plan.steps}:
        raise ValueError("단계별 작업 배정 형식이 올바르지 않습니다.")

    for name in runs.values():
        check_run_name(name)

    if any(step.module_id not in SPECS for step in plan.steps):
        raise ValueError("충돌 제외 분기의 통합 조회는 아직 지원하지 않습니다. 기존 CLI를 사용하세요.")

    reader = WmRunReader()
    verified, rows = {}, []
    stopped = None
    modules = {item.contract.module_id: item for item in wm_modules()}

    for step in plan.steps:
        row = {"step_id": step.step_id, "module_id": step.module_id, "run_name": runs.get(step.step_id)}

        if stopped is not None:
            rows.append({**row, "status": "not_checked", "code": "PREVIOUS_STEP_NOT_VERIFIED"})
            continue

        if step.step_id not in runs:
            stopped = {"step_id": step.step_id, "code": "RUN_NOT_ASSIGNED",
                       "action": "select_or_create_existing_cli_run", "cli": "src.agents." + modules[step.module_id].cli_id}
            rows.append({**row, "status": "pending", "code": "RUN_NOT_ASSIGNED"})
            continue

        try:
            current = reader.read(step.module_id, runs[step.step_id])
            reference = dict(step.inputs)["evidence"]

            if reference != "source.evidence":
                parent_step = reference.split(".", 1)[0]
                actual_parent = getattr(current.tool, SPECS[step.module_id][1], None)

                if actual_parent != verified[parent_step].name:
                    raise ValueError("계획에서 연결한 부모 작업과 실제 선행 작업이 다릅니다.")

            verified[step.step_id] = current
            rows.append({**row, "status": "verified", "code": "COMPLETED_RUN_VERIFIED"})
        except RunNotComplete as error:
            stopped = {"step_id": step.step_id, "code": "RUN_NOT_COMPLETE", "action": "review_existing_run",
                       "cli": "src.agents." + modules[step.module_id].cli_id, "stage": error.stage,
                       "run_status": error.status}
            rows.append({**row, "status": "pending", "code": "RUN_NOT_COMPLETE"})
        except (ValueError, TypeError, OSError, KeyError, RuntimeError):
            # 상세 예외·통계·경로는 상태 응답에 섞지 않는다. 오류를 성공으로 대체하지 않는다.
            stopped = {"step_id": step.step_id, "code": "RUN_EVIDENCE_MISMATCH", "action": "review_local_evidence"}
            rows.append({**row, "status": "blocked", "code": "RUN_EVIDENCE_MISMATCH"})

    try:
        reader.check_unchanged()
    except (ValueError, TypeError, OSError, KeyError, RuntimeError):
        return {"status": "verification_failed", "code": "RUN_CHANGED_DURING_CHECK", "steps": rows,
                "next_action": "repeat_readonly_verification", "worker_started": False, "approval_consumed": False}

    return {"status": "bindings_verified" if stopped is None else "attention_required",
            "pipeline_id": plan.pipeline_id, "pipeline_signature": plan.signature,
            "steps": rows, "next_action": stopped, "worker_started": False,
            "approval_consumed": False, "model_trained": False, "test_maps_read": False,
            "scope": "completed_run_evidence_and_parent_linkage_only"}
