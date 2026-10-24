"""등록된 입력 어댑터 파이프라인의 계약 확인·승인 저장 CLI."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import secrets
import uuid

from src.preprocessing.input_profile import header_dedupe_pipeline, input_profile_pipeline
from src.preprocessing.profiles import profile_from_dict
from .approval_store import ApprovalStore
from .artifacts import _root
from .audit import append_event
from .cli import _current_actor
from .executor import _digest, _run_lock
from .input_profile_pipeline import bind_input_profile
from .input_snapshot import SNAPSHOT_VERSION, Snapshot
from .isolated_eda import _read_file
from .pipeline_approval import open_pipeline_approval_run
from .pipeline_binding_registry import get_pipeline_binder
from .state import LocalStateStore
from .worker_contract import WorkerLimits, _decode
from .worker_paths import _path


SELECTION_VERSION = "pipeline_approval_selection_v1"
# 승인·생성된 v2 표의 고정 진단. 원시 snapshot 대신 생성 출처 계약을 저장하고 열 때마다 다시 검증한다.
TABULAR_SELECTION_VERSION = "pipeline_approval_selection_v2"
TABULAR_SELECTION_FIELDS = {"schema_version", "run_id", "actor_id", "source", "limits", "binding", "execution_plan"}
MAX_SELECTION_BYTES = 256 * 1024

def _directory(name, *, create_parent=False):
    """작업 이름을 검사하고 고정 승인 기록 폴더를 반환한다."""
    if type(name) is not str or re.fullmatch(r"[a-z][a-z0-9_-]{0,47}", name) is None:
        raise ValueError("작업 이름은 영문 소문자·숫자·밑줄·하이픈으로 1~48자여야 합니다.")

    root = Path(__file__).resolve().parents[2]

    for part in ("logs", "agents", "pipeline_approval", "runs"):
        root = root / part

        if create_parent:
            root.mkdir(exist_ok=True)

        _root(root)

    return root / name


def _read_json(path, limit):
    """크기 제한과 JSON 중복 키 검사를 적용해 설정 파일을 읽는다."""
    return _decode(_read_file(path, limit).decode("utf-8"))


def _json_value(value):
    """계약 객체를 저장·재로딩 뒤와 같은 JSON 기본 자료형으로 정규화한다."""
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))


def _limits(value):
    """저장된 자원 요청을 엄격한 WorkerLimits 객체로 복원한다."""
    if type(value) is not dict or set(value) != set(WorkerLimits.__dataclass_fields__):
        raise ValueError("저장된 worker 자원 한도 필드가 올바르지 않습니다.")

    return WorkerLimits(**value)


def _build_binding(selection):
    """선택 기록의 입력·프로파일·계획을 다시 검증해 승인 바인딩을 만든다."""
    if selection.get("schema_version") == TABULAR_SELECTION_VERSION:
        from .input_profile_pipeline import bind_tabular_diagnostic
        from .tabular_source_contract import TabularSourceContract

        # 생성 승인·커밋·묶음 해시를 출처 계약이 다시 확인한다. 고정 진단 계획 외에는 결합하지 않는다.
        return bind_tabular_diagnostic(TabularSourceContract.from_dict(selection["source"]), _limits(selection["limits"]))

    snapshot_directory = selection["snapshot_directory"]
    snapshot = Snapshot(**selection["snapshot"])

    # 프로파일 없는 선택은 고정 입력 진단 계획에만 결합된다. 다른 계획이면 바인더가 거부한다.
    if selection["profile"] is None:
        return bind_input_profile(selection["pipeline"], snapshot_directory, snapshot, _limits(selection["limits"]))

    profile = profile_from_dict(selection["profile"])

    binder = get_pipeline_binder(profile.adapter_id)

    return binder(
        selection["pipeline"],
        snapshot_directory,
        snapshot,
        profile,
        _limits(selection["limits"]),
    )


def create(name, snapshot_directory, profile_path, pipeline_path, limits, *, confirmed=False):
    """선택한 입력과 계획을 다시 확인할 승인 run으로 저장한다."""
    if not confirmed:
        raise ValueError("snapshot·프로파일·파이프라인 선택을 명시적으로 확인해야 합니다.")

    snapshot_dir, snapshot = _read_snapshot(snapshot_directory)
    profile_file = _path(Path(profile_path).absolute(), directory=False)
    pipeline_file = _path(Path(pipeline_path).absolute(), directory=False)
    profile = _read_json(profile_file, 64 * 1024)
    pipeline = _read_json(pipeline_file, 64 * 1024)
    dataset_profile = profile_from_dict(profile)

    binder = get_pipeline_binder(dataset_profile.adapter_id)

    binding = binder(
        pipeline,
        snapshot_dir,
        snapshot,
        dataset_profile,
        limits,
    )

    return _save_run(name, snapshot_dir, snapshot, profile, pipeline, limits, binding)


def create_input_profile(name, snapshot_directory, limits, *, confirmed=False, header_dedupe=False):
    """프로파일 없이 고정 입력 진단(또는 중복 컬럼명 정비) 계획만 승인 run으로 저장한다. profile은 null이다."""
    if not confirmed:
        raise ValueError("snapshot·입력 진단 선택을 명시적으로 확인해야 합니다.")

    if type(header_dedupe) is not bool:
        raise TypeError("중복 컬럼명 정비 여부는 bool이어야 합니다.")

    snapshot_dir, snapshot = _read_snapshot(snapshot_directory)
    pipeline = header_dedupe_pipeline() if header_dedupe else input_profile_pipeline()
    binding = bind_input_profile(pipeline, snapshot_dir, snapshot, limits)

    return _save_run(name, snapshot_dir, snapshot, None, pipeline, limits, binding)


def create_tabular_diagnostic(name, source, limits, *, confirmed=False):
    """승인·생성된 v2 표에 고정 입력 진단만 결합한 승인 run을 저장한다. worker는 시작하지 않는다."""
    from .input_profile_pipeline import bind_tabular_diagnostic

    if not confirmed:
        raise ValueError("생성된 표의 입력 진단 선택을 명시적으로 확인해야 합니다.")

    binding = bind_tabular_diagnostic(source, limits)
    return _save_selection(name, binding, lambda run_id, actor_id: {
        "schema_version": TABULAR_SELECTION_VERSION, "run_id": run_id, "actor_id": actor_id,
        "source": _json_value(source.to_dict()), "limits": asdict(limits)})


def _read_snapshot(snapshot_directory):
    """선택한 snapshot 폴더의 메타데이터를 고정 필드로 읽는다."""
    snapshot_dir = _path(Path(snapshot_directory).absolute(), directory=True)
    snapshot_info = _read_json(snapshot_dir / "snapshot.json", 4096)

    if set(snapshot_info) != {"version", "size_bytes", "sha256", "max_input_bytes"} or snapshot_info["version"] != SNAPSHOT_VERSION:
        raise ValueError("지원되는 snapshot 계약이 필요합니다.")

    return snapshot_dir, Snapshot(
        size_bytes=snapshot_info["size_bytes"],
        sha256=snapshot_info["sha256"],
        max_input_bytes=snapshot_info["max_input_bytes"],
    )


def _save_run(name, snapshot_dir, snapshot, profile, pipeline, limits, binding):
    """검증된 바인딩을 새 run 폴더에 저장한다. 기존 기록은 덮어쓰지 않는다."""
    return _save_selection(name, binding, lambda run_id, actor_id: {
        "schema_version": SELECTION_VERSION,
        "run_id": run_id,
        "actor_id": actor_id,
        "snapshot_directory": str(snapshot_dir),
        "snapshot": asdict(snapshot),
        "profile": profile,
        "pipeline": pipeline,
        "limits": asdict(limits),
    })


def _save_selection(name, binding, fields):
    """선택 기록 형식별 입력 필드에 바인딩·실행 계획을 더해 새 run 폴더에 저장한다."""
    directory = _directory(name, create_parent=True)

    if directory.exists() or directory.is_symlink():
        raise ValueError("기존 승인 기록은 덮어쓰지 않습니다. 새 작업 이름을 사용하세요.")

    directory.mkdir()
    _root(directory)
    actor_id = _current_actor()
    states = LocalStateStore(directory)
    state = states.create(actor_id, policy_version=binding.policy_version)
    run = open_pipeline_approval_run(directory, binding)
    selection = dict(fields(state.run_id, actor_id), binding=_json_value(binding.to_dict()),
                     execution_plan=_json_value(run.plan.to_dict()))
    payload = json.dumps(selection, ensure_ascii=False, sort_keys=True, allow_nan=False)

    if len(payload.encode("utf-8")) > MAX_SELECTION_BYTES:
        raise ValueError("승인 선택 기록이 허용 크기를 초과했습니다.")

    (directory / "selection.json").write_text(payload, encoding="utf-8")
    ApprovalStore(directory / "approvals.sqlite3")
    append_event(
        directory / "events.jsonl", "run_created", "RUN_CREATED",
        stage=state.current_stage, revision=state.revision,
    )

    return {
        "status": "created",
        "run_dir": str(directory),
        "binding_id": binding.binding_id,
        "pipeline_id": binding.plan.pipeline_id,
        "worker_started": False,
        "notice": "승인 전에는 worker를 실행하지 않습니다. 승인 소비 후 별도 격리 worker에서 실행합니다.",
    }


def _open(name):
    """선택 기록을 읽고 현재 원본·프로파일·계획이 그대로인지 재검증한다."""
    directory = _root(_directory(name))
    selection_path = directory / "selection.json"
    selection = _read_json(selection_path, MAX_SELECTION_BYTES)
    expected_fields = {
        "schema_version", "run_id", "actor_id", "snapshot_directory", "snapshot",
        "profile", "pipeline", "limits", "binding", "execution_plan",
    }

    fields = {SELECTION_VERSION: expected_fields, TABULAR_SELECTION_VERSION: TABULAR_SELECTION_FIELDS}

    if type(selection) is not dict or set(selection) != fields.get(selection.get("schema_version")):
        raise ValueError("승인 선택 기록 형식이 올바르지 않습니다.")

    states = LocalStateStore(directory)
    state = states.load()

    if state.run_id != selection["run_id"] or state.actor_id != selection["actor_id"] or state.actor_id != _current_actor():
        raise ValueError("저장된 run·OS 사용자와 현재 사용자가 일치하지 않습니다.")

    binding = _build_binding(selection)

    if _json_value(binding.to_dict()) != selection["binding"]:
        raise ValueError("입력 계약 또는 메타데이터가 저장된 승인 선택과 달라졌습니다.")

    run = open_pipeline_approval_run(directory, binding)

    if _json_value(run.plan.to_dict()) != selection["execution_plan"]:
        raise ValueError("실행 계획이 저장된 승인 선택과 달라졌습니다.")

    return directory, selection, run


def next_step(name, approval_id=None):
    """고정 계획의 한 단계를 진행한다. worker 단계는 유효한 일회 승인을 요구한다."""
    directory, _selection, run = _open(name)
    result = asdict(run.next(approval_id=approval_id))
    result["worker_started"] = run.pipeline_execution.worker_started
    return result


def preview_approval(name):
    """현재 계약 전체를 재검사해 사용자에게 보여 줄 승인 범위를 고정한다.

    CLI와 UI가 같은 검사를 사용한다. 이 함수는 승인 레코드를 발급하거나
    worker를 실행하지 않는다. 변경된 상태는 확인 당시 revision으로 탐지한다.
    """
    directory, selection, run = _open(name)
    return _approval_preview(directory, selection, run)


def _approval_preview(directory, selection, run):
    """동일하게 열린 계약에서 범위를 만든다. 승인 저장도 이 객체를 재사용한다."""
    state = run.states.load()

    if state.status != "awaiting_approval" or state.current_stage != "run_pipeline":
        raise ValueError("먼저 사전 검사를 마치고 승인 대기 상태로 진입해야 합니다.")

    approval_path = directory / "approval.json"

    if approval_path.exists() or approval_path.is_symlink():
        raise ValueError("이 run에는 이미 승인 기록이 있습니다. 새 선택으로 다시 시작하세요.")

    summary = run.binding.approval_summary()
    summary["binding_id"] = run.binding.binding_id
    summary["effect"] = "승인 범위를 저장합니다. 격리 분석 worker는 아직 실행되지 않습니다."
    return {
        "summary": summary,
        "selection_id": _digest(json.dumps(selection, sort_keys=True, ensure_ascii=False)),
        "run_id": state.run_id,
        "revision": state.revision,
        "scope": asdict(run.executor.scope_for("run_pipeline")),
    }


def record_approval(name, preview, *, confirmed=False):
    """사용자가 확인한 바로 그 범위에만 승인한다. 화면 표시로 호출하지 않는다."""
    if confirmed is not True:
        raise ValueError("승인 범위를 명시적으로 확인해야 합니다.")

    return _record_approval(name, preview, "PIPELINE_APPROVAL_RECORDED")


def record_delegated_approval(name, preview, *, authorize):
    """사용자 작업 동의 범위 안에서 조정기가 발급한다. 직접 확인 이벤트로 기록하지 않는다.

    authorize는 잠금 안에서 최신 미리보기로 동의 범위를 판정하고 벗어나면 예외를 낸다.
    """
    if not callable(authorize):
        raise ValueError("동의 범위 판정이 필요합니다.")

    return _record_approval(name, preview, "DELEGATED_APPROVAL_RECORDED", authorize)


def _record_approval(name, preview, code, authorize=None):
    directory = _root(_directory(name))

    with _run_lock(directory) as acquired:
        if not acquired:
            raise ValueError("다른 실행이 진행 중입니다.")

        # 승인 파일 존재 여부까지 잠금 안에서 검사한다. 동시 클릭도 재발급하지 않는다.
        _, selection, run = _open(name)

        if _approval_preview(directory, selection, run) != preview:
            raise ValueError("승인 확인 중 입력·코드·계획 또는 상태가 변경되었습니다.")

        if authorize is not None:
            authorize(preview)

        current_state = run.states.load()
        approval_path = directory / "approval.json"

        scope = run.executor.scope_for("run_pipeline")
        record = run.approvals.record_confirmed_approval(scope)
        approval_record = {
            "approval_id": record.approval_id,
            "expires_at": record.expires_at.isoformat(),
            "scope": asdict(scope),
            "binding_id": run.binding.binding_id,
        }

        with approval_path.open("x", encoding="utf-8") as stream:
            json.dump(approval_record, stream, ensure_ascii=False, sort_keys=True)

        current_state.approval_ids.append(record.approval_id)
        run.states.save(current_state, expected_revision=current_state.revision)
        append_event(
            directory / "events.jsonl", "approval_requested", code,
            stage=current_state.current_stage, revision=current_state.revision,
        )

    return {
        "status": "approval_recorded",
        "approval_id": record.approval_id,
        "expires_at": record.expires_at.isoformat(),
        "binding_id": run.binding.binding_id,
        "worker_started": False,
    }


def approve(name):
    """터미널 확인과 UI 확인이 동일한 승인 저장 경계를 사용한다."""
    if not os.isatty(0):
        raise ValueError("승인은 대화형 터미널에서만 할 수 있습니다.")

    preview = preview_approval(name)
    print(json.dumps(preview["summary"], ensure_ascii=False, indent=2))
    token = "PIPELINE-" + secrets.token_hex(4).upper()

    if input(f"승인하려면 {token} 을 입력하세요: ") != token:
        return {"code": "APPROVAL_DECLINED"}

    return record_approval(name, preview, confirmed=True)


def status(name):
    """계약 일치 여부와 승인 레코드 상태만 표시한다."""
    directory, _selection, run = _open(name)
    state = run.states.load()
    result = {
        "status": state.status,
        "stage": state.current_stage,
        "binding_id": run.binding.binding_id,
    }
    approval_path = directory / "approval.json"

    if approval_path.exists() and not approval_path.is_symlink():
        record = _read_json(approval_path, 16 * 1024)
        approval = run.approvals.get(record["approval_id"])

        if approval is None or asdict(approval.scope) != record["scope"]:
            raise ValueError("승인 DB 기록이 선택된 계약과 일치하지 않습니다.")

        expired = datetime.now(timezone.utc) >= approval.expires_at
        result["approval_status"] = "expired" if expired and approval.status == "pending" else approval.status
        result["approval_id"] = approval.approval_id
        result["expires_at"] = approval.expires_at.isoformat()
        if approval.scope != run.executor.scope_for("run_pipeline"):
            raise ValueError("승인 범위가 현재 실행 계획과 일치하지 않습니다.")

    return result


def main():
    """승인 요청 run 생성·사전 검사·승인·상태 조회 명령을 처리한다."""
    parser = argparse.ArgumentParser(description="등록 입력 어댑터의 파이프라인 승인 범위 관리 (worker 미연결)")
    parser.description = "등록 입력 어댑터의 파이프라인 승인 및 격리 실행 관리"
    commands = parser.add_subparsers(dest="command", required=True)

    create_parser = commands.add_parser("create")
    create_parser.add_argument("--name", required=True)
    create_parser.add_argument("--snapshot-dir", required=True)
    create_parser.add_argument("--profile", required=True)
    create_parser.add_argument("--pipeline", required=True)
    create_parser.add_argument("--wall-seconds", type=int, default=300)
    create_parser.add_argument("--memory-mib", type=int, default=2048)
    create_parser.add_argument("--processes", type=int, default=2)
    create_parser.add_argument("--confirm-selection", action="store_true")

    for command in ("next", "approve", "status"):
        sub = commands.add_parser(command)
        sub.add_argument("--name", required=True)

        if command == "next":
            sub.add_argument("--approval-id")

    args = parser.parse_args()

    try:
        if args.command == "create":
            result = create(
                args.name,
                args.snapshot_dir,
                args.profile,
                args.pipeline,
                WorkerLimits(args.wall_seconds, args.memory_mib, args.processes),
                confirmed=args.confirm_selection,
            )
        elif args.command == "next":
            result = next_step(args.name, args.approval_id)
        elif args.command == "approve":
            result = approve(args.name)
        else:
            result = status(args.name)

        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception:
        print(json.dumps({"status": "failed", "code": "PIPELINE_APPROVAL_REQUEST_FAILED"}, ensure_ascii=False))
        raise


if __name__ == "__main__":
    main()
