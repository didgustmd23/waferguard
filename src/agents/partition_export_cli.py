"""확정 분할의 구간 내보내기 생성·승인·실행 CLI. 새 분할이나 자동 승인은 하지 않는다."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import secrets
import sys

from .approval_store import ApprovalStore
from .approvals import ApprovalScope
from .artifacts import _root
from .audit import append_event
from .cli import _current_actor
from .contracts import ToolResult
from .eda_input_contract import EdaInputContract, bind_eda_input
from .executor import RegisteredTool, RulesExecutor, _run_lock
from .input_snapshot import SNAPSHOT_VERSION, Snapshot
from .isolated_eda import _read_file
from .isolated_eda_cli import cancel
from .partition_export import PartitionExportTool
from .partition_export_contract import PartitionExportContract, POLICY_VERSION
from .partition_input_contract import bind_partition_input
from .split_evidence import read_confirmed_split, read_committed_artifact, read_metadata
from .state import LocalStateStore
from .worker_contract import WorkerLimits, _decode
from .worker_paths import _path
from src.preprocessing.profiles import profile_from_dict


def _directory(name, *, create_parent=False):
    """실행 이름으로 경로를 탈출하지 못하게 고정 루트의 단일 이름만 허용한다."""
    if type(name) is not str or re.fullmatch(r"[a-z][a-z0-9_-]{0,47}", name) is None:
        raise ValueError("실행 이름은 소문자로 시작하는 영문·숫자·밑줄·하이픈 1~48자여야 합니다.")

    root = Path(__file__).resolve().parents[2]

    for part in ("logs", "agents", "partition_export", "runs"):
        root = root / part

        if create_parent:
            root.mkdir(exist_ok=True)

        _root(root)

    return root / name


def _preflight(output, arguments):
    """지원 플랫폼만 확인한다. 이 단계의 통과는 OS 격리 성공 증거가 아니다."""
    if arguments or os.name != "nt" or sys.version_info[:2] not in ((3, 10), (3, 12)):
        return ToolResult("blocked_input")

    (output / "preflight.json").write_text(json.dumps({"version": POLICY_VERSION}), encoding="utf-8")

    return ToolResult("passed")


def _validate_preflight(output, arguments):
    return not arguments and _decode(_read_file(output / "preflight.json", 4096).decode()) == {"version": POLICY_VERSION}


def executor(tool):
    """기존 승인 1회 소비·예약·취소·검증·커밋 구현을 그대로 사용한다."""
    return RulesExecutor(tool.states, ApprovalStore(tool.states.directory / "approvals.sqlite3"), tool.plan, (
        RegisteredTool("run_preflight", _preflight, _validate_preflight), tool.registered_tool(),
    ))


def create(name, split_run, split_scope, snapshot_directory, profile_path, limits, max_partition_bytes, *, confirmed=False):
    """사용자가 명시한 확정 분할과 동일 원본을 고정한다. 생성만으로 worker는 시작하지 않는다."""
    if not confirmed:
        raise ValueError("확정 분할·원본 복사본 선택을 명시적으로 확인해야 합니다.")

    scope = ApprovalScope(**read_metadata(_path(Path(split_scope).absolute(), directory=False)))
    split = read_confirmed_split(Path(split_run).absolute(), scope)

    if scope.actor_id != _current_actor():
        raise ValueError("확정 분할을 승인한 OS 계정에서 내보내기를 생성하세요.")

    selected = _path(Path(snapshot_directory).absolute(), directory=True)
    metadata = _decode(_read_file(selected / "snapshot.json", 4096).decode("utf-8"))

    if set(metadata) != {"version", *Snapshot.__dataclass_fields__} or metadata["version"] != SNAPSHOT_VERSION:
        raise ValueError("등록된 snapshot 계약이 필요합니다.")

    snapshot = Snapshot(**{key: metadata[key] for key in Snapshot.__dataclass_fields__})
    profile = profile_from_dict(_decode(_read_file(_path(Path(profile_path).absolute(), directory=False), 65536).decode("utf-8")))
    source = bind_eda_input(selected, snapshot, profile, limits)
    contract = PartitionExportContract(split, source, max_partition_bytes)
    directory = _directory(name, create_parent=True)

    if directory.exists() or directory.is_symlink():
        raise ValueError("기존 실행을 덮어쓰지 않습니다. 새 이름을 지정하세요.")

    states = LocalStateStore(directory)
    tool = PartitionExportTool(states, contract)
    record = {
        "split_run": split.artifact.run_directory, "split_scope": asdict(scope),
        "split_evidence_id": split.signature, "source": source.to_dict(),
        "max_partition_bytes": max_partition_bytes, "code_id": tool.code_id, "plan_id": tool.plan.plan_id,
    }
    payload = json.dumps(record, ensure_ascii=False, allow_nan=False)
    _decode(payload)
    state = states.create(_current_actor(), policy_version=POLICY_VERSION)
    (directory / "selection.json").write_text(payload, encoding="utf-8")
    ApprovalStore(directory / "approvals.sqlite3")
    append_event(directory / "events.jsonl", "run_created", "RUN_CREATED", stage=state.current_stage, revision=state.revision)

    return {"status": "created", "run_dir": str(directory), "worker_started": False, "partitions": ["train", "validation"]}


def open_run(name):
    """저장된 선택과 현재 증거를 다시 결합한다. 변경된 코드는 새 실행·승인이 필요하다."""
    directory = _root(_directory(name))
    record = _decode(_read_file(directory / "selection.json", 65536).decode("utf-8"))

    if set(record) != {"split_run", "split_scope", "split_evidence_id", "source", "max_partition_bytes", "code_id", "plan_id"}:
        raise ValueError("내보내기 선택 기록의 필드가 올바르지 않습니다.")

    states = LocalStateStore(directory)
    actor = _current_actor()
    scope = ApprovalScope(**record["split_scope"])

    if states.load().actor_id != actor or scope.actor_id != actor:
        raise ValueError("실행과 분할을 승인한 OS 계정에서만 접근할 수 있습니다.")

    split = read_confirmed_split(record["split_run"], scope)
    source = EdaInputContract.from_json(json.dumps(record["source"]))
    tool = PartitionExportTool(states, PartitionExportContract(split, source, record["max_partition_bytes"]))

    if record["split_evidence_id"] != split.signature or record["code_id"] != tool.code_id or record["plan_id"] != tool.plan.plan_id:
        raise ValueError("분할·코드·계획이 변경되었습니다. 새 실행과 승인이 필요합니다.")

    return tool


def approve(tool):
    """구간·상한·Test 읽기 범위를 표시하고 대화형 확인 후 정확한 작업 승인만 발급한다."""
    if not os.isatty(0):
        raise ValueError("승인은 대화형 관리자 터미널에서 진행하세요.")

    manager = executor(tool)
    state = tool.states.load()

    if state.status != "awaiting_approval" or tool.cancel.is_set():
        raise ValueError("취소되지 않은 승인 대기 작업이 필요합니다.")

    tool._check_arguments(tool.arguments)
    scope = manager.scope_for("ingest")
    token = "EXPORT-" + secrets.token_hex(4).upper()
    print(json.dumps({
        "partitions": ["train", "validation"], "split_evidence_id": tool.contract.split.signature,
        "snapshot_bytes": tool.contract.source.snapshot.size_bytes,
        "max_partition_bytes": tool.contract.max_partition_bytes, "limits": asdict(tool.limits),
        "lifetime_minutes": 30,
        "effect": "새 AppContainer+Job에서 확정 행만 내보내고 부모 검증 후 커밋",
        "notice": "원본 전체를 순회하여 Test 배정 행도 읽습니다. Test 출력·통계 산출·대치·스케일링·모델 학습·LLM 전송은 하지 않습니다.",
    }, ensure_ascii=False, indent=2))

    if input(f"승인하려면 {token} 를 입력하세요: ") != token:
        return {"code": "APPROVAL_DECLINED"}

    with _run_lock(tool.states.directory) as acquired:
        if not acquired or tool.states.load() != state or tool.cancel.is_set():
            raise ValueError("승인 중 실행 상태가 변경되었습니다.")

        tool._check_arguments(tool.arguments)

        if manager.scope_for("ingest") != scope:
            raise ValueError("승인 범위가 변경되었습니다.")

        append_event(tool.states.directory / "events.jsonl", "approval_requested", "LOCAL_REVIEW_CONFIRMED", stage=state.current_stage, revision=state.revision)
        record = manager.approvals.record_confirmed_approval(scope)

    return {"approval_id": record.approval_id, "expires_at": record.expires_at.isoformat()}


def status(tool):
    """완료 이력과 구간 입력 계약을 재검증한다. 데이터 내용은 콘솔에 출력하지 않는다."""
    state = tool.states.load()
    result = {"status": state.status, "stage": state.current_stage, "cancel_requested": tool.cancel.is_set()}

    if state.status == "completed":
        scope = executor(tool).scope_for("ingest")
        export = read_committed_artifact(tool.states.directory, scope)

        if not tool.validate(export.directory, tool.arguments):
            raise ValueError("완료 파일 또는 worker 운송 증거가 현재 계약과 다릅니다.")

        for name in ("train", "validation"):
            bind_partition_input(tool.contract.split, export, name, tool.contract.source.profile.roles).check()

        result.update(partition_contracts="passed", local_result_directory=str(export.directory))

    return result


def main():
    parser = argparse.ArgumentParser(description="확정 분할의 승인 기반 Train·Validation 내보내기")
    commands = parser.add_subparsers(dest="command", required=True)

    for command in ("create", "next", "approve", "status", "cancel"):
        sub = commands.add_parser(command)
        sub.add_argument("--name", required=True)

        if command == "create":
            for option in ("split-run-dir", "split-scope", "snapshot-dir", "profile"):
                sub.add_argument("--" + option, required=True)

            sub.add_argument("--max-partition-bytes", type=int, required=True)
            sub.add_argument("--wall-seconds", type=int, default=300)
            sub.add_argument("--memory-mib", type=int, default=2048)
            sub.add_argument("--processes", type=int, default=2)
            sub.add_argument("--confirm-selection", action="store_true")
        elif command == "next":
            sub.add_argument("--approval-id")

    args = parser.parse_args()

    try:
        if args.command == "create":
            result = create(args.name, args.split_run_dir, args.split_scope, args.snapshot_dir, args.profile,
                            WorkerLimits(args.wall_seconds, args.memory_mib, args.processes),
                            args.max_partition_bytes, confirmed=args.confirm_selection)
        else:
            tool = open_run(args.name)

            if args.command == "next":
                tool._check_arguments(tool.arguments)
                result = asdict(executor(tool).run_next(approval_id=args.approval_id))
            elif args.command == "approve":
                result = approve(tool)
            elif args.command == "cancel":
                result = cancel(tool)
            else:
                result = status(tool)

        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception:
        print("내보내기 명령을 완료하지 못했습니다. 확정 분할·원본·프로파일·새 실행 이름·승인·관리자 권한을 확인하세요.")
        print("비격리 재시도·기존 승인 재사용·원본 덮어쓰기는 하지 않습니다.")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
