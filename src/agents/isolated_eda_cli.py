"""합성 EDA 격리 실행의 생성·단계 진행·명시 승인·취소 CLI. 실데이터 인자는 없다."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import secrets
import sys

from .approval_store import ApprovalStore
from .artifacts import _is_link, _root
from .audit import append_event
from .cli import _current_actor
from .contracts import ToolResult
from .executor import RegisteredTool, RulesExecutor, _run_lock
from .isolated_eda import SyntheticEdaTool
from .state import LocalStateStore
from .synthetic_eda_data import VERSION
from .worker_contract import WorkerLimits


def _directory(name, *, create_parent=False):
    """고정 로그 영역의 단일 이름만 허용한다. 외부 경로·데이터 경로를 받지 않는다."""
    if not re.fullmatch(r"[a-z][a-z0-9_-]{0,47}", name):
        raise ValueError("데모 이름은 소문자로 시작하는 영문·숫자·밑줄·하이픈 1~48자여야 합니다.")

    root = Path(__file__).resolve().parents[2]

    for part in ("logs", "agents", "isolated_eda", "demos"):
        root = root / part

        if create_parent:
            root.mkdir(exist_ok=True)

        _root(root)

    return root / name


def _metadata(tool):
    return {"version": VERSION, "limits": asdict(tool.limits), "config_id": tool.config_id, "plan_id": tool.plan.plan_id}


def _open(name):
    root = _root(_directory(name))
    marker = root / "demo.json"

    if _is_link(marker) or marker.stat().st_size > 65536:
        raise ValueError("유효한 합성 EDA 실행 기록이 필요합니다.")

    metadata = json.loads(marker.read_text(encoding="utf-8"))
    states = LocalStateStore(root)

    if states.load().actor_id != _current_actor():
        raise ValueError("데모를 생성한 OS 계정에서만 실행할 수 있습니다.")

    tool = SyntheticEdaTool(states, WorkerLimits(**metadata["limits"]))

    if metadata != _metadata(tool):
        raise ValueError("생성 후 코드·설정이 변경됐습니다. 기존 승인을 재사용하지 말고 새 데모를 만드세요.")

    return tool


def _preflight(output, arguments):
    """프로필·프로세스를 생성하지 않는 환경 검사다."""
    if arguments or os.name != "nt" or sys.version_info[:2] != (3, 10):
        return ToolResult("blocked_input")

    (output / "preflight.json").write_text(json.dumps({"version": VERSION, "synthetic_only": True}), encoding="utf-8")
    return ToolResult("passed")


def _validate_preflight(output, arguments):
    path = output / "preflight.json"

    return not arguments and not _is_link(path) and json.loads(path.read_text(encoding="utf-8")) == {"version": VERSION, "synthetic_only": True}


def _executor(tool):
    return RulesExecutor(tool.states, ApprovalStore(tool.states.directory / "approvals.sqlite3"), tool.plan, (
        RegisteredTool("run_preflight", _preflight, _validate_preflight), tool.registered_tool(),
    ))


def create(name, limits):
    root = _directory(name, create_parent=True)

    if root.exists() or root.is_symlink():
        raise ValueError("존재하지 않는 새 데모 이름을 지정하세요.")

    states = LocalStateStore(root)
    state = states.create(_current_actor(), policy_version=VERSION)
    tool = SyntheticEdaTool(states, limits)
    (root / "demo.json").write_text(json.dumps(_metadata(tool), sort_keys=True), encoding="utf-8")
    ApprovalStore(root / "approvals.sqlite3")
    append_event(root / "events.jsonl", "run_created", "RUN_CREATED", stage=state.current_stage, revision=state.revision)

    return {"status": "created", "synthetic_only": True, "run_dir": str(root), "limits": asdict(limits)}


def approve(tool):
    """일회성 문구를 사람이 입력한 뒤 동일 실행·코드·설정인지 다시 확인한다."""
    if not os.isatty(0):
        raise ValueError("승인은 대화형 관리자 터미널에서 직접 진행하세요.")

    executor = _executor(tool)
    state = tool.states.load()

    if state.status != "awaiting_approval" or tool.cancel.is_set():
        raise ValueError("승인 대기 중인 취소되지 않은 EDA 작업이 필요합니다.")

    scope = executor.scope_for("run_eda")
    token = "EDA-" + secrets.token_hex(4).upper()
    print(json.dumps({
        "tool_id": "run_eda", "synthetic_only": True,
        "run_id": scope.run_id, "plan_id": scope.plan_id,
        "limits": asdict(tool.limits), "lifetime_minutes": 30,
        "effect": "새 AppContainer·검사 전용 폴더 권한·Job 한도로 고정 합성 CSV의 EDA 실행",
    }, ensure_ascii=False, indent=2))

    if input(f"승인하려면 {token} 를 입력하세요: ") != token:
        return {"code": "APPROVAL_DECLINED"}

    with _run_lock(tool.states.directory) as acquired:
        if not acquired or tool.states.load() != state or tool.cancel.is_set():
            raise ValueError("승인 중 상태가 변경됐거나 실행 중입니다.")

        tool._check_arguments(tool.arguments)

        if executor.scope_for("run_eda") != scope:
            raise ValueError("승인 대상이 변경되었습니다.")

        append_event(tool.states.directory / "events.jsonl", "approval_requested", "LOCAL_REVIEW_CONFIRMED", stage=state.current_stage, revision=state.revision)
        approval = executor.approvals.record_confirmed_approval(scope)

    return {"approval_id": approval.approval_id, "expires_at": approval.expires_at.isoformat()}


def cancel(tool):
    """실행 잠금 중에도 전용 취소 신호를 남긴다. worker에게 DB·상태 쓰기 권한을 주지 않는다."""
    if tool.states.load().status == "completed":
        raise ValueError("완료된 데모는 취소할 수 없습니다.")

    path = tool.states.directory / "cancel.requested"

    if path.exists() or path.is_symlink():
        tool.cancel.is_set()
    else:
        with path.open("xb") as stream:
            stream.write(b"cancel_synthetic_eda_v1\n")
            stream.flush()
            os.fsync(stream.fileno())

    approvals = ApprovalStore(tool.states.directory / "approvals.sqlite3")
    approvals.revoke_run(tool.states.load().run_id)

    with _run_lock(tool.states.directory) as acquired:
        if acquired:
            state = tool.states.load()

            if state.status != "completed":
                state.cancel_requested = True
                state.status = "cancelled"
                tool.states.save(state, expected_revision=state.revision)

    return {"code": "RUN_CANCEL_REQUESTED"}


def main():
    parser = argparse.ArgumentParser(description="고정 합성 CSV EDA의 승인 기반 격리 실행 데모")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("create", "next", "approve", "status", "cancel"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--name", required=True)

        if command == "create":
            command_parser.add_argument("--memory-mib", type=int, default=2048)
            command_parser.add_argument("--processes", type=int, default=1)
            command_parser.add_argument("--wall-seconds", type=int, default=300)
        elif command == "next":
            command_parser.add_argument("--approval-id")

    args = parser.parse_args()

    try:
        if args.command == "create":
            result = create(args.name, WorkerLimits(args.wall_seconds, args.memory_mib, args.processes))
        else:
            tool = _open(args.name)

            if args.command == "next":
                result = asdict(_executor(tool).run_next(approval_id=args.approval_id))
            elif args.command == "approve":
                result = approve(tool)
            elif args.command == "cancel":
                result = cancel(tool)
            else:
                state = tool.states.load()
                result = {"status": state.status, "stage": state.current_stage, "cancel_requested": tool.cancel.is_set(), "synthetic_only": True}

        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception:
        print("합성 EDA 명령을 완료하지 못했습니다. 관리자 권한·새 이름·승인 대상·실행 상태를 로컬에서 확인하세요.")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
