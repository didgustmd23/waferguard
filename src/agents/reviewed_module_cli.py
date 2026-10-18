"""참조 모듈의 합성 전용 승인 CLI. 임의 원본·Python 경로를 받지 않는다."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import secrets
import sys

from .approval_store import ApprovalStore
from .artifacts import _root, check_artifacts
from .audit import append_event
from .cli import _current_actor
from .contracts import ToolResult
from .eda_input_contract import EdaInputContract, bind_eda_input
from .executor import RulesExecutor, RegisteredTool, _run_lock
from .input_snapshot import create_snapshot
from .isolated_eda import _read_file
from .isolated_eda_cli import cancel
from .reviewed_module import ReviewedModuleTool
from .reviewed_module_contract import POLICY_VERSION, demo_profile, module_identity
from .state import LocalStateStore
from .synthetic_eda_data import CSV_BYTES
from .worker_contract import WorkerLimits, _decode


def _directory(name, *, create_parent=False):
    if type(name) is not str or re.fullmatch(r"[a-z][a-z0-9_-]{0,47}", name) is None:
        raise ValueError("실행 이름은 소문자로 시작하는 영문·숫자·밑줄·하이픈 1~48자여야 합니다.")

    root = Path(__file__).resolve().parents[2]

    for part in ("logs", "agents", "reviewed_module", "runs"):
        root = root / part

        if create_parent:
            root.mkdir(exist_ok=True)

        _root(root)

    return root / name


def _preflight(output, arguments):
    if arguments or os.name != "nt" or sys.version_info[:2] != (3, 10):
        return ToolResult("blocked_input")

    (output / "preflight.json").write_text(json.dumps({"version": POLICY_VERSION}), encoding="utf-8")
    return ToolResult("passed")


def _validate_preflight(output, arguments):
    return not arguments and _decode(_read_file(output / "preflight.json", 4096).decode()) == {"version": POLICY_VERSION}


def executor(tool):
    return RulesExecutor(tool.states, ApprovalStore(tool.states.directory / "approvals.sqlite3"), tool.plan,
                         (RegisteredTool("run_preflight", _preflight, _validate_preflight), tool.registered_tool()))


def create_demo(name, limits, include_complete=True):
    """고정 fixture·계약만 저장한다. 확장 코드 import·worker 시작·승인은 하지 않는다."""
    directory = _directory(name, create_parent=True)

    if directory.exists() or directory.is_symlink():
        raise ValueError("기존 참조 모듈 실행은 덮어쓰지 않습니다. 새 이름을 지정하세요.")

    examples = directory.parent.parent / "examples"
    examples.mkdir(exist_ok=True)
    _root(examples)
    example = examples / name
    example.mkdir()
    (example / "source.csv").write_bytes(CSV_BYTES)
    snapshot = create_snapshot(example / "source.csv", example / "snapshot", max_input_bytes=1024 * 1024)
    contract = bind_eda_input(example / "snapshot", snapshot, demo_profile(), limits)
    states = LocalStateStore(directory)
    tool = ReviewedModuleTool(states, contract, include_complete)
    record = {"contract": contract.to_dict(), "include_complete": include_complete,
              "code_id": tool.code_id, "plan_id": tool.plan.plan_id}
    payload = json.dumps(record, ensure_ascii=False, allow_nan=False)
    _decode(payload)
    state = states.create(_current_actor(), policy_version=POLICY_VERSION)
    (directory / "selection.json").write_text(payload, encoding="utf-8")
    ApprovalStore(directory / "approvals.sqlite3")
    append_event(directory / "events.jsonl", "run_created", "RUN_CREATED", stage=state.current_stage, revision=state.revision)
    return {"status": "created", "run_dir": str(directory), "worker_started": False,
            "synthetic_only": True, "module": module_identity(include_complete)}


def open_run(name):
    directory = _root(_directory(name))
    record = _decode(_read_file(directory / "selection.json", 65536).decode("utf-8"))

    if set(record) != {"contract", "include_complete", "code_id", "plan_id"}:
        raise ValueError("참조 모듈 선택 기록이 올바르지 않습니다.")

    states = LocalStateStore(directory)

    if states.load().actor_id != _current_actor():
        raise ValueError("작업을 생성한 OS 계정에서만 접근하세요.")

    tool = ReviewedModuleTool(states, EdaInputContract.from_json(json.dumps(record["contract"])), record["include_complete"])

    if record["code_id"] != tool.code_id or record["plan_id"] != tool.plan.plan_id:
        raise ValueError("모듈·명세·설정·한도가 변경되었습니다. 새 실행과 승인이 필요합니다.")

    return tool


def approve(tool):
    if not os.isatty(0):
        raise ValueError("대화형 관리자 터미널에서 승인하세요.")

    manager = executor(tool)
    state = tool.states.load()

    if state.status != "awaiting_approval" or tool.cancel.is_set():
        raise ValueError("취소되지 않은 승인 대기 작업이 필요합니다.")

    tool._check_arguments(tool.arguments)
    scope = manager.scope_for("run_eda")
    token = "MODULE-" + secrets.token_hex(4).upper()
    print(json.dumps({"module": module_identity(tool.include_complete), "code_id": tool.code_id,
                      "source_id": tool.contract.snapshot.source_id, "limits": asdict(tool.limits),
                      "lifetime_minutes": 30,
                      "effect": "고정 합성 CSV 품질 검사 후 참조 모듈을 새 AppContainer+Job에서 실행",
                      "notice": "실데이터·임의 Python·모델 학습·LLM 전송 없음"}, ensure_ascii=False, indent=2))

    if input(f"승인하려면 {token} 를 입력하세요: ") != token:
        return {"code": "APPROVAL_DECLINED"}

    with _run_lock(tool.states.directory) as acquired:
        if not acquired or tool.states.load() != state or tool.cancel.is_set():
            raise ValueError("승인 중 상태가 변경되었습니다.")

        tool._check_arguments(tool.arguments)

        if manager.scope_for("run_eda") != scope:
            raise ValueError("승인 범위가 변경되었습니다.")

        append_event(tool.states.directory / "events.jsonl", "approval_requested", "LOCAL_REVIEW_CONFIRMED", stage=state.current_stage, revision=state.revision)
        approval = manager.approvals.record_confirmed_approval(scope)

    return {"approval_id": approval.approval_id, "expires_at": approval.expires_at.isoformat()}


def status(tool):
    state = tool.states.load()
    result = {"status": state.status, "stage": state.current_stage,
              "synthetic_only": True, "cancel_requested": tool.cancel.is_set()}

    if state.status == "completed":
        manager = executor(tool)
        task = manager.approvals.get_task(manager.scope_for("run_eda"))

        if task is None or task["status"] != "committed":
            raise ValueError("커밋된 참조 모듈 기록이 없습니다.")

        output = manager._output(task)

        if not check_artifacts(output, manager._context(tool.plan.steps[-1])).reusable or not tool.validate(output, tool.arguments):
            raise ValueError("완료 파일 또는 운송 증거가 현재 계약과 다릅니다.")

        result.update(module_status="passed", local_result_directory=str(output / "pipeline"))

    return result


def main():
    parser = argparse.ArgumentParser(description="검토한 참조 모듈의 합성 승인·격리 실행")
    commands = parser.add_subparsers(dest="command", required=True)

    for command in ("create-demo", "next", "approve", "status", "cancel"):
        sub = commands.add_parser(command)
        sub.add_argument("--name", required=True)

        if command == "create-demo":
            sub.add_argument("--memory-mib", type=int, default=2048)
            sub.add_argument("--processes", type=int, default=2)
            sub.add_argument("--wall-seconds", type=int, default=60)
            sub.add_argument("--exclude-complete", action="store_true")
        elif command == "next":
            sub.add_argument("--approval-id")

    args = parser.parse_args()

    try:
        if args.command == "create-demo":
            result = create_demo(args.name, WorkerLimits(args.wall_seconds, args.memory_mib, args.processes), not args.exclude_complete)
        else:
            tool = open_run(args.name)

            if args.command == "next":
                result = asdict(executor(tool).run_next(approval_id=args.approval_id))
            elif args.command == "approve":
                result = approve(tool)
            elif args.command == "cancel":
                result = cancel(tool)
            else:
                result = status(tool)

        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception:
        print("참조 모듈 명령을 완료하지 못했습니다. 새 이름·명세·입력·승인·한도·관리자 권한을 확인하세요.")
        print("실데이터 활성화·비격리 대체 실행·자동 재시도는 하지 않습니다.")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
