"""정확한 후보 상태를 Train·Validation에 적용하는 별도 승인 CLI."""

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
from .executor import RegisteredTool, RulesExecutor, _run_lock
from .isolated_eda import _read_file
from .isolated_eda_cli import cancel
from .preparation_apply import PreparationApplyTool
from .preparation_apply_contract import PreparationApplyContract, POLICY_VERSION, read_candidate_evidence
from .state import LocalStateStore
from .worker_contract import WorkerLimits, _decode


def _directory(name, *, create_parent=False):
    """고정 실행 루트 아래의 단일 이름만 허용한다."""
    if type(name) is not str or re.fullmatch(r"[a-z][a-z0-9_-]{0,47}", name) is None:
        raise ValueError("실행 이름은 소문자로 시작하는 영문·숫자·밑줄·하이픈 1~48자여야 합니다.")

    root = Path(__file__).resolve().parents[2]

    for part in ("logs", "agents", "preparation_apply", "runs"):
        root = root / part

        if create_parent:
            root.mkdir(exist_ok=True)

        _root(root)

    return root / name


def _preflight(output, arguments):
    if arguments or os.name != "nt" or sys.version_info[:2] not in ((3, 10), (3, 12)):
        return ToolResult("blocked_input")

    (output / "preflight.json").write_text(json.dumps({"version": POLICY_VERSION}), encoding="utf-8")
    return ToolResult("passed")


def _validate_preflight(output, arguments):
    return not arguments and _decode(_read_file(output / "preflight.json", 4096).decode()) == {"version": POLICY_VERSION}


def executor(tool):
    """후보 승인을 재사용하지 않고 기존 승인 1회 소비·예약·커밋 구현만 재사용한다."""
    return RulesExecutor(tool.states, ApprovalStore(tool.states.directory / "approvals.sqlite3"), tool.plan, (
        RegisteredTool("run_preflight", _preflight, _validate_preflight), tool.registered_tool(),
    ))


def create(name, candidate_run, limits, maximum, *, confirmed=False):
    """선택한 완료 후보만 결합한다. 데이터 변환·worker 실행은 생성 시 수행하지 않는다."""
    if not confirmed:
        raise ValueError("저장 기준을 Train·Validation에 적용할 대상 선택을 명시적으로 확인하세요.")

    candidate_run = str(_root(Path(candidate_run).absolute()))
    evidence = read_candidate_evidence(candidate_run, _current_actor())
    version = "preparation_apply_v2" if evidence.derivation.schema_version == "preparation_candidate_v2" else POLICY_VERSION
    contract = PreparationApplyContract(evidence, limits, maximum, schema_version=version)
    directory = _directory(name, create_parent=True)

    if directory.exists() or directory.is_symlink():
        raise ValueError("기존 적용 작업은 덮어쓰지 않습니다. 새 이름을 지정하세요.")

    states = LocalStateStore(directory)
    tool = PreparationApplyTool(states, contract)
    record = {"candidate_run": candidate_run, "candidate_id": evidence.signature,
              "limits": asdict(limits), "max_partition_bytes": maximum, "code_id": tool.code_id, "plan_id": tool.plan.plan_id}
    payload = json.dumps(record, ensure_ascii=False, allow_nan=False)
    _decode(payload)
    state = states.create(_current_actor(), policy_version=contract.schema_version)
    (directory / "selection.json").write_text(payload, encoding="utf-8")
    ApprovalStore(directory / "approvals.sqlite3")
    append_event(directory / "events.jsonl", "run_created", "RUN_CREATED", stage=state.current_stage, revision=state.revision)
    return {"status": "created", "run_dir": str(directory), "worker_started": False,
            "candidate_report": str(evidence.directory / "candidate.md"),
            "notice": "Train·Validation에 저장 기준만 적용합니다. 원본 덮어쓰기·재산출·Test 적용·모델 학습 없음."}


def open_run(name):
    directory = _root(_directory(name))
    record = _decode(_read_file(directory / "selection.json", 65536).decode("utf-8"))

    if set(record) != {"candidate_run", "candidate_id", "limits", "max_partition_bytes", "code_id", "plan_id"}:
        raise ValueError("적용 선택 기록의 필드가 올바르지 않습니다.")

    states = LocalStateStore(directory)

    if states.load().actor_id != _current_actor():
        raise ValueError("적용 작업을 생성한 OS 계정에서만 접근하세요.")

    evidence = read_candidate_evidence(record["candidate_run"], _current_actor())
    version = "preparation_apply_v2" if evidence.derivation.schema_version == "preparation_candidate_v2" else POLICY_VERSION
    contract = PreparationApplyContract(evidence, WorkerLimits(**record["limits"]), record["max_partition_bytes"], schema_version=version)
    tool = PreparationApplyTool(states, contract)

    if states.load().policy_version != version or evidence.signature != record["candidate_id"] or tool.code_id != record["code_id"] or tool.plan.plan_id != record["plan_id"]:
        raise ValueError("후보 상태·대상 구간·코드·한도가 변경되었습니다. 새 실행과 승인이 필요합니다.")

    return tool


def approve(tool):
    """정확한 후보 해시·두 구간·출력 상한을 별도 적용 승인에 결합한다."""
    if not os.isatty(0):
        raise ValueError("승인은 대화형 관리자 터미널에서 진행하세요.")

    manager = executor(tool)
    state = tool.states.load()

    if state.status != "awaiting_approval" or tool.cancel.is_set():
        raise ValueError("취소되지 않은 적용 승인 대기 작업이 필요합니다.")

    tool._check_arguments(tool.arguments)
    scope = manager.scope_for("apply_preparation")
    token = "APPLY-" + secrets.token_hex(4).upper()
    print(json.dumps({"stage": "apply_preparation", "candidate_id": tool.contract.candidate.signature,
                      "preprocessor_sha256": tool.contract.candidate.state_sha256,
                      "partitions": ["train", "validation"], "limits": asdict(tool.limits),
                      "max_partition_bytes": tool.contract.max_partition_bytes, "lifetime_minutes": 30,
                      "candidate_report": str(tool.contract.candidate.directory / "candidate.md"),
                      "notice": "원본과 후보를 보존하고 새 파생 파일을 생성합니다. 기준 재산출·Test 적용·모델 학습·LLM 전송 없음."}, ensure_ascii=False, indent=2))

    if input(f"후보 보고서를 확인했다면 {token} 를 입력하세요: ") != token:
        return {"code": "APPROVAL_DECLINED"}

    with _run_lock(tool.states.directory) as acquired:
        if not acquired or tool.states.load() != state or tool.cancel.is_set():
            raise ValueError("승인 중 실행 상태가 변경되었습니다.")

        tool._check_arguments(tool.arguments)

        if manager.scope_for("apply_preparation") != scope:
            raise ValueError("적용 승인 범위가 변경되었습니다.")

        append_event(tool.states.directory / "events.jsonl", "approval_requested", "LOCAL_REVIEW_CONFIRMED", stage=state.current_stage, revision=state.revision)
        approval = manager.approvals.record_confirmed_approval(scope)

    return {"approval_id": approval.approval_id, "expires_at": approval.expires_at.isoformat()}


def status(tool):
    state = tool.states.load()
    result = {"status": state.status, "stage": state.current_stage, "cancel_requested": tool.cancel.is_set()}

    if state.status == "completed":
        manager = executor(tool)
        task = manager.approvals.get_task(manager.scope_for("apply_preparation"))

        if task is None or task["status"] != "committed":
            raise ValueError("커밋된 적용 작업 기록이 없습니다.")

        output = manager._output(task)

        if not check_artifacts(output, manager._context(tool.plan.steps[-1])).reusable or not tool.validate(output, tool.arguments):
            raise ValueError("적용 완료 기록·파일·운송 증거가 현재 계약과 다릅니다.")

        result.update(application_status="applied", transformed=True, refit=False, test_used=False,
                      local_result_directory=str(output / "prepared"), local_report=str(output / "prepared/application.md"))

    return result


def main():
    parser = argparse.ArgumentParser(description="저장된 후보 기준을 Train·Validation에 별도 승인 후 적용")
    commands = parser.add_subparsers(dest="command", required=True)

    for command in ("create", "next", "approve", "status", "cancel"):
        sub = commands.add_parser(command)
        sub.add_argument("--name", required=True)

        if command == "create":
            sub.add_argument("--candidate-run-dir", required=True)
            sub.add_argument("--max-partition-bytes", type=int, default=32 * 1024 * 1024)
            sub.add_argument("--memory-mib", type=int, default=2048)
            sub.add_argument("--processes", type=int, default=2)
            sub.add_argument("--wall-seconds", type=int, default=300)
            sub.add_argument("--confirm-application-selection", action="store_true")
        elif command == "next":
            sub.add_argument("--approval-id")

    args = parser.parse_args()

    try:
        if args.command == "create":
            result = create(args.name, args.candidate_run_dir, WorkerLimits(args.wall_seconds, args.memory_mib, args.processes),
                            args.max_partition_bytes, confirmed=args.confirm_application_selection)
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
        print("적용 명령을 완료하지 못했습니다. 완료 후보·새 이름·대상 구간·승인·관리자 권한을 확인하세요.")
        print("기준 재산출·비격리 재시도·과거 승인 재사용·원본 덮어쓰기는 하지 않습니다.")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
