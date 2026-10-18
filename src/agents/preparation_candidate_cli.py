"""Train 전용 전처리 기준 후보의 생성·승인·격리 실행 CLI."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import secrets
import sys

from src.preprocessing.profiles import profile_from_dict
from src.preprocessing.transform import PreparationRules
from .approval_store import ApprovalStore
from .approvals import ApprovalScope
from .artifacts import _root, check_artifacts
from .audit import append_event
from .cli import _current_actor
from .contracts import ToolResult
from .executor import RegisteredTool, RulesExecutor, _run_lock
from .isolated_eda import _read_file
from .isolated_eda_cli import cancel
from .partition_input_contract import bind_partition_input
from .preparation_candidate import PreparationCandidateTool
from .preparation_candidate_contract import PreparationCandidateContract, POLICY_VERSION, saved_export_scope
from .preparation_candidate_results import read_candidate
from .split_evidence import read_metadata, read_confirmed_split, read_committed_artifact
from .state import LocalStateStore
from .worker_contract import WorkerLimits, _decode


def _directory(name, *, create_parent=False):
    if type(name) is not str or re.fullmatch(r"[a-z][a-z0-9_-]{0,47}", name) is None:
        raise ValueError("실행 이름은 소문자로 시작하는 영문·숫자·밑줄·하이픈 1~48자여야 합니다.")

    root = Path(__file__).resolve().parents[2]

    for part in ("logs", "agents", "preparation_candidate", "runs"):
        root = root / part

        if create_parent:
            root.mkdir(exist_ok=True)

        _root(root)

    return root / name


def _train(split_run, split_scope, export_run, export_scope):
    """저장된 승인 범위와 완료 해시에서 Train만 결합한다. 과거 도구를 재실행하지 않는다."""
    actor = _current_actor()

    if split_scope.actor_id != actor or export_scope.actor_id != actor:
        raise ValueError("분할·내보내기를 승인한 OS 계정에서만 후보를 생성하세요.")

    split = read_confirmed_split(split_run, split_scope)
    export = read_committed_artifact(export_run, export_scope)
    profile = profile_from_dict(read_metadata(Path(split_run) / "profile.json"))

    if profile.signature != split_scope.config_id:
        raise ValueError("분할 프로파일과 확정 승인 범위가 다릅니다.")

    return bind_partition_input(split, export, "train", profile.roles)


def _preflight(output, arguments):
    if arguments or os.name != "nt" or sys.version_info[:2] != (3, 10):
        return ToolResult("blocked_input")

    (output / "preflight.json").write_text(json.dumps({"version": POLICY_VERSION}), encoding="utf-8")
    return ToolResult("passed")


def _validate_preflight(output, arguments):
    return not arguments and _decode(_read_file(output / "preflight.json", 4096).decode()) == {"version": POLICY_VERSION}


def executor(tool):
    return RulesExecutor(tool.states, ApprovalStore(tool.states.directory / "approvals.sqlite3"), tool.plan, (
        RegisteredTool("run_preflight", _preflight, _validate_preflight), tool.registered_tool(),
    ))


def create(name, split_run, split_scope_path, export_run, rules_path, limits, maximum, *, confirmed=False):
    if not confirmed:
        raise ValueError("Train 계약 선택과 후보 기준 산출만 허용함을 명시적으로 확인하세요.")

    split_run = str(_root(Path(split_run).absolute()))
    export_run = str(_root(Path(export_run).absolute()))
    split_scope = ApprovalScope(**read_metadata(Path(split_scope_path).absolute()))
    export_scope = saved_export_scope(export_run)
    train = _train(split_run, split_scope, export_run, export_scope)
    rules = PreparationRules(**_decode(_read_file(Path(rules_path).absolute(), 65536).decode("utf-8")))
    contract = PreparationCandidateContract(train, rules, limits, maximum)
    directory = _directory(name, create_parent=True)

    if directory.exists() or directory.is_symlink():
        raise ValueError("기존 후보 실행은 덮어쓰지 않습니다. 새 이름을 사용하세요.")

    states = LocalStateStore(directory)
    tool = PreparationCandidateTool(states, contract)
    record = {"split_run": split_run, "split_scope": asdict(split_scope), "export_run": export_run,
              "export_scope": asdict(export_scope), "train_contract_id": train.contract_id,
              "rules": asdict(rules), "limits": asdict(limits), "max_candidate_bytes": maximum,
              "code_id": tool.code_id, "plan_id": tool.plan.plan_id}
    payload = json.dumps(record, ensure_ascii=False, allow_nan=False)
    _decode(payload)
    state = states.create(_current_actor(), policy_version=POLICY_VERSION)
    (directory / "selection.json").write_text(payload, encoding="utf-8")
    ApprovalStore(directory / "approvals.sqlite3")
    append_event(directory / "events.jsonl", "run_created", "RUN_CREATED", stage=state.current_stage, revision=state.revision)
    return {"status": "created", "run_dir": str(directory), "worker_started": False,
            "notice": "Train 전용 후보 기준 산출입니다. 실제 변환·모델 학습은 하지 않습니다."}


def open_run(name):
    directory = _root(_directory(name))
    record = _decode(_read_file(directory / "selection.json", 65536).decode("utf-8"))
    fields = {"split_run", "split_scope", "export_run", "export_scope", "train_contract_id", "rules", "limits", "max_candidate_bytes", "code_id", "plan_id"}

    if set(record) != fields:
        raise ValueError("후보 선택 기록의 필드가 올바르지 않습니다.")

    states = LocalStateStore(directory)

    if states.load().actor_id != _current_actor():
        raise ValueError("후보 실행을 생성한 OS 계정에서만 접근하세요.")

    train = _train(record["split_run"], ApprovalScope(**record["split_scope"]), record["export_run"], ApprovalScope(**record["export_scope"]))
    contract = PreparationCandidateContract(train, PreparationRules(**record["rules"]), WorkerLimits(**record["limits"]), record["max_candidate_bytes"])
    tool = PreparationCandidateTool(states, contract)

    if record["train_contract_id"] != train.contract_id or record["code_id"] != tool.code_id or record["plan_id"] != tool.plan.plan_id:
        raise ValueError("Train 증거·코드·설정·계획이 변경되었습니다. 새 실행과 승인이 필요합니다.")

    return tool


def approve(tool):
    if not os.isatty(0):
        raise ValueError("승인은 대화형 관리자 터미널에서 진행하세요.")

    manager = executor(tool)
    state = tool.states.load()

    if state.status != "awaiting_approval" or tool.cancel.is_set():
        raise ValueError("취소되지 않은 후보 산출 승인 대기 작업이 필요합니다.")

    tool._check_arguments(tool.arguments)
    scope = manager.scope_for("derive_preparation")
    token = "CANDIDATE-" + secrets.token_hex(4).upper()
    print(json.dumps({"stage": "derive_preparation", "partition": "train", "rules": asdict(tool.contract.rules),
                      "limits": asdict(tool.limits), "max_candidate_bytes": tool.contract.max_candidate_bytes,
                      "lifetime_minutes": 30,
                      "notice": "Train 기준 후보만 산출합니다. Validation·Test 전달, 데이터 변환 적용, 모델 학습, LLM 전송 없음."}, ensure_ascii=False, indent=2))

    if input(f"승인하려면 {token} 를 입력하세요: ") != token:
        return {"code": "APPROVAL_DECLINED"}

    with _run_lock(tool.states.directory) as acquired:
        if not acquired or tool.states.load() != state or tool.cancel.is_set():
            raise ValueError("승인 중 실행 상태가 변경되었습니다.")

        tool._check_arguments(tool.arguments)

        if manager.scope_for("derive_preparation") != scope:
            raise ValueError("후보 산출 승인 범위가 변경되었습니다.")

        append_event(tool.states.directory / "events.jsonl", "approval_requested", "LOCAL_REVIEW_CONFIRMED", stage=state.current_stage, revision=state.revision)
        approval = manager.approvals.record_confirmed_approval(scope)

    return {"approval_id": approval.approval_id, "expires_at": approval.expires_at.isoformat()}


def status(tool):
    state = tool.states.load()
    result = {"status": state.status, "stage": state.current_stage, "cancel_requested": tool.cancel.is_set()}

    if state.status == "completed":
        manager = executor(tool)
        task = manager.approvals.get_task(manager.scope_for("derive_preparation"))

        if task is None or task["status"] != "committed":
            raise ValueError("커밋된 후보 작업 기록이 없습니다.")

        output = manager._output(task)

        if not check_artifacts(output, manager._context(tool.plan.steps[-1])).reusable or not tool.validate(output, tool.arguments):
            raise ValueError("후보 완료 기록·운송 증거·파일이 현재 계약과 다릅니다.")

        result.update(candidate_status="candidate_ready", transformed=False, local_report=str(output / "candidate/candidate.md"))

    return result


def main():
    parser = argparse.ArgumentParser(description="Train 전용 전처리 기준 후보의 승인 기반 격리 산출")
    commands = parser.add_subparsers(dest="command", required=True)

    for command in ("create", "next", "approve", "status", "cancel"):
        sub = commands.add_parser(command)
        sub.add_argument("--name", required=True)

        if command == "create":
            for option in ("split-run-dir", "split-scope", "export-run-dir", "rules"):
                sub.add_argument("--" + option, required=True)

            sub.add_argument("--max-candidate-bytes", type=int, default=4 * 1024 * 1024)
            sub.add_argument("--memory-mib", type=int, default=2048)
            sub.add_argument("--processes", type=int, default=2)
            sub.add_argument("--wall-seconds", type=int, default=300)
            sub.add_argument("--confirm-candidate-only", action="store_true")
        elif command == "next":
            sub.add_argument("--approval-id")

    args = parser.parse_args()

    try:
        if args.command == "create":
            result = create(args.name, args.split_run_dir, args.split_scope, args.export_run_dir, args.rules,
                            WorkerLimits(args.wall_seconds, args.memory_mib, args.processes), args.max_candidate_bytes,
                            confirmed=args.confirm_candidate_only)
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
        print("후보 명령을 완료하지 못했습니다. Train 완료 증거·설정·새 이름·승인·관리자 권한을 확인하세요.")
        print("기준 적용·비격리 재시도·과거 승인 재사용은 하지 않습니다.")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
