"""완료된 WM 분할 제안의 고정 배정을 별도 승인으로 확정하는 CLI."""

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import re
import secrets
import sys

from src.preprocessing.canonical_data import decode_json
from .approval_store import ApprovalStore
from .artifacts import _root
from .audit import append_event
from .cli import _current_actor
from .contracts import ToolResult
from .executor import RegisteredTool, RulesExecutor, _run_lock
from .isolated_eda import _read_file
from .isolated_eda_cli import cancel
from .split_evidence import read_committed_artifact
from .state import LocalStateStore
from .wm_canonical_contract import WmCanonicalInput, digest
from .wm_split_commit import WmSplitCommitTool
from .wm_split_commit_evidence import read_proposal_evidence
from .wm_split_commit_results import POLICY_VERSION
from .worker_contract import WorkerLimits, _decode


def _directory(name, *, create_parent=False):
    """새 적용 작업만 별도 이름 공간에 저장하고 기존 품질·후보 기록은 보존한다."""
    if type(name) is not str or re.fullmatch(r"[a-z][a-z0-9_-]{0,47}", name) is None:
        raise ValueError("작업 이름은 소문자로 시작하는 영문·숫자·밑줄·하이픈 1~48자여야 합니다.")

    directory = Path(__file__).resolve().parents[2]

    for part in ("logs", "agents", "wm_split_commit", "runs"):
        directory /= part

        if create_parent:
            directory.mkdir(exist_ok=True)

        _root(directory)

    return directory / name


def _preflight(output, arguments):
    """입력 배열을 읽지 않고 지원 실행 환경만 확인한다."""
    if arguments or os.name != "nt" or sys.version_info[:2] != (3, 10):
        return ToolResult("blocked_input")

    (output / "preflight.json").write_text(json.dumps({"version": POLICY_VERSION}), encoding="utf-8")
    return ToolResult("passed")


def _validate_preflight(output, arguments):
    return not arguments and _decode(_read_file(output / "preflight.json", 4096).decode()) == {"version": POLICY_VERSION}


def executor(tool):
    return RulesExecutor(tool.states, ApprovalStore(tool.states.directory / "approvals.sqlite3"), tool.plan, (
        RegisteredTool("run_preflight", _preflight, _validate_preflight), *tool.reference_tools(), tool.registered_tool(),
    ))


def create(name, proposal_run, limits, max_output_bytes=2 * 1024 ** 3, *, confirmed=False):
    """완료된 제안과 그룹 배정을 고정한다. 확정은 새 승인을 받은 뒤 수행한다."""
    if not confirmed:
        raise ValueError("검토한 고정 배정의 확정을 명시적으로 확인해야 합니다.")

    previous, evidence = read_proposal_evidence(proposal_run)
    contract = replace(previous, limits=limits, max_output_bytes=max_output_bytes, include_examples=False)
    directory = _directory(name, create_parent=True)

    if directory.exists() or directory.is_symlink():
        raise ValueError("기존 적용 작업은 덮어쓰지 않습니다. 새 작업 이름을 사용하세요.")

    directory.mkdir()
    states = LocalStateStore(directory)
    tool = WmSplitCommitTool(states, contract, proposal_run, evidence)
    tool._check_arguments(tool.arguments)
    state = states.create(_current_actor(), policy_version=POLICY_VERSION)
    record = {"contract": contract.to_dict(), "proposal_run": proposal_run, "evidence": evidence,
              "code_id": tool.code_id, "plan_id": tool.plan.plan_id}
    payload = json.dumps(record, ensure_ascii=True).encode("utf-8")

    if len(payload) > 1024 * 1024:
        raise ValueError("적용 선택 기록이 크기 상한을 초과했습니다.")

    (directory / "selection.json").write_bytes(payload)
    ApprovalStore(directory / "approvals.sqlite3")
    append_event(directory / "events.jsonl", "run_created", "RUN_CREATED",
                 stage=state.current_stage, revision=state.revision)
    return {"status": "created", "run_dir": str(directory), "worker_started": False,
            "notice": "검토한 분할안의 확정을 준비합니다. 실제 확정에는 새 승인이 필요합니다."}


def open_run(name):
    """같은 사용자·입력·품질 증거·코드·계획으로만 재개한다."""
    directory = _root(_directory(name))
    record = decode_json(_read_file(directory / "selection.json", 1024 * 1024))

    if type(record) is not dict or set(record) != {"contract", "proposal_run", "evidence", "code_id", "plan_id"}:
        raise ValueError("WM 그룹 분할 확정의 선택 기록이 올바르지 않습니다.")

    states = LocalStateStore(directory)

    if states.load().actor_id != _current_actor():
        raise ValueError("작업을 생성한 OS 계정에서 실행하세요.")

    tool = WmSplitCommitTool(states, WmCanonicalInput.from_dict(record["contract"]), record["proposal_run"], record["evidence"])

    if tool.code_id != record["code_id"] or tool.plan.plan_id != record["plan_id"]:
        raise ValueError("입력·코드·계획이 변경되었습니다. 새 작업과 승인이 필요합니다.")

    tool._check_arguments(tool.arguments)
    return tool


def approve(tool):
    """민감한 해시 목록·행 위치는 출력하지 않고 정확한 작업 범위에만 승인한다."""
    if not os.isatty(0):
        raise ValueError("승인은 대화형 터미널에서 직접 진행하세요.")

    state = tool.states.load()
    manager = executor(tool)

    if state.status != "awaiting_approval" or tool.cancel.is_set():
        raise ValueError("취소되지 않은 승인 대기 작업이 필요합니다.")

    tool._check_arguments(tool.arguments)
    if state.current_stage not in {"review_eda", "commit_split"}:
        raise ValueError("검토 또는 확정 승인 단계가 필요합니다.")

    scope = manager.scope_for(state.current_stage)
    token = ("WM-REVIEW-" if state.current_stage == "review_eda" else "WM-COMMIT-") + secrets.token_hex(4).upper()
    print(json.dumps({"source_id": tool.contract.source_id, "proposal_evidence_id": digest(tool.evidence),
                      "stage": state.current_stage,
                      "split_config": tool.evidence["proposal"]["config"],
                      "local_proposal_report": str(Path(tool.evidence["proposal_directory"]) / "proposal.md"),
                      "limits": asdict(tool.limits), "max_output_bytes": tool.contract.max_output_bytes,
                      "lifetime_minutes": 30,
                      "effect": ("기존 EDA·분할안 검토만 확인. 확정 worker 미실행" if state.current_stage == "review_eda"
                                 else "AppContainer+Job에서 고정 배정의 manifest 생성·행 대응 재검증"),
                      "notice": "후보 재탐색·배열 파싱·추가 제외·이미지 변환·학습·LLM 전송 없음"},
                     ensure_ascii=False, indent=2))

    confirmation = "EDA·분할안 검토를 확인하려면" if state.current_stage == "review_eda" else "분할 확정을 승인하려면"

    if input(f"{confirmation} {token} 을 입력하세요: ") != token:
        return {"code": "APPROVAL_DECLINED"}

    with _run_lock(tool.states.directory) as acquired:
        if not acquired or tool.states.load() != state or tool.cancel.is_set():
            raise ValueError("승인 중 작업 상태가 변경되었습니다.")

        tool._check_arguments(tool.arguments)

        if manager.scope_for(state.current_stage) != scope:
            raise ValueError("승인 범위가 변경되었습니다.")

        append_event(tool.states.directory / "events.jsonl", "approval_requested", "LOCAL_REVIEW_CONFIRMED",
                     stage=state.current_stage, revision=state.revision)
        record = manager.approvals.record_confirmed_approval(scope)

    return {"approval_id": record.approval_id, "expires_at": record.expires_at.isoformat()}


def status(tool):
    state = tool.states.load()
    result = {"status": state.status, "stage": state.current_stage, "cancel_requested": tool.cancel.is_set()}

    if state.status == "completed":
        scope = executor(tool).scope_for("commit_split")
        attempt = read_committed_artifact(tool.states.directory, scope).directory
        tool.inspect_attempt(attempt)
        result.update(split_complete=True, training_ready=False,
                      local_split_directory=str(attempt / "split"),
                      local_report=str(attempt / "split/split_summary.md"))

    return result


def main():
    parser = argparse.ArgumentParser(description="WM 고정 그룹 분할 확정의 승인 기반 격리 실행")
    commands = parser.add_subparsers(dest="command", required=True)

    for command in ("create", "next", "approve", "status", "cancel"):
        sub = commands.add_parser(command)
        sub.add_argument("--name", required=True)

        if command == "create":
            sub.add_argument("--proposal-run", required=True)
            sub.add_argument("--memory-mib", type=int, default=8192)
            sub.add_argument("--processes", type=int, default=2)
            sub.add_argument("--wall-seconds", type=int, default=1800)
            sub.add_argument("--max-output-bytes", type=int, default=2 * 1024 ** 3)
            sub.add_argument("--confirm-commit", action="store_true")
        elif command == "next":
            sub.add_argument("--approval-id")

    args = parser.parse_args()

    try:
        if args.command == "create":
            result = create(args.name, args.proposal_run, WorkerLimits(args.wall_seconds, args.memory_mib, args.processes),
                            args.max_output_bytes, confirmed=args.confirm_commit)
        else:
            tool = open_run(args.name)

            if args.command == "next":
                result = {**asdict(executor(tool).run_next(approval_id=args.approval_id)), "worker_started": tool.worker_started}
            elif args.command == "approve":
                result = approve(tool)
            elif args.command == "cancel":
                result = cancel(tool)
            else:
                result = status(tool)

        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception:
        print(json.dumps({"status": "failed", "code": "WM_SPLIT_COMMIT_REQUEST_FAILED",
                          "message": "품질 검사 기록·입력·작업 이름·승인·자원 상한을 확인하세요."}, ensure_ascii=False))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
