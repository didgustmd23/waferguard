"""승인된 WM 품질 검사에 연결한 충돌 제외 후보 CLI. 실제 제외는 수행하지 않는다."""

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
from .wm_conflict import WmConflictTool
from .wm_conflict_evidence import read_quality_evidence
from .wm_conflict_results import POLICY_VERSION
from .worker_contract import WorkerLimits, _decode


def _directory(name, *, create_parent=False):
    """새 후보 작업만 별도 이름 공간에 저장하고 기존 품질 기록은 보존한다."""
    if type(name) is not str or re.fullmatch(r"[a-z][a-z0-9_-]{0,47}", name) is None:
        raise ValueError("작업 이름은 소문자로 시작하는 영문·숫자·밑줄·하이픈 1~48자여야 합니다.")

    directory = Path(__file__).resolve().parents[2]

    for part in ("logs", "agents", "wm_conflict", "runs"):
        directory /= part

        if create_parent:
            directory.mkdir(exist_ok=True)

        _root(directory)

    return directory / name


def _preflight(output, arguments):
    """입력 배열을 읽지 않고 지원 실행 환경만 확인한다."""
    from .wm_runtime import runtime_status

    if arguments or not runtime_status("wm_conflict_candidate")["supported"]:
        return ToolResult("blocked_input")

    (output / "preflight.json").write_text(json.dumps({"version": POLICY_VERSION}), encoding="utf-8")
    return ToolResult("passed")


def _validate_preflight(output, arguments):
    return not arguments and _decode(_read_file(output / "preflight.json", 4096).decode()) == {"version": POLICY_VERSION}


def executor(tool):
    return RulesExecutor(tool.states, ApprovalStore(tool.states.directory / "approvals.sqlite3"), tool.plan, (
        RegisteredTool("run_preflight", _preflight, _validate_preflight), tool.registered_tool(),
    ))


def create(name, quality_run, limits, max_output_bytes=64 * 1024 * 1024, *, confirmed=False):
    """이전 결과의 증거를 고정한다. 파일 복사·진단은 새 승인을 받은 뒤 수행한다."""
    if not confirmed:
        raise ValueError("실제 제외가 아닌 후보 진단임을 명시적으로 확인해야 합니다.")

    previous, evidence = read_quality_evidence(quality_run)
    contract = replace(previous, limits=limits, max_output_bytes=max_output_bytes, include_examples=False)
    directory = _directory(name, create_parent=True)

    if directory.exists() or directory.is_symlink():
        raise ValueError("기존 후보 작업은 덮어쓰지 않습니다. 새 작업 이름을 사용하세요.")

    directory.mkdir()
    states = LocalStateStore(directory)
    tool = WmConflictTool(states, contract, quality_run, evidence)
    tool._check_arguments(tool.arguments)
    state = states.create(_current_actor(), policy_version=POLICY_VERSION)
    record = {"contract": contract.to_dict(), "quality_run": quality_run, "evidence": evidence,
              "code_id": tool.code_id, "plan_id": tool.plan.plan_id}
    payload = json.dumps(record, ensure_ascii=True).encode("utf-8")

    if len(payload) > 1024 * 1024:
        raise ValueError("후보 선택 기록이 크기 상한을 초과했습니다.")

    (directory / "selection.json").write_bytes(payload)
    ApprovalStore(directory / "approvals.sqlite3")
    append_event(directory / "events.jsonl", "run_created", "RUN_CREATED",
                 stage=state.current_stage, revision=state.revision)
    return {"status": "created", "run_dir": str(directory), "worker_started": False,
            "notice": "충돌 제외의 가상 영향만 진단합니다. 실제 제외·분할·EDA·학습은 하지 않습니다."}


def open_run(name):
    """같은 사용자·입력·품질 증거·코드·계획으로만 재개한다."""
    directory = _root(_directory(name))
    record = decode_json(_read_file(directory / "selection.json", 1024 * 1024))

    if type(record) is not dict or set(record) != {"contract", "quality_run", "evidence", "code_id", "plan_id"}:
        raise ValueError("충돌 후보의 선택 기록이 올바르지 않습니다.")

    states = LocalStateStore(directory)

    if states.load().actor_id != _current_actor():
        raise ValueError("작업을 생성한 OS 계정에서 실행하세요.")

    tool = WmConflictTool(states, WmCanonicalInput.from_dict(record["contract"]), record["quality_run"], record["evidence"])

    if tool.code_id != record["code_id"] or tool.plan.plan_id != record["plan_id"]:
        raise ValueError("입력·코드·계획이 변경되었습니다. 새 작업과 승인이 필요합니다.")

    tool._check_arguments(tool.arguments)
    return tool


def approval_summary(tool):
    """현재 단계의 로컬 승인 설명. 표시와 실제 승인 저장을 분리한다."""
    return {
        "source_id": tool.contract.source_id, "quality_evidence_id": digest(tool.evidence),
        "limits": asdict(tool.limits), "max_output_bytes": tool.contract.max_output_bytes,
        "lifetime_minutes": 30,
        "effect": "AppContainer+Job에서 manifest 기반 가상 제외 후보 보고서 생성",
        "notice": "NPZ 배열 재파싱·실제 제외·라벨 변경·분할·EDA·학습·LLM 전송 없음"
    }


def preview_approval(tool):
    """현재 범위의 데이터 전용 승인 미리보기를 반환한다."""
    from .workflow_approval import preview_approval as build_preview

    return build_preview(tool, executor(tool), approval_summary)


def record_approval(tool, preview, *, confirmed=False):
    """CLI·공통 서비스가 같은 잠금·범위 재검사·저장 함수를 사용한다."""
    from .workflow_approval import record_approval as save_approval

    return save_approval(tool, executor(tool), approval_summary, preview, confirmed=confirmed)


def approve(tool):
    """기존 터미널 확인 문구를 유지하고 공통 승인 서비스에 저장을 위임한다."""
    if not os.isatty(0):
        raise ValueError("승인은 대화형 터미널에서 직접 진행하세요.")

    preview = preview_approval(tool)
    token = "CONFLICT-" + secrets.token_hex(4).upper()
    print(json.dumps(preview.to_dict()["summary"], ensure_ascii=False, indent=2))

    if input(f"후보 진단을 승인하려면 {token} 을 입력하세요: ") != token:
        return {"code": "APPROVAL_DECLINED"}

    return record_approval(tool, preview, confirmed=True)


def status(tool):
    state = tool.states.load()
    result = {"status": state.status, "stage": state.current_stage, "cancel_requested": tool.cancel.is_set()}

    if state.status == "completed":
        scope = executor(tool).scope_for("derive_preparation")
        attempt = read_committed_artifact(tool.states.directory, scope).directory
        tool.inspect_attempt(attempt)
        result.update(candidate_status="candidate_ready", data_modified=False, exclusion_approved=False,
                      local_report=str(attempt / "candidate/diagnosis/diagnosis.md"))

    return result


def main():
    parser = argparse.ArgumentParser(description="WM 충돌 제외 후보의 승인 기반 격리 진단")
    commands = parser.add_subparsers(dest="command", required=True)

    for command in ("create", "next", "approve", "status", "cancel"):
        sub = commands.add_parser(command)
        sub.add_argument("--name", required=True)

        if command == "create":
            sub.add_argument("--quality-run", required=True)
            sub.add_argument("--memory-mib", type=int, default=8192)
            sub.add_argument("--processes", type=int, default=2)
            sub.add_argument("--wall-seconds", type=int, default=1800)
            sub.add_argument("--max-output-bytes", type=int, default=64 * 1024 * 1024)
            sub.add_argument("--confirm-candidate-only", action="store_true")
        elif command == "next":
            sub.add_argument("--approval-id")

    args = parser.parse_args()

    try:
        if args.command == "create":
            result = create(args.name, args.quality_run, WorkerLimits(args.wall_seconds, args.memory_mib, args.processes),
                            args.max_output_bytes, confirmed=args.confirm_candidate_only)
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
        print(json.dumps({"status": "failed", "code": "WM_CONFLICT_CANDIDATE_REQUEST_FAILED",
                          "message": "품질 검사 기록·입력·작업 이름·승인·자원 상한을 확인하세요."}, ensure_ascii=False))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
