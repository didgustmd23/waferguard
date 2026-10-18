"""완료된 WM 변환 검사를 바탕으로 Train·Validation 캐시 생성을 승인하는 CLI."""

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
from .wm_cache_evidence import committed_cache
from .state import LocalStateStore
from .wm_canonical_contract import WmCanonicalInput, digest
from .wm_cache import WmCacheTool
from .wm_cache_evidence import read_transform_evidence
from .wm_cache_results import POLICY_VERSION
from .worker_contract import WorkerLimits, _decode


def _directory(name, *, create_parent=False):
    """새 적용 작업만 별도 이름 공간에 저장하고 기존 품질·후보 기록은 보존한다."""
    if type(name) is not str or re.fullmatch(r"[a-z][a-z0-9_-]{0,47}", name) is None:
        raise ValueError("작업 이름은 소문자로 시작하는 영문·숫자·밑줄·하이픈 1~48자여야 합니다.")

    directory = Path(__file__).resolve().parents[2]

    for part in ("logs", "agents", "wm_cache", "runs"):
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


def create(name, check_run, limits, max_output_bytes=8 * 1024 ** 3, *, confirmed=False,
           config_path="configs/wm811k/cache.json"):
    """완료된 분할·변환과 캐시 설정을 고정한다. 실제 생성은 새 승인 뒤 수행한다."""
    if not confirmed:
        raise ValueError("Train·Validation만 새 캐시로 생성함을 확인해야 합니다.")

    from src.wafer_dl.cache import load_cache_config
    from src.wafer_dl.isolated_cache import estimate_cache_bytes

    config = load_cache_config(config_path)
    previous, evidence = read_transform_evidence(check_run, config)
    split = evidence["split_evidence"]
    estimated = estimate_cache_bytes(split["split_report"], split["transform_config"]) + 1024 * 1024
    pixels = split["transform_config"]["target_size"][0] * split["transform_config"]["target_size"][1]
    capacity = min(config["shard_max_rows"], config["buffer_max_bytes"] // pixels)

    if capacity < 1 or any((split["split_report"]["split_counts"][name] + capacity - 1) // capacity > 2048
                           for name in ("train", "validation")):
        raise ValueError("캐시 shard 수가 구간별 2048개 상한을 초과합니다. 버퍼·shard 행 수를 검토하세요.")

    if estimated > max_output_bytes:
        raise ValueError("출력 상한이 보수적 캐시 예상 크기보다 작습니다. 상한을 늘리거나 새 설정을 검토하세요.")

    contract = replace(previous, limits=limits, max_output_bytes=max_output_bytes, include_examples=False)
    directory = _directory(name, create_parent=True)

    if directory.exists() or directory.is_symlink():
        raise ValueError("기존 적용 작업은 덮어쓰지 않습니다. 새 작업 이름을 사용하세요.")

    directory.mkdir()
    states = LocalStateStore(directory)
    tool = WmCacheTool(states, contract, check_run, evidence)
    tool._check_arguments(tool.arguments)
    state = states.create(_current_actor(), policy_version=POLICY_VERSION)
    record = {"contract": contract.to_dict(), "check_run": check_run, "evidence": evidence,
              "code_id": tool.code_id, "plan_id": tool.plan.plan_id}
    payload = json.dumps(record, ensure_ascii=True).encode("utf-8")

    if len(payload) > 1024 * 1024:
        raise ValueError("적용 선택 기록이 크기 상한을 초과했습니다.")

    (directory / "selection.json").write_bytes(payload)
    ApprovalStore(directory / "approvals.sqlite3")
    append_event(directory / "events.jsonl", "run_created", "RUN_CREATED",
                 stage=state.current_stage, revision=state.revision)
    return {"status": "created", "run_dir": str(directory), "worker_started": False,
            "notice": "Train·Validation 캐시만 준비합니다. 그림 확인·기존 분할 재확인·캐시 실행 승인은 별개입니다."}


def open_run(name):
    """같은 사용자·입력·품질 증거·코드·계획으로만 재개한다."""
    directory = _root(_directory(name))
    record = decode_json(_read_file(directory / "selection.json", 1024 * 1024))

    if type(record) is not dict or set(record) != {"contract", "check_run", "evidence", "code_id", "plan_id"}:
        raise ValueError("WM 개발 구간 캐시 생성의 선택 기록이 올바르지 않습니다.")

    states = LocalStateStore(directory)

    if states.load().actor_id != _current_actor():
        raise ValueError("작업을 생성한 OS 계정에서 실행하세요.")

    tool = WmCacheTool(states, WmCanonicalInput.from_dict(record["contract"]), record["check_run"], record["evidence"])

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
    scope = manager.scope_for(state.current_stage)
    token = "WM-CACHE-" + secrets.token_hex(4).upper()
    print(json.dumps({"source_id": tool.contract.source_id, "split_evidence_id": digest(tool.evidence),
                      "stage": state.current_stage,
                      "transform_config": tool.evidence["split_evidence"]["transform_config"],
                      "cache_config": tool.evidence["cache_config"],
                      "local_examples": str(Path(tool.evidence["check_directory"]) / "examples.png"),
                      "local_check_report": str(Path(tool.evidence["check_directory"]) / "transform_check.md"),
                      "limits": asdict(tool.limits), "max_output_bytes": tool.contract.max_output_bytes,
                      "lifetime_minutes": 30,
                      "effect": ("그림의 형태 보존을 사용자가 확인" if state.current_stage == "review_eda" else
                                 "기존 분할 완료 증거만 재확인, 새 분할 없음" if state.current_stage == "commit_split" else
                                 "AppContainer+Job에서 Train·Validation uint8 캐시 생성"),
                      "notice": "그림 확인 승인은 그림을 직접 본 후에만 진행하세요. Test 배열 복원·모델 학습·LLM 전송 없음"},
                     ensure_ascii=False, indent=2))

    if input(f"현재 단계를 승인하려면 {token} 을 입력하세요: ") != token:
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
        attempt = committed_cache(tool, executor(tool))
        tool.inspect_attempt(attempt)
        result.update(cache_status="cache_complete", training_ready=False, test_maps_read=False,
                      local_cache_directory=str(attempt / "cache"),
                      local_report=str(attempt / "cache/cache.md"))
    elif state.status == "failed_runtime" and state.current_stage == "prepare_data":
        # 이미 소비한 승인을 재사용하지 않고 해당 시도의 고정 진단만 조회한다.
        from .wm_cache_transport import PREPARATION_CODES, PREPARATION_STAGES

        manager = executor(tool)
        task = manager.approvals.get_task(manager.scope_for("prepare_data"))

        if task is not None:
            request = tool._request(task["task_id"])
            path = tool._work_root(task["task_id"]) / "transport.json"

            if path.is_file():
                outcome = _decode(_read_file(path, 65536).decode("utf-8"))
                failure = outcome.get("preparation_failure")

                if (outcome.get("request_id") == request.request_id and type(failure) is dict
                    and set(failure) == {"stage", "code", "os_error"}
                    and failure["stage"] in PREPARATION_STAGES and failure["code"] in PREPARATION_CODES
                    and (failure["os_error"] is None or
                         type(failure["os_error"]) is int and 0 <= failure["os_error"] <= 2 ** 32 - 1)):
                    result.update(preparation_failure=failure,
                                  worker_started=outcome.get("launch", {}).get("status") in
                                  {"exited", "timed_out", "resource_exceeded", "cancelled"})

    return result


def main():
    parser = argparse.ArgumentParser(description="WM 개발 구간 캐시의 시각 확인·승인 기반 격리 실행")
    commands = parser.add_subparsers(dest="command", required=True)

    for command in ("create", "next", "approve", "status", "cancel"):
        sub = commands.add_parser(command)
        sub.add_argument("--name", required=True)

        if command == "create":
            sub.add_argument("--check-run", required=True)
            sub.add_argument("--config", default="configs/wm811k/cache.json")
            sub.add_argument("--memory-mib", type=int, default=8192)
            sub.add_argument("--processes", type=int, default=2)
            sub.add_argument("--wall-seconds", type=int, default=1800)
            sub.add_argument("--max-output-bytes", type=int, default=8 * 1024 ** 3)
            sub.add_argument("--confirm-development-cache", action="store_true")
        elif command == "next":
            sub.add_argument("--approval-id")

    args = parser.parse_args()

    try:
        if args.command == "create":
            result = create(args.name, args.check_run, WorkerLimits(args.wall_seconds, args.memory_mib, args.processes),
                            args.max_output_bytes, confirmed=args.confirm_development_cache, config_path=args.config)
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
        print(json.dumps({"status": "failed", "code": "WM_CACHE_REQUEST_FAILED",
                          "message": "품질 검사 기록·입력·작업 이름·승인·자원 상한을 확인하세요."}, ensure_ascii=False))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
