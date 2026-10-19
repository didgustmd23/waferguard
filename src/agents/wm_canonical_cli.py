"""기존 WM canonical의 승인 기반 품질·EDA 실행 CLI. pickle을 다시 읽지 않는다."""

import argparse
from dataclasses import asdict
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
from .wm_canonical import WmCanonicalTool
from .wm_canonical_contract import POLICY_VERSION, WmCanonicalInput, copy_snapshot
from .worker_contract import WorkerLimits, _decode
from .worker_paths import _path


def _directory(name, *, create_parent=False):
    """고정 로그 루트 아래 새 이름만 허용한다. 기존 작업을 덮어쓰지 않는다."""
    if type(name) is not str or re.fullmatch(r"[a-z][a-z0-9_-]{0,47}", name) is None:
        raise ValueError("작업 이름은 소문자로 시작하는 영문·숫자·밑줄·하이픈 1~48자여야 합니다.")

    root = Path(__file__).resolve().parents[2]

    for name_part in ("logs", "agents", "wm_canonical", "runs"):
        root /= name_part

        if create_parent:
            root.mkdir(exist_ok=True)

        _root(root)

    return root / name


def _preflight(output, arguments):
    """실데이터를 읽지 않고 고정 격리 런타임의 운영체제·Python 버전을 확인한다."""
    from .wm_runtime import runtime_status

    if arguments or not runtime_status("wm_canonical_eda")["supported"]:
        return ToolResult("blocked_input")

    (output / "preflight.json").write_text(json.dumps({"version": POLICY_VERSION}), encoding="utf-8")
    return ToolResult("passed")


def _validate_preflight(output, arguments):
    return not arguments and _decode(_read_file(output / "preflight.json", 4096).decode()) == {"version": POLICY_VERSION}


def executor(tool):
    return RulesExecutor(tool.states, ApprovalStore(tool.states.directory / "approvals.sqlite3"), tool.plan, (
        RegisteredTool("run_preflight", _preflight, _validate_preflight), tool.registered_tool(),
    ))


def create(name, source, limits, max_input_bytes, max_output_bytes, *, confirmed=False,
           include_examples=False, max_group_ratio=0.2):
    """선택한 묶음의 바이트만 복사한다. 배열 파싱·품질 검사·EDA는 승인 후에 수행한다."""
    if not confirmed:
        raise ValueError("canonical 입력 선택을 명시적으로 확인해야 합니다.")

    # 설정 오류는 큰 복사를 시작하기 전에 검출한다.
    empty = {key: {"size": 0, "sha256": "0" * 64} for key in
             ("manifest.jsonl", "conversion.json", "provenance.json", "SUCCESS.json")}
    source = _path(Path(source).absolute(), directory=True)
    WmCanonicalInput(str(source), json.dumps(empty), limits, max_input_bytes, max_output_bytes,
                     include_examples, max_group_ratio)
    directory = _directory(name, create_parent=True)

    if directory.exists() or directory.is_symlink():
        raise ValueError("기존 WM 분석 기록은 덮어쓰지 않습니다. 새 이름을 사용하세요.")

    directory.mkdir()
    files = copy_snapshot(source, directory / "snapshot", max_input_bytes)
    contract = WmCanonicalInput(str(_root(directory / "snapshot")), json.dumps(files, sort_keys=True), limits,
                                max_input_bytes, max_output_bytes, include_examples, max_group_ratio)
    states = LocalStateStore(directory)
    tool = WmCanonicalTool(states, contract)
    state = states.create(_current_actor(), policy_version=POLICY_VERSION)
    record = {"contract": contract.to_dict(), "code_id": tool.code_id, "plan_id": tool.plan.plan_id}
    payload = json.dumps(record, ensure_ascii=True).encode("utf-8")

    if len(payload) > 1024 * 1024:
        raise ValueError("WM 실행 선택 기록이 크기 상한을 초과했습니다.")

    (directory / "selection.json").write_bytes(payload)
    ApprovalStore(directory / "approvals.sqlite3")
    append_event(directory / "events.jsonl", "run_created", "RUN_CREATED", stage=state.current_stage, revision=state.revision)
    return {"status": "created", "run_dir": str(directory), "worker_started": False,
            "notice": "canonical 바이트 복사본을 고정했습니다. 품질·EDA는 새 사용자 승인 후 격리 실행합니다."}


def create_demo(name, limits, *, review=False):
    """실데이터 대신 독립 Lot·맵을 가진 고정 합성 canonical로 연결을 검사한다."""
    import numpy as np
    import pandas as pd
    from scripts.wm_sandbox.export_canonical import CLASS_NAMES, export_bundle

    base = _directory(name, create_parent=True).parent.parent / "examples"
    base.mkdir(exist_ok=True)
    _root(base)
    example = base / name
    example.mkdir(exist_ok=False)
    rows = []

    for class_index, label in enumerate(CLASS_NAMES):
        for repeat in range(1 if review else 3):
            index = class_index * 3 + repeat + 1
            array = np.ones((8, 8), dtype=np.uint8)

            for bit in range(6):
                array.flat[bit] = 2 if index & (1 << bit) else 1

            rows.append({"waferMap": array, "failureType": label, "lotName": f"fixture_lot_{index}"})

    export_bundle(pd.DataFrame(rows), example / "canonical", "a" * 64, 100,
                  execution_mode="synthetic_demo", max_output_bytes=1024 * 1024)
    result = create(name, example / "canonical", limits, 1024 * 1024, 32 * 1024 * 1024, confirmed=True)
    return {**result, "synthetic_example": True, "review_example": review}


def open_run(name):
    """같은 OS 사용자·코드·복사본·계획에 한해서만 재개한다."""
    directory = _root(_directory(name))
    record = decode_json(_read_file(directory / "selection.json", 1024 * 1024))

    if type(record) is not dict or set(record) != {"contract", "code_id", "plan_id"}:
        raise ValueError("WM 분석 선택 기록이 올바르지 않습니다.")

    states = LocalStateStore(directory)

    if states.load().actor_id != _current_actor():
        raise ValueError("작업을 생성한 OS 계정에서 실행하세요.")

    tool = WmCanonicalTool(states, WmCanonicalInput.from_dict(record["contract"]))

    if tool.code_id != record["code_id"] or tool.plan.plan_id != record["plan_id"]:
        raise ValueError("WM 입력·코드·설정이 변경되었습니다. 새 실행과 승인이 필요합니다.")

    return tool


def approval_summary(tool):
    """현재 단계의 로컬 승인 설명. 표시와 실제 승인 저장을 분리한다."""
    return {
        "source_id": tool.contract.source_id,
        "file_count": len(tool.contract.files),
        "snapshot_bytes": sum(value["size"] for value in tool.contract.files.values()),
        **tool.arguments, "lifetime_minutes": 30,
        "effect": "기존 AppContainer+Job에서 WM 검증 후 통과한 경우만 EDA 실행",
        "notice": "품질 검토 필요는 자동 해제하지 않습니다. 원본 변경·분할·캐시·학습·LLM 전송 없음."
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
    token = "WM-" + secrets.token_hex(4).upper()
    print(json.dumps(preview.to_dict()["summary"], ensure_ascii=False, indent=2))

    if input(f"승인하려면 {token} 을 입력하세요: ") != token:
        return {"code": "APPROVAL_DECLINED"}

    return record_approval(tool, preview, confirmed=True)


def status(tool):
    state = tool.states.load()
    result = {"status": state.status, "stage": state.current_stage, "cancel_requested": tool.cancel.is_set()}
    manager = executor(tool)
    attempt = None

    if state.status == "completed":
        attempt = read_committed_artifact(tool.states.directory, manager.scope_for("run_eda")).directory
    elif state.status == "blocked_input":
        task = manager.approvals.get_task(manager.scope_for("run_eda"))

        if task is not None:
            candidate = tool.states.directory / "attempts" / task["task_id"]

            if (candidate / "receipt.json").is_file():
                attempt = candidate

    if attempt is not None:
        report = tool.inspect_attempt(attempt)
        result.update(quality_status=report["quality_status"], eda_generated=report["eda_generated"],
                      training_ready=False, local_quality_report=str(attempt / "analysis/quality/validation.md"))

        if report["eda_generated"]:
            result["local_eda_report"] = str(attempt / "analysis/eda/eda.md")

    return result


def main():
    parser = argparse.ArgumentParser(description="승인 기반 WM canonical 품질·EDA 격리 실행")
    commands = parser.add_subparsers(dest="command", required=True)

    for command in ("create", "create-demo", "next", "approve", "status", "cancel"):
        sub = commands.add_parser(command)
        sub.add_argument("--name", required=True)

        if command in {"create", "create-demo"}:
            sub.add_argument("--memory-mib", type=int, default=2048)
            sub.add_argument("--processes", type=int, default=2)
            sub.add_argument("--wall-seconds", type=int, default=300)

        if command == "create":
            sub.add_argument("--canonical-dir", required=True)
            sub.add_argument("--max-input-bytes", type=int, required=True)
            sub.add_argument("--max-output-bytes", type=int, default=64 * 1024 * 1024)
            sub.add_argument("--include-examples", action="store_true")
            sub.add_argument("--max-group-ratio", type=float, default=0.2)
            sub.add_argument("--confirm-selection", action="store_true")
        elif command == "create-demo":
            sub.add_argument("--review-demo", action="store_true")
        elif command == "next":
            sub.add_argument("--approval-id")

    args = parser.parse_args()

    try:
        if args.command == "create-demo":
            result = create_demo(args.name, WorkerLimits(args.wall_seconds, args.memory_mib, args.processes), review=args.review_demo)
        elif args.command == "create":
            result = create(args.name, args.canonical_dir, WorkerLimits(args.wall_seconds, args.memory_mib, args.processes),
                            args.max_input_bytes, args.max_output_bytes, confirmed=args.confirm_selection,
                            include_examples=args.include_examples, max_group_ratio=args.max_group_ratio)
        else:
            tool = open_run(args.name)

            if args.command == "next":
                tool._check_arguments(tool.arguments)
                result = {**asdict(executor(tool).run_next(approval_id=args.approval_id)), "worker_started": tool.worker_started}
            elif args.command == "approve":
                result = approve(tool)
            elif args.command == "cancel":
                result = cancel(tool)
            else:
                result = status(tool)

        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception:
        print(json.dumps({"status": "failed", "code": "WM_CANONICAL_REQUEST_FAILED",
                          "message": "입력·새 작업 이름·승인·자원 상한·로컬 실행 기록을 확인하세요."}, ensure_ascii=False))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
