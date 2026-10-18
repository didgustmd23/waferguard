"""구조 판별형 pickle의 선택·승인·AppContainer 변환·상태 조회 CLI."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import secrets
import sys

from .approval_store import ApprovalStore
from .artifacts import _root
from .audit import append_event
from .cli import _current_actor
from .contracts import ToolResult
from .executor import RegisteredTool, RulesExecutor, _run_lock
from .input_snapshot import create_snapshot
from .isolated_eda import _read_file
from .isolated_eda_cli import cancel
from .split_evidence import read_committed_artifact
from .state import LocalStateStore
from .pickle_contract import ADAPTER_ID, POLICY_VERSION, PickleInput, check_options, CONVERTERS
from .pickle_import import PickleImportTool
from .worker_contract import WorkerLimits, _decode
from .worker_paths import _path


def _directory(name, *, create_parent=False):
    """승인 기록은 고정 로그 루트의 새 실행 이름 아래에 저장한다."""
    if type(name) is not str or re.fullmatch(r"[a-z][a-z0-9_-]{0,47}", name) is None:
        raise ValueError("실행 이름은 소문자로 시작하는 영문·숫자·밑줄·하이픈 1~48자여야 합니다.")

    root = Path(__file__).resolve().parents[2]

    for part in ("logs", "agents", "pickle_import", "runs"):
        root = root / part

        if create_parent:
            root.mkdir(exist_ok=True)

        _root(root)

    return root / name


def _preflight(output, arguments):
    """기존 격리 런타임과 같은 Windows·Python 3.10 실행 조건을 확인한다."""
    if arguments or os.name != "nt" or sys.version_info[:2] != (3, 10):
        return ToolResult("blocked_input")

    (output / "preflight.json").write_text(json.dumps({"version": POLICY_VERSION}), encoding="utf-8")

    return ToolResult("passed")


def _validate_preflight(output, arguments):
    return not arguments and _decode(_read_file(output / "preflight.json", 4096).decode()) == {"version": POLICY_VERSION}


def executor(tool):
    """기존 실행기가 승인 1회 소비·작업 예약·취소·검증·커밋을 관리한다."""
    return RulesExecutor(tool.states, ApprovalStore(tool.states.directory / "approvals.sqlite3"), tool.plan, (
        RegisteredTool("run_preflight", _preflight, _validate_preflight), tool.registered_tool(),
    ))


def create(name, source_path, limits, max_input_bytes, max_output_bytes, *, confirmed=False,
           shard_rows=1000, converter_id="auto", select_path=()):
    """명시적으로 선택한 pickle의 바이트만 복사한다. 생성 시 역직렬화하지 않는다."""
    if not confirmed:
        raise ValueError("pickle 파일 선택을 명시적으로 확인해야 합니다.")

    check_options(max_output_bytes, shard_rows)
    source = _path(Path(source_path).absolute(), directory=False)

    if source.suffix.lower() not in {".pkl", ".pickle"}:
        raise ValueError("pickle 원본 파일을 지정하세요.")

    directory = _directory(name, create_parent=True)

    if directory.exists() or directory.is_symlink():
        raise ValueError("기존 변환 기록은 덮어쓰지 않습니다. 새 실행 이름을 사용하세요.")

    directory.mkdir()
    snapshot = create_snapshot(source, directory / "snapshot", max_input_bytes=max_input_bytes)
    contract = PickleInput(str(_root(directory / "snapshot")), snapshot, limits, max_output_bytes,
                           shard_rows, converter_id, tuple(select_path))
    states = LocalStateStore(directory)
    tool = PickleImportTool(states, contract)
    state = states.create(_current_actor(), policy_version=POLICY_VERSION)
    record = {"contract": contract.to_dict(), "code_id": tool.code_id, "plan_id": tool.plan.plan_id}
    (directory / "selection.json").write_text(json.dumps(record), encoding="utf-8")
    ApprovalStore(directory / "approvals.sqlite3")
    append_event(directory / "events.jsonl", "run_created", "RUN_CREATED", stage=state.current_stage, revision=state.revision)

    return {"status": "created", "run_dir": str(directory), "adapter_id": ADAPTER_ID, "worker_started": False}


def create_demo(name, limits, demo_kind="tabular"):
    """고정 합성 표·배열·WM·다중 후보로 동일한 승인·격리 경로를 준비한다."""
    import numpy as np
    import pandas as pd
    from scripts.wm_sandbox.export_canonical import CLASS_NAMES

    base = _directory(name, create_parent=True).parent.parent / "examples"
    base.mkdir(exist_ok=True)
    _root(base)
    example = base / name
    example.mkdir(exist_ok=False)
    rows = []

    for index, label in enumerate(CLASS_NAMES):
        array = np.ones((3, 3), dtype=np.uint8)
        array.flat[:index] = 2
        rows.append({"waferMap": array, "failureType": label, "lotName": f"fixture_lot_{index}"})

    source = example / "synthetic.pkl"
    examples = {
        "tabular": pd.DataFrame({"value": [1.0, float("nan"), 3.0], "category": ["a", "b", "a"]}),
        "array": np.arange(12, dtype=np.float32).reshape(3, 4),
        "wm": pd.DataFrame(rows),
        "ambiguous": {"first": np.arange(3), "second": np.arange(4)},
    }
    pd.to_pickle(examples[demo_kind], source)
    result = create(name, source, limits, 1024 * 1024, 1024 * 1024, confirmed=True)
    result["synthetic_example"] = True

    return result


def open_run(name):
    """재개 전에 동일 계정·복사본·코드·계획을 확인한다."""
    directory = _root(_directory(name))
    record = _decode(_read_file(directory / "selection.json", 65536).decode("utf-8"))

    if set(record) != {"contract", "code_id", "plan_id"}:
        raise ValueError("pickle 변환 선택 기록이 올바르지 않습니다.")

    states = LocalStateStore(directory)

    if states.load().actor_id != _current_actor():
        raise ValueError("작업을 생성한 OS 계정에서 실행하세요.")

    tool = PickleImportTool(states, PickleInput.from_dict(record["contract"]))

    if record["code_id"] != tool.code_id or record["plan_id"] != tool.plan.plan_id:
        raise ValueError("변환 코드·입력·계획이 변경되었습니다. 새 실행과 승인이 필요합니다.")

    return tool


def approve(tool):
    """파일 해시·출력 상한·자원 한도에 결합된 단일 변환 승인을 발급한다."""
    if not os.isatty(0):
        raise ValueError("승인은 대화형 터미널에서 직접 진행하세요.")

    manager = executor(tool)
    state = tool.states.load()

    if state.status != "awaiting_approval" or tool.cancel.is_set():
        raise ValueError("취소되지 않은 승인 대기 작업이 필요합니다.")

    tool._check_arguments(tool.arguments)
    scope = manager.scope_for("ingest")
    token = "PICKLE-" + secrets.token_hex(4).upper()
    print(json.dumps({
        "adapter_id": ADAPTER_ID, "source_sha256": tool.contract.snapshot.sha256,
        "snapshot_bytes": tool.contract.snapshot.size_bytes,
        "max_input_bytes": tool.contract.snapshot.max_input_bytes,
        "max_output_bytes": tool.contract.max_output_bytes, "limits": asdict(tool.limits),
        "converter_id": tool.contract.converter_id, "select_path": list(tool.contract.select_path),
        "selection_rule": "지원 객체 한 개만 자동 선택. 모호하거나 미지원 구조이면 검토 중단",
        "lifetime_minutes": 30,
        "effect": "기존 AppContainer+Job에서 구조를 판별하고 선택 객체의 canonical 묶음을 생성·검증",
        "notice": "pickle은 코드를 실행할 수 있습니다. 출처와 해시를 확인하세요. 출력 정합성은 원본 동등성·무해성 인증이 아닙니다.",
    }, ensure_ascii=False, indent=2))

    if input(f"승인하려면 {token} 을 입력하세요: ") != token:
        return {"code": "APPROVAL_DECLINED"}

    with _run_lock(tool.states.directory) as acquired:
        if not acquired or tool.states.load() != state or tool.cancel.is_set():
            raise ValueError("승인 중 실행 상태가 변경되었습니다.")

        tool._check_arguments(tool.arguments)

        if manager.scope_for("ingest") != scope:
            raise ValueError("승인 범위가 변경되었습니다.")

        append_event(tool.states.directory / "events.jsonl", "approval_requested", "LOCAL_REVIEW_CONFIRMED",
                     stage=state.current_stage, revision=state.revision)
        record = manager.approvals.record_confirmed_approval(scope)

    return {"approval_id": record.approval_id, "expires_at": record.expires_at.isoformat()}


def status(tool):
    state = tool.states.load()
    result = {"status": state.status, "stage": state.current_stage, "cancel_requested": tool.cancel.is_set()}

    if state.status == "completed":
        artifact = read_committed_artifact(tool.states.directory, executor(tool).scope_for("ingest"))

        if not tool.validate(artifact.directory, tool.arguments):
            raise ValueError("완료된 pickle 변환 결과 또는 worker 종료 증거가 변경되었습니다.")

        result.update(canonical_integrity="passed", training_ready=False,
                      local_canonical_directory=str(artifact.directory / "canonical"))

    if state.status == "blocked_input":
        # 검토 결과는 완료 산출물이 아니므로 예약된 시도의 로컬 보고서만 안내한다.
        manager = executor(tool)
        task = manager.approvals.get_task(manager.scope_for("ingest"))

        if task is not None:
            review = tool.states.directory / "attempts" / task["task_id"] / "review"

            if review.is_dir():
                from .pickle_results import read_structure_review

                read_structure_review(review, tool._request(task["task_id"]), tool.contract.max_output_bytes)
                result.update(structure_status="review_required", local_structure_report=str(review / "structure.md"))

    return result


def main():
    parser = argparse.ArgumentParser(description="승인 기반 AppContainer 구조 판별형 pickle 변환")
    commands = parser.add_subparsers(dest="command", required=True)

    for command in ("create", "create-demo", "next", "approve", "status", "cancel"):
        sub = commands.add_parser(command)
        sub.add_argument("--name", required=True)

        if command in {"create", "create-demo"}:
            sub.add_argument("--wall-seconds", type=int, default=300)
            sub.add_argument("--memory-mib", type=int, default=2048)
            sub.add_argument("--processes", type=int, default=2)

        if command == "create":
            sub.add_argument("--source", required=True)
            sub.add_argument("--max-input-bytes", type=int, required=True)
            sub.add_argument("--max-output-bytes", type=int, required=True)
            sub.add_argument("--shard-rows", type=int, default=1000)
            sub.add_argument("--converter", choices=sorted(CONVERTERS), default="auto")
            sub.add_argument("--select-path", default="[]", help='dict 키·목록 인덱스의 JSON 배열. 예: ["data", 0]')
            sub.add_argument("--confirm-pickle-selection", action="store_true")
        elif command == "create-demo":
            sub.add_argument("--demo-kind", choices=("tabular", "array", "wm", "ambiguous"), default="tabular")
        elif command == "next":
            sub.add_argument("--approval-id")

    args = parser.parse_args()

    try:
        if args.command == "create-demo":
            result = create_demo(args.name, WorkerLimits(args.wall_seconds, args.memory_mib, args.processes), args.demo_kind)
        elif args.command == "create":
            result = create(
                args.name, args.source, WorkerLimits(args.wall_seconds, args.memory_mib, args.processes),
                args.max_input_bytes, args.max_output_bytes, confirmed=args.confirm_pickle_selection,
                shard_rows=args.shard_rows, converter_id=args.converter, select_path=_parse_path(args.select_path),
            )
        else:
            tool = open_run(args.name)

            if args.command == "next":
                tool._check_arguments(tool.arguments)
                result = asdict(executor(tool).run_next(approval_id=args.approval_id))
                result["worker_started"] = tool.worker_started
            elif args.command == "approve":
                result = approve(tool)
            elif args.command == "cancel":
                result = cancel(tool)
            else:
                result = status(tool)

        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception:
        print(json.dumps({"status": "failed", "code": "PICKLE_IMPORT_REQUEST_FAILED",
                          "message": "입력 계약·새 실행 이름·승인·자원 설정·로컬 진단 기록을 확인하세요."}, ensure_ascii=False))
        raise SystemExit(2)


def _parse_path(text):
    """사용자 경로를 표현식이 아닌 길이 제한 JSON으로 해석한다."""
    if len(text) > 4096:
        raise ValueError("선택 경로가 너무 깁니다.")

    value = _decode(text)

    if type(value) is not list:
        raise ValueError("선택 경로는 JSON 배열이어야 합니다.")

    return tuple(value)


if __name__ == "__main__":
    main()
