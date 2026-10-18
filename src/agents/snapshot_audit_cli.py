"""snapshot CSV 품질 검사의 명시적 선택·승인·격리 실행 CLI. LLM 연결은 없다."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import secrets
import sys

from src.preprocessing.contracts import ColumnRoles
from src.preprocessing.profiles import DatasetProfile, profile_from_dict
from .approval_store import ApprovalStore
from .artifacts import _root
from .audit import append_event
from .cli import _current_actor
from .contracts import ToolResult
from .eda_input_contract import EdaInputContract, bind_eda_input
from .executor import RegisteredTool, RulesExecutor, _run_lock
from .input_snapshot import SNAPSHOT_VERSION, Snapshot, create_snapshot
from .isolated_eda import _read_file
from .isolated_eda_cli import cancel
from .snapshot_audit import POLICY_VERSION, SnapshotAuditTool
from .quality_result_validation import read_quality_report
from .state import LocalStateStore
from .synthetic_eda_data import CSV_BYTES
from .worker_contract import WorkerLimits, _decode
from .worker_paths import _path


def _directory(name, *, create_parent=False):
    """고정 실행 영역의 단일 이름만 받는다. 사용자 입력으로 쓰기 루트를 넓히지 않는다."""
    if not re.fullmatch(r"[a-z][a-z0-9_-]{0,47}", name):
        raise ValueError("실행 이름은 소문자로 시작하는 영문·숫자·밑줄·하이픈 1~48자여야 합니다.")

    root = Path(__file__).resolve().parents[2]

    for part in ("logs", "agents", "snapshot_audit", "runs"):
        root = root / part

        if create_parent:
            root.mkdir(exist_ok=True)

        _root(root)

    return root / name


def _preflight(output, arguments):
    """격리 worker 없이 Windows·Python 호환성만 확인한다."""
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


def create(name, snapshot_directory, profile_path, limits, *, confirmed=False):
    """사용자가 명시적으로 선택한 현재 snapshot을 새 계약의 기준으로 검증한다."""
    if not confirmed:
        raise ValueError("snapshot 선택을 명시적으로 확인해야 합니다.")

    selected = _path(Path(snapshot_directory).absolute(), directory=True)
    metadata = _decode(_read_file(selected / "snapshot.json", 4096).decode("utf-8"))

    if set(metadata) != {"version", "size_bytes", "sha256", "max_input_bytes"} or metadata["version"] != SNAPSHOT_VERSION:
        raise ValueError("형식 독립 snapshot 계약이 필요합니다.")

    expected = Snapshot(**{key: metadata[key] for key in Snapshot.__dataclass_fields__})
    profile_file = _path(Path(profile_path).absolute(), directory=False)
    profile = profile_from_dict(_decode(_read_file(profile_file, 65536).decode("utf-8")))
    contract = bind_eda_input(selected, expected, profile, limits)
    directory = _directory(name, create_parent=True)

    if directory.exists() or directory.is_symlink():
        raise ValueError("기존 실행을 덮어쓰지 않습니다. 새 이름을 지정하세요.")

    states = LocalStateStore(directory)
    state = states.create(_current_actor(), policy_version=POLICY_VERSION)
    tool = SnapshotAuditTool(states, contract)
    record = {"contract": contract.to_dict(), "code_id": tool.code_id, "plan_id": tool.plan.plan_id}
    (directory / "selection.json").write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    ApprovalStore(directory / "approvals.sqlite3")
    append_event(directory / "events.jsonl", "run_created", "RUN_CREATED", stage=state.current_stage, revision=state.revision)
    return {"status": "created", "run_dir": str(directory), "adapter_id": profile.adapter_id, "worker_started": False}


def open_run(name):
    """저장된 선택·계정·코드가 바뀐 실행을 기존 승인으로 재개하지 않는다."""
    directory = _root(_directory(name))
    record = _decode(_read_file(directory / "selection.json", 131072).decode("utf-8"))

    if set(record) != {"contract", "code_id", "plan_id"}:
        raise ValueError("실행 선택 계약이 올바르지 않습니다.")

    states = LocalStateStore(directory)

    if states.load().actor_id != _current_actor():
        raise ValueError("실행을 생성한 OS 계정에서만 접근할 수 있습니다.")

    contract = EdaInputContract.from_json(json.dumps(record["contract"], ensure_ascii=False))
    tool = SnapshotAuditTool(states, contract)

    if record["code_id"] != tool.code_id or record["plan_id"] != tool.plan.plan_id:
        raise ValueError("코드·계약이 변경됐습니다. 새 실행과 승인이 필요합니다.")

    return tool


def create_demo(name, limits):
    """실데이터 없이 새 snapshot 경로를 검증할 예제를 준비한다. 승인은 발급하지 않는다."""
    base = _directory(name, create_parent=True).parent.parent
    examples = base / "examples"
    examples.mkdir(exist_ok=True)
    _root(examples)
    example = examples / name
    example.mkdir(exist_ok=False)
    (example / "source.csv").write_bytes(CSV_BYTES)
    profile = DatasetProfile(
        "audit_demo", "tabular_csv_v1", "eda_review_v1",
        ColumnRoles(numeric=("x", "y"), categorical=("category",), target="target"),
        max_plot_features=3,
    )
    profile_path = example / "profile.json"
    profile_path.write_text(json.dumps(profile.to_dict()), encoding="utf-8")
    create_snapshot(example / "source.csv", example / "snapshot", max_input_bytes=1024 * 1024)
    result = create(name, example / "snapshot", profile_path, limits, confirmed=True)
    result["synthetic_example"] = True
    return result


def approve(tool):
    """로컬 입력 역할·권한·한도를 사람이 검토한 뒤 일회성 승인을 발급한다."""
    if not os.isatty(0):
        raise ValueError("승인은 대화형 관리자 터미널에서 진행하세요.")

    manager = executor(tool)
    state = tool.states.load()

    if state.status != "awaiting_approval" or tool.cancel.is_set():
        raise ValueError("취소되지 않은 승인 대기 작업이 필요합니다.")

    tool._check_arguments(tool.arguments)
    scope = manager.scope_for("audit")
    token = "AUDIT-" + secrets.token_hex(4).upper()
    print(json.dumps({
        "adapter_id": tool.contract.profile.adapter_id,
        "roles": tool.contract.profile.to_dict()["roles"],
        "snapshot_bytes": tool.contract.snapshot.size_bytes,
        "max_input_bytes": tool.contract.snapshot.max_input_bytes,
        "quality_rules": asdict(tool.contract.profile.quality_rules),
        "limits": asdict(tool.limits), "lifetime_minutes": 30,
        "effect": "검증한 복사본·프로파일을 새 AppContainer+Job에 전달. 원본 폴더·승인 DB 권한 추가 없음",
        "notice": "비파괴 품질 검사만 수행합니다. 검토 필요·차단 결과는 자동 해제하거나 다음 단계로 진행하지 않습니다.",
    }, ensure_ascii=False, indent=2))

    if input(f"승인하려면 {token} 를 입력하세요: ") != token:
        return {"code": "APPROVAL_DECLINED"}

    with _run_lock(tool.states.directory) as acquired:
        if not acquired or tool.states.load() != state or tool.cancel.is_set():
            raise ValueError("승인 중 실행 상태가 변경됐습니다.")

        tool._check_arguments(tool.arguments)

        if manager.scope_for("audit") != scope:
            raise ValueError("승인 범위가 변경됐습니다.")

        append_event(tool.states.directory / "events.jsonl", "approval_requested", "LOCAL_REVIEW_CONFIRMED", stage=state.current_stage, revision=state.revision)
        record = manager.approvals.record_confirmed_approval(scope)

    return {"approval_id": record.approval_id, "expires_at": record.expires_at.isoformat()}


def main():
    parser = argparse.ArgumentParser(description="snapshot CSV 품질 검사의 사용자 승인 기반 격리 실행")
    commands = parser.add_subparsers(dest="command", required=True)

    for command in ("create-demo", "create", "next", "approve", "status", "cancel"):
        sub = commands.add_parser(command)
        sub.add_argument("--name", required=True)

        if command in ("create", "create-demo"):
            if command == "create":
                sub.add_argument("--snapshot-dir", required=True)
                sub.add_argument("--profile", required=True)
                sub.add_argument("--confirm-snapshot-selection", action="store_true")
            sub.add_argument("--wall-seconds", type=int, default=300)
            sub.add_argument("--memory-mib", type=int, default=2048)
            sub.add_argument("--processes", type=int, default=1)
        elif command == "next":
            sub.add_argument("--approval-id")

    args = parser.parse_args()

    try:
        if args.command == "create-demo":
            result = create_demo(args.name, WorkerLimits(args.wall_seconds, args.memory_mib, args.processes))
        elif args.command == "create":
            result = create(args.name, args.snapshot_dir, args.profile, WorkerLimits(args.wall_seconds, args.memory_mib, args.processes), confirmed=args.confirm_snapshot_selection)
        else:
            tool = open_run(args.name)

            if args.command == "next":
                # 승인 전 변조를 예약·실행 전에 차단한다.
                tool._check_arguments(tool.arguments)
                result = asdict(executor(tool).run_next(approval_id=args.approval_id))
            elif args.command == "approve":
                result = approve(tool)
            elif args.command == "cancel":
                result = cancel(tool)
            else:
                state = tool.states.load()
                result = {"status": state.status, "stage": state.current_stage, "cancel_requested": tool.cancel.is_set()}
                manager = executor(tool)
                task = manager.approvals.get_task(manager.scope_for("audit"))

                if task is not None:
                    directory = manager._output(task) / "quality"

                    if directory.is_dir():
                        report = read_quality_report(directory, tool.contract.profile)
                        result["quality_status"] = report["status"] if report is not None else "invalid_result"
                        result["local_quality_report"] = str(directory / "quality.md")

        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception:
        print("snapshot 품질 검사 명령을 완료하지 못했습니다.")
        print("관리자 권한·새 이름·입력 정합성·등록 프로파일·승인 상태를 확인하세요. 비격리 재실행은 하지 않습니다.")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
