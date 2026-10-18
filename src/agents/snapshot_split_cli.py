"""CSV 복사본의 격리 분할 준비 CLI. 기존 ML 분할·모델·Test 파일을 변경하지 않는다."""

import argparse
from dataclasses import asdict, replace
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
from .eda_input_contract import EdaInputContract, bind_eda_input
from .executor import RegisteredTool, RulesExecutor, _run_lock
from .input_snapshot import create_snapshot, InputTooLarge
from .isolated_eda import _read_file
from .isolated_eda_cli import cancel
from .snapshot_split import SnapshotSplitTool, POLICY_VERSION
from .split_evidence import read_confirmed_split
from .state import LocalStateStore
from .worker_contract import WorkerLimits, _decode
from .worker_paths import _path
from src.preprocessing.profiles import profile_from_dict


def _directory(name, *, create_parent=False):
    """사용자는 고정 작업 루트 아래의 단일 실행 이름만 선택한다."""
    if type(name) is not str or re.fullmatch(r"[a-z][a-z0-9_-]{0,47}", name) is None:
        raise ValueError("실행 이름은 소문자로 시작하는 영문·숫자·밑줄·하이픈 1~48자여야 합니다.")

    root = Path(__file__).resolve().parents[2]

    for part in ("logs", "agents", "snapshot_split", "runs"):
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
    """선행 단계 커밋·승인 소비·확정 의존성을 기존 실행기로 강제한다."""
    return RulesExecutor(tool.states, ApprovalStore(tool.states.directory / "approvals.sqlite3"), tool.plan, (
        RegisteredTool("run_preflight", _preflight, _validate_preflight), *tool.registered_tools(),
    ))


def create(name, source_path, profile_path, limits, max_input_bytes, *, confirmed=False, allow_feature_findings=False):
    """명시한 파일의 바이트 복사본을 만든다. 원본 분석·worker 시작·자동 승인은 하지 않는다."""
    if not confirmed:
        raise ValueError("원본 선택과 기능 검증용 새 분할을 명시적으로 확인해야 합니다.")

    profile = profile_from_dict(_decode(_read_file(_path(Path(profile_path).absolute(), directory=False), 65536).decode("utf-8")))

    if profile.adapter_id != "tabular_csv_v1" or profile.workflow_id != "split_review_v1":
        raise ValueError("CSV 분할 검토 프로파일이 필요합니다.")

    if allow_feature_findings:
        # 분할은 센서 값으로 기준을 학습하지 않는다. 사용자가 명시한 경우에만
        # 결측·상수 센서 통지 수준을 바꾸고, 그 정책 자체를 새 프로파일/승인에 결합한다.
        # 라벨·ID·시간·그룹과 구조 오류의 차단 규칙은 변경하지 않는다.
        profile = replace(profile, quality_rules=replace(profile.quality_rules,
                          high_missing="ignore", all_missing_feature="ignore", constant_feature="ignore"))

    source = _path(Path(source_path).absolute(), directory=False)

    if type(max_input_bytes) is not int or max_input_bytes < 1:
        raise ValueError("입력 파일 상한은 양의 정수 바이트여야 합니다.")

    if source.stat().st_size > max_input_bytes:
        raise InputTooLarge(source.stat().st_size, max_input_bytes)

    directory = _directory(name, create_parent=True)

    if directory.exists() or directory.is_symlink():
        raise ValueError("기존 실행은 덮어쓰지 않습니다. 새 이름을 사용하세요.")

    directory.mkdir()
    snapshot = create_snapshot(source, directory / "snapshot", max_input_bytes=max_input_bytes)
    contract = bind_eda_input(directory / "snapshot", snapshot, profile, limits)
    states = LocalStateStore(directory)
    tool = SnapshotSplitTool(states, contract)
    record = {"contract": contract.to_dict(), "code_id": tool.code_id, "plan_id": tool.plan.plan_id}
    payload = json.dumps(record, ensure_ascii=False, allow_nan=False)
    _decode(payload)
    state = states.create(_current_actor(), policy_version=POLICY_VERSION)
    (directory / "selection.json").write_text(payload, encoding="utf-8")
    (directory / "profile.json").write_text(json.dumps(profile.to_dict(), ensure_ascii=False), encoding="utf-8")
    ApprovalStore(directory / "approvals.sqlite3")
    append_event(directory / "events.jsonl", "run_created", "RUN_CREATED", stage=state.current_stage, revision=state.revision)

    return {"status": "created", "run_dir": str(directory), "worker_started": False,
            "notice": "기능 검증용 새 분할입니다. 기존 ML 실험·Test는 변경하지 않습니다."}


def open_run(name):
    """OS 주체·snapshot·프로파일·소스·전체 계획이 바뀐 실행의 재개를 거부한다."""
    directory = _root(_directory(name))
    record = _decode(_read_file(directory / "selection.json", 65536).decode("utf-8"))

    if set(record) != {"contract", "code_id", "plan_id"}:
        raise ValueError("분할 선택 기록의 필드가 올바르지 않습니다.")

    states = LocalStateStore(directory)

    if states.load().actor_id != _current_actor():
        raise ValueError("실행을 생성한 OS 계정에서만 접근하세요.")

    contract = EdaInputContract.from_json(json.dumps(record["contract"]))
    tool = SnapshotSplitTool(states, contract)
    profile = profile_from_dict(_decode(_read_file(directory / "profile.json", 65536).decode("utf-8")))

    if profile != contract.profile or record["code_id"] != tool.code_id or record["plan_id"] != tool.plan.plan_id:
        raise ValueError("분할 코드·프로파일·계획이 변경되었습니다. 새 실행과 승인이 필요합니다.")

    return tool


def approve(tool):
    """현재 한 단계만 대화형 승인한다. EDA 확인과 분할 확정은 각각 별도 승인이다."""
    if not os.isatty(0):
        raise ValueError("승인은 대화형 관리자 터미널에서 진행하세요.")

    manager = executor(tool)
    state = tool.states.load()

    if state.status != "awaiting_approval" or tool.cancel.is_set():
        raise ValueError("취소되지 않은 승인 대기 작업이 필요합니다.")

    tool._check_arguments(tool.arguments)
    scope = manager.scope_for(state.current_stage)
    token = "SPLIT-" + secrets.token_hex(4).upper()
    print(json.dumps({
        "stage": state.current_stage, "rules": asdict(tool.contract.profile.split_rules),
        "quality_rules": asdict(tool.contract.profile.quality_rules),
        "limits": asdict(tool.limits), "snapshot_bytes": tool.contract.snapshot.size_bytes,
        "lifetime_minutes": 30, "local_reports": local_reports(tool),
        "notice": "현재 한 단계만 승인합니다. 검토 필요는 자동 해제하지 않습니다. 삭제·대치·스케일링·모델 학습·LLM 전송 없음.",
    }, ensure_ascii=False, indent=2))

    if input(f"로컬 보고서와 설정을 확인했다면 {token} 를 입력하세요: ") != token:
        return {"code": "APPROVAL_DECLINED"}

    with _run_lock(tool.states.directory) as acquired:
        if not acquired or tool.states.load() != state or tool.cancel.is_set():
            raise ValueError("승인 중 실행 상태가 변경되었습니다.")

        tool._check_arguments(tool.arguments)

        if manager.scope_for(state.current_stage) != scope:
            raise ValueError("승인 대상이 변경되었습니다.")

        append_event(tool.states.directory / "events.jsonl", "approval_requested", "LOCAL_REVIEW_CONFIRMED", stage=state.current_stage, revision=state.revision)
        record = manager.approvals.record_confirmed_approval(scope)

    return {"approval_id": record.approval_id, "expires_at": record.expires_at.isoformat()}


def local_reports(tool):
    """값·통계 대신 로컬 보고서 경로만 안내한다."""
    state = tool.states.load()
    filenames = {"audit": "quality/quality.md", "run_eda": "eda/eda.md", "propose_split": "proposal.md"}
    return {stage: str(tool.states.directory / state.artifact_refs[stage] / filename)
            for stage, filename in filenames.items() if stage in state.artifact_refs}


def export_scope(tool):
    """실제 승인·커밋 증거를 확인한 뒤 정확한 범위만 저장한다. 승인 이력은 생성하지 않는다."""
    with _run_lock(tool.states.directory) as acquired:
        if not acquired or tool.states.load().status != "completed":
            raise ValueError("완료된 분할 실행이 필요합니다.")

        tool._check_arguments(tool.arguments)
        scope = executor(tool).scope_for("commit_split")
        read_confirmed_split(tool.states.directory, scope)
        path = tool.states.directory / "split_scope.json"
        payload = json.dumps(asdict(scope), sort_keys=True, ensure_ascii=False)

        if path.exists():
            if _read_file(path, 65536).decode("utf-8") != payload:
                raise ValueError("기존 승인 범위 파일이 현재 증거와 다릅니다.")
        else:
            with path.open("x", encoding="utf-8") as stream:
                stream.write(payload)

    return {"split_run_dir": str(tool.states.directory), "split_scope": str(path),
            "snapshot_dir": tool.contract.snapshot_directory, "profile": str(tool.states.directory / "profile.json")}


def main():
    parser = argparse.ArgumentParser(description="복사본의 격리 품질·EDA·분할 검토와 승인 확정")
    commands = parser.add_subparsers(dest="command", required=True)

    for command in ("create", "next", "approve", "status", "export-scope", "cancel"):
        sub = commands.add_parser(command)
        sub.add_argument("--name", required=True)

        if command == "create":
            sub.add_argument("--source", required=True)
            sub.add_argument("--profile", required=True)
            sub.add_argument("--max-input-bytes", type=int, required=True)
            sub.add_argument("--memory-mib", type=int, default=2048)
            sub.add_argument("--processes", type=int, default=2)
            sub.add_argument("--wall-seconds", type=int, default=300)
            sub.add_argument("--confirm-selection", action="store_true")
            sub.add_argument("--allow-feature-quality-findings", action="store_true",
                             help="결측·전체 결측·상수 센서를 삭제하지 않고 통지만 남기는 분할 검증 정책")
        elif command == "next":
            sub.add_argument("--approval-id")

    args = parser.parse_args()

    try:
        if args.command == "create":
            result = create(args.name, args.source, args.profile, WorkerLimits(args.wall_seconds, args.memory_mib, args.processes),
                            args.max_input_bytes, confirmed=args.confirm_selection,
                            allow_feature_findings=args.allow_feature_quality_findings)
        else:
            tool = open_run(args.name)

            if args.command == "next":
                tool._check_arguments(tool.arguments)
                result = asdict(executor(tool).run_next(approval_id=args.approval_id))
            elif args.command == "approve":
                result = approve(tool)
            elif args.command == "export-scope":
                result = export_scope(tool)
            elif args.command == "cancel":
                result = cancel(tool)
            else:
                state = tool.states.load()
                result = {"status": state.status, "stage": state.current_stage,
                          "cancel_requested": tool.cancel.is_set(), "local_reports": local_reports(tool)}

        print(json.dumps(result, ensure_ascii=False, indent=2))
    except InputTooLarge as error:
        print("입력 파일이 복사 상한을 초과했습니다. 자르거나 일부만 분석하지 않았습니다.")
        raise SystemExit(2)
    except Exception:
        print("분할 명령을 완료하지 못했습니다. 새 이름·입력·프로파일·승인·로컬 보고서·관리자 권한을 확인하세요.")
        print("비격리 재시도·자동 승인·행 삭제·재시드는 하지 않습니다.")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
