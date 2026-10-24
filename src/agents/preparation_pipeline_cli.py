"""변환 모듈 계획을 기존 후보·적용 실행기에 연결하는 로컬 통합 CLI."""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import secrets

from . import preparation_candidate_cli as candidate
from . import preparation_apply_cli as application
from .artifacts import _root, _fingerprint
from .cli import _current_actor
from .executor import _run_lock
from .isolated_eda_cli import cancel as cancel_child
from .preparation_apply_contract import read_candidate_evidence
from .preparation_module_plan import validate_preparation_modules
from .split_evidence import read_metadata
from .worker_contract import WorkerLimits, _decode


def _directory(name, *, create_parent=False):
    """하위 실행 이름에 접미사를 붙일 여유를 남기고 경로 입력은 거부한다."""
    if type(name) is not str or re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", name) is None:
        raise ValueError("통합 실행 이름은 소문자로 시작하는 영문·숫자·밑줄·하이픈 1~32자여야 합니다.")

    root = Path(__file__).resolve().parents[2]

    for part in ("logs", "agents", "preparation_pipeline", "runs"):
        root = root / part

        if create_parent:
            root.mkdir(exist_ok=True)

        _root(root)

    return root / name


def _code_id():
    """통합 제어 코드 변경은 재개 전에 차단한다. 기존 worker의 코드 검사는 별도로 유지한다."""
    digest = hashlib.sha256()

    for relative in ("preparation_pipeline_cli.py", "preparation_module_plan.py", "../preprocessing/preparation_modules.py"):
        digest.update(relative.encode("utf-8"))
        digest.update((Path(__file__).parent / relative).read_bytes())

    return digest.hexdigest()


def _save_new(path, value):
    """기존 기록은 덮어쓰지 않는다. 중단된 초기화도 자동 재생성하지 않는다."""
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2))


def _child_identity(tool):
    return {"run_id": tool.states.load().run_id,
            "selection": _fingerprint(tool.states.directory / "selection.json")}


def create(name, pipeline, split_run, split_scope, export_run, limits, candidate_bytes, partition_bytes, *, confirmed=False):
    """설정과 입력 선택을 결합하고 후보 작업만 만든다. 승인·worker 실행은 하지 않는다."""
    return _create(name, pipeline, limits, candidate_bytes, partition_bytes, confirmed=confirmed,
                   legacy=(split_run, split_scope, export_run))


def create_from_tabular(name, pipeline, train, limits, candidate_bytes, partition_bytes, *, confirmed=False):
    """typed Train을 기존 후보 검토·별도 적용 승인 경로에 연결한다."""
    return _create(name, pipeline, limits, candidate_bytes, partition_bytes, confirmed=confirmed, train=train)


def _create(name, pipeline, limits, candidate_bytes, partition_bytes, *, confirmed, legacy=None, train=None):
    """입력 선택만 버전별로 나누고 기존 통합 기록과 검토 절차는 공유한다."""
    if not confirmed:
        raise ValueError("전처리 계획과 입력 선택을 명시적으로 확인하세요.")

    # IPC와 같은 제한 파서를 사용해 중복 키·비표준 숫자·64 KiB 초과를 거부한다.
    with Path(pipeline).open("rb") as stream:
        payload = stream.read(65537)

    if len(payload) > 65536:
        raise ValueError("파이프라인 설정은 64 KiB 이하여야 합니다.")

    config = _decode(payload.decode("utf-8"))
    plan = validate_preparation_modules(config)

    if type(partition_bytes) is not int or not 0 < partition_bytes <= 1024 ** 3:
        raise ValueError("구간 출력 상한은 1 바이트~1 GiB 사이여야 합니다.")
    if legacy is None:
        from .tabular_source_contract import TabularPartitionInputContract
        from .preparation_candidate_contract import PreparationCandidateContract
        from src.preprocessing.transform import PreparationRules
        if type(train) is not TabularPartitionInputContract:
            raise ValueError("고정된 typed Train 계약이 필요합니다.")
        if train.split.artifact.scope.actor_id != _current_actor() or train.export.scope.actor_id != _current_actor():
            raise ValueError("분할·내보내기 계정이 다릅니다.")
        PreparationCandidateContract(train, PreparationRules(**json.loads(plan.rules_json)), limits, candidate_bytes,
                                     schema_version="preparation_candidate_v2").check()

    directory = _directory(name, create_parent=True)
    directory.mkdir()  # 기존 실행·불완전한 초기화는 덮어쓰지 않는다.
    candidate_name = name + "-candidate"
    apply_name = name + "-apply"
    _save_new(directory / "pipeline.json", config)
    _save_new(directory / "rules.json", json.loads(plan.rules_json))
    if legacy is None:
        candidate.create_from_tabular(candidate_name, train, PreparationRules(**json.loads(plan.rules_json)),
                                      limits, candidate_bytes, confirmed=True)
    else:
        candidate.create(candidate_name, *legacy, directory / "rules.json", limits, candidate_bytes, confirmed=True)
    tool = candidate.open_run(candidate_name)
    record = {"schema_version": "preparation_pipeline_run_v1", "actor": _current_actor(),
              "plan_signature": plan.signature, "code_id": _code_id(),
              "candidate_name": candidate_name, "apply_name": apply_name,
              "candidate_identity": _child_identity(tool), "limits": asdict(limits),
              "max_partition_bytes": partition_bytes}
    _save_new(directory / "selection.json", record)

    return {"status": "created", "run_dir": str(directory), "worker_started": False,
            "stage": "candidate", "notice": "후보 산출과 적용은 각각 새 승인이 필요합니다."}


def open_run(name):
    """통합 기록과 하위 실행의 동일성을 검증한다. 상태명만으로 완료를 신뢰하지 않는다."""
    directory = _root(_directory(name))
    record = read_metadata(directory / "selection.json")
    fields = {"schema_version", "actor", "plan_signature", "code_id", "candidate_name", "apply_name",
              "candidate_identity", "limits", "max_partition_bytes"}

    if set(record) != fields or record["schema_version"] != "preparation_pipeline_run_v1" or record["actor"] != _current_actor():
        raise ValueError("통합 실행의 기록 또는 OS 계정이 다릅니다.")

    plan = validate_preparation_modules(read_metadata(directory / "pipeline.json"))

    if (plan.signature != record["plan_signature"] or _code_id() != record["code_id"]
            or record["candidate_name"] != name + "-candidate" or record["apply_name"] != name + "-apply"):
        raise ValueError("계획·코드·하위 실행 연결이 변경되었습니다. 새 실행이 필요합니다.")

    WorkerLimits(**record["limits"])

    if type(record["max_partition_bytes"]) is not int or not 0 < record["max_partition_bytes"] <= 1024 ** 3:
        raise ValueError("저장된 구간 출력 상한이 올바르지 않습니다.")

    tool = candidate.open_run(record["candidate_name"])

    if (_child_identity(tool) != record["candidate_identity"] or asdict(tool.contract.rules) != json.loads(plan.rules_json)
            or asdict(tool.limits) != record["limits"]):
        raise ValueError("후보 작업이 통합 계획과 다릅니다.")

    return directory, record, tool


def _application(directory, record, candidate_tool):
    """사용자 검토 기록과 정확한 후보 상태에 연결된 적용 작업만 복원한다."""
    review_path = directory / "review.json"

    if not review_path.exists():
        return None

    review = read_metadata(review_path)
    evidence = read_candidate_evidence(candidate_tool.states.directory, record["actor"])
    tool = application.open_run(record["apply_name"])

    if (set(review) != {"candidate_id", "apply_identity"} or review["candidate_id"] != evidence.signature
            or tool.contract.candidate != evidence or _child_identity(tool) != review["apply_identity"]
            or asdict(tool.limits) != record["limits"] or tool.contract.max_partition_bytes != record["max_partition_bytes"]):
        raise ValueError("검토한 후보·적용 대상·자원 한도가 변경되었습니다.")

    return tool


def operate(name, command, approval_id=None):
    """한 명령에서 한 하위 단계만 수행한다. 완료 후보를 자동 승인·적용하지 않는다."""
    directory, record, candidate_tool = open_run(name)

    # 취소는 실행 잠금 밖에서 신호를 전달해 실행 중 worker도 중단할 수 있게 한다.
    if command == "cancel":
        marker = directory / "CANCEL_REQUESTED"

        try:
            marker.touch(exist_ok=False)
        except FileExistsError:
            pass

        apply_tool = _application(directory, record, candidate_tool)

        if candidate_tool.states.load().status != "completed":
            cancel_child(candidate_tool)

        if apply_tool is not None and apply_tool.states.load().status != "completed":
            cancel_child(apply_tool)

        return {"status": "cancel_requested", "notice": "이미 커밋된 결과는 삭제하지 않습니다."}

    with _run_lock(directory) as acquired:
        if not acquired:
            raise ValueError("다른 통합 명령이 진행 중입니다.")

        # 잠금 대기 중 바뀐 기록도 다시 검증한다.
        directory, record, candidate_tool = open_run(name)

        if (directory / "CANCEL_REQUESTED").exists():
            return {"status": "cancel_requested"}

        apply_tool = _application(directory, record, candidate_tool)
        backend = candidate if apply_tool is None else application
        tool = candidate_tool if apply_tool is None else apply_tool
        result = backend.status(tool)

        if command == "status":
            if apply_tool is None and result["status"] == "completed":
                result.update(status="awaiting_candidate_review", stage="review_candidate")

            return result

        if command == "review":
            if apply_tool is not None:
                return {"status": "already_reviewed", "stage": "apply_preparation"}

            return _review(directory, record, candidate_tool, result)

        if apply_tool is None and result["status"] == "completed":
            return {"status": "awaiting_candidate_review", "stage": "review_candidate",
                    "local_report": result["local_report"]}

        if command == "approve":
            return backend.approve(tool)

        if command != "next":
            raise ValueError("지원하지 않는 통합 명령입니다.")

        tool._check_arguments(tool.arguments)
        return asdict(backend.executor(tool).run_next(approval_id=approval_id))


def _review(directory, record, tool, result):
    """후보 확인은 적용 작업 선택일 뿐이다. 실제 적용 승인은 다음 명령에서 따로 받는다."""
    if result["status"] != "completed" or not os.isatty(0):
        raise ValueError("완료 후보를 대화형 터미널에서 검토해야 합니다.")

    evidence = read_candidate_evidence(tool.states.directory, record["actor"])
    token = "REVIEW-" + secrets.token_hex(4).upper()
    print(json.dumps({"candidate_report": str(evidence.directory / "candidate.md"),
                      "candidate_id": evidence.signature,
                      "notice": "후보 검토 후 적용 작업만 준비합니다. 실제 적용은 별도 승인이 필요합니다."}, ensure_ascii=False, indent=2))

    if input(f"보고서를 확인했다면 {token} 를 입력하세요: ") != token:
        return {"code": "REVIEW_DECLINED"}

    evidence.check()

    if (directory / "CANCEL_REQUESTED").exists():
        return {"status": "cancel_requested"}

    application.create(record["apply_name"], tool.states.directory, WorkerLimits(**record["limits"]),
                       record["max_partition_bytes"], confirmed=True)
    apply_tool = application.open_run(record["apply_name"])
    _save_new(directory / "review.json", {"candidate_id": evidence.signature, "apply_identity": _child_identity(apply_tool)})

    # 생성 도중 도착한 취소도 적용 실행 전에 전달한다.
    if (directory / "CANCEL_REQUESTED").exists():
        cancel_child(apply_tool)
        return {"status": "cancel_requested"}

    return {"status": "application_created", "worker_started": False, "stage": "preflight"}


def main():
    parser = argparse.ArgumentParser(description="모듈 계획 기반 후보 산출·검토·별도 승인 적용")
    commands = parser.add_subparsers(dest="command", required=True)

    for command in ("create", "next", "approve", "review", "status", "cancel"):
        sub = commands.add_parser(command)
        sub.add_argument("--name", required=True)

        if command == "create":
            for option in ("pipeline", "split-run-dir", "split-scope", "export-run-dir"):
                sub.add_argument("--" + option, required=True)

            sub.add_argument("--memory-mib", type=int, default=2048)
            sub.add_argument("--processes", type=int, default=2)
            sub.add_argument("--wall-seconds", type=int, default=300)
            sub.add_argument("--max-candidate-bytes", type=int, default=4194304)
            sub.add_argument("--max-partition-bytes", type=int, default=33554432)
            sub.add_argument("--confirm-selection", action="store_true")
        elif command == "next":
            sub.add_argument("--approval-id")

    args = parser.parse_args()

    try:
        if args.command == "create":
            result = create(args.name, args.pipeline, args.split_run_dir, args.split_scope, args.export_run_dir,
                            WorkerLimits(args.wall_seconds, args.memory_mib, args.processes),
                            args.max_candidate_bytes, args.max_partition_bytes, confirmed=args.confirm_selection)
        else:
            result = operate(args.name, args.command, getattr(args, "approval_id", None))

        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception:
        print("통합 명령을 완료하지 못했습니다. 계획·완료 증거·승인·실행 이름·권한을 확인하세요.")
        print("불완전한 초기화는 덮어쓰거나 자동 복구하지 않습니다. 기존 기록을 보존하고 새 이름을 사용하세요.")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
