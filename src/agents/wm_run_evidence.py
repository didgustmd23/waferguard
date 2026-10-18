"""WM 정상 개발 경로의 완료 증거를 읽기 전용으로 확인한다. 실행 관리자는 만들지 않는다."""

from contextlib import closing
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import sqlite3

from src.preprocessing.canonical_data import decode_json
from .approvals import ApprovalScope
from .artifacts import ArtifactContext, _fingerprint, _root, check_artifacts
from .cli import _current_actor
from .executor import _digest
from .isolated_eda import _read_file
from .state import LocalStateStore
from .wm_canonical_contract import WmCanonicalInput, digest
from .wm_module_catalogue import _cli, wm_modules
from .worker_contract import WorkerRequest, WorkerResult, _decode
from .worker_paths import _path


# 실제 작업 생성·승인 함수는 등록하지 않는다. 조회 지원 범위는 정상 개발 6단계다.
# 충돌 검토·제외는 기존 CLI를 유지하며 이 조회 경로에 자동 편입하지 않는다.
SPECS = {
    "wm_canonical_eda": ("WmCanonicalTool", None, None, "analysis"),
    "wm_split_propose": ("WmSplitTool", "eda_run", "wm_canonical_eda", "proposal"),
    "wm_split_commit": ("WmSplitCommitTool", "proposal_run", "wm_split_propose", "split"),
    "wm_transform_check": ("WmTransformTool", "split_run", "wm_split_commit", "check"),
    "wm_cache_prepare": ("WmCacheTool", "check_run", "wm_transform_check", "cache"),
    "wm_dataset_check": ("WmDatasetTool", "cache_run", "wm_cache_prepare", "dataset"),
}


class RunNotComplete(ValueError):
    """미완료·취소·검토 중단을 완료 검증 실패와 구분하기 위한 로컬 상태."""

    def __init__(self, status, stage):
        super().__init__("완료되고 취소되지 않은 작업이 필요합니다.")
        self.status, self.stage = status, stage


def check_run_name(name):
    if type(name) is not str or re.fullmatch(r"[a-z][a-z0-9_-]{0,47}", name) is None:
        raise ValueError("작업 이름은 소문자로 시작하는 영문·숫자·밑줄·하이픈 1~48자여야 합니다.")


def _json(path, maximum=1024 * 1024):
    return decode_json(_read_file(path, maximum))


def _scope(tool, stage):
    """기존 계획의 정규화된 인자로 범위를 재구성한다. 승인 DB 초기화는 하지 않는다."""
    state = tool.states.load()
    step = next(item for item in tool.plan.steps if item.stage == stage)

    return ApprovalScope(
        actor_id=state.actor_id, run_id=state.run_id, stage=step.stage, tool_id=step.action_id,
        target_id=f"{state.run_id}:{step.stage}", arguments_id=_digest(step.arguments_json),
        source_id=tool.plan.source_id, config_id=tool.plan.config_id,
        plan_id=tool.plan.plan_id, policy_version=tool.plan.policy_version,
    )


def _committed(tool, stage):
    """정확한 작업 범위의 승인 소비·예약·완료 기록을 읽기 전용으로 대조한다."""
    scope = _scope(tool, stage)
    database = _path(tool.states.directory / "approvals.sqlite3", directory=False)
    scope_json = json.dumps(asdict(scope), ensure_ascii=False, sort_keys=True)

    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        task = connection.execute("SELECT task_id,status FROM task_runs WHERE scope_json=?", (scope_json,)).fetchone()
        approvals = connection.execute(
            "SELECT r.reservation_id,a.status FROM reservations r JOIN approvals a ON a.approval_id=r.approval_id "
            "WHERE r.scope_json=? AND a.scope_json=r.scope_json", (scope_json,),
        ).fetchall()

    if (task is None or task[1] != "committed" or approvals != [(task[0], "consumed")]
            or type(task[0]) is not str or re.fullmatch(r"[a-f0-9]{32}", task[0]) is None):
        raise ValueError("승인 소비·예약·커밋 기록이 일치하지 않습니다.")

    output = tool.states.directory / "attempts" / task[0]
    context = ArtifactContext(scope.source_id, scope.config_id, scope.plan_id, scope.policy_version, scope.tool_id)

    if not check_artifacts(output, context).reusable:
        raise ValueError("완료 표식 또는 산출물이 변경되었습니다.")

    return output


def _result(tool, module_id, folder, request):
    """원래 부모 결과 검증 함수를 직접 호출한다. 도구의 재귀 실행기 생성은 호출하지 않는다."""
    if module_id == "wm_canonical_eda":
        from .wm_canonical_results import inspect_analysis
        return inspect_analysis(folder, request, tool.contract)

    if module_id == "wm_split_propose":
        from .wm_split_results import inspect_proposal as inspect
    elif module_id == "wm_split_commit":
        from .wm_split_commit_results import inspect_split as inspect
    elif module_id == "wm_transform_check":
        from .wm_transform_results import inspect_check as inspect
    elif module_id == "wm_cache_prepare":
        from .wm_cache_results import inspect_cache as inspect
    elif module_id == "wm_dataset_check":
        from .wm_dataset_results import inspect_dataset
        return inspect_dataset(folder, request, tool.contract, tool.evidence)
    else:
        raise ValueError("조회 검증기가 등록되지 않은 WM 작업입니다.")

    return inspect(folder, request, tool.contract, digest(tool.evidence), tool.evidence)


def _inspect(tool, module_id, attempt, folder):
    """저장 요청·OS 종료·Job 정리·worker 응답·부모 복사본을 함께 검증한다."""
    from .appcontainer_probe import launch_succeeded

    if {path.name for path in _root(attempt).iterdir()} != {folder, "receipt.json", "AGENT_COMPLETE.json"}:
        raise ValueError("완료 시도 폴더의 파일 목록이 올바르지 않습니다.")

    request = tool._request(attempt.name)
    work = tool._work_root(attempt.name)
    saved = WorkerRequest.from_json(_read_file(work / "input/request.json", 65536).decode("utf-8"))
    outcome = _decode(_read_file(work / "transport.json", 65536).decode("utf-8"))
    reply = WorkerResult.from_json(_read_file(Path(request.output_directory) / "worker_result.json", 65536).decode("utf-8"))

    if (saved != request or not tool._evidence(request, outcome)
            or not launch_succeeded(outcome.get("launch", {})) or not reply.matches(request)
            or reply.status != "succeeded" or reply.artifact_refs != (folder,)):
        raise ValueError("저장 요청·worker 종료·자원 정리 증거가 일치하지 않습니다.")

    report = _result(tool, module_id, attempt / folder, request)

    if _json(attempt / "receipt.json") != {"request_id": request.request_id, "report": report}:
        raise ValueError("완료 영수증이 검증된 결과와 다릅니다.")

    return report


@dataclass
class VerifiedWmRun:
    module_id: str
    name: str
    tool: object
    attempt: Path
    report: dict
    stamps: dict

    def check_unchanged(self):
        """조회 중 상태·선택·승인 DB가 바뀌면 이전 확인 결과를 재사용하지 않는다."""
        for name, expected in self.stamps.items():
            if _fingerprint(_path(self.tool.states.directory / name, directory=False)) != expected:
                raise ValueError("조회 중 실행 기록이 변경되었습니다.")

        module = next(item for item in wm_modules() if item.contract.module_id == self.module_id)
        scope = _scope(self.tool, module.worker_stage)
        context = ArtifactContext(scope.source_id, scope.config_id, scope.plan_id, scope.policy_version, scope.tool_id)

        if not check_artifacts(self.attempt, context).reusable:
            raise ValueError("조회 중 완료 결과가 변경되었습니다.")

        for stage in module.review_stages:
            reference = _committed(self.tool, stage)

            if _json(reference / "reference.json", 65536) != self.tool._reference_record(stage):
                raise ValueError("조회 중 검토 승인에 연결된 증거가 변경되었습니다.")


def load_run_tool(module_id, name, *, require_complete=True):
    """기존 계약·도구 생성자만 재사용한다. open_run·executor·ApprovalStore 생성은 금지한다."""
    check_run_name(name)

    if module_id not in SPECS:
        raise ValueError("현재 완료 연결 조회는 WM 정상 개발 6단계만 지원합니다.")

    module = next(item for item in wm_modules() if item.contract.module_id == module_id)
    cli = _cli(module.cli_id)
    directory = _root(cli._directory(name))
    states = LocalStateStore(directory)
    state = states.load()

    if state.actor_id != _current_actor():
        raise ValueError("작업을 생성한 OS 계정에서 조회하세요.")

    if ((require_complete and state.status != "completed") or state.cancel_requested
            or state.blockers or (directory / "CANCEL_REQUESTED").exists()):
        raise RunNotComplete(state.status, state.current_stage)

    records = ("state.json", "selection.json", "approvals.sqlite3")

    if module_id == "wm_dataset_check":
        records += ("runtime_files.json",)

    stamps = {file: _fingerprint(_path(directory / file, directory=False)) for file in records}
    record = _json(directory / "selection.json", 4 * 1024 * 1024)
    class_name, parent_field, _, folder = SPECS[module_id]
    fields = {"contract", "code_id", "plan_id"} | ({parent_field, "evidence"} if parent_field else set())

    if type(record) is not dict or set(record) != fields:
        raise ValueError("기존 WM 선택 기록의 필드가 올바르지 않습니다.")

    if module_id == "wm_dataset_check":
        from .wm_dataset_contract import WmDatasetInput
        contract = WmDatasetInput.from_dict(record["contract"])
    else:
        contract = WmCanonicalInput.from_dict(record["contract"])

    factory = getattr(cli, class_name)  # class_name은 위 고정 코드 목록에서만 가져온다.
    arguments = (states, contract, record[parent_field], record["evidence"]) if parent_field else (states, contract)
    tool = factory(*arguments)

    if (tool.code_id != record["code_id"] or tool.plan.plan_id != record["plan_id"]
            or state.policy_version != tool.plan.policy_version):
        raise ValueError("기존 코드·설정·계획·정책이 변경되었습니다.")

    registered = tool.registered_tool()

    if (registered.action_id != module.worker_stage or registered.requires_approval is not True
            or registered.dependency_stage is not None):
        raise ValueError("조회기가 지원하는 기존 WM 실행 계약과 다릅니다.")

    contract.check()

    if module_id == "wm_dataset_check":
        tool._runtime()

    return tool, module, folder, stamps


def _open_record(module_id, name):
    """완료 조회는 기존과 동일하게 검토 승인·최종 커밋·산출물을 모두 검사한다."""
    tool, module, folder, stamps = load_run_tool(module_id, name)

    for stage in module.review_stages:
        reference = _committed(tool, stage)

        if _json(reference / "reference.json", 65536) != tool._reference_record(stage):
            raise ValueError("검토 승인이 현재 선행 증거에 연결되지 않았습니다.")

    attempt = _committed(tool, module.worker_stage)
    result = VerifiedWmRun(module_id, name, tool, attempt, _inspect(tool, module_id, attempt, folder), stamps)
    result.check_unchanged()

    if module_id == "wm_canonical_eda" and (result.report["quality_status"] != "passed" or result.report["eda_generated"] is not True):
        raise ValueError("통과한 WM 품질·EDA만 정상 개발 경로에 연결할 수 있습니다.")

    if module_id == "wm_transform_check" and result.report["status"] != "transform_check_passed":
        raise ValueError("변환 검토가 남아 있는 작업은 연결할 수 없습니다.")

    return result


def parent_evidence(child, parent):
    """이미 검증한 부모 결과로 자식의 저장 증거를 재구성한다. 상세 값은 로컬에서만 비교한다."""
    tool, previous = child.tool, parent.tool
    state = previous.states.load()
    common = {"run_id": state.run_id, "plan_id": previous.plan.plan_id, "source_id": previous.contract.source_id}

    if child.module_id == "wm_split_propose":
        quality = _json(parent.attempt / "analysis/quality/validation.json", 65536)
        config = tool.evidence["split_config"]

        if (config["max_group_ratio"] != previous.contract.max_group_ratio
                or quality["split_assessment"]["status"] != "preconditions_passed"
                or quality["duplicate_audit"]["map_label_conflicts"] != 0
                or quality["duplicate_audit"]["wafer_id_conflicts"] != 0):
            raise ValueError("동일 검토 상한에서 충돌 없이 통과한 EDA가 필요합니다.")

        value = {**common, "report_files": parent.report["files"], "quality": quality, "split_config": config}
    elif child.module_id == "wm_split_commit":
        value = {**common, "proposal_directory": str(parent.attempt / "proposal"),
                 "proposal_files": parent.report["files"], "proposal": _json(parent.attempt / "proposal/proposal.json", 65536),
                 "canonical_root": previous.contract.snapshot_directory}
    elif child.module_id == "wm_transform_check":
        value = {**common, "split_directory": str(parent.attempt / "split"), "split_files": parent.report["files"],
                 "split_report": _json(parent.attempt / "split/split.json", 65536),
                 "canonical_root": previous.contract.snapshot_directory, "transform_config": tool.evidence["transform_config"]}
    elif child.module_id == "wm_cache_prepare":
        value = {**common, "transform_run": parent.name, "split_run": previous.split_run,
                 "split_evidence": previous.evidence, "check_directory": str(parent.attempt / "check"),
                 "check_files": parent.report["files"], "cache_config": tool.evidence["cache_config"]}
    elif child.module_id == "wm_dataset_check":
        value = {"cache_run": parent.name, "run_id": state.run_id, "plan_id": previous.plan.plan_id,
                 "cache_directory": str(parent.attempt / "cache"), "cache_files": parent.report["files"],
                 "cache_report": _json(parent.attempt / "cache/cache.json")}
    else:
        raise ValueError("부모 증거 연결이 정의되지 않은 모듈입니다.")

    value["evidence_id"] = digest(value)
    return value


class WmRunReader:
    """한 조회 안에서만 검증을 재사용한다. 다음 CLI 호출에서는 모두 다시 확인한다."""

    def __init__(self):
        self.checked = {}

    def read(self, module_id, name):
        key = (module_id, name)

        if key in self.checked:
            self.checked[key].check_unchanged()
            return self.checked[key]

        result = _open_record(module_id, name)
        self.check_parent(module_id, result.tool)
        result.check_unchanged()
        self.checked[key] = result
        return result

    def check_parent(self, module_id, tool):
        """미완료 작업도 선행 결과가 현재 계약과 같은지 조회한다. 실행은 하지 않는다."""
        from types import SimpleNamespace
        _, parent_field, parent_module, _ = SPECS[module_id]

        if parent_field:
            parent = self.read(parent_module, getattr(tool, parent_field))
            current = SimpleNamespace(module_id=module_id, tool=tool)

            if tool.evidence != parent_evidence(current, parent):
                raise ValueError("부모 결과의 현재 증거가 자식에 저장된 증거와 다릅니다.")

            if module_id != "wm_dataset_check":
                if (tool.contract.source_id != parent.tool.contract.source_id
                        or tool.contract.max_group_ratio != parent.tool.contract.max_group_ratio):
                    raise ValueError("부모와 자식의 canonical 입력 또는 검토 상한이 다릅니다.")

    def check_unchanged(self):
        inputs = set()

        for result in self.checked.values():
            # 같은 canonical을 공유하는 단계는 최종 입력 해시 검사를 한 번만 한다.
            # 캐시와 canonical은 별도 경로이므로 서로 다른 입력으로 검사한다.
            contract = result.tool.contract
            directory = getattr(contract, "snapshot_directory", None) or contract.cache_directory
            key = (directory, contract.source_id)

            if key not in inputs:
                contract.check()
                inputs.add(key)

            result.check_unchanged()
