"""기존 로컬 전달 CLI를 일회 승인 실행기에 연결한다. 격리 worker를 흉내 내지 않는다."""

from dataclasses import asdict
import json
from pathlib import Path
import re

from .approval_store import ApprovalStore
from .artifacts import _fingerprint, _root
from .cli import _current_actor
from .contracts import ToolResult
from .csv_handoff_cli import export_bundle, _directory, verify_bundle
from .csv_handoff_contract import PREPARED_FILES, MAX_BUNDLE_BYTES
from .csv_workflow_creation import _pipeline, code_id
from .executor import ExecutionPlan, ExecutionStep, RegisteredTool, RulesExecutor, _digest
from .isolated_eda import _read_file
from .state import LocalStateStore
from .split_evidence import read_metadata
from .worker_contract import WorkerLimits


def directory_for(name):
    if type(name) is not str or re.fullmatch(r"[a-z][a-z0-9_-]{0,47}", name) is None:
        raise ValueError("고정 형식의 전달 작업 이름이 필요합니다.")
    return Path(__file__).resolve().parents[2] / "logs/agents/csv_local_handoff/runs" / name


class HandoffTool:
    def __init__(self, states, selection):
        self.states, self.arguments = states, selection
        self.limits = WorkerLimits(**selection["limits"])
        self.cancel = states.directory / "CANCEL_REQUESTED"
        self.plan = ExecutionPlan("csv_handoff:" + selection["parent_id"], _digest(json.dumps(selection, sort_keys=True)),
            "csv_local_handoff_v1", (ExecutionStep("preflight", "run_preflight"),
                                     ExecutionStep("ingest", "run_ingest", json.dumps(selection))))

    def _check_arguments(self, arguments):
        if arguments != self.arguments or self.arguments["code_id"] != code_id():
            raise ValueError("로컬 전달 코드·선택이 변경됐습니다.")
        directory, record, candidate, evidence, application = _pipeline(self.arguments["pipeline_name"], applied=True)
        if _fingerprint(application.states.directory / "state.json")["sha256"] != self.arguments["parent_id"]:
            raise ValueError("적용 완료 증거가 변경됐습니다.")

    def preflight(self, output, arguments):
        self._check_arguments(self.arguments)
        (output / "preflight.json").write_text('{"local_copy":true}', encoding="utf-8")
        return ToolResult("passed")

    def validate_preflight(self, output, arguments):
        return not arguments and _read_file(output / "preflight.json", 4096) == b'{"local_copy":true}'

    def execute(self, output, arguments):
        self._check_arguments(arguments)
        result = export_bundle(arguments["pipeline_name"], arguments["name"], arguments["max_bundle_bytes"], confirmed=True)
        (output / "handoff.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        return ToolResult("passed", artifact_refs=("handoff.json",))

    def validate(self, output, arguments):
        self._check_arguments(arguments)
        if {path.name for path in output.iterdir()} - {"handoff.json", "AGENT_COMPLETE.json"}:
            return False
        result = read_metadata(output / "handoff.json")
        if result.get("directory") != str(_directory(arguments["name"])) or result.get("status") != "handoff_created":
            return False
        verify_bundle(_directory(arguments["name"]))
        bundle = _directory(arguments["name"])
        provenance = read_metadata(bundle / "provenance.json")
        directory, record, candidate, evidence, application = _pipeline(arguments["pipeline_name"], applied=True)
        from .preparation_apply_cli import status

        prepared = Path(status(application)["local_result_directory"])
        if any(_fingerprint(bundle / "prepared" / name) != _fingerprint(prepared / name) for name in PREPARED_FILES):
            return False
        return (provenance.get("pipeline_name") == arguments["pipeline_name"]
                and provenance.get("candidate_id") == evidence.signature
                and provenance.get("apply_run_id") == application.states.load().run_id
                and _fingerprint(bundle / "preprocessor.json")["sha256"] == evidence.state_sha256)


def manager(tool):
    # Path 자체에는 is_set이 없으므로 서비스와 기존 실행기에서 쓸 취소 표면을 제공한다.
    return RulesExecutor(tool.states, ApprovalStore(tool.states.directory / "approvals.sqlite3"), tool.plan, (
        RegisteredTool("run_preflight", tool.preflight, tool.validate_preflight),
        RegisteredTool("run_ingest", tool.execute, tool.validate, requires_approval=True),
    ))


def create_run(name, pipeline_name, maximum, limits):
    if type(maximum) is not int or not 0 < maximum <= MAX_BUNDLE_BYTES:
        raise ValueError("로컬 전달 바이트 상한이 잘못됐습니다.")
    directory, record, candidate, evidence, application = _pipeline(pipeline_name, applied=True)
    selected = {"name": name, "pipeline_name": pipeline_name, "max_bundle_bytes": maximum,
                "limits": asdict(limits), "code_id": code_id(), "actor": _current_actor(),
                "parent_id": _fingerprint(application.states.directory / "state.json")["sha256"]}
    target = directory_for(name)
    if not target.parent.exists():
        parent = Path(__file__).resolve().parents[2]
        for part in ("logs", "agents", "csv_local_handoff", "runs"):
            parent = parent / part
            if not parent.exists():
                parent.mkdir()
            _root(parent)
    _root(target.parent)
    target.mkdir()
    states = LocalStateStore(target)
    states.create(_current_actor(), policy_version="csv_local_handoff_v1")
    (target / "selection.json").write_text(json.dumps(selected, ensure_ascii=False), encoding="utf-8")
    ApprovalStore(target / "approvals.sqlite3")
    return {"run_dir": str(target), "status": "created"}


def open_run(name):
    directory = _root(directory_for(name))
    selection = read_metadata(directory / "selection.json")
    if set(selection) != {"name", "pipeline_name", "max_bundle_bytes", "limits", "code_id", "actor", "parent_id"}:
        raise ValueError("전달 선택 기록이 잘못됐습니다.")
    states = LocalStateStore(directory)
    if selection["actor"] != _current_actor() or states.load().actor_id != _current_actor() or selection["name"] != name:
        raise ValueError("전달 실행 계정·이름이 다릅니다.")
    tool = HandoffTool(states, selection)
    # 기존 승인 도구와 동일한 읽기 전용 취소 검사 인터페이스.
    from types import SimpleNamespace

    marker = tool.cancel
    tool.cancel = SimpleNamespace(is_set=marker.exists)
    tool._check_arguments(tool.arguments)
    return tool
