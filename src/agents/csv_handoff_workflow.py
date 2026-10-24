"""기존 로컬 전달 CLI를 일회 승인 실행기에 연결한다. 격리 worker를 흉내 내지 않는다."""

from dataclasses import asdict
import json
from pathlib import Path
import re
import tempfile

from .approval_store import ApprovalStore
from .artifacts import _fingerprint, _root
from .cli import _current_actor
from .contracts import ToolResult
from .csv_handoff_cli import export_bundle, _directory, verify_bundle
from .csv_handoff_contract import PREPARED_FILES, MAX_BUNDLE_BYTES, _snapshot_bundle
from .csv_workflow_creation import _pipeline, code_id
from .executor import ExecutionPlan, ExecutionStep, RegisteredTool, RulesExecutor, _digest
from .isolated_eda import _read_file
from .state import LocalStateStore
from .split_evidence import read_metadata
from .worker_contract import WorkerLimits


SELECTION_FIELDS = {"name", "pipeline_name", "max_bundle_bytes", "limits", "code_id", "actor", "parent_id"}


def directory_for(name, *, typed=False):
    if type(name) is not str or re.fullmatch(r"[a-z][a-z0-9_-]{0,47}", name) is None:
        raise ValueError("고정 형식의 전달 작업 이름이 필요합니다.")
    module = "tabular_local_handoff" if typed else "csv_local_handoff"
    return Path(__file__).resolve().parents[2] / "logs/agents" / module / "runs" / name


def _connection_code(typed):
    """v1은 기존 CSV 연결 지문, v2는 표형 서비스 연결 지문을 승인·실행 기준으로 쓴다."""
    if typed:
        from .tabular_workflow_service import code_id as tabular_code_id

        return tabular_code_id()
    return code_id()


class HandoffTool:
    def __init__(self, states, selection, *, recorded_code_id=None):
        self.states, self.arguments = states, selection
        # 읽기 전용 검증에서만 실행 당시 코드 지문을 기준으로 삼는다. 이 값이 있으면 실행 함수는 거부한다.
        self.recorded_code_id = recorded_code_id
        # v2 선택 기록에만 표형 출처 ID가 있다. 형식과 정책·계획 ID가 함께 바뀐다.
        self.typed = "source_contract_id" in selection
        self.limits = WorkerLimits(**selection["limits"])
        self.cancel = states.directory / "CANCEL_REQUESTED"
        prefix, policy = ("tabular_handoff:", "tabular_local_handoff_v2") if self.typed else ("csv_handoff:", "csv_local_handoff_v1")
        self.plan = ExecutionPlan(prefix + selection["parent_id"], _digest(json.dumps(selection, sort_keys=True)),
            policy, (ExecutionStep("preflight", "run_preflight"),
                                     ExecutionStep("ingest", "run_ingest", json.dumps(selection))))

    def _check_arguments(self, arguments):
        expected = _connection_code(self.typed) if self.recorded_code_id is None else self.recorded_code_id
        if arguments != self.arguments or self.arguments["code_id"] != expected:
            raise ValueError("로컬 전달 코드·선택이 변경됐습니다.")
        directory, record, candidate, evidence, application = _pipeline(self.arguments["pipeline_name"], applied=True,
                                                                        typed=self.typed)
        if _fingerprint(application.states.directory / "state.json")["sha256"] != self.arguments["parent_id"]:
            raise ValueError("적용 완료 증거가 변경됐습니다.")
        if self.typed and evidence.derivation.train.parent_source_id != self.arguments["source_contract_id"]:
            raise ValueError("적용 결과의 표형 출처가 전달 선택과 다릅니다.")

    def _refuse_recorded(self):
        if self.recorded_code_id is not None:
            raise ValueError("읽기 전용 검증으로 연 전달 실행은 실행할 수 없습니다.")

    def preflight(self, output, arguments):
        self._refuse_recorded()
        self._check_arguments(self.arguments)
        (output / "preflight.json").write_text('{"local_copy":true}', encoding="utf-8")
        return ToolResult("passed")

    def validate_preflight(self, output, arguments):
        return not arguments and _read_file(output / "preflight.json", 4096) == b'{"local_copy":true}'

    def execute(self, output, arguments):
        self._refuse_recorded()
        self._check_arguments(arguments)
        result = export_bundle(arguments["pipeline_name"], arguments["name"], arguments["max_bundle_bytes"], confirmed=True,
                               source_contract_id=arguments.get("source_contract_id"))
        (output / "handoff.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        return ToolResult("passed", artifact_refs=("handoff.json",))

    def validate(self, output, arguments):
        return self.verified_bundle(output, arguments) is not None

    def verified_bundle(self, output, arguments):
        """묶음을 비공개 사본으로 고정하고 자체 검사·승인 기록 대조를 같은 바이트에서 수행한다.

        통과하면 봉인한 완료 기록의 형식·파일 목록·해시를 돌려주고, 아니면 None이다.
        """
        self._check_arguments(arguments)
        if {path.name for path in output.iterdir()} - {"handoff.json", "AGENT_COMPLETE.json"}:
            return None
        result = read_metadata(output / "handoff.json")
        if result.get("directory") != str(_directory(arguments["name"])) or result.get("status") != "handoff_created":
            return None
        expected = "tabular_preparation_handoff_v2" if self.typed else "csv_preparation_handoff_v1"
        directory, record, candidate, evidence, application = _pipeline(arguments["pipeline_name"], applied=True,
                                                                        typed=self.typed)
        from .preparation_apply_cli import status

        prepared = Path(status(application)["local_result_directory"])

        with tempfile.TemporaryDirectory() as scratch:
            bundle = Path(scratch)
            _snapshot_bundle(_directory(arguments["name"]), bundle)
            if verify_bundle(bundle)["schema_version"] != expected:
                return None
            provenance = read_metadata(bundle / "provenance.json")
            if self.typed:
                from .csv_handoff_cli import TYPED_HANDOFF_README

                # v2는 안내서 고정 문구와 계획·적용 계획 식별값까지 대조한다. v1 과거 결과 검증 범위는 바꾸지 않는다.
                if (provenance.get("source_contract_id") != arguments["source_contract_id"]
                        or provenance.get("pipeline_signature") != record["plan_signature"]
                        or provenance.get("apply_plan_id") != application.plan.plan_id
                        or _read_file(bundle / "README.md", 65536) != TYPED_HANDOFF_README.encode("utf-8")):
                    return None
            if any(_fingerprint(bundle / "prepared" / name) != _fingerprint(prepared / name) for name in PREPARED_FILES):
                return None
            if (provenance.get("pipeline_name") != arguments["pipeline_name"]
                    or provenance.get("candidate_id") != evidence.signature
                    or provenance.get("apply_run_id") != application.states.load().run_id
                    or _fingerprint(bundle / "preprocessor.json")["sha256"] != evidence.state_sha256):
                return None
            manifest = read_metadata(bundle / "HANDOFF_COMPLETE.json")
            return {"schema_version": manifest["schema_version"], "files": manifest["files"],
                    "manifest_sha256": _fingerprint(bundle / "HANDOFF_COMPLETE.json")["sha256"]}


def manager(tool):
    # Path 자체에는 is_set이 없으므로 서비스와 기존 실행기에서 쓸 취소 표면을 제공한다.
    return RulesExecutor(tool.states, ApprovalStore(tool.states.directory / "approvals.sqlite3"), tool.plan, (
        RegisteredTool("run_preflight", tool.preflight, tool.validate_preflight),
        RegisteredTool("run_ingest", tool.execute, tool.validate, requires_approval=True),
    ))


def _refused(output, arguments):
    raise ValueError("읽기 전용 검증에서는 실행할 수 없습니다.")


def open_recorded(name, recorded_code_id):
    """과거 실행 결과의 읽기 전용 검증용으로 연다. 실행·승인 경로(open_run·manager)와 분리된 진입점이다.

    전달 선택 기록의 코드 지문 = 호출자가 준 실행 당시 지문(작업 동의에 기록된 값) = 승인 증거 파일들의 코드 지문이고,
    실행 계정·이름·적용 완료 증거(parent_id)가 맞을 때만 연다. 돌려주는 검증용 실행기는 실행 함수 자리에 거부 함수가 있어
    이 경로에서 실행이 호출될 수 없다. 결과 해시·커밋 기록·도메인 검증은 호출자가 기존 결과 조회와 같은 코드로 수행한다.
    """
    from types import SimpleNamespace

    if type(recorded_code_id) is not str or re.fullmatch(r"[0-9a-f]{64}", recorded_code_id) is None:
        raise ValueError("실행 당시 코드 지문이 필요합니다.")
    directory = _root(directory_for(name))
    selection = read_metadata(directory / "selection.json")
    # 읽기 전용 과거 결과 검증은 기존 v1 전달만 허용한다.
    if set(selection) != SELECTION_FIELDS:
        raise ValueError("전달 선택 기록이 잘못됐습니다.")
    states = LocalStateStore(directory)
    if selection["actor"] != _current_actor() or states.load().actor_id != _current_actor() or selection["name"] != name:
        raise ValueError("전달 실행 계정·이름이 다릅니다.")
    if selection["code_id"] != recorded_code_id:
        raise ValueError("전달 실행의 코드 지문이 동의 기록과 다릅니다.")
    tool = HandoffTool(states, selection, recorded_code_id=recorded_code_id)
    marker = tool.cancel
    tool.cancel = SimpleNamespace(is_set=marker.exists)
    tool._check_arguments(tool.arguments)
    if tool.cancel.is_set():
        raise ValueError("취소된 전달 실행입니다.")
    checker = RulesExecutor(tool.states, ApprovalStore(tool.states.directory / "approvals.sqlite3"), tool.plan, (
        RegisteredTool("run_preflight", _refused, tool.validate_preflight),
        RegisteredTool("run_ingest", _refused, tool.validate, requires_approval=True),
    ))
    proofs = sorted(directory.glob("ui_approval_*.json"))
    if not proofs:
        raise ValueError("승인 증거 파일이 없습니다.")
    scopes = {step.stage: json.dumps(asdict(checker.scope_for(step.stage)), sort_keys=True) for step in tool.plan.steps}
    found = []
    for path in proofs:
        if re.fullmatch(r"ui_approval_[a-f0-9]{32}\.json", path.name) is None:
            raise ValueError("승인 증거 파일 이름이 잘못됐습니다.")
        proof = read_metadata(path)
        if (set(proof) != {"code_id", "scope"} or proof["code_id"] != recorded_code_id
                or json.dumps(proof["scope"], sort_keys=True) not in scopes.values()):
            raise ValueError("승인 증거의 코드 지문·범위가 실행 기록과 다릅니다.")
        found.append(json.dumps(proof["scope"], sort_keys=True))
    for step in tool.plan.steps:
        if checker.tools[step.action_id].requires_approval and found.count(scopes[step.stage]) != 1:
            raise ValueError("승인이 필요한 단계의 승인 증거가 하나가 아닙니다.")
    return tool, checker


def create_run(name, pipeline_name, maximum, limits, *, source_contract_id=None):
    if type(maximum) is not int or not 0 < maximum <= MAX_BUNDLE_BYTES:
        raise ValueError("로컬 전달 바이트 상한이 잘못됐습니다.")
    typed = source_contract_id is not None
    directory, record, candidate, evidence, application = _pipeline(pipeline_name, applied=True, typed=typed)
    if typed and evidence.derivation.train.parent_source_id != source_contract_id:
        raise ValueError("적용 결과의 표형 출처가 전달 선택과 다릅니다.")
    selected = {"name": name, "pipeline_name": pipeline_name, "max_bundle_bytes": maximum,
                "limits": asdict(limits), "code_id": _connection_code(typed), "actor": _current_actor(),
                "parent_id": _fingerprint(application.states.directory / "state.json")["sha256"]}
    if typed:
        selected["source_contract_id"] = source_contract_id
    target = directory_for(name, typed=True) if typed else directory_for(name)
    if not target.parent.exists():
        parent = Path(__file__).resolve().parents[2]
        for part in ("logs", "agents", "tabular_local_handoff" if typed else "csv_local_handoff", "runs"):
            parent = parent / part
            if not parent.exists():
                parent.mkdir()
            _root(parent)
    _root(target.parent)
    target.mkdir()
    states = LocalStateStore(target)
    states.create(_current_actor(), policy_version="tabular_local_handoff_v2" if typed else "csv_local_handoff_v1")
    (target / "selection.json").write_text(json.dumps(selected, ensure_ascii=False), encoding="utf-8")
    ApprovalStore(target / "approvals.sqlite3")
    return {"run_dir": str(target), "status": "created"}


def open_run(name, *, typed=False):
    directory = _root(directory_for(name, typed=True) if typed else directory_for(name))
    selection = read_metadata(directory / "selection.json")
    if set(selection) != (SELECTION_FIELDS | {"source_contract_id"} if typed else SELECTION_FIELDS):
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
