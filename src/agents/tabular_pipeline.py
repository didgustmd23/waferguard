"""v2 typed 입력을 기존 진단·분할 도구의 staging 확장 지점에 연결한다."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil

from src.preprocessing.input_profile import bind_input_profile_plan, input_profile_pipeline
from src.preprocessing.tabular_bundle import inspect_bundle, read_bundle_schema
from .artifacts import _root
from .executor import ExecutionPlan, ExecutionStep
from .input_profile_pipeline import InputProfileBinding
from .isolated_eda import CancelSignal
from .module_pipeline import ModulePipelineTool
from .module_pipeline_results import read_tabular_profile_result
from .module_pipeline_transport import code_manifest
from .pipeline_csv_execution import CsvPipelineExecution
from .snapshot_eda import _digest
from .tabular_source_contract import SOURCE_TYPES, TabularSourceContract
from .tabular_worker_input import DIAGNOSTIC_POLICY, MAX_SPLIT_ROWS, table_files
from .worker_contract import WorkerRequest, WorkerLimits, _decode
from .worker_paths import WorkerPathPolicy


ADAPTER_ID = "tabular_canonical_diagnostic_v2"


def check_source(source, limits):
    """선택·동의·worker 생성 전에 생성(또는 시간 복원 적용) 승인 증거와 제한을 확인한다."""
    if type(source) not in SOURCE_TYPES or type(limits) is not WorkerLimits:
        raise TypeError("검증된 v2 표형 출처·자원 계약이 필요합니다.")
    if source.bundle.rows > MAX_SPLIT_ROWS or any(getattr(limits, k) > getattr(source.limits, k) for k in asdict(limits)):
        raise ValueError("현재 표형 실행의 행 수 또는 자원 한도를 초과했습니다.")
    source.check_input()


def stage_table(tool, request, packet, settings, *, extras=None):
    """승인·예약 뒤 새 고정 폴더에 검증된 표와 제한된 선행 메타데이터를 복사한다."""
    source = tool.contract.source
    check_source(source, tool.limits)
    root = tool._work_root(request.task_id)
    root.parent.mkdir(parents=True, exist_ok=True)
    _root(root.parent)
    root.mkdir(exist_ok=False)
    for name in ("runtime", "input", "tools", "output", "scratch"):
        (root / name).mkdir()
    shutil.copytree(source.bundle_directory, root / "input/table")
    inspect_bundle(root / "input/table", expected=source.bundle, max_bytes=source.bundle.size_bytes)
    for name, payload in (extras or {}).items():
        if name not in {"preview.json", "split.json"} or type(payload) is not bytes or len(payload) > 8 * 1024 * 1024:
            raise ValueError("고정된 선행 결과 파일만 전달할 수 있습니다.")
        (root / "input" / name).write_bytes(payload)
    (root / "input" / settings).write_text(json.dumps(packet, ensure_ascii=False, sort_keys=True, allow_nan=False), encoding="utf-8")
    (root / "input/request.json").write_text(request.to_json(), encoding="utf-8")
    Path(request.output_directory).mkdir()
    Path(request.scratch_directory).mkdir()
    request.check_policy(WorkerPathPolicy(tuple(root / p for p in ("runtime", "input", "tools")), root / "output", root / "scratch",
                                         protected_roots=(tool.states.directory, *source.protected_directories)), tool.limits)
    check_source(source, tool.limits)
    return root


@dataclass(frozen=True)
class TabularDiagnosticInput:
    """프로파일 없이 생성 완료 표와 진단 자원만 결합한다."""
    source: TabularSourceContract
    limits: WorkerLimits
    schema_version: str = "tabular_diagnostic_contract_v2"

    def __post_init__(self):
        if self.schema_version != "tabular_diagnostic_contract_v2":
            raise ValueError("지원하지 않는 진단 계약입니다.")
        check_source(self.source, self.limits)

    def check_input(self):
        check_source(self.source, self.limits)
        return True

    @property
    def contract_id(self):
        return _digest({"source": self.source.packet(), "limits": asdict(self.limits), "version": self.schema_version})

    def to_dict(self):
        return {"source": self.source.to_dict(), "limits": asdict(self.limits), "schema_version": self.schema_version}


@dataclass(frozen=True)
class TabularDiagnosticBinding(InputProfileBinding):
    """고정 진단 계획을 승인·생성된 표에만 결합한다."""
    policy_version: str = "tabular_diagnostic_binding_v2"

    @property
    def adapter_id(self):
        return ADAPTER_ID

    @property
    def source_id(self):
        return self.input_contract.source.source_id

    def approval_summary(self):
        return {"adapter_id": self.adapter_id, "source_id": self.source_id, "diagnostic_only": True, "roles": None,
                "llm_transfer": False, "pipeline_signature": self.plan.signature, "execution_code_id": self.execution_code_id,
                "limits": asdict(self.input_contract.limits), "bundle": asdict(self.input_contract.source.bundle)}


class TabularProfileTool(ModulePipelineTool):
    """기존 모듈 도구의 승인·운송·결과 등록을 재사용하고 typed 입력만 바꾼다."""
    def __init__(self, states, contract, config):
        if type(contract) is not TabularDiagnosticInput:
            raise TypeError("v2 고정 진단 입력이 필요합니다.")
        self.states, self.contract, self.limits = states, contract, contract.limits
        self.cancel, self.worker_started = CancelSignal(states), False
        bound, self.module_plan = bind_input_profile_plan(config)
        if bound != input_profile_pipeline():
            raise ValueError("v2 진단은 단일 고정 진단 계획만 허용합니다.")
        self.code_id = _digest(code_manifest())
        self.arguments = {"version": DIAGNOSTIC_POLICY, "source": contract.source.packet(), "profile": None,
                          "limits": asdict(self.limits), "code_id": self.code_id,
                          "pipeline": bound, "pipeline_signature": self.module_plan.signature}
        _decode(json.dumps(self.arguments, ensure_ascii=False))
        self.config_id = _digest(self.arguments)
        self.plan = ExecutionPlan(source_id=contract.source.source_id, config_id=self.config_id, policy_version=DIAGNOSTIC_POLICY,
                                  steps=(ExecutionStep("preflight", "run_preflight"), ExecutionStep("run_eda", "run_eda", json.dumps(self.arguments))))

    def _check_arguments(self, arguments):
        if _digest(arguments) != self.config_id or self.code_id != _digest(code_manifest()):
            raise ValueError("진단 코드·입력·계획이 승인 범위와 다릅니다.")
        bound, plan = bind_input_profile_plan(arguments["pipeline"])
        if bound != input_profile_pipeline() or plan.signature != self.module_plan.signature:
            raise ValueError("고정 진단 계획이 다릅니다.")
        self.contract.check_input()

    def _request(self, task_id):
        root, state = self._work_root(task_id), self.states.load()
        return WorkerRequest(task_id=task_id, run_id=state.run_id, actor_id=state.actor_id, stage="run_eda", action_id="run_eda",
                             source_id=self.plan.source_id, config_id=self.config_id, plan_id=self.plan.plan_id,
                             policy_version=DIAGNOSTIC_POLICY, limits=self.limits, input_files=table_files(root, "eda.json"),
                             dependency_directories=(), output_directory=str(root / "output/result"), scratch_directory=str(root / "scratch/task"))

    def _stage(self, request):
        return stage_table(self, request, self.arguments, "eda.json")

    def _receipt(self, request):
        return {"version": DIAGNOSTIC_POLICY, "contract_id": self.contract.contract_id, "request_id": request.request_id,
                "task_id": request.task_id, "pipeline_signature": self.module_plan.signature, "job_cleanup": True, "profile_cleanup": True}

    def _read_result(self, directory):
        return read_tabular_profile_result(directory, self.module_plan, self.contract.source)


class TabularProfileExecution(CsvPipelineExecution):
    """바인딩 대조와 실행기 연산은 기존 실행 어댑터를 유지한다."""
    def __init__(self, states, binding):
        if type(binding) is not TabularDiagnosticBinding:
            raise TypeError("v2 진단 바인딩이 필요합니다.")
        self.binding = binding
        self.binding_arguments = {"binding_id": binding.binding_id, "binding": binding.to_dict()}
        self.worker_tool = TabularProfileTool(states, binding.input_contract, json.loads(binding.pipeline_config_json))
        if self.worker_tool.code_id != binding.execution_code_id or self.worker_tool.module_plan.signature != binding.plan.signature:
            raise ValueError("승인된 진단 코드·계획이 바뀌었습니다.")


def build_tabular_profile_execution(states, binding):
    """정적 등록부에서만 v2 진단 실행기를 만든다."""
    return TabularProfileExecution(states, binding)
