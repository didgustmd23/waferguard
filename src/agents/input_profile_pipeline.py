"""프로파일 없는 고정 입력 진단 계획을 기존 승인·격리 모듈 실행 경로에 연결한다.

진단 결과는 역할·기준 결정을 돕는 참고값이다. 분할·전처리 fit·라벨 결정의 입력으로 쓰지 않고
LLM에 전달하지 않는다. 일반 분석 계약(EdaInputContract·ReviewedCsvPipelineBinding)은 바꾸지 않는다.
"""

from dataclasses import asdict, dataclass
import csv
import hashlib
import json
from pathlib import Path

from src.preprocessing.input_profile import COUNT_COLUMNS, HEADER_DEDUPE_ID, PROFILE_COLUMNS, bind_input_profile_plan
from src.preprocessing.pipeline import PipelinePlan
from .approvals import ApprovalScope
from .artifacts import _root
from .executor import ExecutionPlan, ExecutionStep, _digest as _arguments_digest
from .input_snapshot import PAYLOAD_NAME, Snapshot, verify_snapshot
from .isolated_eda import CancelSignal
from .module_pipeline import ModulePipelineTool
from .module_pipeline_contract import POLICY_VERSION as WORKER_POLICY_VERSION
from .module_pipeline_results import read_header_dedupe_result, read_input_profile_result
from .module_pipeline_transport import code_manifest
from .pipeline_csv_execution import CsvPipelineExecution
from .reviewed_csv_pipeline_contract import _canonical
from .snapshot_eda import _digest
from .worker_contract import WorkerLimits, _decode


CONTRACT_VERSION = "snapshot_diagnostic_contract_v1"
POLICY_VERSION = "input_profile_binding_v1"
ADAPTER_ID = "tabular_csv_diagnostic_v1"


@dataclass(frozen=True)
class DiagnosticInputContract:
    """snapshot·자원 한도만 가진 진단 입력 계약. profile 속성이 없어 일반 경로에 넣으면 실패한다."""

    snapshot_directory: str
    snapshot: Snapshot
    limits: WorkerLimits
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self):
        if self.schema_version != CONTRACT_VERSION:
            raise ValueError("지원하지 않는 진단 입력 계약 버전입니다.")

        if not isinstance(self.snapshot, Snapshot) or not isinstance(self.limits, WorkerLimits):
            raise TypeError("snapshot·worker 자원 계약이 필요합니다.")

        if not isinstance(self.snapshot_directory, str) or not self.snapshot_directory:
            raise ValueError("로컬 snapshot 경로가 필요합니다.")

    def to_dict(self):
        return {
            "schema_version": self.schema_version, "snapshot_directory": self.snapshot_directory,
            "snapshot": asdict(self.snapshot), "limits": asdict(self.limits),
        }

    @property
    def contract_id(self):
        """입력 크기·상한·경로·자원 변경을 식별한다. 인증 서명이 아니다."""
        return hashlib.sha256(_canonical(self.to_dict()).encode("utf-8")).hexdigest()

    def check_input(self):
        """실행 직전 재호출할 복사본 검사. CSV 내용 해석이나 ACL 부여는 하지 않는다."""
        root = _root(self.snapshot_directory)

        if str(root) != self.snapshot_directory or not verify_snapshot(root, self.snapshot):
            raise ValueError("진단 입력 복사본이 계약과 일치하지 않습니다.")

        return True


@dataclass(frozen=True)
class InputProfileBinding:
    """진단 입력·고정 계획·worker 코드 지문을 함께 고정한 실행 전 계약."""

    input_contract: DiagnosticInputContract
    plan: PipelinePlan
    pipeline_config_json: str
    execution_code_id: str
    policy_version: str = POLICY_VERSION

    @property
    def adapter_id(self):
        return ADAPTER_ID

    @property
    def source_id(self):
        return self.input_contract.snapshot.source_id

    def approval_summary(self):
        """승인 화면용 요약. 역할이 없고 LLM으로 보내지 않는 진단 전용 실행임을 명시한다."""
        summary = {
            "adapter_id": self.adapter_id,
            "source_id": self.source_id,
            "snapshot_bytes": self.input_contract.snapshot.size_bytes,
            "snapshot_sha256": self.input_contract.snapshot.sha256,
            "diagnostic_only": True,
            "roles": None,
            "llm_transfer": False,
            "modules": [
                {"step_id": step.step_id, "module_id": step.module_id, "version": step.version, "settings": json.loads(step.effective_settings_json)}
                for step in self.plan.steps
            ],
            "pipeline_signature": self.plan.signature,
            "execution_code_id": self.execution_code_id,
            "limits": asdict(self.input_contract.limits),
        }

        if self.plan.steps[0].module_id == HEADER_DEDUPE_ID:
            # 데이터 변경이 있는 작업이므로 승인 화면에 바뀌는 범위를 직접 표시한다.
            summary["input_change"] = ("중복 컬럼명만 이름_번호로 바꾼 새 파생 입력을 만듭니다. "
                                       "데이터 값·행 순서·컬럼 순서·컬럼 수와 중복되지 않은 이름은 유지하며 원본과 기존 snapshot은 변경하지 않습니다.")

        return summary

    def to_dict(self):
        return {
            "schema_version": self.policy_version,
            "input_contract": self.input_contract.to_dict(),
            "pipeline_plan": self.plan.to_dict(),
            "pipeline_config": json.loads(self.pipeline_config_json),
            "pipeline_signature": self.plan.signature,
            "execution_code_id": self.execution_code_id,
        }

    @property
    def binding_id(self):
        return hashlib.sha256(_canonical(self.to_dict()).encode("utf-8")).hexdigest()

    def execution_plan(self):
        """일반 CSV 바인딩과 같은 preflight → run_pipeline 승인 단계를 만든다."""
        arguments = _canonical({"binding_id": self.binding_id, "binding": self.to_dict()})

        return ExecutionPlan(
            source_id=self.source_id, config_id=self.binding_id, policy_version=self.policy_version,
            steps=(ExecutionStep("preflight", "run_preflight", arguments), ExecutionStep("run_pipeline", "run_pipeline", arguments)),
        )

    def approval_scope(self, actor_id, run_id):
        plan = self.execution_plan()
        step = plan.steps[-1]

        return ApprovalScope(
            actor_id=actor_id, run_id=run_id, stage=step.stage, tool_id=step.action_id,
            target_id=f"{run_id}:{step.stage}", arguments_id=_arguments_digest(step.arguments_json),
            source_id=plan.source_id, config_id=plan.config_id, plan_id=plan.plan_id,
            policy_version=plan.policy_version,
        )


def bind_input_profile(config, snapshot_directory, snapshot, limits):
    """고정 진단 계획만 snapshot에 결합한다. worker는 시작하지 않는다."""
    contract = DiagnosticInputContract(str(_root(snapshot_directory)), snapshot, limits)
    contract.check_input()
    bound, plan = bind_input_profile_plan(config)

    return InputProfileBinding(contract, plan, _canonical(bound), _digest(code_manifest()))


def bind_tabular_diagnostic(source, limits):
    """승인·생성된 v2 표에 프로파일 없는 단일 고정 진단만 결합한다."""
    from src.preprocessing.input_profile import input_profile_pipeline
    from .tabular_pipeline import TabularDiagnosticInput, TabularDiagnosticBinding
    contract = TabularDiagnosticInput(source, limits)
    bound, plan = bind_input_profile_plan(input_profile_pipeline())
    return TabularDiagnosticBinding(contract, plan, _canonical(bound), _digest(code_manifest()))


class InputProfileTool(ModulePipelineTool):
    """기존 모듈 worker 도구의 예약·운송·커밋을 쓰고 계획 결합과 결과 검증만 진단용으로 바꾼다."""

    def __init__(self, states, contract, config):
        if not isinstance(contract, DiagnosticInputContract):
            raise TypeError("검증된 진단 입력 계약이 필요합니다.")

        self.states = states
        self.contract = contract
        self.limits = contract.limits
        bound, self.module_plan = bind_input_profile_plan(config)
        self.code_id = _digest(code_manifest())
        self.worker_started = False
        # worker 설정 필드 집합은 일반 경로와 같다. profile null은 고정 진단 계획에서만 허용된다.
        self.arguments = {
            "contract_id": contract.contract_id, "snapshot": asdict(contract.snapshot), "profile": None,
            "limits": asdict(contract.limits), "code_id": self.code_id,
            "pipeline": bound, "pipeline_signature": self.module_plan.signature,
        }
        _decode(json.dumps(self.arguments, ensure_ascii=False))
        self.config_id = _digest(self.arguments)
        self.plan = ExecutionPlan(
            source_id=contract.snapshot.source_id, config_id=self.config_id, policy_version=WORKER_POLICY_VERSION,
            steps=(ExecutionStep("preflight", "run_preflight"), ExecutionStep("run_eda", "run_eda", json.dumps(self.arguments))),
        )
        self.cancel = CancelSignal(states)

    def _check_arguments(self, arguments):
        if arguments != self.arguments or self.code_id != _digest(code_manifest()):
            raise ValueError("진단 코드·설정이 승인 대상과 다릅니다.")

        _, plan = bind_input_profile_plan(arguments["pipeline"])

        if plan.signature != self.module_plan.signature or _digest(arguments) != self.config_id:
            raise ValueError("진단 계획 또는 입력 계약이 변경되었습니다.")

        self.contract.check_input()

    def _read_result(self, directory):
        """결과 행의 컬럼명·순서를 승인된 snapshot 헤더와 대조한다. 헤더 외 원본 행은 읽지 않는다.

        중복 컬럼명 정비 계획이면 대응표·파생 입력을 원본 snapshot과 바이트 단위로 대조한다.
        """
        try:
            self.contract.check_input()

            if self.module_plan.steps[0].module_id == HEADER_DEDUPE_ID:
                return read_header_dedupe_result(directory, self.module_plan, self.contract.snapshot_directory,
                                                 self.contract.snapshot)

            with (Path(self.contract.snapshot_directory) / PAYLOAD_NAME).open(encoding="utf-8-sig", newline="") as stream:
                header = next(csv.reader(stream), [])
        except (OSError, ValueError, UnicodeError, csv.Error):
            return None

        return read_input_profile_result(directory, self.module_plan, tuple(header))


class InputProfileExecution(CsvPipelineExecution):
    """승인 바인딩 재대조·실행·검증은 CSV 실행 어댑터를 그대로 쓰고 진단 도구만 연결한다."""

    def __init__(self, states, binding):
        if not isinstance(binding, InputProfileBinding):
            raise TypeError("등록된 입력 진단 바인딩이 필요합니다.")

        self.binding = binding
        self.binding_arguments = {"binding_id": binding.binding_id, "binding": binding.to_dict()}
        self.worker_tool = InputProfileTool(states, binding.input_contract, json.loads(binding.pipeline_config_json))

        if self.worker_tool.module_plan.signature != binding.plan.signature:
            raise ValueError("승인된 진단 계획과 격리 worker 계획이 일치하지 않습니다.")

        if self.worker_tool.code_id != binding.execution_code_id:
            raise ValueError("승인된 worker 코드와 현재 실행 코드가 일치하지 않습니다.")


def build_input_profile_execution(states, binding):
    """정적으로 등록된 진단 실행 어댑터를 만든다. 사용자 지정 import는 하지 않는다."""
    return InputProfileExecution(states, binding)


def committed_diagnostic(run, module_id):
    """완료·커밋된 진단 run의 (도구, 출력 폴더)를 현재 코드·승인 기록·부모 검증기로 다시 확인해 반환한다."""
    from .artifacts import check_artifacts

    from .tabular_pipeline import ADAPTER_ID as TABULAR_ADAPTER_ID, TabularProfileTool

    tool = run.pipeline_execution.worker_tool
    # CSV v1 진단과 생성된 v2 표 진단만 받는다. 바인딩 종류와 도구 종류가 서로 맞아야 한다.
    tools = {ADAPTER_ID: InputProfileTool, TABULAR_ADAPTER_ID: TabularProfileTool}

    if run.binding.adapter_id not in tools or type(tool) is not tools[run.binding.adapter_id]:
        raise ValueError("입력 진단 바인딩이 아닙니다.")

    if run.states.load().status != "completed":
        raise ValueError("완료된 입력 진단 결과만 열 수 있습니다.")

    task = run.approvals.get_task(run.executor.scope_for("run_pipeline"))

    if task is None or task["status"] != "committed":
        raise ValueError("커밋된 진단 작업 기록이 없습니다.")

    output = _root(run.executor._output(task))

    if (not check_artifacts(output, run.executor._context(run.plan.steps[-1])).reusable
            or not run.pipeline_execution.validate(output, run.arguments)):
        raise ValueError("진단 결과의 정합성 확인에 실패했습니다.")

    if tool.module_plan.steps[0].module_id != module_id:
        raise ValueError("요청한 진단 작업 종류가 아닙니다.")

    return tool, output


def _profile_value(name, text):
    if text == "" or name in ("column", "dtype_group"):
        return None if text == "" else text

    return int(text) if name in COUNT_COLUMNS else float(text)


def read_profile_table(tool, output):
    """committed_diagnostic이 확인한 출력에서 요약과 컬럼별 표(자료형 변환 포함)를 읽는다."""
    from .isolated_eda import _read_file
    from .module_pipeline_results import MAX_FILE_BYTES

    child = _root(output / "pipeline" / tool.module_plan.steps[0].step_id)
    return parse_profile_table(_read_file(child / "input_profile.csv", MAX_FILE_BYTES),
                               _read_file(child / "input_summary.json", 65536))


def profile_table_files(tool):
    """진단 출력 폴더 기준의 요약·컬럼 표 상대 경로. 완료 표식 결합 읽기에 쓴다."""
    step = tool.module_plan.steps[0].step_id
    return "pipeline/" + step + "/input_profile.csv", "pipeline/" + step + "/input_summary.json"


def parse_profile_table(table, summary):
    """이미 읽은 진단 표·요약 바이트를 해석한다. 파일을 다시 열지 않는다."""
    from .eda_result_validation import _rows
    from .module_pipeline_results import MAX_FILE_BYTES

    if len(table) > MAX_FILE_BYTES or len(summary) > 65536:
        raise ValueError("진단 결과 파일 크기 제한을 넘었습니다.")

    rows = _rows(table)
    summary = _decode(summary.decode("utf-8"))

    if tuple(rows[0]) != PROFILE_COLUMNS:
        raise ValueError("진단 결과 표 형식이 다릅니다.")

    return summary, [{name: _profile_value(name, text) for name, text in zip(PROFILE_COLUMNS, row)} for row in rows[1:]]
