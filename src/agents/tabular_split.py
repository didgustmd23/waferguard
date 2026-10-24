"""기존 분할·내보내기 도구의 v2 typed 입력 확장. 실행·승인·결과 커밋은 상속한다."""

from dataclasses import asdict
import json

from .artifacts import _fingerprint
from .contracts import ToolResult
from .executor import ExecutionPlan, ExecutionStep
from .isolated_eda import CancelSignal, _read_file
from .partition_export import PartitionExportTool
from .partition_export_contract import TabularExportContract
from .partition_export_transport import code_manifest as export_manifest
from .snapshot_eda import _digest
from .snapshot_split import SnapshotSplitTool
from .snapshot_split_transport import code_manifest as split_manifest
from .split_evidence import read_split_files
from .tabular_pipeline import stage_table, check_source
from .tabular_source_contract import TabularEdaInputContract
from .tabular_worker_input import SPLIT_POLICY, EXPORT_POLICY, table_files
from .worker_contract import WorkerRequest, _decode


class TabularSplitTool(SnapshotSplitTool):
    """v2 표형 출처만 받아 기존 품질·EDA·검토·분할 단계를 사용한다."""
    def __init__(self, states, contract):
        if type(contract) is not TabularEdaInputContract:
            raise TypeError("v2 표형 분석 계약이 필요합니다.")
        check_source(contract.source, contract.limits)
        contract.check_input()
        if contract.profile.workflow_id != "split_review_v1" or contract.profile.split_rules is None:
            raise ValueError("v2 분할 검토 프로파일이 필요합니다.")
        self.states, self.contract, self.limits = states, contract, contract.limits
        self.cancel = CancelSignal(states)
        self.code_id = _digest(split_manifest())
        self.arguments = {"version": SPLIT_POLICY, "source": contract.source.packet(), "profile": contract.profile.to_dict(),
                          "limits": asdict(self.limits), "code_id": self.code_id, "proposal_sha256": None}
        _decode(json.dumps(self.arguments, ensure_ascii=False))
        self.plan = ExecutionPlan(source_id=contract.source.source_id, config_id=contract.profile.signature, policy_version=SPLIT_POLICY,
                                  steps=tuple(ExecutionStep(s, a, "{}" if s == "preflight" else json.dumps(self.arguments))
                                              for s, a in (("preflight", "run_preflight"), ("audit", "run_audit"), ("run_eda", "run_eda"),
                                                           ("review_eda", "review_eda"), ("propose_split", "propose_split"), ("commit_split", "commit_split"))))

    def _check_arguments(self, arguments):
        if _digest(arguments) != _digest(self.arguments) or self.code_id != _digest(split_manifest()):
            raise ValueError("분할 입력·설정·코드가 승인 범위와 다릅니다.")
        self.contract.check_input()

    def _request_for(self, task_id, stage, dependency=None):
        root, state = self._work_root(task_id), self.states.load()
        packet = self._packet(dependency)
        if stage == "commit_split" and dependency is None:
            raise ValueError("커밋에는 검증된 분할 제안이 필요합니다.")
        return WorkerRequest(task_id=task_id, run_id=state.run_id, actor_id=state.actor_id, stage=stage,
                             action_id="run_audit" if stage == "audit" else stage, source_id=self.plan.source_id,
                             config_id=_digest(packet), plan_id=self.plan.plan_id, policy_version=SPLIT_POLICY, limits=self.limits,
                             input_files=table_files(root, "split_task.json", extra=("preview.json",) if stage == "commit_split" else ()),
                             dependency_directories=(), output_directory=str(root / "output/result"), scratch_directory=str(root / "scratch/task"))

    def _check_rows(self, directory, stage):
        return read_split_files(directory, self.plan.source_id, self.plan.config_id, preview=stage == "propose_split",
                                committed=(directory / "AGENT_COMPLETE.json").exists(), allow_review=True,
                                expected_rows=self.contract.source.bundle.rows)

    def _report_status(self, directory, stage):
        if stage not in {"propose_split", "commit_split"}:
            status = super()._report_status(directory, stage)
            if status is not None and stage == "run_eda":
                summary = _decode(_read_file(directory / "summary.json", 65536).decode("utf-8"))
                if summary["row_count"] != self.contract.source.bundle.rows:
                    return None
            return status
        try:
            report = self._check_rows(directory, stage)
            return report["status"] if report["rules"] == asdict(self.contract.profile.split_rules) else None
        except (ValueError, TypeError, OSError, KeyError):
            return None

    def validate_stage(self, stage, output, arguments, dependency=None):
        if not super().validate_stage(stage, output, arguments, dependency):
            return False
        if stage in {"propose_split", "commit_split"}:
            try:
                self._check_rows(output, stage)
            except (ValueError, TypeError, OSError, KeyError):
                return False
        return True

    def _stage_for(self, request, dependency):
        extras = {} if dependency is None else {"preview.json": _read_file(dependency / "proposal.json", 8 * 1024 * 1024)}
        root = stage_table(self, request, self._packet(dependency), "split_task.json", extras=extras)
        if extras and _fingerprint(root / "input/preview.json")["sha256"] != self._packet(dependency)["proposal_sha256"]:
            raise ValueError("복사한 제안의 지문이 다릅니다.")
        return root

    def review(self, output, arguments):
        self._check_arguments(arguments)
        (output / "review.json").write_text(json.dumps({"version": SPLIT_POLICY, "reviewed": True}), encoding="utf-8")
        return ToolResult("passed", artifact_refs=("review.json",))

    def validate_review(self, output, arguments):
        self._check_arguments(arguments)
        return not self.cancel.is_set() and _decode(_read_file(output / "review.json", 4096).decode()) == {"version": SPLIT_POLICY, "reviewed": True}


class TabularExportTool(PartitionExportTool):
    """내보내기 실행과 부모 검증을 유지하고 입력 staging만 typed 묶음으로 바꾼다."""
    def __init__(self, states, contract):
        if type(contract) is not TabularExportContract:
            raise TypeError("v2 분할 내보내기 계약이 필요합니다.")
        self.states, self.contract, self.limits = states, contract, contract.source.limits
        self.cancel = CancelSignal(states)
        self.code_id = _digest(export_manifest())
        self.arguments = contract.packet(self.code_id)
        _decode(json.dumps(self.arguments, ensure_ascii=False))
        self.config_id = _digest(self.arguments)
        self.plan = ExecutionPlan(source_id=contract.source.source.source_id, config_id=self.config_id, policy_version=EXPORT_POLICY,
                                  steps=(ExecutionStep("preflight", "run_preflight"), ExecutionStep("ingest", "run_ingest", json.dumps(self.arguments))))

    def _request(self, task_id):
        root, state = self._work_root(task_id), self.states.load()
        return WorkerRequest(task_id=task_id, run_id=state.run_id, actor_id=state.actor_id, stage="ingest", action_id="run_ingest",
                             source_id=self.plan.source_id, config_id=self.config_id, plan_id=self.plan.plan_id, policy_version=EXPORT_POLICY,
                             limits=self.limits, dependency_directories=(), input_files=table_files(root, "export.json", extra=("split.json",)),
                             output_directory=str(root / "output/result"), scratch_directory=str(root / "scratch/task"))

    def _stage(self, request):
        # stage_table의 공통 입력은 계약의 source 속성이다. export는 EDA 계약 안에 생성 출처를 갖는다.
        from types import SimpleNamespace
        raw = _read_file(self.contract.split.artifact.directory / "split.json", 8 * 1024 * 1024)
        proxy = SimpleNamespace(contract=self.contract.source, limits=self.limits, states=self.states, _work_root=self._work_root)
        root = stage_table(proxy, request, self.arguments, "export.json", extras={"split.json": raw})
        if _fingerprint(root / "input/split.json")["sha256"] != self.arguments["split_file_sha256"]:
            raise ValueError("복사한 확정 분할 지문이 다릅니다.")
        return root
