"""격리 검사·EDA·사람 검토·분할 제안·승인 확정을 하나의 고정 계획으로 연결한다."""

from dataclasses import asdict
import json
from pathlib import Path
import shutil

from .appcontainer_probe import launch_succeeded
from .artifacts import _root, _fingerprint
from .contracts import ToolResult
from .executor import ExecutionPlan, ExecutionStep, RegisteredTool
from .input_snapshot import create_snapshot
from .isolated_eda import CancelSignal, _read_file
from .snapshot_eda import SnapshotEdaTool, _digest
from .snapshot_split_transport import code_manifest, run_split_worker
from .split_evidence import read_split_files
from .quality_result_validation import read_quality_report
from .eda_result_validation import validate_eda_report
from .worker_contract import WorkerRequest, WorkerResult, _decode
from .worker_paths import WorkerPathPolicy


POLICY_VERSION = "snapshot_split_v1"
ARTIFACTS = {"audit": "quality", "run_eda": "eda", "propose_split": "proposal", "commit_split": "split"}


class SnapshotSplitTool(SnapshotEdaTool):
    """부모는 snapshot 바이트와 결과 계약만 검사한다. 원본 분석은 worker만 수행한다."""

    def __init__(self, states, contract):
        self.states = states
        self.contract = contract
        self.limits = contract.limits
        self.cancel = CancelSignal(states)

        if contract.profile.workflow_id != "split_review_v1" or contract.profile.split_rules is None:
            raise ValueError("분할 검토 프로파일이 필요합니다.")

        contract.check_input()
        self.code_id = _digest(code_manifest())
        self.arguments = {"snapshot": asdict(contract.snapshot), "profile": contract.profile.to_dict(),
                          "limits": asdict(self.limits), "code_id": self.code_id, "proposal_sha256": None}
        _decode(json.dumps(self.arguments, ensure_ascii=False))
        # 기존 확정 분할 계약은 profile.signature를 config_id로 사용한다.
        # 소스 코드·한도·snapshot의 결합은 단계 인자와 plan_id에 별도로 포함한다.
        self.plan = ExecutionPlan(
            source_id="csv:" + contract.snapshot.sha256, config_id=contract.profile.signature,
            policy_version=POLICY_VERSION,
            steps=tuple(ExecutionStep(stage, action, json.dumps(self.arguments) if stage != "preflight" else "{}")
                        for stage, action in (("preflight", "run_preflight"), ("audit", "run_audit"),
                                              ("run_eda", "run_eda"), ("review_eda", "review_eda"),
                                              ("propose_split", "propose_split"), ("commit_split", "commit_split"))),
        )

    def _check_arguments(self, arguments):
        if arguments != self.arguments or self.code_id != _digest(code_manifest()):
            raise ValueError("분할 코드·설정이 승인 대상과 다릅니다.")

        self.contract.check_input()

    def _packet(self, dependency):
        packet = dict(self.arguments)

        if dependency is not None:
            packet["proposal_sha256"] = _fingerprint(_root(dependency) / "proposal.json")["sha256"]

        return packet

    def _request_for(self, task_id, stage, dependency=None):
        root = self._work_root(task_id)
        state = self.states.load()
        files = ["snapshot/input.bin", "snapshot/snapshot.json", "split_task.json"]

        if stage == "commit_split":
            if dependency is None:
                raise ValueError("확정에는 실행기가 검증한 선행 분할 제안이 필요합니다.")

            files.append("preview.json")

        return WorkerRequest(
            task_id=task_id, run_id=state.run_id, actor_id=state.actor_id, stage=stage,
            action_id={"audit": "run_audit", "run_eda": "run_eda", "propose_split": "propose_split", "commit_split": "commit_split"}[stage],
            source_id=self.plan.source_id, config_id=_digest(self._packet(dependency)), plan_id=self.plan.plan_id,
            policy_version=POLICY_VERSION, limits=self.limits,
            input_files=tuple(str(root / "input" / name) for name in files), dependency_directories=(),
            output_directory=str(root / "output/result"), scratch_directory=str(root / "scratch/task"),
        )

    def _stage_for(self, request, dependency):
        """승인·예약 뒤 복사한다. 확정 worker에는 제안 JSON만 전달하며 DB는 전달하지 않는다."""
        root = self._work_root(request.task_id)
        root.parent.mkdir(exist_ok=True)
        _root(root.parent)
        root.mkdir()

        for name in ("runtime", "input", "tools", "output", "scratch"):
            (root / name).mkdir()

        staged = create_snapshot(Path(self.contract.snapshot_directory) / "input.bin", root / "input/snapshot",
                                 max_input_bytes=self.contract.snapshot.max_input_bytes)

        if staged != self.contract.snapshot:
            raise ValueError("작업용 복사본이 승인된 내용과 다릅니다.")

        if dependency is not None:
            shutil.copyfile(dependency / "proposal.json", root / "input/preview.json")

        (root / "input/split_task.json").write_text(json.dumps(self._packet(dependency), ensure_ascii=False), encoding="utf-8")
        (root / "input/request.json").write_text(request.to_json(), encoding="utf-8")
        Path(request.output_directory).mkdir()
        Path(request.scratch_directory).mkdir()
        request.check_policy(WorkerPathPolicy(
            tuple(root / name for name in ("runtime", "input", "tools")), root / "output", root / "scratch",
            protected_roots=(self.states.directory, Path(self.contract.snapshot_directory)),
        ), self.limits)

        return root

    def _evidence(self, request, outcome):
        """프로세스 성공과 Job·프로필 정리를 별도로 확인한다."""
        launch = outcome.get("launch", {})
        resources = launch.get("resources", {})

        return (
            outcome.get("request_id") == request.request_id and outcome.get("profile_cleanup") is True
            and isinstance(resources, dict) and resources.get("cleanup_confirmed") is True
            and resources.get("active_processes") == 0
            and resources.get("configured_job_memory_bytes") == self.limits.memory_mib * 1024 * 1024
            and resources.get("configured_process_limit") == self.limits.processes
        )

    def _report_status(self, directory, stage):
        """상세 통계는 로컬에 남기고 고정 상태만 실행기로 반환한다."""
        if stage == "audit":
            report = read_quality_report(directory, self.contract.profile)
            return None if report is None else report["status"]

        if stage == "run_eda":
            return "passed" if validate_eda_report(directory, self.contract.profile) else None

        try:
            report = read_split_files(directory, self.plan.source_id, self.plan.config_id,
                                      preview=stage == "propose_split", allow_review=True)
            if report["rules"] != asdict(self.contract.profile.split_rules):
                return None

            return report["status"]
        except (ValueError, TypeError, OSError, KeyError):
            return None

    def execute_stage(self, stage, output, arguments, dependency=None):
        self._check_arguments(arguments)

        if self.cancel.is_set():
            return ToolResult("blocked_input", errors=("WORKER_CANCELLED",))

        request = self._request_for(output.name, stage, dependency)
        outcome = run_split_worker(request, self._stage_for(request, dependency), self.cancel)
        self._check_arguments(arguments)
        launch = outcome.get("launch", {})

        if self.cancel.is_set() or launch.get("status") == "cancelled":
            return ToolResult("blocked_input", errors=("WORKER_CANCELLED",))

        if not self._evidence(request, outcome):
            return ToolResult("failed_runtime", errors=("WORKER_FAILED",))

        staged = _root(request.output_directory)
        reply_path = staged / "worker_result.json"

        if launch.get("status") == "exited" and launch.get("exit_code") == 2 and reply_path.exists():
            reply = WorkerResult.from_json(_read_file(reply_path, 65536).decode("utf-8"))

            if reply.matches(request) and reply.status == "blocked_input" and not reply.artifact_refs:
                return ToolResult("blocked_input", errors=("WORKER_INPUT_BLOCKED",))

        if not launch_succeeded(launch):
            code = {"timed_out": "WORKER_TIMED_OUT", "resource_exceeded": "WORKER_RESOURCE_EXCEEDED"}.get(launch.get("status"), "WORKER_FAILED")
            return ToolResult("failed_runtime", errors=(code,))

        artifact = ARTIFACTS[stage]

        if {path.name for path in staged.iterdir()} != {artifact, "worker_result.json"}:
            return ToolResult("blocked_input", errors=("WORKER_INPUT_BLOCKED",))

        reply = WorkerResult.from_json(_read_file(reply_path, 65536).decode("utf-8"))
        status = self._report_status(staged / artifact, stage)

        if not reply.matches(request) or reply.status != "succeeded" or reply.artifact_refs != (artifact,) or status is None:
            return ToolResult("blocked_input", errors=("WORKER_INPUT_BLOCKED",))

        if stage in {"propose_split", "commit_split"}:
            filenames = ("split.json", "assignments.csv") if stage == "commit_split" else ("proposal.json", "proposal.md")

            for name in filenames:
                shutil.copyfile(staged / artifact / name, output / name)
        else:
            shutil.copytree(staged / artifact, output / artifact)

        if self.cancel.is_set():
            return ToolResult("blocked_input", errors=("WORKER_CANCELLED",))

        refs = filenames if stage in {"propose_split", "commit_split"} else (artifact,)
        return ToolResult(status, artifact_refs=refs)

    def validate_stage(self, stage, output, arguments, dependency=None):
        """커밋 전에 요청·운송·결과를 다시 확인한다. 원본 CSV를 재파싱하지 않는다."""
        self._check_arguments(arguments)

        if self.cancel.is_set():
            return False

        request = self._request_for(output.name, stage, dependency)
        root = self._work_root(output.name)
        saved = WorkerRequest.from_json(_read_file(root / "input/request.json", 65536).decode("utf-8"))
        outcome = _decode(_read_file(root / "transport.json", 65536).decode("utf-8"))
        reply = WorkerResult.from_json(_read_file(Path(request.output_directory) / "worker_result.json", 65536).decode("utf-8"))

        if saved != request or not self._evidence(request, outcome) or not launch_succeeded(outcome.get("launch", {})) or not reply.matches(request) or reply.status != "succeeded" or reply.artifact_refs != (ARTIFACTS[stage],):
            return False

        if stage == "commit_split":
            proposal = read_split_files(output, self.plan.source_id, self.plan.config_id,
                                       committed=(output / "AGENT_COMPLETE.json").exists())
            expected = read_split_files(dependency, self.plan.source_id, self.plan.config_id, preview=True, committed=True)
            return proposal == expected

        if stage == "propose_split":
            report = read_split_files(output, self.plan.source_id, self.plan.config_id, preview=True,
                                      committed=(output / "AGENT_COMPLETE.json").exists())
            return report["status"] == "passed" and report["rules"] == asdict(self.contract.profile.split_rules)

        names = {path.name for path in _root(output).iterdir()}

        return names <= {ARTIFACTS[stage], "AGENT_COMPLETE.json"} and self._report_status(output / ARTIFACTS[stage], stage) == "passed"

    def review(self, output, arguments):
        """사용자가 로컬 EDA를 확인하고 별도 승인한 사실만 기록한다."""
        self._check_arguments(arguments)
        (output / "review.json").write_text(json.dumps({"version": POLICY_VERSION, "reviewed": True}), encoding="utf-8")
        return ToolResult("passed", artifact_refs=("review.json",))

    def validate_review(self, output, arguments):
        self._check_arguments(arguments)
        return not self.cancel.is_set() and _decode(_read_file(output / "review.json", 4096).decode()) == {"version": POLICY_VERSION, "reviewed": True}

    def registered_tools(self):
        """승인은 분석 실행·검토·제안·확정 모두 명시적으로 받는다."""
        tools = []

        for stage in ARTIFACTS:
            # 기본 인자로 현재 stage를 고정하여 반복문의 마지막 값으로 바뀌지 않게 한다.
            def execute(output, arguments, dependency=None, selected=stage):
                return self.execute_stage(selected, output, arguments, dependency)

            def validate(output, arguments, dependency=None, selected=stage):
                return self.validate_stage(selected, output, arguments, dependency)

            action = "run_audit" if stage == "audit" else stage
            tools.append(RegisteredTool(action, execute, validate, requires_approval=True,
                                        dependency_stage="propose_split" if stage == "commit_split" else None))

        tools.append(RegisteredTool("review_eda", self.review, self.validate_review, requires_approval=True))
        return tuple(tools)
