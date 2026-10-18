"""기존 snapshot 실행 기반을 재사용하는 승인 필수 비파괴 품질 검사 도구."""

from dataclasses import replace
import json
from pathlib import Path
import shutil

from .appcontainer_probe import launch_succeeded
from .artifacts import _root
from .contracts import ToolResult
from .executor import ExecutionPlan, ExecutionStep, RegisteredTool
from .input_snapshot import PAYLOAD_NAME, create_snapshot
from .isolated_eda import _read_file
from .quality_result_validation import read_quality_report
from .snapshot_eda import SnapshotEdaTool, _digest
from .snapshot_audit_transport import code_manifest, run_snapshot_audit
from .worker_contract import WorkerResult
from .worker_paths import WorkerPathPolicy


POLICY_VERSION = "snapshot_audit_execution_v1"


class SnapshotAuditTool(SnapshotEdaTool):
    """공통 계약·취소·작업 경로를 재사용하고 실행 단계와 결과 판정은 별도로 구현한다."""

    def __init__(self, states, contract):
        super().__init__(states, contract)
        self.code_id = _digest(code_manifest())
        self.arguments["code_id"] = self.code_id
        self.config_id = _digest(self.arguments)
        self.plan = ExecutionPlan(
            source_id=contract.snapshot.source_id, config_id=self.config_id, policy_version=POLICY_VERSION,
            steps=(ExecutionStep("preflight", "run_preflight"), ExecutionStep("audit", "run_audit", json.dumps(self.arguments))),
        )

    def _check_arguments(self, arguments):
        if arguments != self.arguments or self.code_id != _digest(code_manifest()):
            raise ValueError("품질 검사 코드·설정이 승인 대상과 다릅니다.")

        self.contract.check_input()

    def _request(self, task_id):
        # 부모의 작업 ID·주체·출력 경로·자원 결합을 재사용하되 audit 단계로 고정한다.
        base = super()._request(task_id)
        root = self._work_root(task_id)
        return replace(
            base, stage="audit", action_id="run_audit", policy_version=POLICY_VERSION,
            input_files=tuple(str(root / "input" / name) for name in ("snapshot/input.bin", "snapshot/snapshot.json", "audit.json")),
        )

    def _stage(self, request):
        """승인·예약 이후에만 작업용 입력을 만들며 원본 폴더 권한은 부여하지 않는다."""
        root = self._work_root(request.task_id)
        root.parent.mkdir(exist_ok=True)
        _root(root.parent)
        root.mkdir(exist_ok=False)

        for name in ("runtime", "input", "tools", "output", "scratch"):
            (root / name).mkdir()

        staged = create_snapshot(
            Path(self.contract.snapshot_directory) / PAYLOAD_NAME, root / "input" / "snapshot",
            max_input_bytes=self.contract.snapshot.max_input_bytes,
        )

        if staged != self.contract.snapshot:
            raise ValueError("품질 검사 작업용 복사본이 승인된 내용과 다릅니다.")

        (root / "input" / "audit.json").write_text(json.dumps(self.arguments, ensure_ascii=False), encoding="utf-8")
        (root / "input" / "request.json").write_text(request.to_json(), encoding="utf-8")
        Path(request.output_directory).mkdir()
        Path(request.scratch_directory).mkdir()
        paths = WorkerPathPolicy(
            tuple(root / name for name in ("runtime", "input", "tools")), root / "output", root / "scratch",
            protected_roots=(self.states.directory, Path(self.contract.snapshot_directory)),
        )
        request.check_policy(paths, self.limits)
        return root

    def _receipt(self, request):
        receipt = super()._receipt(request)
        receipt["version"] = POLICY_VERSION
        return receipt

    def execute(self, output, arguments):
        """검토·차단 보고서도 검증해 보존하지만 통과 산출물로 커밋하지 않는다."""
        self._check_arguments(arguments)

        if self.cancel.is_set():
            return ToolResult("blocked_input", errors=("WORKER_CANCELLED",))

        request = self._request(output.name)
        root = self._stage(request)
        outcome = run_snapshot_audit(request, root, self.cancel)
        self._check_arguments(arguments)
        launch = outcome.get("launch", {})

        if self.cancel.is_set() or launch.get("status") == "cancelled":
            return ToolResult("blocked_input", errors=("WORKER_CANCELLED",))

        resources = launch.get("resources", {})
        job_valid = (
            isinstance(resources, dict) and resources.get("cleanup_confirmed") is True
            and resources.get("active_processes") == 0
            and resources.get("configured_job_memory_bytes") == self.limits.memory_mib * 1024 * 1024
            and resources.get("configured_process_limit") == self.limits.processes
        )

        if outcome.get("request_id") != request.request_id or outcome.get("profile_cleanup") is not True or not job_valid:
            return ToolResult("failed_runtime", errors=("WORKER_FAILED",))

        staged = _root(request.output_directory)

        if launch.get("status") == "exited" and launch.get("exit_code") == 2 and (staged / "worker_result.json").exists():
            reply = WorkerResult.from_json(_read_file(staged / "worker_result.json", 65536).decode("utf-8"))

            if reply.matches(request) and reply.status == "blocked_input" and not reply.artifact_refs:
                return ToolResult("blocked_input", errors=("WORKER_INPUT_BLOCKED",))

        if not launch_succeeded(launch):
            code = {"timed_out": "WORKER_TIMED_OUT", "resource_exceeded": "WORKER_RESOURCE_EXCEEDED"}.get(launch.get("status"), "WORKER_FAILED")
            return ToolResult("failed_runtime", errors=(code,))

        if {path.name for path in staged.iterdir()} != {"quality", "worker_result.json"}:
            return ToolResult("blocked_input", errors=("WORKER_INPUT_BLOCKED",))

        reply = WorkerResult.from_json(_read_file(staged / "worker_result.json", 65536).decode("utf-8"))
        report = read_quality_report(staged / "quality", self.contract.profile)

        if not reply.matches(request) or reply.status != "succeeded" or reply.artifact_refs != ("quality",) or report is None:
            return ToolResult("blocked_input", errors=("WORKER_INPUT_BLOCKED",))

        if self.cancel.is_set():
            return ToolResult("blocked_input", errors=("WORKER_CANCELLED",))

        shutil.copytree(staged / "quality", output / "quality")
        (output / "worker_receipt.json").write_text(json.dumps(self._receipt(request)), encoding="utf-8")

        # 복사 후에도 검사한다. review/block은 검증된 로컬 보고서만 보존한다.
        if read_quality_report(output / "quality", self.contract.profile) != report:
            return ToolResult("blocked_input", errors=("WORKER_INPUT_BLOCKED",))

        status = report["status"]
        return ToolResult(
            status, artifact_refs=("quality",),
            warnings=("QUALITY_REVIEW_REQUIRED",) if status == "review_required" else (),
            errors=("QUALITY_INPUT_BLOCKED",) if status == "blocked_input" else (),
        )

    def validate(self, output, arguments):
        self._check_arguments(arguments)
        names = {path.name for path in _root(output).iterdir()}

        if self.cancel.is_set() or not {"quality", "worker_receipt.json"}.issubset(names) or names - {"quality", "worker_receipt.json", "AGENT_COMPLETE.json"}:
            return False

        report = read_quality_report(output / "quality", self.contract.profile)
        receipt = json.loads(_read_file(output / "worker_receipt.json", 65536))
        return report is not None and report["status"] == "passed" and receipt == self._receipt(self._request(output.name))

    def registered_tool(self):
        return RegisteredTool("run_audit", self.execute, self.validate, requires_approval=True)
