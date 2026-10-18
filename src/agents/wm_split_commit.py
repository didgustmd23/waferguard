"""기존 승인·AppContainer 실행 흐름을 재사용하는 원본 보존형 WM 그룹 분할 확정 도구."""

from dataclasses import replace
from functools import partial
import json
from pathlib import Path
import shutil

from .appcontainer_probe import launch_succeeded
from .artifacts import _root
from .contracts import ToolResult
from .executor import ExecutionPlan, ExecutionStep, RegisteredTool
from .isolated_eda import _read_file
from .wm_canonical import WmCanonicalTool
from .wm_canonical_contract import digest
from .wm_split_commit_evidence import check_proposal_evidence
from .wm_split_commit_results import POLICY_VERSION, inspect_split
from .wm_split_commit_transport import code_manifest, run_commit_worker
from .worker_contract import WorkerRequest, WorkerResult, _decode


class WmSplitCommitTool(WmCanonicalTool):
    """파일 복사·Job 증거 확인을 재사용하고 확정 전 분할안의 결과 계약을 검사한다."""

    def __init__(self, states, contract, proposal_run, evidence):
        super().__init__(states, contract)
        self.proposal_run = proposal_run
        self.evidence = evidence
        self.code_id = digest(code_manifest())
        self.arguments = {**contract.packet(self.code_id), "version": POLICY_VERSION, "commit_evidence": evidence}
        self.config_id = digest(self.arguments)
        self.plan = ExecutionPlan(contract.source_id, self.config_id, POLICY_VERSION, (
            ExecutionStep("preflight", "run_preflight"),
            ExecutionStep("run_eda", "run_eda", json.dumps(self.arguments)),
            ExecutionStep("review_eda", "review_eda", json.dumps(self.arguments)),
            ExecutionStep("propose_split", "propose_split", json.dumps(self.arguments)),
            ExecutionStep("commit_split", "commit_split", json.dumps(self.arguments)),
        ))

    def _reference_record(self, stage):
        return {"stage": stage, "evidence_id": digest(self.evidence), "reference_only": True,
                "recomputed": False, "split_complete": False}

    def reference_execute(self, stage, output, arguments):
        """완료된 EDA·제안을 증거로 연결한다. 새 분석·탐색·확정은 하지 않는다."""
        self._check_arguments(arguments)

        if self.cancel.is_set():
            return ToolResult("blocked_input", errors=("WORKER_CANCELLED",))

        (output / "reference.json").write_text(json.dumps(self._reference_record(stage)), encoding="utf-8")
        return ToolResult("passed", artifact_refs=("reference.json",))

    def reference_validate(self, stage, output, arguments):
        self._check_arguments(arguments)
        return (not self.cancel.is_set()
                and _decode(_read_file(output / "reference.json", 65536).decode("utf-8")) == self._reference_record(stage))

    def reference_tools(self):
        return tuple(RegisteredTool(stage, partial(self.reference_execute, stage),
                                    partial(self.reference_validate, stage), requires_approval=stage == "review_eda")
                     for stage in ("run_eda", "review_eda", "propose_split"))

    def _check_arguments(self, arguments):
        if arguments != self.arguments or self.code_id != digest(code_manifest()):
            raise ValueError("WM 그룹 분할 확정의 코드·설정이 승인 범위와 다릅니다.")

        previous = check_proposal_evidence(self.proposal_run, self.evidence)

        if previous.source_id != self.contract.source_id or previous.max_group_ratio != self.contract.max_group_ratio:
            raise ValueError("WM 그룹 분할 확정과 완료 EDA의 입력·검토 상한이 다릅니다.")

        self.contract.check()

    def _request(self, task_id):
        request = super()._request(task_id)
        return replace(request, stage="commit_split", action_id="commit_split", policy_version=POLICY_VERSION,
                       dependency_directories=request.dependency_directories +
                       (str(self._work_root(task_id) / "input/proposal"),))


    def _stage(self, request):
        """공통 canonical 복사에 검증된 분할 제안 복사본을 추가한다."""
        base_request = replace(request, dependency_directories=request.dependency_directories[:1])
        root = super()._stage(base_request)
        source = _root(self.evidence["proposal_directory"])
        target = root / "input/proposal"
        target.mkdir()

        for name in self.evidence["proposal_files"]:
            shutil.copyfile(source / name, target / name)

        from .wm_split_results import proposal_files
        from .worker_paths import WorkerPathPolicy

        if proposal_files(target, self.contract.max_output_bytes) != self.evidence["proposal_files"]:
            raise ValueError("확정용 제안 복사본이 검토한 지문과 다릅니다.")

        (root / "input/request.json").write_text(request.to_json(), encoding="utf-8")
        request.check_policy(WorkerPathPolicy(
            tuple(root / name for name in ("runtime", "input", "tools")), root / "output", root / "scratch",
            protected_roots=(self.states.directory, Path(self.contract.snapshot_directory), source),
        ), self.limits)
        return root

    def execute(self, output, arguments):
        """성공 응답 외에도 종료·입력·파일 계약을 확인한 결과만 가져온다."""
        self._check_arguments(arguments)

        if self.cancel.is_set():
            return ToolResult("blocked_input", errors=("WORKER_CANCELLED",))

        request = self._request(output.name)
        outcome = run_commit_worker(request, self._stage(request), self.cancel)
        self._check_arguments(arguments)
        launch = outcome.get("launch", {})
        self.worker_started = launch.get("status") in {"exited", "timed_out", "resource_exceeded"}

        if self.cancel.is_set() or launch.get("status") == "cancelled":
            return ToolResult("blocked_input", errors=("WORKER_CANCELLED",))

        if not self._evidence(request, outcome):
            return ToolResult("failed_runtime", errors=("WORKER_FAILED",))

        reply_path = Path(request.output_directory) / "worker_result.json"

        if launch.get("status") == "exited" and launch.get("exit_code") == 2 and reply_path.is_file():
            reply = WorkerResult.from_json(_read_file(reply_path, 65536).decode("utf-8"))

            if reply.matches(request) and reply.status == "blocked_input" and not reply.artifact_refs:
                return ToolResult("blocked_input", errors=("WORKER_INPUT_BLOCKED",))

        if not launch_succeeded(launch):
            code = {"timed_out": "WORKER_TIMED_OUT", "resource_exceeded": "WORKER_RESOURCE_EXCEEDED"}.get(
                launch.get("status"), "WORKER_FAILED")
            return ToolResult("failed_runtime", errors=(code,))

        staged = _root(request.output_directory)
        reply = WorkerResult.from_json(_read_file(staged / "worker_result.json", 65536).decode("utf-8"))

        if (not reply.matches(request) or reply.status != "succeeded" or reply.artifact_refs != ("split",)
            or {path.name for path in staged.iterdir()} != {"split", "worker_result.json"}):
            return ToolResult("blocked_input", errors=("WORKER_INPUT_BLOCKED",))

        report = inspect_split(staged / "split", request, self.contract, digest(self.evidence), self.evidence)
        shutil.copytree(staged / "split", output / "split")

        if inspect_split(output / "split", request, self.contract, digest(self.evidence), self.evidence) != report:
            return ToolResult("blocked_input", errors=("WORKER_INPUT_BLOCKED",))

        (output / "receipt.json").write_text(json.dumps({"request_id": request.request_id, "report": report}), encoding="utf-8")

        if self.cancel.is_set():
            return ToolResult("blocked_input", errors=("WORKER_CANCELLED",))

        return ToolResult("passed", artifact_refs=("split", "receipt.json"))

    def inspect_attempt(self, output):
        """완료·검토 보고서 조회 시에도 요청·종료 증거·해시를 다시 대조한다."""
        self._check_arguments(self.arguments)
        output = _root(output)

        if {path.name for path in output.iterdir()} not in (
            {"split", "receipt.json"}, {"split", "receipt.json", "AGENT_COMPLETE.json"},
        ):
            raise ValueError("WM 그룹 분할 확정 시도 결과 파일 목록이 올바르지 않습니다.")

        request = self._request(output.name)
        work = self._work_root(output.name)
        saved = WorkerRequest.from_json(_read_file(work / "input/request.json", 65536).decode("utf-8"))
        outcome = _decode(_read_file(work / "transport.json", 65536).decode("utf-8"))
        reply = WorkerResult.from_json(_read_file(Path(request.output_directory) / "worker_result.json", 65536).decode("utf-8"))

        if (saved != request or not self._evidence(request, outcome) or not launch_succeeded(outcome.get("launch", {}))
            or not reply.matches(request) or reply.status != "succeeded" or reply.artifact_refs != ("split",)):
            raise ValueError("WM 그룹 분할 확정 worker 종료 증거가 현재 작업과 다릅니다.")

        report = inspect_split(output / "split", request, self.contract, digest(self.evidence), self.evidence)
        receipt = _decode(_read_file(output / "receipt.json", 65536).decode("utf-8"))

        if receipt != {"request_id": request.request_id, "report": report}:
            raise ValueError("WM 그룹 분할 확정 결과 영수증이 검증된 내용과 다릅니다.")

        return report


    def validate(self, output, arguments):
        self._check_arguments(arguments)
        return not self.cancel.is_set() and self.inspect_attempt(output)["status"] == "split_complete"

    def registered_tool(self):
        return RegisteredTool("commit_split", self.execute, self.validate, requires_approval=True)
