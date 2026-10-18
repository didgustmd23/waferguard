"""기존 승인·AppContainer 실행 흐름을 재사용하는 원본 보존형 WM 개발 구간 캐시 생성 도구."""

from dataclasses import replace
import json
from pathlib import Path
import shutil

from .appcontainer_probe import launch_succeeded
from .artifacts import _root
from .contracts import ToolResult
from .executor import ExecutionPlan, ExecutionStep, RegisteredTool
from functools import partial
from src.preprocessing.canonical_data import decode_json
from .isolated_eda import _read_file
from .wm_canonical import WmCanonicalTool
from .wm_canonical_contract import digest
from .wm_cache_evidence import check_transform_evidence
from .wm_cache_results import POLICY_VERSION, inspect_cache
from .wm_cache_transport import code_manifest, run_cache_worker
from .worker_contract import WorkerRequest, WorkerResult, _decode


class WmCacheTool(WmCanonicalTool):
    """확정 분할과 변환 설정을 결합하고 개발 구간 검사만 격리 실행한다."""

    def __init__(self, states, contract, check_run, evidence):
        super().__init__(states, contract)
        self.check_run = check_run
        self.evidence = evidence
        self.code_id = digest(code_manifest())
        self.arguments = {**contract.packet(self.code_id), "version": POLICY_VERSION, "cache_evidence": evidence}
        self.config_id = digest(self.arguments)
        self.plan = ExecutionPlan(contract.source_id, self.config_id, POLICY_VERSION, (
            ExecutionStep("preflight", "run_preflight"),
            ExecutionStep("run_eda", "run_eda", json.dumps(self.arguments)),
            ExecutionStep("review_eda", "review_eda", json.dumps(self.arguments)),
            ExecutionStep("propose_split", "propose_split", json.dumps(self.arguments)),
            ExecutionStep("commit_split", "commit_split", json.dumps(self.arguments)),
            ExecutionStep("prepare_data", "prepare_data", json.dumps(self.arguments)),
        ))

    def _reference_record(self, stage):
        return {"stage": stage, "evidence_id": digest(self.evidence), "reference_only": True, "recomputed": False}

    def reference_execute(self, stage, output, arguments):
        """이미 완료된 EDA·분할만 연결한다. 새 분할·변환·캐시는 만들지 않는다."""
        self._check_arguments(arguments)

        if self.cancel.is_set():
            return ToolResult("blocked_input", errors=("WORKER_CANCELLED",))

        (output / "reference.json").write_text(json.dumps(self._reference_record(stage)), encoding="utf-8")
        return ToolResult("passed", artifact_refs=("reference.json",))

    def reference_validate(self, stage, output, arguments):
        self._check_arguments(arguments)
        return (not self.cancel.is_set() and
                _decode(_read_file(output / "reference.json", 65536).decode()) == self._reference_record(stage))

    def reference_tools(self):
        return tuple(RegisteredTool(stage, partial(self.reference_execute, stage),
                                    partial(self.reference_validate, stage), requires_approval=stage in {"review_eda", "commit_split"})
                     for stage in ("run_eda", "review_eda", "propose_split", "commit_split"))

    def _check_arguments(self, arguments):
        if arguments != self.arguments or self.code_id != digest(code_manifest()):
            raise ValueError("WM 개발 구간 캐시 생성의 코드·설정이 승인 범위와 다릅니다.")

        previous = check_transform_evidence(self.check_run, self.evidence)

        if previous.source_id != self.contract.source_id or previous.max_group_ratio != self.contract.max_group_ratio:
            raise ValueError("캐시 생성과 확정 분할의 입력·검토 상한이 다릅니다.")

        self.contract.check()

    def _request(self, task_id):
        request = super()._request(task_id)
        return replace(request, stage="prepare_data", action_id="prepare_data", policy_version=POLICY_VERSION,
                       dependency_directories=request.dependency_directories +
                       (str(self._work_root(task_id) / "input/split"), str(self._work_root(task_id) / "input/check")))


    def _stage(self, request):
        """공통 canonical 복사에 검증된 확정 분할 복사본을 추가한다."""
        base_request = replace(request, dependency_directories=request.dependency_directories[:1])
        root = super()._stage(base_request)
        from .wm_split_commit_results import split_files
        from .wm_transform_results import check_files
        from .worker_paths import WorkerPathPolicy

        split = self.evidence["split_evidence"]
        protected = [self.states.directory, Path(self.contract.snapshot_directory)]

        for kind, source_path, records, verifier in (
            ("split", split["split_directory"], split["split_files"], split_files),
            ("check", self.evidence["check_directory"], self.evidence["check_files"], check_files),
        ):
            source = _root(source_path)
            protected.append(source)
            target = root / "input" / kind
            target.mkdir()

            for name in records:
                shutil.copyfile(source / name, target / name)

            if verifier(target, sum(value["size"] for value in records.values())) != records:
                raise ValueError("캐시 입력 복사본이 승인한 지문과 다릅니다.")

        (root / "input/request.json").write_text(request.to_json(), encoding="utf-8")
        request.check_policy(WorkerPathPolicy(
            tuple(root / name for name in ("runtime", "input", "tools")), root / "output", root / "scratch",
            protected_roots=tuple(protected),
        ), self.limits)
        return root

    def execute(self, output, arguments):
        """성공 응답 외에도 종료·입력·파일 계약을 확인한 결과만 가져온다."""
        self._check_arguments(arguments)

        if self.cancel.is_set():
            return ToolResult("blocked_input", errors=("WORKER_CANCELLED",))

        request = self._request(output.name)
        outcome = run_cache_worker(request, self._stage(request), self.cancel)
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

        if (not reply.matches(request) or reply.status != "succeeded" or reply.artifact_refs != ("cache",)
            or {path.name for path in staged.iterdir()} != {"cache", "worker_result.json"}):
            return ToolResult("blocked_input", errors=("WORKER_INPUT_BLOCKED",))

        report = inspect_cache(staged / "cache", request, self.contract, digest(self.evidence), self.evidence)
        shutil.copytree(staged / "cache", output / "cache")

        if inspect_cache(output / "cache", request, self.contract, digest(self.evidence), self.evidence) != report:
            return ToolResult("blocked_input", errors=("WORKER_INPUT_BLOCKED",))

        (output / "receipt.json").write_text(json.dumps({"request_id": request.request_id, "report": report}), encoding="utf-8")

        if self.cancel.is_set():
            return ToolResult("blocked_input", errors=("WORKER_CANCELLED",))

        return ToolResult("passed",
                          artifact_refs=("cache", "receipt.json"))

    def inspect_attempt(self, output):
        """완료·검토 보고서 조회 시에도 요청·종료 증거·해시를 다시 대조한다."""
        self._check_arguments(self.arguments)
        output = _root(output)

        if {path.name for path in output.iterdir()} not in (
            {"cache", "receipt.json"}, {"cache", "receipt.json", "AGENT_COMPLETE.json"},
        ):
            raise ValueError("WM 개발 구간 캐시 생성 시도 결과 파일 목록이 올바르지 않습니다.")

        request = self._request(output.name)
        work = self._work_root(output.name)
        saved = WorkerRequest.from_json(_read_file(work / "input/request.json", 65536).decode("utf-8"))
        outcome = _decode(_read_file(work / "transport.json", 65536).decode("utf-8"))
        reply = WorkerResult.from_json(_read_file(Path(request.output_directory) / "worker_result.json", 65536).decode("utf-8"))

        if (saved != request or not self._evidence(request, outcome) or not launch_succeeded(outcome.get("launch", {}))
            or not reply.matches(request) or reply.status != "succeeded" or reply.artifact_refs != ("cache",)):
            raise ValueError("WM 개발 구간 캐시 생성 worker 종료 증거가 현재 작업과 다릅니다.")

        report = inspect_cache(output / "cache", request, self.contract, digest(self.evidence), self.evidence)
        receipt = decode_json(_read_file(output / "receipt.json", 1024 * 1024))

        if receipt != {"request_id": request.request_id, "report": report}:
            raise ValueError("WM 개발 구간 캐시 생성 결과 영수증이 검증된 내용과 다릅니다.")

        return report


    def validate(self, output, arguments):
        self._check_arguments(arguments)
        return not self.cancel.is_set() and self.inspect_attempt(output)["status"] == "cache_complete"

    def registered_tool(self):
        return RegisteredTool("prepare_data", self.execute, self.validate, requires_approval=True)
