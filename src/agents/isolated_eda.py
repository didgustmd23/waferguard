"""승인 실행기에 등록하는 합성 EDA worker 도구. 실데이터 입력 경로를 지원하지 않는다."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import struct
import zlib

from .adapters.tabular import REPORT_FILES
from .appcontainer_probe import launch_succeeded
from .artifacts import _fingerprint, _is_link, _root
from .contracts import ToolResult
from .executor import ExecutionPlan, ExecutionStep, RegisteredTool
from .isolated_eda_transport import code_manifest, run_isolated_eda
from .synthetic_eda_data import CSV_BYTES, SOURCE_ID, VERSION, fixture_input
from .worker_contract import WorkerLimits, WorkerRequest, WorkerResult
from .worker_paths import WorkerPathPolicy
from src.preprocessing.eda import collect_eda
from src.preprocessing.eda_report import report_markdown


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _read_file(path, maximum):
    """링크·하드 링크·과대 파일을 거부하고 제한된 바이트만 읽는다."""
    if not path.is_file() or _is_link(path) or path.stat().st_nlink != 1:
        raise ValueError("검증 가능한 일반 산출물 파일이 필요합니다.")

    with path.open("rb") as stream:
        data = stream.read(maximum + 1)

    if not data or len(data) > maximum:
        raise ValueError("산출물 크기 제한을 위반했습니다.")

    return data


def _valid_png(data):
    """이미지를 렌더링하지 않고 고정 크기·PNG 청크·CRC를 검사한다. 안전성 인증은 아니다."""
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return False

    offset, first, image_data = 8, True, False

    while offset + 12 <= len(data):
        size = struct.unpack_from(">I", data, offset)[0]
        end = offset + 12 + size

        if end > len(data):
            return False

        kind = data[offset + 4:offset + 8]
        payload = data[offset + 8:end - 4]
        crc = struct.unpack_from(">I", data, end - 4)[0]

        if zlib.crc32(kind + payload) & 0xFFFFFFFF != crc:
            return False

        if first:
            if kind != b"IHDR" or size != 13:
                return False

            width, height = struct.unpack_from(">II", payload)

            if not 1 <= width <= 4096 or not 1 <= height <= 4096:
                return False

            first = False

        image_data = image_data or kind == b"IDAT"

        if kind == b"IEND":
            return size == 0 and end == len(data) and image_data

        offset = end

    return False


def validate_report(directory):
    """완료 표식만 믿지 않고 모든 CSV·요약·고정 문구·그림 구조를 부모에서 재검증한다."""
    report = _root(directory)

    if {path.name for path in report.iterdir()} != REPORT_FILES:
        return False

    expected = collect_eda(fixture_input())

    for path in report.iterdir():
        data = _read_file(path, 4 * 1024 * 1024)

        if path.suffix == ".csv":
            text = data.decode("utf-8-sig").replace("\r\n", "\n")
            reference = expected[path.stem].to_csv().replace("\r\n", "\n")

            if text != reference:
                return False
        elif path.suffix == ".png":
            if not _valid_png(data):
                return False
        elif path.name == "summary.json":
            if json.loads(data) != expected["summary"]:
                return False
        elif path.name == "SUCCESS.json":
            if json.loads(data) != {"status": "eda_complete"}:
                return False
        elif path.name == "eda.md" and data.decode("utf-8").replace("\r\n", "\n") != report_markdown():
            return False

    return True


class CancelSignal:
    """worker에게 전달하지 않고 부모만 상태와 전용 취소 요청을 확인한다."""

    def __init__(self, states):
        self.states = states

    def is_set(self):
        path = self.states.directory / "cancel.requested"

        if path.exists() or path.is_symlink():
            if _is_link(path) or _read_file(path, 128) != b"cancel_synthetic_eda_v1\n":
                raise ValueError("취소 요청 파일이 올바르지 않습니다.")

            return True

        state = self.states.load()

        return state.cancel_requested or state.status == "cancelled"


class SyntheticEdaTool:
    """기존 승인·예약·커밋 흐름을 사용하는 단일 등록 도구."""

    def __init__(self, states, limits):
        if not isinstance(limits, WorkerLimits):
            raise TypeError("사용자 지정 WorkerLimits가 필요합니다.")

        self.states = states
        self.limits = limits
        self.code_id = _digest(code_manifest())
        self.arguments = {"demo_id": VERSION, "limits": asdict(limits), "code_id": self.code_id}
        self.config_id = _digest(self.arguments)
        arguments = json.dumps(self.arguments, sort_keys=True)
        self.plan = ExecutionPlan(
            source_id=SOURCE_ID, config_id=self.config_id, policy_version=VERSION,
            steps=(ExecutionStep("preflight", "run_preflight"), ExecutionStep("run_eda", "run_eda", arguments)),
        )
        self.cancel = CancelSignal(states)

    def _check_arguments(self, arguments):
        if arguments != self.arguments or _digest(code_manifest()) != self.code_id:
            raise ValueError("승인한 EDA 코드·설정과 현재 실행이 일치하지 않습니다.")

    def _work_root(self, task_id):
        if len(task_id) != 32 or any(char not in "0123456789abcdef" for char in task_id):
            raise ValueError("내부 작업 ID가 필요합니다.")

        base = _root(self.states.directory.parent.parent)

        return base / "workers" / task_id

    def _request(self, task_id):
        root = self._work_root(task_id)
        state = self.states.load()

        return WorkerRequest(
            task_id=task_id, run_id=state.run_id, actor_id=state.actor_id,
            stage="run_eda", action_id="run_eda", source_id=SOURCE_ID,
            config_id=self.config_id, plan_id=self.plan.plan_id, policy_version=VERSION,
            input_files=(str(root / "input" / "synthetic.csv"),), dependency_directories=(),
            output_directory=str(root / "output" / "result"),
            scratch_directory=str(root / "scratch" / "task"), limits=self.limits,
        )

    def _stage(self, request):
        """예약 후에만 새 작업 영역을 만든다. 기존 작업 영역을 덮어쓰지 않는다."""
        root = self._work_root(request.task_id)
        root.parent.mkdir(exist_ok=True)
        _root(root.parent)
        root.mkdir(exist_ok=False)

        for name in ("runtime", "input", "tools", "output", "scratch"):
            (root / name).mkdir()

        Path(request.output_directory).mkdir()
        Path(request.scratch_directory).mkdir()
        (root / "input" / "synthetic.csv").write_bytes(CSV_BYTES)
        (root / "input" / "request.json").write_text(request.to_json(), encoding="utf-8")
        policy = WorkerPathPolicy(
            read_roots=tuple(root / name for name in ("input", "runtime", "tools")),
            output_root=root / "output", scratch_root=root / "scratch",
            protected_roots=(self.states.directory,),
        )
        request.check_policy(policy, self.limits)

        return root

    def execute(self, output, arguments):
        """승인 소비·작업 예약 이후 호출된다. 부모 검증 전에는 완료 파일을 쓰지 않는다."""
        self._check_arguments(arguments)

        if self.cancel.is_set():
            return ToolResult("blocked_input", errors=("WORKER_CANCELLED",))

        request = self._request(output.name)
        root = self._stage(request)
        transport = run_isolated_eda(request, root, self.cancel)
        self._check_arguments(arguments)

        if self.cancel.is_set() or transport.get("launch", {}).get("status") == "cancelled":
            return ToolResult("blocked_input", errors=("WORKER_CANCELLED",))

        launch = transport.get("launch", {})
        resources = launch.get("resources", {})
        job_valid = (
            isinstance(resources, dict) and resources.get("cleanup_confirmed") is True
            and resources.get("active_processes") == 0
            and resources.get("configured_job_memory_bytes") == self.limits.memory_mib * 1024 * 1024
            and resources.get("configured_process_limit") == self.limits.processes
        )

        if transport.get("request_id") != request.request_id or transport.get("profile_cleanup") is not True or not launch_succeeded(launch) or not job_valid:
            code = {"timed_out": "WORKER_TIMED_OUT", "resource_exceeded": "WORKER_RESOURCE_EXCEEDED"}.get(launch.get("status"), "WORKER_FAILED")
            return ToolResult("failed_runtime", errors=(code,))

        staged = _root(Path(request.output_directory))

        if {path.name for path in staged.iterdir()} != {"eda", "worker_result.json"}:
            return ToolResult("blocked_input", errors=("WORKER_INPUT_BLOCKED",))

        reply = WorkerResult.from_json(_read_file(staged / "worker_result.json", 65536).decode("utf-8"))

        if not reply.matches(request) or reply.status != "succeeded" or reply.artifact_refs != ("eda",):
            return ToolResult("blocked_input", errors=("WORKER_INPUT_BLOCKED",))

        if Path(request.input_files[0]).read_bytes() != CSV_BYTES or not validate_report(staged / "eda"):
            return ToolResult("blocked_input", errors=("WORKER_INPUT_BLOCKED",))

        if self.cancel.is_set():
            return ToolResult("blocked_input", errors=("WORKER_CANCELLED",))

        # 검증한 목록만 부모가 복사한다. worker의 완료 표식·임의 파일은 가져오지 않는다.
        shutil.copytree(staged / "eda", output / "eda")
        receipt = {"version": VERSION, "request_id": request.request_id, "task_id": request.task_id, "job_cleanup": True, "profile_cleanup": True}
        (output / "worker_receipt.json").write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")

        return ToolResult("passed", artifact_refs=("eda",))

    def validate(self, output, arguments):
        self._check_arguments(arguments)

        if self.cancel.is_set():
            return False

        names = {path.name for path in _root(output).iterdir()}

        if not {"eda", "worker_receipt.json"}.issubset(names) or names - {"eda", "worker_receipt.json", "AGENT_COMPLETE.json"}:
            return False

        request = self._request(output.name)
        receipt = json.loads(_read_file(output / "worker_receipt.json", 65536))
        expected = {"version": VERSION, "request_id": request.request_id, "task_id": request.task_id, "job_cleanup": True, "profile_cleanup": True}

        return receipt == expected and validate_report(output / "eda")

    def registered_tool(self):
        return RegisteredTool("run_eda", self.execute, self.validate, requires_approval=True)
