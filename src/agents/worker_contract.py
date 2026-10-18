"""향후 격리 worker의 로컬 요청·종료 결과 계약. 현재 실행 기능은 제공하지 않는다."""

from dataclasses import asdict, dataclass
import hashlib
import json
import re

from .workflow import STAGE_ACTION
from .worker_paths import WorkerPathPolicy
from .artifacts import _safe_relative


WORKER_ACTIONS = frozenset({
    "derive_preparation",
    "apply_preparation",
    "run_ingest",
    "run_audit", "run_eda", "run_pipeline", "propose_split", "commit_split", "prepare_data",
    "validate_contract", "render_report",
})
RESULT_CODES = {
    "succeeded": "WORKER_EXITED_OK", "blocked_input": "WORKER_INPUT_BLOCKED",
    "failed_runtime": "WORKER_FAILED", "cancelled": "WORKER_CANCELLED",
    "timed_out": "WORKER_TIMED_OUT", "resource_exceeded": "WORKER_RESOURCE_EXCEEDED",
}


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_:.\-]{1,256}", value):
        raise ValueError("worker 계약 식별값 형식이 올바르지 않습니다.")


def _decode(payload):
    """IPC에서 과도한 입력·중복 키·비표준 숫자·임의 필드를 조용히 수용하지 않는다."""
    if not isinstance(payload, str) or len(payload.encode("utf-8")) > 64 * 1024:
        raise ValueError("worker 계약 JSON 크기 또는 타입이 올바르지 않습니다.")

    def pairs(items):
        result = {}

        for key, value in items:
            if key in result:
                raise ValueError("worker 계약 JSON에 중복 키가 있습니다.")

            result[key] = value

        return result

    def invalid_constant(value):
        raise ValueError("worker 계약 JSON에 비표준 숫자가 있습니다.")

    value = json.loads(payload, object_pairs_hook=pairs, parse_constant=invalid_constant)

    if not isinstance(value, dict):
        raise ValueError("worker 계약은 JSON 객체여야 합니다.")

    return value


@dataclass(frozen=True)
class WorkerLimits:
    """요청한 제한값이다. 실제 OS 강제 여부를 뜻하지 않는다."""

    wall_seconds: int = 300
    memory_mib: int = 2048
    processes: int = 1

    def __post_init__(self):
        for value, upper in ((self.wall_seconds, 86400), (self.memory_mib, 32768), (self.processes, 8)):
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError("worker 자원 한도는 허용 범위의 양의 정수여야 합니다.")


# ==========================================
# 신뢰하는 로컬 실행기가 만드는 worker 요청
# - 원본 내용 대신 승인된 경로와 식별값만 담는다. 이 객체도 LLM에는 보내지 않는다.
# - 명령·모듈·코드·API 키·자유 인자를 받지 않는다.
# - 허용 루트와 관리자 자원 한도는 요청 외부의 신뢰하는 정책에서 주입한다.
# - OS 격리 검증 및 승인 결합이 없는 현재 단계에서는 실제 실행에 사용하지 않는다.
# ==========================================
@dataclass(frozen=True)
class WorkerRequest:
    task_id: str
    run_id: str
    actor_id: str
    stage: str
    action_id: str
    source_id: str
    config_id: str
    plan_id: str
    policy_version: str
    input_files: tuple[str, ...]
    dependency_directories: tuple[str, ...]
    output_directory: str
    scratch_directory: str
    limits: WorkerLimits
    network_policy: str = "deny_all"
    schema_version: str = "worker_request_v1"

    def __post_init__(self):
        if not isinstance(self.task_id, str) or not re.fullmatch(r"[a-f0-9]{32}", self.task_id):
            raise ValueError("worker 작업 ID는 내부 UUID 형식이어야 합니다.")

        for name in ("run_id", "actor_id", "source_id", "config_id", "plan_id", "policy_version"):
            # OS 계정은 DOMAIN\\user 형식일 수 있으므로 계정 필드만 별도로 검사한다.
            if name == "actor_id":
                value = self.actor_id

                if not isinstance(value, str) or not value.strip() or len(value) > 256 or any(ord(char) < 32 for char in value):
                    raise ValueError("worker 실행 주체 형식이 올바르지 않습니다.")
            else:
                _identifier(getattr(self, name))

        if not isinstance(self.action_id, str) or not isinstance(self.stage, str) or self.action_id not in WORKER_ACTIONS or STAGE_ACTION.get(self.stage) != self.action_id:
            raise ValueError("worker 실행 단계와 등록 작업이 일치하지 않습니다.")

        if self.schema_version != "worker_request_v1" or self.network_policy != "deny_all":
            raise ValueError("지원하지 않는 worker 버전 또는 네트워크 정책입니다.")

        if not isinstance(self.limits, WorkerLimits):
            raise TypeError("worker 자원 한도 계약이 필요합니다.")

        for values in (self.input_files, self.dependency_directories):
            if not isinstance(values, tuple) or len(values) > 32:
                raise ValueError("worker 입력 목록은 중복 없는 제한된 tuple이어야 합니다.")

            if any(not isinstance(value, str) or not value for value in values):
                raise ValueError("worker 입력 경로 형식이 올바르지 않습니다.")

            if len(set(values)) != len(values):
                raise ValueError("worker 입력 경로를 중복 지정할 수 없습니다.")

        if not self.input_files or not all(isinstance(value, str) and value for value in (self.output_directory, self.scratch_directory)):
            raise ValueError("worker 입력 파일과 출력·임시 경로가 필요합니다.")

    def to_json(self):
        """경로를 포함한 로컬 IPC 표현이며 최소 LLM DTO가 아니다."""
        payload = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False, allow_nan=False)

        if len(payload.encode("utf-8")) > 64 * 1024:
            raise ValueError("worker 요청 JSON 크기 상한을 초과했습니다.")

        return payload

    @property
    def request_id(self):
        """경로·자원 한도·정책 변경까지 구분한다. 승인 인증 서명은 아니다."""
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_json(cls, payload):
        value = _decode(payload)

        if set(value) != set(cls.__dataclass_fields__) or not isinstance(value["limits"], dict) or set(value["limits"]) != set(WorkerLimits.__dataclass_fields__):
            raise ValueError("worker 요청 필드가 계약과 일치하지 않습니다.")

        for key in ("input_files", "dependency_directories"):
            if not isinstance(value[key], list):
                raise ValueError("worker 입력 목록은 JSON 배열이어야 합니다.")

            value[key] = tuple(value[key])

        value["limits"] = WorkerLimits(**value["limits"])

        return cls(**value)

    def check_policy(self, paths, maximum):
        """실행 직전 다시 호출할 사전 검사. 파일 생성·권한 부여·실행은 하지 않는다."""
        if not isinstance(paths, WorkerPathPolicy) or not isinstance(maximum, WorkerLimits):
            raise TypeError("신뢰하는 경로 정책과 관리자 자원 상한이 필요합니다.")

        if any(getattr(self.limits, name) > getattr(maximum, name) for name in WorkerLimits.__dataclass_fields__):
            raise ValueError("worker 요청이 관리자 자원 상한을 초과했습니다.")

        files = [paths.check_read(value) for value in self.input_files]
        dependencies = [paths.check_read(value, directory=True) for value in self.dependency_directories]
        output = paths.check_write(self.output_directory)
        scratch = paths.check_write(self.scratch_directory, scratch=True)

        if len(set(files)) != len(files) or len(set(dependencies)) != len(dependencies) or output == scratch:
            raise ValueError("정규화한 worker 경로가 중복됩니다.")

        return True


@dataclass(frozen=True)
class WorkerResult:
    """worker 종료 보고일 뿐이다. 성공이어도 부모 실행기의 산출물 검증이 필요하다."""

    task_id: str
    request_id: str
    status: str
    code: str
    artifact_refs: tuple[str, ...] = ()
    schema_version: str = "worker_result_v1"

    def __post_init__(self):
        if not isinstance(self.task_id, str) or not re.fullmatch(r"[a-f0-9]{32}", self.task_id):
            raise ValueError("worker 결과의 작업 ID가 올바르지 않습니다.")

        if not isinstance(self.request_id, str) or not re.fullmatch(r"[a-f0-9]{64}", self.request_id):
            raise ValueError("worker 결과의 요청 식별값이 올바르지 않습니다.")

        if not isinstance(self.status, str) or not isinstance(self.code, str) or self.schema_version != "worker_result_v1" or RESULT_CODES.get(self.status) != self.code:
            raise ValueError("worker 결과 상태·코드·버전이 일치하지 않습니다.")

        if not isinstance(self.artifact_refs, tuple) or len(self.artifact_refs) > 64:
            raise ValueError("worker 산출물 참조 목록이 올바르지 않습니다.")

        for value in self.artifact_refs:
            if not isinstance(value, str) or len(value) > 512:
                raise ValueError("worker 산출물 참조 길이가 올바르지 않습니다.")

            _safe_relative(value)

        if len(set(self.artifact_refs)) != len(self.artifact_refs) or (self.status != "succeeded" and self.artifact_refs):
            raise ValueError("실패 결과에 완료 산출물 참조를 등록할 수 없습니다.")

    def to_json(self):
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False, allow_nan=False)

    @classmethod
    def from_json(cls, payload):
        value = _decode(payload)

        if set(value) != set(cls.__dataclass_fields__) or not isinstance(value["artifact_refs"], list):
            raise ValueError("worker 결과 필드가 계약과 일치하지 않습니다.")

        value["artifact_refs"] = tuple(value["artifact_refs"])

        return cls(**value)

    def matches(self, request):
        """다른 작업·과거 요청의 결과를 현재 결과로 취급하지 않는다."""
        return isinstance(request, WorkerRequest) and self.task_id == request.task_id and self.request_id == request.request_id
