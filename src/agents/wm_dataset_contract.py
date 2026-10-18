"""완료된 개발 캐시와 CPU 첫 배치 검사 범위를 결합하는 계약."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .wm_canonical_contract import digest
from .wm_cache_results import cache_files
from .worker_contract import WorkerLimits


POLICY_VERSION = 'wm_dataset_check_v1'


def read_cache_evidence(name):
    """완료 승인·OS 종료 증거·행 연결을 부모에서 확인한다. 배열을 LLM으로 보내지 않는다."""
    # 호스트 전용 승인 DB 조회는 worker의 import 경로로 끌어들이지 않는다.
    from .wm_cache_cli import executor, open_run
    from .wm_cache_evidence import committed_cache
    from .isolated_eda import _read_file
    from src.preprocessing.canonical_data import decode_json

    tool = open_run(name)
    directory = committed_cache(tool, executor(tool))
    checked = tool.inspect_attempt(directory)
    root = directory / 'cache'
    report = decode_json(_read_file(root / 'cache.json', 1024 * 1024))
    evidence = {
        'cache_run': name, 'run_id': tool.states.load().run_id, 'plan_id': tool.plan.plan_id,
        'cache_directory': str(root), 'cache_files': checked['files'], 'cache_report': report,
    }
    evidence['evidence_id'] = digest(evidence)
    return evidence


@dataclass(frozen=True)
class WmDatasetInput:
    cache_directory: str
    files_json: str
    limits: WorkerLimits
    max_input_bytes: int
    max_output_bytes: int
    max_runtime_bytes: int
    runtime_sha256: str
    torch_version: str
    batch_size: int = 32

    def __post_init__(self):
        if not isinstance(self.limits, WorkerLimits) or not Path(self.cache_directory).is_absolute():
            raise ValueError('캐시 절대 경로와 worker 자원 계약이 필요합니다.')

        if any(type(value) is not int or not 1 <= value <= 2 ** 63 - 1 for value in
               (self.max_input_bytes, self.max_output_bytes, self.max_runtime_bytes)):
            raise ValueError('입력·출력·런타임 복사 상한은 양의 정수여야 합니다.')

        if (type(self.batch_size) is not int or not 1 <= self.batch_size <= 256 or
            type(self.runtime_sha256) is not str or len(self.runtime_sha256) != 64 or
            any(char not in '0123456789abcdef' for char in self.runtime_sha256) or
            type(self.torch_version) is not str or not 1 <= len(self.torch_version) <= 80):
            raise ValueError('배치 크기·런타임 명세 해시·PyTorch 버전이 올바르지 않습니다.')

        if type(self.files) is not dict:
            raise ValueError('캐시 파일 명세가 필요합니다.')

    @property
    def files(self):
        from src.preprocessing.canonical_data import decode_json

        return decode_json(self.files_json.encode('utf-8'))

    @property
    def source_id(self):
        return 'wm_cache:' + digest(self.files)

    def check(self):
        if cache_files(self.cache_directory, self.max_input_bytes) != self.files:
            raise ValueError('완료된 캐시가 승인한 파일 목록·지문과 다릅니다.')

    def packet(self, code_id, evidence):
        return {
            'version': POLICY_VERSION, 'source_id': self.source_id, 'code_id': code_id,
            'limits': asdict(self.limits), 'max_input_bytes': self.max_input_bytes,
            'max_output_bytes': self.max_output_bytes, 'max_runtime_bytes': self.max_runtime_bytes,
            'runtime_sha256': self.runtime_sha256, 'torch_version': self.torch_version,
            'batch_size': self.batch_size, 'cache_report': evidence['cache_report'],
            'cache_evidence_id': digest(evidence),
        }

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, record):
        if type(record) is not dict or set(record) != set(cls.__dataclass_fields__):
            raise ValueError('Dataset 입력 선택 기록이 올바르지 않습니다.')

        return cls(**{**record, 'limits': WorkerLimits(**record['limits'])})


def committed_dataset(tool, manager):
    """Dataset 검사 단계의 승인 소비·예약·완료 표식을 전용으로 확인한다.

    기존 분할 증거 reader의 허용 작업을 전역 확장하지 않는다. 이 함수는
    신뢰하는 Dataset 검사 실행기가 생성한 validate_contract 범위만 받는다.
    """
    from contextlib import closing
    from dataclasses import asdict
    import json
    import re
    import sqlite3

    from .artifacts import ArtifactContext, _fingerprint, check_artifacts

    state = tool.states.load()
    scope = manager.scope_for('validate_contract')
    state_path = tool.states.directory / 'state.json'
    before = _fingerprint(state_path)

    if (state.status != 'completed' or state.cancel_requested or state.blockers
        or scope.actor_id != state.actor_id or scope.run_id != state.run_id):
        raise ValueError('완료된 Dataset 검사 작업이 필요합니다.')

    scope_json = json.dumps(asdict(scope), ensure_ascii=False, sort_keys=True)
    database = tool.states.directory / 'approvals.sqlite3'

    with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)) as connection:
        connection.execute('BEGIN')
        task = connection.execute('SELECT task_id,status FROM task_runs WHERE scope_json=?', (scope_json,)).fetchone()
        approvals = connection.execute(
            'SELECT r.reservation_id,a.status FROM reservations r JOIN approvals a ON a.approval_id=r.approval_id '
            'WHERE r.scope_json=? AND a.scope_json=r.scope_json', (scope_json,),
        ).fetchall()

    if (task is None or task[1] != 'committed' or approvals != [(task[0], 'consumed')]
        or re.fullmatch('[a-f0-9]{32}', task[0]) is None):
        raise ValueError('Dataset 검사의 승인 소비·커밋 기록이 일치하지 않습니다.')

    output = tool.states.directory / 'attempts' / task[0]
    context = ArtifactContext(scope.source_id, scope.config_id, scope.plan_id, scope.policy_version, scope.tool_id)

    if not check_artifacts(output, context).reusable or _fingerprint(state_path) != before:
        raise ValueError('Dataset 검사 완료 파일·상태가 변경되었습니다.')

    return output
