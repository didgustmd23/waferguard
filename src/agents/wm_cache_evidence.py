"""완료된 변환 검사·확정 분할을 캐시 승인 범위에 읽기 전용으로 결합한다."""

from .split_evidence import read_committed_artifact
from .wm_canonical_contract import digest
from .wm_transform_cli import executor, open_run
from .wm_transform_results import check_files


def committed_cache(tool, manager):
    """캐시 단계의 승인 소비·예약·완료 표식을 전용으로 확인한다.

    기존 분할 증거 reader의 허용 작업을 전역 확장하지 않는다. 이 함수는
    신뢰하는 캐시 실행기가 생성한 prepare_data 범위만 받는다.
    """
    from contextlib import closing
    from dataclasses import asdict
    import json
    import re
    import sqlite3

    from .artifacts import ArtifactContext, _fingerprint, check_artifacts

    state = tool.states.load()
    scope = manager.scope_for('prepare_data')
    state_path = tool.states.directory / 'state.json'
    before = _fingerprint(state_path)

    if (state.status != 'completed' or state.cancel_requested or state.blockers
        or scope.actor_id != state.actor_id or scope.run_id != state.run_id):
        raise ValueError('완료된 캐시 작업이 필요합니다.')

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
        raise ValueError('캐시의 승인 소비·커밋 기록이 일치하지 않습니다.')

    output = tool.states.directory / 'attempts' / task[0]
    context = ArtifactContext(scope.source_id, scope.config_id, scope.plan_id, scope.policy_version, scope.tool_id)

    if not check_artifacts(output, context).reusable or _fingerprint(state_path) != before:
        raise ValueError('캐시 완료 파일·상태가 변경되었습니다.')

    return output


def read_transform_evidence(name, cache_config):
    tool = open_run(name)
    state = tool.states.load()

    if state.status != 'completed' or state.blockers or tool.cancel.is_set():
        raise ValueError('완료된 변환 검사 기록이 필요합니다.')

    artifact = read_committed_artifact(tool.states.directory, executor(tool).scope_for('derive_preparation'))
    report = tool.inspect_attempt(artifact.directory)

    if report['status'] != 'transform_check_passed':
        raise ValueError('영역 소실 검토가 남아 있는 변환 검사는 캐시에 사용할 수 없습니다.')

    directory = artifact.directory / 'check'
    evidence = {
        'run_id': state.run_id, 'plan_id': tool.plan.plan_id,
        'source_id': tool.contract.source_id,
        'transform_run': name, 'split_run': tool.split_run,
        'split_evidence': tool.evidence,
        'check_directory': str(directory),
        'check_files': check_files(directory, tool.contract.max_output_bytes),
        'cache_config': cache_config,
    }
    evidence['evidence_id'] = digest(evidence)
    return tool.contract, evidence


def check_transform_evidence(name, expected):
    contract, actual = read_transform_evidence(name, expected['cache_config'])

    if actual != expected:
        raise ValueError('변환 검사·확정 분할·캐시 설정이 승인 범위와 다릅니다.')

    return contract
