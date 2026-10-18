"""완료된 개발 캐시의 CPU Dataset·첫 배치 검사를 별도 승인하는 CLI."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import secrets
import sys

from .approval_store import ApprovalStore
from .artifacts import _root
from .audit import append_event
from .cli import _current_actor
from .contracts import ToolResult
from .executor import RegisteredTool, RulesExecutor, _run_lock
from .isolated_eda import _read_file
from .isolated_eda_cli import cancel
from .state import LocalStateStore
from .wm_canonical_contract import digest
from .wm_dataset import WmDatasetTool
from .wm_dataset_contract import POLICY_VERSION, WmDatasetInput, read_cache_evidence, committed_dataset
from .wm_dataset_runtime import runtime_manifest, MAX_MANIFEST_BYTES
from .worker_contract import WorkerLimits, _decode


def _directory(name, *, create_parent=False):
    if type(name) is not str or re.fullmatch(r'[a-z][a-z0-9_-]{0,47}', name) is None:
        raise ValueError('작업 이름은 소문자로 시작하는 영문·숫자·밑줄·하이픈 1~48자여야 합니다.')

    directory = Path(__file__).resolve().parents[2]

    for part in ('logs', 'agents', 'wm_dataset', 'runs'):
        directory /= part

        if create_parent:
            directory.mkdir(exist_ok=True)

        _root(directory)

    return directory / name


def _preflight(output, arguments):
    if arguments or os.name != 'nt' or sys.version_info[:2] != (3, 10):
        return ToolResult('blocked_input')

    (output / 'preflight.json').write_text(json.dumps({'version': POLICY_VERSION}), encoding='utf-8')
    return ToolResult('passed')


def _validate_preflight(output, arguments):
    return not arguments and _decode(_read_file(output / 'preflight.json', 4096).decode()) == {'version': POLICY_VERSION}


def executor(tool):
    return RulesExecutor(tool.states, ApprovalStore(tool.states.directory / 'approvals.sqlite3'), tool.plan, (
        RegisteredTool('run_preflight', _preflight, _validate_preflight), tool.registered_tool(),
    ))


def create(name, cache_run, limits, *, batch_size=32, max_input_bytes=8 * 1024 ** 3,
           max_runtime_bytes=8 * 1024 ** 3, max_output_bytes=1024 * 1024, confirmed=False):
    """학습·Test를 제외한 첫 배치 검사만 준비한다. 배열은 승인 후 worker에서 읽는다."""
    if not confirmed:
        raise ValueError('개발 캐시의 CPU 첫 배치 검사만 수행함을 확인해야 합니다.')

    directory = _directory(name, create_parent=True)

    if directory.exists() or directory.is_symlink():
        raise ValueError('기존 Dataset 검사 작업을 덮어쓰지 않습니다. 새 이름을 사용하세요.')

    evidence = read_cache_evidence(cache_run)
    runtime, version = runtime_manifest(Path(sys.prefix), max_runtime_bytes)
    contract = WmDatasetInput(evidence['cache_directory'], json.dumps(evidence['cache_files']), limits,
                              max_input_bytes, max_output_bytes, max_runtime_bytes, digest(runtime), version, batch_size)
    directory.mkdir()
    (directory / 'runtime_files.json').write_text(json.dumps(runtime, sort_keys=True), encoding='utf-8')
    states = LocalStateStore(directory)
    tool = WmDatasetTool(states, contract, cache_run, evidence)
    tool._check_arguments(tool.arguments)
    state = states.create(_current_actor(), policy_version=POLICY_VERSION)
    record = {'contract': contract.to_dict(), 'cache_run': cache_run, 'evidence': evidence,
              'code_id': tool.code_id, 'plan_id': tool.plan.plan_id}
    payload = json.dumps(record, ensure_ascii=True).encode()

    if len(payload) > 4 * 1024 * 1024:
        raise ValueError('Dataset 선택 기록이 상한을 초과했습니다.')

    (directory / 'selection.json').write_bytes(payload)
    ApprovalStore(directory / 'approvals.sqlite3')
    append_event(directory / 'events.jsonl', 'run_created', 'RUN_CREATED',
                 stage=state.current_stage, revision=state.revision)
    return {'status': 'created', 'run_dir': str(directory), 'worker_started': False,
            'notice': 'Train·Validation 캐시의 CPU 첫 배치 검사만 준비했습니다. 실제 검사는 새 승인이 필요합니다.'}


def open_run(name):
    directory = _root(_directory(name))
    payload = _read_file(directory / 'selection.json', 4 * 1024 * 1024)
    # 승인 선택은 4MiB로 제한하고 중복 키를 허용하지 않는다.
    from .wm_cache_results import _unique_pairs

    record = json.loads(payload, object_pairs_hook=_unique_pairs)

    if type(record) is not dict or set(record) != {'contract', 'cache_run', 'evidence', 'code_id', 'plan_id'}:
        raise ValueError('Dataset 선택 기록이 올바르지 않습니다.')

    states = LocalStateStore(directory)

    if states.load().actor_id != _current_actor():
        raise ValueError('작업을 생성한 OS 계정에서 실행하세요.')

    tool = WmDatasetTool(states, WmDatasetInput.from_dict(record['contract']), record['cache_run'], record['evidence'])

    if tool.code_id != record['code_id'] or tool.plan.plan_id != record['plan_id']:
        raise ValueError('Dataset 검사 코드·설정·계획이 변경되었습니다. 새 작업과 승인이 필요합니다.')

    tool._check_arguments(tool.arguments)
    return tool


def approve(tool):
    if not os.isatty(0):
        raise ValueError('승인은 대화형 터미널에서 직접 진행하세요.')

    state = tool.states.load()
    manager = executor(tool)

    if state.status != 'awaiting_approval' or state.current_stage != 'validate_contract' or tool.cancel.is_set():
        raise ValueError('취소되지 않은 Dataset 검사 승인 대기 작업이 필요합니다.')

    tool._check_arguments(tool.arguments)
    scope = manager.scope_for('validate_contract')
    token = 'WM-DATASET-' + secrets.token_hex(4).upper()
    print(json.dumps({'cache_run': tool.cache_run, 'source_id': tool.contract.source_id,
                      'batch_size': tool.contract.batch_size, 'partitions': ['train', 'validation'],
                      'device': 'cpu', 'num_workers': 0, 'torch_version': tool.contract.torch_version,
                      'limits': asdict(tool.limits), 'max_input_bytes': tool.contract.max_input_bytes,
                      'max_runtime_bytes': tool.contract.max_runtime_bytes,
                      'runtime_bytes': sum(item['size'] for item in tool._runtime().values()),
                      'max_output_bytes': tool.contract.max_output_bytes, 'lifetime_minutes': 30,
                      'effect': '캐시·고정 PyTorch 의존성 복사 후 새 AppContainer+Job에서 두 개발 구간의 첫 배치 검사',
                      'notice': '원본 canonical·Test 배열·승인 DB 전달·GPU 연산·모델 학습·LLM 전송 없음'},
                     ensure_ascii=False, indent=2))

    if input(f'Dataset 첫 배치 검사를 승인하려면 {token} 을 입력하세요: ') != token:
        return {'code': 'APPROVAL_DECLINED'}

    with _run_lock(tool.states.directory) as acquired:
        if not acquired or tool.states.load() != state or tool.cancel.is_set():
            raise ValueError('승인 중 작업 상태가 변경되었습니다.')

        tool._check_arguments(tool.arguments)

        if manager.scope_for('validate_contract') != scope:
            raise ValueError('Dataset 검사 승인 범위가 변경되었습니다.')

        append_event(tool.states.directory / 'events.jsonl', 'approval_requested', 'LOCAL_REVIEW_CONFIRMED',
                     stage=state.current_stage, revision=state.revision)
        record = manager.approvals.record_confirmed_approval(scope)

    return {'approval_id': record.approval_id, 'expires_at': record.expires_at.isoformat()}


def status(tool):
    state = tool.states.load()
    result = {'status': state.status, 'stage': state.current_stage, 'cancel_requested': tool.cancel.is_set()}

    if state.status == 'completed':
        attempt = committed_dataset(tool, executor(tool))
        report = tool.inspect_attempt(attempt)
        result.update(dataset_status=report['status'], development_inputs_ready=True,
                      test_maps_read=False, model_trained=False,
                      local_report=str(attempt / 'dataset/dataset_check.md'))
    elif state.status in {'failed_runtime', 'blocked_input'} and state.current_stage == 'validate_contract':
        # 입력 값·예외 원문 대신 현재 시도와 결합된 고정 진단만 표시한다.
        from .wm_dataset_transport import PREPARATION_CODES, PREPARATION_STAGES

        manager = executor(tool)
        task = manager.approvals.get_task(manager.scope_for('validate_contract'))

        if task is not None:
            request = tool._request(task['task_id'])
            root = tool._work_root(task['task_id'])
            transport = root / 'transport.json'

            if transport.is_file():
                outcome = _decode(_read_file(transport, 65536).decode())
                failure = outcome.get('preparation_failure')

                if (outcome.get('request_id') == request.request_id and type(failure) is dict and
                    set(failure) == {'stage', 'code', 'os_error'} and
                    failure['stage'] in PREPARATION_STAGES and failure['code'] in PREPARATION_CODES and
                    (failure['os_error'] is None or type(failure['os_error']) is int and
                     0 <= failure['os_error'] <= 2 ** 32 - 1)):
                    result['preparation_failure'] = failure

                diagnostic = root / 'scratch/task/diagnostic.json'

                if outcome.get('request_id') == request.request_id and diagnostic.is_file():
                    item = _decode(_read_file(diagnostic, 4096).decode())

                    if (set(item) == {'stage', 'code'} and
                        item['stage'] in {'REQUEST', 'IMPORT_TORCH', 'CHECK_TRAIN', 'CHECK_VALIDATION', 'BOOTSTRAP', 'COMPLETE'} and
                        item['code'] in {'START', 'PASSED', 'DATASET_ACCESS_DENIED', 'DATASET_CHECK_FAILED', 'DATASET_BOOTSTRAP_FAILED'}):
                        result['worker_diagnostic'] = item

    return result


def main():
    parser = argparse.ArgumentParser(description='WM 개발 캐시의 CPU Dataset·첫 배치 검사')
    commands = parser.add_subparsers(dest='command', required=True)

    for command in ('create', 'next', 'approve', 'status', 'cancel'):
        sub = commands.add_parser(command)
        sub.add_argument('--name', required=True)

        if command == 'create':
            sub.add_argument('--cache-run', required=True)
            sub.add_argument('--batch-size', type=int, default=32)
            sub.add_argument('--memory-mib', type=int, default=8192)
            sub.add_argument('--processes', type=int, default=2)
            sub.add_argument('--wall-seconds', type=int, default=900)
            sub.add_argument('--max-input-bytes', type=int, default=8 * 1024 ** 3)
            sub.add_argument('--max-runtime-bytes', type=int, default=8 * 1024 ** 3)
            sub.add_argument('--max-output-bytes', type=int, default=1024 * 1024)
            sub.add_argument('--confirm-development-only', action='store_true')
        elif command == 'next':
            sub.add_argument('--approval-id')

    args = parser.parse_args()

    try:
        if args.command == 'create':
            result = create(args.name, args.cache_run, WorkerLimits(args.wall_seconds, args.memory_mib, args.processes),
                            batch_size=args.batch_size, max_input_bytes=args.max_input_bytes,
                            max_runtime_bytes=args.max_runtime_bytes, max_output_bytes=args.max_output_bytes,
                            confirmed=args.confirm_development_only)
        else:
            tool = open_run(args.name)

            if args.command == 'next':
                result = {**asdict(executor(tool).run_next(approval_id=args.approval_id)), 'worker_started': tool.worker_started}
            elif args.command == 'approve':
                result = approve(tool)
            elif args.command == 'cancel':
                result = cancel(tool)
            else:
                result = status(tool)

        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception:
        print(json.dumps({'status': 'failed', 'code': 'WM_DATASET_REQUEST_FAILED',
                          'message': '완료 캐시·런타임 상한·작업 이름·승인 상태를 확인하세요.'}, ensure_ascii=False))
        raise SystemExit(2)


if __name__ == '__main__':
    main()
