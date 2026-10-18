"""Dataset 검사 전용 PyTorch 런타임의 제한 복사 목록. 기존 런타임 정책은 바꾸지 않는다."""

from importlib.metadata import distribution
import json
from pathlib import Path
import re

from .artifacts import _is_link, _root, _safe_relative
from .wm_canonical_contract import file_hash


MAX_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_RUNTIME_FILES = 60000
PACKAGES = (
    'torch', 'functorch', 'torchgen', 'filelock', 'typing_extensions.py',
    'sympy', 'mpmath', 'networkx', 'jinja2', 'markupsafe', 'fsspec',
    'setuptools', 'pkg_resources', '_distutils_hack',
)
METADATA = (
    'torch-', 'filelock-', 'typing_extensions-', 'sympy-', 'mpmath-', 'networkx-',
    'jinja2-', 'markupsafe-', 'fsspec-', 'setuptools-',
)


def runtime_manifest(prefix, maximum):
    """설치된 라이브러리만 해싱한다. 설치·다운로드·모델 파일 로드는 하지 않는다."""
    from .appcontainer_probe import runtime_files

    _check_maximum(maximum)
    prefix = _root(prefix)
    paths = set(runtime_files(prefix))
    site = prefix / 'Lib/site-packages'

    if not (site / 'torch/__init__.py').is_file():
        raise ValueError('현재 Python 환경에 PyTorch가 설치되어 있어야 합니다.')

    selected = [site / name for name in PACKAGES]
    selected.extend(path for path in site.glob('*.dist-info') if path.name.lower().startswith(METADATA))

    for path in selected:
        if path.is_dir():
            paths.update(item for item in path.rglob('*') if item.is_file() and '__pycache__' not in item.parts
                         and item.suffix not in {'.pyc', '.pyo'})
        elif path.is_file():
            paths.add(path)

    records, total = {}, 0

    if len(paths) > MAX_RUNTIME_FILES:
        raise ValueError('Dataset 런타임 파일 개수 상한을 초과했습니다.')

    for path in sorted(paths):
        # Conda 설치 파일은 패키지 저장소와 하드 링크일 수 있다. 신뢰하는
        # 설치 환경은 읽기만 하고 별도 바이트 복사·해시 검증으로 분리한다.
        if (not path.is_file() or
            any(_is_link(part) for part in (path, *path.parents) if part != prefix.parent)):
            raise ValueError('런타임에 심볼릭 링크·재분석 지점·누락 파일을 허용하지 않습니다.')

        relative = path.relative_to(prefix).as_posix()
        runtime_relative(relative)
        record = file_hash(path, maximum - total)
        records[relative] = record
        total += record['size']

    payload = json.dumps(records, sort_keys=True).encode()

    if len(payload) > MAX_MANIFEST_BYTES:
        raise ValueError('Dataset 런타임 명세 크기 상한을 초과했습니다.')

    return records, distribution('torch').version


def decode_manifest(payload, maximum):
    """큰 파일 목록은 IPC 64KiB 제한과 분리해 제한된 전용 명세로 읽는다."""
    _check_maximum(maximum)

    if len(payload) > MAX_MANIFEST_BYTES:
        raise ValueError('Dataset 런타임 명세가 너무 큽니다.')

    def unique(pairs):
        result = {}

        for key, value in pairs:
            if key in result:
                raise ValueError('런타임 명세의 중복 키는 허용하지 않습니다.')

            result[key] = value

        return result

    records = json.loads(payload, object_pairs_hook=unique)

    if type(records) is not dict or not 1 <= len(records) <= MAX_RUNTIME_FILES:
        raise ValueError('Dataset 런타임 명세가 올바르지 않습니다.')

    total = 0

    for name, item in records.items():
        runtime_relative(name)

        if (type(item) is not dict or set(item) != {'size', 'sha256'} or
            type(item['size']) is not int or item['size'] < 0 or
            type(item['sha256']) is not str or len(item['sha256']) != 64 or
            any(char not in '0123456789abcdef' for char in item['sha256'])):
            raise ValueError('런타임 파일의 크기·해시가 올바르지 않습니다.')

        total += item['size']

    if total > maximum or 'Lib/site-packages/torch/__init__.py' not in records:
        raise ValueError('PyTorch 런타임이 누락되었거나 복사 상한을 초과했습니다.')

    return records


def _check_maximum(value):
    if type(value) is not int or not 1 <= value <= 2 ** 63 - 1:
        raise ValueError('런타임 복사 상한은 유효한 양의 정수여야 합니다.')


def runtime_relative(name):
    """설치 라이브러리의 C++ 헤더·공백 파일명만 별도 허용한다.

    데이터 산출물의 공통 경로 규칙은 완화하지 않는다. 치환한 검사명으로도
    상위 이동·예약 이름·빈 경로 조각을 검사하고 실제 복사에는 원래 이름을 쓴다.
    """
    if type(name) is not str or len(name) > 1024 or re.fullmatch(r'[A-Za-z0-9_./+() -]+', name) is None:
        raise ValueError('런타임 파일 상대 경로가 올바르지 않습니다.')

    if any(part.endswith((' ', '.')) for part in name.split('/')):
        raise ValueError('런타임 경로 끝의 공백·마침표는 허용하지 않습니다.')

    _safe_relative(name.translate(str.maketrans({' ': '_', '+': '_', '(': '_', ')': '_'})))


def copy_runtime_file(source, target, record):
    """고정 길이만 바이트 복사하며 런타임 원본의 변경·성장·링크 전환을 거부한다."""
    import hashlib

    if not source.is_file() or any(_is_link(part) for part in (source, *source.parents)):
        raise ValueError('복사할 설치 라이브러리가 변경되었습니다.')

    target.parent.mkdir(parents=True, exist_ok=True)
    remaining = record['size']
    hasher = hashlib.sha256()

    with source.open('rb') as reader, target.open('xb') as writer:
        while True:
            data = reader.read(min(1024 * 1024, remaining + 1))

            if not data:
                break

            remaining -= len(data)

            if remaining < 0:
                raise ValueError('런타임 파일이 승인한 크기보다 커졌습니다.')

            writer.write(data)
            hasher.update(data)

    if remaining != 0 or hasher.hexdigest() != record['sha256'] or file_hash(target, record['size']) != record:
        raise ValueError('런타임 복사본이 승인한 설치 파일 지문과 다릅니다.')
