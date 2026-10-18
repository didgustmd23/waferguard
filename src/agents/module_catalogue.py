"""고정 사용자 모듈 명세·코드 목록 검사. JSON을 import 경로나 실행 명령으로 해석하지 않는다."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import stat

from src.preprocessing.builtin_modules import build_builtin_registry
from src.preprocessing.module_contracts import ModuleContract, ModulePort
from src.preprocessing.module_registry import ModuleRegistry
from .worker_contract import _decode
from .worker_paths import _path
from .artifacts import _is_link


MODULE_ID = "user_missing_row_histogram"
VERSION = "1.0.0"
DESCRIPTOR = "configs/agents/modules/missing_row_histogram.json"
SOURCE_FILES = (
    "src/__init__.py", "src/preprocessing/__init__.py",
    "src/preprocessing/extensions/__init__.py",
    "src/preprocessing/extensions/missing_row_histogram.py",
    "src/preprocessing/contracts.py", "src/preprocessing/module_contracts.py",
    "src/preprocessing/module_registry.py",
)
# 명세 파일까지 고정한다. 명세와 소스만 함께 바꿔도 자동 등록되지 않는다.
# 참조 예제의 정합성 검사 후 계산한 값이며 사람의 운영 사용 승인을 뜻하지 않는다.
DESCRIPTOR_SHA256 = "5dfd9665759d70b9d0f3e9b7c33e7e5f17413f4f1a63265332da0cfc3aa44e72"
_loaded_descriptor = None


def _catalogue_path(path, staged_root=None):
    """worker는 허용된 tools 하위만 검사한다. 접근 불가한 조상 폴더를 탐색하지 않는다."""
    if staged_root is None:
        return _path(path, directory=False)

    root = Path(staged_root)
    path = Path(path)
    relative = path.relative_to(root)

    if relative.as_posix() not in SOURCE_FILES + (DESCRIPTOR,) or _is_link(root):
        raise ValueError("고정 복사 목록 안의 소스 파일만 검사할 수 있습니다.")

    cursor = root

    for part in relative.parts:
        cursor = cursor / part

        if _is_link(cursor):
            raise ValueError("모듈 복사본에 링크를 사용할 수 없습니다.")

    info = path.stat()

    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("모듈 복사본은 일반 파일이어야 합니다.")

    return path


def _source_hash(path, *, staged_root=None):
    """UTF-8 코드의 CRLF만 LF로 정규화한다. 운영 worker의 원시 바이트 해시와 구분한다."""
    path = _catalogue_path(path, staged_root)

    if path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("등록 검사 소스는 파일당 2 MiB 이하여야 합니다.")

    with path.open("rb") as stream:
        value = stream.read(2 * 1024 * 1024 + 1)

    if len(value) > 2 * 1024 * 1024:
        raise ValueError("등록 검사 중 소스 크기 상한을 초과했습니다.")

    value.decode("utf-8")
    return hashlib.sha256(value.replace(b"\r\n", b"\n")).hexdigest()


def _read_descriptor(root, *, staged=False):
    staged_root = root if staged else None
    path = _catalogue_path(root / DESCRIPTOR, staged_root)

    if path.stat().st_size > 65536:
        raise ValueError("모듈 명세는 64 KiB 이하여야 합니다.")

    with path.open("rb") as stream:
        payload = stream.read(65537)

    if len(payload) > 65536 or hashlib.sha256(payload.replace(b"\r\n", b"\n")).hexdigest() != DESCRIPTOR_SHA256:
        raise ValueError("모듈 명세가 고정 등록 목록과 다릅니다. 재검토 후 등록 코드를 갱신하세요.")

    value = _decode(payload.decode("utf-8"))

    if _source_hash(path, staged_root=staged_root) != DESCRIPTOR_SHA256:
        raise ValueError("검사 중 모듈 명세가 변경되었습니다.")

    return value


def _contract(value):
    """입출력·보안 선언을 기존 계약 타입으로 검증하고 누락 필드는 허용하지 않는다."""
    if type(value) is not dict:
        raise ValueError("모듈 계약은 JSON 객체여야 합니다.")

    fields = dict(value)

    for key in ("inputs", "outputs"):
        if type(fields.get(key)) is not list:
            raise ValueError("모듈 입출력은 목록이어야 합니다.")

        ports = []

        for port in fields[key]:
            if type(port) is not dict or set(port) != {"name", "contract_id", "contract_version", "required"}:
                raise ValueError("입출력 포트 명세가 올바르지 않습니다.")

            ports.append(ModulePort(**port))

        fields[key] = tuple(ports)

    for key in ("settings_fields", "preconditions"):
        if type(fields.get(key)) is not list:
            raise ValueError("설정명과 선행 조건은 목록이어야 합니다.")

        fields[key] = tuple(fields[key])

    contract = ModuleContract(**fields)

    if json.loads(json.dumps(asdict(contract))) != value:
        raise ValueError("모듈 계약의 모든 필드를 명시해야 합니다.")

    return contract


def inspect_module(module_id, version, *, project_root=None, staged=False):
    """파일 읽기·해시 검사만 수행한다. 사용자 모듈 import·계산·테스트 실행은 하지 않는다."""
    if (module_id, version) != (MODULE_ID, VERSION):
        raise ValueError("고정 목록에 없는 모듈 또는 버전입니다.")

    root = Path(__file__).resolve().parents[2] if project_root is None else Path(project_root)
    value = _read_descriptor(root, staged=staged)

    if (set(value) != {"schema_version", "contract", "source_files", "review", "execution_scope", "hash_policy"}
            or value["schema_version"] != "reviewed_module_descriptor_v1"
            or value["hash_policy"] != "utf8_lf_sha256"
            or value["execution_scope"] != "synthetic_local_only"
            or type(value["source_files"]) is not dict or set(value["source_files"]) != set(SOURCE_FILES)
            or value["review"] != {"kind": "reference_example", "test_suite": "tests.agents.test_module_catalogue"}):
        raise ValueError("초기 모듈 등록 명세 또는 검토 범위가 올바르지 않습니다.")

    contract = _contract(value["contract"])

    if (contract.module_id, contract.version) != (MODULE_ID, VERSION):
        raise ValueError("명세의 모듈 ID·버전이 고정 목록과 다릅니다.")

    for relative in SOURCE_FILES:
        expected = value["source_files"][relative]

        if type(expected) is not str or len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
            raise ValueError("소스 해시는 SHA-256 형식이어야 합니다.")

        if _source_hash(root / relative, staged_root=root if staged else None) != expected:
            raise ValueError("모듈 또는 선언한 의존 소스가 변경되었습니다. 재검토가 필요합니다.")

    return contract


def build_reviewed_registry(*, staged=False):
    """신뢰된 초기화 코드용 참조 등록부. 실제 worker 허용 목록에 추가하는 API가 아니다."""
    global _loaded_descriptor
    contract = inspect_module(MODULE_ID, VERSION, staged=staged)

    if _loaded_descriptor is not None and _loaded_descriptor != DESCRIPTOR_SHA256:
        raise ValueError("실행 중 등록 목록을 교체하지 않습니다. 새 프로세스에서 재검토하세요.")

    # import 대상은 JSON이 아닌 고정 Python 코드에만 있다. 검사 전에 사용자 코드를 읽지 않는다.
    from src.preprocessing.extensions.missing_row_histogram import build_module

    module = build_module()
    inspect_module(MODULE_ID, VERSION, staged=staged)

    if module.contract != contract:
        raise ValueError("등록한 계산 함수의 계약이 검토 명세와 다릅니다.")

    _loaded_descriptor = DESCRIPTOR_SHA256
    registry = ModuleRegistry()
    builtins = build_builtin_registry()

    for item in builtins.contracts():
        registry.register(builtins.resolve(item.module_id, item.version))

    registry.register(module)
    return registry.seal()
