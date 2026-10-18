"""worker 요청의 로컬 경로 사전 검사. OS 접근 권한을 대신하지 않는다."""

from dataclasses import dataclass
from pathlib import Path
import stat

from .artifacts import _is_link


def _path(value, *, directory):
    """절대 로컬 경로만 받고 상위 이동·ADS·예약 이름·링크를 거부한다."""
    if not isinstance(value, (str, Path)):
        raise TypeError("worker 경로는 문자열 또는 Path여야 합니다.")

    raw = str(value)

    if not raw or any(ord(character) < 32 for character in raw) or raw.startswith(("\\\\", "//")):
        raise ValueError("worker에는 유효한 로컬 경로만 허용합니다.")

    # Windows 드라이브 구분자 외의 콜론은 alternate data stream 등에 사용될 수 있다.
    tail = raw[2:] if len(raw) >= 2 and raw[0].isalpha() and raw[1] == ":" else raw

    if ":" in tail:
        raise ValueError("worker 경로에 특수 스트림을 사용할 수 없습니다.")

    parts = tail.replace("\\", "/").split("/")
    reserved = {"CON", "PRN", "AUX", "NUL"}
    reserved.update(f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10))

    for part in parts:
        if part in {".", ".."} or (part and (part.endswith((".", " ")) or part.split(".")[0].upper() in reserved)):
            raise ValueError("worker 경로에 상위 이동·예약 이름을 사용할 수 없습니다.")

    path = Path(raw)

    if not path.is_absolute():
        raise ValueError("worker에는 절대 경로가 필요합니다.")

    try:
        # resolve 전에 검사하여 링크가 가리키는 곳으로 먼저 이동하지 않는다.
        for component in (path, *path.parents):
            if _is_link(component):
                raise ValueError("worker 경로에 링크·재분석 지점을 사용할 수 없습니다.")

        mode = path.stat().st_mode

        if directory and not stat.S_ISDIR(mode):
            raise ValueError("worker 폴더 경로가 필요합니다.")

        if not directory and (not stat.S_ISREG(mode) or path.stat().st_nlink != 1):
            raise ValueError("worker 입력은 하드 링크가 아닌 일반 파일이어야 합니다.")

        resolved = path.resolve(strict=True)

        if str(resolved).startswith(("\\\\", "//")):
            raise ValueError("정규화한 worker 경로가 네트워크 경로입니다.")

        return resolved
    except OSError:
        raise ValueError("worker 경로가 없거나 안전하게 확인할 수 없습니다.") from None


def _inside(path, root):
    return path == root or root in path.parents


def _overlap(first, second):
    return _inside(first, second) or _inside(second, first)


@dataclass(frozen=True)
class WorkerPathPolicy:
    """신뢰하는 관리자가 구성한다. worker 요청이나 LLM이 허용 루트를 정하지 않는다."""

    read_roots: tuple[Path, ...]
    output_root: Path
    scratch_root: Path
    protected_roots: tuple[Path, ...] = ()

    def __post_init__(self):
        if not isinstance(self.read_roots, tuple) or not self.read_roots or not isinstance(self.protected_roots, tuple):
            raise ValueError("읽기·보호 루트는 고정 tuple이어야 합니다.")

        reads = tuple(_path(path, directory=True) for path in self.read_roots)
        protected = tuple(_path(path, directory=True) for path in self.protected_roots)
        output = _path(self.output_root, directory=True)
        scratch = _path(self.scratch_root, directory=True)

        if any(path == Path(path.anchor) for path in (*reads, output, scratch, *protected)):
            raise ValueError("드라이브 전체를 worker 허용 루트로 지정할 수 없습니다.")

        if len(set(reads)) != len(reads) or _overlap(output, scratch):
            raise ValueError("worker 루트가 중복되거나 출력·임시 영역이 겹칩니다.")

        if any(_overlap(write, read) for write in (output, scratch) for read in (*reads, *protected)):
            raise ValueError("worker 쓰기 영역과 읽기·보호 영역은 분리해야 합니다.")

        object.__setattr__(self, "read_roots", reads)
        object.__setattr__(self, "output_root", output)
        object.__setattr__(self, "scratch_root", scratch)
        object.__setattr__(self, "protected_roots", protected)

    def check_read(self, value, *, directory=False):
        """요청한 정확한 입력만 허용한다. 폴더 내부 파일 전체 검사는 여기서 수행하지 않는다."""
        path = _path(value, directory=directory)

        if not any(_inside(path, root) for root in self.read_roots):
            raise ValueError("worker 읽기 경로가 허용 범위를 벗어났습니다.")

        if any(_overlap(path, root) for root in self.protected_roots):
            raise ValueError("worker는 승인 DB·비밀 등 보호 영역을 읽을 수 없습니다.")

        return path

    def check_write(self, value, *, scratch=False):
        """관리자가 미리 만든 빈 작업 하위 폴더만 사용하며 파일을 생성하지 않는다."""
        path = _path(value, directory=True)
        root = self.scratch_root if scratch else self.output_root

        if path == root or not _inside(path, root):
            raise ValueError("worker 쓰기는 작업별 허용 하위 폴더만 가능합니다.")

        if any(path.iterdir()):
            raise ValueError("worker 작업 폴더는 비어 있어야 합니다.")

        return path
