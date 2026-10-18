"""첫 WM EDA 생성의 canonical 바이트 선택 검사. 배열·pickle은 파싱하지 않는다."""

import json
from pathlib import Path

from .wm_canonical_contract import WmCanonicalInput, digest, inventory
from .worker_paths import _path


def canonical_selection(directory, options, limits, max_output_bytes):
    """허용 파일의 지문과 복사 예산을 고정한다. 품질 통과나 실행 승인은 아니다."""
    if directory is None or type(options) is not dict or set(options) != {
            "max_input_bytes", "include_examples", "max_group_ratio"}:
        raise ValueError("첫 EDA에는 canonical 폴더와 입력 상한·예시 그림·그룹 검토 설정이 필요합니다.")

    maximum = options["max_input_bytes"]

    if type(maximum) is not int or not 1 <= maximum <= 32 * 1024 ** 3:
        raise ValueError("canonical 복사 상한은 1 byte~32 GiB 범위의 정수로 지정하세요.")

    root = _path(Path(directory).absolute(), directory=True)
    # 실제 파일을 해싱하기 전에 설정 타입·범위를 기존 계약으로 확인한다.
    from .wm_canonical_contract import MANAGEMENT

    empty = {key: {"size": 0, "sha256": "0" * 64} for key in MANAGEMENT}
    WmCanonicalInput(str(root), json.dumps(empty), limits, maximum, max_output_bytes,
                     options["include_examples"], options["max_group_ratio"])
    files = inventory(root, maximum)
    return {"directory": str(root), **options, "files_id": digest(files),
            "source_id": "wm_canonical:" + digest(files), "file_count": len(files),
            "snapshot_bytes": sum(item["size"] for item in files.values())}


def check_canonical_source(selection, limits, maximum):
    """생성 전후 원본의 바이트 선택이 유지되는지 확인한다. 명세 전체는 응답에 넣지 않는다."""
    options = {key: selection[key] for key in ("max_input_bytes", "include_examples", "max_group_ratio")}

    if canonical_selection(selection["directory"], options, limits, maximum) != selection:
        raise ValueError("확인한 canonical 원본이 변경되었습니다. 새 작업을 배정하지 않습니다.")
