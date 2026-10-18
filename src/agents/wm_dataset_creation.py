"""Dataset 작업 생성 전의 배치·복사 예산·설치 런타임 확인. 실행 기능은 없다."""

from pathlib import Path
import sys

from .wm_canonical_contract import digest


DEFAULT_OPTIONS = {
    "batch_size": 32,
    "max_input_bytes": 8 * 1024 ** 3,
    "max_runtime_bytes": 8 * 1024 ** 3,
}
MAX_COPY_BYTES = 32 * 1024 ** 3


def dataset_selection(parent, options):
    """완료 캐시와 설치 환경만 읽고 확인할 선택 범위를 반환한다.

    입력·런타임 복사 예산은 RAM 제한과 다르다. 캐시 배열을 복원하거나
    PyTorch로 배치를 생성하지 않는다. 설치·다운로드도 수행하지 않는다.
    """
    if options is None:
        options = dict(DEFAULT_OPTIONS)

    if type(options) is not dict or set(options) != set(DEFAULT_OPTIONS):
        raise ValueError("Dataset 설정은 배치 크기·입력 복사 상한·런타임 복사 상한만 지정해야 합니다.")

    if type(options["batch_size"]) is not int or not 1 <= options["batch_size"] <= 256:
        raise ValueError("Dataset 배치 크기는 1~256의 정수여야 합니다.")

    for key in ("max_input_bytes", "max_runtime_bytes"):
        if type(options[key]) is not int or not 1 <= options[key] <= MAX_COPY_BYTES:
            raise ValueError("Dataset 입력·런타임 복사 상한은 1 byte~32 GiB 범위의 정수여야 합니다.")

    cache_bytes = sum(record["size"] for record in parent.report["files"].values())

    if cache_bytes > options["max_input_bytes"]:
        raise ValueError("Dataset 입력 복사 상한이 완료 캐시 크기보다 작습니다. 복사 예산을 검토하세요.")

    # 기존 CLI가 사용하는 고정 런타임 수집기를 그대로 재사용한다.
    # 실제 생성 시에도 재수집하므로 확인 중 설치 환경이 바뀌면 배정하지 않는다.
    from .wm_dataset_cli import runtime_manifest

    records, version = runtime_manifest(Path(sys.prefix), options["max_runtime_bytes"])
    return {**options, "runtime_sha256": digest(records), "torch_version": version,
            "cache_bytes": cache_bytes}
