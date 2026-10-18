"""WM worker 결과를 제한된 형식으로 검사한다. 부모는 원본 맵을 재분석하지 않는다."""

import csv
from pathlib import Path

from src.wafer_dl.validate_canonical import split_review_checks
from .artifacts import _is_link, _root
from .wm_canonical_contract import POLICY_VERSION, file_hash
from .worker_contract import _decode


EDA_FILES = frozenset({
    "distributions.png", "die_statistics.png", "correlations.png", "correlation_scatter.png",
    "correlation_pearson.csv", "correlation_spearman.csv", "correlation_support.csv", "correlation_by_class.csv",
    "classes.csv", "statuses.csv", "shapes.csv", "eda.json", "eda.md", "SUCCESS.json",
})
QUALITY_FILES = frozenset({"quality/validation.json", "quality/validation.md"})


def _read_small(path, maximum):
    """부모 검증용 작은 텍스트만 제한된 크기로 읽는다."""
    with path.open("rb") as stream:
        payload = stream.read(maximum + 1)

    if len(payload) > maximum:
        raise ValueError("WM 결과 관리 파일 크기 상한을 초과했습니다.")

    return payload


def result_files(directory, maximum):
    """임의 파일·하위 폴더·링크를 거부하고 전체 출력 크기를 제한한다."""
    root = Path(directory)

    if _is_link(root) or not root.is_dir():
        raise ValueError("분석 결과 폴더가 올바르지 않습니다.")

    paths = []

    for path in root.iterdir():
        if _is_link(path):
            raise ValueError("분석 결과에 링크를 사용할 수 없습니다.")

        if path.name in {"quality", "eda"} and path.is_dir():
            for child in path.iterdir():
                if len(paths) >= 32:
                    raise ValueError("WM 분석 결과 파일 개수 상한을 초과했습니다.")

                paths.append(child)
        elif path.name == "result.json" and path.is_file():
            paths.append(path)
        else:
            raise ValueError("분석 결과에 허용되지 않은 항목이 있습니다.")

    files = {}
    total = 0

    for path in paths:
        if _is_link(path) or not path.is_file() or path.stat().st_nlink != 1:
            raise ValueError("분석 결과에는 일반 파일만 허용합니다.")

        record = file_hash(path, maximum - total)
        total += record["size"]
        files[path.relative_to(root).as_posix()] = record

    return files


def inspect_analysis(directory, request, contract):
    """요청 결합·검토 상태·파일 해시·JSON/CSV/PNG 구조를 확인한다."""
    root = _root(directory)
    fingerprints = result_files(root, contract.max_output_bytes)
    result = _decode(_read_small(root / "result.json", 65536).decode("utf-8"))
    expected_fields = {"version", "request_id", "source_id", "config_id", "quality_status", "files"}

    if (set(result) != expected_fields or result["version"] != POLICY_VERSION
        or result["request_id"] != request.request_id or result["source_id"] != request.source_id
        or result["config_id"] != request.config_id or result["quality_status"] not in {"passed", "review_required"}):
        raise ValueError("WM 분석 결과가 현재 승인 요청과 다릅니다.")

    expected = set(QUALITY_FILES)

    if result["quality_status"] == "passed":
        expected.update("eda/" + name for name in EDA_FILES)

        if contract.include_examples:
            expected.add("eda/examples.png")

    if (set(fingerprints) != expected | {"result.json"}
        or result["files"] != {name: fingerprints[name] for name in expected}):
        raise ValueError("WM 분석 결과 파일 목록·크기·해시가 올바르지 않습니다.")

    report = _decode(_read_small(root / "quality/validation.json", 65536).decode("utf-8"))

    if (report.get("status") != "integrity_passed" or report.get("integrity_passed") is not True
        or report.get("training_ready") is not False or report.get("source_equivalence_verified") is not False
        or report.get("is_security_certification") is not False
        or type(report.get("row_count")) is not int or not 0 <= report["row_count"] <= 2_000_000
        or type(report.get("shard_count")) is not int or not 0 <= report["shard_count"] <= 4096):
        raise ValueError("WM 품질 보고서의 구조가 올바르지 않습니다.")

    checks = split_review_checks(report)
    quality_passed = all(passed for _, passed, _ in checks)
    assessment = report["split_assessment"]

    if (quality_passed != (result["quality_status"] == "passed")
        or assessment.get("status") != ("preconditions_passed" if quality_passed else "review_required")
        or assessment.get("max_group_ratio") != contract.max_group_ratio
        or assessment.get("split_complete") is not False):
        raise ValueError("WM 품질 판정과 검토 조건이 일치하지 않습니다.")

    for name in expected:
        path = root / name

        if name.endswith(".png"):
            # 기존 부모 검증기의 PNG 청크·CRC·픽셀 크기 제한을 그대로 재사용한다.
            # 이미지 렌더링·이미지 라이브러리의 외부 데이터 복원은 하지 않는다.
            from .isolated_eda import _valid_png

            if not _valid_png(_read_small(path, 8 * 1024 * 1024)):
                raise ValueError("WM 그림의 PNG 구조·CRC·크기가 올바르지 않습니다.")
        elif name.endswith(".csv"):
            # CSV는 문자열로만 구조를 확인한다. 수식·Python·pickle을 실행하지 않는다.
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.reader(stream, strict=True)
                header = next(reader, None)

                if header is None or not 1 <= len(header) <= 64:
                    raise ValueError("WM CSV 헤더가 올바르지 않습니다.")

                for count, row in enumerate(reader, start=1):
                    if len(row) != len(header) or count > 2_000_000:
                        raise ValueError("WM CSV 행 구조 또는 개수가 올바르지 않습니다.")
        elif name.endswith(".json"):
            value = _decode(_read_small(path, 65536).decode("utf-8"))

            if name == "eda/SUCCESS.json" and value != {"status": "eda_complete", "training_ready": False}:
                raise ValueError("WM EDA 완료 기록이 올바르지 않습니다.")

            if name == "eda/eda.json" and (value.get("row_count") != report["row_count"]
                                           or value.get("training_ready") is not False
                                           or value.get("split_complete") is not False):
                raise ValueError("WM EDA 요약의 처리 범위가 품질 보고서와 다릅니다.")
        else:
            _read_small(path, 65536).decode("utf-8")

    if result_files(root, contract.max_output_bytes) != fingerprints:
        raise ValueError("검증 중 WM 분석 결과가 변경되었습니다.")

    return {"quality_status": result["quality_status"], "eda_generated": quality_passed,
            "files": fingerprints, "training_ready": False, "model_trained": False}
