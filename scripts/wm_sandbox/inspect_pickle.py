"""Sandbox 전용 최초 구조 검사. 완전한 품질 audit·변환·안전성 인증이 아니다."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid
import numpy as np
import pandas as pd


# ==========================================
# 고정 공유 루트 밖의 경로 거부
# - 경로 검사는 보조 통제이며 OS 격리를 대신하지 않음
# ==========================================
def within_root(path, root):
    resolved = Path(path).resolve(strict=True)
    resolved.relative_to(Path(root).resolve(strict=True))
    return resolved


# ==========================================
# 파일 식별값 계산
# - 블록 단위로 읽어 추가 메모리 사용을 제한
# - 해시는 파일 안전성 보증이 아니라 변경 확인용
# ==========================================
def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# ==========================================
# 앞부분 최대 1,000행의 맵·라벨 구조 검사
# - 전체 pickle 로드 메모리를 줄이는 기능이 아님
# - 맵·라벨 값·Lot 식별자를 복사하지 않고 구조 집계만 기록
# - 전체 품질·중복·Lot·클래스 검증은 후속 audit에서 수행
# ==========================================
def inspect_frame(frame, limit=1000):
    import numpy as np
    import pandas as pd

    if type(frame) is not pd.DataFrame:
        raise ValueError("지원하는 최상위 객체가 아닙니다.")
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError("검사 행 한도를 벗어났습니다.")
    if frame.columns.has_duplicates:
        raise ValueError("원본 컬럼이 중복됩니다.")
    if not all(type(column) is str for column in frame.columns):
        raise ValueError("문자열 컬럼명만 지원합니다.")

    required = ("waferMap", "failureType", "lotName")
    missing = [column for column in required if column not in frame.columns]

    if missing:
        return {"status": "blocked_schema", "row_count": len(frame),
                "columns": list(frame.columns), "missing_required_columns": missing,
                "full_quality_audit": False}

    map_status = Counter()
    label_structures = Counter()
    shapes = Counter()
    sampled = frame.iloc[:limit]

    for value in sampled["waferMap"]:
        if type(value) is not np.ndarray or value.ndim != 2 or not value.size:
            map_status["invalid_array_structure"] += 1
            continue

        if value.dtype.kind not in "biuf" or value.size > 1_000_000:
            map_status["unsupported_dtype_or_size"] += 1
            continue

        # 값과 형태만 확인하고 clipping·결측 보정·resize는 하지 않는다.
        shapes[f"{value.shape[0]}x{value.shape[1]}"] += 1
        if not np.isfinite(value).all() or not np.isin(value, (0, 1, 2)).all():
            map_status["invalid_map_values"] += 1
        elif not np.any(value > 0):
            map_status["no_valid_die"] += 1
        else:
            map_status["valid_structure"] += 1

    for value in sampled["failureType"]:
        # 라벨 자체가 아니라 빈 배열·문자열 등 입력 구조만 기록한다.
        if type(value) is np.ndarray:
            key = "empty_array" if value.size == 0 else "array"
        elif type(value) is str:
            key = "string"
        elif value is None:
            key = "null"
        else:
            key = "other"

        label_structures[key] += 1

    return {"status": "inspection_complete", "row_count": len(frame),
            "columns": list(frame.columns), "sampled_rows": len(sampled),
            "sample_selection": "first_rows_not_representative",
            "sample_map_status": dict(map_status), "sample_map_shapes": dict(shapes),
            "sample_label_structures": dict(label_structures),
            "full_quality_audit": False, "raw_samples_exported": False}


# ==========================================
# 승인된 원본의 최초 검사
# - 호스트·미승인 입력은 역직렬화 전에 거부
# - 결과는 전용 출력 폴더의 새 하위 경로에만 기록
# - 계정·경로 확인은 보조 방어이며 악성 pickle 실행을 막지는 못함
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="격리 환경의 최초 원본 구조 검사")
    parser.add_argument("--input", required=True)
    parser.add_argument("--trusted-pickle", action="store_true")
    args = parser.parse_args()

    if os.environ.get("USERNAME") != "WDAGUtilityAccount" or not args.trusted_pickle:
        parser.error("Sandbox 내부에서 출처 확인 후 명시적 승인으로만 실행하세요.")

    source = within_root(args.input, "C:/wm_input")
    output_root = within_root("C:/wm_output", "C:/wm_output")

    if not source.is_file() or source.suffix.lower() not in (".pkl", ".pickle"):
        parser.error("지원하는 pickle 파일을 지정하세요.")

    if source.stat().st_size > 4 * 1024 ** 3:
        parser.error("최초 읽기 파일 크기 상한을 초과했습니다. 자원 검토가 필요합니다.")

    output = output_root / ("inspection_" + uuid.uuid4().hex)
    output.mkdir(exist_ok=False)

    record = {"schema_version": "wm_first_inspection_v1",
              "created_at": datetime.now(timezone.utc).isoformat(),
              "is_security_certification": False, "full_quality_audit": False,
              "raw_samples_exported": False}
    
    try:
        record["environment"] = {"python": sys.version.split()[0],
                                 "numpy": np.__version__, "pandas": pd.__version__}
        record["source_sha256"] = file_digest(source)
        record["source_bytes"] = source.stat().st_size

        # 실제 코드 실행 위험이 있는 지점. 명시적 승인과 OS 격리가 전제다.
        frame = pd.read_pickle(source)
        record.update(inspect_frame(frame))

    except Exception:
        # 원문 예외·객체 repr을 콘솔·LLM에 전달하지 않는다.
        record["status"] = "inspection_failed"
        record["error_code"] = "LOAD_OR_INSPECTION_FAILED"

    with (output / "inspection.json").open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=True, indent=2, allow_nan=False)

    with (output / "inspection.md").open("x", encoding="utf-8") as stream:
        stream.write("# WM-811K 최초 구조 검사\n\n상세 결과는 inspection.json에서 로컬로 확인하세요.\n\n")
        stream.write("앞부분 최대 1,000행의 구조 검사이며 전체 품질·중복·Lot·라벨 검증은 아닙니다.\n")
        stream.write("전체 pickle은 메모리에 로드됩니다. 원본 안전성·무해성을 보장하지 않습니다.\n")
        stream.write("출력은 비신뢰 파일입니다. 상세 수치·구조·식별값을 LLM에 붙여 넣지 마세요.\n")

    print("검사 기록을 C:\\wm_output에 저장했습니다. 상세 내용은 로컬에서만 확인하세요.")
    
    return 0 if record["status"] == "inspection_complete" else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print("검사를 중단했습니다. 경로·권한·출력 설정을 로컬에서 확인하세요.", file=sys.stderr)
        sys.exit(1)
