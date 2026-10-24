"""적용 결과의 제한된 JSON 저장과 고정 보고서 안내."""

import hashlib
import json


def write_json_bounded(path, value, maximum):
    """각 UTF-8 조각을 쓰기 전에 상한을 확인한다. 부분 파일에는 완료를 표시하지 않는다."""
    size = 0
    digest = hashlib.sha256()

    with path.open("xb") as stream:
        for chunk in json.JSONEncoder(ensure_ascii=False, allow_nan=False).iterencode(value):
            payload = chunk.encode("utf-8")

            if size + len(payload) > maximum:
                raise ValueError("적용 결과 JSON 크기 상한을 초과했습니다.")

            stream.write(payload)
            size += len(payload)
            digest.update(payload)

    return {"size_bytes": size, "sha256": digest.hexdigest()}


def application_markdown(*, typed=False):
    """구간 메타데이터 형식에 맞는 고정 사용 안내를 만든다."""
    metadata = ("각 *_rows.json에는 원본 행 위치·인덱스와 라벨·ID·그룹·시간을 안전한 스칼라 및 자료형 스키마로 보존했습니다.\n\n"
                if typed else "각 *_rows.json에는 원본 행 위치와 라벨·ID·그룹·시간 등의 문자열을 보존했습니다.\n\n")
    return ("# 저장 기준 적용 결과\n\n"
            "승인된 후보 상태를 복원해 Train·Validation에 적용했습니다. 기준을 다시 계산하지 않았습니다.\n\n"
            "각 *_features.csv는 특징만 포함합니다. " + metadata +
            "파일별 출처·해시·행 수는 application.json에서 로컬로 확인하세요. "
            "원본·후보는 변경하지 않았고 Test 적용·모델 학습·성능 평가는 하지 않았습니다.\n\n"
            "결과·통계·라벨·식별자는 로컬 전용이며 LLM으로 보내지 마세요.\n")
