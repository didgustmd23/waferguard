"""부모 프로세스의 로컬 기록·검증된 결과 파일용 엄격한 JSON 읽기. 상한은 호출부가 정한다.

worker IPC 계약(worker_contract._decode, 64 KiB)과 같은 엄격도(중복 키·비표준 숫자 거부, 객체만 허용)를 쓰되
크기 상한만 기록 종류에 맞게 받는다. worker IPC 상한은 바꾸지 않는다.
"""

import json


class JsonTooLarge(ValueError):
    """상한을 넘는 JSON. 읽기 실패 원인을 구분해 안내하기 위해 따로 둔다."""


def loads(payload, limit):
    """상한 이하의 JSON 객체만 읽는다. 중복 키·NaN·Infinity·객체가 아닌 최상위 값은 거부한다."""
    if not isinstance(payload, str) or type(limit) is not int or limit <= 0:
        raise ValueError("JSON 입력 형식이 올바르지 않습니다.")

    if len(payload.encode("utf-8")) > limit:
        raise JsonTooLarge("JSON 크기가 상한을 초과했습니다.")

    def pairs(items):
        result = {}

        for key, value in items:
            if key in result:
                raise ValueError("JSON에 중복 키가 있습니다.")

            result[key] = value

        return result

    def invalid_constant(value):
        raise ValueError("JSON에 비표준 숫자가 있습니다.")

    value = json.loads(payload, object_pairs_hook=pairs, parse_constant=invalid_constant)

    if not isinstance(value, dict):
        raise ValueError("JSON 최상위 값은 객체여야 합니다.")

    return value
