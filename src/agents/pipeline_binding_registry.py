"""입력 어댑터별 파이프라인 계약 바인더 등록부."""

from .reviewed_csv_pipeline_contract import bind_reviewed_csv_pipeline
from .pipeline_csv_execution import build_csv_pipeline_execution


# 등록 ID만 고정 구현에 연결한다. profile 값으로 임의 모듈을 import하지 않는다.
_PIPELINE_BINDERS = {
    "tabular_csv_v1": bind_reviewed_csv_pipeline,
}

# 실행 어댑터도 정적 허용 목록으로 관리한다. 입력 JSON에서 import 경로를 받지 않는다.
_PIPELINE_EXECUTORS = {
    "tabular_csv_v1": build_csv_pipeline_execution,
}


def get_pipeline_binder(adapter_id):
    """등록된 입력 어댑터의 바인더를 반환하고 미등록 형식은 거부한다."""
    if type(adapter_id) is not str:
        raise TypeError("입력 어댑터 ID는 문자열이어야 합니다.")

    binder = _PIPELINE_BINDERS.get(adapter_id)

    if binder is None:
        raise ValueError(f"지원하지 않는 파이프라인 입력 어댑터입니다: {adapter_id}")

    return binder


def registered_pipeline_adapters():
    """파이프라인 바인더가 등록된 adapter ID를 정렬해 반환한다."""
    return tuple(sorted(_PIPELINE_BINDERS))


def build_pipeline_execution(adapter_id, states, binding):
    """지원 어댑터의 고정 실행기를 만들고 미등록 형식은 거부한다."""
    if type(adapter_id) is not str:
        raise TypeError("입력 어댑터 ID는 문자열이어야 합니다.")

    factory = _PIPELINE_EXECUTORS.get(adapter_id)

    if factory is None:
        raise ValueError(f"실행기가 등록되지 않은 입력 형식입니다: {adapter_id}")

    return factory(states, binding)
