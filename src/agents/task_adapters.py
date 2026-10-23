"""작업 유형 어댑터 등록부. 공통 조정기·화면은 이 등록부로만 유형별 기능을 찾는다.

pipeline_binding_registry와 같이 고정 dict로 등록한다. 플러그인 탐색·동적 import는 없다.
각 어댑터가 제공하는 이름은 설계 문서 7.3절(공통/유형 책임)에 따른다.
"""

from . import array_task, tabular_task


ADAPTERS = {tabular_task.KIND: tabular_task, array_task.KIND: array_task}


def adapter(kind):
    """기록의 유형으로 어댑터를 찾는다. 등록되지 않은 유형은 거부한다."""
    if type(kind) is not str or kind not in ADAPTERS:
        raise ValueError("등록되지 않은 작업 유형입니다.")

    return ADAPTERS[kind]


def select(path):
    """입력 형식 로더가 받아들이는 첫 유형을 고른다. 받아들이는 유형이 없으면 거부한다."""
    matches = [kind for kind, item in ADAPTERS.items() if item.accepts(path)]

    if len(matches) != 1:
        raise ValueError("지원하는 입력 형식이 아니거나 유형을 하나로 정할 수 없습니다.")

    return matches[0]


def input_specs():
    """요청 화면의 입력 안내. 등록된 유형의 입력 형식만 나열한다."""
    return [dict(item.INPUT, kind_id=kind) for kind, item in ADAPTERS.items()]
