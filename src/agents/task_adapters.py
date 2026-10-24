"""작업 유형 어댑터 등록부. 공통 조정기·화면은 이 등록부로만 유형별 기능을 찾는다.

pipeline_binding_registry와 같이 고정 dict로 등록한다. 플러그인 탐색·동적 import는 없다.
각 어댑터가 제공하는 이름은 설계 문서 7.3절(공통/유형 책임)에 따른다.
"""

from . import array_task, tabular_task, tabular_v2_task


ADAPTERS = {tabular_task.KIND: tabular_task, tabular_v2_task.KIND: tabular_v2_task, array_task.KIND: array_task}
# 새 작업에서 고를 수 있는 유형 집합은 등록된 유형에서 세대별 제외 유형을 뺀 것이다. legacy는 기존 호출·테스트의 기본값
# (v2 도입 전과 같은 후보)이고, 통합 화면의 새 작업은 current를 쓴다. CSV는 세대마다 한 유형만 받는다.
# 저장된 작업은 기록의 kind로만 연다(ADAPTERS). 외부 입력이 유형·import 대상을 지정할 수 없다.
EXCLUDED = {"legacy": frozenset({tabular_v2_task.KIND}), "current": frozenset({tabular_task.KIND})}


def adapter(kind):
    """기록의 유형으로 어댑터를 찾는다. 등록되지 않은 유형은 거부한다."""
    if type(kind) is not str or kind not in ADAPTERS:
        raise ValueError("등록되지 않은 작업 유형입니다.")

    return ADAPTERS[kind]


def _generation(generation):
    if type(generation) is not str or generation not in EXCLUDED:
        raise ValueError("지원하지 않는 입력 세대입니다.")

    return [kind for kind in ADAPTERS if kind not in EXCLUDED[generation]]


def select(path, *, generation="legacy"):
    """선택한 세대의 유형 중 입력 형식 로더가 받아들이는 유형 하나를 고른다. 하나로 정할 수 없으면 거부한다."""
    matches = [kind for kind in _generation(generation) if ADAPTERS[kind].accepts(path)]

    if len(matches) != 1:
        raise ValueError("지원하는 입력 형식이 아니거나 유형을 하나로 정할 수 없습니다.")

    return matches[0]


def input_specs(*, generation="legacy"):
    """요청 화면의 입력 안내. 선택한 세대의 입력 형식만 나열한다."""
    return [dict(ADAPTERS[kind].INPUT, kind_id=kind) for kind in _generation(generation)]
