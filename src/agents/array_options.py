"""배열 작업 계획 항목의 쉬운 설명과 확정 전 사전 경고. 화면·보고서 표시 전용이며 계획 값을 바꾸지 않는다.

설명 문구는 고정이다. 적용 대상 수와 사전 경고는 로컬 진단 요약(전체 입력 기준)에서 계산한 사실이고,
모델 의견(권장 이유)은 따로 표시한다. 실제 영향은 Train 후보 계산에서 다시 집계한다. 모델에 보내지 않는다.
"""

import numpy as np


# 결과는 항상 float32로 저장한다(dtype 보존 기능 없음). 모든 옵션이 none이어도 값이 바뀔 수 있다.
FLOAT32_NOTE = ("결과는 모든 옵션이 none이어도 float32로 저장합니다. float64·큰 정수(약 1,677만 초과) 입력은 값이 반올림되거나 "
                "서로 다른 값이 같은 값이 될 수 있습니다.")


# ==========================================
# 항목 설명 (고정 문구)
# - what: 처리 내용 / caution: 주의 / off: 쓰지 않거나 기본값일 때의 동작
# ==========================================
GUIDE = {
    "mapping.samples": {"label": "표본 배열", "what": "첫 축이 표본인 숫자 배열을 학습 입력으로 씁니다.",
                        "caution": "표본 수가 다른 1차원 배열은 라벨·그룹으로 고를 수 없습니다.", "off": "고르지 않으면 진행할 수 없습니다."},
    "mapping.channel_axis": {"label": "채널 축", "what": "채널별 기준(대치·clipping·크기 조정)을 계산할 축입니다.",
                             "caution": "축을 잘못 고르면 공간 위치마다 따로 기준이 생기거나 채널이 섞입니다.",
                             "off": "none이면 표본 전체 값을 한 채널로 봅니다.",
                             "values": {"none": "채널 없음(전체 한 채널)", "last": "마지막 축", "first": "두 번째 축"}},
    "mapping.label": {"label": "라벨 배열", "what": "표본별 정답(분류 등)을 특징이 아닌 메타데이터로 보존합니다.",
                      "caution": "모델에는 클래스 이름 대신 순번과 개수만 보냅니다. 모델은 클래스 의미를 해석하지 못합니다.",
                      "off": "없음이면 라벨 없이 처리하며 계층 분할을 쓸 수 없습니다."},
    "mapping.group": {"label": "그룹 배열", "what": "같은 그룹(예: 같은 대상·묶음)의 표본을 한 구간에 모읍니다.",
                      "caution": "그룹이 비어 있는 표본이 있으면 진행하지 않습니다.", "off": "없음이면 표본 단위로 나눕니다."},
    "mapping.sample_id": {"label": "표본 ID 배열", "what": "결과 메타데이터에 표본 ID로 보존합니다.",
                          "caution": "비어 있거나 중복이면 진행하지 않습니다.", "off": "없음이면 원래 위치 번호를 ID로 씁니다."},
    "split.strategy": {"label": "분할 방식", "what": "Train·Validation·Test를 나누는 방식입니다. 배열 값은 보지 않고 메타데이터로만 나눕니다.",
                       "caution": "group은 그룹 배열이 필요합니다. 시간 기준 분할은 지원하지 않습니다.", "off": "기본은 random입니다.",
                       "values": {"random": "무작위", "group": "그룹 단위"}},
    "preparation.nonfinite": {"label": "비유한 값(NaN·무한대)", "what": "Train 채널 중앙값으로 채우거나, 있으면 멈춥니다.",
                              "caution": "채운 값은 실제 측정값이 아닙니다.", "off": "block이면 비유한 값이 있을 때 진행하지 않습니다.",
                              "values": {"block": "있으면 중단", "median": "Train 채널 중앙값으로 채움"}},
    "preparation.clip": {"label": "극단값 자르기(clipping)", "what": "Train 채널 분위수 하한·상한 밖의 값을 경계값으로 바꿉니다. 표본은 지우지 않습니다.",
                         "caution": "값 종류가 적은 채널(예: 0/1 맵)은 하한과 상한이 같아져 모든 값이 하나가 될 수 있습니다.",
                         "off": "none이면 값을 바꾸지 않습니다.", "values": {"none": "사용 안 함", "percentile": "Train 분위수 경계"}},
    "preparation.scaling": {"label": "값 크기 조정", "what": "채널별로 Train 기준 표준화·정규화하거나 고정 값으로 나눕니다.",
                            "caution": "fixed는 Train 통계를 쓰지 않습니다(예: 0~255 영상을 255로 나눔). " + FLOAT32_NOTE,
                            "off": "none이면 원래 값 범위를 유지하지만 float32로 저장하므로 정밀도가 줄 수 있습니다.",
                            "values": {"none": "사용 안 함", "standard": "Train 평균 0·표준편차 1", "minmax": "Train 최솟값 0·최댓값 1",
                                       "fixed": "고정 값으로 나눔"}},
    "preparation.missing_labels": {"label": "라벨 없는 표본", "what": "라벨이 비어 있는 표본을 분할 전에 제외하거나 라벨 없이 유지합니다.",
                                   "caution": "제외는 결과에서 빠질 뿐 원본을 지우지 않습니다. 유지하면 계층 분할을 쓸 수 없습니다.",
                                   "off": "라벨 결측이 있으면 반드시 골라야 합니다.", "values": {"exclude": "분할 전 제외", "keep": "라벨 없이 유지"}},
    "preparation.duplicate_grouping": {"label": "누수 방지: 같은 표본을 한 구간에 모으기",
                                       "what": "값이 완전히 같은 표본을 한 묶음으로 보아 같은 구간에 둡니다(그룹 단위 분할).",
                                       "caution": "켜면 계층 분할을 쓸 수 없습니다. 표본을 지우지 않습니다. 근사 중복은 찾지 않습니다.",
                                       "off": "끄면 같은 표본이 Train과 Validation·Test에 나뉠 수 있고 결과에 누수 경고를 남깁니다.",
                                       "values": {True: "사용", False: "사용 안 함"}},
}
SOURCES = {"base": "기본 계획", "model": "모델 권장", "user": "사용자 변경", "answer": "질문 답변", "previous": "직전 확정 계획"}
PREPARATION = ("nonfinite", "clip", "scaling", "missing_labels", "duplicate_grouping")


def value_text(path, value):
    labels = GUIDE.get(path, {}).get("values", {})
    return "없음" if value is None else labels.get(value, str(value)) if type(value) in (str, bool) else str(value)


def _sample_stats(record, plan):
    key = plan["mapping"]["samples"]
    return None if key is None else record["diagnostic"]["summary"]["samples"].get(key)


def _facts(path, record, plan):
    """진단 요약으로 확인한 사실과 사전 경고. 진단은 전체 입력 기준이며 Train과 다를 수 있다."""
    summary, mapping, options = record["diagnostic"]["summary"], plan["mapping"], plan["preparation"]
    stats = _sample_stats(record, plan)
    facts, warnings = [], []

    if stats is None:
        return facts, warnings

    overall = stats["overall"]

    if path == "preparation.nonfinite":
        facts.append(f"표본 배열의 비유한 값 {overall['nonfinite']}개")
    elif path == "preparation.clip":
        facts.append(f"표본 배열 값 종류 {overall['distinct'] if overall['distinct'] <= 256 else '256 초과'}")

        if options["clip"] == "percentile" and overall["distinct"] <= 2:
            warnings.append("값 종류가 2개 이하입니다. 분위수 clipping의 하한·상한이 같아지면 모든 값이 하나(상수)가 될 수 있습니다. "
                            "실제 영향은 Train 후보 계산에서 확인합니다.")
    elif path == "preparation.duplicate_grouping":
        facts.append(f"완전히 같은 표본 {stats['duplicate_samples']}개(첫 표본 제외 개수)")

        if stats["duplicate_samples"] and not options["duplicate_grouping"]:
            warnings.append("같은 표본이 서로 다른 구간에 들어가면 평가가 낙관적으로 나올 수 있습니다(누수). "
                            "'같은 표본을 한 구간에 모으기'를 켜면 막을 수 있습니다. 설정은 자동으로 바꾸지 않았습니다.")
    elif path == "preparation.scaling":
        facts.append(f"상수 표본 {stats['constant_samples']}개")
        header = summary["headers"][mapping["samples"]]
        largest = max(abs(overall["min"] or 0), abs(overall["max"] or 0))
        facts.append(f"입력 자료형 {header['dtype']}")

        # float32로 정확히 담기지 않는 입력이면 값과 무관하게 사전 경고한다. 설정은 바꾸지 않는다.
        if header["kind"] == "f" and np.dtype(header["dtype"]).itemsize > 4 or header["kind"] in "iu" and largest > 2 ** 24:
            warnings.append("입력이 float32보다 정밀한 자료형이거나 큰 정수입니다. 저장 결과에서 값이 반올림되거나 서로 다른 값이 같아질 수 "
                            "있습니다(dtype 보존 기능 없음).")
    elif path == "preparation.missing_labels" and mapping["label"] is not None:
        facts.append(f"라벨 결측 {summary['meta'][mapping['label']]['missing']}개")
    elif path == "mapping.label" and mapping["label"] is not None:
        meta = summary["meta"][mapping["label"]]
        facts.append(f"클래스 {meta['unique']}개(모델에는 순번과 개수만 전달)")
    elif path == "mapping.group" and mapping["group"] is not None:
        sizes = summary["meta"][mapping["group"]]["group_sizes"]
        facts.append(f"그룹 {summary['meta'][mapping['group']]['unique']}개" + (f", 그룹당 표본 {sizes['min']}~{sizes['max']}" if sizes else ""))

    return facts, warnings


def option_guide(record, plan, origins):
    """확정 전 화면의 항목별 설명. 처리 내용·대상·계산 사실·모델 의견·주의·기본 동작·사전 경고를 돌려준다."""
    result = []

    for path, spec in GUIDE.items():
        section, name = path.split(".", 1)
        value = plan[section][name]
        origin = origins.get(path, {"source": "base"})
        facts, warnings = _facts(path, record, plan)
        result.append({"path": path, "name": name, "label": spec["label"], "value": value, "value_text": value_text(path, value),
                       "basic": True, "source": SOURCES.get(origin["source"], origin["source"]),
                       "applies": "표본 배열" if section != "mapping" else "매핑", "what": spec["what"], "caution": spec["caution"],
                       "off": spec["off"], "facts": facts, "opinion": origin.get("reason"), "warnings": warnings})

    return result
