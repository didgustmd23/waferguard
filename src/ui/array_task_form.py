"""배열 작업의 계획 편집 부품. 매핑·분할·전처리 입력만 그리고, 검증·확정은 공통 화면이 조정기에 맡긴다.

배열 이름은 입력 자료에서 온 문자열이므로 선택지 값으로만 쓰고 위젯 이름(마크다운 해석)에는 넣지 않는다.
"""

from src.agents.array_options import GUIDE, value_text


NONE = "(없음)"


def _choice(st, column, path, field, value, key):
    options = ([None] if field.get("nullable") else []) + list(field["options"])
    label = GUIDE[path]["label"] if path in GUIDE else path
    return column.selectbox(label, options, index=options.index(value) if value in options else None, key=key,
                            format_func=lambda item: NONE if item is None else value_text(path, item) if path in GUIDE
                            and "values" in GUIDE[path] else str(item))


def render(st, record, plan, fields, key):
    """현재 계획을 편집 양식으로 그리고 편집한 계획을 돌려준다. 양식 제출 전에는 아무것도 실행하지 않는다."""
    by_path = {field["path"]: field for field in fields}
    mapping, split, preparation = dict(plan["mapping"]), dict(plan["split"]), dict(plan["preparation"])
    st.caption("매핑: 표본 배열과 라벨·그룹·표본 ID 배열을 고릅니다. 라벨·그룹·ID는 특징이 아니라 메타데이터로 보존합니다.")
    ui = st.columns(3)

    for index, name in enumerate(("samples", "channel_axis", "label", "group", "sample_id")):
        path = "mapping." + name
        mapping[name] = _choice(st, ui[index % 3], path, by_path[path], mapping[name], key + "_" + name)

    st.caption("분할: 배열 값은 보지 않고 표본 메타데이터로만 나눕니다. 시간 기준 분할은 지원하지 않습니다.")
    ui = st.columns(3)
    split["strategy"] = _choice(st, ui[0], "split.strategy", by_path["split.strategy"], split["strategy"], key + "_strategy")
    split["train_ratio"] = ui[1].number_input("Train 비율", 0.01, 0.98, float(split["train_ratio"]), 0.05, key=key + "_train")
    split["validation_ratio"] = ui[2].number_input("Validation 비율", 0.01, 0.98, float(split["validation_ratio"]), 0.05, key=key + "_validation")
    split["test_ratio"] = ui[0].number_input("Test 비율", 0.01, 0.98, float(split["test_ratio"]), 0.05, key=key + "_test")
    split["random_state"] = int(ui[1].number_input("난수 시드", 0, 2 ** 32 - 1, int(split["random_state"]), key=key + "_seed"))
    split["stratify"] = ui[2].checkbox("라벨 기준 계층 분할", value=split["stratify"], key=key + "_stratify")
    st.caption("전처리: 모든 방법은 질문과 관계없이 고치거나 끌 수 있습니다. 기준값은 Train에서만 채널별로 계산합니다.")
    ui = st.columns(3)

    for index, name in enumerate(("nonfinite", "clip", "scaling", "missing_labels")):
        path = "preparation." + name
        preparation[name] = _choice(st, ui[index % 3], path, by_path[path], preparation[name], key + "_" + name)

    preparation["clip_low"] = float(ui[1].number_input("clipping 하한 분위수(%)", 0.0, 49.0, float(preparation["clip_low"]), 0.5, key=key + "_clip_low"))
    preparation["clip_high"] = float(ui[2].number_input("clipping 상한 분위수(%)", 51.0, 100.0, float(preparation["clip_high"]), 0.5, key=key + "_clip_high"))
    preparation["fixed_divisor"] = float(ui[0].number_input("고정 나눗값(fixed)", 1e-6, 1e9, float(preparation["fixed_divisor"]), key=key + "_divisor"))
    preparation["duplicate_grouping"] = st.checkbox(GUIDE["preparation.duplicate_grouping"]["label"], value=preparation["duplicate_grouping"],
                                                    key=key + "_duplicate_grouping",
                                                    help="같은 표본을 한 구간에 모아 누수를 막습니다. 표본은 지우지 않습니다.")
    return {"mapping": mapping, "split": split, "preparation": preparation}
