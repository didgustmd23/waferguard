"""표형 작업의 계획 편집 부품. 역할 표와 분할·전처리 입력만 그리고, 검증·확정은 공통 화면이 조정기에 맡긴다."""

import pandas as pd

from src.agents.profile_store import ROLE_LABELS
from src.agents.tabular_options import GUIDE, value_text


SHOWN = ("column", "dtype_group", "missing_ratio", "observed_unique", "finite_count")
# 모델 질문과 관계없이 모든 전처리 방법을 고치거나 끌 수 있게 한다. 안전 상한은 접힌 고급 영역에 둔다.
OPTIONS = ("numeric_imputation", "outlier", "scaling", "binning", "bin_encoding", "categorical_imputation", "unknown_category")
LIMITS = ("max_categories", "max_output_features", "max_output_cells")


def render(st, record, plan, fields, key):
    """현재 계획을 편집 양식으로 그리고 편집한 계획을 돌려준다. 양식 제출 전에는 아무것도 실행하지 않는다."""
    by_path = {field["path"]: field for field in fields}
    columns = record["diagnostic"]["columns"]
    table = pd.DataFrame([{**{name: row[name] for name in SHOWN}, "role": plan["roles"][row["column"]]} for row in columns])
    edited = st.data_editor(table, key=key + "_roles", hide_index=True, disabled=list(SHOWN),
                            column_config={"role": st.column_config.SelectboxColumn("역할", options=list(ROLE_LABELS), required=True)})
    split, preparation = dict(plan["split"]), dict(plan["preparation"])
    strategy = by_path["split.strategy"]
    strategies = [option for option in strategy["options"] if option not in strategy.get("unsupported", [])]
    ui = st.columns(3)
    split["strategy"] = ui[0].selectbox("분할 방식", strategies, placeholder="분할 방식 선택", key=key + "_strategy",
                                        index=strategies.index(split["strategy"]) if split["strategy"] in strategies else None)
    split["train_ratio"] = ui[1].number_input("Train 비율", 0.01, 0.98, float(split["train_ratio"]), 0.05, key=key + "_train")
    split["validation_ratio"] = ui[2].number_input("Validation 비율", 0.01, 0.98, float(split["validation_ratio"]), 0.05, key=key + "_validation")
    split["test_ratio"] = ui[0].number_input("Test 비율", 0.01, 0.98, float(split["test_ratio"]), 0.05, key=key + "_test")
    split["random_state"] = int(ui[1].number_input("난수 시드", 0, 2 ** 32 - 1, int(split["random_state"]), key=key + "_seed"))
    split["stratify"] = ui[2].checkbox("target 기준 계층 분할", value=split["stratify"], key=key + "_stratify")
    # 기존 분할 계약의 클래스 조건. 끄면 target 클래스가 빠진 구간도 그대로 진행하고, 분할 결과의 구간별 클래스 수로 확인한다.
    split["require_all_classes"] = ui[0].checkbox("모든 구간에 모든 클래스 요구", value=split["require_all_classes"],
                                                  key=key + "_require_all_classes",
                                                  help="켜면(기본) Train·Validation·Test 중 target 클래스가 빠진 구간이 생길 때 분할 "
                                                       "제안에서 멈춥니다. 끄면 그런 구간도 진행하며, 작은 데이터·group 분할에서 "
                                                       "평가에 쓸 클래스가 없을 수 있습니다.")

    if strategy.get("unsupported"):
        st.text("time 분할은 통합 작업에서 지원하지 않습니다(시간 형식 지정 불가). group 역할이 있으면 group, 없으면 random을 고르세요.")

    time = _time_inputs(st, plan, by_path, key, split["strategy"]) if "time.column" in by_path else None

    st.caption("전처리 방법은 질문과 관계없이 고치거나 끌 수 있습니다(none·사용 안 함). 각 방법의 설명은 위 '옵션 설명'에 있습니다.")
    ui = st.columns(3)

    for index, name in enumerate(OPTIONS):
        choices = by_path["preparation." + name]["options"]
        preparation[name] = ui[index % 3].selectbox(GUIDE[name]["label"] + f" ({name})", choices, index=choices.index(preparation[name]),
                                                    format_func=lambda value, name=name: value_text(name, value) + f" [{value}]",
                                                    key=key + "_" + name)

    preparation["iqr_multiplier"] = float(ui[0].number_input(GUIDE["iqr_multiplier"]["label"], 0.01, 100.0,
                                                             float(preparation["iqr_multiplier"]), 0.5, key=key + "_iqr_multiplier"))
    bins = by_path["preparation.n_bins"]
    preparation["n_bins"] = int(ui[1].number_input(GUIDE["n_bins"]["label"], int(bins["minimum"]), int(bins["maximum"]),
                                                   int(preparation["n_bins"]), key=key + "_n_bins"))
    preparation["drop_constant"] = st.checkbox("값이 하나뿐인 특징 제거(Train 기준)", value=preparation["drop_constant"],
                                               key=key + "_drop_constant")
    threshold = preparation["missing_ratio_threshold"]

    if st.checkbox(GUIDE["missing_ratio_threshold"]["label"] + "(Train 결측 비율 기준)", value=threshold is not None,
                   key=key + "_missing_threshold_on"):
        preparation["missing_ratio_threshold"] = float(st.number_input("제거 기준 결측 비율(초과 시 제거)", 0.0, 1.0,
                                                                       0.5 if threshold is None else float(threshold), 0.05,
                                                                       key=key + "_missing_threshold"))
    else:
        preparation["missing_ratio_threshold"] = None

    with st.expander("고급: 안전 상한(끌 수 없음)"):
        for name in LIMITS:
            field = by_path["preparation." + name]
            preparation[name] = int(st.number_input(GUIDE[name]["label"] + f" ({name})", int(field["minimum"]), int(field["maximum"]),
                                                    int(preparation[name]), key=key + "_" + name))
    roles = dict(zip(edited["column"], edited["role"]))
    flagged = sorted({item["subject"] for item in ((record.get("problem") or {}).get("findings") or [])
                      if item["code"] != "CLASS_MISSING_IN_SPLIT" and item["subject"] in roles
                      and roles[item["subject"]] in ("수치형", "범주형")})

    # 품질 검사가 지적한 특징을 사용자가 고를 때만 제외로 바꾼다. 기본은 바꾸지 않으며 서버가 계획 전체를 다시 검증한다.
    if flagged and st.checkbox(f"품질 검사가 지적한 특징 {len(flagged)}개를 제외로 표시합니다", value=False, key=key + "_exclude_flagged"):
        roles.update({name: "제외" for name in flagged})

    result = {"roles": roles, "split": split, "preparation": preparation}

    if time is not None:
        result["time"] = time

    return result


FREQUENCY_LABELS = {"daily": "매일 (같은 현지 시각, 달력상 하루 간격)", "weekdays": "평일 (월~금, 같은 현지 시각)",
                    "fixed": "고정 간격 (분 단위)"}


def _time_inputs(st, plan, by_path, key, strategy):
    """시간 분할 설정 입력. 사용자가 직접 정하며 모델 권장값이 없다. time 분할이 아니면 입력값을 계획에 넣지 않고 기본값을 돌려준다."""
    names = [path.split(".", 1)[1] for path in by_path if path.startswith("time.")]
    default = {name: False if by_path["time." + name]["type"] == "boolean" else None for name in names}
    current = dict(default, **(plan.get("time") or {}))

    with st.expander("시간 분할 설정 (분할 방식 time일 때만 사용)", expanded=strategy == "time"):
        st.caption("모델은 시간 설정을 정하지 않습니다. 직접 지정하세요. 결측 시각은 앞뒤 유효 시각 사이에서 지정한 주기·행 순서로 하나로 "
                   "정해질 때만 복원 대상이 되며, 앞뒤 끝 결측·중복 시각·주기 불일치·서머타임 모호성은 복원하지 않고 멈춥니다. "
                   "이번 단계는 설정 검증까지이며 실제 복원·time 분할 실행은 하지 않습니다.")
        columns = by_path["time.column"]["options"]
        time = {"column": st.selectbox("시간 컬럼 (timestamp 역할로 지정한 컬럼)", columns, placeholder="시간 컬럼 선택",
                                       index=columns.index(current["column"]) if current["column"] in columns else None,
                                       key=key + "_time_column")}
        text = st.text_input("시간 형식 (예: %Y-%m-%d %H:%M · 허용: %Y %y %m %d %H %M %S %f %z)", value=current["timestamp_format"] or "",
                             max_chars=by_path["time.timestamp_format"]["maxLength"], key=key + "_time_format")
        time["timestamp_format"] = text.strip() or None
        zone = st.text_input("시간대 (IANA 이름, 예: Asia/Seoul · 형식에 %z가 있으면 비움)", value=current["timezone"] or "",
                             max_chars=by_path["time.timezone"]["maxLength"], key=key + "_time_zone")
        time["timezone"] = zone.strip() or None
        frequencies = by_path["time.frequency"]["options"]
        time["frequency"] = st.selectbox("복원 주기", frequencies, placeholder="복원 주기 선택", format_func=lambda value: FREQUENCY_LABELS.get(value, value),
                                         index=frequencies.index(current["frequency"]) if current["frequency"] in frequencies else None,
                                         key=key + "_time_frequency")
        interval = by_path["time.interval_minutes"]
        minutes = int(st.number_input("고정 간격(분) — 고정 간격일 때만 사용", 0, int(interval["maximum"]),
                                      int(current["interval_minutes"] or 0), key=key + "_time_interval"))
        time["interval_minutes"] = minutes if time["frequency"] == "fixed" and minutes else None
        time["row_order_confirmed"] = st.checkbox("원본 행 순서가 시간 순서이고, 각 행이 주기상 한 시점임을 확인합니다",
                                                  value=current["row_order_confirmed"], key=key + "_time_row_order")

    return time if strategy == "time" else default
