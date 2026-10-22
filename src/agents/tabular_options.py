"""표형 전처리 옵션의 쉬운 설명과 확정 전 사전 경고. 화면·보고서 표시 전용이며 계획 값을 바꾸지 않는다.

설명 문구는 고정이다. 적용 대상 수와 사전 경고는 로컬 진단 집계(전체 입력 기준)에서 계산한 사실이고,
모델 의견(권장 이유)은 따로 표시한다. 실제 영향은 Train 후보 계산에서 다시 집계한다. 모델에 보내지 않는다.
설계 기준: docs/agents/design/CSV_TASK_AGENT.md 7.22
"""

from . import llm_task_assist as rules


# ==========================================
# 옵션 설명 (고정 문구)
# - what: 처리 내용 / caution: 주의사항 / off: 사용하지 않을 때의 동작
# - target: 적용 대상 역할(numeric·categorical·features·output)
# - values: 선택값의 쉬운 이름. 없는 값은 원래 값을 그대로 보여 준다.
# ==========================================
GUIDE = {
    "missing_ratio_threshold": {
        "label": "결측이 많은 특징 제거", "target": "features",
        "what": "Train에서 결측 비율이 이 값을 넘는 특징을 입력 특징에서 뺍니다.",
        "caution": "Train 기준이라 전체 입력의 결측률과 다를 수 있습니다. 결측 자체가 의미 있는 정보라면 함께 사라집니다.",
        "off": "비워 두면(null) 결측 비율로는 제거하지 않고 대치 설정을 따릅니다.", "values": {None: "사용 안 함"}},
    "drop_constant": {
        "label": "값이 하나뿐인 특징 제거", "target": "features",
        "what": "Train에서 관측값이 한 종류뿐이거나 전부 결측인 특징을 입력 특징에서 뺍니다.",
        "caution": "Train 기준 판정입니다. Validation에서만 다른 값이 나타나는 특징도 제거됩니다.",
        "off": "끄면 상수 특징도 그대로 출력합니다(학습에 정보는 주지 않음).", "values": {True: "사용", False: "사용 안 함"}},
    "numeric_imputation": {
        "label": "수치 결측 채우기", "target": "numeric",
        "what": "수치 특징의 빈 값을 Train에서 계산한 대표값으로 채웁니다.",
        "caution": "채운 값은 실제 측정값이 아닙니다. 결측이 많은 특징은 대표값 쪽으로 분포가 몰립니다.",
        "off": "none이면 빈 값이 출력 파일에 빈 칸으로 남습니다. 구간화는 결측 대치가 필요합니다.",
        "values": {"median": "Train 중앙값", "mean": "Train 평균", "none": "채우지 않음"}},
    "outlier": {
        "label": "극단값 자르기(clipping)", "target": "numeric",
        "what": "Train 사분위 범위(IQR)로 정한 하한·상한 밖의 값을 경계값으로 바꿉니다. 행은 지우지 않습니다.",
        "caution": "값 종류가 적은 특징(예: 대부분 0인 0/1 특징)은 하한과 상한이 같아져 모든 값이 하나로 바뀔 수 있습니다.",
        "off": "none이면 값을 바꾸지 않습니다. 극단값은 스케일링·학습에 그대로 영향을 줍니다.",
        "values": {"none": "사용 안 함", "iqr_clip": "IQR 경계로 자르기"}},
    "iqr_multiplier": {
        "label": "clipping 경계 폭(IQR 배수)", "target": "numeric",
        "what": "사분위 범위에 곱해 하한·상한을 정합니다. 클수록 덜 자릅니다.",
        "caution": "사분위 범위가 0인 특징은 배수와 관계없이 경계 폭이 0입니다.",
        "off": "clipping을 사용하지 않으면 이 값은 쓰이지 않습니다."},
    "scaling": {
        "label": "수치 크기 맞추기(스케일링)", "target": "numeric",
        "what": "수치 특징을 Train 기준 중심·폭으로 나눠 크기를 맞춥니다.",
        "caution": "구간화와 함께 쓸 수 없습니다. 값의 순서·상대 거리는 유지되지만 원래 단위는 사라집니다.",
        "off": "none이면 원래 단위를 유지합니다.",
        "values": {"none": "사용 안 함", "standard": "평균 0·표준편차 1", "minmax": "최솟값 0·최댓값 1", "robust": "중앙값·IQR 기준"}},
    "binning": {
        "label": "수치 구간화", "target": "numeric",
        "what": "수치 특징을 Train 기준 구간 번호(또는 구간별 0/1 열)로 바꿉니다.",
        "caution": "구간 안의 차이는 사라집니다. 값 종류가 적으면 경계가 합쳐져 구간이 줄거나 한 구간(상수)만 남을 수 있습니다.",
        "off": "none이면 구간화하지 않고 수치를 그대로(스케일링 설정에 따라) 출력합니다.",
        "values": {"none": "사용 안 함", "quantile": "같은 개수씩(분위수)", "uniform": "같은 폭으로"}},
    "n_bins": {
        "label": "구간 수", "target": "numeric",
        "what": "구간화할 때 나눌 최대 구간 수입니다. 같은 경계는 합쳐 실제 구간 수가 줄 수 있습니다.",
        "caution": "원핫 출력이면 특징마다 최대 이 수만큼 열이 생깁니다.", "off": "구간화를 사용하지 않으면 쓰이지 않습니다."},
    "bin_encoding": {
        "label": "구간 표현 방식", "target": "numeric",
        "what": "ordinal은 구간 번호 한 열, onehot은 구간마다 0/1 열을 만듭니다.",
        "caution": "onehot은 출력 열 수가 구간 수만큼 늘어납니다.", "off": "구간화를 사용하지 않으면 쓰이지 않습니다.",
        "values": {"ordinal": "구간 번호 한 열", "onehot": "구간별 0/1 열"}},
    "categorical_imputation": {
        "label": "범주 결측 채우기", "target": "categorical",
        "what": "범주 특징의 빈 값을 '결측' 범주로 두거나 Train 최빈값으로 채웁니다.",
        "caution": "최빈값으로 채우면 결측이었다는 정보가 사라집니다.",
        "off": "항상 둘 중 하나가 적용됩니다(범주 특징은 원핫 출력).",
        "values": {"missing_category": "'결측' 범주로 따로 둠", "most_frequent": "Train 최빈값"}},
    "unknown_category": {
        "label": "Train에 없던 범주 처리", "target": "categorical",
        "what": "Validation 등에 Train에 없던 범주가 나오면 적용을 멈추거나(error) 모든 범주 열을 0으로 둡니다(ignore).",
        "caution": "error는 값이 행마다 다른 범주 특징(ID 등)이 있으면 적용 단계에서 실패합니다. ignore는 새 범주 정보를 버립니다.",
        "off": "항상 둘 중 하나가 적용됩니다.", "values": {"error": "적용 중단", "ignore": "0으로 두고 계속"}},
    "max_categories": {
        "label": "범주 수 상한", "target": "categorical",
        "what": "특징 하나의 원핫 범주 수가 이 값을 넘으면 후보 계산을 멈춥니다.",
        "caution": "고유값이 매우 많은 범주 특징은 식별자나 제외로 바꿔야 진행할 수 있습니다.", "off": "끌 수 없는 안전 상한입니다."},
    "max_output_features": {
        "label": "출력 열 수 상한", "target": "output",
        "what": "전처리 후 출력 열 수가 이 값을 넘으면 후보 계산을 멈춥니다.", "caution": "원핫 범주·구간이 많으면 넘을 수 있습니다.",
        "off": "끌 수 없는 안전 상한입니다."},
    "max_output_cells": {
        "label": "출력 셀 수 상한", "target": "output",
        "what": "Train 행 수 × 출력 열 수가 이 값을 넘으면 후보 계산을 멈춥니다.", "caution": "넓은 표에서 원핫을 쓰면 넘을 수 있습니다.",
        "off": "끌 수 없는 안전 상한입니다."},
}
TARGETS = {"numeric": "수치형 특징", "categorical": "범주형 특징", "features": "모든 입력 특징", "output": "전체 출력"}
SOURCES = {"base": "기본 계획", "model": "모델 권장", "user": "사용자 변경", "answer": "질문 답변", "previous": "직전 확정 계획"}
BASIC = ("numeric_imputation", "outlier", "scaling", "binning", "categorical_imputation", "unknown_category", "drop_constant",
         "missing_ratio_threshold")


def value_text(name, value):
    """선택값의 쉬운 이름. 설명이 없는 값은 원래 값을 그대로 쓴다."""
    labels = GUIDE.get(name, {}).get("values", {})
    return labels.get(value, str(value)) if (value is None or type(value) in (str, bool)) else str(value)


def _names(items, limit=5):
    items = list(items)
    return ", ".join(items[:limit]) + (f" 외 {len(items) - limit}개" if len(items) > limit else "")


def _facts(name, preparation, numeric, categorical, features, summary, columns, plan):
    """진단 집계로 확인한 사실과 사전 경고. 진단은 전체 입력 기준이며 Train과 다를 수 있다."""
    by_name = {row["column"]: row for row in columns}
    facts, warnings = [], []

    if name == "numeric_imputation":
        missing = [column for column in numeric if by_name[column]["missing_count"]]
        facts.append(f"결측이 있는 수치형 특징 {len(missing)}개" + (f" ({_names(missing)})" if missing else ""))
    elif name == "categorical_imputation":
        missing = [column for column in categorical if by_name[column]["missing_count"]]
        facts.append(f"결측이 있는 범주형 특징 {len(missing)}개" + (f" ({_names(missing)})" if missing else ""))
    elif name == "outlier":
        flagged = [column for column in numeric if by_name[column].get("outlier_candidate_count")]
        facts.append(f"IQR 1.5 기준 이상치 후보가 있는 수치형 특징 {len(flagged)}개" + (f" ({_names(flagged)})" if flagged else ""))
        binary = [column for column in flagged if by_name[column]["observed_unique"] == 2]

        if binary and preparation["outlier"] == "iqr_clip":
            warnings.append(f"값이 두 종류뿐인 수치 특징 {len(binary)}개({_names(binary)})에 이상치 후보가 있습니다. clipping하면 적은 쪽 값이 "
                            "바뀌어 두 값의 구분이 줄거나 모든 값이 하나(상수)가 될 수 있습니다. 실제 영향은 Train 후보 계산에서 확인합니다.")
    elif name == "binning" and preparation["binning"] != "none":
        few = [column for column in numeric if (by_name[column]["observed_unique"] or 0) <= preparation["n_bins"]]

        if few:
            warnings.append(f"값 종류가 구간 수({preparation['n_bins']}) 이하인 수치 특징 {len(few)}개({_names(few)})는 구간 경계가 합쳐져 "
                            "구간이 줄거나 한 구간(상수)만 남을 수 있습니다. 실제 구간 수는 Train 후보 계산에서 확인합니다.")
    elif name == "drop_constant":
        constant = [column for column in features if (by_name[column]["observed_unique"] or 0) <= 1]
        facts.append(f"관측값이 한 종류 이하인 입력 특징 {len(constant)}개" + (f" ({_names(constant)})" if constant else ""))
    elif name == "missing_ratio_threshold" and preparation["missing_ratio_threshold"] is not None:
        high = [column for column in features if by_name[column]["missing_ratio"] > preparation["missing_ratio_threshold"]]
        facts.append(f"전체 입력에서 결측 비율이 기준을 넘는 특징 {len(high)}개" + (f" ({_names(high)})" if high else ""))
    elif name == "unknown_category" and preparation["unknown_category"] == "error":
        risky = rules.identifier_like(summary, columns, plan["roles"])

        if risky:
            warnings.append(f"값이 행마다 다른 범주 특징 {len(risky)}개({_names(risky)})가 있어 적용 단계에서 Train에 없던 범주로 실패할 수 있습니다.")
    elif name == "max_categories":
        over = rules.category_limit_risk(summary, columns, plan)

        if over:
            warnings.append(f"범주 수가 상한을 넘을 수 있는 특징 {len(over)}개({_names(item['column'] for item in over)})가 있습니다.")
    elif name == "max_output_features":
        width = estimate_width(preparation, numeric, categorical, by_name)
        facts.append(f"예상 출력 열 수 상한 {width}개(구간·범주가 합쳐지면 줄어듦)")

        if width > preparation["max_output_features"]:
            warnings.append(f"예상 출력 열 수 상한({width})이 설정 상한({preparation['max_output_features']})보다 큽니다. 후보 계산이 멈출 수 있습니다.")

    return facts, warnings


def estimate_width(preparation, numeric, categorical, by_name):
    """진단 고유값으로 본 출력 열 수의 상한. 실제 값은 Train 후보 계산의 출력 폭이다."""
    per_numeric = preparation["n_bins"] if preparation["binning"] != "none" and preparation["bin_encoding"] == "onehot" else 1
    extra = 1 if preparation["categorical_imputation"] == "missing_category" else 0
    return len(numeric) * per_numeric + sum((by_name[column]["observed_unique"] or 0) + extra for column in categorical)


def option_guide(record, plan, origins):
    """확정 전 화면의 옵션별 설명. 처리 내용·적용 대상·계산 사실·모델 의견·주의·미사용 동작·사전 경고를 돌려준다."""
    preparation, roles = plan["preparation"], plan["roles"]
    summary, columns = record["diagnostic"]["summary"], record["diagnostic"]["columns"]
    numeric = [name for name, role in roles.items() if role == "수치형"]
    categorical = [name for name, role in roles.items() if role == "범주형"]
    features = numeric + categorical
    counts = {"numeric": len(numeric), "categorical": len(categorical), "features": len(features), "output": None}
    result = []

    for name in rules.PREPARATION_FIELDS:
        spec = GUIDE[name]
        origin = origins.get("preparation." + name, {"source": "base"})
        facts, warnings = _facts(name, preparation, numeric, categorical, features, summary, columns, plan)
        count = counts[spec["target"]]
        result.append({"path": "preparation." + name, "name": name, "label": spec["label"], "value": preparation[name],
                       "value_text": value_text(name, preparation[name]), "basic": name in BASIC,
                       "source": SOURCES.get(origin["source"], origin["source"]),
                       "applies": TARGETS[spec["target"]] + ("" if count is None else f" {count}개"),
                       "what": spec["what"], "caution": spec["caution"], "off": spec["off"],
                       "facts": facts, "opinion": origin.get("reason"), "warnings": warnings})

    return result
