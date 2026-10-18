"""에이전트·파일 저장·그래프와 독립적인 표형 EDA 계산 기능."""

import pandas as pd

from .contracts import TabularInput


def analyze_summary(data: TabularInput):
    """입력 역할과 행·그룹·중복 특징 행의 기본 집계를 반환한다."""
    data.validate()
    frame, roles = data.frame, data.roles
    features = frame.loc[:, list(roles.feature_columns)]

    return {
        "row_count": len(frame),
        "feature_count": len(roles.feature_columns),
        "numeric_feature_count": len(roles.numeric),
        "categorical_feature_count": len(roles.categorical),
        "repeated_feature_rows": int(features.duplicated(keep=False).sum()),
        "group_count": int(frame[roles.group].nunique()) if roles.group else None,
        "target_present": roles.target is not None,
    }


def analyze_missing(data: TabularInput):
    """지정 특징의 결측 수·비율·관측 고유값을 계산한다. 대치는 하지 않는다."""
    data.validate()
    features = data.frame.loc[:, list(data.roles.feature_columns)]
    table = pd.DataFrame({
        "missing_count": features.isna().sum(),
        "missing_ratio": features.isna().mean(),
        "observed_unique": features.nunique(dropna=True),
    })
    table.index.name = "feature"

    return table


def analyze_numeric(data: TabularInput):
    """수치 특징의 기술 통계만 계산한다. 상관분석은 호출하지 않는다."""
    data.validate()

    if not data.roles.numeric:
        return pd.DataFrame()

    numeric = data.frame.loc[:, list(data.roles.numeric)].astype(float)

    return numeric.describe().T


def analyze_categories(data: TabularInput):
    """지정한 범주 특징의 빈도를 계산하며 결측 범주도 기록한다."""
    data.validate()
    rows = []

    for name in data.roles.categorical:
        for value, count in data.frame[name].value_counts(dropna=False).items():
            rows.append({"feature": name, "value": value, "count": int(count)})

    return pd.DataFrame(rows, columns=["feature", "value", "count"])


def analyze_target(data: TabularInput):
    """선택적 라벨의 빈도를 반환한다. 라벨 변경·모델 학습은 하지 않는다."""
    data.validate()

    if data.roles.target is None:
        return pd.DataFrame(columns=["label", "count"])

    return data.frame[data.roles.target].value_counts(dropna=False).rename_axis("label").reset_index(name="count")


def analyze_correlations(data: TabularInput):
    """수치 특징만의 상관계수와 변수 쌍별 실제 관측 수를 반환한다."""
    data.validate()
    numeric = data.frame.loc[:, list(data.roles.numeric)].astype(float)
    observed = numeric.notna().astype("int64")

    # NaN은 관측 부족·상수 등의 의미를 보존한다. 0으로 바꾸지 않는다.
    return {
        "pearson": numeric.corr(method="pearson", min_periods=3),
        "spearman": numeric.corr(method="spearman", min_periods=3),
        "correlation_support": observed.T.dot(observed),
    }


EDA_ANALYSES = ("missing", "numeric", "categories", "target", "correlations")


def collect_eda(data: TabularInput, *, analyses=EDA_ANALYSES):
    """선택한 분석만 실행한다. 등록부가 아닌 내장 기능의 명시적 조합 입구다."""
    if not isinstance(analyses, (tuple, list)) or any(type(name) is not str for name in analyses):
        raise ValueError("EDA 분석 목록은 문자열의 tuple 또는 list여야 합니다.")

    if len(set(analyses)) != len(analyses) or any(name not in EDA_ANALYSES for name in analyses):
        raise ValueError("EDA 분석 목록에 중복되거나 지원하지 않는 기능이 있습니다.")

    # 요약은 표본 수·역할을 설명하는 공통 근거다. 실제 분석은 선택한 것만 호출한다.
    tables = {"summary": analyze_summary(data)}
    functions = {
        "missing": ("missing", analyze_missing),
        "numeric": ("numeric_statistics", analyze_numeric),
        "categories": ("categorical_distribution", analyze_categories),
        "target": ("target_distribution", analyze_target),
    }

    for name in analyses:
        if name == "correlations":
            tables.update(analyze_correlations(data))
        else:
            output_name, calculate = functions[name]
            tables[output_name] = calculate(data)

    return tables
