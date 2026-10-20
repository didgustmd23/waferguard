"""검증된 표형 EDA 집계에서 그림의 표시 범위만 추출한다. 원본·그림 OCR은 사용하지 않는다."""

from io import BytesIO
import math

import pandas as pd


EVIDENCE_TABLES = {"missing.csv", "numeric_statistics.csv", "pearson.csv", "correlation_support.csv"}


def _number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def context_for_graph(name, evidence):
    """같은 보고서 폴더의 검증된 집계만 사용한다. 다른 작업의 표를 혼합하지 않는다."""
    normalized = name.replace("\\", "/")
    prefix, _, basename = normalized.rpartition("/")
    group = {}
    limits = set()
    for item in evidence:
        parent, _, table = item["name"].replace("\\", "/").rpartition("/")
        if parent == prefix and table in EVIDENCE_TABLES:
            limit = item.get("plot_limit")
            if type(limit) is not int or not 1 <= limit <= 30 or table in group:
                raise ValueError("COMMENTARY_GRAPH_CONTEXT_INVALID")
            if len(item["data"]) > 8 * 1024 ** 2:
                raise ValueError("COMMENTARY_GRAPH_CONTEXT_INVALID")
            group[table] = item["data"]
            limits.add(limit)

    if not group:
        return {"status": "unavailable", "notice": "대응 집계가 없습니다. 전체 특징으로 일반화하거나 정확한 수치를 단정하지 마세요."}
    if len(limits) != 1:
        raise ValueError("COMMENTARY_GRAPH_CONTEXT_INVALID")
    limit = limits.pop()

    def table(key):
        if key not in group:
            raise ValueError("COMMENTARY_GRAPH_CONTEXT_INVALID")
        frame = pd.read_csv(BytesIO(group[key]), index_col=0)
        if not frame.index.is_unique:
            raise ValueError("COMMENTARY_GRAPH_CONTEXT_INVALID")
        return frame

    if basename == "missing.png":
        frame = table("missing.csv")
        selected = frame.sort_values("missing_ratio", ascending=False).head(limit)
        return {"status": "available", "selection": "결측률 내림차순 상위 특징만 표시; 전체 특징으로 일반화 금지",
                "total_feature_count": len(frame), "displayed_feature_count": len(selected), "unit": "ratio_0_to_1",
                "values": [{"feature": str(feature), "missing_ratio": _number(row["missing_ratio"])}
                           for feature, row in selected.iterrows()]}

    if basename == "distributions.png":
        frame = table("numeric_statistics.csv")
        selected = frame.iloc[:limit]
        return {"status": "available", "selection": "입력 수치 특징 순서의 앞쪽 일부; 중요도 상위가 아님",
                "total_numeric_feature_count": len(frame), "displayed_feature_count": len(selected),
                "statistics_basis": "결측을 제외한 관측값 기준. count는 관측 건수이며 전체 행 수가 아님. 상수 여부도 관측값에 한정함.",
                "values": [{"feature": str(feature), **{str(key): _number(value) for key, value in row.items()}}
                           for feature, row in selected.iterrows()]}

    if basename == "correlations.png":
        full = table("pearson.csv")
        if list(full.index) != list(full.columns):
            raise ValueError("COMMENTARY_GRAPH_CONTEXT_INVALID")
        selected = full.iloc[:limit, :limit]
        support = table("correlation_support.csv").reindex(index=selected.index, columns=selected.columns)
        pairs = []
        for first in range(len(selected)):
            for second in range(first + 1, len(selected)):
                coefficient = _number(selected.iloc[first, second])
                count = _number(support.iloc[first, second])
                if (coefficient is not None and not -1 <= coefficient <= 1
                        or count is None or count < 0 or count != int(count)):
                    raise ValueError("COMMENTARY_GRAPH_CONTEXT_INVALID")
                pairs.append({"feature_a": str(selected.index[first]), "feature_b": str(selected.columns[second]),
                              "pearson": coefficient, "observed_count": int(count)})

        pairs.sort(key=lambda pair: (pair["pearson"] is None, -abs(pair["pearson"] or 0)))
        return {"status": "available", "selection": "입력 수치 특징 순서의 앞쪽 일부; 전체 상관행렬이 아님",
                "total_numeric_feature_count": len(full), "features": [str(value) for value in selected.index],
                "displayed_feature_count": len(selected), "pair_count": len(pairs), "pairs": pairs,
                "pair_order": "상관계수 절댓값 내림차순, 계산 불가 항목은 마지막. 표시 특징의 모든 비중복 쌍 포함.",
                "notice": "null은 계산 불가이며 0이 아님. 자기상관·대칭 중복 쌍은 제외. 센서명·계수·관측 수는 같은 항목을 함께 인용할 것."}

    return {"status": "unavailable"}
