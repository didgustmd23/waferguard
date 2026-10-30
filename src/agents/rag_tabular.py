"""실험적 Graph RAG의 표형(v1·v2) 프로필: 지식 위치·지원 유형·적용 단서·검색 질의 구성.

질의는 이미 만든 권장 요청(user 메시지)에서만 만든다. 목표 문장과 상세로 보낸 컬럼 집계·근거 코드에서 고정 단서를 계산하며,
컬럼 이름·개별 값은 질의에 넣지 않는다. 이름만·요약만 보낸 컬럼은 집계가 없으므로 단서를 추정하지 않는다.
내부 진단(record)이나 요청에 없는 필드는 읽지 않는다.
"""

import json
from pathlib import Path


PROFILE_ID = "tabular_v1"
TASK_KINDS = ("tabular_csv", "tabular_v2")
REPO_ROOT = Path(__file__).resolve().parents[2]
PACK_DIR = REPO_ROOT / "configs" / "agents" / "rag" / "tabular"
INDEX_ROOT = REPO_ROOT / "models" / "rag" / "tabular"
EMBED_MODEL = "bge-m3:567m"
DIMENSIONS = 1024
QUERY_VERSION = "tabular_query_v2"
MAX_QUERY_CHARS = 1500
# 고정 단서 문구(질의 구성 v2). 검색 임계값은 이 구성으로 만든 질의에서 보정했으므로 문구를 바꾸면 다시 보정해야 한다.
CUES = {
    "few_values": "값 종류가 적은 숫자형 컬럼(범주 코드 가능성, 구간화·clipping 시 상수화)",
    "sparse_binary": "대부분 같은 값인 0/1 수치 컬럼(IQR clipping 상수화)",
    "outliers": "IQR 이상치 후보가 있는 수치 특징(극단값, 스케일링·clipping·대치 선택)",
    "all_missing": "전체 결측 수치 특징(대치 불가, 제외 또는 제거 기준)",
    "constant": "값이 하나뿐인 상수 특징",
    "high_missing": "결측률이 높은 특징(결측률 제거 기준, 결측 의미)",
    "all_distinct": "행마다 다른 값(식별 번호, 새 범주로 적용 실패)",
    "many_categories": "고유값이 많은 범주 특징(범주 수 상한)",
    "repeated_text": "행 수보다 고유값이 훨씬 적은 문자열 단위 컬럼(반복 측정 단위 가능성, group 분할)",
    "non_numeric": "문자열 자료형 컬럼(숫자 모양 문자열은 수치형 불가)",
}
# 단서 계산에 쓰는 요청 필드. 권장 요청의 detail_fields에 없는 필드는 쓰지 않는다.
USED_FIELDS = ("dtype_group", "missing_ratio", "observed_unique", "mean", "outlier_candidate_count")


class QueryError(ValueError):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def _cues(row_count, rows):
    found = []

    def add(name):
        if name not in found:
            found.append(name)

    for row in rows:
        unique, numeric = row["observed_unique"], row["dtype_group"] == "numeric"
        ratio = row["missing_ratio"] or 0
        reasons = row["reasons"]

        if numeric and ratio >= 1:
            add("all_missing")
        elif unique is not None and unique <= 1:
            add("constant")
        if numeric and unique is not None and 2 <= unique <= 20:
            add("few_values")
        if numeric and unique == 2 and row["mean"] is not None and (row["mean"] <= 0.1 or row["mean"] >= 0.9):
            add("sparse_binary")
        if numeric and (row["outlier_candidate_count"] or 0) > 0 and unique is not None and unique > 20:
            add("outliers")
        if 0.05 <= ratio < 1:
            add("high_missing")
        # 행마다 다른 값은 요청에 보낸 근거 코드로만 판단한다(결측 개수는 보내지 않는다).
        if "all_distinct" in reasons and unique is not None and unique > 1:
            add("all_distinct")
        if not numeric:
            add("non_numeric")
            observed = round(row_count * (1 - ratio))

            if unique is not None and unique > 256:
                add("many_categories")
            if unique is not None and 20 < unique <= observed // 5:
                add("repeated_text")

    return found


def build_query(payload):
    """권장 요청 본문에서 질의 문자열과 단서 코드를 만든다. 형식이 다르면 QueryError(고정 코드)."""
    try:
        content = json.loads(payload["messages"][1]["content"])
        goal, row_count, subjects = content["goal"], content["row_count"], content["subjects"]
        fields, detailed = subjects["detail_fields"], subjects["detailed"]
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise QueryError("RAG_QUERY_INVALID") from error

    if (type(goal) is not str or not goal.strip() or type(row_count) is not int or row_count < 0
            or type(fields) is not list or fields[:1] != ["id"] or fields[-1:] != ["reasons"]
            or not set(USED_FIELDS) <= set(fields) or type(detailed) is not list):
        raise QueryError("RAG_QUERY_INVALID")

    rows = []

    for values in detailed:
        if type(values) is not list or len(values) != len(fields):
            raise QueryError("RAG_QUERY_INVALID")

        row = dict(zip(fields, values))

        if type(row["reasons"]) is not list:
            raise QueryError("RAG_QUERY_INVALID")

        # 컬럼 이름(id)은 단서 계산에 넘기지 않는다.
        rows.append({name: row[name] for name in (*USED_FIELDS, "reasons")})

    try:
        names = _cues(row_count, rows)
    except (TypeError, ValueError) as error:
        raise QueryError("RAG_QUERY_INVALID") from error

    text = "목표: " + goal.strip() + "\n데이터 특성: " + "; ".join(CUES[name] for name in names)

    if len(text) > MAX_QUERY_CHARS or any(ord(char) < 32 and char != "\n" for char in text):
        raise QueryError("RAG_QUERY_TOO_LONG" if len(text) > MAX_QUERY_CHARS else "RAG_QUERY_INVALID")

    return {"text": text, "cues": names}
