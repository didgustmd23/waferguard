"""실제 데이터와 독립된 EDA fixture. 외부 입력·설정을 받지 않는다."""

import hashlib
from io import StringIO


VERSION = "isolated_synthetic_eda_v1"
CSV_BYTES = (
    "x,y,category,target\n"
    "1,2,A,0\n2,4,B,0\n3,6,A,1\n4,8,B,0\n"
    "5,,A,1\n6,12,B,0\n7,14,A,1\n8,16,B,0\n"
).encode("ascii")
SOURCE_ID = "synthetic:" + hashlib.sha256(CSV_BYTES).hexdigest()


def fixture_input(payload=CSV_BYTES):
    """공통 입력 계약을 사용하며 공장명·센서 접두사·외부 파일에 의존하지 않는다."""
    import pandas as pd
    from src.preprocessing.contracts import ColumnRoles, TabularInput

    if payload != CSV_BYTES:
        raise ValueError("독립 합성 CSV 이외의 입력은 지원하지 않습니다.")

    return TabularInput(
        pd.read_csv(StringIO(payload.decode("ascii"))),
        ColumnRoles(numeric=("x", "y"), categorical=("category",), target="target"),
    ).validate()
