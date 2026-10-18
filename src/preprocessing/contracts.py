"""로컬 데이터 처리에 사용하는 명시적 컬럼 역할과 표형 입력 계약."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ==========================================
# 공통 컬럼 역할
# - label 개수·값·공장명·센서 접두사를 고정하지 않는다.
# - 수치/범주 특징과 target·group·ID를 명시적으로 분리한다.
# - 선택된 특징만 EDA 상관분석 대상으로 사용한다.
# ==========================================
@dataclass(frozen=True)
class ColumnRoles:
    numeric: tuple[str, ...] = ()
    categorical: tuple[str, ...] = ()
    target: str | None = None
    group: str | None = None
    timestamp: str | None = None
    identifiers: tuple[str, ...] = ()

    def __post_init__(self):
        for names in (self.numeric, self.categorical, self.identifiers):
            if not isinstance(names, tuple):
                raise TypeError("컬럼 목록은 tuple이어야 합니다.")

            if any(not isinstance(name, str) or not name.strip() for name in names):
                raise ValueError("컬럼 이름이 올바르지 않습니다.")

        for name in (self.target, self.group, self.timestamp):
            if name is not None and (not isinstance(name, str) or not name.strip()):
                raise ValueError("target·group·timestamp 컬럼 이름이 올바르지 않습니다.")

        names = self.required_columns

        if not self.numeric and not self.categorical:
            raise ValueError("분석할 특징 컬럼이 하나 이상 필요합니다.")

        if len(names) != len(set(names)):
            raise ValueError("컬럼 역할은 중복될 수 없습니다.")

    @property
    def feature_columns(self):
        return self.numeric + self.categorical

    @property
    def required_columns(self):
        optional = tuple(name for name in (self.target, self.group, self.timestamp) if name is not None)

        return self.feature_columns + self.identifiers + optional


@dataclass
class TabularInput:
    """어댑터가 전달하는 로컬 표. 전체 객체를 상태·LLM 입력으로 직렬화하지 않는다."""

    frame: pd.DataFrame
    roles: ColumnRoles

    def validate(self):
        """데이터를 자동 보정하지 않고 계약 위반을 거부한다."""
        if not isinstance(self.frame, pd.DataFrame) or not isinstance(self.roles, ColumnRoles):
            raise TypeError("DataFrame과 ColumnRoles가 필요합니다.")

        if not self.frame.columns.is_unique or any(not isinstance(name, str) for name in self.frame.columns):
            raise ValueError("데이터 컬럼은 중복 없는 문자열이어야 합니다.")

        if self.frame.empty:
            raise ValueError("분석할 데이터 행이 없습니다.")

        if not set(self.roles.required_columns).issubset(self.frame.columns):
            raise ValueError("프로파일의 필수 컬럼이 없습니다.")

        for name in self.roles.numeric:
            series = self.frame[name]

            if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
                raise ValueError("수치 특징에는 bool을 제외한 수치형 컬럼이 필요합니다.")

            values = series.dropna().to_numpy(dtype=float)

            if not np.isfinite(values).all():
                raise ValueError("수치 특징에 무한대가 포함되어 있습니다.")

        return self
