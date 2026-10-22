"""Train에서만 기준을 학습하는 표형 전처리와 안전한 JSON 저장 계약."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from .contracts import ColumnRoles, TabularInput
from .split import validate_proposal


class PreparationInputError(ValueError):
    """입력 데이터 때문에 후보를 만들 수 없는 경우. 진단에는 고정 코드만 남기고 메시지·값·컬럼명은 남기지 않는다."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class PreparationRules:
    """방법은 명시적으로 선택한다. 이상치 행 삭제·자동 최적화는 제공하지 않는다."""

    missing_ratio_threshold: float | None = None
    drop_constant: bool = False
    numeric_imputation: str = "median"
    scaling: str = "none"
    outlier: str = "none"
    iqr_multiplier: float = 1.5
    binning: str = "none"
    n_bins: int = 5
    bin_encoding: str = "ordinal"
    categorical_imputation: str = "missing_category"
    unknown_category: str = "error"
    max_categories: int = 256
    max_output_features: int = 4096
    max_output_cells: int = 10_000_000
    algorithm_version: str = "tabular_prepare_v1"

    def __post_init__(self):
        choices = {
            "numeric_imputation": {"median", "mean", "none"},
            "scaling": {"none", "standard", "minmax", "robust"},
            "outlier": {"none", "iqr_clip"},
            "binning": {"none", "quantile", "uniform"},
            "bin_encoding": {"ordinal", "onehot"},
            "categorical_imputation": {"missing_category", "most_frequent"},
            "unknown_category": {"error", "ignore"},
        }

        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise ValueError("지원하지 않는 전처리 방법입니다.")

        threshold = self.missing_ratio_threshold

        if threshold is not None and (type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1):
            raise ValueError("제거용 결측 비율은 0~1 유한수 또는 null이어야 합니다.")

        if type(self.drop_constant) is not bool:
            raise ValueError("상수 특징 제거 여부는 bool이어야 합니다.")

        if type(self.iqr_multiplier) not in (int, float) or not math.isfinite(self.iqr_multiplier) or self.iqr_multiplier <= 0:
            raise ValueError("IQR 배수는 양의 유한수여야 합니다.")

        for name, upper in (("n_bins", 100), ("max_categories", 4096), ("max_output_features", 100_000), ("max_output_cells", 100_000_000)):
            value = getattr(self, name)

            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError("구간 수·출력 상한은 허용 범위의 양의 정수여야 합니다.")

        if self.n_bins < 2:
            raise ValueError("범주화 구간 수는 2 이상이어야 합니다.")

        if self.binning != "none" and (self.scaling != "none" or self.numeric_imputation == "none"):
            raise ValueError("수치 범주화에는 결측 대치가 필요하며 스케일링과 동시에 적용하지 않습니다.")

        if self.algorithm_version != "tabular_prepare_v1":
            raise ValueError("지원하지 않는 전처리 알고리즘 버전입니다.")


def _category(value):
    """문자열·숫자·bool 범주를 구분하고 실제 값과 결측 토큰의 충돌을 막는다."""
    if value is None or value is pd.NA or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        return "missing"

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, str):
        if not value.strip():
            return "missing"

        kind = "str"
    elif type(value) is bool:
        kind = "bool"
    elif type(value) in (int, float) and math.isfinite(value):
        # CSV의 정수·실수 표현 차이는 동일 숫자 범주로 취급한다.
        kind = "number"
        value = float(value)
    else:
        raise ValueError("범주 특징에는 유한 숫자·문자열·bool·결측만 허용합니다.")

    return json.dumps([kind, value], ensure_ascii=False, allow_nan=False)


def _numbers(series):
    return series.to_numpy(dtype=float, na_value=np.nan, copy=True)


def _numeric_values(values, record):
    """저장한 Train 통계로만 대치·clip한다. 전달된 배치에서 통계를 계산하지 않는다."""
    values = values.copy()

    if record["impute"] is not None:
        values[np.isnan(values)] = record["impute"]

    if record["clip"] is not None:
        values = np.clip(values, *record["clip"])

    return values


# ==========================================
# sklearn Pipeline 안에서 사용하는 범용 변환기
# - fit: 호출자가 전달한 학습 구간에서만 제거 목록·대치·변환 기준을 계산한다.
# - transform: 저장된 기준만 적용하며 새 통계·범주·컬럼을 학습하지 않는다.
# - target·group·timestamp·ID는 모델 입력과 학습 통계에서 제외한다.
# - 입력 행을 삭제하지 않고 열 순서가 달라도 이름 기준으로 정렬한다.
# - 출력 크기를 제한하여 원핫 인코딩의 과도한 메모리 사용을 차단한다.
# ==========================================
class TabularPreprocessor(TransformerMixin, BaseEstimator):
    def __init__(self, roles, rules):
        # sklearn.clone이 생성자 인자를 그대로 복제할 수 있도록 여기서는 변환하지 않는다.
        self.roles = roles
        self.rules = rules

    def _input(self, frame):
        if not isinstance(self.roles, ColumnRoles) or not isinstance(self.rules, PreparationRules):
            raise TypeError("ColumnRoles와 PreparationRules가 필요합니다.")

        # 추론에는 정답·Lot·시간·ID 없이 특징만 전달할 수 있다.
        feature_roles = ColumnRoles(numeric=self.roles.numeric, categorical=self.roles.categorical)
        TabularInput(frame, feature_roles).validate()

        return frame.loc[:, list(self.roles.feature_columns)]

    def fit(self, X, y=None):
        """학습 데이터에서만 처리 기준을 계산한다. y를 특징 선택에 사용하지 않는다."""
        frame = self._input(X)
        records = []
        removed = []

        for column in self.roles.feature_columns:
            numeric = column in self.roles.numeric
            series = frame[column]
            values = _numbers(series) if numeric else series.map(_category)
            missing = np.isnan(values) if numeric else values.eq("missing")
            missing_ratio = float(np.mean(missing))
            unique = int(pd.Series(values[~missing]).nunique())
            reason = None

            if self.rules.missing_ratio_threshold is not None and missing_ratio > self.rules.missing_ratio_threshold:
                reason = "HIGH_MISSING_RATIO"
            elif self.rules.drop_constant and unique <= 1:
                reason = "CONSTANT_OR_ALL_MISSING"

            if reason is not None:
                removed.append({"column": column, "reason": reason})
                continue

            if numeric:
                record = self._fit_numeric(column, values)
            else:
                record = self._fit_categorical(column, values)

            records.append(record)

        if not records:
            raise ValueError("전처리 후 남는 특징이 없습니다.")

        width = sum(record["width"] for record in records)

        if width > self.rules.max_output_features:
            raise ValueError("전처리 출력 특징 수 상한을 초과했습니다.")

        self.state_ = {"train_rows": len(frame), "records": records, "removed": removed}
        self.n_features_in_ = len(self.roles.feature_columns)
        self.feature_names_in_ = np.asarray(self.roles.feature_columns, dtype=object)
        self.feature_names_out_ = np.asarray([
            f"feature_{index}__{record['column']}__{offset}"
            for index, record in enumerate(records) for offset in range(record["width"])
        ], dtype=object)

        return self

    def _fit_numeric(self, column, values):
        """대치 → 선택적 IQR clip → 스케일링 또는 범주화 순서로 Train 기준을 만든다."""
        observed = values[~np.isnan(values)]
        needs_statistics = self.rules.numeric_imputation != "none" or self.rules.scaling != "none" or self.rules.outlier != "none"

        if not len(observed) and needs_statistics:
            raise ValueError("전체 결측 수치 특징은 제거 기준을 지정하거나 별도로 검토하세요.")

        impute = None

        if self.rules.numeric_imputation != "none":
            function = np.median if self.rules.numeric_imputation == "median" else np.mean
            impute = float(function(observed))

        clip = None

        if self.rules.outlier == "iqr_clip":
            q1, q3 = np.quantile(observed, [0.25, 0.75])
            margin = self.rules.iqr_multiplier * (q3 - q1)
            clip = [float(q1 - margin), float(q3 + margin)]

        record = {"column": column, "kind": "numeric", "impute": impute, "clip": clip, "center": 0.0, "scale": 1.0, "edges": None, "width": 1}
        prepared = _numeric_values(values, record)
        observed_prepared = prepared[~np.isnan(prepared)]

        if self.rules.binning != "none":
            boundaries = (
                np.quantile(prepared, np.linspace(0, 1, self.rules.n_bins + 1))
                if self.rules.binning == "quantile"
                else np.linspace(prepared.min(), prepared.max(), self.rules.n_bins + 1)
            )
            # 동점 경계는 합친다. 상수 특징은 한 구간으로 남으며 실제 구간 수를 저장한다.
            record["edges"] = np.unique(boundaries)[1:-1].tolist()
            record["width"] = len(record["edges"]) + 1 if self.rules.bin_encoding == "onehot" else 1
        elif self.rules.scaling == "standard":
            record["center"] = float(observed_prepared.mean())
            record["scale"] = float(observed_prepared.std(ddof=0)) or 1.0
        elif self.rules.scaling == "minmax":
            record["center"] = float(observed_prepared.min())
            record["scale"] = float(observed_prepared.max() - observed_prepared.min()) or 1.0
        elif self.rules.scaling == "robust":
            q1, q3 = np.quantile(observed_prepared, [0.25, 0.75])
            record["center"] = float(np.median(observed_prepared))
            record["scale"] = float(q3 - q1) or 1.0

        statistics = [record["center"], record["scale"]] + ([impute] if impute is not None else []) + (clip or []) + (record["edges"] or [])

        if not np.isfinite(statistics).all():
            raise ValueError("전처리 통계 계산 결과가 유한수가 아닙니다.")

        return record

    def _fit_categorical(self, column, values):
        """Train 범주만 등록한다. 결측 전용 범주는 실제 문자열과 구분한다."""
        fill = "missing"

        if self.rules.categorical_imputation == "most_frequent":
            counts = values[values.ne("missing")].value_counts()

            if counts.empty:
                raise ValueError("전체 결측 범주 특징에는 최빈값 대치를 적용할 수 없습니다.")

            # 동점에서는 정규화 토큰 순서로 선택하여 결과를 재현한다.
            fill = sorted(counts[counts.eq(counts.max())].index)[0]

        prepared = values.mask(values.eq("missing"), fill)
        categories = sorted(set(prepared))

        if self.rules.categorical_imputation == "missing_category" and "missing" not in categories:
            categories.append("missing")
            categories.sort()

        if len(categories) > self.rules.max_categories:
            raise PreparationInputError("CATEGORY_LIMIT_EXCEEDED", "범주 특징의 원핫 범주 수 상한을 초과했습니다.")

        return {"column": column, "kind": "categorical", "fill": fill, "categories": categories, "width": len(categories)}

    def transform(self, X):
        """새 배치의 평균·범주를 계산하지 않고 고정 순서의 float 특징을 반환한다."""
        check_is_fitted(self, "state_")
        frame = self._input(X)
        width = len(self.feature_names_out_)

        if len(frame) * width > self.rules.max_output_cells:
            raise ValueError("전처리 출력 셀 수 상한을 초과했습니다. 입력을 배치로 나누세요.")

        return pd.DataFrame(self._output(frame), index=frame.index, columns=self.feature_names_out_)

    def _output(self, frame):
        """저장한 기준만으로 출력 배열을 만든다. transform과 영향 집계(impact)가 같은 계산을 쓴다."""
        output = np.zeros((len(frame), len(self.feature_names_out_)), dtype=np.float64)
        offset = 0

        for record in self.state_["records"]:
            column = frame[record["column"]]

            if record["kind"] == "numeric":
                values = _numeric_values(_numbers(column), record)

                if record["edges"] is not None:
                    bins = np.searchsorted(record["edges"], values, side="right")

                    if self.rules.bin_encoding == "onehot":
                        output[np.arange(len(frame)), offset + bins] = 1.0
                    else:
                        output[:, offset] = bins
                else:
                    output[:, offset] = (values - record["center"]) / record["scale"]
            else:
                tokens = column.map(_category).replace("missing", record["fill"])
                lookup = {value: index for index, value in enumerate(record["categories"])}
                indices = tokens.map(lookup)

                if indices.isna().any() and self.rules.unknown_category == "error":
                    raise ValueError("Train에 없던 범주가 입력되었습니다.")

                known = indices.notna().to_numpy()
                output[np.flatnonzero(known), offset + indices[known].to_numpy(dtype=int)] = 1.0

            offset += record["width"]

        if np.isinf(output).any():
            raise ValueError("변환 결과에 무한대가 발생했습니다.")

        return output

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_out_")

        return self.feature_names_out_.copy()

    def diagnose(self, X):
        """결측·새 범주 건수만 로컬 기록용으로 계산한다. 처리 기준은 변경하지 않는다."""
        check_is_fitted(self, "state_")
        frame = self._input(X)
        result = []

        for record in self.state_["records"]:
            column = frame[record["column"]]

            if record["kind"] == "numeric":
                missing_count = int(np.isnan(_numbers(column)).sum())
                unknown_count = 0
            else:
                tokens = column.map(_category)
                missing_count = int(tokens.eq("missing").sum())
                prepared = tokens.replace("missing", record["fill"])
                unknown_count = int((~prepared.isin(record["categories"])).sum())

            result.append({"column": record["column"], "missing_count": missing_count, "unknown_category_count": unknown_count})

        return result

    def impact(self, X, transformed=None):
        """저장한 기준을 이 배치에 적용했을 때의 특징별 영향 건수만 계산한다. 기준·출력은 바꾸지 않는다.

        clipped는 결측이 아닌 관측값 중 clip 범위 밖이라 값이 바뀌는 개수(clip이 없으면 None),
        constant_after는 해당 특징의 출력 열이 이 배치 전체에서 모두 한 값인지, missing_after는 출력에 남은 결측 칸 수다.
        개별 값·경계는 담지 않는다.
        transformed에 같은 배치의 transform 결과를 주면 다시 변환하지 않는다.
        """
        check_is_fitted(self, "state_")
        frame = self._input(X)
        # 후보 단계는 transform을 호출하지 않는다(변환 결과를 만들지 않는 계약). 같은 계산으로 메모리에서만 센다.
        output = self._output(frame) if transformed is None else transformed.to_numpy(dtype=np.float64)

        if output.shape != (len(frame), len(self.feature_names_out_)):
            raise ValueError("영향 집계의 변환 결과 크기가 입력과 다릅니다.")

        result = []
        offset = 0

        for record in self.state_["records"]:
            column = frame[record["column"]]
            block = output[:, offset:offset + record["width"]]
            offset += record["width"]

            if record["kind"] == "numeric":
                values = _numbers(column)
                observed = values[~np.isnan(values)]
                missing = int(np.isnan(values).sum())
                clipped = None if record["clip"] is None else int(((observed < record["clip"][0]) | (observed > record["clip"][1])).sum())
                distinct = int(len(np.unique(observed)))
                unknown = 0
            else:
                tokens = column.map(_category)
                missing = int(tokens.eq("missing").sum())
                clipped = None
                distinct = int(tokens[tokens.ne("missing")].nunique())
                unknown = int((~tokens.replace("missing", record["fill"]).isin(record["categories"])).sum())

            # 대치하지 않은 결측(NaN)만 있는 열은 상수, 결측과 값이 섞이면 상수가 아니다.
            constant = all(_constant(block[:, index]) for index in range(block.shape[1]))
            result.append({"column": record["column"], "kind": record["kind"], "width": record["width"], "missing": missing,
                           "imputed": missing if record["kind"] == "categorical" or record.get("impute") is not None else 0,
                           "clipped": clipped, "distinct_before": distinct, "constant_after": constant, "unknown": unknown,
                           "missing_after": int(np.isnan(block).sum())})

        return result


def _constant(values):
    missing = np.isnan(values)
    return bool(missing.all() or (not missing.any() and (values == values[0]).all()))


def impact_totals(items):
    """특징별 영향을 구간 합계 정수로 줄인다. 적용 기록의 크기 상한 안에 두기 위한 요약이다."""
    return {"clipped_values": sum(item["clipped"] or 0 for item in items),
            "clipped_features": sum(1 for item in items if item["clipped"]),
            "constant_features": sum(1 for item in items if item["constant_after"]),
            "unknown_categories": sum(item["unknown"] for item in items),
            "missing_values": sum(item["missing_after"] for item in items)}


def build_preparation(roles, rules):
    """교차검증 시 clone하여 각 학습 Fold 안에서 fit할 수 있는 Pipeline."""
    return Pipeline([("preprocess", TabularPreprocessor(roles, rules))])


def prepare_splits(data, proposal, split_rules, rules, *, source_id, config_id):
    """확정 분할을 검증하고 Train만 fit한다. Test 변환은 명시적 후속 단계로 남긴다."""
    validate_proposal(proposal, data, split_rules, source_id=source_id, config_id=config_id)
    pipeline = build_preparation(data.roles, rules)
    train = data.frame.iloc[proposal["partitions"]["train"]]
    pipeline.fit(train)
    outputs = {}

    for name in ("train", "validation"):
        outputs[name] = pipeline.transform(data.frame.iloc[proposal["partitions"][name]])

    return pipeline, outputs


def save_preparation(pipeline, path, *, max_bytes=None):
    """pickle/joblib 없이 통계와 계약만 JSON으로 저장한다. 기존 파일을 덮어쓰지 않는다."""
    transformer = pipeline.named_steps["preprocess"]
    check_is_fitted(transformer, "state_")
    payload = {
        "schema_version": "fitted_tabular_prepare_v1", "roles": asdict(transformer.roles),
        "rules": asdict(transformer.rules), "state": transformer.state_,
        "output_columns": transformer.get_feature_names_out().tolist(),
    }

    if max_bytes is not None and (type(max_bytes) is not int or max_bytes < 1):
        raise ValueError("전처리 상태 파일 상한은 양의 정수 바이트여야 합니다.")

    # 후보 worker에서는 각 JSON 조각을 쓰기 전에 크기를 검사한다.
    # 초과 시 부분 파일은 남길 수 있지만 완료 후보로 등록하지 않는다.
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        size = 0

        for chunk in json.JSONEncoder(ensure_ascii=False, indent=2, allow_nan=False).iterencode(payload):
            size += len(chunk.encode("utf-8"))

            if max_bytes is not None and size > max_bytes:
                raise ValueError("전처리 상태 파일 크기 상한을 초과했습니다.")

            stream.write(chunk)


def load_preparation(path):
    """로컬 JSON 계약만 복원한다. 함수·모듈을 가져오거나 fit을 다시 호출하지 않는다."""
    path = Path(path)

    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("저장 전처리 파일 크기 상한을 초과했습니다.")

    def pairs(items):
        result = {}

        for key, value in items:
            if key in result:
                raise ValueError("저장 전처리 JSON에 중복 키가 있습니다.")

            result[key] = value

        return result

    payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs)

    if not isinstance(payload, dict) or set(payload) != {"schema_version", "roles", "rules", "state", "output_columns"} or payload["schema_version"] != "fitted_tabular_prepare_v1":
        raise ValueError("저장 전처리 형식이 계약과 일치하지 않습니다.")

    role_values = dict(payload["roles"])

    for key in ("numeric", "categorical", "identifiers"):
        if not isinstance(role_values.get(key), list):
            raise ValueError("저장된 컬럼 목록 형식이 올바르지 않습니다.")

        role_values[key] = tuple(role_values[key])

    roles = ColumnRoles(**role_values)
    rules = PreparationRules(**payload["rules"])
    pipeline = build_preparation(roles, rules)
    transformer = pipeline.named_steps["preprocess"]
    transformer.state_ = _validate_state(payload["state"], roles, rules)
    transformer.feature_names_out_ = np.asarray([
        f"feature_{index}__{record['column']}__{offset}"
        for index, record in enumerate(transformer.state_["records"]) for offset in range(record["width"])
    ], dtype=object)

    if payload["output_columns"] != transformer.feature_names_out_.tolist():
        raise ValueError("저장된 출력 컬럼 순서가 처리 계약과 일치하지 않습니다.")

    transformer.n_features_in_ = len(roles.feature_columns)
    transformer.feature_names_in_ = np.asarray(roles.feature_columns, dtype=object)

    return pipeline


def _validate_state(state, roles, rules):
    """복원 통계의 타입·범위·역할·출력 크기를 검사한다. 해시 인증을 대신하지 않는다."""
    if not isinstance(state, dict) or set(state) != {"train_rows", "records", "removed"} or type(state["train_rows"]) is not int or state["train_rows"] < 1:
        raise ValueError("저장된 학습 상태 형식이 올바르지 않습니다.")

    if not isinstance(state["records"], list) or not state["records"] or not isinstance(state["removed"], list):
        raise ValueError("저장된 특징 목록이 올바르지 않습니다.")

    columns = []
    width = 0

    def finite(value):
        return type(value) in (int, float) and math.isfinite(value)

    for record in state["records"]:
        if not isinstance(record, dict) or record.get("column") not in roles.feature_columns:
            raise ValueError("저장된 특징 역할이 올바르지 않습니다.")

        columns.append(record["column"])

        if record.get("kind") == "numeric" and record["column"] in roles.numeric:
            if set(record) != {"column", "kind", "impute", "clip", "center", "scale", "edges", "width"}:
                raise ValueError("저장된 수치 처리 필드가 올바르지 않습니다.")

            impute, clip, edges = record["impute"], record["clip"], record["edges"]

            if (impute is not None and not finite(impute)) or not finite(record["center"]) or not finite(record["scale"]) or record["scale"] <= 0:
                raise ValueError("저장된 수치 통계가 올바르지 않습니다.")

            if clip is not None and (not isinstance(clip, list) or len(clip) != 2 or not all(finite(value) for value in clip) or clip[0] > clip[1]):
                raise ValueError("저장된 이상치 경계가 올바르지 않습니다.")

            if edges is not None and (not isinstance(edges, list) or len(edges) >= rules.n_bins or not all(finite(value) for value in edges) or edges != sorted(set(edges))):
                raise ValueError("저장된 범주화 경계가 올바르지 않습니다.")

            if (edges is None) != (rules.binning == "none") or (clip is None) != (rules.outlier == "none") or (impute is None) != (rules.numeric_imputation == "none"):
                raise ValueError("저장된 통계와 전처리 설정이 일치하지 않습니다.")

            expected_width = len(edges) + 1 if edges is not None and rules.bin_encoding == "onehot" else 1
        elif record.get("kind") == "categorical" and record["column"] in roles.categorical:
            if set(record) != {"column", "kind", "fill", "categories", "width"}:
                raise ValueError("저장된 범주 처리 필드가 올바르지 않습니다.")

            categories = record["categories"]

            if not isinstance(categories, list) or not categories or not all(isinstance(value, str) for value in categories) or categories != sorted(set(categories)) or len(categories) > rules.max_categories or record["fill"] not in categories:
                raise ValueError("저장된 범주 목록이 올바르지 않습니다.")

            expected_width = len(categories)
        else:
            raise ValueError("저장된 처리 종류가 역할과 일치하지 않습니다.")

        if type(record["width"]) is not int or record["width"] != expected_width:
            raise ValueError("저장된 출력 폭이 올바르지 않습니다.")

        width += expected_width

    for removed in state["removed"]:
        if not isinstance(removed, dict) or set(removed) != {"column", "reason"} or removed["column"] not in roles.feature_columns or removed["reason"] not in {"HIGH_MISSING_RATIO", "CONSTANT_OR_ALL_MISSING"}:
            raise ValueError("저장된 제거 목록이 올바르지 않습니다.")

        columns.append(removed["column"])

    if len(columns) != len(set(columns)) or set(columns) != set(roles.feature_columns) or width > rules.max_output_features:
        raise ValueError("저장된 특징 목록의 중복·누락 또는 크기 위반이 있습니다.")

    return state
