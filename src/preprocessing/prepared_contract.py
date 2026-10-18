"""저장된 표형 학습 입력을 재학습 없이 독립 검사하는 로컬 계약 코어."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .split import validate_proposal
from .transform import load_preparation


PREPARED_FILES = frozenset({
    "preprocessor.json", "preparation.json", "preparation.md", "split_contract.json",
    "train_features.csv", "validation_features.csv", "row_contract.json",
})


class ContractViolation(ValueError):
    """값·경로·원본 예외 대신 고정 검사 코드만 보존한다."""


def read_local_json(path):
    """중복 키·비표준 NaN·과도한 크기를 거부하는 로컬 산출물 로더."""
    path = Path(path)

    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("로컬 계약 JSON 크기 상한을 초과했습니다.")

    def pairs(items):
        result = {}

        for key, value in items:
            if key in result:
                raise ValueError("로컬 계약 JSON에 중복 키가 있습니다.")

            result[key] = value

        return result

    def invalid_constant(value):
        raise ValueError("로컬 계약 JSON에 비표준 숫자가 있습니다.")

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=invalid_constant)


def _equal(left, right):
    """bool·숫자·문자열을 느슨하게 동등 취급하지 않고 JSON 타입까지 비교한다."""
    return json.dumps(left, sort_keys=True, ensure_ascii=False, allow_nan=False) == json.dumps(right, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _rows(data, proposal):
    result = {}

    for name in ("train", "validation"):
        positions = proposal["partitions"][name]
        row = {"row_positions": positions}

        if data.roles.target is not None:
            labels = data.frame.iloc[positions][data.roles.target].tolist()
            row["targets"] = [value.item() if isinstance(value, np.generic) else value for value in labels]

        result[name] = row

    return result


# ==========================================
# 독립 최종 계약 검사
# - 입력은 원본 계약·확정 분할·저장 변환기·Train/Validation 산출물이다.
# - 저장 변환기를 복원하여 transform만 호출하며 fit은 호출하지 않는다.
# - 파일 구성·식별값·행 위치/정답·컬럼 순서/타입·NaN 정책을 따로 검사한다.
# - 실패 시 고정 코드만 반환하고 부분 통계나 원본 예외는 결과에 복사하지 않는다.
# - 상세 성공 요약은 로컬 보고서 전용이며 LLM 입력이 아니다.
# ==========================================
def inspect_prepared(directory, data, rules, split_rules, *, source_id, config_id):
    checks = []
    current = "PREPARED_FILES"

    def require(condition, code):
        if not condition:
            raise ContractViolation(code)

        checks.append({"code": code, "passed": True})

    try:
        directory = Path(directory)
        files = tuple(directory.iterdir())
        require(
            {path.name for path in files} - {"AGENT_COMPLETE.json"} == PREPARED_FILES
            and all(path.is_file() and not path.is_symlink() and path.stat().st_size > 0 for path in files),
            current,
        )

        current = "TRANSFORMER_CONTRACT"
        pipeline = load_preparation(directory / "preprocessor.json")
        transformer = pipeline.named_steps["preprocess"]
        require(transformer.roles == data.roles and transformer.rules == rules, current)

        current = "SPLIT_BINDING"
        proposal = read_local_json(directory / "split_contract.json")
        validate_proposal(proposal, data, split_rules, source_id=source_id, config_id=config_id)
        require(True, current)

        current = "PREPARATION_BINDING"
        preparation = read_local_json(directory / "preparation.json")
        expected_binding = {
            "status": "preparation_complete", "source_id": source_id,
            "config_id": config_id, "proposal_id": proposal["proposal_id"],
            "rules": vars(rules), "train_rows": len(proposal["partitions"]["train"]),
            "input_features": len(data.roles.feature_columns),
            "output_features": len(transformer.get_feature_names_out()),
            "removed": transformer.state_["removed"],
            "diagnostics": {
                name: transformer.diagnose(data.frame.iloc[proposal["partitions"][name]])
                for name in ("train", "validation")
            },
            "prepared_splits": ["train", "validation"],
            "test_transformed": False, "model_trained": False,
        }
        require(_equal(preparation, expected_binding) and transformer.state_["train_rows"] == expected_binding["train_rows"], current)

        current = "ROW_ALIGNMENT"
        require(_equal(read_local_json(directory / "row_contract.json"), _rows(data, proposal)), current)
        summaries = []
        columns = transformer.get_feature_names_out().tolist()

        # NaN 보존은 none 대치를 선택한 비범주화 수치 출력에만 허용한다.
        nullable_columns = []

        for record in transformer.state_["records"]:
            nullable = record["kind"] == "numeric" and record["edges"] is None and rules.numeric_imputation == "none"
            nullable_columns.extend([nullable] * record["width"])

        for name in ("train", "validation"):
            frame = data.frame.iloc[proposal["partitions"][name]]
            path = directory / f"{name}_features.csv"
            current = f"{name.upper()}_OUTPUT_SCHEMA"

            if path.stat().st_size > 256 * 1024 * 1024:
                raise ContractViolation(current)

            saved = pd.read_csv(path, float_precision="round_trip")
            require(
                len(saved) == len(frame) and saved.columns.tolist() == columns
                and all(dtype == np.dtype("float64") for dtype in saved.dtypes), current,
            )

            current = f"{name.upper()}_VALUE_POLICY"
            values = saved.to_numpy()
            missing = np.isnan(values)
            require(
                not np.isinf(values).any()
                and not missing[:, ~np.asarray(nullable_columns, dtype=bool)].any(), current,
            )

            current = f"{name.upper()}_TRANSFORM_MATCH"
            expected = pipeline.transform(frame).reset_index(drop=True)
            require(saved.equals(expected), current)
            diagnostics = transformer.diagnose(frame)
            input_missing_count = 0

            for column in data.roles.feature_columns:
                series = frame[column]
                blank = series.astype(object).map(lambda value: isinstance(value, str) and not value.strip())
                input_missing_count += int((series.isna() | blank).sum())

            summaries.append({
                "split": name, "rows": len(frame), "output_features": len(columns),
                "input_missing_count": input_missing_count,
                "retained_input_missing_count": sum(row["missing_count"] for row in diagnostics),
                "output_missing_count": int(missing.sum()),
                "unknown_category_count": sum(row["unknown_category_count"] for row in diagnostics),
            })

        return {
            "schema_version": "prepared_check_v1", "status": "passed", "checks": checks,
            "summary": {
                "source_id": source_id, "config_id": config_id, "proposal_id": proposal["proposal_id"],
                "rules": vars(rules), "input_features": len(data.roles.feature_columns),
                "retained_input_features": len(transformer.state_["records"]),
                "output_features": len(columns), "output_columns": columns,
                "removed": transformer.state_["removed"], "splits": summaries,
                "test_transformed": False, "model_trained": False,
            },
        }
    except (ValueError, TypeError, KeyError, AttributeError, OSError, OverflowError):
        return {
            "schema_version": "prepared_check_v1", "status": "blocked_input",
            "checks": checks + [{"code": current, "passed": False}],
        }
