"""LLM 제안에 공개할 정적 CSV 작업 계약. UI 생성 폼이나 실제 로컬 경로는 공개하지 않는다."""

from dataclasses import fields

from src.preprocessing.split import SplitRules
from src.preprocessing.transform import PreparationRules
from .workflow_service_registry import TABULAR_VERSION, csv_workflow_records, tabular_workflow_records


VERSION = "1.0.0"


def settings_schema(module_id):
    """모든 유효 설정을 명시한다. 모델이 생략한 값을 임의 기본값으로 채우지 않는다."""
    if module_id == "csv_split_review":
        properties = {
            "strategy": {"enum": ["random", "group", "time"]},
            **{key: {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1}
               for key in ("train_ratio", "validation_ratio", "test_ratio")},
            "random_state": {"type": "integer", "minimum": 0, "maximum": 2 ** 32 - 1},
            "stratify": {"type": "boolean"}, "require_all_classes": {"type": "boolean"},
            "timestamp_format": {"type": ["string", "null"], "maxLength": 128},
            "algorithm_version": {"const": "tabular_split_v1"},
        }
        required = [field.name for field in fields(SplitRules)]
    elif module_id == "csv_prepare_candidate":
        choices = {
            "numeric_imputation": ["median", "mean", "none"], "scaling": ["none", "standard", "minmax", "robust"],
            "outlier": ["none", "iqr_clip"], "binning": ["none", "quantile", "uniform"],
            "bin_encoding": ["ordinal", "onehot"], "categorical_imputation": ["missing_category", "most_frequent"],
            "unknown_category": ["error", "ignore"],
        }
        properties = {key: {"enum": value} for key, value in choices.items()}
        properties.update({"missing_ratio_threshold": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
                           "drop_constant": {"type": "boolean"},
                           "iqr_multiplier": {"type": "number", "exclusiveMinimum": 0},
                           "algorithm_version": {"const": "tabular_prepare_v1"}})
        for key, lower, upper in (("n_bins", 2, 100), ("max_categories", 1, 4096),
                                  ("max_output_features", 1, 100000), ("max_output_cells", 1, 100000000)):
            properties[key] = {"type": "integer", "minimum": lower, "maximum": upper}
        required = [field.name for field in fields(PreparationRules)]
    elif module_id in ("csv_partition_export", "csv_prepare_apply", "csv_local_handoff"):
        properties, required = {}, []
    else:
        raise ValueError("공개 제안에 등록되지 않은 CSV 작업입니다.")

    if set(required) != set(properties):
        raise ValueError("코어 규칙과 공개 설정 명세가 다릅니다.")
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


def workflow_proposal_specifications(*, include_tabular=False):
    """공개 정보만 새 dict로 반환한다. 이 목록은 승인이나 실행 가능 증거가 아니다.

    기본값은 기존 CSV 명세만 돌려준다. 표형 v2 명세는 로컬 계약 검증·결합에서만 명시적으로 포함한다.
    """
    ports = {
        "csv_split_review": ({"data": "source.selected_input"}, "split.confirmed"),
        "csv_partition_export": ({"split": "split.confirmed"}, "export.development"),
        "csv_prepare_candidate": ({"split": "split.confirmed", "development": "export.development"}, "candidate.saved"),
        "csv_prepare_apply": ({"split": "split.confirmed", "development": "export.development",
                               "candidate": "candidate.saved"}, "prepared.saved"),
        "csv_local_handoff": ({"prepared": "prepared.saved"}, "handoff.local"),
    }
    approvals = {
        "csv_split_review": ["audit", "run_eda", "review_eda", "propose_split", "commit_split"],
        "csv_partition_export": ["ingest"], "csv_prepare_candidate": ["derive_preparation"],
        "csv_prepare_apply": ["review_candidate", "apply_preparation"], "csv_local_handoff": ["ingest"],
    }
    local_checks = {
        "csv_split_review": ["snapshot_and_roles_valid", "quality_gate_passed", "eda_reviewed", "split_contract_valid"],
        "csv_partition_export": ["committed_split", "same_source_and_row_linkage"],
        "csv_prepare_candidate": ["committed_split_and_export", "same_source_and_row_linkage", "derive_on_train_only"],
        "csv_prepare_apply": ["committed_candidate", "candidate_reviewed", "same_source_and_row_linkage", "no_rederivation"],
        "csv_local_handoff": ["committed_application", "bundle_integrity_and_row_linkage"],
    }
    result = []
    for item in csv_workflow_records():
        module_id = item["module_id"]
        inputs, output = ports[module_id]
        result.append({**item, "version": VERSION, "inputs": inputs, "output": output,
                       "settings_schema": settings_schema(module_id), "approval_boundaries": approvals[module_id],
                       "required_local_checks": local_checks[module_id],
                       "execution_approval": "separate_explicit_approval", "test_application_allowed": False,
                       "criteria_policy": "derive_on_train" if module_id == "csv_prepare_candidate"
                       else "apply_saved_without_rederivation" if module_id == "csv_prepare_apply" else "none"})

    if not include_tabular:
        return result

    # v2는 같은 역할의 승인 경계·설정 명세를 쓰고, 논리 참조 이름을 구분해 CSV 결과와 섞이지 않게 한다.
    legacy = {item["module_id"].split("_", 1)[1]: item for item in result}
    for item in tabular_workflow_records():
        base = legacy[item["module_id"].split("_", 1)[1]]
        rename = lambda reference: reference if reference == "source.selected_input" else "tabular_" + reference
        result.append({**base, **item, "version": TABULAR_VERSION,
                       "inputs": {key: rename(value) for key, value in base["inputs"].items()},
                       "output": rename(base["output"]),
                       "required_local_checks": base["required_local_checks"] + ["same_tabular_source_contract"]})
    return result
