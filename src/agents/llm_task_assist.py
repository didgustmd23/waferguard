"""표형 작업의 계획 규칙: 모델 전송 집계, 결정 항목 명세, 기본 계획, 계획 전체 검증, 결정적 질문.

공통 조정기·권장안 모듈(task_recommend)이 이 규칙을 호출한다. 모델 출력은 허용 값·계약 검사를 통과한 값만 쓰며
셸·경로·코드로 해석하지 않는다. 원본 행 값과 최솟값·최댓값·분위수 같은 개별 값은 보내지 않는다.
"""

from dataclasses import asdict, fields
import hashlib
import json
import math
import re

from src.preprocessing.split import SplitRules
from src.preprocessing.transform import PreparationRules
from .csv_workflow_proposal_catalogue import settings_schema
from .profile_store import EXCLUDED, LIST_ROLES, ROLE_LABELS, SINGLE_ROLES, SPLIT_FIELDS, build_profile
from .task_plan import PlanError, question


SHARED_FIELDS = ("column", "dtype_group", "missing_ratio", "observed_unique", "blank_string_count",
                 "pos_inf_count", "neg_inf_count", "finite_count", "mean", "std", "outlier_candidate_count")
PREPARATION_FIELDS = tuple(field.name for field in fields(PreparationRules) if field.name != "algorithm_version")
SECTIONS = ("roles", "split", "preparation")

# ==========================================
# 모델 지시문의 역할·분할 제약
# - 역할 이름·개수 제한은 profile_store, 기본값은 SplitRules·PreparationRules에서 가져온다.
# - 분할 방식별 필요 역할은 split._partition의 실행 검사와 같다(테스트로 대조한다).
# - 문자열 시간은 timestamp_format이 필요한데, 작업 분할 필드에 없으면 time 분할을 안내하지 않는다.
# ==========================================
ROLE_MEANINGS = {"numeric": "수치 입력 특징", "categorical": "범주 입력 특징", "identifiers": "행 식별용, 특징에서 제외",
                 "target": "예측 대상", "group": "같은 묶음(예: lot·장비 단위)을 한 구간에 모으는 분할 기준",
                 "timestamp": "측정 시각"}
STRATEGY_ROLES = {"group": "group", "time": "timestamp"}
TIME_SPLIT_AVAILABLE = "timestamp_format" in SPLIT_FIELDS
NO_TARGET = "__no_target__"
TIME_UNSUPPORTED_MESSAGE = ("통합 작업에서는 시간 형식을 지정할 수 없어 time 분할을 지원하지 않습니다. "
                            "random으로 자동 변경하지 않았습니다. 분할 방식을 직접 선택하세요.")
CODE_LABELS = {
    "PLAN_FIELDS_INVALID": "역할·분할·전처리 항목 구성이 계약과 다릅니다",
    "ROLES_INVALID": "역할이 역할 계약(1개만 허용되는 역할 중복, 특징 0개 등)에 맞지 않습니다",
    "SPLIT_INVALID": "분할 기준이 분할 계약(비율 합, 방식 조건 등)에 맞지 않습니다",
    "PROFILE_INVALID": "역할과 분할 기준의 조합이 계약에 맞지 않습니다",
    "PREPARATION_INVALID": "전처리 기준이 허용 선택지에 맞지 않습니다",
    "BINNING_WITH_SCALING": "수치 범주화(binning)와 스케일링(scaling)은 함께 적용할 수 없습니다. 둘 중 하나를 none으로 고르세요",
    "BINNING_NEEDS_IMPUTATION": "수치 범주화(binning)에는 수치 결측 대치가 필요합니다. numeric_imputation을 none 외의 값으로 고르거나 binning을 none으로 고르세요",
    "CATALOGUE_PLAN_INVALID": "처리 단계 계획이 작업 카탈로그 검증을 통과하지 못했습니다",
    "SPLIT_STRATEGY_MISSING": "분할 방식이 정해지지 않았습니다",
    "TIME_SPLIT_UNSUPPORTED": "통합 작업에서 지원하지 않는 time 분할입니다",
    "SPLIT_ROLE_MISMATCH": "역할과 분할 방식이 맞지 않습니다(group·target 역할 조건)",
    # 분할 계약 거부의 하위 원인(split_violation). 확인된 조건만 이름을 붙이고 나머지는 미확인으로 둔다.
    "SPLIT_RATIO_RANGE": "분할 비율 중 0 이하·1 이상이거나 유한수가 아닌 값이 있습니다",
    "SPLIT_RATIO_SUM": "Train·Validation·Test 비율의 합이 1이 아닙니다",
    "SPLIT_STRATIFY_STRATEGY": "target 기준 계층 분할을 random이 아닌 분할 방식과 함께 정했습니다",
    # 역할×분할 교차 거부(check_split_roles)의 하위 원인. 역할 컬럼명은 남기지 않고 분할 방식·계층 여부로만 구분한다.
    "SPLIT_ROLE_REQUIRED_MISSING": "분할 방식에 필요한 역할 컬럼이 없습니다(group 분할에는 group 역할 필요)",
    "SPLIT_GROUP_STRATEGY": "group 역할을 지정했는데 group이 아닌 분할 방식을 정했습니다",
    "SPLIT_STRATIFY_TARGET_MISSING": "target 기준 계층 분할을 정했지만 target 역할이 없습니다",
    "SPLIT_REASON_UNCONFIRMED": "분할 계약의 다른 조건에 맞지 않습니다(하위 원인을 분류하지 않음)",
}
STRATIFY_MESSAGE = "target 기준 계층 분할은 random 분할이고 target 역할이 있을 때만 쓸 수 있습니다."
# 분할 거부 진단에 남길 수 있는 계약 항목. 자유 문자열(timestamp_format)·시드·역할 컬럼명은 넣지 않는다.
SPLIT_DIAGNOSTIC_KEYS = ("train_ratio", "validation_ratio", "test_ratio", "strategy", "stratify")
SPLIT_RATIO_KEYS = ("train_ratio", "validation_ratio", "test_ratio")

# ==========================================
# 분할 거부 진단의 허용 명세(생성·표시 공용)
# - 구획은 split만, 하위 원인은 계획 검증 코드별 목록 안에서만 허용한다
# - values·omitted는 SPLIT_DIAGNOSTIC_KEYS를 겹치지 않고 빠짐없이 나눈다. 값 조건은 _split_value_ok 하나로 판정한다
# - from_model은 values에 있는 키 이름만 중복 없이 정렬해 둔다
# - 하위 원인과 기록한 비율 값이 split_violation과 같은 식으로 맞아야 한다
# - 비율 0·1을 values에 남기는 생성 정책(계약은 열린 구간)은 그대로 둔다
# ==========================================
REJECTION_KEYS = {"section", "reason", "values", "from_model", "omitted"}
REJECTION_REASONS = {"SPLIT_INVALID": ("SPLIT_RATIO_RANGE", "SPLIT_RATIO_SUM", "SPLIT_REASON_UNCONFIRMED"),
                     "SPLIT_ROLE_MISMATCH": ("SPLIT_STRATIFY_STRATEGY", "SPLIT_ROLE_REQUIRED_MISSING", "SPLIT_GROUP_STRATEGY",
                                             "SPLIT_STRATIFY_TARGET_MISSING")}


def _label(field):
    return next(label for roles in (LIST_ROLES, SINGLE_ROLES) for label, key in roles.items() if key == field)


def _defaults():
    split = {key: value for key, value in asdict(SplitRules()).items() if key in SPLIT_FIELDS}
    preparation = {key: value for key, value in asdict(PreparationRules()).items() if key in PREPARATION_FIELDS}
    return split, preparation


def instructions():
    """표형 계획의 역할 의미·분할 제약. 기본값은 user 메시지의 base_plan으로 따로 보낸다.

    표형 v1·v2가 공유하는 안내일 뿐 계약이 아니다. 모델이 따르지 않아도 병합한 계획은 별도 검증(허용 값·분할 제약)에서
    전체 거부되고 기본 계획으로 진행한다. 길이는 전달 범위(token_budget)를 줄이므로 판단에 필요한 의미만 둔다.
    """
    roles = [f"{label}={ROLE_MEANINGS[key]}(여러 컬럼 가능)" for label, key in LIST_ROLES.items()]
    roles += [f"{label}={ROLE_MEANINGS[key]}(최대 1개, 해당 컬럼이 없으면 쓰지 않음)" for label, key in SINGLE_ROLES.items()]
    roles.append(f"{EXCLUDED}=사용하지 않음")
    strategies = [f"strategy={strategy}는 {_label(role)} 역할이 있을 때만" for strategy, role in STRATEGY_ROLES.items()
                  if strategy != "time" or TIME_SPLIT_AVAILABLE]
    strategies.append(f"{_label('group')} 역할이 있으면 strategy=group")

    if not TIME_SPLIT_AVAILABLE:
        strategies.append(f"이 작업에서는 시간 형식을 지정할 수 없으므로 strategy=time을 쓰지 말고 {_label('timestamp')}는 역할만 표시")

    strategies.append(f"stratify=true는 strategy=random이고 {_label('target')} 역할이 있을 때만")
    return (
        "이 작업은 표형 데이터입니다. path는 roles.<컬럼>, split.<항목>, preparation.<항목> 형식입니다. "
        "subjects는 컬럼 목록입니다. detailed는 detail_fields 순서의 컬럼별 집계, named는 집계 없이 이름만, "
        "summarized는 개수만 제공한 컬럼입니다. reasons는 상세로 보낸 근거 코드입니다. "
        "base_plan.roles.by_value는 역할별 컬럼 목록입니다. "
        "base_plan의 roles는 자료형만 보고 정한 초안이므로 의미에 맞게 바꿔야 할 컬럼만 changes에 쓰세요. "
        "역할: " + "; ".join(roles) + ". 수치형·범주형 특징이 하나 이상 있어야 합니다. "
        "값이 하나뿐이거나 모두 비어 있는 컬럼, observed_unique가 row_count와 같은 문자열 컬럼처럼 행마다 다른 식별 번호로 "
        "보이는 컬럼은 특징으로 쓰지 마세요. 목표에 예측 대상이 있으면 해당 컬럼을 target으로 두세요. "
        "split 제약: train_ratio·validation_ratio·test_ratio는 각각 0과 1 사이이고 합이 1; " + "; ".join(strategies) + ". "
        "병합한 계획 전체가 이 제약을 만족해야 하며(아니면 권장안 전체 거부), 역할을 바꾸면 관련 split 항목도 함께 쓰세요. "
        "Test 구간에 맞추는 규칙은 없습니다. "
        # 전처리 옵션의 적용 범위와 few_values 의미. 컬럼별 의미는 데이터마다 다르므로 특정 컬럼명을 예로 들지 않는다.
        "전처리 옵션은 최종 역할 기준으로 일괄 적용됩니다(numeric_imputation·outlier·scaling·binning은 수치형 전체, "
        "categorical_imputation·unknown_category는 범주형 전체). binning은 수치형 특징의 구간화이며 범주형 인코딩이 아닙니다. "
        "few_values는 관측값 종류가 적다는 뜻으로 숫자 범주 코드일 수 있으니 역할을 검토하되 이것만으로 단정하지 마세요. "
        "역할과 옵션을 함께 적용한 계획 전체를 고려하고, 데이터 설명은 제공된 집계와 일치시키세요."
    )


def shared_columns(columns):
    """모델에 보낼 컬럼 집계. 원본 행과 개별 값(최솟값·최댓값·분위수·IQR 경계)은 포함하지 않는다.

    값이 하나뿐인 컬럼이나 유한값이 2개 이하인 컬럼은 평균·표준편차로 개별 값이 복원되므로 둘 다 비운다.
    """
    shared = []

    for row in columns:
        item = {name: row[name] for name in SHARED_FIELDS}
        unique, finite = row["observed_unique"], row["finite_count"]

        if (unique is not None and unique <= 1) or (finite is not None and finite <= 2):
            item.update(mean=None, std=None)

        shared.append(item)

    return shared


def model_data(summary, columns):
    return {"row_count": summary["row_count"], "repeated_rows": summary["repeated_rows"], "columns": shared_columns(columns)}


def _round(value):
    """모델에 보낼 집계 실수는 유효숫자 4자리로 줄인다. 집계 값이며 개별 값이 아니다."""
    return float(f"{value:.4g}") if isinstance(value, float) else value


ID_NAME = re.compile(r"(^|[^a-z0-9])(id|no|num|idx|index|key|seq|번호|순번|일련번호|아이디)($|[^a-z0-9])")


TIME_NAME = re.compile(r"(time|date|timestamp|datetime|일시|시각|날짜|시간)")


def time_like_name(name):
    """이름에 시각·날짜 단서가 있는지. 값은 보지 않으므로 역할 확인 질문의 근거로만 쓴다."""
    return bool(TIME_NAME.search(name.casefold()))


def category_limit_risk(summary, columns, plan):
    """범주형 특징 중 고유값이 원핫 범주 상한을 넘는 것. 반드시 실패하는지(certain)도 함께 돌려준다.

    Train에만 있는 범주로 계산하므로 전체 고유값이 상한을 넘어도 Train은 넘지 않을 수 있다. Train에 빠질 수 있는 최대
    행 수를 빼도 상한을 넘으면 반드시 실패한다. missing_category는 결측 범주 하나를 항상 더한다.
    """
    roles, preparation = plan["roles"], plan["preparation"]
    limit = preparation.get("max_categories", PreparationRules().max_categories)
    extra = 1 if preparation.get("categorical_imputation") == "missing_category" else 0
    rows = summary["row_count"]
    outside = rows - int(rows * float(plan["split"].get("train_ratio") or 0))
    # group 분할은 묶음 단위로 나눠 Train 행 수가 비율보다 적을 수 있다. 반드시 실패한다고 판정하지 않는다.
    exact = plan["split"].get("strategy") == "random"
    result = []

    for row in columns:
        unique = row["observed_unique"]

        if roles.get(row["column"]) == "범주형" and unique is not None and unique + extra > limit:
            result.append({"column": row["column"], "unique": unique, "limit": limit,
                           "certain": exact and unique - outside + extra > limit})

    return result


def id_like_name(name):
    """이름에 식별 번호 단서가 있는지. 단어 경계 또는 이름 끝의 'id'(예: PassengerId)만 본다."""
    text = name.casefold()
    return bool(ID_NAME.search(text)) or (text.endswith("id") and len(text) > 2)


def recommendation_items(goal, summary, columns, plan):
    """권장안 판단 대상(컬럼)별 상세 집계·우선순위·근거·요약 구분·연결 계획 항목. 배치와 예산은 공통 계층이 정한다.

    우선순위 3(반드시 상세): target·식별자 판단 근거가 있는 컬럼(목표 문장에 이름 등장·값 종류 2~20·식별 번호 이름·
    비수치·행마다 다른 값). 우선순위 2(상세 우선): 결측 5% 이상. 우선순위 0: 값이 하나뿐이거나 모두 비어 기본 계획에서
    이미 제외한 컬럼. 나머지는 1. 같은 우선순위 안에서는 목표 문장 이름 → 값 종류 적음 순서다.
    """
    detail_keys = [name for name in SHARED_FIELDS if name != "column"]
    observed_rows = summary["row_count"]
    items = []

    for row, shared in zip(columns, shared_columns(columns)):
        name, unique = row["column"], row["observed_unique"]
        reasons = []

        if unique is not None and unique <= 1:
            reasons.append("single_or_empty")
        else:
            if row["dtype_group"] != "numeric":
                reasons.append("non_numeric")
            if unique is not None and 2 <= unique <= 20:
                reasons.append("few_values")
            if unique is not None and unique == observed_rows - (row.get("missing_count") or 0):
                reasons.append("all_distinct")
            if (row.get("missing_ratio") or 0) >= 0.05:
                reasons.append("missing")
            if name.casefold() in goal.casefold():
                reasons.append("named_in_goal")
            if id_like_name(name):
                reasons.append("id_like_name")

        decisive = {"named_in_goal", "few_values", "id_like_name", "non_numeric", "all_distinct"}
        priority = (0 if "single_or_empty" in reasons else 3 if decisive.intersection(reasons) else 2 if reasons else 1)
        missing = row.get("missing_ratio") or 0
        bucket = f"{row['dtype_group']}·결측 {'없음' if missing == 0 else '5% 미만' if missing < 0.05 else '5% 이상'}"
        items.append({"id": name, "priority": priority, "reasons": reasons, "bucket": bucket, "paths": ["roles." + name],
                      "detail": {key: _round(shared[key]) for key in detail_keys}})

    # 공통 계층은 우선순위만 보고 안정 정렬한다. 같은 우선순위 안의 순서는 어댑터가 판단 근거 강도로 정한다.
    return sorted(items, key=lambda item: ("named_in_goal" not in item["reasons"], "few_values" not in item["reasons"]))


# ==========================================
# 결정 항목 명세와 기본 계획
# ==========================================

def _field(path, schema):
    if "enum" in schema:
        return {"path": path, "type": "enum", "options": list(schema["enum"])}

    kinds = schema["type"] if type(schema["type"]) is list else [schema["type"]]
    kind = next(item for item in kinds if item != "null")
    field = {"path": path, "type": kind, "nullable": "null" in kinds}

    for key in ("minimum", "maximum"):
        if key in schema:
            field[key] = schema[key]

    # 비율의 열린 구간은 형식 확인에서 닫힌 구간으로 두고 SplitRules가 최종 판정한다.
    if "exclusiveMinimum" in schema:
        field["minimum"] = schema["exclusiveMinimum"]

    if "exclusiveMaximum" in schema:
        field["maximum"] = schema["exclusiveMaximum"]

    return field


def plan_fields(columns):
    split = settings_schema("csv_split_review")["properties"]
    preparation = settings_schema("csv_prepare_candidate")["properties"]
    result = [{"path": "roles." + row["column"], "type": "enum", "options": list(ROLE_LABELS)} for row in columns]
    result += [_field("split." + key, split[key]) for key in sorted(SPLIT_FIELDS)]
    result += [_field("preparation." + key, preparation[key]) for key in PREPARATION_FIELDS]

    strategy = next(item for item in result if item["path"] == "split.strategy")
    # 분할 방식은 time 제안 거부 뒤 비워 둘 수 있다. time은 형식상 받아 계획 검증에서 미지원으로 거부하고, 화면 선택지에서는 뺀다.
    strategy["nullable"] = True

    if not TIME_SPLIT_AVAILABLE:
        strategy["unsupported"] = ["time"]

    return result


def base_plan(columns):
    """자료형만 보고 정한 기본 계획. 확정은 사용자가 한다."""
    roles = {}

    for row in columns:
        if row["observed_unique"] is not None and row["observed_unique"] <= 1:
            roles[row["column"]] = EXCLUDED
        elif row["dtype_group"] == "numeric":
            roles[row["column"]] = "수치형"
        else:
            roles[row["column"]] = "범주형"

    split, preparation = _defaults()
    return {"roles": roles, "split": split, "preparation": preparation}


# ==========================================
# 계획 전체 검증 (초안 병합·화면 선검사·서버 확인이 같은 함수를 쓴다)
# ==========================================

class SplitRoleError(ValueError):
    """역할과 분할 방식의 조합 오류. 메시지는 입력 자료가 들어가지 않는 고정 문구, reason은 SPLIT_ROLE_MISMATCH의 고정 하위 코드다."""

    def __init__(self, code, message, reason=None):
        super().__init__(message)
        self.code, self.message, self.reason = code, message, reason


def check_split_roles(roles, split):
    """역할×분할 교차 검사. split._partition 실행 검사와 같은 조건이다. 기존 계약 검사를 대신하지 않고 더한다."""
    assigned = set(roles.values())
    strategy = split.get("strategy")

    if strategy not in ("random", *STRATEGY_ROLES):
        raise SplitRoleError("SPLIT_STRATEGY_MISSING", "분할 방식을 선택하세요.")

    if strategy == "time" and not TIME_SPLIT_AVAILABLE:
        raise SplitRoleError("TIME_SPLIT_UNSUPPORTED", TIME_UNSUPPORTED_MESSAGE)

    required = STRATEGY_ROLES.get(strategy)

    if required is not None and _label(required) not in assigned:
        raise SplitRoleError("SPLIT_ROLE_MISMATCH", f"{strategy} 분할에는 {_label(required)} 역할 컬럼이 필요합니다.",
                             "SPLIT_ROLE_REQUIRED_MISSING")

    if strategy == "random" and _label("group") in assigned:
        # 같은 묶음을 행 단위로 섞어 누수 위험을 숨기지 않는다.
        raise SplitRoleError("SPLIT_ROLE_MISMATCH", "group 역할을 지정했으면 group 분할을 선택하세요.", "SPLIT_GROUP_STRATEGY")

    if split.get("stratify") is True and (strategy != "random" or _label("target") not in assigned):
        raise SplitRoleError("SPLIT_ROLE_MISMATCH", STRATIFY_MESSAGE,
                             "SPLIT_STRATIFY_STRATEGY" if strategy != "random" else "SPLIT_STRATIFY_TARGET_MISSING")


def split_violation(split):
    """분할 계약(SplitRules.__post_init__)과 같은 순서·같은 식으로 첫 실패 조건의 고정 하위 코드를 돌려준다. 통과하면 None.

    예외 문장을 해석하지 않는다. 이름 붙인 세 조건 밖의 실패(분할 방식·시드·bool·시간 형식·버전)는 미확인 코드로 둔다.
    """
    values = {**{field.name: field.default for field in fields(SplitRules)}, **split}
    ratios = (values["train_ratio"], values["validation_ratio"], values["test_ratio"])

    if any(type(value) not in (int, float) or not math.isfinite(value) or not 0 < value < 1 for value in ratios):
        return "SPLIT_RATIO_RANGE"

    if not math.isclose(sum(ratios), 1.0, abs_tol=1e-12):
        return "SPLIT_RATIO_SUM"

    # 집합 대신 튜플로 비교해 해시할 수 없는 값도 TypeError 없이 미확인으로 둔다(SplitRules도 이 값을 거부한다).
    if (type(values["strategy"]) is not str or values["strategy"] not in ("random", "group", "time")
            or type(values["random_state"]) is not int or not 0 <= values["random_state"] < 2**32
            or type(values["stratify"]) is not bool or type(values["require_all_classes"]) is not bool):
        return "SPLIT_REASON_UNCONFIRMED"

    if values["stratify"] and values["strategy"] != "random":
        return "SPLIT_STRATIFY_STRATEGY"

    try:
        SplitRules(**split)
    except (ValueError, TypeError):
        return "SPLIT_REASON_UNCONFIRMED"

    return None


def _split_values(split):
    """분할 거부 진단에 남길 계약 값. 허용 목록·타입·범위·유한성을 통과한 값만 남기고 나머지는 키 이름만 omitted로 둔다."""
    values, omitted = {}, []

    for key in SPLIT_DIAGNOSTIC_KEYS:
        value = split.get(key)

        if _split_value_ok(key, value):
            values[key] = value
        else:
            omitted.append(key)

    return values, omitted


def _split_value_ok(key, value):
    # 범위 비교를 먼저 해 큰 정수의 float 변환 오류(OverflowError) 없이 거부한다. bool은 비율로 받지 않는다.
    if key in SPLIT_RATIO_KEYS:
        return type(value) in (int, float) and 0 <= value <= 1 and math.isfinite(value)

    if key == "strategy":
        # 분할 방식 없음(None)은 계획 검증이 진단 없이 먼저 거부한다. 진단 값으로 생길 수 없어 허용하지 않는다.
        return type(value) is str and value in ("random", "group", "time")

    return key == "stratify" and type(value) is bool


def _reason_matches(reason, values):
    # 기록한 값으로 split_violation의 비율 조건을 다시 판정한다. 비율이 빠졌거나 0·1이면 범위 위반만 맞다.
    inside = all(key in values and 0 < values[key] < 1 for key in SPLIT_RATIO_KEYS)

    if reason == "SPLIT_RATIO_RANGE" or not inside:
        return reason == "SPLIT_RATIO_RANGE" and not inside

    summed = math.isclose(sum(values[key] for key in SPLIT_RATIO_KEYS), 1.0, abs_tol=1e-12)

    if reason == "SPLIT_RATIO_SUM" or not summed:
        return reason == "SPLIT_RATIO_SUM" and not summed

    strategy, stratify = values.get("strategy"), values.get("stratify")

    if reason == "SPLIT_STRATIFY_STRATEGY":
        return stratify is True and strategy in ("group", "time")

    # 역할×분할 교차 하위 원인은 분할 계약을 통과한 값에서만 생긴다(계층+비random은 계약 단계가 먼저 거부한다).
    if reason == "SPLIT_ROLE_REQUIRED_MISSING":
        return stratify is False and strategy in STRATEGY_ROLES and (strategy != "time" or TIME_SPLIT_AVAILABLE)

    if reason == "SPLIT_GROUP_STRATEGY":
        return strategy == "random" and type(stratify) is bool

    if reason == "SPLIT_STRATIFY_TARGET_MISSING":
        return strategy == "random" and stratify is True

    return True


def rejection_valid(code, rejection):
    """저장된 분할 거부 진단이 생성 규칙의 허용 명세에 맞는지 확인한다(표시 직전 경계). 값을 고치거나 버리지 않는다."""
    if type(code) is not str or type(rejection) is not dict or set(rejection) != REJECTION_KEYS:
        return False

    values, omitted, from_model = rejection["values"], rejection["omitted"], rejection["from_model"]

    if (rejection["section"] != "split" or type(rejection["reason"]) is not str
            or rejection["reason"] not in REJECTION_REASONS.get(code, ())
            or type(values) is not dict or type(omitted) is not list or type(from_model) is not list
            or not all(type(key) is str for key in [*values, *omitted, *from_model])):
        return False

    if (len(set(omitted)) != len(omitted) or set(values) & set(omitted) or set(values) | set(omitted) != set(SPLIT_DIAGNOSTIC_KEYS)
            or not all(_split_value_ok(key, value) for key, value in values.items())
            or from_model != sorted(set(from_model)) or not set(from_model) <= set(values)):
        return False

    return _reason_matches(rejection["reason"], values)


def _profile_reason(names, roles, split):
    try:
        build_profile(tuple(names), roles, purpose="eda", profile_id="plan_check")
    except (ValueError, TypeError):
        return "ROLES_INVALID"

    try:
        SplitRules(**split)
    except (ValueError, TypeError):
        return "SPLIT_INVALID"

    return "PROFILE_INVALID"


def validate_plan(columns, plan):
    """계획 전체를 기존 계약으로 확인하고 프로파일·전처리 규칙을 돌려준다. 실패하면 고정 코드의 PlanError."""
    names = tuple(row["column"] for row in columns)

    if (type(plan) is not dict or set(plan) != set(SECTIONS) or any(type(plan[key]) is not dict for key in SECTIONS)
            or set(plan["roles"]) != set(names) or set(plan["split"]) != SPLIT_FIELDS
            or set(plan["preparation"]) != set(PREPARATION_FIELDS)):
        raise PlanError("PLAN_FIELDS_INVALID", CODE_LABELS["PLAN_FIELDS_INVALID"])

    roles, split = plan["roles"], plan["split"]

    if split.get("strategy") is None:
        raise PlanError("SPLIT_STRATEGY_MISSING", CODE_LABELS["SPLIT_STRATEGY_MISSING"])

    try:
        profile = build_profile(names, roles, purpose="split", profile_id="plan_check", split=split)
    except (ValueError, TypeError) as error:
        code = _profile_reason(names, roles, split)

        if code != "SPLIT_INVALID":
            raise PlanError(code, CODE_LABELS[code]) from error

        # 분할 계약 거부는 계약과 같은 순서의 첫 실패 조건으로 하위 원인을 정한다. 역할·형식 오류의 우선순위는 그대로다.
        reason = split_violation(split)
        values, omitted = _split_values(split)
        detail = {"section": "split", "reason": reason, "values": values, "omitted": omitted}

        if reason == "SPLIT_STRATIFY_STRATEGY":
            # group 등과 계층 분할의 조합은 역할·분할 방식 교차 검사와 같은 코드·고정 문구로 알린다(일반 SPLIT_INVALID에 가리지 않음).
            raise PlanError("SPLIT_ROLE_MISMATCH", STRATIFY_MESSAGE, subjects=["split.strategy", "split.stratify"],
                            detail=detail) from error

        raise PlanError(code, CODE_LABELS[code], detail=detail) from error

    try:
        check_split_roles(roles, split)
    except SplitRoleError as error:
        detail = None

        if error.reason is not None:
            # 분할 계약 거부와 같은 진단 구조. 하위 코드와 허용 명세를 통과한 분할 값만 두고 역할 컬럼명은 넣지 않는다.
            values, omitted = _split_values(split)
            detail = {"section": "split", "reason": error.reason, "values": values, "omitted": omitted}

        raise PlanError(error.code, error.message, detail=detail) from error

    preparation = plan["preparation"]

    # 항목별로는 허용된 값이어도 함께 쓸 수 없는 조합이 있다. 전처리 계약과 같은 조건을 어느 항목끼리 충돌하는지와 함께 알린다.
    if preparation.get("binning", "none") != "none":
        if preparation.get("scaling", "none") != "none":
            raise PlanError("BINNING_WITH_SCALING", CODE_LABELS["BINNING_WITH_SCALING"],
                            subjects=["preparation.binning", "preparation.scaling"])

        if preparation.get("numeric_imputation") == "none":
            raise PlanError("BINNING_NEEDS_IMPUTATION", CODE_LABELS["BINNING_NEEDS_IMPUTATION"],
                            subjects=["preparation.binning", "preparation.numeric_imputation"])

    try:
        rules = PreparationRules(**preparation)
    except (ValueError, TypeError) as error:
        raise PlanError("PREPARATION_INVALID", CODE_LABELS["PREPARATION_INVALID"]) from error

    return profile, rules


# ==========================================
# 결정적 질문 (설계 7.5절의 표형 질문)
# ==========================================

def numeric_identifier_like(summary, columns, roles):
    """수치형 특징 중 식별 번호로 보이는 컬럼. 고유값이 많다는 것만으로는 고르지 않는다.

    결측을 뺀 모든 값이 서로 다르고, 이름에 식별 번호 단서가 있거나 값이 빈틈없는 연속 정수 범위
    (최댓값−최솟값+1 = 고유값 수)일 때만 고른다. 최솟값·최댓값은 로컬 진단 표에서만 쓰고 모델에 보내지 않는다.
    """
    names = []

    for row in columns:
        unique = row["observed_unique"]
        observed = summary["row_count"] - (row.get("missing_count") or 0)

        if roles.get(row["column"]) != "수치형" or unique is None or unique <= 2 or unique != observed:
            continue

        low, high = row.get("min"), row.get("max")
        consecutive = (isinstance(low, (int, float)) and isinstance(high, (int, float)) and float(low).is_integer()
                       and float(high).is_integer() and high - low + 1 == unique)

        if id_like_name(row["column"]) or consecutive:
            names.append(row["column"])

    return names


def identifier_like(summary, columns, roles):
    """범주형 특징 중 관측 고유값 수가 결측을 뺀 행 수와 같은 컬럼. 수치형 연속값은 대상이 아니다."""
    names = []

    for row in columns:
        observed = summary["row_count"] - (row.get("missing_count") or 0)

        if (roles.get(row["column"]) == "범주형" and row["observed_unique"] is not None
                and row["observed_unique"] > 1 and row["observed_unique"] == observed):
            names.append(row["column"])

    return names


def target_checks(goal, roles):
    """목표 문장과 target 지정이 맞물리지 않으면 확인 문구를 돌려준다. 역할을 바꾸거나 진행을 막지 않는다."""
    targets = [name for name, role in roles.items() if role == _label("target")]

    if not targets:
        return ["target이 지정되지 않았습니다. 예측 대상이 있으면 역할 표에서 target을 지정하세요."]

    if targets[0].casefold() not in goal.casefold():
        return [f"목표 문장에 target 컬럼 이름이 없어 추정한 지정입니다. target={targets[0]}이(가) 맞는지 확인하세요."]

    return []


def plan_questions(goal, summary, columns, plan):
    """현재 계획에서 사용자가 정하거나 확인해야 할 항목. 필수 여부는 최종 계획에서 다시 계산한다."""
    roles, split = plan["roles"], plan["split"]
    role_options = [{"value": label, "label": label} for label in ROLE_LABELS]
    result = []
    strategies = [option for option in ("random", "group", "time") if option != "time" or TIME_SPLIT_AVAILABLE]

    if split.get("strategy") is None:
        result.append(question("split.strategy", "required", "분할 방식을 고르세요. " + TIME_UNSUPPORTED_MESSAGE,
                               path="split.strategy", options=[{"value": item, "label": item} for item in strategies]))
    else:
        try:
            check_split_roles(roles, split)
        except SplitRoleError as error:
            result.append(question("split.strategy.mismatch", "required", error.message, path="split.strategy",
                                   options=[{"value": item, "label": item} for item in strategies]))

    for name in identifier_like(summary, columns, roles):
        text = "값이 행마다 모두 다른 범주형 특징입니다. 식별 번호라면 식별자로 지정하세요."

        if plan["preparation"].get("unknown_category") == "error":
            text += " 범주형으로 두면 Validation에 Train에 없던 값이 생겨 전처리 적용이 실패할 수 있습니다."

        # 질문 ID는 형식이 고정된 해시로 만든다. 컬럼 이름(대문자·한글·공백 가능)은 path·subjects에만 둔다.
        qid = "identifier:" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]
        result.append(question(qid, "advisory", text, path="roles." + name,
                               options=[option for option in role_options if option["value"] in ("식별자", "범주형", EXCLUDED)],
                               recommended="식별자", subjects=[name]))

    for name in numeric_identifier_like(summary, columns, roles):
        qid = "identifier:" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]
        result.append(question(qid, "advisory", "값이 행마다 모두 다른 수치형 특징이며 이름이나 값 범위가 식별 번호처럼 보입니다. "
                                                "측정값이면 수치형으로 두고, 행 번호·ID라면 식별자로 지정하세요.",
                               path="roles." + name,
                               options=[option for option in role_options if option["value"] in ("식별자", "수치형", EXCLUDED)],
                               recommended="식별자", subjects=[name]))

    # timestamp 역할은 시간 형식을 지정할 수 있을 때만 쓸 수 있다. 형식 없이 지정하면 분할 전 품질 검사가 검토 필요로 멈춘다.
    time_role = TIME_SPLIT_AVAILABLE and _label("timestamp") not in roles.values()
    rows = summary["row_count"]
    asked = set()

    for row in columns:
        name = row["column"]

        if roles.get(name) == _label("timestamp") and not TIME_SPLIT_AVAILABLE:
            asked.add(name)
            qid = "time_role:" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]
            result.append(question(qid, "required", "timestamp 역할은 시간 형식이 필요하지만 통합 작업에서는 형식을 지정할 수 없어 "
                                                    "분할 전 품질 검사가 멈춥니다. 시간 정보를 특징에서 빼려면 제외를 고르세요.",
                                   path="roles." + name, options=[option for option in role_options if option["value"] == EXCLUDED],
                                   recommended=EXCLUDED, subjects=[name]))

    for item in category_limit_risk(summary, columns, plan):
        name = item["column"]
        asked.add(name)
        row = next(row for row in columns if row["column"] == name)
        time_like = time_like_name(name)
        # 식별자는 중복 검사 대상이다. 값이 행마다 모두 다를 때만 선택지로 둔다.
        distinct = row["observed_unique"] == rows - (row["missing_count"] or 0)
        allowed = [EXCLUDED] + (["식별자"] if distinct else []) + ([_label("timestamp")] if time_like and time_role else [])
        text = f"범주형 특징의 고유값이 {item['unique']}개로 원핫 범주 상한 {item['limit']}개를 넘습니다. "
        text += ("Train만으로도 상한을 넘으므로 범주형으로는 전처리 후보 계산이 실패합니다. " if item["certain"] else
                 "Train에 포함되는 범주 수에 따라 전처리 후보 계산이 실패할 수 있습니다. ")

        if time_like:
            text += ("이름이 시각·날짜처럼 보입니다. 시간 정보라면 timestamp로 지정하세요. " if time_role else
                     "이름이 시각·날짜처럼 보입니다. 통합 작업은 시간 형식을 지정할 수 없어 timestamp로 쓸 수 없으니 제외를 고르세요. ")

        text += "행마다 다른 ID라면 식별자로 지정하세요." if distinct else "특징으로 쓰지 않으려면 제외를 고르세요."
        qid = "category_limit:" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]
        # 반드시 실패하면 필수 질문이다. 역할을 바꿔야 해결되며 자동으로 바꾸지 않는다. 불확실하면 안내 질문이다.
        result.append(question(qid, "required" if item["certain"] else "advisory", text, path="roles." + name,
                               options=[option for option in role_options
                                        if option["value"] in allowed or (not item["certain"] and option["value"] == "범주형")],
                               recommended=_label("timestamp") if time_like and time_role else EXCLUDED if time_like or not distinct
                               else "식별자", subjects=[name]))

    for row in columns:
        name = row["column"]

        if name not in asked and row["dtype_group"] != "numeric" and roles.get(name) == "범주형" and time_like_name(name):
            qid = "time_candidate:" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]
            text = ("이름이 시각·날짜처럼 보이는 범주형 특징입니다. 시간 정보라면 timestamp로 지정해 특징에서 빼고 보존하세요."
                    if time_role else "이름이 시각·날짜처럼 보이는 범주형 특징입니다. 통합 작업은 시간 형식을 지정할 수 없어 timestamp로 "
                                      "쓸 수 없습니다. 시간 정보를 특징으로 쓰지 않으려면 제외를 고르세요.")
            result.append(question(qid, "advisory", text, path="roles." + name,
                                   options=[option for option in role_options
                                            if option["value"] in ("범주형", EXCLUDED) or (time_role and option["value"] == _label("timestamp"))],
                                   recommended=_label("timestamp") if time_role else EXCLUDED, subjects=[name]))

    targets = [name for name, role in roles.items() if role == _label("target")]

    if not targets:
        # target이 없으면 목적을 확인하기 전에는 확정하지 않는다. 예측 목적이면 target 컬럼을, 아니면 "예측 대상 없음"을
        # 고른다. 고른 컬럼은 roles.<컬럼> 한 항목만 target으로 바꾼다. 다른 역할은 바꾸지 않는다.
        candidates = sorted((row for row in columns if row["observed_unique"] is not None and 1 < row["observed_unique"] <= 20
                             and row["column"] != NO_TARGET), key=lambda row: row["observed_unique"])[:10]
        options = [{"value": row["column"], "label": row["column"], "path": "roles." + row["column"], "set": _label("target")}
                   for row in candidates]
        options.append({"value": NO_TARGET, "label": "예측 대상 없음(예측 목적이 아닌 전처리)"})
        result.append(question("target", "required", "target(예측 대상)이 지정되지 않았습니다. 목표가 예측이면 target 컬럼을 고르세요. "
                                                     "후보에 없으면 역할 표에서 지정하세요. 예측 목적이 아니면 '예측 대상 없음'을 고르세요.",
                               options=options, subjects=[row["column"] for row in candidates]))
    elif targets[0].casefold() not in goal.casefold():
        result.append(question("target", "advisory", "목표 문장에 target 컬럼 이름이 없어 추정한 지정입니다. 맞는지 확인하세요.",
                               subjects=targets))

    return result


def on_rejected(draft):
    """time 분할 제안이 거부되면 분할 방식을 비워 사용자가 직접 고르게 한다. random으로 대체하지 않는다."""
    if draft["code"] == "TIME_SPLIT_UNSUPPORTED":
        plan = json.loads(json.dumps(draft["plan"]))
        plan["split"]["strategy"] = None
        return dict(draft, plan=plan, notes=[*draft["notes"], "모델이 time 분할을 제안했습니다. " + TIME_UNSUPPORTED_MESSAGE])

    return draft
