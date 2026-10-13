# ==========================================
# OOF 후보 표에서 Recall·재검사 조건에 맞는 threshold 선택
# - 재검사 비율은 (TP + FP) / 전체 표본 수로 계산
# - 재검사 최소 → Recall 최대 → threshold 최대 순으로 동률 처리
# - 조건 미충족은 None으로 반환하며 F1 최대값으로 대체하지 않음
# ==========================================

import numpy as np
import pandas as pd

try:
    from src.split_contract import PROTOCOL_ID
except ModuleNotFoundError:
    from split_contract import PROTOCOL_ID


# ==========================================
# 후보 표를 검증하고 재검사 비율·정책 충족 여부 추가
# - table은 같은 OOF 평가 대상에 여러 문턱을 적용한 결과
# - probability: threshold도 0~1의 확률 범위로 제한
# - anomaly: threshold에 유한한 실수를 허용하고 점수 방향은 호출자가 통일
# - policy는 최소 Recall·최대 재검사 비율을 가진 설정 객체
# - 복사본을 반환하므로 원본 후보 표와 기존 로그는 수정하지 않음
# ==========================================
def policy_candidates(table, policy, *, score_kind="probability"):
    """OOF 후보표를 검증하고 재검사 비율 및 정책 충족 여부를 계산한다.

    입력 표를 변경하지 않고 복사본을 반환한다. 확률 점수와 anomaly 점수의
    threshold 범위는 다르게 검사하며, 표본·오류 건수 및 Recall 일관성도 확인한다.
    """
    # 점수 종류 오타로 확률 범위 검사가 우회되지 않게 먼저 확인한다.
    if score_kind not in {"probability", "anomaly"}:
        raise ValueError("점수 종류는 probability 또는 anomaly여야 합니다.")
    
    # 로그에 이미 저장된 비율은 신뢰하지 않고 오류 건수에서 다시 계산한다.
    columns = ["threshold", "recall", "support", "positive_support",
               "true_positive", "false_positive"]
    
    if table.empty or not set(columns).issubset(table.columns):
        raise ValueError("정책 선택용 OOF 후보 표가 비어 있거나 필수 지표가 없습니다.")
    
    # 계산에 필요한 최소 열만 수치형으로 검증하고 출처 metadata는 그대로 유지한다.
    result = table.copy()

    try:
        values = result[columns].astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("OOF 후보 지표는 수치형이어야 합니다.") from error
    
    # NaN을 포함하면 비교 결과가 조용히 False가 되어 미충족으로 오해할 수 있다.
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("OOF 후보 지표에 결측 또는 무한대 값이 있습니다.")
    if ((values.recall < 0) | (values.recall > 1)).any():
        raise ValueError("OOF 후보의 Recall은 0~1이어야 합니다.")
    if score_kind == "probability" and ((values.threshold < 0) | (values.threshold > 1)).any():
        raise ValueError("확률용 OOF 후보의 threshold는 0~1이어야 합니다.")
    
    # support=N, positive_support=실제 불량 수이며 TP·FP는 제품 건수다.
    # CSV에서 10.0으로 읽혀도 정수값이면 허용하되 10.5건이나 음수는 차단한다.
    counts = values[["support", "positive_support", "true_positive", "false_positive"]]

    if (counts < 0).any().any() or (counts != np.floor(counts)).any().any():
        raise ValueError("OOF 후보의 표본 수와 오류 건수는 음수가 아닌 정수여야 합니다.")
    
    # 실제 불량이 없으면 Recall 목표의 충족 여부를 판단할 수 없다.
    if (values.support <= 0).any() or (values.positive_support <= 0).any():
        raise ValueError("정책 선택에는 전체 표본과 양성 표본이 하나 이상 필요합니다.")
    
    # TP는 실제 불량 수, FP는 실제 정상 수보다 클 수 없다.
    # 이렇게 분모·분자를 검증해야 재검사 비율이 잘못된 값으로 계산되지 않는다.
    if (values.positive_support > values.support).any() or (
        values.true_positive > values.positive_support).any() or (
        values.false_positive > values.support - values.positive_support).any():
        raise ValueError("OOF 후보의 표본 수와 오류 건수가 일치하지 않습니다.")
    
    # 모델·구간이 다른 후보를 섞으면 평가 대상이 달라 공정한 문턱 선택이 아니다.
    # 호출자는 출처도 따로 확인해야 하며, 여기서는 최소한 표본 수 일치를 검사한다.
    if values.support.nunique() != 1 or values.positive_support.nunique() != 1:
        raise ValueError("서로 다른 평가 대상의 후보 표를 섞어 선택할 수 없습니다.")
    
    # 저장된 Recall이 오류 건수와 일치하는지 확인한다.
    # CSV 부동소수점 표현 차이는 작은 오차만 허용하고 계산 불일치는 차단한다.
    if not np.allclose(values.recall, values.true_positive / values.positive_support,
                       rtol=1e-10, atol=1e-12):
        raise ValueError("OOF 후보의 Recall과 양성 검출 건수가 일치하지 않습니다.")
    
    result[columns] = values

    # 모든 양성 판정 제품을 추가 검사 대상으로 지정한다는 가정이다.
    # (TP+FP)/N은 재검사 비율이고, FP/(FP+TN)인 FPR과 분모가 다르다.
    result["reinspection_ratio"] = (values.true_positive + values.false_positive) / values.support

    # 두 조건을 모두 만족해야 True다. 경계값과 같은 경우도 충족으로 처리한다.
    result["policy_feasible"] = ((values.recall >= policy.min_recall) &
                                 (result.reinspection_ratio <= policy.max_reinspection_ratio))
    
    return result


# ==========================================
# 정책을 충족하는 OOF 후보 중 문턱 하나 선택
# - 우선순위: 재검사 비율 최소 → Recall 최대 → threshold 최대
# - 마지막 threshold 순서로 동률에도 항상 같은 후보를 선택
# - 충족 후보가 없으면 None이며 F1 최대 후보로 자동 대체하지 않음
# - 반환값은 원본 수치이며 화면 표시용 반올림을 적용하지 않음
# ==========================================
def select_policy_threshold(table, policy, *, score_kind="probability"):
    """정책 제약을 충족한 후보 중 우선순위 규칙에 따라 threshold 하나를 고른다.

    우선순위는 재검사 비율 최소화, Recall 최대화, threshold 최대화 순이다.
    조건을 충족하는 후보가 없으면 다른 지표의 최적값으로 대체하지 않고 None을 반환한다.
    """
    # 선택은 OOF 후보만 사용하며 미래 평가 label을 받지 않는다.
    candidates = policy_candidates(table, policy, score_kind=score_kind)

    # 조건을 통과한 후보만 남기며 목표 미달인 인접 후보는 선택하지 않는다.
    feasible = candidates.loc[candidates.policy_feasible]

    if feasible.empty:
        return None
    
    # ascending=True는 최소, False는 최대를 먼저 배치한다.
    # 문턱의 숫자를 단순히 높이는 것이 아니라 검사량·검출률 순서로 선택한다.
    row = feasible.sort_values(["reinspection_ratio", "recall", "threshold"],
                               ascending=[True, False, False]).iloc[0]
    
    return float(row.threshold)


# ==========================================
# 시간 검증에 적용할 threshold 비교표의 출처 확인
# - 동일한 Train 생성 계약·데이터셋·모델의 단일 OOF 후보만 허용
# - 반복 평균 분석 결과와 legacy 출처 불명 결과는 최종 threshold 근거로 사용하지 않음
# ==========================================
def validate_threshold_report(report, train_frame, config, experiment, threshold):
    """사용하려는 threshold가 현재 학습 실행에서 나온 후보인지 확인한다.

    데이터셋, split 계약, 실험 이름, OOF 방식과 수치 범위를 대조한다.
    이를 통해 다른 데이터나 과거 실행의 보고서를 실수로 적용하는 것을 막는다.
    """
    required = {"dataset_id", "experiment", "training_protocol_id", "threshold",
                "oof_score_method", "threshold_use"}
    
    if report.empty or not required.issubset(report.columns):
        raise ValueError("threshold 비교표가 비어 있거나 출처 확인 컬럼이 없습니다.")
    
    if PROTOCOL_ID not in train_frame:
        raise ValueError("threshold 출처 확인에는 원본 split 생성 계약이 필요합니다.")
    
    expected = {"dataset_id": config.dataset.dataset_id, "experiment": experiment,
                "training_protocol_id": train_frame[PROTOCOL_ID].iloc[0],
                "oof_score_method": "single", "threshold_use": "candidate"}
    
    for column, value in expected.items():
        if report[column].isna().any() or set(report[column]) != {value}:
            raise ValueError(f"threshold 비교표의 학습 출처 또는 방식이 일치하지 않습니다: {column}")
        
    try:
        values = report["threshold"].to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("threshold 비교표의 threshold는 수치형이어야 합니다.") from error
    
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("threshold 비교표에는 0~1의 유한한 threshold만 허용됩니다.")
    
    if not np.isclose(values, threshold, rtol=1e-12, atol=0).any():
        raise ValueError("지정한 threshold가 비교표에 없습니다. 반올림하지 않은 값을 사용하세요.")
