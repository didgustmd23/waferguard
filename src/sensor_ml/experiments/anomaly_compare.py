# ==========================================
# 정상만 학습하는 이상 탐지와 지도학습의 시간순 비교
# - 품질 필터·대치·scaling·모델 모두 각 학습 구간의 정상만 fit
# - 과거 내부 시간순 OOF에서 문턱을 선택하고 다음 시간 구간에 적용
# - 원본 이상 점수는 확률이 아니며 미래 데이터로 정규화하지 않음
# - 최종 Test와 외부 Validation을 읽지 않는 Train 내부 진단
#
# 전체 처리 순서:
# - 1. Time Train을 과거 학습 / 다음 시간 평가 구간으로 분리
# - 2. 과거 학습 구간을 다시 나누어 정상 전용 모델의 내부 OOF 생성
# - 3. 내부 OOF에서 F1 진단용 문턱과 평가 정책용 문턱을 각각 선택
# - 4. 과거 학습 구간의 정상 전체로 새 모델을 학습
# - 5. 선택한 문턱을 다음 시간 구간에 적용하고 결과·학습 근거 저장
#
# 용어:
# - outer: 모델이 미래 구간에도 적용되는지 확인하는 외부 시간 fold
# - inner: outer의 과거 학습 구간 안에서 문턱을 고르는 내부 시간 fold
# - fit: 센서 제거 기준·대치값·scaling·모델 파라미터를 학습하는 작업
# - score: 이미 학습한 모델로 각 제품의 이상 정도를 계산하는 작업
# - diagnostic: 정책 충족과 무관하게 비교하는 F1 최대 진단 후보
# - chosen: Recall·재검사 정책을 충족할 때만 선택되는 후보
# ==========================================

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

try:
    from src.modeling_config import DEFAULT_CONFIG_PATH, load_modeling_config
    from src.modeling_metrics import evaluate_anomaly_scores
    from src.modeling_preprocessing import preprocessing_steps, quality_filter_record, quality_filter_json
    from src.dataset_schema import split_frame_to_xy
    from src.split_contract import SOURCE_ROW_ID, PROTOCOL_ID, validate_train_role
    from src.split_contract import temporal_folds
    from src.threshold_policy import policy_candidates, select_policy_threshold
except ModuleNotFoundError:
    from modeling_config import DEFAULT_CONFIG_PATH, load_modeling_config
    from modeling_metrics import evaluate_anomaly_scores
    from modeling_preprocessing import preprocessing_steps, quality_filter_record, quality_filter_json
    from dataset_schema import split_frame_to_xy
    from split_contract import SOURCE_ROW_ID, PROTOCOL_ID, validate_train_role
    from split_contract import temporal_folds
    from threshold_policy import policy_candidates, select_policy_threshold


# 두 모델을 항상 같은 시간 구간에서 평가한다. 이 목록은 실행 대상 이름이다.
DETECTORS = ("isolation_forest", "pca_reconstruction")


# ==========================================
# 정상 행만 사용해 전처리와 모델을 하나의 Pipeline으로 학습
# - 정상·불량 label 값은 Dataset Profile에서 읽음
# - contamination은 목표 Recall·재검사 비율과 독립적인 auto 설정
#
# train:
# - 정상·불량을 포함한 현재 학습 구간이며 내부에서 정상만 추출
# - 불량 행을 전처리 fit에 사용하지 않는 것이 핵심
#
# 반환값:
# - 품질 필터부터 이상 탐지 모델까지 학습된 Pipeline
# - 다음 구간에서는 이 Pipeline을 다시 fit하지 않고 transform만 사용
# ==========================================
def fit_detector(train, config, name, *, n_jobs=1, n_estimators=200):
    # Profile에 맞는 센서·label 구조를 검증한 뒤 모델 입력과 정답을 분리한다.
    features, labels, _ = split_frame_to_xy(train, config.dataset)

    # 정상 기준도 -1로 고정하지 않고 Dataset Profile의 negative_label을 사용한다.
    normal = features.loc[labels.eq(config.dataset.negative_label)]

    # PCA의 분산 계산과 정상 패턴 학습을 위해 정상 표본이 최소 두 개는 필요하다.
    if len(normal) < 2:
        raise ValueError("이상 탐지 학습에는 정상 표본이 두 개 이상 필요합니다.")
    
    if name == "isolation_forest":
        # bool도 Python에서는 int이므로 True를 트리 1개로 해석하지 않도록 차단한다.
        if isinstance(n_estimators, bool) or not isinstance(n_estimators, int) or n_estimators < 1:
            raise ValueError("Isolation Forest 트리 수는 양의 정수여야 합니다.")
        
        # 정상 패턴에서 쉽게 고립되는 관측값에 높은 이상 점수를 주는 트리 모델이다.
        # predict()의 기본 문턱은 사용하지 않고 score_samples의 원본 점수를 평가한다.
        # 따라서 contamination을 재검사 상한 20%와 같은 값으로 맞추지 않는다.
        estimator = IsolationForest(n_estimators=n_estimators, contamination="auto",
                                    random_state=config.experiment.cv.random_state, n_jobs=n_jobs)
        
    elif name == "pca_reconstruction":
        # 0.9는 센서 90% 선택이 아니라 정상 데이터 설명분산 90% 보존을 뜻한다.
        # 실제 성분 수는 정상 표본·센서 수와 상관 구조에 따라 fit마다 달라진다.
        # 분산 비율로 성분 수를 정할 수 있도록 full SVD를 명시한다.
        estimator = PCA(n_components=config.experiment.pca_explained_variance, svd_solver="full",
                        random_state=config.experiment.cv.random_state)
        
    else:
        raise ValueError(f"지원하지 않는 이상 탐지 모델입니다: {name}")
    
    # 공통 전처리: 센서 품질 필터 → 대치 → scaling → 모델.
    # PCA에서 큰 단위의 센서가 복원오차를 독점하지 않도록 scaling을 적용한다.
    # 선언된 범주형 센서는 공통 코어가 대치·One-Hot Encoding으로 처리한다.
    pipeline = Pipeline(preprocessing_steps(config.dataset, scale=True) + [("model", estimator)])

    # 작은 표본의 PCA 과도한 BLAS 병렬화를 줄이고 트리 병렬화는 n_jobs로 관리한다.
    with threadpool_limits(limits=1):
        # normal만 전달하므로 품질 필터·대치값·scaler도 정상 데이터만 학습한다.
        pipeline.fit(normal)

    return pipeline


# ==========================================
# 학습된 정상 패턴으로 제품별 원본 이상 점수 계산
# - Isolation Forest: 낮을수록 이상인 score_samples의 부호를 반전
# - PCA: 표준화된 입력과 복원값의 차이를 센서 전체에서 평균
# - 두 점수 모두 클수록 이상이며 불량 확률은 아님
# - label은 입력 구조 검증에만 사용하고 점수 계산에는 전달하지 않음
# ==========================================
def anomaly_scores(pipeline, frame, dataset, name):
    # 학습된 전처리만 적용한다. 평가 구간의 label은 점수 계산에 사용하지 않는다.
    features, _, _ = split_frame_to_xy(frame, dataset)

    # Pipeline의 마지막 모델을 제외한 전처리 부분만 평가 데이터에 적용한다.
    # transform은 과거 fit의 센서 목록·대치값·scaling을 그대로 사용한다.
    transformed = pipeline[:-1].transform(features)
    estimator = pipeline.named_steps["model"]

    if name == "isolation_forest":
        # 부호만 바꾸므로 순위 정보는 유지되고 클수록 이상이라는 방향을 통일한다.
        scores = -estimator.score_samples(transformed)

    elif name == "pca_reconstruction":
        # transform: 원본 센서 공간 → PCA 성분 공간.
        # inverse_transform: PCA 성분 공간 → 복원된 센서 공간.
        reconstructed = estimator.inverse_transform(estimator.transform(transformed))

        # 행 하나가 제품 하나다. axis=1로 각 제품의 센서별 제곱 오차를 평균한다.
        # 오차는 raw 단위가 아니라 전처리된 센서 공간의 값이다.
        scores = np.mean(np.square(transformed - reconstructed), axis=1)

    else:
        raise ValueError(f"지원하지 않는 이상 탐지 모델입니다: {name}")
    
    # 이후 분위수·AP 계산에 NaN/무한대가 섞이지 않도록 즉시 중단한다.
    if not np.isfinite(scores).all():
        raise ValueError("이상 점수에 결측 또는 무한대가 발생했습니다.")
    
    return scores


# ==========================================
# Profile의 날짜 형식으로 구간의 실제 시작·종료 시각 조회
# - 문자열 사전순이 아니라 파싱된 timestamp의 최소·최대값 사용
# - 정상만 전달하면 정상 전용 fit 기간, 전체 구간이면 구간 기간 반환
# ==========================================
def _period(frame, dataset):
    times = pd.to_datetime(frame[dataset.timestamp_column], format=dataset.timestamp_format)

    return str(times.min()), str(times.max())


# ==========================================
# 학습 하나의 정상 전용 근거와 센서 제거 내역 생성
# - metadata: 모델 이름·outer fold·inner fold 등 실행 위치
# - train_samples는 원래 학습 구간, normal_fit_samples는 실제 fit 건수
# - 정상 원본 ID를 보존해 불량 포함 여부·평가 데이터 중복을 사후 확인
# - PCA가 아닌 모델은 pca_components를 None으로 기록
# ==========================================
def _fit_record(pipeline, train, config, metadata):
    # 실제 fit 대상인 정상 표본 수·ID·기간과 품질 제거 기록을 함께 남긴다.
    normal = train.loc[train[config.dataset.label_column].eq(config.dataset.negative_label)]
    model = pipeline.named_steps["model"]

    # 센서 제거 개수는 fit된 품질 필터에서 읽는다. 전체 데이터의 EDA 값이 아니다.
    # fail_fit_samples=0은 fit_detector가 정상만 전달하는 학습 계약의 기록이다.
    # 실제 포함 행은 normal_fit_source_ids와 함께 확인해야 한다.
    return {**metadata, **quality_filter_record(pipeline), "train_samples": len(train),
            "normal_fit_samples": len(normal), "fail_fit_samples": 0,
            "fit_start": _period(normal, config.dataset)[0], "fit_end": _period(normal, config.dataset)[1],
            "normal_fit_source_ids": quality_filter_json(normal[SOURCE_ROW_ID].tolist()) if SOURCE_ROW_ID in normal else "[]",
            "pca_components": getattr(model, "n_components_", None),
            "model_parameters": quality_filter_json(model.get_params(deep=False))}


# ==========================================
# OOF 또는 다음 구간의 예측 점수를 행 단위 로그로 변환
# - source_row_index: 시간 정렬 전에도 유지한 원본 CSV 행 위치
# - source_row_id: split 생성 시 보존한 원본 ID, 없으면 None
# - label: 모델 학습용이 아니라 이후 평가 지표 계산용 정답
# - 여러 모델의 같은 제품 기록은 metadata로 구분
# ==========================================
def _prediction_rows(frame, config, scores, metadata):
    # 원본 ID로 OOF·외부 구간의 중복 및 시간 누수를 검증할 수 있게 한다.
    return pd.DataFrame({**metadata, "source_row_index": frame.index,
        "source_row_id": frame[SOURCE_ROW_ID].to_numpy() if SOURCE_ROW_ID in frame else None,
        "label": frame[config.dataset.label_column].to_numpy(), "anomaly_score": scores})


# ==========================================
# 내부 OOF 점수로 문턱 후보별 성능·정책 충족 여부 계산
# - 0~100% 분위수를 1% 간격으로 계산해 점수 분포에 맞는 후보 생성
# - 중복 분위수는 제거해 동일한 문턱을 반복 평가하지 않음
# - 최대 점수 바로 위 후보를 추가해 아무것도 선별하지 않는 경우도 포함
# - 이상 점수는 0~1 범위가 아닐 수 있으므로 anomaly 계약으로 평가
# - 다음 시간 구간의 점수·정답은 이 함수에 전달하지 않음
# ==========================================
def _thresholds(oof, config, metadata):

    # 원본 점수 분위수를 사용하며 0.5 확률 기준을 이상 점수에 적용하지 않는다.
    values = oof.anomaly_score.to_numpy()

    # 판정 조건이 score >= threshold이므로 최대값 자체는 최소 한 건을 선별한다.
    # nextafter(max, +inf)는 최대값보다 큰 가장 가까운 부동소수점 수를 만든다.
    candidates = np.unique(np.append(np.quantile(values, np.linspace(0, 1, 101)),
                                     np.nextafter(values.max(), np.inf)))
    
    rows = []
    for threshold in candidates:
        # 같은 OOF 점수에 문턱만 바꾸므로 AP는 같고 Recall·오류 건수는 달라진다.
        metrics = evaluate_anomaly_scores(oof.label, values,
            positive_label=config.dataset.positive_label, negative_label=config.dataset.negative_label,
            threshold=float(threshold))
        
        rows.append({**metadata, **asdict(metrics)})

    # 재검사 비율과 policy_feasible을 추가한다. 이 단계에서 문턱을 선택하지는 않는다.
    return policy_candidates(pd.DataFrame(rows), config.threshold_policy, score_kind="anomaly")


def _inner_anomaly_oof(train, config, name, inner, metadata, *, n_jobs, n_estimators):
    """과거 구간의 정상만 학습해 시간순 내부 OOF와 학습 근거를 생성한다."""
    parts, fit_records = [], []

    for inner_fold, (past_i, future_i) in enumerate(inner, 1):

        past, future = train.iloc[past_i], train.iloc[future_i]

        # 내부 past의 정상만 학습하고 아직 학습하지 않은 내부 future를 점수화한다.
        # 점수 계산 대상은 future의 정상·불량 전체이며 fit에는 사용하지 않는다.
        model = fit_detector(past, config, name, n_jobs=n_jobs, n_estimators=n_estimators)
        scores = anomaly_scores(model, future, config.dataset, name)
        parts.append(_prediction_rows(future, config, scores, {**metadata, "inner_fold": inner_fold}))
        fit_records.append({**_fit_record(model, past, config, {**metadata, "inner_fold": inner_fold}),
                     "evaluation_start": _period(future, config.dataset)[0]})
        
    # 4. 내부 평가 부분만 연결한다. 최초 past는 예측 대상이 아니므로 제외된다.
    # 이 초기 행을 0점으로 채우면 OOF 결과와 threshold가 왜곡된다.
    oof = pd.concat(parts, ignore_index=True)

    if oof.source_row_index.duplicated().any():
        raise RuntimeError("시간순 OOF 행이 중복 예측됐습니다.")
    
    return oof, fit_records


def _evaluate_threshold(evaluation, scores, config, threshold):
    """사전 선택한 이상 점수 문턱으로 미래 구간의 오류 건수를 평가한다."""
    # 미래 정답은 지표 계산에만 사용하며 이 함수에서는 문턱을 탐색하지 않는다.
    metrics = evaluate_anomaly_scores(
        evaluation[config.dataset.label_column], scores,
        positive_label=config.dataset.positive_label,
        negative_label=config.dataset.negative_label, threshold=threshold,
    )
    return {
        **asdict(metrics),
        "reinspection_ratio": (metrics.true_positive + metrics.false_positive) / metrics.support,
    }


def _summarize_anomalies(result):
    """시간 구간별 이상 탐지 결과를 AP 평균과 합산 오류량으로 요약한다."""
    summaries = []

    for name, group in result.groupby("model_name", sort=False):
        # 8. 외부 평가 행은 시간 구간별로 겹치지 않으므로 오류 건수를 합산한다.
        # AP는 구간별 평균이며 합산 Recall·Precision은 건수에서 다시 계산한다.
        # 구간마다 모델·문턱이 달라 하나의 최종 모델 성능으로 해석하면 안 된다.
        tp, fp, fn, tn = (int(group[column].sum()) for column in
                         ["true_positive", "false_positive", "false_negative", "true_negative"])
        summaries.append({"model_name": name, "mode": "temporal_oof", "model_family": "anomaly",
            "ap_mean": group.average_precision.mean(), "ap_std": group.average_precision.std(ddof=0),
            "pooled_recall": tp / (tp + fn), "pooled_precision": tp / (tp + fp) if tp + fp else 0,
            "alarm_ratio": (tp + fp) / (tp + fp + fn + tn),
            "true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn})
        
    return pd.DataFrame(summaries)


# ==========================================
# 지도학습과 같은 세 외부 시간 구간에서 이상 탐지 비교
# - F1 최대값은 진단용이고 정책 미충족 시 정책 기반 평가를 만들지 않음
# - 시간순 OOF 초기 구간 제외 수·모든 학습의 정상 전용 기록 보존
#
# outer_splits / inner_splits:
# - 외부 구간 수와 과거 학습 내부 OOF 구간 수이며 각각 확장형 시간 분할
# - inner에서 threshold를 정하고 outer 평가 구간에서 전이 성능을 확인
#
# 반환값:
# - folds / quality_filter: 시간 구간과 실제 학습 근거
# - oof_predictions / oof_coverage: 문턱 선택 근거 및 초기 제외 행 수
# - threshold_compare / policy_selection: 후보 전체와 정책 선택 결과
# - predictions / fold_results / summary: 다음 구간 점수·지표·진단 요약
# ==========================================
def compare_anomalies(frame, config, *, outer_splits=3, inner_splits=2,
                      n_jobs=1, n_estimators=200):
    # 1. Test/Validation 역할의 입력과 잘못된 데이터 구조를 먼저 차단한다.
    validate_train_role(frame)
    split_frame_to_xy(frame, config.dataset)
    if config.dataset.timestamp_column is None:
        raise ValueError("시간순 이상 탐지 비교에는 timestamp가 필요합니다.")
    times = pd.to_datetime(frame[config.dataset.timestamp_column],
                           format=config.dataset.timestamp_format, errors="coerce")
    if times.isna().any():
        raise ValueError("timestamp 파싱 실패 또는 결측값이 있습니다.")
    
    # 실제 날짜로 정렬하고 같은 timestamp의 원래 순서를 유지한다.
    # reset_index를 하지 않아 source_row_index가 원본 CSV 행 위치를 유지한다.
    frame = frame.iloc[np.argsort(times.to_numpy(), kind="stable")].copy()

    # 결과 종류마다 별도 목록을 사용해 학습 근거·OOF·미래 평가를 섞지 않는다.
    periods, fits, oofs, thresholds, predictions, results, policies, coverage = [], [], [], [], [], [], [], []

    for outer, (fit_i, eval_i) in enumerate(temporal_folds(frame, config.dataset, outer_splits), 1):
        # 2. 외부 구간: 과거 train으로만 모든 선택을 하고 이후 evaluation은 평가용이다.
        # temporal_folds는 동일 timestamp·선언된 그룹·ID가 경계를 넘는지 검사한다.
        train, evaluation = frame.iloc[fit_i], frame.iloc[eval_i]
        periods.append({"outer_fold": outer, "train_samples": len(train),
            "train_fail": int(train[config.dataset.label_column].eq(config.dataset.positive_label).sum()),
            "evaluation_samples": len(evaluation),
            "evaluation_fail": int(evaluation[config.dataset.label_column].eq(config.dataset.positive_label).sum()),
            "train_start": _period(train, config.dataset)[0], "train_end": _period(train, config.dataset)[1],
            "evaluation_start": _period(evaluation, config.dataset)[0], "evaluation_end": _period(evaluation, config.dataset)[1]})
        
        # 3. 문턱 선택용 구간은 외부 evaluation이 아니라 과거 train 안에서 생성한다.
        # 두 이상 탐지 모델은 이 같은 인덱스를 재사용해 공정하게 비교한다.
        inner = temporal_folds(train, config.dataset, inner_splits)

        for name in DETECTORS:

            metadata = {"dataset_id": config.dataset.dataset_id, "outer_fold": outer,
                        "model_name": name, "mode": "temporal_oof", "score_kind": "anomaly"}
            
            # 초기 미예측 구간은 제외한 OOF와 정상 전용 fit 기록을 받는다.
            oof, inner_fits = _inner_anomaly_oof(
                train, config, name, inner, metadata,
                n_jobs=n_jobs, n_estimators=n_estimators,
            )

            fits.extend(inner_fits)
            oofs.append(oof)
            table = _thresholds(oof, config, metadata)
            thresholds.append(table)

            coverage.append({**metadata, "train_samples": len(train), "oof_samples": len(oof),
                             "excluded_initial_samples": len(train) - len(oof),
                             "oof_fail": int(oof.label.eq(config.dataset.positive_label).sum())})
            
            # 5. 진단 후보와 정책 후보는 목적이 다르므로 따로 보관한다.
            # diagnostic: 목표 미충족이어도 모델 비교를 위해 계산하는 F1 최대 후보.
            # chosen: Recall·재검사 상한을 동시에 만족할 때만 값이 있고 아니면 None.
            diagnostic = float(table.loc[table.f1.idxmax(), "threshold"])
            chosen = select_policy_threshold(table, config.threshold_policy, score_kind="anomaly")

            # 문턱을 고른 다음 과거 정상 전체로 학습하고 이후 구간의 점수를 생성한다.
            # 6. 내부 OOF에서 사용한 모델을 재사용하지 않고 과거 정상 전체로 새로 fit한다.
            # fit 시간만 측정하며 점수 계산·내부 OOF 생성 시간은 포함하지 않는다.
            started = perf_counter()
            model = fit_detector(train, config, name, n_jobs=n_jobs, n_estimators=n_estimators)
            elapsed = perf_counter() - started
            scores = anomaly_scores(model, evaluation, config.dataset, name)
            fits.append({**_fit_record(model, train, config, {**metadata, "inner_fold": None}),
                         "evaluation_start": _period(evaluation, config.dataset)[0]})
            predictions.append(_prediction_rows(evaluation, config, scores, metadata))

            # 7. F1 진단 결과는 항상 기록하되 정책 통과 결과와 구분한다.
            results.append({**metadata, "threshold_rule": "oof_max_f1_diagnostic", "fit_seconds": elapsed,
                            **_evaluate_threshold(evaluation, scores, config, diagnostic)})
            policy = {**metadata, **asdict(config.threshold_policy), "threshold": chosen,
                      "policy_status": "feasible" if chosen is not None else "infeasible"}
            
            if chosen is not None:
                # OOF에서 조건을 충족해도 다음 구간에서는 미충족일 수 있으므로 재확인한다.
                policy.update(_evaluate_threshold(evaluation, scores, config, chosen))
                policy["evaluation_meets_policy"] = (policy["recall"] >= config.threshold_policy.min_recall and
                    policy["reinspection_ratio"] <= config.threshold_policy.max_reinspection_ratio)
                
            # chosen=None이면 정책 미충족만 기록하고 미래 정책 지표는 만들지 않는다.
            policies.append(policy)

    result = pd.DataFrame(results)

    return {"folds": pd.DataFrame(periods), "fold_results": result, "summary": _summarize_anomalies(result),
            "predictions": pd.concat(predictions, ignore_index=True),
            "oof_predictions": pd.concat(oofs, ignore_index=True),
            "threshold_compare": pd.concat(thresholds, ignore_index=True),
            "policy_selection": pd.DataFrame(policies), "oof_coverage": pd.DataFrame(coverage),
            "quality_filter": pd.DataFrame(fits)}


# ==========================================
# 이미 실행한 지도학습 로그와 이상 탐지의 같은 구간 결과 연결
# - 모델 재학습 없이 기존 temporal_run.json·CSV를 조회
# - 정책값은 F1 진단 결과를 바꾸지 않으므로 설정 비교에서 제외
# - 나머지 학습 설정·기간·표본 수가 다르면 비교를 중단
# - 내부 OOF 방식이 동일한 temporal_oof 행만 선택
# ==========================================
def supervised_comparison(results, directory, config):

    # 생성 계약·학습 설정·시간 경계가 맞는 과거 지도학습 진단만 비교에 사용한다.
    record = json.loads((directory / "temporal_run.json").read_text(encoding="utf-8"))

    # 현재 Path 객체도 문자열로 바꿔 JSON에 저장된 과거 설정과 같은 형식으로 비교한다.
    current = json.loads(json.dumps(asdict(config), default=str))
    original = dict(record["config"])

    # 여기서는 정책 threshold가 아니라 OOF 최대 F1 진단 결과끼리 비교한다.
    current.pop("threshold_policy", None)
    original.pop("threshold_policy", None)

    if current != original:
        raise ValueError("지도학습 비교 로그의 학습 설정이 다릅니다. 같은 조건으로 다시 생성하세요.")
    
    # 평균 AP만 비슷한 로그를 연결하지 않도록 실제 평가 구간 자체를 확인한다.
    periods = pd.read_csv(directory / "folds.csv")

    try:
        pd.testing.assert_frame_equal(results["folds"], periods, check_dtype=False)
    except AssertionError as error:
        raise ValueError("지도학습과 이상 탐지의 시간 구간·표본 수가 일치하지 않습니다.") from error
    
    previous = pd.read_csv(directory / "fold_results.csv")
    previous = previous.loc[previous["mode"].eq("temporal_oof")].copy()

    if previous.empty:
        raise ValueError("지도학습 비교 로그에 시간순 OOF 평가가 없습니다.")
    
    # 노트북에서 학습 방식별로 구분할 수 있게 접근 종류를 명시한다.
    previous["model_family"] = "supervised"
    anomaly = results["fold_results"].assign(model_family="anomaly")

    return pd.concat([previous, anomaly], ignore_index=True).drop(columns="quality_filter_log", errors="ignore")


# ==========================================
# CLI 입력 → 정상 전용 비교 → CSV·실행 조건 저장
# - --train만 받아 외부 Validation·최종 Test의 평가를 제공하지 않음
# - --supervised-dir는 기존 지도학습 비교 결과를 연결할 때만 사용
# - 이미 결과가 있는 폴더는 덮어쓰지 않아 과거 실험을 보존
# ==========================================
def parse_args(argv=None):
    """실험 옵션과 기존 출력 경로 조건을 확인하고 실행 인자를 반환한다."""
    parser = argparse.ArgumentParser(description="정상 전용 Isolation Forest·PCA 시간순 비교")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--supervised-dir", type=Path)
    parser.add_argument("--outer-splits", type=int, default=3)
    parser.add_argument("--inner-splits", type=int, default=2)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args(argv)

    # 학습을 시작하기 전에 기존 산출물을 덮어쓰는 실행인지 확인한다.
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("결과 폴더가 비어 있지 않습니다. 새 폴더를 지정하세요.")
    
    return args


def save_results(args, config, train, results):
    """계산된 결과와 실행 설정을 기존 파일 이름·형식으로 저장한다."""
    # 저장 단계에서는 모델을 학습하거나 문턱을 다시 선택하지 않는다.
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name, table in results.items():
        if name == "quality_filter":
            # 센서 이름 목록은 CSV 셀 하나에 들어가도록 JSON 문자열로 변환한다.
            # copy를 사용해 계산 단계에서 만든 DataFrame 원본은 변경하지 않는다.
            table = table.copy()

            for column in ("high_missing_features", "constant_features"):
                table[column] = table[column].map(quality_filter_json)

        table.to_csv(args.output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    # CSV 수치와 함께 입력·정책·모델 옵션·환경·점수 정의를 보존한다.
    # 점수 정의가 달라지면 threshold 숫자의 의미도 달라지므로 함께 기록한다.
    record = {"config": asdict(config), "train_path": str(args.train.resolve()),
              "supervised_dir": str(args.supervised_dir.resolve()) if args.supervised_dir else None,
              "outer_splits": args.outer_splits, "inner_splits": args.inner_splits,
              "n_estimators": args.n_estimators, "n_jobs": args.n_jobs,
              "training_protocol_id": train[PROTOCOL_ID].iloc[0] if PROTOCOL_ID in train else None,
              "python": sys.version.split()[0], "sklearn": sklearn.__version__,
              "score_definition": {"isolation_forest": "negative_score_samples", "pca_reconstruction": "standardized_reconstruction_mse"}}
    
    (args.output_dir / "anomaly_run.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def main():
    """인자 검증 → 기존 실험 코어 실행 → 결과 저장 → 콘솔 요약을 수행한다."""
    args = parse_args()
    config = load_modeling_config(args.config)
    train = pd.read_csv(args.train)

    # 결과 계산 중에는 CSV를 쓰지 않는다. 비교 검증까지 통과한 뒤 저장한다.
    results = compare_anomalies(train, config, outer_splits=args.outer_splits,
                               inner_splits=args.inner_splits, n_jobs=args.n_jobs, n_estimators=args.n_estimators)
    
    if args.supervised_dir:
        # 생성 계약과 내부 fold 수는 기간 비교만으로 확인할 수 없으므로 별도 검사한다.
        metadata = json.loads((args.supervised_dir / "temporal_run.json").read_text(encoding="utf-8"))
        
        if (metadata.get("outer_splits") != args.outer_splits or
            metadata.get("inner_temporal_splits") != args.inner_splits):
            raise ValueError("지도학습과 이상 탐지의 시간 fold 설정이 다릅니다.")
        
        if PROTOCOL_ID in train and metadata.get("training_protocol_id") != train[PROTOCOL_ID].iloc[0]:
            raise ValueError("지도학습 비교 로그의 Train 생성 계약이 다릅니다.")
        
        results["anomaly_compare"] = supervised_comparison(results, args.supervised_dir, config)

    # 결과 dict의 각 항목을 같은 이름의 CSV로 저장해 노트북과 연결한다.
    save_results(args, config, train, results)
    
    # 콘솔에는 요약만 표시하고 상세 근거는 생성된 CSV에서 확인한다.
    print(results["summary"].to_string(index=False))
    print(results["policy_selection"][["outer_fold", "model_name", "policy_status", "threshold"]].to_string(index=False))
    print(f"결과 저장 경로: {args.output_dir}")


if __name__ == "__main__":
    main()
