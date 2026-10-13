# ==========================================
# Train 내부 시간순 검증의 전체 과거·최근 기간 학습 비교
# - 동일한 미래 평가 구간에 확장형·최근 30일·최근 60일 학습 적용
# - 전체 센서와 S0 RF Top-K 모두 동일한 M3 분류기 설정 사용
# - 각 학습 구간 안에서 품질 필터·대치·센서 선택을 새로 fit
# - AP·ROC-AUC를 우선 비교하고 기본 문턱은 보조 진단에만 사용
# - 외부 Validation·Test와 전체 Train OOF 문턱은 사용하지 않음
# ==========================================

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

try:
    from src.modeling_config import DEFAULT_CONFIG_PATH, load_modeling_config
    from src.modeling_models import build_pipeline, configure_topk_xgb, fit_pipeline, positive_scores
    from src.modeling_metrics import evaluate_binary_scores
    from src.modeling_preprocessing import quality_filter_record, quality_filter_json, fitted_feature_count
    from src.dataset_schema import split_frame_to_xy
    from src.split_contract import SOURCE_ROW_ID, PROTOCOL_ID, validate_train_role, temporal_folds
except ModuleNotFoundError:
    from modeling_config import DEFAULT_CONFIG_PATH, load_modeling_config
    from modeling_models import build_pipeline, configure_topk_xgb, fit_pipeline, positive_scores
    from modeling_metrics import evaluate_binary_scores
    from modeling_preprocessing import quality_filter_record, quality_filter_json, fitted_feature_count
    from dataset_schema import split_frame_to_xy
    from split_contract import SOURCE_ROW_ID, PROTOCOL_ID, validate_train_role, temporal_folds


def recent_training_rows(frame, indices, dataset, days):
    """과거 학습 구간의 마지막 시각을 기준으로 최근 기간 행만 선택한다."""
    # 0은 전체 과거 구간이며 양수는 마지막 관측 시각을 포함한 달력 일수다.
    if isinstance(days, bool) or not isinstance(days, int) or days < 0:
        raise ValueError("학습 기간은 0 이상의 정수 일수여야 합니다. 0은 전체 과거입니다.")
    
    past = frame.iloc[indices]

    if days == 0:
        return past.copy()
    
    times = pd.to_datetime(past[dataset.timestamp_column], format=dataset.timestamp_format)
    cutoff = times.max() - pd.Timedelta(days=days)

    # 동일 timestamp의 행은 함께 유지하며 평가 구간의 시각은 참조하지 않는다.
    return past.loc[times >= cutoff].copy()


def compare_recent_windows(frame, config, *, windows=(0, 30, 60), n_splits=3, n_jobs=1, n_estimators=300,
                           fixed_sensor_control=False):
    """같은 시간 fold에서 학습 기간과 특징 축소 효과를 비교한다."""
    validate_train_role(frame)

    if not windows or len(set(windows)) != len(windows):
        raise ValueError("학습 기간은 중복 없이 하나 이상 지정하세요.")
    
    if any(isinstance(days, bool) or not isinstance(days, int) or days < 0 for days in windows):
        raise ValueError("학습 기간은 0 이상의 정수 일수여야 합니다.")
    
    if config.reduction_policy is None:
        raise ValueError("센서 수 비교에는 reduction_policy 설정이 필요합니다.")
    
    if fixed_sensor_control and (0 not in windows or config.dataset.categorical_feature_columns):
        raise ValueError("고정 센서 대조 실험에는 전체 과거(0일)와 수치형 센서 Profile이 필요합니다.")
    
    if isinstance(n_estimators, bool) or not isinstance(n_estimators, int) or n_estimators < 1:
        raise ValueError("트리 수는 양의 정수여야 합니다.")
    
    count = config.reduction_policy.max_sensor_count
    top_name = f"xgboost_rf_top_{count}"
    
    top = build_pipeline(config, top_name, n_jobs=n_jobs)
    configure_topk_xgb({top_name: top}, max_depth=2, reg_lambda=5)
    top.set_params(model__n_estimators=n_estimators, selector__estimator__n_estimators=n_estimators)

    whole = build_pipeline(config, "xgboost_all", n_jobs=n_jobs)
    whole.named_steps["model"].set_params(**top.named_steps["model"].get_params())

    folds = temporal_folds(frame, config.dataset, n_splits)

    # 모든 기간에 같은 미래 행을 사용하며 과거 학습 행만 줄인다.
    results, predictions, selections, quality = [], [], [], []

    for fold, (fit_indices, evaluation_indices) in enumerate(folds, 1):
        evaluation = frame.iloc[evaluation_indices]
        reference, fixed_names = None, None

        if fixed_sensor_control:
            # 전체 Train이 아니라 이 fold의 과거 학습 부분에서만 센서를 정한다.
            past_x, past_y, _ = split_frame_to_xy(frame.iloc[fit_indices], config.dataset)
            reference = fit_pipeline(top, past_x, past_y, config, "s0_m3_topk", n_jobs=n_jobs)
            fixed_names = reference[:-1].get_feature_names_out().tolist()

            # 품질 제거와 RF 선택은 과거 기준으로 고정한다. 대치 통계는 각 기간에서 새로 학습한다.
            fixed = Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True).set_output(transform="pandas")),
                              ("model", clone(top.named_steps["model"]))])
            
        candidates = (("s0_m3_topk", top), ("s0_m3_fixed_sensors", fixed)) if fixed_sensor_control else (
            ("m3_all", whole), ("s0_m3_topk", top))
        
        for days in windows:
            train = recent_training_rows(frame, fit_indices, config.dataset, days)
            train_x, train_y, features = split_frame_to_xy(train, config.dataset)
            eval_x, eval_y, _ = split_frame_to_xy(evaluation, config.dataset, expected_feature_columns=features)
            times = pd.to_datetime(train[config.dataset.timestamp_column], format=config.dataset.timestamp_format)
            eval_times = pd.to_datetime(evaluation[config.dataset.timestamp_column], format=config.dataset.timestamp_format)

            for name, template in candidates:
                metadata = {"outer_fold": fold, "window_days": days, "model_name": name,
                            "train_samples": len(train), "train_fail": int(train_y.eq(config.dataset.positive_label).sum()),
                            "evaluation_samples": len(evaluation), "evaluation_fail": int(eval_y.eq(config.dataset.positive_label).sum()),
                            "train_start": str(times.min()), "train_end": str(times.max()),
                            "evaluation_start": str(eval_times.min()), "evaluation_end": str(eval_times.max()),
                            "window_truncated": len(train) < len(fit_indices)}
                
                metadata["sensor_selection_scope"] = "outer_full_past" if name == "s0_m3_fixed_sensors" else "current_training_window"
                metadata["selection_train_samples"] = len(fit_indices) if name == "s0_m3_fixed_sensors" else len(train)

                if train_y.nunique() < 2:
                    results.append({**metadata, "status": "skipped", "reason": "학습 구간에 두 클래스가 없습니다."})
                    continue

                # 고정 경로는 같은 센서 순서를 유지하고 미래 구간의 관측값으로 통계를 fit하지 않는다.
                fit_x = train_x.loc[:, fixed_names] if name == "s0_m3_fixed_sensors" else train_x
                predict_x = eval_x.loc[:, fixed_names] if name == "s0_m3_fixed_sensors" else eval_x

                model = reference if fixed_sensor_control and name == "s0_m3_topk" and days == 0 else fit_pipeline(
                    template, fit_x, train_y, config, name, n_jobs=n_jobs)
                
                scores = positive_scores(model, predict_x, config.dataset)

                # 전체 Train OOF 문턱은 내부 평가 행을 이미 보았으므로 사용하지 않는다.
                metrics = evaluate_binary_scores(eval_y, scores, positive_label=config.dataset.positive_label,
                                                  negative_label=config.dataset.negative_label,
                                                  threshold=config.experiment.default_threshold)
                
                results.append({**metadata, "status": "evaluated", "selected_feature_count": fitted_feature_count(model),
                                "empty_input_sensor_count": int(fit_x.isna().all().sum()),
                                **asdict(metrics), "reinspection_ratio": (metrics.true_positive + metrics.false_positive) / metrics.support})
                
                # 고정 경로의 품질 기준은 전체 과거 fit에서 정했다. 최근 기간 제거로 센서 수를 바꾸지 않는다.
                quality.append({**metadata, **quality_filter_record(reference if name == "s0_m3_fixed_sensors" else model),
                                "quality_fit_scope": "outer_full_past" if name == "s0_m3_fixed_sensors" else "current_training_window"})
                
                names = model[:-1].get_feature_names_out()

                selections.extend({**metadata, "feature": str(feature)} for feature in names)
                predictions.extend({"outer_fold": fold, "window_days": days, "model_name": name,
                                    "source_row_index": index, "source_row_id": evaluation[SOURCE_ROW_ID].iloc[position]
                                    if SOURCE_ROW_ID in evaluation else None,
                                    "label": eval_y.iloc[position], "positive_score": float(score)}
                                   for position, (index, score) in enumerate(zip(evaluation.index, scores)))
                
    return {"fold_results": pd.DataFrame(results), "predictions": pd.DataFrame(predictions),
            "selected_features": pd.DataFrame(selections), "quality_filter": pd.DataFrame(quality)}


# ==========================================
# 저장된 시간 fold 결과를 전체 과거 기준과 짝지어 분석
# - 모델을 재학습하지 않고 AP 차이와 학습 표본 감소를 확인
# - 평가 기간·표본 수가 같을 때만 같은 fold의 차이를 계산
# - 미평가 구간과 AP=0의 상대 변화는 결측값으로 남김
# ==========================================
def analyze_window_folds(rows):
    """각 기간을 같은 모델·시간 fold의 전체 과거 결과와 대조한다."""
    keys = ["model_name", "outer_fold"]
    required = {*keys, "window_days", "status", "train_samples", "train_fail", "train_start", "train_end",
                "evaluation_samples", "evaluation_fail", "evaluation_start", "evaluation_end"}
    
    if rows.empty or not required.issubset(rows.columns):
        raise ValueError("구간별 분석에 필요한 fold 결과 열이 없거나 표가 비어 있습니다.")
    
    if rows.duplicated(keys + ["window_days"]).any():
        raise ValueError("같은 모델·fold·기간의 결과가 중복됩니다.")
    
    baseline = rows.loc[rows.window_days.eq(0)].copy()

    if baseline.empty:
        raise ValueError("전체 과거 기간(window_days=0)의 기준 결과가 필요합니다.")
    
    # 모두 미평가였다면 지표 열이 없을 수 있으므로 계산 불가로 보존한다.
    prepared = rows.copy()

    for column in ("average_precision", "roc_auc"):
        if column not in prepared:
            prepared[column] = np.nan
            
    baseline = prepared.loc[prepared.window_days.eq(0)].rename(
        columns={column: f"baseline_{column}" for column in prepared.columns if column not in keys})
    
    paired = prepared.loc[prepared.window_days.ne(0)].merge(baseline, on=keys, how="left", validate="many_to_one")

    if paired.empty or paired.baseline_window_days.isna().any():
        raise ValueError("기간 후보가 없거나 같은 모델·fold의 전체 과거 기준 결과가 없습니다.")
    
    for column in ("evaluation_samples", "evaluation_fail", "evaluation_start", "evaluation_end"):
        if not paired[column].eq(paired[f"baseline_{column}"]).all():
            raise ValueError("비교 대상의 미래 평가 기간·표본 수가 기준 결과와 다릅니다.")
        
    paired["same_training_period"] = (paired.train_samples.eq(paired.baseline_train_samples)
                                      & paired.train_start.eq(paired.baseline_train_start)
                                      & paired.train_end.eq(paired.baseline_train_end))
    
    paired["removed_train_samples"] = paired.baseline_train_samples - paired.train_samples
    paired["removed_train_fail"] = paired.baseline_train_fail - paired.train_fail
    paired["train_fail_ratio"] = paired.train_fail / paired.train_samples
    paired["baseline_train_fail_ratio"] = paired.baseline_train_fail / paired.baseline_train_samples
    comparable = paired.status.eq("evaluated") & paired.baseline_status.eq("evaluated")
    paired["ap_delta"] = (paired.average_precision - paired.baseline_average_precision).where(comparable)
    paired["ap_relative_change"] = (paired.ap_delta / paired.baseline_average_precision.where(
        paired.baseline_average_precision.gt(0))).where(comparable)
    paired["roc_auc_delta"] = (paired.roc_auc - paired.baseline_roc_auc).where(comparable)

    return paired.sort_values(["model_name", "window_days", "outer_fold"])


def compare_window_sensors(selected, paired):
    """저장된 선택 목록의 교집합과 추가·제외 센서를 같은 시간 fold끼리 비교한다."""
    keys = ["model_name", "outer_fold", "window_days"]

    if not {*keys, "feature"}.issubset(selected.columns) or selected.feature.isna().any():
        raise ValueError("선택 센서 로그에 필수 열 또는 센서 이름이 없습니다.")
    
    if selected.duplicated(keys + ["feature"]).any():
        raise ValueError("같은 모델·fold·기간의 선택 센서가 중복됩니다.")
    
    summaries, changes = [], []

    for _, row in paired.iterrows():
        metadata = {key: row[key] for key in keys}

        if row.status != "evaluated" or row.baseline_status != "evaluated":
            summaries.append({**metadata, "status": "skipped", "ap_delta": row.ap_delta})
            continue

        group = selected.loc[selected.model_name.eq(row.model_name) & selected.outer_fold.eq(row.outer_fold)]
        baseline = set(group.loc[group.window_days.eq(0), "feature"])
        recent = set(group.loc[group.window_days.eq(row.window_days), "feature"])

        if (not baseline or not recent or len(baseline) != row.baseline_selected_feature_count
                or len(recent) != row.selected_feature_count):
            raise ValueError("선택 센서 목록이 누락됐거나 fold 결과의 센서 수와 다릅니다.")
        
        common, removed, added = baseline & recent, baseline - recent, recent - baseline

        summaries.append({**metadata, "status": "compared", "baseline_sensor_count": len(baseline),
                          "recent_sensor_count": len(recent), "common_count": len(common),
                          "removed_count": len(removed), "added_count": len(added),
                          "jaccard": len(common) / len(baseline | recent),
                          "same_sensor_set": baseline == recent, "ap_delta": row.ap_delta,
                          "removed_train_fail": row.removed_train_fail})
        
        for status, names in (("common", common), ("removed", removed), ("added", added)):
            changes.extend({**metadata, "sensor_status": status, "feature": name} for name in sorted(names))

    return pd.DataFrame(summaries), pd.DataFrame(changes)


def analyze_saved_windows(run_dir, output_dir, *, sensor_overlap=False):
    """기존 실행의 구간별 표를 읽어 새 진단 파일만 생성한다."""
    record = json.loads((run_dir / "execution.json").read_text(encoding="utf-8"))

    if record.get("external_validation_used") is not False or record.get("test_used") is not False:
        raise ValueError("이 분석에는 Train 내부 시간 검증 실행만 사용할 수 있습니다.")
    
    paired = analyze_window_folds(pd.read_csv(run_dir / "fold_results.csv"))
    sensor_summary, sensor_changes = compare_window_sensors(
        pd.read_csv(run_dir / "selected_features.csv"), paired) if sensor_overlap else (None, None)
    
    output_dir.mkdir(parents=True, exist_ok=True)

    paired.to_csv(output_dir / "window_fold_diagnostics.csv", index=False, encoding="utf-8-sig")

    columns = ["model_name", "window_days", "outer_fold", "train_samples", "train_fail",
               "removed_train_samples", "removed_train_fail", "baseline_average_precision",
               "average_precision", "ap_delta", "ap_relative_change", "same_training_period"]
    
    text = ["# 최근 기간 학습의 구간별 변화 분석", "",
            "AP 차이는 최근 기간−전체 과거이며, 음수는 최근 기간에서 저하됐다는 뜻입니다.", "",
            "```text", paired[columns].to_string(index=False), "```", "",
            "## 해석 주의", "",
            "- removed_train_fail은 최근 기간 선택으로 학습에서 제외된 실제 불량 표본 수입니다. 평가에서 놓친 불량(FN)이 아닙니다.",
            "- same_training_period는 기록된 학습 시작·종료·행 수가 같은지 확인합니다. 원본 값의 해시 비교는 아닙니다.",
            "- AP 상대 변화는 기준 AP 대비 비율입니다. 기준 AP가 0이거나 미평가인 경우 계산하지 않습니다.",
            "- 미평가 구간은 성능 0으로 대체하지 않습니다. 원본 status와 reason도 CSV에서 확인합니다.",
            "- 학습 불량 표본 감소와 성능 저하가 함께 나타나도 인과관계를 입증하지는 않습니다.",
            "- 외부 Validation·최종 Test를 읽거나 모델·threshold를 변경하지 않았습니다."]
    
    (output_dir / "window_fold_diagnostics.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    (output_dir / "source.json").write_text(json.dumps(
        {"source_run": str(run_dir.resolve()), "execution": record, "model_retrained": False},
        ensure_ascii=False, indent=2), encoding="utf-8")
    
    print(paired[columns].to_string(index=False))

    if sensor_summary is not None:
        sensor_summary.to_csv(output_dir / "sensor_overlap.csv", index=False, encoding="utf-8-sig")
        sensor_changes.to_csv(output_dir / "sensor_changes.csv", index=False, encoding="utf-8-sig")

        note = ["# 학습 기간별 선택 센서 교집합", "", "```text", sensor_summary.to_string(index=False), "```", "",
                "- common은 두 학습 기간의 공통 센서, removed는 전체 과거에만 있는 센서, added는 최근 기간에만 있는 센서입니다.",
                "- Jaccard는 교집합/합집합입니다. 센서 중요도 순위나 모델 가중치의 일치 여부는 나타내지 않습니다.",
                "- s0_m3_topk는 RF 선택 목록을 비교합니다. m3_all은 품질 필터를 통과한 전체 특징 목록의 차이입니다.",
                "- 목록이 달라져도 구현 오류라고 단정하지 않습니다. 각 학습 구간에서 선택기를 다시 학습한 결과입니다.",
                "- AP 차이와 센서 교체가 함께 나타나도 교체 센서가 저하 원인이라고 확정할 수 없습니다.",
                "- 동일 센서로도 학습 표본·대치 통계·분류기가 달라질 수 있습니다. 새 실험이나 센서 제거를 수행하지 않았습니다."]
        
        (output_dir / "sensor_overlap.md").write_text("\n".join(note) + "\n", encoding="utf-8")
        print("\n학습 기간별 센서 교집합:")
        print(sensor_summary.to_string(index=False))

    print(f"구간별 분석 저장 경로: {output_dir}")


def parse_args(argv=None):
    """실험 옵션과 기존 출력 경로 조건을 확인하고 실행 인자를 반환한다."""
    parser = argparse.ArgumentParser(description="M3 전체·Top-K의 Train 내부 최근 기간 학습 비교")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--analyze-only", action="store_true", help="기존 fold 결과만 분석하며 학습하지 않습니다.")
    parser.add_argument("--run-dir", type=Path, help="분석할 최근 기간 실험의 결과 폴더입니다.")
    parser.add_argument("--sensor-overlap", action="store_true", help="분석 전용 모드에서 선택 센서 교집합도 확인합니다.")
    parser.add_argument("--fixed-sensor-control", action="store_true", help="fold별 전체 과거의 센서를 고정한 대조 경로를 비교합니다.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--windows", type=int, nargs="+", default=[0, 30, 60])
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args(argv)

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("결과 폴더가 비어 있지 않습니다. 새 폴더를 지정하세요.")
    
    return args


def save_results(args, config, frame, tables):
    """계산된 결과와 실행 설정을 기존 파일 이름·형식으로 저장한다."""
    # 저장 단계에서는 모델을 학습하거나 문턱을 다시 선택하지 않는다.
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name, table in tables.items():
        stored = table.copy()

        if name == "quality_filter":

            for column in ("high_missing_features", "constant_features"):

                if column in stored:
                    stored[column] = stored[column].map(quality_filter_json)

        stored.to_csv(args.output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    rows = tables["fold_results"]
    evaluated = rows.loc[rows.status.eq("evaluated")]

    summary = evaluated.groupby(["model_name", "window_days"], as_index=False).agg(
        evaluated_folds=("outer_fold", "count"), ap_mean=("average_precision", "mean"),
        ap_std=("average_precision", lambda values: values.std(ddof=0)), roc_auc_mean=("roc_auc", "mean"),
        train_samples_mean=("train_samples", "mean")) if not evaluated.empty else pd.DataFrame()
    
    summary.to_csv(args.output_dir / "summary.csv", index=False, encoding="utf-8-sig")

    execution = {"config": asdict(config), "train_path": str(args.train.resolve()), "windows": args.windows,
                 "n_splits": args.n_splits, "n_jobs": args.n_jobs, "model_variant": "M3",
                 "max_depth": 2, "reg_lambda": 5, "n_estimators": 300,
                 "rf_selector": "S0", "threshold_source": "configured_comparison_default",
                 "training_protocol_id": frame[PROTOCOL_ID].iloc[0] if PROTOCOL_ID in frame else None,
                 "external_validation_used": False, "test_used": False}
    
    execution["fixed_sensor_control"] = args.fixed_sensor_control
    (args.output_dir / "execution.json").write_text(json.dumps(execution, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    report = ["# Train 내부 최근 기간 학습 비교", "", "0일은 전체 과거 구간, 양수는 각 학습 구간의 마지막 시각 기준 최근 일수입니다.", "",
              "```text", summary.to_string(index=False), "```", "",
              "- fold별 AP와 실제 학습 표본 수를 함께 확인합니다. 최근 기간은 시간 변화 대응과 표본 감소를 동시에 일으킵니다.",
              "- 3개 시간 fold는 독립 반복 실험이 아닙니다. 평균뿐 아니라 fold_results.csv의 구간별 저하도 확인합니다.",
              "- window_truncated=False이면 해당 fold에서 전체 과거와 학습 행이 같으므로 별도 개선 증거가 아닙니다.",
              "- 미평가 fold가 있으면 다른 기간과 평균만 직접 비교하지 않고 동일 평가 fold끼리 확인합니다.",
              "- AP·ROC-AUC가 우선 지표입니다. 기본 문턱의 Recall은 보조 진단이며 정책 threshold를 결정하지 않았습니다.",
              "- 외부 Validation·최종 Test를 사용하지 않았습니다. 후보 채택은 사용자 검토 후 결정합니다."]
    
    if args.fixed_sensor_control:
        report.extend(["", "## 고정 센서 대조 경로", "",
                       "- s0_m3_topk는 각 기간에서 센서를 재선택합니다. s0_m3_fixed_sensors는 각 fold의 전체 과거에서 선택한 센서를 고정합니다.",
                       "- 고정 센서와 품질 기준은 최근 기간 밖의 과거 정보도 사용합니다. 순수한 최근 데이터만 사용하는 운영 전략이 아니라 효과 분리를 위한 대조 실험입니다.",
                       "- 고정 경로의 median과 분류기는 해당 학습 기간에서 새로 fit합니다. 최근 기간에서 센서를 다시 제거하지 않습니다.",
                       "- 고정 센서가 최근 기간에 전부 결측이면 keep_empty_features=True에 따라 0으로 대치됩니다. empty_input_sensor_count를 확인하고 그 경우 단순한 센서 선택 효과로 해석하지 않습니다.",
                       "- 0일의 두 경로는 같은 확률이 나오는지 predictions.csv로 대조할 수 있습니다. 센서 고정 효과와 분류기 학습 효과를 구분하는 기준점입니다."])
        
    (args.output_dir / "comparison.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    return summary


def main():
    """인자 검증 → 기존 실험 코어 실행 → 결과 저장 → 콘솔 요약을 수행한다."""
    args = parse_args()
    if args.analyze_only:
        if args.run_dir is None or args.train is not None or args.fixed_sensor_control:
            raise ValueError("분석 전용 모드에는 --run-dir만 지정하고 --train은 지정하지 마세요.")
        
        analyze_saved_windows(args.run_dir, args.output_dir, sensor_overlap=args.sensor_overlap)
        return
    
    if args.train is None or args.run_dir is not None or args.sensor_overlap:
        raise ValueError("학습 모드에는 --train을 지정하고 --run-dir은 지정하지 마세요.")
    
    config = load_modeling_config(args.config)
    frame = pd.read_csv(args.train)

    tables = compare_recent_windows(frame, config, windows=tuple(args.windows), n_splits=args.n_splits, n_jobs=args.n_jobs,
                                    fixed_sensor_control=args.fixed_sensor_control)
    
    summary = save_results(args, config, frame, tables)
    print(summary.to_string(index=False))
    print(f"최근 기간 학습 비교 저장 경로: {args.output_dir}")


if __name__ == "__main__":
    main()
