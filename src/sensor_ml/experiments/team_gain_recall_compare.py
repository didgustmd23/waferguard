# ==========================================
# 기존 시간순 로그의 Recall 목표 80%·90% 문턱 비교
# - 재학습 없이 각 외부 fold의 과거 OOF 표에서만 문턱 선택
# - 목표 이상 후보 중 선별 비율 최소, 정밀도·문턱 순으로 동점 처리
# - 이후 평가 구간 label은 선택 완료 후 지표 계산에만 사용
# - 저장된 V2 모델·문턱·Test·원본 로그는 변경하지 않음
# ==========================================
import argparse
import json
from pathlib import Path

import pandas as pd

from src.modeling_metrics import evaluate_binary_scores
from src.sensor_ml.experiments.temporal_validation import _summarize_results


def compare_recall_targets(thresholds, predictions, *, positive_label, negative_label,
                           targets=(0.8, 0.9)):
    """과거 OOF에서 고른 문턱을 다음 구간에 적용하고 fold별 결과를 반환한다."""
    tables = thresholds.loc[thresholds["mode"].eq("temporal_oof")]

    # default와 temporal_oof는 같은 모델 확률이므로 default 행만 사용해 중복을 막는다.
    future = predictions.loc[predictions["mode"].eq("default")]

    if tables.empty or future.empty:
        raise ValueError("시간순 OOF 문턱표와 기본 문턱 예측 기록이 필요합니다.")
    
    keys = ["model_name", "outer_fold"]

    if set(map(tuple, tables[keys].to_numpy())) != set(map(tuple, future[keys].to_numpy())):
        raise ValueError("OOF와 이후 예측의 모델·fold 구성이 다릅니다.")
    
    rows = []

    for (model, fold), table in tables.groupby(keys, sort=False):
        evaluation = future.loc[future.model_name.eq(model) & future.outer_fold.eq(fold)]

        if evaluation.source_row_id.isna().any() or evaluation.source_row_id.duplicated().any():
            raise ValueError("이후 구간의 원본 행 ID에 결측 또는 중복이 있습니다.")
        
        for target in targets:
            # 이후 평가 정답은 이 문턱 선택에 전달하지 않는다.
            candidates = table.loc[table.recall.ge(target)].copy()

            if candidates.empty:
                raise ValueError(f"OOF Recall 목표 후보가 없습니다: {model}, fold {fold}, {target}")
            
            candidates["alarm_ratio"] = (candidates.true_positive + candidates.false_positive) / candidates.support
            best = candidates.sort_values(["alarm_ratio", "precision", "threshold"],
                                          ascending=[True, False, False]).iloc[0]
            
            threshold = float(best.threshold)
            metrics = evaluate_binary_scores(evaluation.label, evaluation.positive_score,
                        positive_label=positive_label, negative_label=negative_label, threshold=threshold).to_dict()
            
            rows.append({"model_name": model, "outer_fold": int(fold),
                         "mode": f"recall{int(target * 100)}", "target_recall": target,
                         "oof_recall": float(best.recall), "oof_alarm_ratio": float(best.alarm_ratio),
                         "oof_support": int(best.support), **metrics,
                         "evaluation_target_met": metrics["recall"] >= target})
            
    return pd.DataFrame(rows)


def main():
    """기존 로그만 읽고 새 결과 폴더에 문턱·구간·합산 비교를 저장한다."""
    parser = argparse.ArgumentParser(description="시간순 로그 재사용: OOF Recall 80%·90% 비교")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        raise ValueError("비어 있는 새 출력 폴더를 지정하세요.")
    
    execution = json.loads((args.source_dir / "temporal_run.json").read_text(encoding="utf-8"))
    dataset = execution["config"]["dataset"]

    # round_trip으로 저장된 OOF 문턱의 정밀도를 유지한다.
    read = lambda name: pd.read_csv(args.source_dir / name, float_precision="round_trip")
    folds = compare_recall_targets(read("threshold_compare.csv"), read("predictions.csv"),
                                  positive_label=dataset["positive_label"], negative_label=dataset["negative_label"])
    
    summary = _summarize_results(folds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    folds.to_csv(args.output_dir / "fold_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_dir / "summary.csv", index=False, encoding="utf-8-sig")

    record = {"source_dir": str(args.source_dir.resolve()), "source_execution": execution,
              "targets": [0.8, 0.9], "selection_rule": "min_alarm_then_max_precision_then_max_threshold",
              "threshold_candidates": "existing_oof_quantile_grid", "refit": False,
              "test_used": False, "saved_bundles_changed": False}
    
    (args.output_dir / "execution.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    
    print(summary.to_string(index=False))
    print("Recall 80%·90%는 과거 OOF 선택 목표입니다. 이후 구간에서의 달성을 보장하지 않습니다.")
    print(f"Recall 목표 비교 저장 경로: {args.output_dir}")


if __name__ == "__main__":
    main()
