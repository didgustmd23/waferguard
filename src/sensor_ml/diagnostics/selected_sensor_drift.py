# ==========================================
# 저장된 최종 학습 센서의 시간순 입력 분포 진단
# - 기존 Step 3 분포 요약을 재사용하고 선택된 센서만 따로 표시
# - 원본 결측률·PSI·표준화 평균 차이와 월별 평균을 기록
# - 모델 학습·센서 재선택·threshold 변경·최종 Test 평가는 수행하지 않음
# ==========================================

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.modeling_config import DEFAULT_CONFIG_PATH, load_modeling_config
    from src.split_contract import PROTOCOL_ID, validate_split_pair
    from src.sensor_ml.diagnostics.step3_drift_check import build_temporal_drift_report
    from src.sensor_ml.diagnostics.sensor_detail import validate_detail_features, write_sensor_details
except ModuleNotFoundError:
    from modeling_config import DEFAULT_CONFIG_PATH, load_modeling_config
    from split_contract import PROTOCOL_ID, validate_split_pair
    from step3_drift_check import build_temporal_drift_report
    from sensor_detail import validate_detail_features, write_sensor_details


# ==========================================
# 동일한 시간순 split에서 저장된 센서 목록의 분포만 요약
# - 선택 목록 누락·중복·존재하지 않는 센서는 조용히 제외하지 않고 오류 처리
# - 월별 평균은 원본 관측값만 사용하며 모델 입력을 대치하거나 변환하지 않음
# ==========================================
def diagnose_selected_sensors(train, validation, selected, dataset):
    """전체 특징 요약·선택 센서 요약·월별 관측 통계를 반환한다."""
    validate_split_pair(train, validation, dataset, temporal=True)

    if "feature" not in selected or selected.empty or selected.feature.isna().any():
        raise ValueError("선택 센서 목록이 비어 있거나 feature 값이 누락되었습니다.")
    
    names = selected.feature.tolist()

    if len(set(names)) != len(names):
        raise ValueError("선택 센서 목록에 중복이 있습니다.")
    
    # 기존 범용 요약을 사용하여 이전 EDA와 지표 정의·우선순위가 달라지지 않게 한다.
    report = build_temporal_drift_report(train, validation, dataset)

    if not set(names).issubset(set(report.feature)):
        raise ValueError("선택 센서가 현재 데이터의 특징 목록에 없습니다.")
    
    report["selected_in_full_train"] = report.feature.isin(names)
    chosen = report.loc[report.selected_in_full_train].copy()

    monthly = []

    if dataset.timestamp_column is not None:
        for role, frame in (("train", train), ("validation", validation)):
            months = pd.to_datetime(frame[dataset.timestamp_column], format=dataset.timestamp_format).dt.to_period("M")
            for name in names:
                # 범주형 평균은 의미가 없으므로 월별 평균 대상에서 제외한다.
                if name in dataset.categorical_feature_columns:
                    continue
                
                values = pd.to_numeric(frame[name], errors="raise").replace([np.inf, -np.inf], np.nan)

                for month, index in frame.groupby(months).groups.items():
                    observed = values.loc[index]
                    failure_count = int(frame.loc[index, dataset.label_column].eq(dataset.positive_label).sum())
    
                    monthly.append({"split": role, "month": str(month), "feature": name,
                                    "row_count": len(observed), "observed_count": observed.count(),
                                    "missing_ratio": observed.isna().mean(), "mean": observed.mean(),
                                    "median": observed.median(), "failure_count": failure_count,
                                    "failure_ratio": failure_count / len(observed)})
                    
    return report, chosen, pd.DataFrame(monthly, columns=["split", "month", "feature", "row_count",
                                                         "observed_count", "missing_ratio", "mean", "median",
                                                         "failure_count", "failure_ratio"])


def write_sensor_diagnosis(report, selected_report, monthly, output_dir, figure_dir):
    """입력 로그는 변경하지 않고 선택 센서의 그림과 해석 문서를 저장한다."""
    import matplotlib
    matplotlib.use("Agg")
    
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    output_dir, figure_dir = Path(output_dir), Path(figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    report.to_csv(output_dir / "all_sensor_drift.csv", index=False, encoding="utf-8-sig")
    selected_report.to_csv(output_dir / "selected_sensor_drift.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output_dir / "selected_sensor_monthly.csv", index=False, encoding="utf-8-sig")

    fonts = {font.name for font in font_manager.fontManager.ttflist}
    font = next((name for name in ("Malgun Gothic", "NanumGothic", "Noto Sans CJK KR") if name in fonts), "DejaVu Sans")

    # 같은 센서 순서로 세 지표를 표시하고 단위가 다른 값을 한 막대에 섞지 않는다.
    ordered = selected_report.iloc[::-1]

    with plt.rc_context({"font.family": font, "axes.unicode_minus": False}):
        fig, axes = plt.subplots(1, 3, figsize=(14, max(4, len(ordered) * 0.3)), sharey=True)
        for ax, column, title, factor in zip(axes,
                ("missing_ratio_delta", "absolute_standardized_mean_difference", "population_stability_index"),
                ("결측률 변화 (%p)", "표준화 평균 차이 절댓값", "PSI"), (100, 1, 1)):
            
            values = pd.to_numeric(ordered[column], errors="coerce") * factor
            ax.barh(ordered.feature, values)

            # 계산 불가를 0으로 바꾸지 않으며 그림에서도 해당 센서에 표시한다.
            for index, missing in enumerate(values.isna()):
                if missing:
                    ax.text(0, index, "계산 불가", va="center", fontsize=8)

            ax.set_title(title)
            ax.grid(axis="x", alpha=0.2)

        fig.tight_layout()
        figure_path = figure_dir / "selected_sensor_drift.png"
        fig.savefig(figure_path, dpi=150)
        plt.close(fig)

    link = Path(os.path.relpath(figure_path, output_dir)).as_posix()
    columns = ["feature", "train_missing_ratio", "validation_missing_ratio", "missing_ratio_delta",
               "absolute_standardized_mean_difference", "population_stability_index", "drift_priority"]
    
    summary = ["# 선택 센서의 시간순 분포 변화 진단", "",
               f"선택 센서 {len(selected_report)}개. 기존 전체 Train 학습의 센서 목록을 재사용했습니다.", "",
               f"![선택 센서 변화 지표]({link})", "", "```text", selected_report[columns].to_string(index=False), "```", "",
               "## 해석 기준", "",
               "- 결측률 차이는 Validation−Train입니다. CSV의 0.10은 10%p이며 그림은 %p로 표시합니다.",
               "- 표준화 평균 차이는 두 구간의 표준편차로 정규화한 평균 차이입니다. PSI는 Train 분위수 구간별 비율 변화를 요약합니다.",
               "- 높음·중간·낮음은 기존 Step 3의 EDA 확인 순서입니다. 공정 이상 판정이나 통계적 유의성 기준이 아닙니다.",
               "- 상수·관측값 부족으로 계산하지 못한 값은 빈 값으로 남겼으며 안정적이라는 뜻이 아닙니다.",
               "- 월별 평균과 결측률은 selected_sensor_monthly.csv에서 확인합니다. 월별 표본 수·불량 구성 차이도 함께 고려해야 합니다.",
               "- 값 분포 변화는 성능 저하의 원인 후보입니다. 원인 확정·센서 자동 제거·threshold 조정에 사용하지 않았습니다.",
               "- Train 학습 통계로 결측을 대치하기 전의 원본 분포이며 모델 재학습과 최종 Test 사용은 없습니다."]
    
    (output_dir / "sensor_diagnostics.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


def main():
    """기존 시간 검증 실행에 기록된 Train·Validation으로 진단만 수행한다."""
    parser = argparse.ArgumentParser(description="저장된 선택 센서의 시간순 분포·결측률 진단")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--detail-features", nargs="+", help="추가 상세 진단할 저장된 선택 센서 이름입니다.")
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("결과 폴더가 비어 있지 않습니다. 새 폴더를 지정하세요.")
    
    if (args.figure_dir / "selected_sensor_drift.png").exists():
        raise ValueError("진단 그림이 이미 있습니다. 새 그림 폴더를 지정하세요.")
    
    if args.detail_features and (args.figure_dir / "sensor_detail.png").exists():
        raise ValueError("상세 진단 그림이 이미 있습니다. 새 그림 폴더를 지정하세요.")
    
    config = load_modeling_config(args.config)
    execution = json.loads((args.run_dir / "execution.json").read_text(encoding="utf-8"))
    current = json.loads(json.dumps(asdict(config), default=str))

    if current != execution["config"] or execution.get("is_final_test") is not False:
        raise ValueError("설정이나 평가 대상이 기존 시간 검증 실행과 다릅니다.")
    
    selected = pd.read_csv(args.run_dir / "selected_features.csv")

    if ("experiment" not in selected or not selected.experiment.eq(execution["experiment"]).all()
            or len(selected) != config.reduction_policy.max_sensor_count):
        raise ValueError("선택 센서 목록의 모델 출처 또는 센서 수가 실행 기록과 다릅니다.")
    
    train, validation = pd.read_csv(execution["train_path"]), pd.read_csv(execution["validation_path"])

    if PROTOCOL_ID not in train or not train[PROTOCOL_ID].eq(execution["training_protocol_id"]).all():
        raise ValueError("현재 Train 생성 계약이 시간 검증 실행 기록과 다릅니다.")
    
    report, chosen, monthly = diagnose_selected_sensors(train, validation, selected, config.dataset)

    if args.detail_features:
        validate_detail_features(args.detail_features, selected, config.dataset)

    write_sensor_diagnosis(report, chosen, monthly, args.output_dir, args.figure_dir)

    if args.detail_features:
        write_sensor_details(train, validation, args.detail_features, monthly, args.output_dir, args.figure_dir)

    (args.output_dir / "source.json").write_text(json.dumps(
        {"source_run": str(args.run_dir.resolve()), "execution": execution,
         "model_retrained": False, "threshold_changed": False,
         "detail_features": args.detail_features}, ensure_ascii=False, indent=2), encoding="utf-8")
    
    print(chosen[["feature", "missing_ratio_delta", "absolute_standardized_mean_difference",
                  "population_stability_index", "drift_priority"]].to_string(index=False))
    
    print(f"센서 분포 진단 저장 경로: {args.output_dir}")


if __name__ == "__main__":
    main()
