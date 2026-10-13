# ==========================================
# Train 내부 정상·불량별 센서 원본 분포 진단
# - 센서명은 실행 인자로 받아 데이터셋에 고정하지 않음
# - 기존 시간 Fold의 경계로 겹치지 않는 구간을 구성
# - 대치 전 관측값·결측률·표본 수를 클래스별로 비교
# - 모델 학습·센서 선정·문턱 조정·외부 평가는 하지 않음
# ==========================================

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.dataset_schema import validate_dataset_frame
from src.modeling_config import DEFAULT_CONFIG_PATH, load_modeling_config
from src.split_contract import temporal_folds, validate_train_role
from src.sensor_ml.experiments.time_weight_compare import DEFAULT_PRESET


def prepare_distribution(frame, dataset, features, n_splits):
    """입력·센서·시간 경계를 검증하고 구간 및 클래스별 통계를 구성한다."""
    validate_train_role(frame)

    available = validate_dataset_frame(frame, dataset)

    if not features or len(features) != len(set(features)):
        raise ValueError("진단 센서를 중복 없이 하나 이상 지정하세요.")
    
    if not set(features).issubset(available):
        raise ValueError("진단 대상은 Profile에서 정의한 입력 센서여야 합니다.")
    
    if set(features) & set(dataset.categorical_feature_columns):
        raise ValueError("값 분포 비교에는 수치형 센서만 지정하세요.")

    folds = temporal_folds(frame, dataset, n_splits)

    # 확장형 학습 Fold끼리는 중첩되므로 직접 분포를 비교하지 않는다.
    # 최초 학습 구간과 이후 평가 구간을 사용해 전체 Train을 한 번씩만 나눈다.
    indices = [folds[0][0]] + [evaluation for _, evaluation in folds]
    flattened = np.concatenate(indices)

    if len(flattened) != len(frame) or len(np.unique(flattened)) != len(frame):
        raise ValueError("시간 구간이 전체 Train을 중복 없이 포함하지 못했습니다.")
    
    times = pd.to_datetime(frame[dataset.timestamp_column], format=dataset.timestamp_format)
    groups, periods = {}, []

    for number, selected in enumerate(indices, start=1):
        name = f"구간 {number}"
        groups[name] = frame.iloc[selected].copy()
        periods.append({"period": name, "row_count": len(selected),
                        "start": str(times.iloc[selected].min()), "end": str(times.iloc[selected].max())})

    # 표본이 없는 클래스도 행을 남긴다. 결측률·중앙값은 0이 아닌 NaN이다.
    rows = []
    for period, data in {"전체 Train": frame, **groups}.items():
        for label, role in ((dataset.negative_label, "정상"), (dataset.positive_label, "불량")):
            subset = data.loc[data[dataset.label_column].eq(label)]
            for feature in features:
                values = pd.to_numeric(subset[feature], errors="raise")
                if np.isinf(values.to_numpy(dtype=float)).any():
                    raise ValueError(f"센서에 무한대 값이 있습니다: {feature}")
                
                rows.append({"period": period, "class": role, "feature": feature,
                             "row_count": len(values), "observed_count": int(values.count()),
                             "missing_ratio": values.isna().mean(),
                             "median": values.median() if values.count() else np.nan,
                             "p10": values.quantile(0.1), "p90": values.quantile(0.9)})
                
    return groups, pd.DataFrame(periods), pd.DataFrame(rows)


def draw_distribution(frame, groups, summary, dataset, features, figure_dir):
    """전체·시간 구간별 원본 ECDF와 결측률을 저장하고 관측 수를 표시한다."""
    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    figure_dir.mkdir(parents=True, exist_ok=True)
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    font = next((name for name in ("Malgun Gothic", "NanumGothic", "Noto Sans CJK KR")
                 if name in fonts), "DejaVu Sans")
    
    classes = ((dataset.negative_label, "정상", "tab:blue"),
               (dataset.positive_label, "불량", "tab:orange"))

    # ECDF는 그룹별 관측값 비율이다. 클래스 건수 차이에 덜 영향을 받으며 대치하지 않는다.
    def plot_ecdf(axis, data, feature):
        for label, role, color in classes:
            values = np.sort(data.loc[data[dataset.label_column].eq(label), feature].dropna().to_numpy(dtype=float))
            caption = f"{role} 관측 n={len(values)}"

            if len(values):
                axis.step(values, np.arange(1, len(values) + 1) / len(values),
                          where="post", color=color, label=caption)
            else:
                axis.plot([], [], color=color, label=caption)

        axis.set_ylabel("누적 관측 비율")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)

    with plt.rc_context({"font.family": font, "axes.unicode_minus": False}):
        figure, axes = plt.subplots(len(features), 3, figsize=(16, 3 * len(features)), squeeze=False)
        temporal, x = summary.loc[summary.period.ne("전체 Train")], np.arange(len(groups))

        for row, feature in enumerate(features):
            plot_ecdf(axes[row, 0], frame, feature)
            axes[row, 0].set_title(f"{feature}: 전체 원본 분포")

            for _, role, color in classes:
                points = temporal.loc[temporal.feature.eq(feature) & temporal["class"].eq(role)].set_index("period").reindex(groups)
                axes[row, 1].plot(x, points["median"], "o-", color=color, label=role)
                axes[row, 2].plot(x, points["missing_ratio"] * 100, "o-", color=color, label=role)

            for column, title in ((1, "구간별 관측 중앙값"), (2, "구간별 결측률 (%)")):
                axes[row, column].set(title=f"{feature}: {title}", xticks=x, xticklabels=list(groups))
                axes[row, column].legend()
                axes[row, column].grid(alpha=0.2)

        figure.tight_layout()
        figure.savefig(figure_dir / "class_distribution.png", dpi=140)
        plt.close(figure)

        # 구간마다 정상·불량의 분포가 분리되는지 직접 비교한다.
        figure, axes = plt.subplots(len(features), len(groups),
                                    figsize=(5 * len(groups), 3 * len(features)), squeeze=False)
        
        for row, feature in enumerate(features):
            for column, (period, data) in enumerate(groups.items()):
                plot_ecdf(axes[row, column], data, feature)
                axes[row, column].set_title(f"{feature}: {period}")

            # 같은 센서는 모든 시간 구간에 같은 가로축 범위를 사용한다.
            observed = frame[feature].dropna()

            if len(observed) and observed.min() < observed.max():
                for axis in axes[row]:
                    axis.set_xlim(observed.min(), observed.max())

        figure.tight_layout()
        figure.savefig(figure_dir / "class_distribution_by_period.png", dpi=140)

        plt.close(figure)


def parse_args(argv=None):
    """입력 센서와 새 출력 폴더를 확인한다."""
    parser = argparse.ArgumentParser(description="Train 정상·불량별 센서 원본 분포 비교")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--preset", type=Path, default=DEFAULT_PRESET)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    for directory in (args.output_dir, args.figure_dir):
        if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
            parser.error(f"비어 있는 새 출력 폴더를 지정하세요: {directory}")

    return args


def main():
    """Train 검증 → 원본 통계 → 그림·해석 주의사항 저장 순서로 실행한다."""
    args = parse_args()
    config = load_modeling_config(args.config)
    preset = json.loads(args.preset.read_text(encoding="utf-8"))
    frame = pd.read_csv(args.train)
    groups, periods, summary = prepare_distribution(frame, config.dataset, args.features, preset["outer_splits"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    draw_distribution(frame, groups, summary, config.dataset, args.features, args.figure_dir)

    periods.to_csv(args.output_dir / "periods.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_dir / "class_summary.csv", index=False, encoding="utf-8-sig")

    record = {"train_path": str(args.train.resolve()), "features": args.features,
              "outer_splits": preset["outer_splits"], "test_used": False,
              "external_validation_used": False, "model_fitted": False, "imputation_applied": False}
    
    (args.output_dir / "execution.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    report = ["# 정상·불량별 센서 분포 진단", "",
              "- Train 내부 탐색입니다. 모델 학습·결측 대치·센서 확정·외부 평가는 하지 않았습니다.",
              "- 시간 구간은 서로 겹치지 않습니다. 실제 경계는 periods.csv를 확인하세요.",
              "- 불량 표본이 적거나 관측값이 없는 구간은 분포를 일반화할 수 없습니다.",
              "- 분포 차이는 drift의 원인 또는 불량 원인의 확정이 아닙니다.", ""]
    
    for filename in ("class_distribution.png", "class_distribution_by_period.png"):
        relative = Path(os.path.relpath(args.figure_dir / filename, args.output_dir)).as_posix()
        report.extend([f"![{filename}]({relative})", ""])

    (args.output_dir / "distribution.md").write_text("\n".join(report), encoding="utf-8")
    
    print(summary.to_string(index=False))
    print(f"센서 분포 보고서: {args.output_dir / 'distribution.md'}")


if __name__ == "__main__":
    main()
