# ==========================================
# 지정 센서의 원본 값 분포와 월별 변화 상세 확인
# - 센서 이름은 실행 인자로 받아 특정 데이터셋에 고정하지 않음
# - 결측 대치 전 관측값의 누적분포·월별 평균·결측률을 비교
# - 월별 관측 수와 불량 비율을 함께 남겨 표본 구성 차이를 확인
# - 학습·센서 제거·문턱 탐색은 수행하지 않음
# ==========================================

import os
from pathlib import Path

import numpy as np
import pandas as pd


def validate_detail_features(features, selected, dataset):
    """상세 진단 대상이 저장된 선택 목록의 수치형 센서인지 확인한다."""
    if not features or len(set(features)) != len(features):
        raise ValueError("상세 진단 센서는 중복 없이 하나 이상 지정하세요.")
    
    if not set(features).issubset(set(selected.feature)):
        raise ValueError("상세 진단 대상은 저장된 선택 센서 목록에 있어야 합니다.")
    
    if set(features) & set(dataset.categorical_feature_columns):
        raise ValueError("상세 값 분포·월별 평균 진단에는 수치형 센서만 지정하세요.")
    
    if dataset.timestamp_column is None:
        raise ValueError("월별 상세 진단에는 timestamp 설정이 필요합니다.")


def write_sensor_details(train, validation, features, monthly, output_dir, figure_dir):
    """상세 요약·월별 표·원본 누적분포 그림을 생성한다. 입력은 변경하지 않는다."""
    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    output_dir, figure_dir = Path(output_dir), Path(figure_dir)
    figure_path = figure_dir / "sensor_detail.png"

    if figure_path.exists():
        raise ValueError("상세 진단 그림이 이미 있습니다. 새 그림 폴더를 지정하세요.")
    
    details = monthly.loc[monthly.feature.isin(features)].copy()
    summary = []

    for name in features:
        for role, frame in (("train", train), ("validation", validation)):
            values = pd.to_numeric(frame[name], errors="raise").replace([np.inf, -np.inf], np.nan)
            summary.append({"feature": name, "split": role, "row_count": len(values),
                            "observed_count": values.count(), "missing_ratio": values.isna().mean(),
                            "min": values.min(), "p10": values.quantile(0.1), "median": values.median(),
                            "p90": values.quantile(0.9), "max": values.max(), "std": values.std(ddof=0)})
            
    summary = pd.DataFrame(summary)
    summary.to_csv(output_dir / "detail_value_summary.csv", index=False, encoding="utf-8-sig")
    details.to_csv(output_dir / "detail_monthly.csv", index=False, encoding="utf-8-sig")
    
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    font = next((name for name in ("Malgun Gothic", "NanumGothic", "Noto Sans CJK KR") if name in fonts), "DejaVu Sans")

    with plt.rc_context({"font.family": font, "axes.unicode_minus": False}):
        fig, axes = plt.subplots(len(features), 3, figsize=(15, max(3, len(features) * 2.8)), squeeze=False)
        for row, name in enumerate(features):
            periods = sorted(details.loc[details.feature.eq(name), "month"].unique())
            for role, frame, color, offset in (("train", train, "tab:blue", -0.08),
                                               ("validation", validation, "tab:orange", 0.08)):
    
                # 클래스 구성이 섞인 원본 관측값이다. 결측을 중앙값으로 채우지 않는다.
                values = pd.to_numeric(frame[name], errors="raise").to_numpy(dtype=float)
                values = np.sort(values[np.isfinite(values)])

                if len(values):
                    axes[row, 0].step(values, np.arange(1, len(values) + 1) / len(values),
                                      where="post", color=color, label=f"{role} (n={len(values)})")
                else:
                    axes[row, 0].plot([], [], color=color, label=f"{role}: 관측값 없음")

                # 같은 달에 두 split이 있으면 옆으로 조금 나눠 표시하고 빈 달은 연결하지 않는다.
                points = details.loc[details.feature.eq(name) & details.split.eq(role)].set_index("month").reindex(periods)
                x = np.arange(len(periods)) + offset

                axes[row, 1].plot(x, points["mean"], "o-", color=color, label=role)
                axes[row, 2].plot(x, points["missing_ratio"] * 100, "o-", color=color, label=role)

            for column, title in enumerate(("원본 값 누적분포", "월별 관측 평균", "월별 결측률 (%)")):
                ax = axes[row, column]
                ax.set_title(f"{name} · {title}")
                ax.grid(alpha=0.2)
                ax.legend(fontsize=8)

                if column:
                    ax.set_xticks(np.arange(len(periods)), periods, rotation=30)
                else:
                    ax.set(xlabel="센서 원본 값", ylabel="관측값 내 누적 비율", ylim=(0, 1.02))

            axes[row, 2].set_ylim(-2, 102)

        fig.tight_layout()
        figure_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_path, dpi=150)

        plt.close(fig)

    link = Path(os.path.relpath(figure_path, output_dir)).as_posix()

    text = ["# 우선 확인 센서 상세 진단", "", f"대상: {', '.join(features)}", "",
            f"![원본 분포와 월별 변화]({link})", "",
            "## 구간별 원본 값 요약", "", "```text", summary.to_string(index=False), "```", "",
            "## 월별 평균·결측률·관측 수·불량 구성", "", "```text", details.to_string(index=False), "```", "",
            "## 확인 방법", "",
            "- 누적분포 곡선의 위치·기울기 차이로 평균만으로 드러나지 않는 분포 변화를 확인합니다.",
            "- 월별 평균은 관측값만의 평균입니다. 결측률·관측 수·해당 월의 불량 비율을 함께 확인합니다.",
            "- 같은 달의 Train과 Validation은 서로 다른 시간 구간입니다. 한 달 전체의 동일 표본으로 해석하지 않습니다.",
            "- 센서는 익명이며 단위·장비 이력이 없으므로 공정 변경, 측정 변경, 우연한 표본 차이 중 원인을 확정할 수 없습니다.",
            "- PSI는 빈 구간·분위수 경계에 민감합니다. 큰 PSI만으로 센서를 제거하지 않습니다.",
            "- 모델 재학습·센서 재선택·threshold 조정·최종 Test 사용은 없습니다."]
    
    (output_dir / "sensor_detail.md").write_text("\n".join(text) + "\n", encoding="utf-8")
