# ==========================================
# OOF·시간 검증 확률 분포와 센서 선택 차이 진단
# - 문턱과 모델 설정을 변경하지 않고 저장된 예측만 요약
# - 정상·불량별 분포, AP·기본 불량률, 센서 교집합을 기록
# - 확률 변화만으로 공정 drift나 센서 축소가 원인이라고 단정하지 않음
# ==========================================

from dataclasses import asdict
from pathlib import Path
import os

import numpy as np
import pandas as pd

try:
    from src.modeling_metrics import evaluate_binary_scores
except ModuleNotFoundError:
    from modeling_metrics import evaluate_binary_scores


def summarize_scores(oof, validation, config, threshold):
    """같은 문턱을 적용한 두 평가 대상의 지표와 클래스별 확률 분포를 반환한다."""
    metrics, distributions = [], []
    
    for role, frame, score_column in (("train_oof", oof, "oof_positive_score"),
                                       ("time_validation", validation, "positive_score")):
        
        # 공통 평가 함수가 빈 입력·잘못된 label·확률 범위를 검사한다.
        measured = evaluate_binary_scores(frame.label, frame[score_column],
                                          positive_label=config.dataset.positive_label,
                                          negative_label=config.dataset.negative_label, threshold=threshold)
        
        metrics.append({"evaluation": role, **asdict(measured),
                        "positive_prevalence": measured.positive_support / measured.support,
                        "reinspection_ratio": (measured.true_positive + measured.false_positive) / measured.support})
        
        for name, label in (("정상", config.dataset.negative_label), ("불량", config.dataset.positive_label)):
            scores = frame.loc[frame.label.eq(label), score_column]
        
            # 해당 클래스가 없으면 임의의 0 대신 결측 통계와 count=0을 남긴다.
            distributions.append({"evaluation": role, "class": name, "count": len(scores),
                                  "mean": scores.mean(), "p10": scores.quantile(0.1),
                                  "median": scores.median(), "p90": scores.quantile(0.9),
                                  "above_threshold_ratio": (scores >= threshold).mean()})
            
    return pd.DataFrame(metrics), pd.DataFrame(distributions)


def sensor_overlap(oof_sensors, selected_sensors):
    """OOF 각 학습 fold와 전체 Train에서 선택한 센서의 교집합을 기록한다."""
    full = set(selected_sensors.feature)
    rows = []

    for fold, frame in oof_sensors.groupby("cv_fold"):
        sensors = set(frame.feature)
        rows.append({"cv_fold": fold, "oof_sensor_count": len(sensors), "train_sensor_count": len(full),
                     "common_count": len(sensors & full),
                     "jaccard": len(sensors & full) / len(sensors | full) if sensors | full else np.nan})
        
    return pd.DataFrame(rows)


def write_score_diagnostics(oof, validation, oof_sensors, selected_sensors, config,
                            threshold, output_dir, figure_dir):
    """진단 CSV·확률 누적분포 그림·Markdown 요약을 생성한다. 학습은 수행하지 않는다."""

    # 그래프 라이브러리는 진단 옵션을 선택했을 때만 사용한다.
    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    output_dir, figure_dir = Path(output_dir), Path(figure_dir)
    figure_path = figure_dir / "score_distribution.png"

    if figure_path.exists():
        raise ValueError("진단 그림이 이미 있습니다. 새 --figure-dir을 지정하세요.")
    
    metrics, distributions = summarize_scores(oof, validation, config, threshold)
    overlap = sensor_overlap(oof_sensors, selected_sensors)
    metrics.to_csv(output_dir / "score_metrics.csv", index=False, encoding="utf-8-sig")
    distributions.to_csv(output_dir / "score_distribution.csv", index=False, encoding="utf-8-sig")
    overlap.to_csv(output_dir / "sensor_overlap.csv", index=False, encoding="utf-8-sig")

    # Windows·Linux에서 설치된 한글 글꼴을 우선 사용하며 전역 설정은 바꾸지 않는다.
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    font = next((name for name in ("Malgun Gothic", "NanumGothic", "Noto Sans CJK KR") if name in fonts), "DejaVu Sans")

    with plt.rc_context({"font.family": font, "axes.unicode_minus": False}):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)

        for ax, (name, label) in zip(axes, (("정상", config.dataset.negative_label), ("불량", config.dataset.positive_label))):
        
            # 표본 수가 다른 두 집단을 비교하므로 빈도 대신 클래스 내 누적 비율을 그린다.
            for role, frame, column in (("Train OOF", oof, "oof_positive_score"),
                                         ("Time Validation", validation, "positive_score")):
                scores = np.sort(frame.loc[frame.label.eq(label), column].to_numpy())
        
                if len(scores):
                    ax.step(scores, np.arange(1, len(scores) + 1) / len(scores), where="post", label=f"{role} (n={len(scores)})")

            ax.axvline(threshold, color="black", linestyle="--", label=f"threshold={threshold:.6g}")
            ax.set(title=f"{name} 제품 확률 분포", xlabel="불량 예측 확률", ylabel="누적 비율", xlim=(0, 1), ylim=(0, 1.02))
            ax.legend(fontsize=8)
            ax.grid(alpha=0.25)

        fig.tight_layout()
        figure_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_path, dpi=150)
        plt.close(fig)

    # Markdown에서 다른 폴더의 그림을 상대 경로로 연결한다.
    image_link = Path(os.path.relpath(figure_path, output_dir)).as_posix()

    sections = ["# S0·M3 시간 검증 성능 저하 진단", "",
                f"고정 threshold: `{threshold!r}`. 모델·센서 선택 설정은 OOF 실행 기록에서 복원했습니다.", "",
                "## 평가 지표", "", "```text", metrics.to_string(index=False), "```", "",
                "## 정상·불량별 예측 확률", "", f"![예측 확률 누적분포]({image_link})", "",
                "```text", distributions.to_string(index=False), "```", "",
                "## OOF fold와 전체 Train 센서 교집합", "", "```text", overlap.to_string(index=False), "```", "",
                "## 해석 주의", "",
                "- 불량 그래프에서 고정 문턱 왼쪽의 누적 비율이 크면 미검출이 많다는 뜻입니다.",
                "- AP와 해당 구간의 기본 불량률을 함께 보되, 그 차이만으로 유의성을 판단하지 않습니다.",
                "- OOF는 여러 fold 모델, Validation은 전체 Train 모델의 예측입니다. 시간 차이와 학습량·선택 센서 차이가 함께 섞여 있습니다.",
                "- 센서 목록이 다른 것은 오류 자체가 아닙니다. 시간 drift나 Top-20 축소가 저하 원인인지 이 자료만으로 확정하지 않습니다.",
                "- Validation으로 threshold를 재탐색하지 않았으며 최종 Test는 사용하지 않았습니다."]
    
    (output_dir / "diagnostics.md").write_text("\n".join(sections) + "\n", encoding="utf-8")
