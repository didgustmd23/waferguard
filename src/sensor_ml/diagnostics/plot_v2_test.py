# ==========================================
# 저장 V2의 기존 Test 혼동행렬·PR 곡선 생성
# - 평가 로그의 정답·확률·판정만 읽으며 모델을 불러오지 않음
# - 두 시나리오의 평가 행·확률과 기록된 지표가 일치하는지 검증
# - 혼동행렬은 문턱별로, PR 곡선은 공통 모델 하나로 표현
# - 이미 사용한 Test의 비교 결과이며 독립 미사용 평가가 아님
# ==========================================
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from src.modeling_metrics import evaluate_binary_scores


def read_evaluation(directory, target):
    """예측 로그를 평가 기록과 대조해 잘못 연결된 결과의 시각화를 막는다."""
    directory = Path(directory)
    record = json.loads((directory / "evaluation.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(directory / "test_predictions.csv", float_precision="round_trip")
    required = {"source_row_id", "label", "positive_score", "predicted_positive", "predicted_label"}

    if not required.issubset(frame.columns) or frame[list(required)].isna().any().any():
        raise ValueError("예측 로그의 필수 컬럼이 없거나 결측값이 있습니다.")
    
    if frame.source_row_id.duplicated().any():
        raise ValueError("예측 로그에 중복된 평가 행이 있습니다.")
    
    if (record.get("status") != "completed" or record.get("refit") is not False
            or record.get("evaluation_scope") != "existing_v1_test_comparison"
            or record.get("target_recall") != target):
        raise ValueError("완료된 V2 기존 Test 비교 기록과 목표 시나리오를 확인하세요.")

    # 저장된 문턱으로 판정만 대조한다. 문턱 탐색·모델 추론은 수행하지 않는다.
    threshold = record["threshold"]
    selected = frame.positive_score >= threshold

    if (frame.predicted_positive.dtype != bool
            or not np.array_equal(frame.predicted_positive, selected)
            or not np.array_equal(frame.predicted_label, np.where(selected, 1, -1))):
        raise ValueError("저장된 판정이 평가 문턱·양성 라벨과 일치하지 않습니다.")
    
    measured = asdict(evaluate_binary_scores(
        frame.label, frame.positive_score, positive_label=1, negative_label=-1,
        threshold=threshold,
    ))

    # 소수점 직렬화 오차만 허용하며 혼동행렬·AP 등 원본 지표를 검증한다.
    for key, value in measured.items():
        if not np.isclose(value, record["metrics"][key], rtol=0, atol=1e-12):
            raise ValueError(f"예측 로그와 평가 기록의 지표가 다릅니다: {key}")
        
    ratio = selected.mean()

    if not np.isclose(ratio, record["metrics"]["reinspection_ratio"], rtol=0, atol=1e-12):
        raise ValueError("예측 로그와 평가 기록의 선별 비율이 다릅니다.")
    
    return frame.sort_values("source_row_id").reset_index(drop=True), record


def validate_pair(first, second):
    """두 문턱 시나리오가 같은 평가 행·정답·확률을 공유하는지 확인한다."""
    columns = ["source_row_id", "label", "positive_score"]

    if not first[columns].equals(second[columns]):
        raise ValueError("두 시나리오의 평가 행·정답·확률이 달라 공통 PR 곡선을 그릴 수 없습니다.")


def draw_confusion(plt, records, destination, title="V2 고정 센서 20개 — 기존 Test 혼동행렬"):
    """실제 클래스는 행, 모델의 위험 선별 판정은 열로 표시한다."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.3))

    for ax, record in zip(axes, records):
        metrics = record["metrics"]
        matrix = np.array([
            [metrics["true_negative"], metrics["false_positive"]],
            [metrics["false_negative"], metrics["true_positive"]],
        ])

        # 공통 색 범위를 사용해 두 시나리오의 건수를 직접 비교할 수 있게 한다.
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=metrics["support"])

        for (row, column), count in np.ndenumerate(matrix):
            name = (("TN", "FP"), ("FN", "TP"))[row][column]
            ax.text(column, row, f"{name}\n{count}건", ha="center", va="center",
                    fontsize=17, color="white" if count > metrics["support"] / 2 else "#172b4d")
            
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["미선별", "위험 대상 선별"],
               yticklabels=["정상", "불량"], xlabel="모델 판정", ylabel="실제 정답")
        
        ax.set_title(
            f"{record.get('display_name', '')}OOF Recall 목표 {record['target_recall']:.0%} 시나리오\n"
            f"Test Recall {metrics['recall']:.2%} · 선별 {metrics['reinspection_ratio']:.2%}\n"
            f"문턱 {record['threshold']:.9f}", fontsize=11, pad=12,
        )

    fig.suptitle(title, fontsize=15)
    metrics = records[0]["metrics"]
    support, positives = metrics["support"], metrics["positive_support"]

    fig.text(0.5, 0.025, f"기존 Test {support}건 (정상 {support - positives} · 불량 {positives}) | 독립 미사용 평가 아님",
             ha="center", fontsize=10)
    
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(destination, dpi=170)

    plt.close(fig)


def draw_pr(plt, frame, records, destination):
    """공통 PR 곡선과 저장된 두 문턱의 위치를 전체·확대 범위로 표시한다."""
    precision, recall, _ = precision_recall_curve(frame.label.eq(1), frame.positive_score)
    prevalence = frame.label.eq(1).mean()
    ap = records[0]["metrics"]["average_precision"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for ax, upper, title in zip(axes, (1.02, 0.2), ("전체 범위", "정밀도 0~20% 확대")):
        # AP는 계단식 평균 정밀도다. 사다리꼴 적분 PR-AUC와 혼용하지 않는다.
        ax.step(recall, precision, where="post", color="#2463a5", label=f"V2 공통 곡선 (AP={ap:.4f})")
        ax.axhline(prevalence, color="gray", linestyle="--", label=f"불량 비율 {prevalence:.2%}")

        for record, color, marker in zip(records, ("#c33d35", "#187d65"), ("o", "s")):
            metrics = record["metrics"]
            ax.scatter(metrics["recall"], metrics["precision"], color=color, marker=marker,
                       s=65, zorder=4, label=f"OOF 목표 {record['target_recall']:.0%} 문턱")
            
        ax.set(title=title, xlabel="Recall (불량 검출률)", ylabel="Precision (선별 대상 중 불량 비율)",
               xlim=(-0.025, 1.025), ylim=(0, upper))
        
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9, loc="upper right")

    fig.suptitle("V2 고정 센서 20개 — 기존 Test PR 곡선", fontsize=15)

    positions = " | ".join(
        f"목표 {record['target_recall']:.0%}: Recall {record['metrics']['recall']:.2%} / Precision {record['metrics']['precision']:.2%}"
        for record in records
    )

    fig.text(0.5, 0.025,
             f"같은 모델·확률, 두 문턱 | {positions}\n"
             f"기존 Test {len(frame)}건 · 불량 {frame.label.eq(1).sum()}건 | 문턱 재선택 없음 · 독립 미사용 평가 아님",
             ha="center", fontsize=9)
    
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))
    fig.savefig(destination, dpi=170)

    plt.close(fig)


def main():
    """로그 검증 후 새 그림 폴더에만 저장한다. 학습·추론 모듈은 사용하지 않는다."""
    parser = argparse.ArgumentParser(description="V2 기존 Test 로그의 혼동행렬·PR 곡선 생성 (재학습·재예측 없음)")
    parser.add_argument("--recall90-dir", type=Path, required=True)
    parser.add_argument("--recall80-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    args = parser.parse_args()

    first, record90 = read_evaluation(args.recall90_dir, 0.9)
    second, record80 = read_evaluation(args.recall80_dir, 0.8)

    validate_pair(first, second)

    if args.figure_dir.exists() and any(args.figure_dir.iterdir()):
        raise ValueError("기존 그림을 보존하려면 비어 있는 새 그림 폴더를 지정하세요.")

    # 화면 없는 실행도 지원하며 설치된 한글 글꼴을 지역 설정으로 적용한다.
    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    fonts = {font.name for font in font_manager.fontManager.ttflist}
    font = next((name for name in ("Malgun Gothic", "NanumGothic", "Noto Sans CJK KR")
                 if name in fonts), "DejaVu Sans")
    
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    with plt.rc_context({"font.family": font, "axes.unicode_minus": False}):
        draw_confusion(plt, (record90, record80), args.figure_dir / "confusion_matrices.png")
        draw_pr(plt, first, (record90, record80), args.figure_dir / "precision_recall_curve.png")
        
    print(f"평가 기록·공통 확률 검증 완료: {len(first)}행")
    print(f"그림 저장 경로: {args.figure_dir}")
    print("기존 Test 예측 로그만 재사용했습니다. 재학습·재예측·문턱 변경은 하지 않았습니다.")


if __name__ == "__main__":
    main()
