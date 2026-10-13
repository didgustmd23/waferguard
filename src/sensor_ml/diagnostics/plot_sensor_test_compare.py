# ==========================================
# 전체 센서·고정 Top-20의 동일 Test 비교 그림
# - 기존 평가 로그만 읽고 모델 추론·학습·문턱 탐색은 하지 않음
# - 모델 내 두 시나리오의 공통 확률과 모델 간 평가 행·정답을 대조
# - 목표별 혼동행렬과 모델별 PR 곡선을 새 그림 폴더에 저장
# ==========================================
import argparse
from pathlib import Path

from sklearn.metrics import precision_recall_curve

from src.sensor_ml.diagnostics.plot_v2_test import read_evaluation, validate_pair, draw_confusion


def read_comparison(all_dir, top20_80_dir, top20_90_dir):
    """모델마다 문턱만 다르고, 모델 간에는 같은 Test 행·정답인지 검증한다."""
    all80 = read_evaluation(all_dir / "recall80", 0.8)
    all90 = read_evaluation(all_dir / "recall90", 0.9)
    top80 = read_evaluation(top20_80_dir, 0.8)
    top90 = read_evaluation(top20_90_dir, 0.9)

    validate_pair(all80[0], all90[0])
    validate_pair(top80[0], top90[0])

    columns = ["source_row_id", "label"]

    if not all80[0][columns].equals(top80[0][columns]):
        raise ValueError("모델 간 평가 행·정답이 달라 동일 Test 비교를 그릴 수 없습니다.")
    
    for _, record in (all80, all90, top80, top90):
        if (record["model_fit_rows"] != all80[1]["model_fit_rows"]
                or record["training_protocol_id"] != all80[1]["training_protocol_id"]):
            raise ValueError("모델 간 학습 행 수·분할 계약이 다릅니다.")
        
    for _, record in (all80, all90):
        record["display_name"] = f"전체 센서 {len(record['sensors'])}개\n"

    for _, record in (top80, top90):
        record["display_name"] = f"고정 Top-{len(record['sensors'])}\n"

    return all80, all90, top80, top90


def draw_comparison_pr(plt, groups, destination):
    """모델별 곡선을 분리하고 80%·90% OOF 목표 문턱 위치를 표시한다."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))

    for ax, upper, title in zip(axes, (1.02, 0.2), ("전체 범위", "정밀도 0~20% 확대")):
        for name, color, first, second in groups:
            frame, record = first
            precision, recall, _ = precision_recall_curve(frame.label.eq(1), frame.positive_score)
            
            ax.step(recall, precision, where="post", color=color,
                    label=f"{name} (AP={record['metrics']['average_precision']:.4f})")
            
            # 모델 색은 동일하게, 목표는 사각형·원으로 구분한다.
            for (_, scenario), marker in zip((first, second), ("s", "o")):
                metrics = scenario["metrics"]
                ax.scatter(metrics["recall"], metrics["precision"], color=color, marker=marker,
                           s=70, zorder=4, edgecolor="white", linewidth=0.5,
                           label=f"{name} · 목표 {scenario['target_recall']:.0%}")
                
        frame = groups[0][2][0]
        prevalence = frame.label.eq(1).mean()

        ax.axhline(prevalence, color="gray", linestyle="--", label=f"불량 비율 {prevalence:.2%}")
        ax.set(title=title, xlabel="Recall (불량 검출률)", ylabel="Precision (선별 대상 중 불량 비율)",
               xlim=(-0.025, 1.025), ylim=(0, upper))
        
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("전체 센서 vs 고정 Top-20 — 동일 기존 Test PR 곡선", fontsize=15)
    
    fig.text(0.5, 0.025,
             f"동일 Test {len(frame)}건 · 불량 {frame.label.eq(1).sum()}건 | 동일 학습 범위·XGBoost 설정\n"
             "각 모델의 OOF 문턱 적용 · Test에서 문턱 재선택 없음 · 독립 미사용 평가 아님",
             ha="center", fontsize=10)
    
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))
    
    fig.savefig(destination, dpi=170)

    plt.close(fig)


def main():
    """네 시나리오를 검증한 후 기존 그림을 덮어쓰지 않고 저장한다."""
    parser = argparse.ArgumentParser(description="전체 센서·고정 Top-20의 동일 Test 로그 시각화")
    parser.add_argument("--all-result-dir", type=Path, required=True)
    parser.add_argument("--top20-recall80-dir", type=Path, required=True)
    parser.add_argument("--top20-recall90-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    args = parser.parse_args()
    
    all80, all90, top80, top90 = read_comparison(
        args.all_result_dir, args.top20_recall80_dir, args.top20_recall90_dir)
    
    if args.figure_dir.exists() and (not args.figure_dir.is_dir() or any(args.figure_dir.iterdir())):
        raise ValueError("비어 있는 새 그림 폴더를 지정하세요.")
    
    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # 설치된 한글 글꼴을 우선 사용하고 다른 실행기의 전역 설정은 바꾸지 않는다.
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    font = next((name for name in ("Malgun Gothic", "NanumGothic", "Noto Sans CJK KR")
                 if name in fonts), "DejaVu Sans")
    
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    with plt.rc_context({"font.family": font, "axes.unicode_minus": False}):
        for target, first, second in ((80, all80, top80), (90, all90, top90)):
            draw_confusion(plt, (first[1], second[1]),
                           args.figure_dir / f"confusion_recall{target}.png",
                           title=f"전체 센서 vs 고정 Top-20 — OOF 목표 {target}%의 기존 Test 혼동행렬")
            
        draw_comparison_pr(plt, (
            ("전체 센서", "#2463a5", all80, all90),
            ("고정 Top-20", "#c33d35", top80, top90),
        ), args.figure_dir / "precision_recall_comparison.png")
        
    print(f"동일 평가 행·정답·기록 지표 검증 완료: {len(all80[0])}행")
    print(f"비교 그림 저장 경로: {args.figure_dir}")
    print("저장된 예측 로그만 사용했으며 재학습·재예측·문턱 변경은 하지 않았습니다.")


if __name__ == "__main__":
    main()
