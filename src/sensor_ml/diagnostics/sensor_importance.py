# ==========================================
# V2 후보의 전체 센서 중요도 확인용 실행기
# - 지정한 Time Train만 사용해 후보 하나를 진단용으로 학습
# - 품질 필터 통과 센서의 gain 중요도와 제거 사유를 원본 센서에 연결
# - 전체 목록·전체 그래프·Top-20 그래프와 실제 설정을 저장
# - Validation/Test 평가·센서 재선정·최종 모델 저장은 하지 않음
# ==========================================

import argparse
import json
import math
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.dataset_schema import split_frame_to_xy
from src.modeling_config import DEFAULT_CONFIG_PATH, load_modeling_config
from src.modeling_models import fit_pipeline
from src.modeling_preprocessing import quality_filter_record
from src.split_contract import PROTOCOL_ID, validate_train_role
from src.sensor_ml.experiments.time_weight_compare import DEFAULT_PRESET, build_weight_candidates


def collect_importances(pipeline):
    """원본 센서 전체에 학습 여부·제거 사유·중요도를 이름 기준으로 연결한다."""
    if "selector" in pipeline.named_steps:
        raise ValueError("전체 센서 진단에는 특징 선택기가 없는 Pipeline이 필요합니다.")
    
    quality = pipeline.named_steps["quality_filter"]
    classifier = pipeline.named_steps["model"]

    if classifier.importance_type != "gain":
        raise ValueError("이 진단은 XGBoost gain 중요도를 기준으로 합니다.")
    
    names = quality.retained_features_
    values = np.asarray(classifier.feature_importances_, dtype=float)

    if values.shape != (len(names),) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("센서 이름과 중요도 개수 또는 중요도 값이 올바르지 않습니다.")
    
    scores = dict(zip(names, values))
    rows = []

    for name in quality.feature_names_in_:
        # 학습되지 않은 센서는 중요도 0으로 해석하지 않도록 NaN으로 남긴다.
        status = ("removed_high_missing" if name in quality.high_missing_features_
                  else "removed_constant" if name in quality.constant_features_ else "used")
        
        rows.append({"feature": name, "status": status,
                     "train_missing_ratio": float(quality.missing_ratios_[name]),
                     "gain_importance": scores.get(name, np.nan)})
        
    result = pd.DataFrame(rows)
    result["importance_rank"] = result.gain_importance.rank(method="min", ascending=False).astype("Int64")

    # 동점은 원래 센서 순서를 보존하고 제거된 센서는 목록 마지막에 둔다.
    return result.sort_values("gain_importance", ascending=False, kind="stable", na_position="last").reset_index(drop=True)


def fit_importance_candidate(frame, config, preset, model_name, *, n_jobs=1):
    """실험 프리셋의 후보 하나를 Train 전체에서 진단용으로 학습한다."""
    validate_train_role(frame)

    if PROTOCOL_ID not in frame or frame.empty:
        raise ValueError("전체 센서 중요도 확인에는 생성 계약을 포함한 Train split이 필요합니다.")
    
    candidates = build_weight_candidates(config, preset, n_jobs=n_jobs)

    if model_name not in candidates:
        raise ValueError(f"알 수 없는 V2 후보입니다: {model_name}")
    
    features, labels, _ = split_frame_to_xy(frame, config.dataset)

    # 저장된 V2 CSV에는 학습 객체가 없으므로 후보 하나를 새로 학습한다.
    # 시간순 CV에서 계산한 중요도가 아니라 전체 Train 학습의 설명용 결과다.
    fitted = fit_pipeline(candidates[model_name], features, labels, config, model_name, n_jobs=n_jobs)

    return collect_importances(fitted), fitted


def draw_importances(table, figure_dir, *, model_name):
    """전체 센서를 여러 패널로 나누고 학습 센서 Top-20도 별도로 표시한다."""
    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Patch

    directory = Path(figure_dir)
    directory.mkdir(parents=True, exist_ok=True)

    fonts = {font.name for font in font_manager.fontManager.ttflist}
    font = next((name for name in ("Malgun Gothic", "NanumGothic", "Noto Sans CJK KR") if name in fonts), "DejaVu Sans")

    reason = {"used": "", "removed_high_missing": " [결측률 제거]", "removed_constant": " [상수 제거]"}
    paths = {}

    with plt.rc_context({"font.family": font, "axes.unicode_minus": False}):
        # 원본 센서 전체를 누락 없이 표시한다. 제거 센서는 회색과 사유 표기를 함께 사용한다.
        columns = min(4, max(1, math.ceil(len(table) / 150)))
        per_panel = math.ceil(len(table) / columns)
        fig, axes = plt.subplots(1, columns, figsize=(6 * columns, max(4, per_panel * 0.19)),
                                 squeeze=False, sharex=True)
        
        for index, axis in enumerate(axes[0]):
            section = table.iloc[index * per_panel:(index + 1) * per_panel]
            labels = [row.feature + reason[row.status] for row in section.itertuples()]
            colors = ["#2864a0" if status == "used" else "#888888" for status in section.status]
            axis.barh(np.arange(len(section)), section.gain_importance.fillna(0), color=colors)
            axis.set_yticks(np.arange(len(section)), labels, fontsize=7)
            axis.invert_yaxis()
            axis.set(xlabel="gain 중요도 (정규화)", title=f"전체 목록 {index * per_panel + 1}~{index * per_panel + len(section)}")
            axis.grid(axis="x", alpha=0.2)

        fig.suptitle(f"{model_name}: 원본 센서 전체 / 회색 표기는 학습 제외이며 중요도 0이 아님", fontsize=12)
        fig.legend(handles=[Patch(color="#2864a0", label="학습 센서"), Patch(color="#888888", label="학습 제외")], loc="lower center")
        fig.tight_layout(rect=(0, 0.02, 1, 0.98))

        paths["all"] = directory / "feature_importance_all.png"
        fig.savefig(paths["all"], dpi=120)

        plt.close(fig)

        top = table.loc[table.status.eq("used")].head(20).iloc[::-1]
        fig, axis = plt.subplots(figsize=(10, max(4, len(top) * 0.3)))

        axis.barh(top.feature, top.gain_importance, color="#2864a0")
        axis.set(title=f"{model_name}: 중요도 상위 {len(top)}개", xlabel="gain 중요도 (정규화)")
        axis.grid(axis="x", alpha=0.2)

        fig.tight_layout()

        paths["top20"] = directory / "feature_importance_top20.png"

        fig.savefig(paths["top20"], dpi=150)

        plt.close(fig)
    return paths


def save_results(args, config, preset, frame, table, fitted):
    """중요도 목록·그림·학습 출처와 해석 한계를 새 폴더에 저장한다."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")

    paths = draw_importances(table, args.figure_dir, model_name=args.model)

    record = {"config": asdict(config), "preset": preset, "model_name": args.model,
              "model_parameters": fitted.named_steps["model"].get_params(),
              "effective_scale_pos_weight": fitted.named_steps["model"].effective_scale_pos_weight_,
              "train_path": str(args.train.resolve()), "input_rows": len(frame),
              "training_protocol_id": frame[PROTOCOL_ID].iloc[0],
              "quality_filter": quality_filter_record(fitted), "importance_type": "normalized_gain",
              "fit_scope": "full_train_diagnostic", "external_validation_used": False,
              "test_used": False, "is_final_selection": False}
    
    (args.output_dir / "execution.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    report = ["# 전체 센서 Feature Importance", "",
              f"후보: `{args.model}`. 전체 Train에서 새로 학습한 설명용 모델이며 최종 모델이 아닙니다.", "",
              f"원본 {len(table)}개 / 학습 {int(table.status.eq('used').sum())}개 / 제거 {int(table.status.ne('used').sum())}개.", ""]
    
    for title, path in paths.items():
        link = Path(os.path.relpath(path, args.output_dir)).as_posix()
        report.extend([f"![{title}]({link})", ""])

    report.extend(["- 중요도는 분할의 평균 손실 개선량(gain)을 정규화한 값입니다. 불량 검출률이나 인과적 영향 비율이 아닙니다.",
                   "- 제거 센서의 중요도는 빈 값입니다. 학습 센서의 0은 이 학습에서 분할에 사용되지 않았음을 뜻합니다.",
                   "- 상위 20개 그림은 표시 범위일 뿐이며 Top-20 모델을 학습하거나 센서 목록을 확정하지 않습니다.",
                   "- 상관 센서·학습 시기·모델 설정에 따라 순위가 달라질 수 있습니다. 시간 fold별 선택 안정성 결과와 구분합니다.",
                   "- Validation/Test를 읽지 않았으며 성능 평가나 threshold 탐색을 하지 않았습니다."])
    
    (args.output_dir / "importance.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def parse_args(argv=None):
    """진단 후보·입력과 새 출력 경로를 확인한다."""
    parser = argparse.ArgumentParser(description="V2 전체 센서 XGBoost gain 중요도 확인")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--preset", type=Path, default=DEFAULT_PRESET)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--model", default="v2_m0_ratio")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args(argv)

    if args.n_jobs < 1:
        raise ValueError("병렬 수는 1 이상의 정수여야 합니다.")
    
    for directory in (args.output_dir, args.figure_dir):
        if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
            raise ValueError(f"비어 있는 새 출력 폴더를 지정하세요: {directory}")
        
    return args


def main():
    """인자 확인 → Train 진단 학습 → 중요도 저장 순서로 실행한다."""
    args = parse_args()
    config = load_modeling_config(args.config)
    preset = json.loads(args.preset.read_text(encoding="utf-8"))
    frame = pd.read_csv(args.train)
    table, fitted = fit_importance_candidate(frame, config, preset, args.model, n_jobs=args.n_jobs)

    save_results(args, config, preset, frame, table, fitted)
    
    print(table.loc[table.status.eq("used")].head(20).to_string(index=False))
    print(f"전체 목록: {args.output_dir / 'feature_importance.csv'}")
    print(f"그림 연결 보고서: {args.output_dir / 'importance.md'}")
    print("전체 Train 진단용 학습 결과이며 최종 센서 선정·성능 평가가 아닙니다.")


if __name__ == "__main__":
    main()
