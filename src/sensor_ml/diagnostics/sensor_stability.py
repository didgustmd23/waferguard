# ==========================================
# 시간순 학습 구간별 XGBoost 센서 중요도 안정성 진단
# - 기존 V2 시간 분할과 모델 설정을 재사용
# - 각 과거 학습 구간에서만 전처리와 모델을 새로 fit
# - gain 순위·Top-K 등장 횟수·구간 간 공통 센서를 기록
# - 외부 Validation/Test·문턱 탐색·최종 센서 선정은 수행하지 않음
# ==========================================

import argparse
import json
from itertools import combinations
from pathlib import Path

import pandas as pd

from src.modeling_config import DEFAULT_CONFIG_PATH, load_modeling_config
from src.sensor_ml.diagnostics.sensor_importance import fit_importance_candidate
from src.split_contract import temporal_folds, validate_train_role
from src.sensor_ml.experiments.time_weight_compare import DEFAULT_PRESET


# ==========================================
# Fold별 기록을 센서 안정성 표와 겹침 표로 집계
# - 제거된 센서의 순위는 결측으로 유지
# - 평균 순위는 해당 센서가 학습에 사용된 Fold만 반영
# - 동점의 Top-K 포함 여부는 원본 센서 순서를 기준으로 결정
# ==========================================
def summarize_stability(records):
    """Fold별 원본 기록에서 센서 등장 횟수와 공통 센서 수를 계산한다."""
    ranks = records.pivot(index="feature", columns="fold", values="importance_rank")
    ranks.columns = [f"fold_{fold}_rank" for fold in ranks.columns]
    counts = records.groupby("feature")["in_topk"].sum().rename("topk_count")
    stability = ranks.join(counts)
    stability["mean_rank_when_used"] = ranks.mean(axis=1)
    stability["used_fold_count"] = ranks.notna().sum(axis=1)
    stability = stability.sort_values(
        ["topk_count", "mean_rank_when_used"], ascending=[False, True], kind="stable"
    )

    # Fold별 센서 집합을 직접 비교하여 같은 센서를 중복 집계하지 않는다.
    top_sets = {
        fold: set(rows.loc[rows["in_topk"], "feature"])
        for fold, rows in records.groupby("fold", sort=True)
    }

    overlaps = []

    for first, second in combinations(top_sets, 2):
        common = top_sets[first] & top_sets[second]
        union = top_sets[first] | top_sets[second]

        overlaps.append({
            "comparison": f"Fold {first} / Fold {second}",
            "first_sensor_count": len(top_sets[first]),
            "second_sensor_count": len(top_sets[second]),
            "common_sensor_count": len(common),
            "jaccard_similarity": len(common) / len(union) if union else None,
            "common_sensors": json.dumps(sorted(common), ensure_ascii=False),
        })

    return stability, pd.DataFrame(overlaps)


def compare_stability(frame, config, preset, model_name, *, top_k=20, n_jobs=1):
    """시간순 Fold의 학습 부분만 사용해 중요도와 학습 범위를 수집한다."""
    validate_train_role(frame)

    if top_k < 1 or n_jobs < 1:
        raise ValueError("센서 수와 병렬 수는 1 이상의 정수여야 합니다.")
    
    folds = temporal_folds(frame, config.dataset, preset["outer_splits"])

    tables, periods = [], []

    for fold, (train_indices, valid_indices) in enumerate(folds, start=1):
        train = frame.iloc[train_indices].copy()
        print(f"Fold {fold} 학습 시작: {len(train)}건")

        table, _ = fit_importance_candidate(
            train, config, preset, model_name, n_jobs=n_jobs
        )

        # 중요도 내림차순 결과에서 학습 센서만 선택한다. 제거 센서는 제외한다.
        table = table.copy()
        table["fold"] = fold
        table["in_topk"] = False
        used_indices = table.index[table["status"].eq("used")]
        table.loc[used_indices[:top_k], "in_topk"] = True
        tables.append(table)

        # 미래 구간은 크기·기간만 기록하며 학습 함수에 전달하지 않는다.
        times = pd.to_datetime(
            frame[config.dataset.timestamp_column], format=config.dataset.timestamp_format
        )

        train_times, valid_times = times.iloc[train_indices], times.iloc[valid_indices]

        periods.append({
            "fold": fold, "train_samples": len(train_indices),
            "evaluation_samples": len(valid_indices), "used_sensor_count": len(used_indices),
            "train_start": str(train_times.min()), "train_end": str(train_times.max()),
            "evaluation_start": str(valid_times.min()), "evaluation_end": str(valid_times.max()),
        })

        print(f"Fold {fold} 완료: 학습 센서 {len(used_indices)}개")

    records = pd.concat(tables, ignore_index=True)
    stability, overlaps = summarize_stability(records)

    return {"fold_importances": records, "stability": stability.reset_index(),
            "overlap": overlaps, "fold_summary": pd.DataFrame(periods)}


def save_results(args, config, preset, tables):
    """집계 표와 설정·실행 범위를 저장하고 결과의 용도를 명시한다."""
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name, table in tables.items():
        table.to_csv(args.output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    record = {
        "train": str(args.train.resolve()), "config": str(args.config.resolve()),
        "dataset_id": config.dataset.dataset_id,
        "model_name": args.model, "top_k": args.top_k, "preset": preset,
        "n_jobs": args.n_jobs, "external_validation_used": False, "test_used": False,
        "is_performance_evaluation": False, "final_sensor_selection": False,
    }

    (args.output_dir / "execution.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    repeated = tables["stability"].query("topk_count >= 2")

    report = ["# 시간순 센서 중요도 안정성 진단", "",
              f"- 모델: `{args.model}` / Fold 수: {preset['outer_splits']} / Top-K: {args.top_k}",
              "- 각 과거 학습 구간에서만 전처리와 모델을 학습했습니다.",
              "- 평가 구간은 기간 기록에만 사용했습니다. 외부 Validation/Test는 읽지 않았습니다.",
              "- 평균 순위는 학습에 사용된 Fold만 반영합니다. 제거 상태는 원본 기록에서 확인하세요.",
              "- 목록 변동은 drift의 증거나 성능 저하의 확정이 아닙니다.",
              "- 시간 Fold의 학습 구간은 중첩됩니다. 최종 센서 목록을 확정하지 않습니다.", "",
              "## Fold 간 공통 센서", "", "```text",
              tables["overlap"].to_string(index=False), "```", "",
              "## 두 번 이상 등장한 센서", "", "```text",
              repeated.to_string(index=False), "```", "",
              "전체 목록은 stability.csv, Fold별 중요도·제거 사유는 fold_importances.csv에 기록했습니다."]
    
    (args.output_dir / "stability.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def parse_args(argv=None):
    """입력과 새 출력 경로를 확인하여 기존 실험 결과 덮어쓰기를 방지한다."""
    parser = argparse.ArgumentParser(description="시간순 Fold별 센서 gain 순위 안정성 진단")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--preset", type=Path, default=DEFAULT_PRESET)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--model", default="v2_m0_ratio")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.top_k < 1 or args.n_jobs < 1:
        parser.error("센서 수와 병렬 수는 1 이상의 정수여야 합니다.")

    if args.output_dir.exists() and (
        not args.output_dir.is_dir() or any(args.output_dir.iterdir())
    ):
        parser.error(f"비어 있는 새 출력 폴더를 지정하세요: {args.output_dir}")

    return args


def main():
    """인자 확인 → Fold별 학습 → 집계 → 결과 저장 순서로 실행한다."""
    args = parse_args()
    config = load_modeling_config(args.config)
    preset = json.loads(args.preset.read_text(encoding="utf-8"))
    tables = compare_stability(pd.read_csv(args.train), config, preset, args.model,
                               top_k=args.top_k, n_jobs=args.n_jobs)
    save_results(args, config, preset, tables)
    
    print(tables["overlap"].to_string(index=False))
    print(f"중요도 안정성 보고서: {args.output_dir / 'stability.md'}")
    print("진단 결과이며 최종 센서 선정·성능 평가가 아닙니다.")


if __name__ == "__main__":
    main()
