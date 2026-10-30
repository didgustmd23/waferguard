"""지식집의 '독립 계산 확인' 항목 근거. 합성 배열만 쓰고 제품 코드는 import하지 않는다.

제품 계산 순서를 따라 하는 부분(대치 → IQR clip, 구간 경계 np.unique 병합)은 transform.py를 읽고 같은 식으로 다시 쓴 것이며,
제품 함수의 실행 결과가 아니다. 결과는 checks/results.json에 고정 키로 저장한다.
"""

import json
from pathlib import Path

import numpy as np
import sklearn
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def iqr_clip_low_card():
    # 3%만 1인 0/1 특징: q1=q3=0이라 경계가 [0,0]이 되어 모든 값이 0으로 바뀐다.
    values = np.array([1.0] * 30 + [0.0] * 970)
    q1, q3 = np.quantile(values, [0.25, 0.75])
    margin = 1.5 * (q3 - q1)
    clipped = np.clip(values, q1 - margin, q3 + margin)
    return {"bounds": [float(q1 - margin), float(q3 + margin)], "unique_before": int(np.unique(values).size),
            "unique_after": int(np.unique(clipped).size)}


def quantile_binning_low_card():
    # 값 3종 특징에 분위수 5구간: 동점 경계가 합쳐져 실제 구간 수가 줄어든다(np.unique(...)[1:-1] 내부 경계).
    rng = np.random.default_rng(0)
    values = rng.choice([1.0, 2.0, 3.0], size=1000, p=[0.6, 0.3, 0.1])
    edges = np.unique(np.quantile(values, np.linspace(0, 1, 6)))[1:-1]
    binary = np.array([0.0] * 900 + [1.0] * 100)
    binary_edges = np.unique(np.quantile(binary, np.linspace(0, 1, 6)))[1:-1]
    return {"requested_bins": 5, "three_valued_bins": int(len(edges) + 1), "binary_10pct_bins": int(len(binary_edges) + 1)}


def uniform_binning_outlier():
    rng = np.random.default_rng(1)
    values = np.concatenate([rng.normal(50, 5, 995), [1000.0] * 5])
    edges = np.linspace(values.min(), values.max(), 6)[1:-1]
    counts = np.bincount(np.searchsorted(edges, values, side="right"), minlength=5)
    return {"bin_counts": counts.tolist(), "largest_bin_share": float(counts.max() / counts.sum())}


def mean_imputation_outlier():
    # 제품은 대치값을 clip 전 관측값으로 계산한다. 극단값이 있으면 평균 대치값이 중앙값보다 크게 끌린다.
    rng = np.random.default_rng(2)
    values = np.concatenate([rng.normal(10, 1, 970), [500.0] * 30])
    return {"mean": round(float(values.mean()), 3), "median": round(float(np.median(values)), 3)}


def standard_scaling_outlier():
    rng = np.random.default_rng(3)
    inliers = rng.normal(10, 1, 990)
    values = np.concatenate([inliers, [500.0] * 10])
    z = (inliers - values.mean()) / values.std()
    q1, q3 = np.quantile(values, [0.25, 0.75])
    r = (inliers - np.median(values)) / (q3 - q1)
    return {"standard_inlier_std": round(float(z.std()), 4), "robust_inlier_std": round(float(r.std()), 4)}


def stratify_min_members():
    # 제품 random 분할은 train_test_split을 두 번 호출한다(두 번째는 남은 행의 라벨로 계층화).
    rng = np.random.default_rng(4)
    results = {}
    for members in (1, 2, 5, 6, 7, 10):
        labels = np.array(["a"] * 600 + ["b"] * 200 + ["c"] * (100 - members) + ["d"] * members)
        positions = np.arange(len(labels))
        try:
            _, remaining = train_test_split(positions, train_size=0.7, random_state=42, stratify=labels)
            train_test_split(remaining, test_size=0.5, random_state=42, stratify=labels[remaining])
            results[str(members)] = "ok"
        except ValueError as error:
            results[str(members)] = "ValueError:" + ("least_populated" if "least populated" in str(error) else "other")
    del rng
    # 이 관찰의 재현 조건. 다른 구성·비율·시드로 일반화하지 않는다.
    return {"conditions": {"classes": "a 600, b 200, c 100-n, d n", "train_ratio": 0.7, "validation_ratio": 0.15,
                           "test_ratio": 0.15, "random_state": 42, "stages": "train_test_split 2회(두 번째는 남은 행 계층화)"},
            "result_by_n": results, "scope": "단일 구성 관찰이며 최소 표본 규칙이 아님"}


def random_split_group_leakage():
    groups = np.repeat(np.arange(200), 10)
    positions = np.arange(len(groups))
    train, test = train_test_split(positions, train_size=0.7, random_state=42)
    shared = set(groups[train]) & set(groups[test])
    splitter = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=42)
    g_train, g_test = next(splitter.split(positions, groups=groups))
    uneven = np.repeat(np.arange(60), np.random.default_rng(5).integers(5, 100, 60))
    u_train, _ = next(GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=42).split(np.arange(len(uneven)), groups=uneven))
    return {"random_shared_groups": len(shared), "group_shared_groups": len(set(groups[g_train]) & set(groups[g_test])),
            "uneven_group_train_row_ratio": round(len(u_train) / len(uneven), 4)}


def main():
    results = {"sklearn_version": sklearn.__version__, "numpy_version": np.__version__,
               "iqr_clip_low_card": iqr_clip_low_card(), "quantile_binning_low_card": quantile_binning_low_card(),
               "uniform_binning_outlier": uniform_binning_outlier(), "mean_imputation_outlier": mean_imputation_outlier(),
               "standard_scaling_outlier": standard_scaling_outlier(), "stratify_min_members": stratify_min_members(),
               "random_split_group_leakage": random_split_group_leakage()}
    path = Path(__file__).with_name("results.json")
    path.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
