"""계층 분할이 성공해도 일부 구간에 클래스가 빠지는지 비율별로 확인한다(합성 라벨, 제품 코드 미사용).

제품 split._partition의 random 경로와 같은 호출 순서(train_test_split 2회, 두 번째는 남은 행을 다시 계층화)를 sklearn으로 재현한다.
"""
import json
from pathlib import Path
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

results = []
for ratios in ((0.7, 0.15, 0.15), (0.8, 0.1, 0.1), (0.9, 0.05, 0.05), (0.5, 0.4, 0.1)):
    for n in (6, 7, 10):
        labels = np.array(["a"] * 600 + ["b"] * 200 + ["c"] * (100 - n) + ["d"] * n)
        positions = np.arange(len(labels))
        row = {"ratios": list(ratios), "rare_class_rows": n, "random_state": 42}
        try:
            train, remaining = train_test_split(positions, train_size=ratios[0], random_state=42, stratify=labels)
            valid, test = train_test_split(remaining, test_size=ratios[2] / (ratios[1] + ratios[2]), random_state=42,
                                           stratify=labels[remaining])
            row["missing_class_splits"] = [name for name, part in zip(("train", "validation", "test"), (train, valid, test))
                                           if set(labels[part]) != set(labels)]
        except ValueError as error:
            row["error"] = type(error).__name__ + ": " + str(error)[:120]
        results.append(row)
Path(__file__).with_suffix(".json").write_text(json.dumps({"sklearn": sklearn.__version__, "results": results}, ensure_ascii=False, indent=1), encoding="utf-8")
