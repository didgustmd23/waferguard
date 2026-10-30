"""그룹 수가 적을 때 2단계 GroupShuffleSplit 결과(합성 배열, 제품 코드 미사용).

제품 split._partition의 group 경로와 같은 호출 순서를 sklearn으로 직접 재현한다: Train을 train_size로 먼저 떼고,
남은 행을 test/(validation+test) 비율로 다시 나눈다. 행 내용은 쓰지 않고 그룹 번호만 쓴다.
"""
import json
from pathlib import Path
import numpy as np
import sklearn
from sklearn.model_selection import GroupShuffleSplit

results = []
for n_groups in (2, 3, 4, 5, 6, 8, 10, 20):
    groups = np.repeat(np.arange(n_groups), 10)
    positions = np.arange(len(groups))
    row = {"n_groups": n_groups, "rows_per_group": 10, "ratios": [0.7, 0.15, 0.15], "random_state": 42}
    try:
        first = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=42)
        train, remaining = next(first.split(positions, groups=groups))
        second = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
        valid_local, test_local = next(second.split(remaining, groups=groups[remaining]))
        valid, test = remaining[valid_local], remaining[test_local]
        row["groups_per_split"] = [len(set(groups[x])) for x in (train, valid, test)]
        row["empty_split"] = any(len(x) == 0 for x in (train, valid, test))
    except ValueError as error:
        row["error"] = type(error).__name__ + ": " + str(error)[:160]
    results.append(row)
Path(__file__).with_suffix('.json').write_text(json.dumps({"sklearn": sklearn.__version__, "results": results}, ensure_ascii=False, indent=1), encoding='utf-8')
