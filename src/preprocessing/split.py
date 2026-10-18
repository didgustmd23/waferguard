"""명시적 역할과 설정에 따른 범용 분할 제안·검증. 학습 통계는 계산하지 않는다."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .contracts import TabularInput


SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class SplitRules:
    """비율은 행 수 목표다. Group·Time 경계를 보존하면 실제 비율은 달라질 수 있다."""

    strategy: str = "random"
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    random_state: int = 42
    stratify: bool = False
    require_all_classes: bool = True
    timestamp_format: str | None = None
    algorithm_version: str = "tabular_split_v1"

    def __post_init__(self):
        ratios = (self.train_ratio, self.validation_ratio, self.test_ratio)

        if any(type(value) not in (int, float) or not math.isfinite(value) or not 0 < value < 1 for value in ratios):
            raise ValueError("분할 비율은 0보다 크고 1보다 작은 유한수여야 합니다.")

        if not math.isclose(sum(ratios), 1.0, abs_tol=1e-12):
            raise ValueError("Train·Validation·Test 비율의 합은 1이어야 합니다.")

        if self.strategy not in {"random", "group", "time"}:
            raise ValueError("분할 방식은 random·group·time 중 하나여야 합니다.")

        if type(self.random_state) is not int or not 0 <= self.random_state < 2**32:
            raise ValueError("분할 난수 시드는 32비트 범위의 음이 아닌 정수여야 합니다.")

        if type(self.stratify) is not bool or type(self.require_all_classes) is not bool:
            raise ValueError("계층 분할·클래스 조건은 bool이어야 합니다.")

        if self.stratify and self.strategy != "random":
            raise ValueError("stratify는 Random 분할에만 적용할 수 있습니다.")

        if self.timestamp_format is not None and (not isinstance(self.timestamp_format, str) or not self.timestamp_format.strip()):
            raise ValueError("시간 형식은 비어 있지 않은 문자열 또는 null이어야 합니다.")

        if self.algorithm_version != "tabular_split_v1":
            raise ValueError("지원하지 않는 분할 알고리즘 버전입니다.")


def _digest(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _required_values(series):
    """빈 문자열도 결측으로 판단한다. 임의 보정·삭제는 하지 않는다."""
    missing = series.isna() | series.map(lambda value: isinstance(value, str) and not value.strip())

    if missing.any():
        raise ValueError("분할에 필요한 역할 컬럼에 결측값이 있습니다.")


def _timestamps(data, rules):
    column = data.roles.timestamp

    if column is None:
        raise ValueError("Time 분할에는 timestamp 역할이 필요합니다.")

    values = data.frame[column]
    _required_values(values)

    if not pd.api.types.is_datetime64_any_dtype(values) and rules.timestamp_format is None:
        raise ValueError("문자열 시간에는 명시적인 timestamp_format이 필요합니다.")

    parsed = pd.to_datetime(values, format=rules.timestamp_format, errors="raise", utc=True)

    if parsed.isna().any():
        raise ValueError("유효한 시간 값이 필요합니다.")

    return parsed


# ==========================================
# 고정 규칙으로 Train·Validation·Test 위치 목록 생성
# - Random: 명시한 경우에만 target 비율을 보존한다.
# - Group: 같은 그룹의 행을 하나의 구간에만 배치한다.
# - Time: 같은 시각을 나누지 않고 과거 → 미래 순서를 보존한다.
# - 입력의 index 대신 원본 행 위치를 사용하여 중복 index와 구분한다.
# - 실패 시 다른 시드·방식으로 자동 재시도하지 않는다.
# ==========================================
def _partition(data, rules):
    frame = data.frame
    positions = np.arange(len(frame))

    if rules.strategy == "random":
        if data.roles.group is not None:
            # 그룹 역할이 있는 입력을 행 단위로 섞어 누수 위험을 숨기지 않는다.
            raise ValueError("group 역할이 있는 입력은 Group 분할을 사용하세요.")

        labels = frame[data.roles.target] if rules.stratify and data.roles.target else None

        if rules.stratify and labels is None:
            raise ValueError("stratify를 사용하려면 target 역할이 필요합니다.")

        train, remaining = train_test_split(
            positions, train_size=rules.train_ratio,
            random_state=rules.random_state, stratify=labels,
        )
        remaining_labels = labels.iloc[remaining] if labels is not None else None
        valid, test = train_test_split(
            remaining, test_size=rules.test_ratio / (rules.validation_ratio + rules.test_ratio),
            random_state=rules.random_state, stratify=remaining_labels,
        )

    elif rules.strategy == "group":
        if data.roles.group is None:
            raise ValueError("Group 분할에는 group 역할이 필요합니다.")

        groups = frame[data.roles.group]
        _required_values(groups)
        codes, _ = pd.factorize(groups, sort=False)

        first = GroupShuffleSplit(n_splits=1, train_size=rules.train_ratio, random_state=rules.random_state)
        train, remaining = next(first.split(positions, groups=codes))
        second = GroupShuffleSplit(
            n_splits=1, test_size=rules.test_ratio / (rules.validation_ratio + rules.test_ratio),
            random_state=rules.random_state,
        )
        valid_local, test_local = next(second.split(remaining, groups=codes[remaining]))
        valid, test = remaining[valid_local], remaining[test_local]

    else:
        times = _timestamps(data, rules)
        ordered = np.argsort(times.to_numpy(), kind="stable")
        ordered_times = times.iloc[ordered].to_numpy()
        boundaries = np.flatnonzero(ordered_times[1:] != ordered_times[:-1]) + 1

        if len(boundaries) < 2:
            raise ValueError("Time 분할에는 서로 다른 시각 경계가 두 개 이상 필요합니다.")

        # 세 구간을 비우지 않는 경계만 선택한다. 같은 시각은 절대 쪼개지 않는다.
        first = int(min(boundaries[:-1], key=lambda point: (abs(point / len(frame) - rules.train_ratio), point)))
        later = boundaries[boundaries > first]
        second = int(min(later, key=lambda point: (abs(point / len(frame) - rules.train_ratio - rules.validation_ratio), point)))
        train, valid, test = ordered[:first], ordered[first:second], ordered[second:]

    return {name: sorted(int(value) for value in indices) for name, indices in zip(SPLITS, (train, valid, test))}


def _check_boundaries(data, rules, partitions):
    """완전성·상호 배타성·그룹 및 시간 경계를 생성 로직과 별도로 검사한다."""
    flattened = [position for name in SPLITS for position in partitions[name]]

    if any(not partitions[name] for name in SPLITS) or sorted(flattened) != list(range(len(data.frame))):
        raise ValueError("분할에 빈 구간·행 중복·누락이 있습니다.")

    if data.roles.group is not None:
        groups = [set(data.frame.iloc[partitions[name]][data.roles.group]) for name in SPLITS]

        if any(groups[left] & groups[right] for left, right in ((0, 1), (0, 2), (1, 2))):
            raise ValueError("동일 그룹이 여러 분할 구간에 포함됩니다.")

    if rules.strategy == "time":
        times = _timestamps(data, rules)
        bounds = [(times.iloc[partitions[name]].min(), times.iloc[partitions[name]].max()) for name in SPLITS]

        if not (bounds[0][1] < bounds[1][0] and bounds[1][1] < bounds[2][0]):
            raise ValueError("시간 경계가 중복되거나 과거·미래 순서를 위반합니다.")


def propose_split(data, rules, *, source_id, config_id):
    """원본을 변경하지 않고 로컬 검토용 분할안·분포·자동 식별값을 만든다."""
    if not isinstance(data, TabularInput) or not isinstance(rules, SplitRules):
        raise TypeError("공통 입력 계약과 SplitRules가 필요합니다.")

    data.validate()

    if any(not isinstance(value, str) or not value.strip() for value in (source_id, config_id)):
        raise ValueError("원본·설정 식별값이 필요합니다.")

    if data.roles.target is not None:
        _required_values(data.frame[data.roles.target])

    if data.roles.group is not None:
        _required_values(data.frame[data.roles.group])

    if data.roles.identifiers:
        for column in data.roles.identifiers:
            _required_values(data.frame[column])

        if data.frame.duplicated(subset=list(data.roles.identifiers)).any():
            raise ValueError("중복 식별자 행은 먼저 검토해야 합니다.")

    partitions = _partition(data, rules)
    _check_boundaries(data, rules, partitions)
    summary = []
    missing_classes = False
    labels = data.frame[data.roles.target] if data.roles.target else None
    all_labels = set(labels) if labels is not None else set()

    for name in SPLITS:
        subset = data.frame.iloc[partitions[name]]
        counts = []

        if labels is not None:
            missing_classes |= set(subset[data.roles.target]) != all_labels

            for label, count in subset[data.roles.target].value_counts(sort=False).items():
                counts.append({"label": label.item() if isinstance(label, np.generic) else label, "count": int(count)})

        row = {"split": name, "rows": len(subset), "ratio": len(subset) / len(data.frame), "classes": counts}

        if data.roles.group is not None:
            row["group_count"] = int(subset[data.roles.group].nunique())

        if rules.strategy == "time":
            times = _timestamps(data, rules).iloc[partitions[name]]
            row["time_min"] = times.min().isoformat()
            row["time_max"] = times.max().isoformat()

        summary.append(row)

    result = {
        "schema_version": "split_proposal_v1", "source_id": source_id, "config_id": config_id,
        "rules": asdict(rules), "partitions": partitions, "summary": summary,
        "status": "review_required" if missing_classes and rules.require_all_classes else "passed",
        "warnings": ["CLASS_MISSING_IN_SPLIT"] if missing_classes else [],
    }
    result["proposal_id"] = _digest(result)

    return result


def validate_proposal(proposal, data, rules, *, source_id, config_id):
    """승인 대상 분할안을 같은 규칙으로 재계산하여 대조한다. 다른 시드로 탐색하지 않는다."""
    expected = propose_split(data, rules, source_id=source_id, config_id=config_id)

    if proposal != expected or expected["status"] != "passed":
        raise ValueError("분할안이 현재 원본·설정·검증 결과와 일치하지 않습니다.")

    return True
