"""사용자가 지정한 컬럼 역할·기준으로 기존 프로파일 계약 JSON을 만들고 새 로컬 파일로만 저장한다.

역할을 추론하지 않는다. 컬럼 목록은 부모 검증을 통과한 입력 진단 결과에서만 받는다.
저장은 실행 승인·데이터 검사가 아니며 컬럼 실재·자료형의 최종 판정은 이후 격리 worker가 한다.
"""

import json
import os
from pathlib import Path

from src.preprocessing.module_contracts import check_id
from src.preprocessing.profiles import load_profile, profile_from_dict
from .llm_proposal_store import _store_root
from .worker_paths import _path


STORE_ROOT = Path(__file__).resolve().parents[2] / "logs/agents/profiles/ui"
MAX_PROFILE_BYTES = 64 * 1024
EXCLUDED = "제외"
LIST_ROLES = {"수치형": "numeric", "범주형": "categorical", "식별자": "identifiers"}
SINGLE_ROLES = {"target": "target", "group": "group", "timestamp": "timestamp"}
ROLE_LABELS = (EXCLUDED, *LIST_ROLES, *SINGLE_ROLES)
PURPOSES = {"eda": "eda_review_v1", "split": "split_review_v1"}
QUALITY_FIELDS = {"missing_ratio_threshold", "high_missing"}
# worker_paths._path와 같은 Windows 예약 장치 이름. 저장 전에 거부해 파일을 남기지 않는다.
RESERVED_NAMES = {"con", "prn", "aux", "nul", *(f"{prefix}{number}" for prefix in ("com", "lpt") for number in range(1, 10))}
SPLIT_FIELDS = {"strategy", "train_ratio", "validation_ratio", "test_ratio", "random_state", "stratify", "require_all_classes"}


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def build_profile(columns, roles, *, purpose, profile_id, max_plot_features=12, quality=None, split=None):
    """역할 표를 기존 프로파일 계약으로 변환한다. 최종 검증은 profile_from_dict가 한다."""
    if type(columns) is not tuple or not columns or len(set(columns)) != len(columns):
        raise ValueError("검증된 진단 결과의 컬럼 목록이 필요합니다.")

    if type(roles) is not dict or set(roles) != set(columns) or any(value not in ROLE_LABELS for value in roles.values()):
        raise ValueError("모든 컬럼에 허용된 역할을 하나씩 지정해야 합니다.")

    if purpose not in PURPOSES:
        raise ValueError("지원하지 않는 프로파일 목적입니다.")

    assigned = {"numeric": [], "categorical": [], "identifiers": [], "target": None, "group": None, "timestamp": None}

    # 진단 결과의 컬럼 순서를 유지한다. 제외 컬럼은 어느 목록에도 쓰지 않는다.
    for name in columns:
        label = roles[name]

        if label in LIST_ROLES:
            assigned[LIST_ROLES[label]].append(name)
        elif label in SINGLE_ROLES:
            key = SINGLE_ROLES[label]

            if assigned[key] is not None:
                raise ValueError("target·group·timestamp 역할은 각각 한 컬럼에만 지정할 수 있습니다.")

            assigned[key] = name

    value = {
        "schema_version": "dataset_profile_v1", "profile_id": profile_id, "adapter_id": "tabular_csv_v1",
        "workflow_id": PURPOSES[purpose], "roles": assigned, "max_plot_features": max_plot_features,
    }

    if quality is not None:
        if type(quality) is not dict or set(quality) - QUALITY_FIELDS:
            raise ValueError("품질 기준은 결측률 기준과 통지 수준만 지정할 수 있습니다.")

        value["quality_rules"] = dict(quality)

    if purpose == "split":
        if type(split) is not dict or set(split) - SPLIT_FIELDS:
            raise ValueError("분할 목적에는 허용된 분할 기준이 필요합니다.")

        value["split_rules"] = dict(split)
    elif split is not None:
        raise ValueError("EDA 목적에는 분할 기준을 지정하지 않습니다.")

    return profile_from_dict(value)


def save_profile(profile, *, root=None):
    """profile_id를 파일 이름으로 새 파일만 배타 생성하고 기존 읽기 경로로 다시 검증한다."""
    root = STORE_ROOT if root is None else root
    check_id(profile.profile_id)

    if profile.profile_id in RESERVED_NAMES:
        raise ValueError("Windows 예약 이름은 프로파일 이름으로 쓸 수 없습니다.")
    payload = _canonical(profile.to_dict()).encode("utf-8")

    if len(payload) > MAX_PROFILE_BYTES:
        raise ValueError("프로파일은 64 KiB 이하여야 합니다.")

    target = _store_root(root, create=True) / f"{profile.profile_id}.json"

    # 같은 이름은 덮어쓰지 않는다. 중단으로 생긴 불완전 파일도 재적재 검사에서 거부된다.
    with target.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())

    restored = load_saved_profile(target, root=root)

    if restored.signature != profile.signature:
        raise ValueError("저장한 프로파일을 다시 읽은 결과가 다릅니다.")

    return {"status": "profile_saved", "name": profile.profile_id, "local_file": str(target),
            "profile_signature": restored.signature, "workflow_id": restored.workflow_id,
            "execution_approved": False, "worker_started": False}


def load_saved_profile(path, *, root=None):
    """저장소 안의 일반 파일만 기존 load_profile(크기·중복 키 검사)로 읽는다."""
    root = STORE_ROOT if root is None else root
    store = _store_root(root, create=False)
    local = _path(Path(path).absolute(), directory=False)

    if local.parent != store or local.suffix != ".json":
        raise ValueError("UI 프로파일 저장소의 파일이 아닙니다.")

    return load_profile(local)
