"""참조 모듈의 합성 전용 실행 계약. 부모에서는 확장 모듈을 import하지 않는다."""

from dataclasses import asdict
import hashlib

from src.preprocessing.contracts import ColumnRoles
from src.preprocessing.profiles import DatasetProfile
from src.preprocessing.quality import QualityRules
from .module_catalogue import MODULE_ID, VERSION, DESCRIPTOR_SHA256, inspect_module
from .synthetic_eda_data import CSV_BYTES


POLICY_VERSION = "reviewed_module_demo_v1"


def demo_profile():
    return DatasetProfile("reviewed_module_demo", "tabular_csv_v1", "eda_review_v1",
                          ColumnRoles(numeric=("x", "y"), categorical=("category",), target="target"),
                          quality_rules=QualityRules(allowed_labels=(0, 1)))


def module_settings(include_complete):
    if type(include_complete) is not bool:
        raise ValueError("완전 관측 행 포함 여부는 bool이어야 합니다.")

    return {"include_complete": include_complete}


def check_demo(snapshot, profile, include_complete, *, staged=False):
    """해시가 다른 CSV·임의 프로파일은 같은 파일명이라도 받지 않는다."""
    module_settings(include_complete)
    inspect_module(MODULE_ID, VERSION, staged=staged)

    if (snapshot.sha256 != hashlib.sha256(CSV_BYTES).hexdigest()
            or snapshot.size_bytes != len(CSV_BYTES) or profile != demo_profile()):
        raise ValueError("검토된 모듈의 고정 합성 입력만 실행할 수 있습니다.")


def module_identity(include_complete):
    return {"module_id": MODULE_ID, "version": VERSION, "descriptor_sha256": DESCRIPTOR_SHA256,
            "settings": module_settings(include_complete), "synthetic_only": True}
