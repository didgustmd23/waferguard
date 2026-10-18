"""프로파일의 어댑터 ID를 검토된 코드에만 연결한다."""

from src.preprocessing.profiles import DatasetProfile


ADAPTER_IDS = frozenset({"tabular_csv_v1", "wm811k_metadata_v1"})


def create_eda_adapter(profile, input_path):
    """임의 모듈 import나 자동 어댑터 추측 없이 등록 ID를 대조한다."""
    if not isinstance(profile, DatasetProfile):
        raise TypeError("검증된 DatasetProfile이 필요합니다.")

    if profile.adapter_id == "tabular_csv_v1":
        from .tabular import csv_eda_adapter

        return csv_eda_adapter(input_path, profile)

    if profile.adapter_id == "wm811k_metadata_v1":
        from .wm811k_metadata import WmMetadataEdaAdapter

        return WmMetadataEdaAdapter(input_path, profile)

    raise ValueError("코드에 등록되지 않은 데이터 어댑터입니다.")
