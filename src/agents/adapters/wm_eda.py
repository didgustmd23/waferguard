"""기존 import 경로 호환용. 신규 코드는 wm811k_metadata를 사용한다."""

from .wm811k_metadata import WmMetadataEdaAdapter

__all__ = ["WmMetadataEdaAdapter"]
