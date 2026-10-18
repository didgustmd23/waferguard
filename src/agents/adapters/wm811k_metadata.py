"""WM canonical metadata를 공통 표형 EDA 입력으로 변환하는 어댑터."""

import hashlib
from pathlib import Path

from src.preprocessing.contracts import TabularInput
from src.preprocessing.profiles import load_profile
from .tabular import TabularEdaAdapter
from ..artifacts import _is_link, _root


METADATA_FILES = (
    "SUCCESS.json", "conversion.json", "provenance.json", "manifest.jsonl",
)


def _metadata_digest(root):
    """EDA가 읽는 관리 파일·manifest를 스트리밍 해싱한다. NPZ는 읽지 않는다."""
    _root(root)
    digest = hashlib.sha256()

    for name in METADATA_FILES:
        path = root / name

        if not path.is_file() or _is_link(path):
            raise ValueError("EDA 입력 관리 파일이 없거나 안전한 일반 파일이 아닙니다.")

        file_digest = hashlib.sha256()

        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                file_digest.update(chunk)

        # 파일 이름과 개별 고정 길이 digest를 함께 묶어 파일 경계를 구분한다.
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(file_digest.digest())

    return digest.hexdigest()


# ==========================================
# WM 전용 EDA 어댑터
# - WM 클래스·metadata 필드 해석은 wafer_dl 코어에 둔다.
# - 공통 실행기는 WM 모듈을 직접 import하지 않는다.
# - 최초 버전은 metadata 전용이며 예시 NPZ·원본 pickle을 읽지 않는다.
# - 결과 표·그림은 로컬 산출물에만 저장하고 일반 성공 상태만 반환한다.
# - 실제 CLI 도구 목록에는 아직 연결하지 않는다.
# ==========================================
class WmMetadataEdaAdapter(TabularEdaAdapter):
    """WM 읽기·적격 행 선택만 수행하고 공통 분석은 부모 어댑터에 위임한다."""

    def __init__(self, input_dir, profile=None):
        self.input_dir = _root(input_dir)

        if profile is None:
            path = Path(__file__).resolve().parents[3] / "configs/agents/profiles/wm811k_metadata.json"
            profile = load_profile(path)

        if profile.adapter_id != "wm811k_metadata_v1":
            raise ValueError("WM metadata 어댑터와 프로파일이 일치하지 않습니다.")

        def load():
            from ...wafer_dl.eda import load_eda_metadata

            frame = load_eda_metadata(self.input_dir)
            accepted = frame.loc[frame["quality_status"].eq("accepted")].copy()

            return TabularInput(accepted, profile.roles)

        def signature():
            return "wm_metadata:" + _metadata_digest(self.input_dir)

        super().__init__(profile, load, signature)
