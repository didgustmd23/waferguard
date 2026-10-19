"""WM의 현재 검증된 런타임 정책. UI Python과 worker 지원 여부를 구분한다."""

import os
from pathlib import Path
import sys


SUPPORTED_PYTHON = ((3, 10), (3, 12))
CPU_312_BUILD = '2.14.1+cpu'
SUPPORTED_312_BUILDS = frozenset({CPU_312_BUILD, '2.14.1+cu130'})
CPU_312_MODULES = frozenset({
    'wm_canonical_eda', 'wm_split_propose', 'wm_split_commit',
    'wm_transform_check', 'wm_cache_prepare', 'wm_dataset_check',
})


def _installed_torch_build():
    """현재 환경의 작은 버전 파일만 검사한다. torch import·GPU 초기화는 하지 않는다."""
    from importlib.metadata import distribution
    from .wm_dataset_runtime import pinned_torch_version
    from .wm_canonical_contract import file_hash

    prefix = Path(sys.prefix)
    name = 'Lib/site-packages/torch/version.py'
    record = file_hash(prefix / name, 65536)
    return pinned_torch_version(prefix, {name: record}, distribution('torch').version)


def runtime_status(module_id=None):
    """지원 정책 조회다. 실제 입력·승인·OS 실행 검증을 대신하지 않는다.

    3.10은 기존 정책을 유지한다. 3.12는 지정 빌드의 정상 개발 6단계에 한정한다.
    CUDA 빌드 설치와 worker의 CUDA 장치 사용은 별개다. 전처리는 CPU에서
    수행하며, 이 허용 목록은 해당 환경의 실제 AppContainer 검사 통과 증명이 아니다.
    """
    python = sys.version_info[:2]
    supported = os.name == 'nt' and python == (3, 10)
    build = None
    if os.name == 'nt' and python == (3, 12) and type(module_id) is str and module_id in CPU_312_MODULES:
        try:
            build = _installed_torch_build()
            supported = build in SUPPORTED_312_BUILDS
        except (OSError, ValueError, ImportError, SyntaxError, UnicodeError):
            supported = False

    return {"supported": supported, "code": "WM_RUNTIME_SUPPORTED" if supported else "WM_RUNTIME_UNVERIFIED",
            "python": list(sys.version_info[:2]), "supported_python": [list(item) for item in SUPPORTED_PYTHON],
            "module_id": module_id, "torch_build": build,
            "worker_device": "cpu",
            "notice": "3.12는 2.14.1+cpu 또는 2.14.1+cu130의 정상 개발 6단계를 허용합니다. CUDA 빌드에서도 전처리는 CPU 연산입니다. 실제 격리 호환성은 합성 검사로 확인하며 충돌 분기·GPU 장치 실행·비격리 대체 실행은 허용하지 않습니다."}
