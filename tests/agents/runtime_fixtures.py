"""레거시 CLI의 계산·승인 검사에만 쓰는 제한된 환경 모의."""

from types import SimpleNamespace
from unittest.mock import patch


def run_supported_preflight(manager, cli):
    """사전 검사 한 단계에서만 CLI 참조를 바꾸고 worker 호출 전 복구한다."""
    if manager.states.load().current_stage != "preflight":
        raise AssertionError("환경 모의는 preflight 단계에서만 사용할 수 있습니다.")

    with patch.object(cli, "os", SimpleNamespace(name="nt")), patch.object(
        cli, "sys", SimpleNamespace(version_info=(3, 10, 0))
    ):
        return manager.run_next()
