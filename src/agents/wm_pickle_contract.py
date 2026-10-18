"""기존 WM import 경로 호환용. 실제 계약은 범용 pickle 계약에서 관리한다."""

from .pickle_contract import ADAPTER_ID, POLICY_VERSION, PickleInput, check_options, digest

WmPickleInput = PickleInput
