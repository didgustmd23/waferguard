"""검토된 Python 구현을 고정 ID에 연결하는 등록부. 임의 import·실행 관리 기능은 없다."""

from dataclasses import dataclass
from typing import Callable

from .module_contracts import ModuleContract, check_id


@dataclass(frozen=True)
class RegisteredModule:
    """계약·계산 함수·설정 검증기·결과 검증기의 검토된 연결."""

    contract: ModuleContract
    calculate: Callable
    settings_validator: Callable
    result_validator: Callable

    def __post_init__(self):
        if not isinstance(self.contract, ModuleContract):
            raise TypeError("검증된 ModuleContract가 필요합니다.")

        if not all(callable(value) for value in (self.calculate, self.settings_validator, self.result_validator)):
            raise ValueError("계산·설정 검증·결과 검증 함수를 모두 등록해야 합니다.")

    def validate_settings(self, settings=None):
        """허용된 키만 복사해 값 검증기로 전달한다. 입력 설정을 변경하지 않는다."""
        settings = {} if settings is None else settings

        if type(settings) is not dict or any(type(key) is not str for key in settings):
            raise ValueError("모듈 설정은 문자열 키를 가진 dict여야 합니다.")

        if set(settings) - set(self.contract.settings_fields):
            raise ValueError("모듈에 허용되지 않은 설정이 있습니다.")

        return self.settings_validator(dict(settings))


class ModuleRegistry:
    """한 번 구성한 목록을 봉인한다. 사용자 코드 등록 승인을 대신하지 않는다."""

    def __init__(self):
        self._modules = {}
        self._sealed = False

    def register(self, module):
        """동일 ID·버전은 덮어쓰지 않는다. 등록은 신뢰된 초기화 코드에서만 한다."""
        if self._sealed:
            raise ValueError("봉인된 모듈 등록부는 변경할 수 없습니다.")

        if not isinstance(module, RegisteredModule):
            raise TypeError("RegisteredModule 객체가 필요합니다.")

        key = (module.contract.module_id, module.contract.version)

        if key in self._modules:
            raise ValueError("동일 ID·버전의 모듈이 이미 등록되어 있습니다.")

        # 기본 설정도 실제 스키마 검증을 통과해야 등록할 수 있다.
        module.validate_settings()
        self._modules[key] = module

    def seal(self):
        self._sealed = True

        return self

    def resolve(self, module_id, version):
        """최신 버전 자동 선택이나 미등록 함수 경로 해석 없이 정확한 버전만 조회한다."""
        check_id(module_id)

        if type(version) is not str:
            raise ValueError("모듈 버전은 문자열로 명시해야 합니다.")

        module = self._modules.get((module_id, version))

        if module is None:
            raise ValueError("등록되지 않은 모듈 ID 또는 지원하지 않는 버전입니다.")

        return module

    def contracts(self):
        """데이터·함수 객체를 제외한 불변 계약 목록을 반환한다."""
        return tuple(module.contract for _, module in sorted(self._modules.items()))
