"""표형 계산 모듈과 WM 승인 작업을 함께 조회하는 에이전트 계층 등록부."""

from src.preprocessing.builtin_modules import build_builtin_registry
from src.preprocessing.preparation_modules import build_preparation_registry
from src.preprocessing.module_contracts import check_id
from .wm_module_catalogue import wm_modules


class AgentModuleRegistry:
    """기존 코어 등록부를 변경하지 않고 검토된 모듈을 하나의 조회 목록으로 묶는다.

    계산 모듈과 여러 단계의 승인 작업은 실행 방식이 다르다. 계약 조회는
    통합하지만 WM 작업을 CSV worker의 calculate 호출 경로에 추가하지 않는다.
    등록은 고정 초기화에서만 수행하며 구성 후에는 봉인한다.
    """

    def __init__(self):
        self._modules = {}
        self._sealed = False

    def register(self, module):
        if self._sealed:
            raise ValueError("봉인된 에이전트 등록부는 변경할 수 없습니다.")

        from src.preprocessing.module_registry import RegisteredModule
        from .wm_module_catalogue import WmWorkflowModule

        if not isinstance(module, (RegisteredModule, WmWorkflowModule)):
            raise TypeError("검토된 계산 모듈 또는 WM 승인 작업만 등록할 수 있습니다.")

        key = (module.contract.module_id, module.contract.version)

        if key in self._modules:
            raise ValueError("동일 ID·버전의 모듈을 덮어쓸 수 없습니다.")

        module.validate_settings()
        self._modules[key] = module

    def seal(self):
        self._sealed = True

        return self

    def resolve(self, module_id, version):
        check_id(module_id)

        if type(version) is not str:
            raise ValueError("모듈 버전을 문자열로 명시하세요.")

        module = self._modules.get((module_id, version))

        if module is None:
            raise ValueError("등록되지 않은 모듈 ID 또는 버전입니다.")

        return module

    def contracts(self):
        return tuple(module.contract for _, module in sorted(self._modules.items()))


def build_agent_registry():
    """내장 EDA·품질 6개, 표형 변환 2개, WM 작업 8개를 고정 등록한다."""
    registry = AgentModuleRegistry()

    for source in (build_builtin_registry(), build_preparation_registry()):
        for contract in source.contracts():
            registry.register(source.resolve(contract.module_id, contract.version))

    for module in wm_modules():
        registry.register(module)

    return registry.seal()
