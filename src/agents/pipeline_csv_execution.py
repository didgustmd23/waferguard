"""CSV 어댑터를 공통 승인 단계에 연결하는 고정 실행 어댑터."""

import json

from .module_pipeline import ModulePipelineTool
from .module_pipeline_transport import code_manifest
from .reviewed_csv_pipeline_contract import ReviewedCsvPipelineBinding
from .snapshot_eda import _digest


class CsvPipelineExecution:
    """기존 격리 모듈 파이프라인을 공통 run_pipeline 도구 계약으로 감싼다."""

    def __init__(self, states, binding):
        if not isinstance(binding, ReviewedCsvPipelineBinding):
            raise TypeError("등록된 CSV 파이프라인 바인딩이 필요합니다.")

        self.binding = binding
        self.binding_arguments = {
            "binding_id": binding.binding_id,
            "binding": binding.to_dict(),
        }
        self.worker_tool = ModulePipelineTool(
            states,
            binding.input_contract,
            json.loads(binding.pipeline_config_json),
        )

        if self.worker_tool.module_plan.signature != binding.plan.signature:
            raise ValueError("승인된 모듈 계획과 격리 worker 계획이 일치하지 않습니다.")

        if self.worker_tool.code_id != binding.execution_code_id:
            raise ValueError("승인된 worker 코드와 현재 실행 코드가 일치하지 않습니다.")

    @property
    def worker_started(self):
        """격리 운송 계층이 실제 자식 프로세스 실행을 시작했는지 반환한다."""
        return self.worker_tool.worker_started

    def _check_binding(self, arguments):
        """실행 직전에도 승인된 입력·계획·코드 지문이 그대로인지 확인한다."""
        # 승인 계획은 JSON으로 저장되므로 tuple 값이 list로 복원될 수 있다.
        # JSON 표현으로 정규화한 뒤 비교해 자료형 표기 차이만 허용한다.
        serialized_arguments = json.dumps(
            arguments,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        serialized_binding = json.dumps(
            self.binding_arguments,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )

        if serialized_arguments != serialized_binding:
            raise ValueError("실행 요청이 승인된 바인딩과 다릅니다.")

        if _digest(code_manifest()) != self.binding.execution_code_id:
            raise ValueError("worker 코드가 승인 후 변경되었습니다.")

        self.binding.input_contract.check_input()

    def execute(self, output, arguments):
        """검증된 요청만 기존 AppContainer·Job 격리 worker로 전달한다."""
        self._check_binding(arguments)
        return self.worker_tool.execute(output, self.worker_tool.arguments)

    def validate(self, output, arguments):
        """기존 CSV 파이프라인 결과 검증을 재사용한다."""
        self._check_binding(arguments)
        return self.worker_tool.validate(output, self.worker_tool.arguments)


def build_csv_pipeline_execution(states, binding):
    """정적으로 등록된 CSV 실행 어댑터를 만든다. 사용자 지정 import는 하지 않는다."""
    return CsvPipelineExecution(states, binding)
