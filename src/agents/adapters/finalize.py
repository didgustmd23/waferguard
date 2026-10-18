"""독립 입력 계약 검사와 결정적 로컬 보고서 생성 도구."""

import json

from src.preprocessing.prepared_contract import inspect_prepared, read_local_json
from src.preprocessing.report import render_preparation_report
from ..artifacts import _fingerprint, _is_link, _root
from ..contracts import ToolResult
from ..executor import RegisteredTool


class TabularFinalizationAdapter:
    """계산은 공통 코어에 위임하고 원본 결합·파일 안전성·도구 계약만 관리한다."""

    def __init__(self, input_adapter):
        profile = input_adapter.profile

        if profile.adapter_id != "tabular_csv_v1" or profile.preparation_rules is None or profile.split_rules is None:
            raise ValueError("분할·준비 기준이 있는 표형 CSV 프로파일이 필요합니다.")

        self.input = input_adapter

    def registered_tools(self):
        return (
            RegisteredTool(
                "validate_contract", self.inspect, self.validate_inspection,
                dependency_stage="prepare_data",
            ),
            RegisteredTool(
                "render_report", self.render, self.validate_report,
                dependency_stage="validate_contract",
            ),
        )

    def _directory(self, directory):
        """선행 폴더의 링크·reparse 항목을 거부한다. OS 격리를 대신하지 않는다."""
        directory = _root(directory)

        if any(_is_link(path) or not path.is_file() for path in directory.iterdir()):
            raise ValueError("최종 검사 입력에는 안전한 일반 파일만 허용합니다.")

        return directory

    def _inspect(self, arguments, dependency):
        self.input._check_input(arguments)
        dependency = self._directory(dependency)
        result = inspect_prepared(
            dependency, self.input.load_input(), self.input.profile.preparation_rules,
            self.input.profile.split_rules, source_id=self.input.source_id, config_id=self.input.config_id,
        )
        self.input._check_input(arguments)

        return result

    def _context(self, dependency):
        """출처 경로는 신뢰하는 실행기가 전달한 경로로만 만든다. LLM 인자가 아니다."""
        dependency = self._directory(dependency)

        return {
            "schema_version": "report_context_v1", "source_id": self.input.source_id,
            "config_id": self.input.config_id, "prepared_artifact": str(dependency),
            "prepared_completion": _fingerprint(dependency / "AGENT_COMPLETE.json"),
        }

    def inspect(self, output, arguments, dependency):
        """최종 검사 결과를 별도 파일로 남긴다. 실패 결과에는 완료 표식을 붙이지 않는다."""
        output = _root(output)
        result = self._inspect(arguments, dependency)
        (output / "contract_check.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
        )

        if result["status"] != "passed":
            return ToolResult(status="blocked_input", errors=("PREPARED_CONTRACT_FAILED",))

        (output / "report_context.json").write_text(
            json.dumps(self._context(dependency), ensure_ascii=False, indent=2), encoding="utf-8",
        )

        return ToolResult(status="passed", artifact_refs=("contract_check.json", "report_context.json"))

    def validate_inspection(self, output, arguments, dependency):
        """검사 파일도 현재 원본·준비 산출물에서 재계산한 결과와 대조한다."""
        output = self._directory(output)

        if {path.name for path in output.iterdir()} - {"AGENT_COMPLETE.json"} != {"contract_check.json", "report_context.json"}:
            return False

        saved = read_local_json(output / "contract_check.json")

        return (
            saved["status"] == "passed" and saved == self._inspect(arguments, dependency)
            and read_local_json(output / "report_context.json") == self._context(dependency)
        )

    def _report_input(self, arguments, dependency):
        self.input._check_input(arguments)
        dependency = self._directory(dependency)
        contract = read_local_json(dependency / "contract_check.json")
        context = read_local_json(dependency / "report_context.json")

        if context.get("source_id") != self.input.source_id or context.get("config_id") != self.input.config_id:
            raise ValueError("보고서 출처가 현재 원본·설정과 일치하지 않습니다.")

        # 이 함수는 원본 특징을 다시 읽거나 LLM을 호출하지 않는다.
        markdown = render_preparation_report(contract)
        self.input._check_input(arguments)

        return contract, context, markdown

    def render(self, output, arguments, dependency):
        """검사 완료 결과를 고정 한국어 템플릿으로 작성한다. 자유문 해석은 하지 않는다."""
        output = _root(output)
        contract, context, markdown = self._report_input(arguments, dependency)
        (output / "report.md").write_text(markdown, encoding="utf-8")
        (output / "report.json").write_text(
            json.dumps(contract, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
        )
        (output / "report_context.json").write_text(
            json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8",
        )

        return ToolResult(status="passed", artifact_refs=("report.md", "report.json", "report_context.json"))

    def validate_report(self, output, arguments, dependency):
        """Markdown·기계 판독 요약·출처가 검증 결과의 결정적 렌더링과 일치하는지 확인한다."""
        output = self._directory(output)

        if {path.name for path in output.iterdir()} - {"AGENT_COMPLETE.json"} != {"report.md", "report.json", "report_context.json"}:
            return False

        contract, context, markdown = self._report_input(arguments, dependency)

        return (
            (output / "report.md").read_text(encoding="utf-8") == markdown
            and read_local_json(output / "report.json") == contract
            and read_local_json(output / "report_context.json") == context
        )
