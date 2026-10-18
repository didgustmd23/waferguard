"""공통 품질 검사를 run_audit 등록 도구로 연결한다."""

import json
from pathlib import Path

from src.preprocessing.quality import (
    QualityFinding, QualityReport, inspect_quality, write_quality_report,
)
from ..artifacts import _is_link, _root
from ..contracts import ToolResult
from ..executor import RegisteredTool


class QualityAuditAdapter:
    """입력 어댑터의 로더·변경 감지·프로파일을 그대로 재사용한다."""

    def __init__(self, input_adapter):
        self.input_adapter = input_adapter

    def registered_tool(self):
        # 품질 검사는 이미 등록된 로컬 입력만 읽는다. 원본 역직렬화 권한은 부여하지 않는다.
        return RegisteredTool("run_audit", self.execute, self.validate)

    def execute(self, output, arguments):
        """검사 실패도 상세 근거를 보존하며 원문을 도구 반환값에 복사하지 않는다."""
        adapter = self.input_adapter
        adapter._check_input(arguments)
        try:
            data = adapter.load_input(validate=False)
        except (ValueError, TypeError):
            report = QualityReport(
                "blocked_input", (QualityFinding("INPUT_LOAD_CONTRACT_INVALID", "block"),),
            )
        else:
            report = inspect_quality(data, adapter.profile.quality_rules)
        write_quality_report(report, _root(output) / "quality")
        adapter._check_input(arguments)

        warnings = ("QUALITY_REVIEW_REQUIRED",) if report.status == "review_required" else ()
        errors = ("QUALITY_INPUT_BLOCKED",) if report.status == "blocked_input" else ()

        return ToolResult(
            status=report.status, artifact_refs=("quality",), warnings=warnings, errors=errors,
        )

    def validate(self, output, arguments):
        """통과 보고서만 완료 산출물로 등록하도록 내용과 현재 입력을 대조한다."""
        adapter = self.input_adapter
        adapter._check_input(arguments)
        report_dir = _root(Path(output) / "quality")
        paths = tuple(report_dir.iterdir())

        if {path.name for path in paths} != {"quality.json", "quality.md"}:
            return False

        if any(_is_link(path) or not path.is_file() or path.stat().st_size == 0 for path in paths):
            return False

        actual = json.loads((report_dir / "quality.json").read_text(encoding="utf-8"))
        expected = inspect_quality(adapter.load_input(validate=False), adapter.profile.quality_rules)
        adapter._check_input(arguments)

        return expected.status == "passed" and actual == expected.to_dict()
