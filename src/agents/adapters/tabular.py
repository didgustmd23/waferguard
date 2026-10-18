"""공통 표형 입력 계약과 EDA를 실행기에 연결하는 등록 도구."""

import csv
import json
from pathlib import Path

import pandas as pd

from src.preprocessing.contracts import TabularInput
from src.preprocessing.eda import analyze_summary
from src.preprocessing.eda_report import generate_eda_report
from src.preprocessing.profiles import DatasetProfile
from ..artifacts import _fingerprint, _is_link, _root
from ..contracts import ToolResult
from ..executor import RegisteredTool


REPORT_FILES = frozenset({
    "missing.csv", "numeric_statistics.csv", "categorical_distribution.csv",
    "target_distribution.csv", "pearson.csv", "spearman.csv", "correlation_support.csv",
    "missing.png", "distributions.png", "correlations.png",
    "summary.json", "eda.md", "SUCCESS.json",
})


# ==========================================
# 공통 EDA 어댑터
# - 로더는 파일 형식을 공통 TabularInput으로 변환한다.
# - 분석 계산은 preprocessing.eda 한 곳에서 수행한다.
# - 실제 로더 함수는 신뢰하는 코드만 등록하며 JSON에서 import하지 않는다.
# - 원본·프로파일 변경을 감지하면 기존 승인·결과를 재사용하지 않는다.
# ==========================================
class TabularEdaAdapter:
    def __init__(self, profile, loader, source_signature):
        if not isinstance(profile, DatasetProfile) or not callable(loader) or not callable(source_signature):
            raise TypeError("프로파일과 등록된 입력 로더·식별 함수가 필요합니다.")

        self.profile = profile
        self._loader = loader
        self._source_signature = source_signature
        self.source_id = source_signature()
        self.config_id = profile.signature

        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("원본 식별값이 필요합니다.")

    @property
    def arguments(self):
        return json.loads(self.profile.tool_arguments_json)

    def _check_input(self, arguments):
        if arguments != self.arguments:
            raise ValueError("EDA 인자가 검증된 프로파일과 일치하지 않습니다.")

        if self._source_signature() != self.source_id:
            raise ValueError("EDA 입력이 계획 생성 이후 변경되었습니다.")

    def load_input(self, *, validate=True):
        """어댑터가 프로파일과 다른 컬럼 역할을 전달하지 못하게 검사한다."""
        data = self._loader()

        if not isinstance(data, TabularInput) or data.roles != self.profile.roles:
            raise ValueError("어댑터 입력 계약이 프로파일과 일치하지 않습니다.")

        return data.validate() if validate else data

    def audit_tool(self):
        """동일 입력 로더와 프로파일로 품질 검사 도구를 연결한다."""
        from .quality import QualityAuditAdapter

        return QualityAuditAdapter(self).registered_tool()

    def split_tools(self):
        """검토용 분할 제안과 승인 필수 확정 도구를 연결한다."""
        from .split import TabularSplitAdapter

        return TabularSplitAdapter(self).registered_tools()

    def preparation_tool(self):
        """확정 분할에 의존하는 Train-only 학습 입력 준비 도구를 연결한다."""
        from .prepare import TabularPreparationAdapter

        return TabularPreparationAdapter(self).registered_tool()

    def finalization_tools(self):
        """학습 입력의 최종 계약 검사와 로컬 보고서 도구를 연결한다."""
        from .finalize import TabularFinalizationAdapter

        return TabularFinalizationAdapter(self).registered_tools()

    def registered_tool(self):
        return RegisteredTool("run_eda", self.execute, self.validate, requires_approval=True)

    def execute(self, output, arguments):
        """상세 통계는 로컬 파일에만 남기고 일반 완료 상태만 반환한다."""
        self._check_input(arguments)
        output = _root(output)
        data = self.load_input()
        result = generate_eda_report(data, output / "eda", max_plot_features=self.profile.max_plot_features)
        self._check_input(arguments)

        if result != {"status": "eda_complete"}:
            return ToolResult(status="blocked_input", errors=("EDA_NOT_COMPLETE",))

        return ToolResult(status="passed", artifact_refs=("eda",))

    def validate(self, output, arguments):
        self._check_input(arguments)
        report = _root(Path(output) / "eda")
        paths = tuple(report.iterdir())

        if {path.name for path in paths} != REPORT_FILES:
            return False

        if any(_is_link(path) or not path.is_file() or path.stat().st_size == 0 for path in paths):
            return False

        completion = json.loads((report / "SUCCESS.json").read_text(encoding="utf-8"))
        summary = json.loads((report / "summary.json").read_text(encoding="utf-8"))
        expected = analyze_summary(self.load_input())
        self._check_input(arguments)

        return completion == {"status": "eda_complete"} and summary == expected


def csv_eda_adapter(path, profile):
    """UTF-8 CSV 입력용 명시적 어댑터. pickle·자동 형식 추정·외부 URL은 지원하지 않는다."""
    if profile.adapter_id != "tabular_csv_v1":
        raise ValueError("CSV 어댑터와 프로파일 ID가 일치하지 않습니다.")

    source = Path(path).absolute()
    _root(source.parent)

    def signature():
        if not source.is_file() or _is_link(source):
            raise ValueError("안전한 로컬 CSV 파일이 필요합니다.")

        return "csv:" + _fingerprint(source)["sha256"]

    def load():
        # 특수 read_csv 옵션·사용자 converter 함수는 설정으로 받지 않는다.
        with source.open(encoding="utf-8-sig", newline="") as stream:
            header = next(csv.reader(stream), [])

        if not header or len(header) != len(set(header)):
            raise ValueError("CSV 헤더가 비어 있거나 중복 컬럼이 있습니다.")

        frame = pd.read_csv(source, encoding="utf-8-sig")

        return TabularInput(frame, profile.roles)

    return TabularEdaAdapter(profile, load, signature)
