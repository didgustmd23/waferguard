"""공통 분할 코어와 제안 → 승인 → 저장 단계를 연결하는 로컬 어댑터."""

import json
from pathlib import Path

import pandas as pd

from src.preprocessing.split import SPLITS, propose_split, validate_proposal
from ..artifacts import _is_link, _root
from ..contracts import ToolResult
from ..executor import RegisteredTool


class TabularSplitAdapter:
    """등록 입력 로더를 재사용한다. 분할 중 원본·특징·라벨은 수정하지 않는다."""

    def __init__(self, input_adapter):
        if input_adapter.profile.split_rules is None:
            raise ValueError("분할 기준이 있는 프로파일이 필요합니다.")

        # WM의 Lot·동일 맵 연결 그룹은 별도 도메인 계약이다.
        # 메타데이터의 Lot 컬럼만으로 기존 WM 분할을 대체하지 않는다.
        if input_adapter.profile.adapter_id != "tabular_csv_v1":
            raise ValueError("현재 공통 분할 연결은 표형 CSV 어댑터만 지원합니다.")

        self.input = input_adapter

    def registered_tools(self):
        return (
            RegisteredTool("propose_split", self.propose, self.validate_preview),
            RegisteredTool(
                "commit_split", self.commit, self.validate_commit,
                requires_approval=True, dependency_stage="propose_split",
            ),
        )

    def _proposal(self, arguments):
        """매번 원본 식별값과 역할을 대조한다. 재계산은 동일 시드·설정으로만 수행한다."""
        self.input._check_input(arguments)
        proposal = propose_split(
            self.input.load_input(), self.input.profile.split_rules,
            source_id=self.input.source_id, config_id=self.input.config_id,
        )
        self.input._check_input(arguments)

        return proposal

    def propose(self, output, arguments):
        """상세 분포·위치 목록은 로컬 검토 파일로만 기록한다."""
        output = _root(output)

        try:
            proposal = self._proposal(arguments)
        except (ValueError, TypeError):
            # 입력 조건 실패를 실행 장애와 구분한다. 원본 예외나 행은 기록하지 않는다.
            diagnosis = {"status": "blocked_input", "code": "SPLIT_CONSTRAINTS_UNMET"}
            (output / "split_check.json").write_text(json.dumps(diagnosis), encoding="utf-8")
            (output / "proposal.md").write_text(
                "# 분할 제안 중단\n\n입력 역할·표본 수·그룹·시간 경계와 분할 설정을 검토하세요.\n"
                "행 삭제·다른 시드·다른 분할 방식으로 자동 우회하지 않았습니다.\n",
                encoding="utf-8",
            )

            return ToolResult(status="blocked_input", errors=("SPLIT_CONSTRAINTS_UNMET",))

        (output / "proposal.json").write_text(
            json.dumps(proposal, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
        )
        lines = [
            "# 분할 제안", "", f"방식: `{proposal['rules']['strategy']}`",
            f"상태: `{proposal['status']}`", f"분할안 식별값: `{proposal['proposal_id']}`", "",
            "| 구간 | 행 수 | 실제 비율 |", "| --- | ---: | ---: |",
        ]
        lines.extend(f"| {row['split']} | {row['rows']} | {row['ratio']:.2%} |" for row in proposal["summary"])
        lines.extend([
            "", "클래스별 개수와 정확한 위치 목록은 proposal.json에서 로컬로 확인하세요.",
            "Group은 그룹 수 기준 무작위 배치이므로 행 비율은 목표와 다를 수 있습니다.",
            "Time은 동일 시각을 나누지 않으며 그룹 역할이 있다면 그룹 교차도 거부합니다.",
            "입력 계약에 선언하지 않은 중복 개체·도메인 연결 관계까지 보장하지 않습니다.",
            "아직 분할을 확정하거나 학습 입력 변환을 수행하지 않았습니다.",
            "상세 데이터·통계·식별값은 LLM으로 전송하지 마세요.",
        ])
        (output / "proposal.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

        return ToolResult(status=proposal["status"], artifact_refs=("proposal.json", "proposal.md"))

    def _read(self, output, filename):
        path = _root(output) / filename

        if _is_link(path) or not path.is_file():
            raise ValueError("안전한 분할 산출물 파일이 필요합니다.")

        return json.loads(path.read_text(encoding="utf-8"))

    def validate_preview(self, output, arguments):
        proposal = self._read(output, "proposal.json")

        return proposal == self._proposal(arguments) and proposal["status"] == "passed"

    def commit(self, output, arguments, dependency):
        """실행기가 검증한 제안만 저장한다. 새 분할·자동 재시드·특징 학습은 하지 않는다."""
        output = _root(output)
        proposal = self._read(dependency, "proposal.json")
        expected = self._proposal(arguments)

        if proposal != expected or proposal["status"] != "passed":
            raise ValueError("승인 대상 분할안이 현재 입력·설정과 일치하지 않습니다.")

        rows = [
            {"row_position": position, "split": name}
            for name in SPLITS for position in proposal["partitions"][name]
        ]
        pd.DataFrame(rows).sort_values("row_position").to_csv(output / "assignments.csv", index=False)
        (output / "split.json").write_text(
            json.dumps(proposal, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
        )
        self.input._check_input(arguments)

        return ToolResult(status="passed", artifact_refs=("assignments.csv", "split.json"))

    def validate_commit(self, output, arguments, dependency):
        proposal = self._read(dependency, "proposal.json")
        data = self.input.load_input()
        self.input._check_input(arguments)
        validate_proposal(
            proposal, data, self.input.profile.split_rules,
            source_id=self.input.source_id, config_id=self.input.config_id,
        )

        if self._read(output, "split.json") != proposal:
            return False

        path = _root(output) / "assignments.csv"

        if _is_link(path) or not path.is_file():
            return False

        actual = pd.read_csv(path)
        expected = pd.DataFrame([
            {"row_position": position, "split": name}
            for name in SPLITS for position in proposal["partitions"][name]
        ]).sort_values("row_position").reset_index(drop=True)
        self.input._check_input(arguments)

        return actual.equals(expected)
