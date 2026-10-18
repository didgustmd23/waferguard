"""승인·확정된 표형 분할을 Train-only 전처리 코어에 연결한다."""

from dataclasses import asdict
import json

import numpy as np
import pandas as pd

from src.preprocessing.split import validate_proposal
from src.preprocessing.transform import load_preparation, prepare_splits, save_preparation
from ..artifacts import _is_link, _root
from ..contracts import ToolResult
from ..executor import RegisteredTool


OUTPUT_FILES = frozenset({
    "preprocessor.json", "preparation.json", "preparation.md",
    "train_features.csv", "validation_features.csv", "row_contract.json", "split_contract.json",
})


class TabularPreparationAdapter:
    def __init__(self, input_adapter):
        profile = input_adapter.profile

        if profile.adapter_id != "tabular_csv_v1" or profile.split_rules is None or profile.preparation_rules is None:
            raise ValueError("분할·전처리 기준이 있는 표형 CSV 프로파일이 필요합니다.")

        self.input = input_adapter

    def registered_tool(self):
        return RegisteredTool(
            "prepare_data", self.execute, self.validate,
            requires_approval=True, dependency_stage="commit_split",
        )

    def _source(self, arguments, dependency):
        """실행기가 검증한 확정 분할을 읽고 현재 원본·프로파일과 다시 대조한다."""
        self.input._check_input(arguments)
        path = _root(dependency) / "split.json"

        if _is_link(path) or not path.is_file():
            raise ValueError("확정 분할 파일이 필요합니다.")

        proposal = json.loads(path.read_text(encoding="utf-8"))
        data = self.input.load_input()
        validate_proposal(
            proposal, data, self.input.profile.split_rules,
            source_id=self.input.source_id, config_id=self.input.config_id,
        )
        self.input._check_input(arguments)

        return data, proposal

    def _rows(self, data, proposal):
        """특징과 분리된 원본 행 위치·정답 계약. Test 정답은 복사하지 않는다."""
        result = {}

        for name in ("train", "validation"):
            positions = proposal["partitions"][name]
            row = {"row_positions": positions}

            if data.roles.target is not None:
                labels = data.frame.iloc[positions][data.roles.target].tolist()
                row["targets"] = [value.item() if isinstance(value, np.generic) else value for value in labels]

            result[name] = row

        return result

    def execute(self, output, arguments, dependency):
        """Train에서 한 번만 fit하고 Train·Validation 특징과 변환 계약을 저장한다."""
        output = _root(output)
        data, proposal = self._source(arguments, dependency)

        try:
            pipeline, outputs = prepare_splits(
                data, proposal, self.input.profile.split_rules, self.input.profile.preparation_rules,
                source_id=self.input.source_id, config_id=self.input.config_id,
            )
        except (ValueError, TypeError):
            # 새 범주·전체 결측·출력 상한 같은 입력 조건 실패를 실행 장애와 구분한다.
            # 상세 예외·값을 상태에 복사하거나 처리 정책을 자동 완화하지 않는다.
            (output / "preparation_check.json").write_text(json.dumps({
                "status": "blocked_input", "code": "PREPARATION_CONSTRAINTS_UNMET",
            }), encoding="utf-8")

            return ToolResult(status="blocked_input", errors=("PREPARATION_CONSTRAINTS_UNMET",))

        save_preparation(pipeline, output / "preprocessor.json")
        (output / "split_contract.json").write_text(
            json.dumps(proposal, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
        )

        for name, features in outputs.items():
            features.to_csv(output / f"{name}_features.csv", index=False)

        transformer = pipeline.named_steps["preprocess"]
        report = {
            "status": "preparation_complete", "source_id": self.input.source_id,
            "config_id": self.input.config_id, "proposal_id": proposal["proposal_id"],
            "rules": asdict(self.input.profile.preparation_rules),
            "train_rows": transformer.state_["train_rows"],
            "input_features": len(data.roles.feature_columns),
            "output_features": len(transformer.get_feature_names_out()),
            "removed": transformer.state_["removed"],
            "diagnostics": {
                name: transformer.diagnose(data.frame.iloc[proposal["partitions"][name]])
                for name in ("train", "validation")
            },
            "prepared_splits": ["train", "validation"],
            "test_transformed": False, "model_trained": False,
        }
        (output / "preparation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "row_contract.json").write_text(json.dumps(self._rows(data, proposal), ensure_ascii=False), encoding="utf-8")
        (output / "preparation.md").write_text(
            "# 학습 입력 전처리\n\n"
            f"Train {report['train_rows']}행에서만 fit했습니다.\n\n"
            f"입력 특징 {report['input_features']}개 → 출력 특징 {report['output_features']}개.\n\n"
            "Train·Validation 특징과 행 위치·정답 계약을 분리하여 저장했습니다.\n\n"
            "Test는 변환하지 않았으며 모델 학습·성능 평가는 하지 않았습니다.\n\n"
            "전처리 통계·범주·제거 목록은 로컬 JSON에서 확인하고 LLM에 보내지 마세요.\n",
            encoding="utf-8",
        )
        self.input._check_input(arguments)

        return ToolResult(status="passed", artifact_refs=tuple(sorted(OUTPUT_FILES)))

    def validate(self, output, arguments, dependency):
        """JSON 복원 후 재학습 없이 특징·행 연결·설정 결합을 검사한다."""
        output = _root(output)
        files = tuple(output.iterdir())

        if {path.name for path in files} - {"AGENT_COMPLETE.json"} != OUTPUT_FILES:
            return False

        if any(_is_link(path) or not path.is_file() for path in files):
            return False

        data, proposal = self._source(arguments, dependency)

        if json.loads((output / "split_contract.json").read_text(encoding="utf-8")) != proposal:
            return False

        pipeline = load_preparation(output / "preprocessor.json")
        transformer = pipeline.named_steps["preprocess"]

        if transformer.roles != data.roles or transformer.rules != self.input.profile.preparation_rules:
            return False

        report = json.loads((output / "preparation.json").read_text(encoding="utf-8"))
        expected = {
            "status": "preparation_complete", "source_id": self.input.source_id,
            "config_id": self.input.config_id, "proposal_id": proposal["proposal_id"],
            "rules": asdict(self.input.profile.preparation_rules),
            "train_rows": len(proposal["partitions"]["train"]),
            "input_features": len(data.roles.feature_columns),
            "output_features": len(transformer.get_feature_names_out()),
            "removed": transformer.state_["removed"],
            "diagnostics": {
                name: transformer.diagnose(data.frame.iloc[proposal["partitions"][name]])
                for name in ("train", "validation")
            },
            "prepared_splits": ["train", "validation"],
            "test_transformed": False, "model_trained": False,
        }

        if report != expected or transformer.state_["train_rows"] != expected["train_rows"]:
            return False

        rows = json.loads((output / "row_contract.json").read_text(encoding="utf-8"))

        if rows != self._rows(data, proposal):
            return False

        for name in ("train", "validation"):
            features = pipeline.transform(data.frame.iloc[proposal["partitions"][name]]).reset_index(drop=True)
            saved = pd.read_csv(output / f"{name}_features.csv", float_precision="round_trip")

            if not saved.equals(features):
                return False

        self.input._check_input(arguments)

        return True
