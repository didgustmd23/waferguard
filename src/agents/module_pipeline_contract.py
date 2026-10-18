"""정적 모듈 계획을 현재 CSV worker가 지원하는 실행 범위에 결합한다."""

from dataclasses import asdict, replace
import json

from src.preprocessing.pipeline import validate_pipeline
from src.preprocessing.quality import QualityRules


POLICY_VERSION = "snapshot_module_pipeline_v1"
MODULE_FILES = {
    "eda_missing": ("missing.csv",),
    "eda_numeric": ("numeric_statistics.csv",),
    "eda_categories": ("categorical_distribution.csv",),
    "eda_target": ("target_distribution.csv",),
    "eda_correlations": ("pearson.csv", "spearman.csv", "correlation_support.csv"),
    "tabular_quality": ("quality.json", "quality.md"),
}


def bind_pipeline(config, profile):
    """프로파일 품질 규칙을 기본값으로 결합하고 지원하지 않는 실행 연결은 거부한다."""
    # 먼저 원래 설정을 검증해 임의 필드·실행 옵션을 복사 과정에서 잃어버리지 않는다.
    validate_pipeline(config)
    bound = json.loads(json.dumps(config, ensure_ascii=False, allow_nan=False))

    for step in bound["steps"]:
        if step["module_id"] not in MODULE_FILES or step["version"] != "1.0.0":
            raise ValueError("현재 격리 실행에 연결되지 않은 모듈입니다.")

        if step["inputs"] != {"data": "source.data"}:
            raise ValueError("초기 격리 실행은 같은 CSV 입력을 읽는 분석·검사만 지원합니다.")

        if step["module_id"] == "tabular_quality":
            # 프로파일의 허용 라벨·시간 형식 등 기존 보호 기준을 조용히 누락하지 않는다.
            # 사용자가 명시한 변경은 전체 승인 미리보기에서 확인한다.
            step["settings"] = {**asdict(profile.quality_rules), **step.get("settings", {})}
            # target이 없는데 허용 라벨을 지정하는 등 프로파일과 모순된 설정은 실행 전에 거부한다.
            rules = dict(step["settings"])

            if isinstance(rules["allowed_labels"], list):
                rules["allowed_labels"] = tuple(rules["allowed_labels"])

            replace(profile, quality_rules=QualityRules(**rules))

    # 승인 인자·저장·IPC는 JSON 배열을 사용한다. tuple/list 차이 때문에 동일 계획이
    # 실행 시 변경으로 오인되지 않도록 결합 후에도 같은 JSON 표현으로 고정한다.
    bound = json.loads(json.dumps(bound, ensure_ascii=False, allow_nan=False))

    return bound, validate_pipeline(bound)
