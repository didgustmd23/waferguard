"""데이터 프로파일 JSON의 엄격한 파싱과 식별값 생성."""

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import re

from .contracts import ColumnRoles
from .workflows import WORKFLOW_TEMPLATES
from .quality import QualityRules
from .split import SplitRules
from .transform import PreparationRules


@dataclass(frozen=True)
class DatasetProfile:
    """데이터 역할과 검토된 실행 템플릿을 선택하는 로컬 설정."""

    profile_id: str
    adapter_id: str
    workflow_id: str
    roles: ColumnRoles
    max_plot_features: int = 12
    schema_version: str = "dataset_profile_v1"
    quality_rules: QualityRules = field(default_factory=QualityRules)
    split_rules: SplitRules | None = None
    preparation_rules: PreparationRules | None = None

    def __post_init__(self):
        # 실행 모듈을 프로파일에서 import하지 않는다. ID만 코드 등록표와 대조한다.
        for value in (self.profile_id, self.adapter_id):
            if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", value):
                raise ValueError("프로파일·어댑터 ID 형식이 올바르지 않습니다.")

        if self.schema_version != "dataset_profile_v1" or self.workflow_id not in WORKFLOW_TEMPLATES:
            raise ValueError("지원하지 않는 프로파일 버전 또는 실행 템플릿입니다.")

        if not isinstance(self.roles, ColumnRoles):
            raise TypeError("프로파일에는 ColumnRoles가 필요합니다.")

        if type(self.max_plot_features) is not int or not 1 <= self.max_plot_features <= 30:
            raise ValueError("그래프 특징 상한은 1~30 사이 정수여야 합니다.")

        if not isinstance(self.quality_rules, QualityRules):
            raise TypeError("프로파일 품질 기준은 QualityRules여야 합니다.")

        if self.roles.target is None and self.quality_rules.allowed_labels is not None:
            raise ValueError("허용 라벨을 지정하려면 target 컬럼 역할이 필요합니다.")

        if self.split_rules is not None and not isinstance(self.split_rules, SplitRules):
            raise TypeError("분할 기준은 SplitRules 또는 null이어야 합니다.")

        if "propose_split" in WORKFLOW_TEMPLATES[self.workflow_id] and self.split_rules is None:
            raise ValueError("분할 실행 템플릿에는 split_rules가 필요합니다.")

        if self.preparation_rules is not None and not isinstance(self.preparation_rules, PreparationRules):
            raise TypeError("학습 입력 준비 기준은 PreparationRules 또는 null이어야 합니다.")

        if "prepare_data" in WORKFLOW_TEMPLATES[self.workflow_id] and self.preparation_rules is None:
            raise ValueError("학습 입력 준비 템플릿에는 preparation_rules가 필요합니다.")

    def to_dict(self):
        """JSON으로 저장 가능한 정규화 프로파일을 반환한다."""
        value = asdict(self)

        if self.split_rules is None:
            value.pop("split_rules")

        if self.preparation_rules is None:
            value.pop("preparation_rules")

        for key in ("numeric", "categorical", "identifiers"):
            value["roles"][key] = list(value["roles"][key])

        if self.quality_rules == QualityRules():
            # 품질 기준을 생략한 기존 프로파일의 식별값을 보존한다.
            value.pop("quality_rules")
        elif self.quality_rules.allowed_labels is not None:
            value["quality_rules"]["allowed_labels"] = list(self.quality_rules.allowed_labels)

        return value

    @property
    def signature(self):
        payload = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False, allow_nan=False)

        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def tool_arguments_json(self):
        return json.dumps({"profile_id": self.profile_id, "profile_signature": self.signature}, sort_keys=True)


def _unique_pairs(pairs):
    """중복 JSON 키를 조용히 덮어쓰지 않는다."""
    result = {}

    for key, value in pairs:
        if key in result:
            raise ValueError("프로파일에 중복 키가 있습니다.")

        result[key] = value

    return result


def profile_from_dict(value):
    """명시된 설정만 허용하고 임의 명령·모듈·경로 추가를 거부한다."""
    expected = {
        "schema_version", "profile_id", "adapter_id", "workflow_id",
        "roles", "max_plot_features",
    }

    optional = {"quality_rules", "split_rules", "preparation_rules"}

    if not isinstance(value, dict) or not expected.issubset(value) or set(value) - expected - optional:
        raise ValueError("프로파일 필드가 계약과 일치하지 않습니다.")

    raw_roles = value["roles"]

    if not isinstance(raw_roles, dict) or set(raw_roles) != {
        "numeric", "categorical", "target", "group", "timestamp", "identifiers",
    }:
        raise ValueError("컬럼 역할 필드가 계약과 일치하지 않습니다.")

    converted = dict(raw_roles)

    for key in ("numeric", "categorical", "identifiers"):
        if not isinstance(raw_roles[key], list):
            raise ValueError("프로파일 컬럼 목록은 JSON 배열이어야 합니다.")

        converted[key] = tuple(raw_roles[key])

    values = dict(value)
    values["roles"] = ColumnRoles(**converted)

    if "preparation_rules" in values and values["preparation_rules"] is not None:
        raw_preparation = values["preparation_rules"]

        if not isinstance(raw_preparation, dict) or set(raw_preparation) - set(PreparationRules.__dataclass_fields__):
            raise ValueError("학습 입력 준비 기준 필드가 계약과 일치하지 않습니다.")

        values["preparation_rules"] = PreparationRules(**raw_preparation)

    if "split_rules" in values and values["split_rules"] is not None:
        raw_split = values["split_rules"]

        if not isinstance(raw_split, dict) or set(raw_split) - set(SplitRules.__dataclass_fields__):
            raise ValueError("분할 기준 필드가 계약과 일치하지 않습니다.")

        values["split_rules"] = SplitRules(**raw_split)

    if "quality_rules" in values:
        raw_rules = values["quality_rules"]

        if not isinstance(raw_rules, dict) or set(raw_rules) - set(QualityRules.__dataclass_fields__):
            raise ValueError("품질 기준 필드가 계약과 일치하지 않습니다.")

        rules = dict(raw_rules)
        labels = rules.get("allowed_labels")

        if labels is not None:
            if not isinstance(labels, list):
                raise ValueError("허용 라벨은 JSON 배열 또는 null이어야 합니다.")

            rules["allowed_labels"] = tuple(labels)

        values["quality_rules"] = QualityRules(**rules)

    return DatasetProfile(**values)


def load_profile(path):
    """크기를 제한한 로컬 JSON 프로파일을 읽는다."""
    path = Path(path)

    if path.stat().st_size > 64 * 1024:
        raise ValueError("프로파일 파일 크기 상한을 초과했습니다.")

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs)

    return profile_from_dict(value)
