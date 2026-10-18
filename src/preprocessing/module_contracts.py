"""전처리 모듈 등록 명세. 선언은 실행 승인이나 OS 격리 증거가 아니다."""

from dataclasses import dataclass
import re


def check_id(value):
    """파일 경로·import 문자열이 아닌 고정 기능 ID만 허용한다."""
    if type(value) is not str or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", value) is None:
        raise ValueError("모듈 계약 ID 형식이 올바르지 않습니다.")


@dataclass(frozen=True)
class ModulePort:
    """입출력 이름과 논리 계약 버전. 실제 파일 형식 검증과는 구분한다."""

    name: str
    contract_id: str
    contract_version: int = 1
    required: bool = True

    def __post_init__(self):
        check_id(self.name)
        check_id(self.contract_id)

        if type(self.contract_version) is not int or self.contract_version < 1:
            raise ValueError("입출력 계약 버전은 양의 정수여야 합니다.")

        if type(self.required) is not bool:
            raise ValueError("입출력 필수 여부는 bool이어야 합니다.")


@dataclass(frozen=True)
class ModuleContract:
    """데이터와 무관한 불변 등록 정보. 모델 학습 기능은 초기 종류에 포함하지 않는다."""

    module_id: str
    version: str
    kind: str
    inputs: tuple[ModulePort, ...]
    outputs: tuple[ModulePort, ...]
    implementation_id: str
    result_validator_id: str
    settings_fields: tuple[str, ...] = ()
    preconditions: tuple[str, ...] = ("snapshot_verified", "tabular_roles_validated")
    original_policy: str = "preserve"
    criteria_policy: str = "none"
    derivation_partition: str | None = None
    approval: str = "explicit_execution"
    read_scope: str = "approved_staged_inputs"
    write_scope: str = "task_outputs_and_scratch"
    network: str = "deny_all"
    resources: str = "explicit_request_within_security_caps"
    schema_version: str = "module_contract_v1"

    def __post_init__(self):
        for identifier in (self.module_id, self.implementation_id, self.result_validator_id):
            check_id(identifier)

        if type(self.version) is not str or re.fullmatch(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", self.version) is None:
            raise ValueError("모듈 버전은 MAJOR.MINOR.PATCH 형식이어야 합니다.")

        if self.kind not in ("analysis", "validation", "transformation", "reporting"):
            raise ValueError("지원하지 않는 전처리 모듈 종류입니다.")

        for ports in (self.inputs, self.outputs):
            if type(ports) is not tuple or not ports or any(not isinstance(port, ModulePort) for port in ports):
                raise ValueError("입출력은 비어 있지 않은 ModulePort tuple이어야 합니다.")

            if len({port.name for port in ports}) != len(ports):
                raise ValueError("입출력 포트 이름이 중복되었습니다.")

        for names in (self.settings_fields, self.preconditions):
            if type(names) is not tuple:
                raise ValueError("설정명·선행 조건은 tuple이어야 합니다.")

            for name in names:
                check_id(name)

            if len(set(names)) != len(names):
                raise ValueError("설정명·선행 조건을 중복 지정할 수 없습니다.")

        if self.criteria_policy not in ("none", "derive_on_train", "apply_saved"):
            raise ValueError("지원하지 않는 전처리 기준 처리 방식입니다.")

        expected = "train" if self.criteria_policy == "derive_on_train" else None

        if self.derivation_partition != expected or (self.kind != "transformation" and self.criteria_policy != "none"):
            raise ValueError("전처리 기준 산출 방식과 구간·모듈 종류가 일치하지 않습니다.")

        # 초기 등록 규격에서는 권한 확장·원본 덮어쓰기·네트워크 사용을 허용하지 않는다.
        expected_policy = {
            "original_policy": "preserve", "approval": "explicit_execution",
            "read_scope": "approved_staged_inputs", "write_scope": "task_outputs_and_scratch",
            "network": "deny_all", "resources": "explicit_request_within_security_caps",
            "schema_version": "module_contract_v1",
        }

        if any(getattr(self, name) != value for name, value in expected_policy.items()):
            raise ValueError("초기 모듈 계약의 보안 정책과 일치하지 않습니다.")
