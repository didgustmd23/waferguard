"""확정 분할과 동일 원본 snapshot을 구간 내보내기 요청에 결합한다. 승인 발급은 하지 않는다."""

from dataclasses import asdict, dataclass

from .artifacts import _fingerprint
from .eda_input_contract import EdaInputContract
from .snapshot_eda import _digest
from .split_evidence import ConfirmedSplit


POLICY_VERSION = "partition_export_v1"


@dataclass(frozen=True)
class PartitionExportContract:
    """내보내기 준비 계약. 직접 호출은 사용자 승인·OS 격리를 대신하지 않는다."""

    split: ConfirmedSplit
    source: EdaInputContract
    max_partition_bytes: int

    def __post_init__(self):
        if not isinstance(self.split, ConfirmedSplit) or not isinstance(self.source, EdaInputContract):
            raise TypeError("확정 분할과 원본 snapshot 입력 계약이 필요합니다.")

        if type(self.max_partition_bytes) is not int or self.max_partition_bytes < 1:
            raise ValueError("구간별 출력 크기 상한은 양의 정수 바이트여야 합니다.")

        profile = self.source.profile

        if profile.adapter_id != "tabular_csv_v1" or profile.split_rules is None:
            raise ValueError("분할 규칙이 있는 CSV 프로파일이 필요합니다.")

        # 기존 CSV 어댑터의 출처 ID와 형식 독립 snapshot의 바이트 해시를 명시적으로 연결한다.
        if self.split.artifact.scope.source_id != "csv:" + self.source.snapshot.sha256:
            raise ValueError("확정 분할과 원본 CSV 복사본의 내용 식별값이 다릅니다.")

        if self.split.artifact.scope.config_id != profile.signature:
            raise ValueError("확정 분할에 사용한 프로파일과 역할·규칙이 다릅니다.")

    def check(self):
        self.split.check()
        self.source.check_input()

        return True

    def packet(self, code_id):
        """작업 폴더 밖의 경로·승인 DB 없이 전달할 로컬 IPC 메타데이터."""
        if type(code_id) is not str or len(code_id) != 64 or any(char not in "0123456789abcdef" for char in code_id):
            raise ValueError("검토한 내보내기 코드의 SHA-256 식별값이 필요합니다.")

        self.check()

        return {
            "version": POLICY_VERSION, "snapshot": asdict(self.source.snapshot),
            "profile": self.source.profile.to_dict(), "limits": asdict(self.source.limits),
            "split_evidence_id": self.split.signature,
            "split_file_sha256": _fingerprint(self.split.artifact.directory / "split.json")["sha256"],
            "max_partition_bytes": self.max_partition_bytes, "code_id": code_id,
        }

    def config_id(self, code_id):
        return _digest(self.packet(code_id))


@dataclass(frozen=True)
class TabularExportContract:
    """v2 생성 출처·분할·프로파일·구간 출력 상한을 결합한다."""
    split: ConfirmedSplit
    source: object
    max_partition_bytes: int

    def __post_init__(self):
        from .tabular_source_contract import TabularEdaInputContract
        if type(self.split) is not ConfirmedSplit or type(self.source) is not TabularEdaInputContract:
            raise TypeError("확정 분할·v2 표형 분석 계약이 필요합니다.")
        if type(self.max_partition_bytes) is not int or self.max_partition_bytes < 1:
            raise ValueError("구간별 출력 상한은 양의 정수여야 합니다.")
        if (self.split.artifact.scope.policy_version != "tabular_split_v2"
                or self.split.artifact.scope.source_id != self.source.source.source_id
                or self.split.artifact.scope.config_id != self.source.profile.signature
                or self.source.profile.split_rules is None):
            raise ValueError("확정 분할과 생성 출처·프로파일이 다릅니다.")

    def check(self):
        self.split.check()
        self.source.check_input()
        return True

    def packet(self, code_id):
        """승인 DB·외부 경로 없이 생성 참조와 확정 배정만 결합한다."""
        from .tabular_worker_input import EXPORT_POLICY
        if type(code_id) is not str or len(code_id) != 64 or any(c not in "0123456789abcdef" for c in code_id):
            raise ValueError("검토된 실행 코드 지문이 필요합니다.")
        self.check()
        return {"version": EXPORT_POLICY, "source": self.source.source.packet(), "profile": self.source.profile.to_dict(),
                "limits": asdict(self.source.limits), "split_evidence_id": self.split.signature,
                "split_file_sha256": _fingerprint(self.split.artifact.directory / "split.json")["sha256"],
                "max_partition_bytes": self.max_partition_bytes, "code_id": code_id}

    def config_id(self, code_id):
        return _digest(self.packet(code_id))
