"""UI용 승인 작업 서비스 목록. 기존 계산 모듈·WM 실행 해시는 변경하지 않는다."""

from pathlib import Path

from .agent_module_registry import build_agent_registry
from .split_evidence import read_metadata


CSV_IDS = ("csv_split_review", "csv_partition_export", "csv_prepare_candidate", "csv_prepare_apply", "csv_local_handoff")
CSV_CATALOGUE = Path(__file__).resolve().parents[2] / "configs/agents/modules/csv_workflows.json"
TABULAR_IDS = ("tabular_split_review", "tabular_partition_export", "tabular_prepare_candidate",
               "tabular_prepare_apply", "tabular_local_handoff")
TABULAR_VERSION = "2.0.0"
# v2는 통합 작업 조정기 전용이다. 공통 UI 목록(workflow_specifications)에는 넣지 않는다.
TABULAR_RECORDS = (
    {"module_id": "tabular_split_review", "title": "표형 v2 품질·EDA·분할 확정",
     "description": "승인·생성된 typed 표의 품질·EDA·검토·분할 제안·확정을 각각 승인합니다."},
    {"module_id": "tabular_partition_export", "title": "표형 v2 Train·Validation 내보내기",
     "description": "확정 분할에서 Train·Validation 구간만 자료형 보존 묶음으로 내보냅니다."},
    {"module_id": "tabular_prepare_candidate", "title": "표형 v2 전처리 기준 후보 산출",
     "description": "typed Train 구간에서만 기준 후보를 산출하고 원본·Validation·Test를 변경하지 않습니다."},
    {"module_id": "tabular_prepare_apply", "title": "표형 v2 저장 기준 적용",
     "description": "검토한 후보 기준을 다시 산출하지 않고 Train·Validation에 적용합니다."},
    {"module_id": "tabular_local_handoff", "title": "표형 v2 로컬 전달 묶음",
     "description": "적용 결과와 자료형 보존 행 메타데이터를 승인 후 로컬 복사·검증합니다. 외부 공유는 하지 않습니다."},
)


def csv_workflow_records():
    """검토된 정적 CSV 설명만 읽는다. 생성 폼·데이터·실행 서비스를 열지 않는다."""
    document = read_metadata(CSV_CATALOGUE)
    if set(document) != {"schema_version", "modules"} or document["schema_version"] != "csv_workflows_v1":
        raise ValueError("CSV 작업 명세 형식이 다릅니다.")

    records = document["modules"]
    if type(records) is not list or len(records) != len(CSV_IDS):
        raise ValueError("CSV 작업 명세가 누락되거나 중복됐습니다.")

    seen = set()
    for item in records:
        if (type(item) is not dict or set(item) != {"module_id", "title", "description"}
                or item["module_id"] not in CSV_IDS or item["module_id"] in seen
                or any(type(value) is not str or not value.strip() or len(value) > 2000
                       or any(ord(char) < 32 for char in value) for value in item.values())):
            raise ValueError("검토된 CSV 작업의 고정 명세만 허용합니다.")

        seen.add(item["module_id"])
    return tuple(dict(item) for item in records)


def workflow_specifications():
    legacy = [item for item in build_agent_registry().specifications(audience="ui")
              if item["execution"]["connection_status"] == "service_registered"]
    for item in csv_workflow_records():
        legacy.append({"contract": {"module_id": item["module_id"], "version": "1.0.0"},
                       "title": item["title"], "description": item["description"],
                       "execution": {"service_id": "csv_workflow_v1", "connection_status": "service_registered"}})
    return legacy


def tabular_workflow_records():
    """제안 카탈로그에 공개할 고정 v2 설명. 생성 폼·데이터·실행 서비스를 열지 않는다."""
    return tuple(dict(item) for item in TABULAR_RECORDS)


def service_version(module_id):
    """정적 등록된 CSV·표형 작업의 유일한 서비스 버전. 다른 작업은 None."""
    return "1.0.0" if module_id in CSV_IDS else TABULAR_VERSION if module_id in TABULAR_IDS else None


def workflow_service(module_id, version):
    if module_id in TABULAR_IDS:
        if version != TABULAR_VERSION:
            raise ValueError("등록된 표형 v2 작업 버전이 필요합니다.")
        from .tabular_workflow_service import TabularWorkflowService

        return TabularWorkflowService(module_id)

    if module_id in CSV_IDS:
        if version != "1.0.0":
            raise ValueError("등록된 CSV 작업 버전이 필요합니다.")
        from .csv_workflow_service import CsvWorkflowService

        return CsvWorkflowService(module_id)

    return build_agent_registry().service(module_id, version)
