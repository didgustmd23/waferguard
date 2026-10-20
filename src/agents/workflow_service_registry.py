"""UI용 승인 작업 서비스 목록. 기존 계산 모듈·WM 실행 해시는 변경하지 않는다."""

from pathlib import Path

from .agent_module_registry import build_agent_registry
from .split_evidence import read_metadata


CSV_IDS = ("csv_split_review", "csv_partition_export", "csv_prepare_candidate", "csv_prepare_apply", "csv_local_handoff")
CSV_CATALOGUE = Path(__file__).resolve().parents[2] / "configs/agents/modules/csv_workflows.json"


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


def workflow_service(module_id, version):
    if module_id in CSV_IDS:
        if version != "1.0.0":
            raise ValueError("등록된 CSV 작업 버전이 필요합니다.")
        from .csv_workflow_service import CsvWorkflowService

        return CsvWorkflowService(module_id)

    return build_agent_registry().service(module_id, version)
