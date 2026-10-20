"""공통 서비스 결과 참조를 제한된 로컬 표시 복사본으로 읽는다."""

import hashlib
from pathlib import Path

from src.agents.artifacts import _fingerprint
from src.agents.isolated_eda import _read_file
from .image_validation import valid_report_png
from src.agents.worker_paths import _path


def _result_references(service, name):
    """캐시·Dataset은 이미 존재하는 전용 커밋 검증기를 사용한다.

    분할 증거 reader의 허용 단계를 확장하거나 승인된 실행 코드를 변경하지 않는다.
    UI는 보고서 참조만 구성하며 캐시 배열을 읽거나 worker를 실행하지 않는다.
    """
    module_id = getattr(service, "module_id", None)
    if module_id not in {"wm_cache_prepare", "wm_dataset_check"}:
        return service.results(name)

    _cli, tool, manager = service._open(name)
    before = tool.states.load()
    if before.status != "completed":
        return service.results(name)

    if module_id == "wm_cache_prepare":
        from src.agents.wm_cache_evidence import committed_cache

        output = committed_cache(tool, manager)
        folder, report = "cache", "cache.md"
    else:
        from src.agents.wm_dataset_contract import committed_dataset

        output = committed_dataset(tool, manager)
        folder, report = "dataset", "dataset_check.md"

    tool.inspect_attempt(output)
    directory = _path(output / folder, directory=True)
    path = _path(directory / report, directory=False)
    if tool.states.load() != before:
        raise ValueError("결과 조회 중 상태가 변경되었습니다.")

    return {"status": before.status, "items": [
        {"kind": "artifact_list", "path": str(directory), "label": folder},
        {"kind": "report", "path": str(path), "label": folder + "/" + report},
    ]}


def load_workflow_results(workflow):
    service = workflow.service()
    result = _result_references(service, workflow.name)
    if len(result["items"]) > 128:
        raise ValueError("화면 표시 항목 상한을 초과했습니다.")

    files, directories, stamps, graph_evidence = [], [], {}, []
    total = 0
    for item in result["items"]:
        if item["kind"] == "artifact_list":
            directories.append(item["path"])
            continue

        path = _path(Path(item["path"]), directory=False)
        if path.suffix.lower() not in {".csv", ".png", ".md", ".json"}:
            continue

        stamp = _fingerprint(path)
        if stamp["size"] > 8 * 1024 ** 2 or total + stamp["size"] > 32 * 1024 ** 2:
            raise ValueError("화면 표시 크기 상한을 초과했습니다. 로컬 보고서를 확인하세요.")

        data = _read_file(path, 8 * 1024 ** 2)
        if _fingerprint(path) != stamp or len(data) != stamp["size"]:
            raise ValueError("표시 자료를 읽는 중 파일이 변경되었습니다.")

        if hashlib.sha256(data).hexdigest() != stamp["sha256"]:
            raise ValueError("표시 복사본의 해시가 일치하지 않습니다.")

        if path.suffix.lower() == ".png" and not valid_report_png(data):
            raise ValueError("PNG 구조 검증에 실패했습니다.")

        total += len(data)
        stamps[path] = stamp
        if item["kind"] == "graph_evidence":
            graph_evidence.append({"name": item["label"], "data": data, "plot_limit": item["plot_limit"]})
        else:
            files.append({"name": item["label"], "suffix": path.suffix.lower(), "data": data})

    # 참조를 얻은 뒤 바뀐 결과를 정상 완료로 표시하지 않는다.
    if _result_references(service, workflow.name) != result or any(_fingerprint(p) != stamp for p, stamp in stamps.items()):
        raise ValueError("검증 중 결과가 변경되었습니다.")

    return {"status": result["status"], "files": files, "directories": directories, "graph_evidence": graph_evidence,
            "notice": "검사 시점의 로컬 표시 복사본입니다. LLM·외부 추적으로 전송하지 않습니다."}
