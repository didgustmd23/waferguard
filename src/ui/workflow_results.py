"""공통 서비스 결과 참조를 제한된 로컬 표시 복사본으로 읽는다."""

import hashlib
from pathlib import Path

from src.agents.artifacts import _fingerprint
from src.agents.isolated_eda import _read_file, _valid_png
from src.agents.worker_paths import _path


def load_workflow_results(workflow):
    service = workflow.service()
    result = service.results(workflow.name)
    if len(result["items"]) > 128:
        raise ValueError("화면 표시 항목 상한을 초과했습니다.")

    files, directories, stamps = [], [], {}
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

        if path.suffix.lower() == ".png" and not _valid_png(data):
            raise ValueError("PNG 구조 검증에 실패했습니다.")

        total += len(data)
        stamps[path] = stamp
        files.append({"name": item["label"], "suffix": path.suffix.lower(), "data": data})

    # 참조를 얻은 뒤 바뀐 결과를 정상 완료로 표시하지 않는다.
    if service.results(workflow.name) != result or any(_fingerprint(p) != stamp for p, stamp in stamps.items()):
        raise ValueError("검증 중 결과가 변경되었습니다.")

    return {"status": result["status"], "files": files, "directories": directories,
            "notice": "검사 시점의 로컬 표시 복사본입니다. LLM·외부 추적으로 전송하지 않습니다."}
