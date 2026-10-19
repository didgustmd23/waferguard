"""검증된 로컬 결과만 제한된 크기의 표시 자료로 읽는다. LLM 전송은 없다."""

import hashlib

from src.agents.artifacts import _inventory, _root, check_artifacts
from src.agents.isolated_eda import _read_file, _valid_png
from src.agents.module_pipeline_results import read_pipeline_result


def load_results(execution):
    """실행 이력·결과 계약을 재검사한다. 검토 중단 결과를 완료로 표시하지 않는다."""
    state = execution.read_state()

    if state.status not in {"completed", "blocked_input"}:
        raise ValueError("완료 또는 품질 검토 중단 결과만 열 수 있습니다.")

    run = execution._check_current()
    task = run.approvals.get_task(run.executor.scope_for("run_pipeline"))

    if task is None:
        raise ValueError("해당 단계의 작업 기록이 없습니다.")

    output = _root(run.executor._output(task))
    tool = run.pipeline_execution.worker_tool
    pipeline = _root(output / "pipeline")
    result = read_pipeline_result(pipeline, tool.module_plan, tool.contract.profile)

    if result is None:
        raise ValueError("표시할 결과 계약을 확인하지 못했습니다.")

    if state.status == "completed":
        step = run.plan.steps[-1]

        if (task["status"] != "committed" or result["status"] != "passed"
                or not check_artifacts(output, run.executor._context(step)).reusable
                or not run.pipeline_execution.validate(output, run.arguments)):
            raise ValueError("완료 결과의 정합성 확인에 실패했습니다.")
    elif result["status"] not in {"review_required", "blocked_input"}:
        raise ValueError("중단 상태와 품질 결과가 일치하지 않습니다.")

    # 이미 계약 검증을 거친 폴더도 표시 단계에서 링크·파일 수·크기를 재확인한다.
    inventory = _inventory(pipeline)

    if len(inventory) > 128:
        raise ValueError("화면에 표시할 파일 수 상한을 초과했습니다.")

    files = []
    total = 0

    for name, path in sorted(inventory.items()):
        if path.suffix.lower() not in {".md", ".csv", ".png"}:
            continue

        data = _read_file(path, 8 * 1024 * 1024)
        total += len(data)

        if total > 32 * 1024 * 1024:
            raise ValueError("화면 결과 자료의 합계 크기 상한을 초과했습니다.")

        if path.suffix.lower() == ".png" and not _valid_png(data):
            raise ValueError("PNG 구조 확인에 실패했습니다.")

        files.append({"name": name, "suffix": path.suffix.lower(), "data": data,
                      "sha256": hashlib.sha256(data).hexdigest()})

    return {"status": result["status"], "directory": str(pipeline), "files": files,
            "notice": "검사 시점의 로컬 표시 복사본입니다. LLM에는 보내지 않았습니다."}
