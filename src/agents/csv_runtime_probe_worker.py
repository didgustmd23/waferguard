"""합성 CSV 검사 전용 bootstrap. 고정한 기존 worker 함수만 실행한다."""

import os
from pathlib import Path
import sys


def main():
    if len(sys.argv) != 4 or sys.argv[3] not in {"split", "export", "candidate", "apply"}:
        return 2
    task = Path(sys.argv[1]).absolute()
    root = task.parent.parent
    runtime = root / "runtime"
    tools = root / "tools" / sys.argv[3]
    sys.path[:0] = [str(tools), str(runtime / "Lib/site-packages")]
    handles = [os.add_dll_directory(str(path)) for path in (runtime, runtime / "DLLs", runtime / "Library/bin") if path.is_dir()]
    from src.agents.pickle_import_worker import _isolated_process

    if not _isolated_process():
        return 2

    from src.agents.worker_contract import WorkerRequest
    import json

    with (task / "input/request.json").open("rb") as stream:
        request = WorkerRequest.from_json(stream.read(65537).decode("utf-8"))
    if request.request_id != sys.argv[2] or request.task_id != task.name or request.run_id != root.name:
        return 2
    if sys.argv[3] == "split":
        from src.agents.snapshot_split_worker import execute_request
    elif sys.argv[3] == "export":
        from src.agents.partition_export_worker import execute_request
    elif sys.argv[3] == "candidate":
        from src.agents.preparation_candidate_worker import execute_request
    else:
        from src.agents.preparation_apply_worker import execute_request

    result = execute_request(request, task)
    (task / "output/result/worker_result.json").write_text(result.to_json(), encoding="utf-8")
    (task / "output/probe_receipt.json").write_text(json.dumps({"schema_version": "csv_probe_child_v1",
        "probe_id": root.name, "request_id": request.request_id, "family": sys.argv[3],
        "isolated_token_and_job": True, "python": list(sys.version_info[:3])}), encoding="utf-8")
    return 0 if result.status == "succeeded" else 2


if __name__ == "__main__":
    raise SystemExit(main())
