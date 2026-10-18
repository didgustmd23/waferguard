"""고정 합성 CSV EDA worker 진입점. 임의 모듈·명령·실데이터를 실행하지 않는다."""

from pathlib import Path
import sys


def execute_request(request, root):
    """등록 ID·정확한 경로·fixture 내용·요청 ID를 검증한 후 범용 코어를 호출한다."""
    from src.agents.synthetic_eda_data import CSV_BYTES, SOURCE_ID, fixture_input
    from src.agents.worker_contract import WorkerResult
    from src.preprocessing.eda_report import generate_eda_report

    root = Path(root).resolve()
    expected_input = root / "input" / "synthetic.csv"
    expected_output = root / "output" / "result"
    expected_scratch = root / "scratch" / "task"
    payload = expected_input.read_bytes()

    valid = (
        request.action_id == "run_eda" and request.stage == "run_eda"
        and request.source_id == SOURCE_ID and request.task_id == root.name
        and request.input_files == (str(expected_input),)
        and not request.dependency_directories
        and request.output_directory == str(expected_output)
        and request.scratch_directory == str(expected_scratch)
        and payload == CSV_BYTES
    )

    if not valid:
        return WorkerResult(request.task_id, request.request_id, "blocked_input", "WORKER_INPUT_BLOCKED")

    # 검증한 CSV 바이트를 다시 읽지 않고 분석한다. 실데이터 경로는 만들지 않는다.
    outcome = generate_eda_report(fixture_input(payload), expected_output / "eda", max_plot_features=3)

    if outcome != {"status": "eda_complete"}:
        return WorkerResult(request.task_id, request.request_id, "failed_runtime", "WORKER_FAILED")

    return WorkerResult(request.task_id, request.request_id, "succeeded", "WORKER_EXITED_OK", ("eda",))


def main():
    root = Path(sys.argv[1]).resolve()
    expected_request_id = sys.argv[2]
    runtime = root / "runtime"
    handles = []

    # 복사한 고정 코드와 명시한 런타임만 검색한다.
    sys.path.insert(0, str(root / "tools"))
    sys.path.insert(0, str(runtime / "Lib" / "site-packages"))

    import os

    for directory in (runtime, runtime / "DLLs", runtime / "Library" / "bin"):
        if directory.is_dir():
            handles.append(os.add_dll_directory(str(directory)))

    from src.agents.worker_contract import WorkerRequest, WorkerResult

    # 요청을 해석하지 못하면 상세 예외를 파일·stdout에 복사하지 않는다.
    try:
        path = root / "input" / "request.json"

        if path.stat().st_size > 65536:
            return 2

        request = WorkerRequest.from_json(path.read_text(encoding="utf-8"))

        if request.request_id != expected_request_id:
            return 2
    except Exception:
        return 2

    try:
        import matplotlib
        matplotlib.use("Agg")
        result = execute_request(request, root)
    except Exception:
        result = WorkerResult(request.task_id, request.request_id, "failed_runtime", "WORKER_FAILED")

    output = root / "output" / "result" / "worker_result.json"
    output.write_text(result.to_json(), encoding="utf-8")

    return 0 if result.status == "succeeded" else 2


if __name__ == "__main__":
    raise SystemExit(main())
