"""Train 복사본만 읽어 후보 기준을 계산한다. 변환 파일·모델·pickle을 만들지 않는다."""

from pathlib import Path
import sys


DIAGNOSTIC_DETAILS = frozenset({"CATEGORY_LIMIT_EXCEEDED"})


def _detail(root, detail):
    """공용 진단 기록에 허용 목록의 세부 코드만 덧붙인다. 분류 기록이 없거나 목록 밖이면 아무것도 쓰지 않는다."""
    import json

    path = root / "scratch" / "task" / "diagnostic.json"

    try:
        item = json.loads(path.read_text(encoding="utf-8"))

        if detail in DIAGNOSTIC_DETAILS and type(item) is dict and set(item) == {"stage", "code"}:
            path.write_text(json.dumps(dict(item, detail=detail)), encoding="utf-8")
    except (OSError, ValueError):
        # 세부 코드 기록 실패가 기존 분류·오류를 바꾸지 않는다.
        pass


def execute_request(request, root):
    """고정 요청과 Train 바이트를 확인하고 기존 복합 전처리 코어의 fit만 수행한다."""
    import csv
    from dataclasses import asdict
    import hashlib
    import json

    from src.agents.input_snapshot import Snapshot
    from src.agents.preparation_candidate_report import candidate_markdown
    from src.agents.snapshot_eda_worker import _diagnostic, _verify_staged_snapshot
    from src.agents.worker_contract import WorkerResult, _decode
    from src.preprocessing.contracts import ColumnRoles
    from src.preprocessing.split import _digest
    from src.preprocessing.transform import PreparationInputError, PreparationRules, build_preparation, save_preparation
    from src.agents.tabular_preparation_worker import CANDIDATE_POLICY, input_files, load_part, feature_view, check_part
    from src.preprocessing.tabular_bundle import worker_filesystem

    root = Path(root).resolve()
    source = root / "input/snapshot/input.bin"
    settings = root / "input/candidate_task.json"
    output = root / "output/result"
    stage = "REQUEST_PATHS"
    old_limit = csv.field_size_limit(1024 * 1024)
    typed = request.policy_version == CANDIDATE_POLICY

    try:
        if (request.task_id != root.name or request.stage != "derive_preparation" or request.action_id != "derive_preparation"
                or request.policy_version not in ("preparation_candidate_v1", CANDIDATE_POLICY) or request.dependency_directories
                or request.input_files != (input_files(root) if typed else (str(source), str(source.parent / "snapshot.json"), str(settings)))
                or request.output_directory != str(output) or request.scratch_directory != str(root / "scratch/task")):
            raise ValueError("후보 산출 작업의 고정 경로가 다릅니다.")

        stage = "SETTINGS"

        with settings.open("rb") as stream:
            packet = _decode(stream.read(65537).decode("utf-8"))

        fields = {"version", "train_contract_id", "split_evidence_id", "export_marker_sha256", "snapshot", "roles",
                  "train_rows", "rules", "limits", "max_candidate_bytes", "code_id"}
        if typed:
            fields = fields - {"snapshot"} | {"input", "input_warnings"}

        if set(packet) != fields or _digest(packet) != request.config_id or packet["version"] != request.policy_version or packet["limits"] != asdict(request.limits):
            raise ValueError("후보 산출 설정이 승인 계약과 다릅니다.")

        expected = None if typed else Snapshot(**packet["snapshot"])
        roles_dict = dict(packet["roles"])

        for name in ("numeric", "categorical", "identifiers"):
            if type(roles_dict[name]) is not list:
                raise ValueError("후보 입력 역할은 명시된 배열이어야 합니다.")

            roles_dict[name] = tuple(roles_dict[name])

        roles = ColumnRoles(**roles_dict)
        rules = PreparationRules(**packet["rules"])
        maximum = packet["max_candidate_bytes"]
        rows = packet["train_rows"]

        if (request.source_id != (packet["input"]["source_id"] if typed else "file_snapshot:" + expected.sha256) or type(rows) is not int or not 1 <= rows <= 1_000_000
                or type(maximum) is not int or not 1 <= maximum <= 8 * 1024 * 1024):
            raise ValueError("Train 출처·행 수·후보 크기 상한이 올바르지 않습니다.")

        stage = "SNAPSHOT"

        if typed:
            if (packet["input"]["contract_id"] != packet["train_contract_id"] or packet["input"]["split_signature"] != packet["split_evidence_id"]
                    or packet["input"]["export_marker_sha256"] != packet["export_marker_sha256"] or packet["input"]["rows"] != rows):
                raise ValueError("typed Train 기준 참조가 다릅니다.")
            frame = feature_view(load_part(root, packet["input"], "train"), roles)
        elif not _verify_staged_snapshot(root, expected):
            raise ValueError("선택한 Train 복사본이 변경되었습니다.")

        stage = "CSV_HEADER"

        if not typed:
            frame = _load_csv_train(source, roles, rows)
        if len(frame) != rows:
            raise ValueError("Train 행 수가 확정 분할과 다릅니다.")

        stage = "INPUT_VALIDATION"
        pipeline = build_preparation(roles, rules)
        pipeline.fit(frame)
        transformer = pipeline.named_steps["preprocess"]

        if rows * len(transformer.get_feature_names_out()) > rules.max_output_cells:
            raise ValueError("후보의 예상 출력 셀 수가 설정 상한을 초과했습니다.")

        candidate = output / "candidate"
        candidate.mkdir()
        save_preparation(pipeline, candidate / "preprocessor.json", max_bytes=maximum)
        state = transformer.state_

        with (candidate / "preprocessor.json").open("rb") as stream:
            digest = hashlib.sha256(stream.read(maximum + 1)).hexdigest()

        report = {"schema_version": "preparation_candidate_report_v3" if typed else "preparation_candidate_report_v2", "status": "candidate_ready", "binding": packet,
                  "preprocessor_sha256": digest, "removed": state["removed"],
                  "retained": [record["column"] for record in state["records"]],
                  "output_feature_count": len(transformer.get_feature_names_out()), "impact": transformer.impact(frame),
                  "transformed": False, "model_trained": False}

        for filename, content in (("candidate.json", json.dumps(report, ensure_ascii=False, allow_nan=False)),
                                  ("candidate.md", candidate_markdown(report))):
            payload = content.encode("utf-8")
            if len(payload) > maximum:
                raise ValueError("후보 보고서 크기 상한을 초과했습니다.")
            (candidate / filename).write_bytes(payload)

        stage = "SNAPSHOT_RECHECK"
        if typed:
            with worker_filesystem(root):
                check_part(root / "input/train", packet["input"], partition="train")
        elif not _verify_staged_snapshot(root, expected):
            raise ValueError("후보 계산 중 Train 복사본이 변경되었습니다.")
    except (ValueError, TypeError, KeyError, OSError, UnicodeError, csv.Error) as error:
        code = "ACCESS_DENIED" if isinstance(error, PermissionError) else "IO_FAILED" if isinstance(error, OSError) else "INPUT_INVALID"
        _diagnostic(root, stage, code)
        if isinstance(error, PreparationInputError):
            _detail(root, error.code)
        return WorkerResult(request.task_id, request.request_id, "blocked_input", "WORKER_INPUT_BLOCKED")
    except Exception:
        _diagnostic(root, stage, "RUNTIME_FAILED")
        return WorkerResult(request.task_id, request.request_id, "failed_runtime", "WORKER_FAILED")
    finally:
        csv.field_size_limit(old_limit)

    _diagnostic(root, "COMPLETE", "PASSED")
    return WorkerResult(request.task_id, request.request_id, "succeeded", "WORKER_EXITED_OK", ("candidate",))


def _load_csv_train(source, roles, rows):
    """기존 v1 CSV 헤더·행·문자열 자료형 규칙을 그대로 적용한다."""
    import pandas as pd
    from src.agents.partition_csv import iter_csv_rows
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = iter_csv_rows(stream)
        header = next(reader, [])

        if not header or any(not name for name in header) or len(set(header)) != len(header) or not set(roles.required_columns).issubset(header):
            raise ValueError("Train CSV의 필수 역할·헤더가 다릅니다.")

        count = 0

        for row in reader:
            count += 1

            if count > rows or len(row) != len(header):
                raise ValueError("Train의 행·열 구조가 계약과 다릅니다.")

    if count != rows:
        raise ValueError("Train 행 수가 확정 분할과 다릅니다.")

    frame = pd.read_csv(source, encoding="utf-8-sig", usecols=list(roles.feature_columns),
                            dtype={name: "string" for name in roles.categorical})
    if len(frame) != rows:
        raise ValueError("표형 로더의 Train 행 수가 확정 계약과 다릅니다.")

    return frame


def main():
    """고정 복사 런타임을 초기화한다. 요청 식별값이 다르면 실행하지 않는다."""
    import os

    root = Path(sys.argv[1]).resolve()
    runtime = root / "runtime"
    sys.path[:0] = [str(root / "tools"), str(runtime / "Lib/site-packages")]
    handles = [os.add_dll_directory(str(path)) for path in (runtime, runtime / "DLLs", runtime / "Library/bin") if path.is_dir()]

    from src.agents.worker_contract import WorkerRequest

    try:
        with (root / "input/request.json").open("rb") as stream:
            request = WorkerRequest.from_json(stream.read(65537).decode("utf-8"))

        if request.request_id != sys.argv[2]:
            return 2

        result = execute_request(request, root)
        (root / "output/result/worker_result.json").write_text(result.to_json(), encoding="utf-8")
        return 0 if result.status == "succeeded" else 2
    except Exception:
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
