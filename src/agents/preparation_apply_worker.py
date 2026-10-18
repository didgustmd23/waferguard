"""저장된 후보를 복원해 Train·Validation에 transform만 수행하는 격리 진입점."""

from pathlib import Path
import sys


def execute_request(request, root):
    import csv
    from dataclasses import asdict
    import json
    import pandas as pd

    from src.agents.input_snapshot import Snapshot, _scan
    from src.agents.partition_csv import iter_csv_rows
    from src.agents.partition_export_worker import BoundedCsvWriter
    from src.agents.preparation_apply_io import write_json_bounded, application_markdown
    from src.agents.snapshot_eda_worker import _diagnostic, _verify_staged_snapshot
    from src.agents.worker_contract import WorkerResult, _decode
    from src.preprocessing.profiles import _unique_pairs
    from src.preprocessing.split import _digest
    from src.preprocessing.transform import load_preparation

    root = Path(root).resolve()
    inputs = root / "input"
    output = root / "output/result"
    stage = "REQUEST_PATHS"
    old_limit = csv.field_size_limit(1024 * 1024)

    try:
        expected_files = tuple(str(inputs / name) for name in (
            "train/input.bin", "train/snapshot.json", "validation/input.bin", "validation/snapshot.json",
            "preprocessor.json", "positions.json", "apply_task.json",
        ))

        if (request.task_id != root.name or request.stage != "apply_preparation" or request.action_id != "apply_preparation"
                or request.policy_version != "preparation_apply_v1" or request.dependency_directories
                or request.input_files != expected_files or request.output_directory != str(output)
                or request.scratch_directory != str(root / "scratch/task")):
            raise ValueError("적용 작업의 고정 요청 경로가 다릅니다.")

        stage = "SETTINGS"

        with (inputs / "apply_task.json").open("rb") as stream:
            packet = _decode(stream.read(65537).decode("utf-8"))

        fields = {"version", "candidate_id", "preprocessor_sha256", "max_state_bytes", "roles", "rules", "output_columns_sha256",
                  "output_feature_count", "partitions", "limits", "max_partition_bytes", "code_id"}

        if set(packet) != fields or packet["version"] != "preparation_apply_v1" or _digest(packet) != request.config_id or packet["limits"] != asdict(request.limits):
            raise ValueError("적용 설정이 승인 범위와 다릅니다.")

        maximum = packet["max_partition_bytes"]
        state_maximum = packet["max_state_bytes"]

        if type(maximum) is not int or not 1 <= maximum <= 1024 * 1024 * 1024 or type(state_maximum) is not int or not 1 <= state_maximum <= 8 * 1024 * 1024:
            raise ValueError("적용 상태·출력 파일 상한이 올바르지 않습니다.")

        if type(packet["partitions"]) is not dict or set(packet["partitions"]) != {"train", "validation"}:
            raise ValueError("적용 대상은 Train·Validation만 허용합니다.")

        state_path = inputs / "preprocessor.json"

        with state_path.open("rb") as stream:
            _, digest = _scan(stream, state_maximum)

        if digest != packet["preprocessor_sha256"]:
            raise ValueError("저장 후보의 정확한 바이트 해시가 다릅니다.")

        with (inputs / "positions.json").open("rb") as stream:
            raw = stream.read(8 * 1024 * 1024 + 1)

        if len(raw) > 8 * 1024 * 1024:
            raise ValueError("행 위치 기록 크기 상한을 초과했습니다.")

        positions = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_pairs)

        if type(positions) is not dict or set(positions) != {"train", "validation"}:
            raise ValueError("행 위치 구간 목록이 올바르지 않습니다.")

        all_positions = []
        snapshots = {}

        for name in ("train", "validation"):
            binding = packet["partitions"][name]

            if type(binding) is not dict or set(binding) != {"contract_id", "snapshot", "rows", "positions_sha256"}:
                raise ValueError("구간 입력 계약 필드가 다릅니다.")

            rows = binding["rows"]
            values = positions[name]

            if (type(rows) is not int or not 1 <= rows <= 1_000_000 or type(values) is not list or len(values) != rows
                    or any(type(value) is not int or value < 0 for value in values)
                    or values != sorted(set(values)) or _digest(values) != binding["positions_sha256"]):
                raise ValueError("적용 행 수·순서가 승인 범위와 다릅니다.")

            snapshots[name] = Snapshot(**binding["snapshot"])
            all_positions.extend(values)

            if not _verify_staged_snapshot(root, snapshots[name], name=name):
                raise ValueError("적용 입력 복사본이 변경되었습니다.")

        if len(set(all_positions)) != len(all_positions) or request.source_id != snapshots["train"].source_id:
            raise ValueError("적용 구간에 행 중복이 있거나 Train 출처가 다릅니다.")

        stage = "INPUT_VALIDATION"
        # 복원은 설정·통계 JSON만 사용한다. fit·fit_transform·prepare_splits는 호출하지 않는다.
        pipeline = load_preparation(state_path)
        transformer = pipeline.named_steps["preprocess"]

        if (json.loads(json.dumps(asdict(transformer.roles))) != packet["roles"] or asdict(transformer.rules) != packet["rules"]
                or _digest(transformer.get_feature_names_out().tolist()) != packet["output_columns_sha256"]
                or len(transformer.get_feature_names_out()) != packet["output_feature_count"]
                or transformer.state_["train_rows"] != packet["partitions"]["train"]["rows"]):
            raise ValueError("복원 상태의 역할·규칙·Train 행 수·출력 순서가 승인 후보와 다릅니다.")

        roles = transformer.roles
        metadata_columns = list(dict.fromkeys(value for value in (roles.target, roles.group, roles.timestamp, *roles.identifiers) if value is not None))
        prepared = output / "prepared"
        prepared.mkdir()
        records = {}

        for name in ("train", "validation"):
            source = inputs / name / "input.bin"
            metadata = []
            stage = "CSV_HEADER"

            with source.open(encoding="utf-8-sig", newline="") as stream:
                reader = iter_csv_rows(stream)
                header = next(reader, [])

                if not header or any(not value for value in header) or len(header) != len(set(header)) or not set(roles.required_columns).issubset(header):
                    raise ValueError("적용 입력의 역할·CSV 헤더가 다릅니다.")

                indices = [header.index(column) for column in metadata_columns]

                for row in reader:
                    if len(metadata) >= len(positions[name]) or len(row) != len(header):
                        raise ValueError("적용 입력 행·열 구조가 계약과 다릅니다.")

                    metadata.append([row[index] for index in indices])

            if len(metadata) != len(positions[name]):
                raise ValueError("적용 입력 행 수가 계약과 다릅니다.")

            stage = "CSV_LOAD"
            frame = pd.read_csv(source, encoding="utf-8-sig", usecols=list(roles.feature_columns),
                                dtype={column: "string" for column in roles.categorical})

            if len(frame) != len(positions[name]):
                raise ValueError("표형 로더의 행 수가 계약과 다릅니다.")

            stage = "INPUT_VALIDATION"
            transformed = pipeline.transform(frame)

            with (prepared / f"{name}_features.csv").open("xb") as stream:
                writer = BoundedCsvWriter(stream, maximum)
                transformed.to_csv(writer, index=False, float_format="%.17g", lineterminator="\n")

            row_record = write_json_bounded(prepared / f"{name}_rows.json",
                                            {"row_positions": positions[name], "metadata_columns": metadata_columns, "metadata": metadata},
                                            min(maximum, 8 * 1024 * 1024))
            records[name] = {"rows": len(frame), "features": {"size_bytes": writer.size, "sha256": writer.digest.hexdigest()}, "row_contract": row_record}

        stage = "SNAPSHOT_RECHECK"

        for name in snapshots:
            if not _verify_staged_snapshot(root, snapshots[name], name=name):
                raise ValueError("적용 중 입력 복사본이 변경되었습니다.")

        with state_path.open("rb") as stream:
            _, digest = _scan(stream, state_maximum)

        if digest != packet["preprocessor_sha256"]:
            raise ValueError("적용 중 저장 기준 파일이 변경되었습니다.")

        write_json_bounded(prepared / "application.json", {"schema_version": "preparation_application_v1", "status": "applied",
                           "binding": packet, "partitions": records, "transformed": True, "refit": False, "test_used": False, "model_trained": False}, 65536)
        (prepared / "application.md").write_text(application_markdown(), encoding="utf-8", newline="")
    except (OSError, ValueError, TypeError, KeyError, UnicodeError, csv.Error) as error:
        code = "ACCESS_DENIED" if isinstance(error, PermissionError) else "IO_FAILED" if isinstance(error, OSError) else "INPUT_INVALID"
        _diagnostic(root, stage, code)
        return WorkerResult(request.task_id, request.request_id, "blocked_input", "WORKER_INPUT_BLOCKED")
    except Exception:
        _diagnostic(root, stage, "RUNTIME_FAILED")
        return WorkerResult(request.task_id, request.request_id, "failed_runtime", "WORKER_FAILED")
    finally:
        csv.field_size_limit(old_limit)

    _diagnostic(root, "COMPLETE", "PASSED")
    return WorkerResult(request.task_id, request.request_id, "succeeded", "WORKER_EXITED_OK", ("prepared",))


def main():
    """복사한 런타임과 고정 패키지만 사용한다."""
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
