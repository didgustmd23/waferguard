"""승인된 분할 위치대로 CSV를 스트리밍 추출하는 격리용 진입점. 분할 재생성·전처리는 하지 않는다."""

import hashlib
from pathlib import Path
import sys


class BoundedCsvWriter:
    """CSV writer의 문자열을 제한된 UTF-8 바이트로 저장하고 동시에 해시한다."""

    def __init__(self, stream, maximum):
        self.stream = stream
        self.maximum = maximum
        self.size = 0
        self.digest = hashlib.sha256()

    def write(self, text):
        payload = text.encode("utf-8")

        if self.size + len(payload) > self.maximum:
            raise ValueError("구간 출력의 크기 상한을 초과했습니다.")

        self.stream.write(payload)
        self.digest.update(payload)
        self.size += len(payload)

        return len(text)


def execute_request(request, root):
    """고정 경로·원본·분할·설정을 대조하고 Train·Validation만 새 파일에 기록한다."""
    if request.policy_version == "partition_export_v2":
        from src.agents.tabular_worker_input import execute_export
        return execute_export(request, root)
    from contextlib import ExitStack
    from dataclasses import asdict
    import csv
    import json
    import os

    from src.agents.input_snapshot import Snapshot, SNAPSHOT_VERSION, _scan
    from src.agents.partition_csv import iter_csv_rows
    from src.agents.snapshot_eda_worker import _diagnostic, _verify_staged_snapshot
    from src.agents.worker_contract import WorkerResult, _decode
    from src.preprocessing.profiles import profile_from_dict, _unique_pairs
    from src.preprocessing.split import _digest

    root = Path(root).resolve()
    source = root / "input/snapshot/input.bin"
    settings = root / "input/export.json"
    split_file = root / "input/split.json"
    output = root / "output/result"
    stage = "REQUEST_PATHS"
    old_field_limit = csv.field_size_limit(1024 * 1024)

    try:
        if (
            request.task_id != root.name or request.stage != "ingest" or request.action_id != "run_ingest"
            or request.policy_version != "partition_export_v1" or request.dependency_directories
            or request.input_files != (str(source), str(source.parent / "snapshot.json"), str(settings), str(split_file))
            or request.output_directory != str(output) or request.scratch_directory != str(root / "scratch/task")
        ):
            raise ValueError("내보내기 작업의 고정 요청 경로가 일치하지 않습니다.")

        stage = "SETTINGS"

        with settings.open("rb") as stream:
            packet = _decode(stream.read(65537).decode("utf-8"))

        if set(packet) != {"version", "snapshot", "profile", "limits", "split_evidence_id", "split_file_sha256", "max_partition_bytes", "code_id"}:
            raise ValueError("내보내기 요청의 설정 필드가 올바르지 않습니다.")

        expected = Snapshot(**packet["snapshot"])
        profile = profile_from_dict(packet["profile"])
        maximum = packet["max_partition_bytes"]

        if (
            packet["version"] != "partition_export_v1" or _digest(packet) != request.config_id
            or packet["limits"] != asdict(request.limits) or profile.adapter_id != "tabular_csv_v1"
            or request.source_id != "csv:" + expected.sha256
            or type(maximum) is not int or maximum < 1
        ):
            raise ValueError("내보내기 원본·설정·자원 계약이 일치하지 않습니다.")

        stage = "SNAPSHOT"

        if not _verify_staged_snapshot(root, expected):
            raise ValueError("내보내기 원본 복사본이 변경되었습니다.")

        with split_file.open("rb") as stream:
            raw_split = stream.read(8 * 1024 * 1024 + 1)

        if len(raw_split) > 8 * 1024 * 1024 or hashlib.sha256(raw_split).hexdigest() != packet["split_file_sha256"]:
            raise ValueError("내보내기 분할 파일이 승인 대상과 다릅니다.")

        proposal = json.loads(raw_split.decode("utf-8"), object_pairs_hook=_unique_pairs)

        if (
            proposal["schema_version"] != "split_proposal_v1" or proposal["status"] != "passed"
            or proposal["source_id"] != request.source_id or proposal["config_id"] != profile.signature
            or proposal["rules"] != asdict(profile.split_rules)
            or proposal["proposal_id"] != _digest({key: value for key, value in proposal.items() if key != "proposal_id"})
        ):
            raise ValueError("원본·프로파일과 확정 분할이 일치하지 않습니다.")

        partitions = proposal["partitions"]

        if type(partitions) is not dict or set(partitions) != {"train", "validation", "test"}:
            raise ValueError("확정 분할 구간 목록이 올바르지 않습니다.")

        all_positions = []

        for positions in partitions.values():
            if type(positions) is not list or not positions or any(type(value) is not int or value < 0 for value in positions):
                raise ValueError("분할 위치 목록이 올바르지 않습니다.")

            if positions != sorted(set(positions)):
                raise ValueError("분할 위치가 중복되거나 정렬되지 않았습니다.")

            all_positions.extend(positions)

        count = len(all_positions)

        if count > 1_000_000 or sorted(all_positions) != list(range(count)):
            raise ValueError("분할 위치에 중복·누락이 있거나 한도를 초과했습니다.")

        destinations = {}

        for name in ("train", "validation"):
            destinations.update({position: name for position in partitions[name]})

        stage = "CSV_LOAD"
        target = output / "exports"
        written = {"train": 0, "validation": 0}

        with source.open(encoding="utf-8-sig", newline="") as stream, ExitStack() as stack:
            reader = iter_csv_rows(stream)
            header = next(reader, [])

            if not header or any(not name for name in header) or len(header) != len(set(header)) or not set(profile.roles.required_columns).issubset(header):
                raise ValueError("원본 CSV 헤더와 역할 계약이 일치하지 않습니다.")

            target.mkdir(exist_ok=False)
            streams, bounded, writers = {}, {}, {}

            for name in ("train", "validation"):
                directory = target / name
                directory.mkdir()
                streams[name] = stack.enter_context((directory / "input.bin").open("xb"))
                bounded[name] = BoundedCsvWriter(streams[name], maximum)
                writers[name] = csv.writer(bounded[name], lineterminator="\n")
                writers[name].writerow(header)

            observed = 0

            for index, row in enumerate(reader):
                if index >= count or len(row) != len(header):
                    raise ValueError("원본 행 수·컬럼 수와 분할 계약이 다릅니다.")

                name = destinations.get(index)

                if name is not None:
                    # 값·문자열 표현은 그대로 두고 CSV 구분자 인용만 표준 writer로 기록한다.
                    writers[name].writerow(row)
                    written[name] += 1

                observed += 1

            if observed != count or any(written[name] != len(partitions[name]) for name in written):
                raise ValueError("원본 또는 출력의 행 수가 확정 분할과 다릅니다.")

            for stream in streams.values():
                stream.flush()
                os.fsync(stream.fileno())

        stage = "SNAPSHOT_RECHECK"

        if not _verify_staged_snapshot(root, expected):
            raise ValueError("내보내기 중 원본 복사본이 변경되었습니다.")

        records = {}

        for name in ("train", "validation"):
            snapshot = Snapshot(bounded[name].size, bounded[name].digest.hexdigest(), maximum)

            with (target / name / "input.bin").open("rb") as stream:
                if _scan(stream, maximum) != (snapshot.size_bytes, snapshot.sha256):
                    raise ValueError("내보낸 복사본의 바이트 검증에 실패했습니다.")

            with (target / name / "snapshot.json").open("x", encoding="utf-8") as stream:
                json.dump({"version": SNAPSHOT_VERSION, **asdict(snapshot)}, stream)

            records[name] = {"row_count": written[name], "row_positions_sha256": _digest(partitions[name]), "snapshot": asdict(snapshot)}

        with (target / "partition_exports.json").open("x", encoding="utf-8") as stream:
            json.dump({
                "schema_version": "partition_exports_v1", "split_evidence_id": packet["split_evidence_id"],
                "source_id": proposal["source_id"], "proposal_id": proposal["proposal_id"],
                "roles": asdict(profile.roles), "partitions": records,
            }, stream, ensure_ascii=False, allow_nan=False)
    except (ValueError, TypeError, KeyError, UnicodeError, OSError, csv.Error) as error:
        code = "ACCESS_DENIED" if isinstance(error, PermissionError) else "IO_FAILED" if isinstance(error, OSError) else "INPUT_INVALID"
        _diagnostic(root, stage, code)
        return WorkerResult(request.task_id, request.request_id, "blocked_input", "WORKER_INPUT_BLOCKED")
    except Exception:
        _diagnostic(root, stage, "RUNTIME_FAILED")
        return WorkerResult(request.task_id, request.request_id, "failed_runtime", "WORKER_FAILED")
    finally:
        csv.field_size_limit(old_field_limit)

    _diagnostic(root, "COMPLETE", "PASSED")
    return WorkerResult(request.task_id, request.request_id, "succeeded", "WORKER_EXITED_OK", ("exports",))


def main():
    """격리 런타임의 고정 진입점. 실행 관리자·승인·Job 연결은 상위 계층의 책임이다."""
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
