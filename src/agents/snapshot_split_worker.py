"""분할 준비용 고정 worker. 품질·EDA·제안·확정만 수행하며 학습·변환은 하지 않는다."""

from pathlib import Path
import sys


def execute_request(request, root):
    """부모가 승인한 복사본·프로파일·제안 해시를 확인하고 등록된 동작만 수행한다."""
    import csv
    from dataclasses import asdict
    import hashlib
    import json
    import pandas as pd

    from src.agents.input_snapshot import Snapshot
    from src.agents.partition_csv import iter_csv_rows
    from src.agents.snapshot_eda_worker import _diagnostic, _verify_staged_snapshot
    from src.agents.worker_contract import WorkerResult, _decode
    from src.preprocessing.contracts import TabularInput
    from src.preprocessing.eda_report import generate_eda_report
    from src.preprocessing.profiles import profile_from_dict, _unique_pairs
    from src.preprocessing.quality import inspect_quality, write_quality_report
    from src.preprocessing.split import propose_split, _digest

    root = Path(root).resolve()
    source = root / "input/snapshot/input.bin"
    settings = root / "input/split_task.json"
    output = root / "output/result"
    stage = "REQUEST_PATHS"
    old_limit = csv.field_size_limit(1024 * 1024)

    try:
        actions = {"audit": "run_audit", "run_eda": "run_eda", "propose_split": "propose_split", "commit_split": "commit_split"}
        expected_files = [str(source), str(source.parent / "snapshot.json"), str(settings)]

        if request.stage == "commit_split":
            expected_files.append(str(root / "input/preview.json"))

        if (
            request.task_id != root.name or actions.get(request.stage) != request.action_id
            or request.policy_version != "snapshot_split_v1" or request.dependency_directories
            or request.input_files != tuple(expected_files)
            or request.output_directory != str(output) or request.scratch_directory != str(root / "scratch/task")
        ):
            raise ValueError("분할 작업의 고정 요청 경로가 다릅니다.")

        stage = "SETTINGS"

        with settings.open("rb") as stream:
            packet = _decode(stream.read(65537).decode("utf-8"))

        if set(packet) != {"snapshot", "profile", "limits", "code_id", "proposal_sha256"} or _digest(packet) != request.config_id:
            raise ValueError("분할 요청 설정이 일치하지 않습니다.")

        snapshot = Snapshot(**packet["snapshot"])
        profile = profile_from_dict(packet["profile"])

        if (
            profile.adapter_id != "tabular_csv_v1" or profile.workflow_id != "split_review_v1"
            or profile.split_rules is None or request.source_id != "csv:" + snapshot.sha256
            or packet["limits"] != asdict(request.limits)
            or (request.stage != "commit_split" and packet["proposal_sha256"] is not None)
        ):
            raise ValueError("분할 원본·프로파일·자원 계약이 다릅니다.")

        stage = "SNAPSHOT"

        if not _verify_staged_snapshot(root, snapshot):
            raise ValueError("원본 복사본이 변경되었습니다.")

        # 먼저 제한된 CSV 파서로 행·열·레코드를 검사한다. 부모는 원본을 파싱하지 않는다.
        stage = "CSV_HEADER"

        with source.open(encoding="utf-8-sig", newline="") as stream:
            reader = iter_csv_rows(stream)
            header = next(reader, [])

            if not header or any(not value for value in header) or len(header) != len(set(header)) or not set(profile.roles.required_columns).issubset(header):
                raise ValueError("CSV 헤더와 역할 계약이 다릅니다.")

            count = 0

            for row in reader:
                count += 1

                if count > 1_000_000 or len(row) != len(header):
                    raise ValueError("CSV 행·열 계약 또는 크기 제한을 벗어났습니다.")

        stage = "CSV_LOAD"
        # ID·그룹·범주·시간의 문자열 표현을 수치로 추정하여 합치지 않는다.
        text_columns = set(profile.roles.identifiers + profile.roles.categorical)
        text_columns.update(value for value in (profile.roles.group, profile.roles.timestamp) if value is not None)
        frame = pd.read_csv(source, encoding="utf-8-sig", dtype={name: "string" for name in text_columns})

        if len(frame) != count or list(frame.columns) != header:
            raise ValueError("안전 파서와 표형 로더의 행·컬럼 결과가 다릅니다.")

        data = TabularInput(frame, profile.roles)
        artifact = {"audit": "quality", "run_eda": "eda", "propose_split": "proposal", "commit_split": "split"}[request.stage]
        destination = output / artifact

        if request.stage == "audit":
            write_quality_report(inspect_quality(data, profile.quality_rules), destination)
        elif request.stage == "run_eda":
            stage = "EDA"
            import matplotlib

            # AppContainer에는 GUI 창을 열지 않는다. pyplot을 불러오기 전에
            # 파일 저장 전용 Agg를 고정하여 기본 Tk/Qt 렌더러 선택을 막는다.
            # 테스트 환경에서 우연히 Agg가 선택되는 것에 의존하지 않는다.
            matplotlib.use("Agg", force=True)
            result = generate_eda_report(data, destination, max_plot_features=profile.max_plot_features)

            if result != {"status": "eda_complete"}:
                raise RuntimeError("EDA 결과 상태가 올바르지 않습니다.")
        else:
            proposal = propose_split(data, profile.split_rules, source_id=request.source_id, config_id=profile.signature)
            destination.mkdir()

            if request.stage == "commit_split":
                with (root / "input/preview.json").open("rb") as stream:
                    raw = stream.read(8 * 1024 * 1024 + 1)

                if len(raw) > 8 * 1024 * 1024 or hashlib.sha256(raw).hexdigest() != packet["proposal_sha256"]:
                    raise ValueError("승인된 분할 제안의 해시가 다릅니다.")

                expected = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_pairs)

                if proposal != expected or proposal["status"] != "passed":
                    raise ValueError("동일 설정의 재계산 결과가 승인 제안과 다릅니다.")

                assignments = [None] * count

                for name, positions in proposal["partitions"].items():
                    for position in positions:
                        assignments[position] = name

                with (destination / "assignments.csv").open("x", encoding="utf-8", newline="") as stream:
                    writer = csv.writer(stream, lineterminator="\n")
                    writer.writerow(["row_position", "split"])
                    writer.writerows(enumerate(assignments))

                filename = "split.json"
            else:
                filename = "proposal.json"
                lines = ["# 분할 제안", "", f"방식: `{profile.split_rules.strategy}`", f"상태: `{proposal['status']}`", "",
                         "| 구간 | 행 수 | 비율 |", "| --- | ---: | ---: |"]
                lines.extend(f"| {row['split']} | {row['rows']} | {row['ratio']:.2%} |" for row in proposal["summary"])
                lines.extend(["", "상세 배정·클래스 분포는 proposal.json에서 로컬로 검토하세요.",
                              "기존 ML 분할을 변경하지 않는 기능 검증용 분할입니다. 아직 확정되지 않았습니다."])
                (destination / "proposal.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

            (destination / filename).write_text(json.dumps(proposal, ensure_ascii=False, allow_nan=False), encoding="utf-8")

        stage = "SNAPSHOT_RECHECK"

        if not _verify_staged_snapshot(root, snapshot):
            raise ValueError("작업 중 원본 복사본이 변경되었습니다.")
    except (ValueError, TypeError, KeyError, OSError, UnicodeError, csv.Error) as error:
        code = "ACCESS_DENIED" if isinstance(error, PermissionError) else "IO_FAILED" if isinstance(error, OSError) else "INPUT_INVALID"
        _diagnostic(root, stage, code)
        return WorkerResult(request.task_id, request.request_id, "blocked_input", "WORKER_INPUT_BLOCKED")
    except Exception:
        _diagnostic(root, stage, "RUNTIME_FAILED")
        return WorkerResult(request.task_id, request.request_id, "failed_runtime", "WORKER_FAILED")
    finally:
        csv.field_size_limit(old_limit)

    _diagnostic(root, "COMPLETE", "PASSED")
    return WorkerResult(request.task_id, request.request_id, "succeeded", "WORKER_EXITED_OK", (artifact,))


def main():
    """고정 복사 패키지만 불러오고 요청 식별값을 확인한다."""
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
