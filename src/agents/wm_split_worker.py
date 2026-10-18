"""완료 EDA의 같은 canonical에서 그룹 분할안만 생성하는 격리 worker."""

from pathlib import Path
import sys


def execute_request(request, root):
    from dataclasses import asdict
    import json
    import sqlite3

    from scripts.wm_sandbox.export_canonical import OutputBudget
    from src.agents.pickle_import_worker import _diagnostic
    from src.agents.wm_canonical_contract import WmCanonicalInput, digest, inventory
    from src.agents.wm_split_results import POLICY_VERSION, proposal_files
    from src.agents.worker_contract import WorkerResult, _decode
    from src.preprocessing.canonical_data import decode_json
    from src.wafer_dl.split import SPLITS, build_groups, choose_assignment, load_split_config, verify_assignments
    from src.wafer_dl.validate_canonical import assess_split_feasibility, validate_manifest, validate_metadata

    root = Path(root).absolute()
    source = root / "input/canonical"
    settings, manifest = root / "input/analysis.json", root / "input/files.json"
    stage = "REQUEST"

    try:
        if (request.task_id != root.name or request.stage != "derive_preparation"
            or request.action_id != "derive_preparation" or request.policy_version != POLICY_VERSION
            or request.input_files != (str(settings), str(manifest)) or request.dependency_directories != (str(source),)
            or request.output_directory != str(root / "output/result")
            or request.scratch_directory != str(root / "scratch/task")):
            raise ValueError("WM 분할안 요청이 올바르지 않습니다.")

        with settings.open("rb") as stream:
            packet = _decode(stream.read(65537).decode("utf-8"))

        if (set(packet) != {"version", "source_id", "limits", "max_input_bytes", "max_output_bytes",
                            "include_examples", "max_group_ratio", "code_id", "split_evidence"}
            or packet["version"] != POLICY_VERSION or packet["source_id"] != request.source_id
            or packet["limits"] != asdict(request.limits) or digest(packet) != request.config_id
            or packet["include_examples"] is not False):
            raise ValueError("WM 분할안 설정이 승인 범위와 다릅니다.")

        with manifest.open("rb") as stream:
            files = decode_json(stream.read(1024 * 1024 + 1))

        contract = WmCanonicalInput(str(source), json.dumps(files), request.limits,
                                    packet["max_input_bytes"], packet["max_output_bytes"], False, packet["max_group_ratio"])
        evidence = packet["split_evidence"]

        if (contract.source_id != request.source_id or evidence["source_id"] != request.source_id
            or evidence["evidence_id"] != digest({key: value for key, value in evidence.items() if key != "evidence_id"})
            or inventory(source, contract.max_input_bytes, staged=True) != files):
            raise ValueError("WM 분할안 입력과 EDA 증거가 다릅니다.")

        config_path = root / "scratch/task/split_config.json"
        config_path.write_text(json.dumps(evidence["split_config"]), encoding="utf-8")
        config = load_split_config(config_path)

        if config["max_group_ratio"] != contract.max_group_ratio:
            raise ValueError("EDA 검토 상한과 분할안 검토 상한이 다릅니다.")

        stage = "MANIFEST"
        summary, version = validate_metadata(source)

        if summary.get("label_normalization_version") != "wm_label_v2":
            raise ValueError("검증된 최신 라벨 정규화 계약이 필요합니다.")

        database = sqlite3.connect(str(root / "scratch/task/proposal.sqlite"))

        try:
            validate_manifest(source, summary, version, database)
            counts, audit, groups, class_groups = build_groups(database)
            assessment = assess_split_feasibility(audit, groups, class_groups, config["max_group_ratio"])
            conflicts = database.execute("SELECT COUNT(*) FROM conflicts").fetchone()[0]

            # 기존 코어는 충돌 격리를 지원하지만, 이 어댑터는 추가 자동 제외를 금지한다.
            if (conflicts != 0 or assessment["status"] != "preconditions_passed"
                or summary["row_count"] != evidence["quality"]["row_count"]
                or audit != evidence["quality"]["duplicate_audit"]
                or assessment != evidence["quality"]["split_assessment"]):
                raise ValueError("분할안 기준이 완료 EDA의 품질 결과와 다릅니다.")

            stage = "PROPOSAL"
            (score, selected, assignment, matrix), candidates = choose_assignment(counts, config)
            database.execute("CREATE TABLE assignments(gid INTEGER PRIMARY KEY, split TEXT)")
            database.executemany("INSERT INTO assignments VALUES (?,?)", (
                (gid, SPLITS[int(value)] if value >= 0 else None) for gid, value in enumerate(assignment)
            ))
            verify_assignments(database)
            proposal = Path(request.output_directory) / "proposal"
            proposal.mkdir(exist_ok=False)
            budget = OutputBudget(contract.max_output_bytes)
            report = {"status": "proposal_ready", "split_complete": False, "training_ready": False,
                      "source_modified": False, "arrays_read": False, "overlap_check_passed": True,
                      "source_id": request.source_id, "config": config, "row_count": summary["row_count"],
                      "excluded_conflict_groups": conflicts, "group_count": len(assignment),
                      "selected_candidate": selected, "objective": score, "candidates": candidates,
                      "split_counts": {name: int(matrix[index].sum()) for index, name in enumerate(SPLITS)},
                      "class_counts": {name: matrix[index].tolist() for index, name in enumerate(SPLITS)}}
            budget.write_json(proposal / "proposal.json", report)

            with budget.open(proposal / "groups.csv") as stream:
                stream.write(("group_id,split," + ",".join(f"class_{index}" for index in range(9)) + "\n").encode("ascii"))

                for gid, value in enumerate(assignment):
                    name = SPLITS[int(value)] if value >= 0 else ""
                    line = ",".join([f"group_{gid:08d}", name] + [str(int(count)) for count in counts[gid]]) + "\n"
                    stream.write(line.encode("ascii"))

            lines = ["# WM 그룹 분할 제안", "", "분할안만 생성했습니다. 확정·변환·학습은 하지 않았습니다.", "",
                     "Lot·동일 맵 연결 그룹을 유지하며 표본·클래스 비율 편차로 고정 후보를 비교했습니다.", "",
                     "| 클래스 번호 | Train | Validation | Test |", "| --- | ---: | ---: | ---: |"]

            for index in range(9):
                lines.append("| " + " | ".join([str(index)] + [str(int(matrix[split, index])) for split in range(3)]) + " |")

            total = int(matrix.sum())
            lines += ["", "## 실제 지도학습 표본 비율", ""]

            for index, name in enumerate(SPLITS):
                lines.append(f"- {name}: {int(matrix[index].sum())}건 / {matrix[index].sum() / total:.2%}")

            lines += ["", "미라벨 전용 그룹의 빈 배정은 학습 허가가 아닙니다.",
                      "분할 확정에는 보고서 검토와 별도 승인이 필요합니다. 상세 집계는 로컬에서만 확인하세요."]

            with budget.open(proposal / "proposal.md") as stream:
                stream.write(("\n".join(lines) + "\n").encode("utf-8"))
        finally:
            database.close()

        if inventory(source, contract.max_input_bytes, staged=True) != files:
            raise ValueError("분할안 생성 중 입력 복사본이 변경되었습니다.")

        records = proposal_files(proposal, contract.max_output_bytes)
        budget.write_json(proposal / "result.json", {"version": POLICY_VERSION, "request_id": request.request_id,
                          "source_id": request.source_id, "config_id": request.config_id,
                          "evidence_id": digest(evidence), "status": "proposal_ready", "files": records})
        _diagnostic(root, "COMPLETE", "PASSED")
        return WorkerResult(request.task_id, request.request_id, "succeeded", "WORKER_EXITED_OK", ("proposal",))
    except Exception:
        _diagnostic(root, stage, "SPLIT_PROPOSAL_FAILED")
        return WorkerResult(request.task_id, request.request_id, "blocked_input", "WORKER_INPUT_BLOCKED")


def main():
    """기존 AppContainer 토큰·Job 확인 후에만 요청을 읽는다."""
    import os

    if len(sys.argv) != 3:
        return 2

    root = Path(sys.argv[1]).absolute()
    sys.path.insert(0, str(root / "tools"))
    from src.agents.pickle_import_worker import _isolated_process

    if not _isolated_process():
        return 2

    runtime = root / "runtime"
    sys.path.insert(1, str(runtime / "Lib/site-packages"))
    handles = [os.add_dll_directory(str(path)) for path in
               (runtime, runtime / "DLLs", runtime / "Library/bin") if path.is_dir()]
    from src.agents.worker_contract import WorkerRequest

    try:
        with (root / "input/request.json").open("rb") as stream:
            request = WorkerRequest.from_json(stream.read(65537).decode("utf-8"))

        if request.request_id != sys.argv[2]:
            return 2

        reply = execute_request(request, root)
        (root / "output/result/worker_result.json").write_text(reply.to_json(), encoding="utf-8")
        return 0 if reply.status == "succeeded" else 2
    except Exception:
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
