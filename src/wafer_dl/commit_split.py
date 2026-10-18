"""검토한 WM 그룹 배정을 재탐색 없이 확정 manifest로 저장하는 코어."""

import csv
import hashlib
import io
import json
from pathlib import Path
import sqlite3

import numpy as np

from scripts.wm_sandbox.export_canonical import OutputBudget
from .check_transform import index_split
from .split import SPLITS, build_groups, verify_assignments
from .validate_canonical import CLASS_NAMES, decode_json, validate_manifest, validate_metadata


def commit_reviewed_split(source, output, workspace, proposal, groups_path, canonical_root, maximum):
    """승인된 배정만 적용한다. 입력 배열·모델·후보 탐색은 실행하지 않는다.

    source는 격리 작업용 복사본이고 canonical_root는 이후 변환기가 참조할 보존된 입력 경로다.
    승인과 두 경로의 동일 지문 검사는 에이전트 경계에서 담당한다.
    """
    source, output, workspace = map(Path, (source, output, workspace))

    if (output.exists() or output.is_symlink() or output.resolve().is_relative_to(source.resolve())
        or source.resolve().is_relative_to(output.resolve())):
        raise ValueError("원본 밖의 존재하지 않는 분할 출력 경로가 필요합니다.")

    if proposal.get("status") != "proposal_ready" or proposal.get("split_complete") is not False:
        raise ValueError("검토할 분할 제안이 필요합니다.")

    summary, version = validate_metadata(source)
    database = sqlite3.connect(str(workspace / "commit.sqlite"))

    try:
        validate_manifest(source, summary, version, database)
        counts, _audit, _groups, _class_groups = build_groups(database)

        if (database.execute("SELECT COUNT(*) FROM conflicts").fetchone()[0] != 0
            or len(counts) != proposal["group_count"] or summary["row_count"] != proposal["row_count"]):
            raise ValueError("검토한 분할안과 원본 그룹·행 수가 다릅니다.")

        database.execute("CREATE TABLE assignments(gid INTEGER PRIMARY KEY, split TEXT)")
        matrix = np.zeros((3, 9), dtype=np.int64)
        group_count = 0

        with Path(groups_path).open("r", encoding="ascii", newline="") as stream:
            reader = csv.reader(stream, strict=True)

            if next(reader, None) != ["group_id", "split"] + [f"class_{index}" for index in range(9)]:
                raise ValueError("검토한 그룹 배정 목록의 헤더가 올바르지 않습니다.")

            for row in reader:
                if (group_count >= len(counts) or len(row) != 11 or row[0] != f"group_{group_count:08d}"
                    or row[1] not in (*SPLITS, "") or any(not value.isascii() or not value.isdecimal()
                                                         or len(value) > 7 for value in row[2:])):
                    raise ValueError("그룹 배정의 순서·중복·표본 수 계약이 올바르지 않습니다.")

                observed = list(map(int, row[2:]))

                if observed != counts[group_count].tolist() or bool(sum(observed)) != bool(row[1]):
                    raise ValueError("검토한 그룹의 표본 구성이 현재 입력과 다릅니다.")

                database.execute("INSERT INTO assignments VALUES (?,?)", (group_count, row[1] or None))

                if row[1]:
                    matrix[SPLITS.index(row[1])] += counts[group_count]

                group_count += 1

        if (group_count != len(counts)
            or {name: matrix[index].tolist() for index, name in enumerate(SPLITS)} != proposal["class_counts"]
            or {name: int(matrix[index].sum()) for index, name in enumerate(SPLITS)} != proposal["split_counts"]):
            raise ValueError("고정 그룹 배정과 검토한 분할 집계가 다릅니다.")

        verify_assignments(database)
        manifest_digest = hashlib.sha256()

        with (source / "manifest.jsonl").open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                manifest_digest.update(block)

        manifest_hash = manifest_digest.hexdigest()
        # 기존 변환·캐시 검증기와 같은 프로토콜 공식을 유지한다.
        manifest_digest.update(json.dumps(proposal["config"], sort_keys=True).encode("utf-8"))
        protocol = "wm_split_" + manifest_digest.hexdigest()
        output.mkdir(exist_ok=False)
        budget = OutputBudget(maximum)
        report = {"status": "split_complete", "split_complete": True, "training_ready": False,
                  "dataset_version": version, "split_protocol_id": protocol, "canonical_root": canonical_root,
                  "source_manifest_sha256": manifest_hash, "config": proposal["config"],
                  "selected_candidate": proposal["selected_candidate"], "objective": proposal["objective"],
                  "candidates": proposal["candidates"], "arrays_revalidated": False,
                  "array_validation_source": "completed_eda_same_inventory", "source_modified": False,
                  "resampled": False, "overlap_check_passed": True,
                  "class_mapping": {str(index): name for index, name in enumerate(CLASS_NAMES)},
                  "excluded_conflict_groups": 0, "excluded_rows": 0, "row_count": summary["row_count"],
                  "split_counts": proposal["split_counts"], "class_counts": proposal["class_counts"]}
        report["warnings"] = [{"split": name, "class": CLASS_NAMES[label], "count": int(matrix[index, label])}
                              for index, name in enumerate(SPLITS) if name != "train" for label in range(9)
                              if matrix[index, label] < proposal["config"]["rare_class_warning_count"]]
        written = 0

        with budget.open(output / "split_manifest.jsonl") as stream:
            query = ("SELECT r.record,g.gid,s.split FROM rows r LEFT JOIN lot_groups g ON r.lot=g.lot "
                     "LEFT JOIN assignments s ON g.gid=s.gid ORDER BY r.position")

            for record, gid, group_split in database.execute(query):
                row = decode_json(record)
                row.update(split_protocol_id=protocol, split_group_id=f"group_{gid:08d}" if gid is not None else None,
                           group_split=group_split, exclusion_reasons=[])

                if row["quality_status"] != "accepted":
                    row["split"] = "quarantined"
                elif row["label_status"] != "labeled":
                    row["split"] = "unlabeled"
                else:
                    if group_split not in SPLITS:
                        raise ValueError("지도학습 행에 검토한 그룹 배정이 없습니다.")

                    row["split"] = group_split

                payload = (json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")

                if len(payload) > 65536:
                    raise ValueError("분할 manifest 행 크기가 검증기 상한을 초과했습니다.")

                stream.write(payload)
                written += 1

        if written != summary["row_count"]:
            raise ValueError("분할 확정 중 행이 누락되었습니다.")

        budget.write_json(output / "split.json", report)
        csv_text = io.StringIO(newline="")
        writer = csv.writer(csv_text)
        writer.writerow(["split", "class_index", "pattern_label", "samples", "split_samples", "split_ratio"])
        total = int(matrix.sum())
        lines = ["# WM 그룹 분할 확정", "", "검토한 그룹 배정을 그대로 사용했습니다. 후보 재탐색은 하지 않았습니다.", "",
                 "| 클래스 | Train | Validation | Test |", "| --- | ---: | ---: | ---: |"]

        for label, name in enumerate(CLASS_NAMES):
            lines.append("| " + " | ".join([name] + [str(int(matrix[index, label])) for index in range(3)]) + " |")

        for index, name in enumerate(SPLITS):
            for label, class_name in enumerate(CLASS_NAMES):
                writer.writerow([name, label, class_name, int(matrix[index, label]), int(matrix[index].sum()),
                                 float(matrix[index].sum() / total)])

        with budget.open(output / "split_summary.csv") as stream:
            stream.write(csv_text.getvalue().encode("utf-8-sig"))

        lines += ["", "원본은 보존했습니다. NPZ 배열은 읽지 않았습니다.",
                  "미라벨은 학습 대상이 아니며 같은 그룹의 Test 미라벨도 학습에 사용하지 마세요.",
                  "이미지 변환·캐시·학습 준비는 아직 완료되지 않았습니다. 상세 집계는 로컬 전용입니다."]

        with budget.open(output / "split_summary.md") as stream:
            stream.write(("\n".join(lines) + "\n").encode("utf-8"))

        # 기존 검증기가 요구하는 내부 표식이다. 에이전트 완료 표식은 재검증 뒤 부모가 별도로 만든다.
        budget.write_json(output / "SUCCESS.json", {"status": "split_complete", "split_protocol_id": protocol,
                                                   "training_ready": False})
    finally:
        database.close()

    # 별도 DB에서 모든 원본 행과 재대조한다. 누락·추가·중복·라벨/참조 변경·그룹 겹침을 검사한다.
    verification = sqlite3.connect(str(workspace / "verify.sqlite"))

    try:
        checked = index_split(source, output, verification)
    finally:
        verification.close()

    if checked != report:
        raise ValueError("분할 확정 후 검증 보고서가 저장 결과와 다릅니다.")

    return report
