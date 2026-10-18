"""검토된 충돌 맵만 제외하여 새 WM canonical을 생성하는 비파괴 코어."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import zipfile

from scripts.wm_sandbox.export_canonical import OutputBudget
from .validate_canonical import (
    CLASS_NAMES, decode_json, read_json, validate_manifest, validate_metadata,
)


def derive_canonical(source, output, workspace, evidence, max_output_bytes):
    """manifest·NPY 바이트를 보존하며 제외·재색인한다. 승인·OS 권한은 호출자가 관리한다.

    evidence에는 검토한 후보의 선택 파일 지문과 전후 집계가 들어간다.
    NPZ의 숫자 NPY 본문은 재계산하지 않고 그대로 복사하고 이름만 새 위치에 맞춘다.
    성공 반환 뒤에도 호출자는 validate_bundle로 새 배열·해시·집계를 전부 재검증해야 한다.
    """
    source, output, workspace = map(Path, (source, output, workspace))

    if output.exists() or output.is_symlink() or output.resolve().is_relative_to(source.resolve()):
        raise ValueError("원본 밖의 존재하지 않는 새 canonical 경로가 필요합니다.")

    summary, _version = validate_metadata(source)
    database = sqlite3.connect(str(workspace / "derive.sqlite"))

    try:
        validate_manifest(source, summary, _version, database)
        database.execute("CREATE TEMP TABLE excluded_hashes AS SELECT hash FROM rows WHERE hash IS NOT NULL "
                         "GROUP BY hash HAVING COUNT(DISTINCT label)>1")
        database.execute("CREATE UNIQUE INDEX excluded_hash_index ON excluded_hashes(hash)")
        hashes = hashlib.sha256()
        hash_bytes = 0
        row_hash = hashlib.sha256(b"source_row_position\n")
        row_bytes = len(b"source_row_position\n")
        excluded = 0

        # 같은 개수라도 다른 맵·행을 제외하지 않도록 검토한 선택 파일의 바이트 지문과 대조한다.
        for (value,) in database.execute("SELECT hash FROM excluded_hashes ORDER BY hash"):
            payload = (json.dumps({"map_hash": value}) + "\n").encode("ascii")
            hashes.update(payload)
            hash_bytes += len(payload)

        for (position,) in database.execute(
            "SELECT position FROM rows WHERE hash IN (SELECT hash FROM excluded_hashes) ORDER BY position",
        ):
            payload = (str(position) + "\n").encode("ascii")
            row_hash.update(payload)
            row_bytes += len(payload)
            excluded += 1

        if ({"size": hash_bytes, "sha256": hashes.hexdigest()} != evidence["selection_files"]["conflict_hashes.jsonl"]
            or {"size": row_bytes, "sha256": row_hash.hexdigest()} != evidence["selection_files"]["excluded_rows.csv"]
            or excluded != evidence["excluded_rows"] or excluded <= 0
            or summary["row_count"] - excluded != evidence["hypothetical"]["manifest_rows"]):
            raise ValueError("실제 제외 대상이 검토한 후보와 다릅니다.")

        output.mkdir(exist_ok=False)
        (output / "maps").mkdir()
        budget = OutputBudget(max_output_bytes)
        counts, reasons, classes = Counter(), Counter(), Counter()
        parent_manifest = hashlib.sha256()

        with (source / "manifest.jsonl").open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                parent_manifest.update(block)

        # 출처는 원본 pickle이 아니라 이번 변환의 직접 입력인 부모 manifest로 명시한다.
        source_hash = parent_manifest.hexdigest()
        version = "wm811k_" + source_hash
        database.execute("CREATE TEMP TABLE retained (old_key TEXT PRIMARY KEY, new_key TEXT, shard TEXT)")
        retained = 0

        with budget.open(output / "manifest.jsonl") as stream:
            for (record,) in database.execute(
                "SELECT record FROM rows WHERE hash IS NULL OR hash NOT IN (SELECT hash FROM excluded_hashes) "
                "ORDER BY position",
            ):
                row = decode_json(record)
                row["parent_sample_id"] = row["sample_id"]
                row["parent_source_row_position"] = row["source_row_position"]
                row["dataset_version"] = version
                row["sample_id"] = f"{version}:{retained}"
                row["source_row_position"] = retained
                reference = row["map_ref"]

                if reference is not None:
                    new_key = f"map_{retained:09d}"
                    database.execute("INSERT INTO retained VALUES (?,?,?)", (reference["key"], new_key, reference["shard"]))
                    row["map_ref"] = {"shard": reference["shard"], "key": new_key}

                stream.write((json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8"))
                counts[row["quality_status"]] += 1
                counts[row["label_status"]] += 1
                reasons.update(row["reason_codes"])

                if row["label_status"] == "labeled":
                    classes[row["pattern_label"]] += 1

                retained += 1

        if retained <= 0:
            raise ValueError("제외 후 남은 행이 없습니다.")

        shards = [row[0] for row in database.execute("SELECT DISTINCT shard FROM retained ORDER BY shard")]

        for shard in shards:
            mappings = dict(database.execute("SELECT old_key,new_key FROM retained WHERE shard=?", (shard,)))

            # ZIP 경로로 디스크에 풀지 않는다. 기존 검증기의 크기·키 제약 안에서만 NPY를 복사한다.
            with zipfile.ZipFile(source / shard) as original, budget.open(output / shard) as writer, \
                    zipfile.ZipFile(writer, "w", compression=zipfile.ZIP_DEFLATED) as derived:
                members = original.infolist()

                if len(members) > 1000 or len({member.filename for member in members}) != len(members):
                    raise ValueError("원본 shard의 구성원 수 또는 중복이 올바르지 않습니다.")

                by_name = {member.filename: member for member in members}

                for old_key, new_key in mappings.items():
                    member = by_name[old_key + ".npy"]

                    if member.flag_bits & 1 or not 0 < member.file_size <= 1_010_000:
                        raise ValueError("원본 NPY의 암호화·크기 계약이 올바르지 않습니다.")

                    derived.writestr(new_key + ".npy", original.read(member))

        updated = {**summary, "row_count": retained, "counts": dict(counts), "reason_counts": dict(reasons),
                   "class_counts": dict(classes), "shard_count": len(shards), "split_complete": False,
                   "duplicate_audit_complete": False, "training_ready": False,
                   "label_coverage_status": "complete" if all(classes[name] > 0 for name in CLASS_NAMES) else "incomplete"}
        parent_provenance = read_json(source / "provenance.json")
        provenance = {"source_sha256": source_hash, "source_bytes": (source / "manifest.jsonl").stat().st_size,
                      "source_kind": "parent_canonical_manifest", "parent_source_id": evidence["source_id"],
                      "original_source_sha256": parent_provenance["source_sha256"],
                      "candidate_evidence_id": evidence["candidate_evidence_id"],
                      "operation": "exclude_all_conflicting_map_rows_v1",
                      "created_at": datetime.now(timezone.utc).isoformat(), "training_ready": False}
        budget.write_json(output / "conversion.json", updated)
        budget.write_json(output / "provenance.json", provenance)
        budget.write_json(output / "SUCCESS.json", {"status": "canonical_export_complete", "training_ready": False})
        return {"retained_rows": retained, "excluded_rows": excluded, "source_id": evidence["source_id"]}
    finally:
        database.close()
