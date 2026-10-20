"""고정 합성 CSV로만 기존 worker 요청을 만든다. 사용자 파일·승인 DB는 받지 않는다."""

import csv
from dataclasses import asdict
from datetime import datetime, timedelta
import json
from pathlib import Path
import shutil
import uuid

from .artifacts import _fingerprint
from .input_snapshot import create_snapshot, Snapshot
from .split_evidence import read_metadata
from .worker_contract import WorkerRequest
from src.preprocessing.contracts import ColumnRoles
from src.preprocessing.profiles import DatasetProfile
from src.preprocessing.quality import QualityRules
from src.preprocessing.split import SplitRules, _digest
from src.preprocessing.transform import PreparationRules


STAGES = ("audit", "run_eda", "propose_split", "commit_split", "ingest", "derive_preparation", "apply_preparation")
MAXIMUM = 1024 * 1024


def profile():
    return DatasetProfile("csv_runtime_probe", "tabular_csv_v1", "split_review_v1",
        ColumnRoles(numeric=("x", "y"), categorical=("category",), target="label", timestamp="time", identifiers=("id",)),
        max_plot_features=2, quality_rules=QualityRules(allowed_labels=(0, 1), target_missing="block",
            duplicate_id="block", timestamp_format="%Y-%m-%d %H:%M:%S"),
        split_rules=SplitRules(strategy="time", timestamp_format="%Y-%m-%d %H:%M:%S"))


def fixture(root):
    """외부 경로를 읽지 않고 120행만 생성한다. 결측·수치·범주·라벨·시간·ID를 포함한다."""
    path = root / "synthetic.csv"
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["x", "y", "category", "label", "time", "id"])
        for index in range(120):
            writer.writerow([index, "" if index % 10 == 0 else index * 2, "abc"[index % 3], index % 2,
                             (datetime(2026, 1, 1) + timedelta(hours=index)).strftime("%Y-%m-%d %H:%M:%S"), f"{index:04d}"])
    return path


def _save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)


def _copy_snapshot(source, destination):
    destination.mkdir()
    for name in ("input.bin", "snapshot.json"):
        shutil.copyfile(source / name, destination / name)
    return {key: value for key, value in read_metadata(destination / "snapshot.json").items() if key != "version"}


def prepare_task(root, stage, outputs, limits):
    if stage not in STAGES or set(outputs) != set(STAGES[:STAGES.index(stage)]):
        raise ValueError("고정 합성 검사 순서만 허용합니다.")
    task = root / "tasks" / uuid.uuid4().hex
    task.mkdir()
    for part in ("input", "output", "output/result", "scratch", "scratch/task"):
        (task / part).mkdir()
    source, roles = root / "synthetic.csv", profile().roles
    common = {"limits": asdict(limits), "code_id": "0" * 64}
    rules = asdict(PreparationRules(scaling="standard"))

    if stage in STAGES[:4]:
        snapshot = create_snapshot(source, task / "input/snapshot", max_input_bytes=MAXIMUM)
        packet = {"snapshot": asdict(snapshot), "profile": profile().to_dict(), **common, "proposal_sha256": None}
        files = ["snapshot/input.bin", "snapshot/snapshot.json", "split_task.json"]
        if stage == "commit_split":
            preview = outputs["propose_split"] / "proposal/proposal.json"
            packet["proposal_sha256"] = _fingerprint(preview)["sha256"]
            shutil.copyfile(preview, task / "input/preview.json")
            files.append("preview.json")
        family, settings, policy = "split", "split_task.json", "snapshot_split_v1"
        action = "run_audit" if stage == "audit" else stage
        source_id = "csv:" + snapshot.sha256
    elif stage == "ingest":
        snapshot = create_snapshot(source, task / "input/snapshot", max_input_bytes=MAXIMUM)
        split = outputs["commit_split"] / "split/split.json"
        shutil.copyfile(split, task / "input/split.json")
        packet = {"version": "partition_export_v1", "snapshot": asdict(snapshot), "profile": profile().to_dict(),
                  **common, "split_evidence_id": "1" * 64, "split_file_sha256": _fingerprint(split)["sha256"],
                  "max_partition_bytes": MAXIMUM}
        family, settings, policy = "export", "export.json", "partition_export_v1"
        files = ["snapshot/input.bin", "snapshot/snapshot.json", settings, "split.json"]
        action, source_id = "run_ingest", "csv:" + snapshot.sha256
    elif stage == "derive_preparation":
        snapshot = _copy_snapshot(outputs["ingest"] / "exports/train", task / "input/snapshot")
        packet = {"version": "preparation_candidate_v1", "snapshot": snapshot, "roles": asdict(roles),
            "train_contract_id": "2" * 64, "split_evidence_id": "1" * 64, "export_marker_sha256": "3" * 64,
            "train_rows": 84, "rules": rules, **common, "max_candidate_bytes": MAXIMUM}
        family, settings, policy = "candidate", "candidate_task.json", "preparation_candidate_v1"
        files = ["snapshot/input.bin", "snapshot/snapshot.json", settings]
        action, source_id = stage, Snapshot(**snapshot).source_id
    else:
        positions = read_metadata(outputs["commit_split"] / "split/split.json")["partitions"]
        partitions = {}
        for name in ("train", "validation"):
            snapshot = _copy_snapshot(outputs["ingest"] / "exports" / name, task / "input" / name)
            partitions[name] = {"contract_id": "2" * 64 if name == "train" else "4" * 64,
                "snapshot": snapshot, "rows": len(positions[name]), "positions_sha256": _digest(positions[name])}
        state = outputs["derive_preparation"] / "candidate/preprocessor.json"
        saved = read_metadata(state)
        shutil.copyfile(state, task / "input/preprocessor.json")
        _save(task / "input/positions.json", {name: positions[name] for name in partitions})
        packet = {"version": "preparation_apply_v1", "candidate_id": "5" * 64,
            "preprocessor_sha256": _fingerprint(state)["sha256"], "max_state_bytes": MAXIMUM,
            "roles": asdict(roles), "rules": rules, "output_columns_sha256": _digest(saved["output_columns"]),
            "output_feature_count": len(saved["output_columns"]), "partitions": partitions,
            **common, "max_partition_bytes": MAXIMUM}
        family, settings, policy = "apply", "apply_task.json", "preparation_apply_v1"
        files = ["train/input.bin", "train/snapshot.json", "validation/input.bin", "validation/snapshot.json",
                 "preprocessor.json", "positions.json", settings]
        action, source_id = stage, Snapshot(**partitions["train"]["snapshot"]).source_id

    # JSON과 동일한 tuple/list 표현으로 식별값을 계산한다.
    packet = json.loads(json.dumps(packet))
    _save(task / "input" / settings, packet)
    request = WorkerRequest(task.name, root.name, "synthetic_probe", stage, action, source_id, _digest(packet),
        "csv_probe:" + root.name, policy, tuple(str(task / "input" / name) for name in files), (),
        str(task / "output/result"), str(task / "scratch/task"), limits)
    (task / "input/request.json").write_text(request.to_json(), encoding="utf-8")
    return task, family, request
