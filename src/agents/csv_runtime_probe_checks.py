"""합성 CSV 검사 결과를 부모가 고정 기대값과 대조한다. 지원 정책은 바꾸지 않는다."""

import csv
from io import BytesIO, StringIO
import json

from .artifacts import _fingerprint
from .csv_runtime_probe_tasks import MAXIMUM, profile
from .eda_result_validation import validate_eda_report
from .isolated_eda import _read_file
from .quality_result_validation import read_quality_report
from .split_evidence import read_metadata, read_split_files
from .worker_contract import WorkerResult


ARTIFACTS = {"audit": "quality", "run_eda": "eda", "propose_split": "proposal", "commit_split": "split",
             "ingest": "exports", "derive_preparation": "candidate", "apply_preparation": "prepared"}


def launch_passed(launch, limits):
    resources = launch.get("resources", {})
    return (launch.get("status") == "exited" and launch.get("exit_code") == 0
        and resources.get("status") == "succeeded" and resources.get("cleanup_confirmed") is True
        and resources.get("active_processes") == 0
        and resources.get("configured_job_memory_bytes") == limits.memory_mib * 1024 ** 2
        and resources.get("configured_process_limit") == limits.processes)


def validate_reply(root, task, family, request, launch):
    if not launch_passed(launch, request.limits):
        return False
    reply = WorkerResult.from_json(_read_file(task / "output/result/worker_result.json", 65536).decode("utf-8"))
    receipt = read_metadata(task / "output/probe_receipt.json")
    return (reply.matches(request) and reply.status == "succeeded" and reply.artifact_refs == (ARTIFACTS[request.stage],)
        and receipt == {"schema_version": "csv_probe_child_v1", "probe_id": root.name,
            "request_id": request.request_id, "family": family, "isolated_token_and_job": True,
            "python": receipt.get("python")}
        and type(receipt["python"]) is list and len(receipt["python"]) == 3
        and receipt["python"][:2] == [3, 12] and all(type(value) is int for value in receipt["python"]))


def _rows(path):
    return list(csv.reader(StringIO(_read_file(path, MAXIMUM).decode("utf-8-sig"))))


def verify_outputs(root, outputs):
    """CSV 원본은 고정 합성 파일만 읽는다. 일반 데이터에 대한 독립 수학 증명은 아니다."""
    import numpy as np
    import pandas as pd
    from src.preprocessing.transform import build_preparation, PreparationRules

    selected = profile()
    quality = read_quality_report(outputs["audit"] / "quality", selected)
    proposal = read_split_files(outputs["propose_split"] / "proposal", "csv:" + _fingerprint(root / "synthetic.csv")["sha256"],
                                selected.signature, preview=True)
    split = read_split_files(outputs["commit_split"] / "split", proposal["source_id"], selected.signature)
    partitions = {"train": list(range(84)), "validation": list(range(84, 102)), "test": list(range(102, 120))}
    raw = _rows(root / "synthetic.csv")
    exports = outputs["ingest"] / "exports"
    candidate = outputs["derive_preparation"] / "candidate"
    prepared = outputs["apply_preparation"] / "prepared"
    saved = read_metadata(candidate / "preprocessor.json")
    application = read_metadata(prepared / "application.json")
    expected = build_preparation(selected.roles, PreparationRules(scaling="standard"))
    frame = pd.read_csv(BytesIO(_read_file(root / "synthetic.csv", MAXIMUM)), dtype={"category": "string", "id": "string", "time": "string"})
    expected.fit(frame.iloc[partitions["train"]])
    state = json.loads(json.dumps(expected.named_steps["preprocess"].state_))
    checks = {
        "quality_passed": quality is not None and quality["status"] == "passed",
        "eda_report_verified": validate_eda_report(outputs["run_eda"] / "eda", selected),
        "split_matches_proposal": proposal == split and split["partitions"] == partitions,
        "export_train_matches": _rows(exports / "train/input.bin") == [raw[0]] + [raw[index + 1] for index in partitions["train"]],
        "export_validation_matches": _rows(exports / "validation/input.bin") == [raw[0]] + [raw[index + 1] for index in partitions["validation"]],
        "candidate_train_state_matches": saved["state"] == state,
        "application_no_refit_or_test": application.get("refit") is False and application.get("test_used") is False and application.get("model_trained") is False,
        "no_test_outputs": not (exports / "test").exists() and not (prepared / "test_features.csv").exists(),
    }
    for name in ("train", "validation"):
        actual = pd.read_csv(BytesIO(_read_file(prepared / (name + "_features.csv"), MAXIMUM)))
        wanted = expected.transform(frame.iloc[partitions[name]])
        rows = read_metadata(prepared / (name + "_rows.json"))
        checks[name + "_values_and_order"] = (list(actual.columns) == list(wanted.columns)
            and actual.shape == wanted.shape and bool(np.allclose(actual.to_numpy(), wanted.to_numpy(), rtol=1e-12, atol=1e-12))
            and rows["row_positions"] == partitions[name])
    return checks
