"""공통 데이터 변환기와 WM 도메인 변환기를 정적 ID로 조립한다."""

import numpy as np
import pandas as pd

from scripts.wm_sandbox.export_canonical import OutputBudget, export_bundle
from src.preprocessing.canonical_data import (
    encode_scalar, export_array, export_table, supported_array, supported_table,
)
from .artifacts import _fingerprint


CONVERTER_IDS = frozenset({"tabular_v1", "numeric_array_v1", "wm811k_v1"})
ROOT_KINDS = frozenset({"dataframe", "ndarray", "dict", "list", "tuple", "unsupported"})
REVIEW_CODES = frozenset({
    "MULTIPLE_CANDIDATES", "UNSUPPORTED_STRUCTURE", "STRUCTURE_LIMIT",
    "SELECT_PATH_INVALID", "CONVERTER_MISMATCH",
})
REVIEW_MARKDOWN = ("# Pickle 구조 선택 필요\n\n"
                  "변환 후보를 자동 확정하지 않았습니다. structure.json의 후보 경로와 변환기를 확인하세요.\n\n"
                  "새 실행에서 --select-path 또는 --converter를 지정하고 새 승인을 받으세요.\n"
                  "이 보고서는 로컬 전용입니다. 데이터 값·구조 보고서를 LLM에 전달하지 마세요.\n")


def root_kind(value):
    types = {pd.DataFrame: "dataframe", np.ndarray: "ndarray", dict: "dict", list: "list", tuple: "tuple"}

    return types.get(type(value), "unsupported")


def _converter(value):
    """값의 의미 대신 등록된 구조 서명으로 변환 후보를 분류한다."""
    if type(value) is pd.DataFrame:
        # WM 전용 클래스·픽셀 규칙은 기존 변환기에 남겨 두며 이 서명으로만 연결한다.
        if {"waferMap", "failureType", "lotName"}.issubset(value.columns):
            return "wm811k_v1"

        return "tabular_v1" if supported_table(value) else None

    if supported_array(value):
        return "numeric_array_v1"

    return None


def _resolve(value, path):
    """dict 키·list/tuple 인덱스만 따라간다. 속성 접근이나 표현식을 실행하지 않는다."""
    for key in path:
        if type(value) is dict and type(key) is str:
            value = value[key]
        elif type(value) in (list, tuple) and type(key) is int:
            value = value[key]
        else:
            raise ValueError("선택 경로와 객체 구조가 일치하지 않습니다.")

    return value


def analyze_structure(value, converter_id="auto", select_path=()):
    """한 개의 지원 객체는 선택하고 모호함·검사 상한 초과는 로컬 검토로 남긴다."""
    report = {"version": "pickle_structure_v1", "status": "review_required",
              "root_kind": root_kind(value), "reason": "UNSUPPORTED_STRUCTURE",
              "candidates": [], "selected_path": None, "converter_id": None}

    try:
        selected_root = _resolve(value, select_path)
    except (ValueError, KeyError, IndexError):
        report["reason"] = "SELECT_PATH_INVALID"
        return None, report

    nodes = 0
    limited = False
    unsupported = False

    def visit(node, path):
        nonlocal nodes, limited, unsupported
        nodes += 1

        if nodes > 2048 or len(path) > 4 or len(report["candidates"]) >= 64:
            limited = True
            return

        candidate = _converter(node)

        # 변환기를 명시한 경우에는 자동 구조 서명보다 해당 변환기의 허용 구조를 우선한다.
        if converter_id == "tabular_v1" and type(node) is pd.DataFrame and supported_table(node):
            candidate = converter_id

        if candidate is not None:
            descriptor = {"path": list(path), "converter_id": candidate, "kind": root_kind(node)}

            if type(node) is pd.DataFrame:
                descriptor.update(rows=len(node), columns=len(node.columns))
            else:
                descriptor.update(shape=list(node.shape), dtype=node.dtype.str)

            report["candidates"].append(descriptor)
            return

        if type(node) is dict:
            if len(node) > 256 or any(type(key) is not str or not 0 < len(key) <= 256 for key in node):
                limited = True
                return

            for key, child in node.items():
                visit(child, (*path, key))
        elif type(node) in (list, tuple):
            if len(node) > 2048:
                limited = True
                return

            for index, child in enumerate(node):
                visit(child, (*path, index))
        else:
            try:
                encode_scalar(node)
            except ValueError:
                unsupported = True

    visit(selected_root, tuple(select_path))
    candidates = report["candidates"]
    eligible = [item for item in candidates if converter_id == "auto" or item["converter_id"] == converter_id]

    if limited:
        report["reason"] = "STRUCTURE_LIMIT"
    elif unsupported:
        report["reason"] = "UNSUPPORTED_STRUCTURE"
    elif len(eligible) > 1:
        report["reason"] = "MULTIPLE_CANDIDATES"
    elif len(eligible) == 0:
        report["reason"] = "CONVERTER_MISMATCH" if candidates else "UNSUPPORTED_STRUCTURE"
    else:
        choice = eligible[0]
        report.update(status="selected", reason="STRUCTURE_RECOGNIZED",
                      selected_path=choice["path"], converter_id=choice["converter_id"])

        return _resolve(value, choice["path"]), report

    return None, report


def export_selected(value, output, packet, report, request):
    """선택된 객체만 변환하고 선택 경로·원본 보존 범위를 출처 정보에 기록한다."""
    selected = {"converter_id": report["converter_id"], "selected_path": report["selected_path"],
                "selection_scope": "selected_object", "request_id": request.request_id,
                "unselected_original": "input_snapshot"}
    snapshot = packet["snapshot"]

    if report["converter_id"] == "wm811k_v1":
        export_bundle(value, output, snapshot["sha256"], snapshot["size_bytes"], execution_mode="appcontainer",
                      max_output_bytes=packet["max_output_bytes"], shard_rows=packet["shard_rows"],
                      extra_provenance=selected)
        return

    budget = OutputBudget(packet["max_output_bytes"])
    output.mkdir(exist_ok=False)
    converter = {"tabular_v1": export_table, "numeric_array_v1": export_array}[report["converter_id"]]
    files = list(converter(value, output, budget))
    budget.write_json(output / "source.json", {
        "source_sha256": snapshot["sha256"], "source_bytes": snapshot["size_bytes"],
        "execution_mode": "appcontainer", **selected,
    })
    files.append("source.json")
    budget.write_json(output / "manifest.json", {
        "version": "canonical_bundle_v1", "converter_id": report["converter_id"],
        "files": {name: _fingerprint(output / name) for name in files},
    })
    budget.write_json(output / "SUCCESS.json", {"status": "canonical_export_complete", "training_ready": False})


def export_review(output, report, request, maximum):
    """구조 후보와 고정 안내만 로컬에 보존한다. 완료 표식을 만들지 않는다."""
    budget = OutputBudget(maximum)
    output.mkdir(exist_ok=False)
    budget.write_json(output / "structure.json", {
        **report, "request_id": request.request_id, "source_id": request.source_id,
    })
    with budget.open(output / "structure.md") as stream:
        stream.write(REVIEW_MARKDOWN.encode("utf-8"))
