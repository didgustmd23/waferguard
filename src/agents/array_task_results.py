"""배열 통합 작업 worker 출력의 부모 검증기. 원본 npz를 다시 열지 않고 결과 기록의 형식·범위·계보를 검사한다.

검증은 형식·일관성 검사다. 진단 통계·clipping 건수처럼 원본 없이 다시 셀 수 없는 값은 범위만 보고, 출력 배열에서
다시 셀 수 있는 값(유한성·상수 채널·표본 수·순서)은 정확히 대조한다. 하나라도 다르면 None을 돌려 등록하지 않는다.
"""

import math
from pathlib import Path
import struct

import numpy as np

from src.preprocessing.array_dataset import (KEY, MAX_CHANNELS, MAX_CLASSES, MAX_MEMBERS, PREVIEW_LIMIT,
                                             base_mapping, check_mapping, impact_totals, sample_candidates)
from . import strict_json
from .artifacts import _is_link
from .isolated_eda import _read_file, _valid_png


MAX_RECORD_BYTES = 64 * 1024 * 1024
MAX_PREVIEW_BYTES = 512 * 1024
SPLITS = ("train", "validation", "test")


def _json(path):
    return strict_json.loads(_read_file(Path(path), MAX_RECORD_BYTES).decode("utf-8"), MAX_RECORD_BYTES)


def _int(value, low=0, high=None):
    return type(value) is int and value >= low and (high is None or value <= high)


def _number(value, nullable=False):
    return (value is None and nullable) or (type(value) in (int, float) and math.isfinite(value))


def _names(directory):
    return {path.name for path in Path(directory).iterdir()}


def _scalar(value):
    """메타데이터 표시 값: 문자열(200자 이하)·bool·정수·유한 실수·null만 받는다. 중첩 JSON은 받지 않는다."""
    return value is None or (type(value) is str and len(value) <= 200) or type(value) in (bool, int) or (type(value) is float and math.isfinite(value))


def _preview_png(data):
    """미리보기 고정 형식: 8비트 회색·RGB, 한 변 256 이하, IHDR·IDAT·IEND 청크만. 공통 PNG 검사도 함께 통과해야 한다."""
    if not _valid_png(data) or data[:8] != b"\x89PNG\r\n\x1a\n":
        return False

    offset, kinds = 8, []

    while offset + 12 <= len(data):
        length, kind = struct.unpack(">I4s", data[offset:offset + 8])
        kinds.append(kind)

        if kind == b"IHDR":
            if length != 13 or offset != 8:
                return False

            width, height, depth, color, compression, method, interlace = struct.unpack(">IIBBBBB", data[offset + 8:offset + 21])

            if not (1 <= width <= 256 and 1 <= height <= 256 and depth == 8 and color in (0, 2) and (compression, method, interlace) == (0, 0, 0)):
                return False

        offset += 12 + length

    return offset == len(data) and kinds[:1] == [b"IHDR"] and kinds[-1:] == [b"IEND"] and set(kinds) <= {b"IHDR", b"IDAT", b"IEND"}


# ==========================================
# 진단 요약 (결정 A)
# ==========================================

def _stats(item, count):
    fields = {"values", "nonfinite", "zero_ratio", "mean", "std", "min", "max", "distinct"}

    if type(item) is not dict or set(item) != fields or not _int(item["values"]) or (count is not None and item["values"] != count):
        return False

    empty = item["values"] == item["nonfinite"]
    return (_int(item["nonfinite"], 0, item["values"]) and _int(item["distinct"], 0, MAX_CLASSES + 1)
            and all(_number(item[key], nullable=empty) and (item[key] is None) == empty for key in ("zero_ratio", "mean", "std", "min", "max"))
            and (empty or (0 <= item["zero_ratio"] <= 1 and item["std"] >= 0 and item["min"] <= item["mean"] <= item["max"])))


def _valid_headers(headers):
    if type(headers) is not dict or not 1 <= len(headers) <= MAX_MEMBERS:
        return False

    return all(type(key) is str and KEY.fullmatch(key) and type(item) is dict and set(item) == {"dtype", "kind", "shape"}
               and type(item["dtype"]) is str and len(item["dtype"]) <= 32 and item["kind"] in "biufU" and len(item["kind"]) == 1
               and np.dtype(item["dtype"]).kind == item["kind"] and type(item["shape"]) is list and len(item["shape"]) <= 5
               and all(_int(value) for value in item["shape"]) for key, item in headers.items())


def read_profile(directory):
    """진단 요약과 미리보기를 검증해 돌려준다. 미리보기 PNG 바이트는 담지 않고 파일 이름만 남긴다."""
    try:
        root = Path(directory)
        summary = _json(root / "summary.json")

        if type(summary) is not dict or set(summary) != {"headers", "samples", "meta", "base_mapping", "previews"}:
            return None

        headers = summary["headers"]

        if not _valid_headers(headers) or summary["base_mapping"] != base_mapping(headers):
            return None

        candidates = sample_candidates(headers)

        if type(summary["samples"]) is not dict or set(summary["samples"]) != set(candidates):
            return None

        for key, item in summary["samples"].items():
            shape = headers[key]["shape"]

            if (type(item) is not dict or set(item) != {"overall", "constant_samples", "duplicate_samples", "channels"}
                    or not _stats(item["overall"], math.prod(shape)) or not _int(item["constant_samples"], 0, shape[0])
                    or not _int(item["duplicate_samples"], 0, max(shape[0] - 1, 0)) or type(item["channels"]) is not dict):
                return None

            for axis, values in item["channels"].items():
                size = shape[-1] if axis == "last" else shape[1]

                if (axis not in ("last", "first") or len(shape) < 3 or not 1 <= size <= MAX_CHANNELS or type(values) is not list
                        or len(values) != size or not all(_stats(value, math.prod(shape) // size) for value in values)):
                    return None

        meta_keys = {key for key, item in headers.items() if len(item["shape"]) == 1 and key not in candidates}

        if type(summary["meta"]) is not dict or set(summary["meta"]) != meta_keys:
            return None

        for key, item in summary["meta"].items():
            count = headers[key]["shape"][0]

            if (type(item) is not dict or set(item) - {"classes"} != {"missing", "unique", "group_sizes"}
                    or not _int(item["missing"], 0, count) or not _int(item["unique"], 0, count - item["missing"])):
                return None

            sizes = item["group_sizes"]

            # 그룹 크기는 화면에 진단 사실로 보인다. 형식과 값의 순서·범위를 확인한다.
            if (sizes is None) != (item["unique"] == 0) or (sizes is not None and (
                    type(sizes) is not dict or set(sizes) != {"min", "median", "max"} or not _int(sizes["min"], 1)
                    or not _int(sizes["max"], sizes["min"], count) or not _number(sizes["median"])
                    or not sizes["min"] <= sizes["median"] <= sizes["max"])):
                return None

            classes = item.get("classes")

            if (item["unique"] <= MAX_CLASSES) != (classes is not None) or (classes is not None and (
                    type(classes) is not list or len(classes) != item["unique"]
                    or any(type(entry) is not dict or set(entry) != {"value", "count"} or type(entry["value"]) is not str
                           or len(entry["value"]) > 200 or not _int(entry["count"], 1) for entry in classes)
                    or sum(entry["count"] for entry in classes) != count - item["missing"])):
                return None

        previews = summary["previews"]
        samples = summary["base_mapping"]["samples"]
        expected = {"summary.json"} | ({"previews"} if samples is not None else set())

        if _names(root) != expected or type(previews) is not list or len(previews) > PREVIEW_LIMIT:
            return None

        if samples is not None:
            names = []

            for index, entry in enumerate(previews):
                # 미리보기는 표본 순서 앞에서부터 고른다. 다른 표본을 고르거나 순서를 바꾼 기록은 받지 않는다.
                if (type(entry) is not dict or set(entry) != {"file", "position", "key"} or entry["position"] != index
                        or entry["key"] != samples or entry["file"] != f"preview_{index}.png"):
                    return None

                data = _read_file(root / "previews" / entry["file"], MAX_PREVIEW_BYTES)

                if not _preview_png(data):
                    return None

                names.append(entry["file"])

            if _names(root / "previews") != set(names) or (previews and len(previews) != min(headers[samples]["shape"][0], PREVIEW_LIMIT)):
                return None

        return summary
    except (OSError, ValueError, TypeError, KeyError, IndexError, UnicodeError, RecursionError):
        return None


# ==========================================
# 분할·Train 기준 후보 (결정 C)
# ==========================================

def _shape(packet):
    return packet["headers"][packet["mapping"]["samples"]]["shape"]


def _valid_state(state, packet, channels):
    options = packet["preparation"]

    if type(state) is not dict or set(state) != {"version", "channels"} or state["version"] != "array_preparation_v1":
        return False

    if type(state["channels"]) is not list or len(state["channels"]) != channels:
        return False

    for index, record in enumerate(state["channels"]):
        if type(record) is not dict or set(record) != {"channel", "impute", "clip", "center", "scale"} or record["channel"] != index:
            return False

        if not _number(record["center"]) or not _number(record["scale"]) or record["scale"] <= 0:
            return False

        if (record["impute"] is None) != (options["nonfinite"] == "block") or not _number(record["impute"], nullable=True):
            return False

        clip = record["clip"]

        if (clip is None) != (options["clip"] == "none") or (clip is not None and (
                type(clip) is not list or len(clip) != 2 or not all(_number(value) for value in clip) or clip[0] > clip[1])):
            return False

        if options["scaling"] == "none" and (record["center"], record["scale"]) != (0.0, 1.0):
            return False

        if options["scaling"] == "fixed" and (record["center"], record["scale"]) != (0.0, float(options["fixed_divisor"])):
            return False

    return True


def _valid_impact(impact, state, samples, per_channel):
    fields = {"channel", "values", "nonfinite", "imputed", "clipped", "constant_after"}

    if type(impact) is not list or len(impact) != len(state["channels"]):
        return False

    for item, record in zip(impact, state["channels"]):
        values = samples * per_channel

        if (type(item) is not dict or set(item) != fields or item["channel"] != record["channel"] or item["values"] != values
                or not _int(item["nonfinite"], 0, values) or type(item["constant_after"]) is not bool):
            return False

        if item["imputed"] != (item["nonfinite"] if record["impute"] is not None else 0) or (record["impute"] is None and item["nonfinite"]):
            return False

        if (record["clip"] is None) != (item["clipped"] is None) or (item["clipped"] is not None and not _int(item["clipped"], 0, values)):
            return False

        # 하한과 상한이 같으면 모든 값이 한 값이 된다. 저장 기준으로 결정되는 결과는 대조한다.
        if record["clip"] is not None and record["clip"][0] == record["clip"][1] and not item["constant_after"]:
            return False

    return True


def read_candidate(directory, packet):
    """분할 배정·채널 기준·Train 예상 영향을 검증해 돌려준다. 변환 파일은 없어야 한다."""
    try:
        root = Path(directory)

        if _names(root) != {"assignment.json", "state.json", "candidate.json"}:
            return None

        assignment, state, candidate = (_json(root / name) for name in ("assignment.json", "state.json", "candidate.json"))
        count, channels = check_mapping(packet["headers"], packet["mapping"])
        fields = {"version", "samples", "partitions", "excluded", "summary", "cross_split_duplicates", "group_overlap", "warnings"}

        if (type(assignment) is not dict or set(assignment) != fields or assignment["version"] != "array_assignment_v1"
                or assignment["samples"] != count or type(assignment["partitions"]) is not dict or set(assignment["partitions"]) != set(SPLITS)):
            return None

        positions = []

        for name in SPLITS + ("excluded",):
            values = assignment["excluded"] if name == "excluded" else assignment["partitions"][name]

            if type(values) is not list or values != sorted(set(values)) or not all(_int(value, 0, count - 1) for value in values):
                return None

            positions += values

        # 세 구간과 제외 표본이 겹치지 않고 모든 표본을 정확히 한 번씩 덮어야 한다.
        if sorted(positions) != list(range(count)) or not assignment["partitions"]["train"] or not assignment["partitions"]["validation"]:
            return None

        if (assignment["excluded"] and packet["preparation"]["missing_labels"] != "exclude"
                or not _int(assignment["cross_split_duplicates"]) or not _int(assignment["group_overlap"])
                or (packet["preparation"]["duplicate_grouping"] and assignment["cross_split_duplicates"])
                or (packet["mapping"]["group"] is not None and packet["split"]["strategy"] == "group" and assignment["group_overlap"])
                or type(assignment["warnings"]) is not list or not all(item == "CLASS_MISSING_IN_SPLIT" for item in assignment["warnings"])):
            return None

        summary = assignment["summary"]

        if type(summary) is not list or [item.get("split") if type(item) is dict else None for item in summary] != list(SPLITS):
            return None

        for item in summary:
            if (set(item) != {"split", "samples", "classes", "groups"} or item["samples"] != len(assignment["partitions"][item["split"]])
                    or type(item["classes"]) is not list or (item["groups"] is not None and not _int(item["groups"], 0, item["samples"]))
                    or any(type(entry) is not dict or set(entry) != {"value", "count"} or type(entry["value"]) is not str
                           or len(entry["value"]) > 200 or not _int(entry["count"], 1) for entry in item["classes"])
                    or sum(entry["count"] for entry in item["classes"]) > item["samples"]):
                return None

        shape = _shape(packet)
        per_channel = math.prod(shape[1:]) // channels

        if (not _valid_state(state, packet, channels) or type(candidate) is not dict
                or set(candidate) != {"version", "binding", "impact", "transformed", "model_trained"}
                or candidate["version"] != "array_candidate_v1" or candidate["binding"] != packet
                or candidate["transformed"] is not False or candidate["model_trained"] is not False
                or not _valid_impact(candidate["impact"], state, len(assignment["partitions"]["train"]), per_channel)):
            return None

        return {"assignment": assignment, "state": state, "candidate": candidate}
    except (OSError, ValueError, TypeError, KeyError, IndexError, UnicodeError, RecursionError, AttributeError):
        return None


# ==========================================
# 적용 결과·로컬 전달 묶음 (결정 D)
# ==========================================

def _npy(path, shape):
    """worker가 쓴 NPY를 pickle 없이 읽는다. 링크·하드링크를 거부하고 헤더·크기·float32·유한성을 확인한다."""
    if _is_link(path) or not path.is_file() or path.stat().st_nlink != 1:
        return None

    with path.open("rb") as stream:
        version = np.lib.format.read_magic(stream)

        if version not in ((1, 0), (2, 0)):
            return None

        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        header_shape, fortran, dtype = reader(stream, max_header_size=10000)

        if list(header_shape) != shape or fortran or dtype != np.dtype("<f4") or path.stat().st_size != stream.tell() + math.prod(shape) * 4:
            return None

    # 메모리 맵은 Windows에서 파일 핸들을 오래 잡아 결과 폴더 정리를 막을 수 있어 한 번에 읽는다.
    array = np.load(path, allow_pickle=False)
    return array if np.isfinite(array).all() else None


def read_application(directory, packet, candidate):
    """적용 출력·메타데이터·기준 복사본·실측 합계를 검증한다. Test 출력은 없어야 한다."""
    from src.preprocessing.array_dataset import channel_first

    try:
        root = Path(directory)
        expected = {"train_samples.npy", "validation_samples.npy", "train_meta.json", "validation_meta.json",
                    "preprocessor.json", "application.json"}

        if _names(root) != expected or _json(root / "preprocessor.json") != candidate["state"]:
            return None

        report = _json(root / "application.json")

        if (type(report) is not dict or set(report) != {"version", "binding", "impact", "transformed", "refit", "test_used", "model_trained"}
                or report["version"] != "array_application_v1" or report["binding"] != packet or report["transformed"] is not True
                or report["refit"] is not False or report["test_used"] is not False or report["model_trained"] is not False
                or type(report["impact"]) is not dict or set(report["impact"]) != {"train", "validation"}):
            return None

        shape, mapping = _shape(packet), packet["mapping"]
        fields = {"values", "imputed", "clipped", "constant_channels"}

        for name in ("train", "validation"):
            positions = candidate["assignment"]["partitions"][name]
            array = _npy(root / f"{name}_samples.npy", [len(positions)] + shape[1:])
            meta = _json(root / f"{name}_meta.json")
            totals = report["impact"][name]

            if array is None or type(meta) is not dict or set(meta) != {"positions", "sample_id", "label", "group"} or meta["positions"] != positions:
                return None

            if (type(meta["sample_id"]) is not list or len(meta["sample_id"]) != len(positions)
                    or any(type(value) is not str or len(value) > 200 for value in meta["sample_id"])
                    or any((meta[role] is None) != (mapping[role] is None) or (meta[role] is not None and (
                        type(meta[role]) is not list or len(meta[role]) != len(positions)
                        or not all(_scalar(value) for value in meta[role]))) for role in ("label", "group"))):
                return None

            view = channel_first(array, mapping["channel_axis"])
            constant = sum(1 for index in range(view.shape[1]) if view[:, index].size == 0
                           or bool((np.asarray(view[:, index]) == np.asarray(view[:, index]).flat[0]).all()))

            if (type(totals) is not dict or set(totals) != fields or not all(_int(value) for value in totals.values())
                    or totals["values"] != view.size or totals["constant_channels"] != constant
                    or totals["imputed"] > totals["values"] or totals["clipped"] > totals["values"]):
                return None

        # 같은 Train·같은 기준이므로 후보가 센 Train 합계와 적용 Train 합계가 같아야 한다.
        if report["impact"]["train"] != impact_totals(candidate["candidate"]["impact"]):
            return None

        return report
    except (OSError, ValueError, TypeError, KeyError, IndexError, UnicodeError, RecursionError, AttributeError):
        return None
