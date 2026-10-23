"""배열 표본 묶음(.npz)의 로딩·진단·분할 입력·Train 기준 산출·적용 계산 코어.

에이전트·pickle·도메인(웨이퍼) 규칙을 모른다. npz는 allow_pickle=False로만 읽고, 배열을 읽기 전에 zip 항목 수·
선언된 압축 해제 크기·각 배열 헤더(dtype·shape)를 상한과 대조한다. 기준값은 호출자가 넘긴 Train 표본에서만 계산한다.
설계 기준: docs/agents/DESIGN.md "배열·이미지 통합 작업 연결"
"""

import hashlib
import math
import re
import struct
import zipfile
import zlib

import numpy as np


MAX_MEMBERS = 32
MAX_TOTAL_BYTES = 2 * 1024 ** 3
MAX_SAMPLES = 1_000_000
MAX_ELEMENTS = 500_000_000
MAX_CHANNELS = 16
MAX_CLASSES = 256
PREVIEW_LIMIT = 12
PREVIEW_SIDE = 256
VALUE_KINDS = "biuf"
META_KINDS = {"label": "biufU", "group": "biuU", "sample_id": "biuU"}
CHANNEL_AXES = ("none", "last", "first")
KEY = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.\-]{0,63}")


class ArrayInputError(ValueError):
    """입력 배열 때문에 진행할 수 없는 경우. 진단에는 고정 코드만 남기고 값·경로는 남기지 않는다."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


# ==========================================
# npz 헤더 검사와 배열 읽기
# - 배열 본문을 읽기 전에 zip 항목·선언 크기·NPY 헤더를 확인한다
# - object·구조체 dtype, 암호화 항목, 하위 경로 이름은 거부한다
# ==========================================

def read_headers(path, max_total=MAX_TOTAL_BYTES):
    """키별 dtype·shape. 배열 값을 읽지 않는다."""
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()

        if not infos or len(infos) > MAX_MEMBERS:
            raise ArrayInputError("NPZ_MEMBERS_INVALID", "npz 항목 수가 허용 범위를 벗어났습니다.")

        if sum(info.file_size for info in infos) > max_total:
            raise ArrayInputError("NPZ_TOO_LARGE", "npz의 압축 해제 크기가 상한을 넘습니다.")

        headers = {}

        for info in infos:
            name = info.filename

            # 키가 .npy로 끝나면 np.load의 이름 해석과 겹칠 수 있어 받지 않는다(예: a.npy와 a.npy.npy).
            if (not name.endswith(".npy") or KEY.fullmatch(name[:-4]) is None or name[:-4].endswith(".npy")
                    or info.flag_bits & 0x1):
                raise ArrayInputError("NPZ_MEMBER_INVALID", "npz 항목 이름 또는 형식이 올바르지 않습니다.")

            with archive.open(info) as stream:
                version = np.lib.format.read_magic(stream)

                if version not in ((1, 0), (2, 0)):
                    raise ArrayInputError("NPY_HEADER_INVALID", "지원하지 않는 NPY 헤더 버전입니다.")

                reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, _, dtype = reader(stream, max_header_size=10000)
                offset = stream.tell()

            if (dtype.hasobject or dtype.names is not None or dtype.kind not in "biufU" or dtype.subdtype is not None
                    or dtype.itemsize == 0):
                raise ArrayInputError("NPY_DTYPE_UNSUPPORTED", "숫자·bool·문자열 배열만 지원합니다.")

            # 원소 크기가 0이 아니어도 원소 수 자체로 메모리를 쓰는 계산(결측 표시 등)이 있어 원소 수도 제한한다.
            if any(type(value) is not int or value < 0 for value in shape) or math.prod(shape) > MAX_ELEMENTS:
                raise ArrayInputError("NPY_HEADER_INVALID", "배열 형상이 허용 범위를 벗어났습니다.")

            if len(shape) > 5 or info.file_size != offset + math.prod(shape) * dtype.itemsize:
                raise ArrayInputError("NPY_HEADER_INVALID", "NPY 헤더와 항목 크기가 다릅니다.")

            headers[name[:-4]] = {"dtype": dtype.str, "kind": dtype.kind, "shape": [int(value) for value in shape]}

    return headers


def load_array(path, key, header):
    """헤더를 확인한 한 배열만 읽는다. pickle 복원은 허용하지 않는다."""
    with np.load(path, allow_pickle=False) as archive:
        array = archive[key]

    if array.dtype.str != header["dtype"] or list(array.shape) != header["shape"]:
        raise ArrayInputError("NPY_HEADER_INVALID", "읽은 배열이 헤더와 다릅니다.")

    return array


# ==========================================
# 매핑: 표본 배열·채널 축·라벨·그룹·표본 ID
# ==========================================

def sample_candidates(headers):
    return sorted(key for key, item in headers.items() if item["kind"] in VALUE_KINDS and 2 <= len(item["shape"]) <= 5)


def meta_candidates(headers, samples):
    count = headers[samples]["shape"][0] if samples in headers else None
    return sorted(key for key, item in headers.items() if key != samples and len(item["shape"]) == 1 and item["shape"][0] == count)


def base_mapping(headers):
    """형상만 보고 정한 기본 매핑. 이름 단서로 라벨·그룹을 추정하지 않는다. 표본 후보가 하나가 아니면 비워 둔다."""
    candidates = sample_candidates(headers)
    return {"samples": candidates[0] if len(candidates) == 1 else None, "channel_axis": "none",
            "label": None, "group": None, "sample_id": None}


def check_mapping(headers, mapping):
    """매핑이 헤더 계약에 맞는지 확인하고 (표본 수, 채널 수)를 돌려준다."""
    if type(mapping) is not dict or set(mapping) != {"samples", "channel_axis", "label", "group", "sample_id"}:
        raise ArrayInputError("MAPPING_INVALID", "매핑 항목 구성이 계약과 다릅니다.")

    samples = mapping["samples"]

    if samples not in sample_candidates(headers):
        raise ArrayInputError("MAPPING_SAMPLES_INVALID", "표본 배열은 2~5차원 숫자 배열이어야 합니다.")

    shape = headers[samples]["shape"]
    count = shape[0]

    if not 1 <= count <= MAX_SAMPLES:
        raise ArrayInputError("SAMPLE_COUNT_INVALID", "표본 수가 허용 범위를 벗어났습니다.")

    axis = mapping["channel_axis"]

    if axis not in CHANNEL_AXES:
        raise ArrayInputError("MAPPING_INVALID", "채널 축 선택이 올바르지 않습니다.")

    channels = 1 if axis == "none" else shape[-1] if axis == "last" else shape[1] if len(shape) >= 3 else 0

    if axis != "none" and (len(shape) < 3 or not 1 <= channels <= MAX_CHANNELS):
        raise ArrayInputError("MAPPING_CHANNEL_INVALID", "채널 축은 3차원 이상 배열에서 길이 1~16인 축만 고를 수 있습니다.")

    used = [samples]

    for role in ("label", "group", "sample_id"):
        key = mapping[role]

        if key is None:
            continue

        item = headers.get(key)

        if key in used or item is None or item["shape"] != [count] or item["kind"] not in META_KINDS[role]:
            raise ArrayInputError("MAPPING_META_INVALID", "라벨·그룹·표본 ID는 표본 수와 길이가 같은 1차원 배열이어야 합니다.")

        used.append(key)

    return count, channels


def channel_first(array, axis):
    """(N, C, ...) 보기. 원본 배열은 바꾸지 않는다."""
    if axis == "none":
        return array[:, None]

    return np.moveaxis(array, -1, 1) if axis == "last" else array


def meta_missing(values):
    """라벨 결측: 실수 NaN 또는 공백 문자열."""
    if values.dtype.kind == "f":
        return np.isnan(values)

    if values.dtype.kind == "U":
        return np.char.strip(values) == ""

    return np.zeros(len(values), dtype=bool)


def meta_text(value):
    """메타데이터 값의 로컬 표시용 문자열. 실수 정수값은 정수로 쓴다."""
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))

    return str(value.item() if isinstance(value, np.generic) else value)


def sample_hashes(array):
    """표본별 바이트 해시. 같은 dtype·형상 안에서 완전히 같은 표본만 같은 값이 된다."""
    flat = np.ascontiguousarray(array.reshape(len(array), -1))
    return [hashlib.sha256(row.tobytes()).hexdigest() for row in flat]


# ==========================================
# 진단 (결정 A): 매핑과 무관하게 키별로 계산한다
# - min·max는 로컬 표시용이며 모델 전송 집계에서 빠진다(어댑터 책임)
# ==========================================

def _value_stats(values):
    finite = np.isfinite(values) if values.dtype.kind == "f" else np.ones(values.shape, dtype=bool)
    observed = values[finite].astype(np.float64)
    distinct = int(len(np.unique(observed))) if observed.size else 0
    return {"values": int(values.size), "nonfinite": int(values.size - observed.size),
            "zero_ratio": float(np.mean(observed == 0)) if observed.size else None,
            "mean": float(observed.mean()) if observed.size else None, "std": float(observed.std()) if observed.size else None,
            "min": float(observed.min()) if observed.size else None, "max": float(observed.max()) if observed.size else None,
            "distinct": min(distinct, MAX_CLASSES + 1)}


def _sample_summary(array):
    flat = array.reshape(len(array), -1)
    values = flat.astype(np.float64) if flat.dtype.kind != "f" else flat
    low, high = np.nanmin(values, axis=1) if values.size else [], np.nanmax(values, axis=1) if values.size else []
    hashes = sample_hashes(array)
    summary = {"overall": _value_stats(array), "constant_samples": int(np.sum(np.asarray(low) == np.asarray(high))),
               "duplicate_samples": len(hashes) - len(set(hashes)), "channels": {}}

    for axis in ("last", "first"):
        if array.ndim >= 3 and 1 <= (array.shape[-1] if axis == "last" else array.shape[1]) <= MAX_CHANNELS:
            view = channel_first(array, axis)
            summary["channels"][axis] = [_value_stats(view[:, index]) for index in range(view.shape[1])]

    return summary


def _meta_summary(values):
    missing = meta_missing(values)
    observed = values[~missing]
    keys, counts = np.unique(observed, return_counts=True) if observed.size else (np.array([]), np.array([], dtype=int))
    result = {"missing": int(missing.sum()), "unique": int(len(keys)),
              "group_sizes": ({"min": int(counts.min()), "median": float(np.median(counts)), "max": int(counts.max())}
                              if len(counts) else None)}

    if len(keys) <= MAX_CLASSES:
        result["classes"] = [{"value": meta_text(key), "count": int(count)} for key, count in zip(keys, counts)]

    return result


def diagnose(path, max_total=MAX_TOTAL_BYTES):
    """키별 진단 요약과 기본 매핑. 표본 배열 후보는 값·표본 통계, 1차원 배열은 결측·고유값·분포를 센다."""
    headers = read_headers(path, max_total)
    samples, meta = {}, {}

    for key, header in sorted(headers.items()):
        if key in sample_candidates(headers):
            samples[key] = _sample_summary(load_array(path, key, header))
        elif len(header["shape"]) == 1:
            meta[key] = _meta_summary(load_array(path, key, header))

    return {"headers": headers, "samples": samples, "meta": meta, "base_mapping": base_mapping(headers)}


# ==========================================
# 표본 미리보기: 입력 확인용 로컬 PNG. 표시용 밝기 조정만 하고 데이터에 적용하지 않는다
# ==========================================

def _png(pixels):
    """8비트 회색조(H,W) 또는 RGB(H,W,3) PNG. 외부 이미지 라이브러리 없이 고정 형식으로 쓴다."""
    height, width = pixels.shape[:2]
    color = 2 if pixels.ndim == 3 else 0
    raw = b"".join(b"\x00" + pixels[row].tobytes() for row in range(height))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def _display(sample):
    """표본 하나를 표시 가능한 2차원(회색) 또는 RGB 배열로. 지원하지 않는 형상이면 None."""
    if sample.ndim == 3 and sample.shape[-1] in (1, 3):
        image = sample if sample.shape[-1] == 3 else sample[..., 0]
    elif sample.ndim == 3 and sample.shape[0] in (1, 3):
        image = np.moveaxis(sample, 0, -1) if sample.shape[0] == 3 else sample[0]
    elif sample.ndim == 2:
        image = sample
    else:
        return None

    values = image.astype(np.float64)
    finite = np.isfinite(values)
    low, high = (values[finite].min(), values[finite].max()) if finite.any() else (0.0, 0.0)
    scaled = np.where(finite, (values - low) / (high - low) if high > low else 0.0, 0.0)
    step = max(1, math.ceil(max(scaled.shape[:2]) / PREVIEW_SIDE))
    return np.ascontiguousarray((scaled[::step, ::step] * 255).round().astype(np.uint8))


def previews(array):
    """표본 순서 기준 앞에서부터 최대 PREVIEW_LIMIT개. (위치, PNG 바이트) 목록. 형상이 이미지가 아니면 빈 목록."""
    result = []

    for position in range(min(len(array), PREVIEW_LIMIT)):
        image = _display(array[position])

        if image is None:
            return []

        result.append((position, _png(image)))

    return result


# ==========================================
# 분할 입력 (결정 C): 배열 값이 아닌 표본 메타데이터로만 나눈다
# ==========================================

def _components(keys_list):
    """여러 키(그룹 값·표본 해시)를 공유하는 표본을 한 묶음으로 합친다(union-find)."""
    count = len(keys_list[0])
    parent = list(range(count))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for keys in keys_list:
        first = {}

        for index, key in enumerate(keys):
            if key in first:
                parent[find(index)] = find(first[key])
            else:
                first[key] = index

    return [find(index) for index in range(count)]


def split_frame(kept, labels, groups, ids, hashes, *, use_label, duplicate_grouping):
    """split.propose_split이 받을 표본 메타데이터 표와 역할. 위치 열은 계약상 필요한 특징 자리이며 분할 기준이 아니다."""
    import pandas as pd

    from .contracts import ColumnRoles, TabularInput

    frame = pd.DataFrame({"__position": np.asarray(kept, dtype=np.int64),
                          "__sample_id": [ids[index] for index in kept]})
    target = group = None

    if use_label:
        frame["__label"] = [labels[index] for index in kept]
        target = "__label"

    sources = []

    if groups is not None:
        sources.append([groups[index] for index in kept])

    if duplicate_grouping:
        sources.append([hashes[index] for index in kept])

    if sources:
        frame["__group"] = ["g" + str(value) for value in _components(sources)] if len(sources) > 1 or duplicate_grouping else sources[0]
        group = "__group"

    roles = ColumnRoles(numeric=("__position",), target=target, group=group, identifiers=("__sample_id",))
    return TabularInput(frame, roles)


# ==========================================
# Train 기준 산출과 적용 (결정 C·D)
# - 순서: 비유한 값 대치 → 분위수 clipping → 크기 조정. 기준은 채널별
# ==========================================

def derive_state(train, options):
    """Train (N, C, ...)에서 채널별 기준을 만든다. Validation·Test는 받지 않는다."""
    channels = []

    for index in range(train.shape[1]):
        values = train[:, index].astype(np.float64).ravel()
        finite = np.isfinite(values)

        if not finite.all() and options["nonfinite"] == "block":
            raise ArrayInputError("NONFINITE_PRESENT", "Train에 비유한 값이 있습니다. 대치 방법을 고르거나 입력을 확인하세요.")

        if not finite.any():
            raise ArrayInputError("CHANNEL_ALL_NONFINITE", "유한값이 없는 채널은 기준을 만들 수 없습니다.")

        impute = float(np.median(values[finite])) if options["nonfinite"] == "median" else None
        prepared = np.where(finite, values, impute if impute is not None else 0.0)
        clip = None

        if options["clip"] == "percentile":
            clip = [float(value) for value in np.percentile(prepared, [options["clip_low"], options["clip_high"]])]
            prepared = np.clip(prepared, *clip)

        center, scale = 0.0, 1.0

        if options["scaling"] == "standard":
            center, scale = float(prepared.mean()), float(prepared.std()) or 1.0
        elif options["scaling"] == "minmax":
            center, scale = float(prepared.min()), float(prepared.max() - prepared.min()) or 1.0
        elif options["scaling"] == "fixed":
            scale = float(options["fixed_divisor"])

        record = {"channel": index, "impute": impute, "clip": clip, "center": center, "scale": scale}

        if not np.isfinite([center, scale] + (clip or []) + ([impute] if impute is not None else [])).all():
            raise ArrayInputError("STATE_NONFINITE", "기준 계산 결과가 유한수가 아닙니다.")

        channels.append(record)

    return {"version": "array_preparation_v1", "channels": channels}


def apply_state(batch, state, options):
    """저장 기준만으로 (N, C, ...) 배치를 float32로 바꾸고 채널별 영향 건수를 센다. 새 통계는 계산하지 않는다."""
    output = np.empty(batch.shape, dtype=np.float32)
    impact = []

    for record in state["channels"]:
        index = record["channel"]
        values = batch[:, index].astype(np.float64)
        finite = np.isfinite(values)
        nonfinite = int(values.size - finite.sum())

        if nonfinite and record["impute"] is None:
            raise ArrayInputError("NONFINITE_PRESENT", "비유한 값이 있지만 대치 방법이 없습니다.")

        prepared = np.where(finite, values, record["impute"] if record["impute"] is not None else 0.0)
        clipped = None

        if record["clip"] is not None:
            low, high = record["clip"]
            clipped = int(np.sum((prepared < low) | (prepared > high)))
            prepared = np.clip(prepared, low, high)

        result = ((prepared - record["center"]) / record["scale"]).astype(np.float32)

        if not np.isfinite(result).all():
            raise ArrayInputError("OUTPUT_NONFINITE", "변환 결과에 비유한 값이 생겼습니다.")

        output[:, index] = result
        flat = result.ravel()
        impact.append({"channel": index, "values": int(values.size), "nonfinite": nonfinite,
                       "imputed": nonfinite if record["impute"] is not None else 0, "clipped": clipped,
                       "constant_after": bool(flat.size == 0 or (flat == flat[0]).all())})

    return output, impact


def impact_totals(impact):
    """채널별 영향의 합계 정수. 적용 기록과 후보 기록의 교차 대조에 쓴다."""
    return {"values": sum(item["values"] for item in impact), "imputed": sum(item["imputed"] for item in impact),
            "clipped": sum(item["clipped"] or 0 for item in impact),
            "constant_channels": sum(1 for item in impact if item["constant_after"])}


def restore_layout(array, axis):
    """(N, C, ...)를 입력과 같은 축 순서로 되돌린다."""
    if axis == "none":
        return array[:, 0]

    return np.moveaxis(array, 1, -1) if axis == "last" else array
