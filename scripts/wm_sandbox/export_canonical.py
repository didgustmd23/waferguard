"""명시적 신뢰 승인 기반 원본 변환기. 분할·resize·증강은 수행하지 않는다."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

import numpy as np
import pandas as pd


CLASS_NAMES = ("None", "Center", "Donut", "Edge-Loc", "Edge-Ring",
               "Loc", "Near-Full", "Random", "Scratch")
LABEL_ALIASES = {"none": "None", "Near-full": "Near-Full"}
NORMALIZATION_VERSION = "wm_label_v2"
DEFAULT_MAX_OUTPUT_BYTES = 8 * 1024 ** 3


class OutputBudget:
    """manifest·NPZ·관리 JSON이 공유하는 실제 기록 바이트 상한."""

    def __init__(self, maximum):
        if type(maximum) is not int or maximum < 1:
            raise ValueError("출력 크기 상한은 양의 정수 바이트여야 합니다.")

        self.maximum = maximum
        self.used = 0

    def open(self, path):
        """기존 파일을 덮어쓰지 않는 제한된 바이너리 writer를 만든다."""
        return _BoundedWriter(Path(path).open("xb"), self)

    def write_json(self, path, value):
        """줄바꿈 변환 없이 UTF-8 바이트를 직접 기록한다."""
        payload = json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False).encode("utf-8")

        if self.used + len(payload) > self.maximum:
            raise ValueError("canonical 출력 크기 상한을 초과했습니다.")

        with self.open(path) as stream:
            stream.write(payload)


class _BoundedWriter:
    """ZIP 헤더의 seek·덮어쓰기를 지원하며 파일 크기 증가 전에 상한을 검사한다."""

    def __init__(self, stream, budget):
        self.stream = stream
        self.budget = budget
        self.size = 0

    def write(self, payload):
        # ZIP은 앞선 헤더를 다시 쓰므로 누적 write 길이 대신 파일 최대 길이를 센다.
        end = self.stream.tell() + memoryview(payload).nbytes
        growth = max(0, end - self.size)

        if self.budget.used + growth > self.budget.maximum:
            raise ValueError("canonical 출력 크기 상한을 초과했습니다.")

        written = self.stream.write(payload)
        actual_end = self.stream.tell()
        self.budget.used += max(0, actual_end - self.size)
        self.size = max(self.size, actual_end)

        return written

    def tell(self):
        return self.stream.tell()

    def read(self, size=-1):
        # NumPy는 read 속성으로 파일 객체를 판별한다. 쓰기 전용 스트림의 읽기는 거부된다.
        return self.stream.read(size)

    def seek(self, offset, whence=0):
        return self.stream.seek(offset, whence)

    def flush(self):
        self.stream.flush()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.stream.close()


# ==========================================
# 단일 원소 중첩을 풀어 문자열 라벨 정규화
# - 빈 배열·결측은 미라벨이며 정상 클래스 None과 구분
# - 복수 라벨·알 수 없는 문자열은 격리
# - 원본 클래스 값으로 매핑 순서를 다시 계산하지 않음
# ==========================================
def normalize_label(value):
    for _ in range(8):
        if type(value) is np.ndarray:
            if value.size == 0:
                return None, None, "unlabeled"
            if value.size != 1:
                return None, None, "invalid"
            value = value.reshape(-1)[0]
        elif type(value) in (list, tuple):
            if not value:
                return None, None, "unlabeled"
            if len(value) != 1:
                return None, None, "invalid"
            value = value[0]
        else:
            break

    if value is None or value is pd.NA:
        return None, None, "unlabeled"
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None, None, "unlabeled"
    if not isinstance(value, (str, np.str_)):
        return None, None, "invalid"
    
    label = str(value).strip()

    if not label:
        return None, None, "unlabeled"
    
    # 승인된 원본 표기만 정규화하며 미지 라벨을 임의로 추측하지 않는다.
    label = LABEL_ALIASES.get(label, label)

    if label not in CLASS_NAMES:
        return None, None, "invalid"
    
    return label, CLASS_NAMES.index(label), "labeled"


# ==========================================
# 문자열·숫자형 메타데이터만 보존
# - 임의 객체의 repr·str 메서드를 호출하지 않음
# - 배열형 원본 partition은 단일 원소만 허용
# ==========================================
def scalar_metadata(value):
    for _ in range(8):
        if type(value) is np.ndarray and value.size == 1:
            value = value.reshape(-1)[0]
        else:
            break

    if isinstance(value, (str, np.str_)):
        return str(value).strip() or None
    if type(value) is int or isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        return float(value)
    
    return None


# ==========================================
# 맵을 검증한 뒤 uint8 배열과 내용 식별값 생성
# - 2D·유한값·0/1/2·유효 다이 조건을 먼저 검사
# - 부적절한 값을 clipping·대치하여 정상으로 만들지 않음
# - 해시에 shape를 포함해 배열 모양이 다른 입력을 구분
# ==========================================
def normalize_map(value):
    if type(value) is not np.ndarray or value.ndim != 2 or not value.size:
        return None, "INVALID_MAP_STRUCTURE"
    if value.size > 1_000_000 or value.dtype.kind not in "biuf":
        return None, "INVALID_MAP_SIZE_OR_DTYPE"
    if not np.isfinite(value).all() or not np.isin(value, (0, 1, 2)).all():
        return None, "INVALID_MAP_VALUES"
    if not np.any(value > 0):
        return None, "NO_VALID_DIE"
    
    result = np.ascontiguousarray(value, dtype=np.uint8)

    return result, None


# ==========================================
# 전체 행을 manifest와 숫자 전용 NPZ shard로 변환
# - 모든 원본 행 위치와 격리 사유를 기록
# - 원본 해상도 유지, 가변 크기 맵은 각각 별도 NPZ 키에 저장
# - shard별로 flush해 전체 변환 배열의 추가 메모리 적재 방지
# - 중복 그룹 검사·분할 승인 전 데이터는 학습 준비 완료가 아님
# ==========================================
def export_frame(frame, output, dataset_version, shard_rows=1000, progress=None,
                 max_output_bytes=DEFAULT_MAX_OUTPUT_BYTES, *, budget=None):
    if type(frame) is not pd.DataFrame or frame.columns.has_duplicates:
        raise ValueError("최상위 객체 또는 컬럼 구조가 올바르지 않습니다.")
    
    required = ("waferMap", "failureType", "lotName")

    if any(column not in frame.columns for column in required):
        raise ValueError("필수 컬럼이 없습니다.")
    
    if type(shard_rows) is not int or not 1 <= shard_rows <= 1000:
        raise ValueError("shard 행 수 범위를 벗어났습니다.")
    
    if budget is None:
        budget = OutputBudget(max_output_bytes)
    elif not isinstance(budget, OutputBudget) or budget.maximum != max_output_bytes:
        raise ValueError("출력 바이트 상한과 공유 기록 계약이 일치하지 않습니다.")

    output = Path(output)
    output.mkdir(exist_ok=False)

    maps_dir = output / "maps"
    maps_dir.mkdir()
    shard_index, pending = 0, {}
    counts, reasons, classes = Counter(), Counter(), Counter()
    positions = {name: index for index, name in enumerate(frame.columns)}

    def flush():
        # object 배열을 만들지 않으므로 복원 시 allow_pickle=False 사용 가능.
        nonlocal shard_index

        if pending:
            shard_path = maps_dir / f"shard_{shard_index:06d}.npz"

            # NPZ의 ZIP 헤더와 배열 본문도 같은 상한 안에서 실제 바이트로 기록한다.
            with budget.open(shard_path) as stream:
                np.savez(stream, **pending)

            pending.clear()
            shard_index += 1

    with budget.open(output / "manifest.jsonl") as stream:
        for position, values in enumerate(frame.itertuples(index=False, name=None)):
            label, class_index, label_status = normalize_label(values[positions["failureType"]])
            lot = scalar_metadata(values[positions["lotName"]])
            
            # Lot 식별자 계약은 문자열로 한정하여 숫자/문자열 ID 혼합을 피한다.
            lot = lot if type(lot) is str else None
            wafer = scalar_metadata(values[positions["waferIndex"]]) if "waferIndex" in positions else None
            partition = scalar_metadata(values[positions["trianTestLabel"]]) if "trianTestLabel" in positions else None
            map_value, map_error = normalize_map(values[positions["waferMap"]])
            row_reasons = [map_error] if map_error else []

            if lot is None:
                row_reasons.append("MISSING_OR_INVALID_LOT")

            if label_status == "invalid":
                row_reasons.append("INVALID_LABEL")

            record = {
                "sample_id": f"{dataset_version}:{position}",
                "dataset_version": dataset_version, "source_row_position": position,
                "source_index": scalar_metadata(frame.index[position]),
                "lot_id": lot, "wafer_index": wafer, "source_partition": partition,
                "pattern_label": label, "class_index": class_index, "label_status": label_status,
                "quality_status": "quarantined" if row_reasons else "accepted",
                "reason_codes": row_reasons, "split": "quarantined" if row_reasons else
                    ("unlabeled" if label_status == "unlabeled" else "pending"),
                "map_ref": None, "map_hash": None,
            }

            if map_value is not None:
                # 유효 배열의 내용은 로컬 산출물에만 기록하고 콘솔에 출력하지 않는다.
                digest = hashlib.sha256(np.asarray(map_value.shape, dtype="<u8").tobytes())
                digest.update(map_value.tobytes())

                key = f"map_{position:09d}"

                record.update({"map_ref": {"shard": f"maps/shard_{shard_index:06d}.npz", "key": key},
                               "map_hash": digest.hexdigest(),
                               "map_height": int(map_value.shape[0]), "map_width": int(map_value.shape[1]),
                               "valid_die_count": int(np.count_nonzero(map_value)),
                               "failed_die_count": int(np.count_nonzero(map_value == 2))})
                
                record["failed_die_ratio"] = record["failed_die_count"] / record["valid_die_count"]
                pending[key] = map_value

            manifest_line = json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n"
            stream.write(manifest_line.encode("utf-8"))
            counts[record["quality_status"]] += 1
            counts[label_status] += 1
            reasons.update(row_reasons)

            if label_status == "labeled":
                classes[label] += 1

            if len(pending) >= shard_rows:
                flush()

            # 맵·라벨·Lot 값 대신 전체 작업의 진행 비율만 알린다.
            if progress and ((position + 1) % 1000 == 0 or position + 1 == len(frame)):
                progress((position + 1) / len(frame))

        flush()

    summary = {"status": "canonical_export_complete", "row_count": len(frame),
               "counts": dict(counts), "reason_counts": dict(reasons), "class_counts": dict(classes),
               "shard_count": shard_index, "split_complete": False,
               "duplicate_audit_complete": False, "training_ready": False,
               "label_normalization_version": NORMALIZATION_VERSION,
               "label_aliases": LABEL_ALIASES,
               "label_coverage_status": "complete" if all(classes[name] > 0 for name in CLASS_NAMES) else "incomplete",
               "class_mapping": dict(enumerate(CLASS_NAMES))}
    
    budget.write_json(output / "conversion.json", summary)

    return summary


def export_bundle(frame, output, source_sha256, source_bytes, *, execution_mode,
                  max_output_bytes=DEFAULT_MAX_OUTPUT_BYTES, shard_rows=1000, progress=None,
                  extra_provenance=None):
    """출처 기록·마지막 성공 표식까지 동일한 바이트 상한으로 변환 묶음을 만든다."""
    budget = OutputBudget(max_output_bytes)
    output = Path(output)
    summary = export_frame(
        frame, output, "wm811k_" + source_sha256, shard_rows=shard_rows,
        progress=progress, max_output_bytes=max_output_bytes, budget=budget,
    )
    provenance = {
        "source_sha256": source_sha256, "source_bytes": source_bytes,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
        "label_normalization_version": NORMALIZATION_VERSION, "execution_mode": execution_mode,
    }

    if extra_provenance is not None:
        if type(extra_provenance) is not dict or set(extra_provenance).intersection(provenance):
            raise ValueError("추가 출처 정보가 기존 필드와 충돌합니다.")

        provenance.update(extra_provenance)

    budget.write_json(output / "provenance.json", provenance)
    # 성공 표식이 들어갈 공간까지 먼저 확인해 빈 성공 파일도 남기지 않는다.
    success = {"status": "canonical_export_complete", "training_ready": False}
    payload = json.dumps(success, ensure_ascii=True, indent=2, allow_nan=False).encode("utf-8")

    if budget.used + len(payload) > budget.maximum:
        raise ValueError("canonical 출력 크기 상한을 초과했습니다.")

    with budget.open(output / "SUCCESS.json") as stream:
        stream.write(payload)

    return summary


# ==========================================
# 실행 환경·명시적 승인 확인 후 원본 읽기
# - 기본은 Sandbox이며 로컬 실행은 별도 옵션과 출력 경로 필수
# - 변환 실패 시 부분 산출물을 학습 준비 완료로 표시하지 않음
# - 원본·집계·경로·예외 상세는 콘솔로 노출하지 않음
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="명시적으로 승인한 원본 canonical 변환")
    parser.add_argument("--input", required=True)
    parser.add_argument("--trusted-pickle", action="store_true")
    parser.add_argument("--allow-host", action="store_true", help="신뢰한 pickle의 로컬 실행을 명시적으로 허용")
    parser.add_argument("--output-root", help="새 canonical 묶음을 저장할 상위 폴더")
    parser.add_argument("--max-output-bytes", type=int, default=DEFAULT_MAX_OUTPUT_BYTES,
                        help="출처 기록과 성공 표식을 포함한 canonical 출력 바이트 상한")
    args = parser.parse_args()
    OutputBudget(args.max_output_bytes)

    # 환경 변수 확인은 악성 파일을 차단하는 보안 경계가 아니라 오실행 방지다.
    if not args.trusted_pickle:
        parser.error("출처 확인 후 --trusted-pickle로 승인하세요.")
    if os.environ.get("USERNAME") != "WDAGUtilityAccount" and not args.allow_host:
        parser.error("로컬 실행에는 --trusted-pickle과 --allow-host가 모두 필요합니다.")
    if args.allow_host and not args.output_root:
        parser.error("로컬 실행에는 --output-root를 지정하세요.")

    source = Path(args.input).resolve(strict=True)

    if not args.allow_host:
        source.relative_to(Path("C:/wm_input").resolve(strict=True))
        if args.output_root:
            parser.error("Sandbox 기본 실행에서는 고정 출력 공유 폴더를 사용합니다.")

    if not source.is_file() or source.suffix.lower() not in (".pkl", ".pickle"):
        raise ValueError("승인된 pickle 파일을 지정하세요.")
    
    if source.stat().st_size > 4 * 1024 ** 3:
        raise ValueError("파일 크기 상한을 초과했습니다.")
    
    output_root = Path(args.output_root).resolve() if args.allow_host else Path("C:/wm_output").resolve(strict=True)

    # 원본 폴더에 변환 결과를 섞거나 원본을 덮어쓰지 않는다.
    if output_root.is_relative_to(source.parent):
        parser.error("원본 폴더 밖의 출력 경로를 지정하세요.")

    output_root.mkdir(parents=True, exist_ok=True)

    # 입력 파일의 식별값은 로컬에만 보존하며 안전성 인증에 쓰지 않는다.
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)

    output = output_root / ("canonical_" + uuid.uuid4().hex)

    print("원본 읽기 시작. pickle은 신뢰 승인 여부와 별개로 코드를 실행할 수 있습니다.", flush=True)

    frame = pd.read_pickle(source)

    print("원본 읽기 완료. 수정된 라벨 규칙으로 새 묶음에 변환합니다.", flush=True)

    previous_percent = -1

    def show_progress(fraction):
        nonlocal previous_percent
        percent = int(fraction * 100)

        if percent != previous_percent:
            print(f"변환 진행: {percent}%", flush=True)
            previous_percent = percent

    summary = export_bundle(
        frame, output, digest.hexdigest(), source.stat().st_size,
        execution_mode="trusted_host" if args.allow_host else "sandbox",
        progress=show_progress, max_output_bytes=args.max_output_bytes,
    )

    print("canonical_export_complete: 변환 완료. 분할·중복 검증은 미완료입니다.")
    
    if summary['label_coverage_status'] == 'incomplete':
        print("주의: 9개 클래스 중 미검출 클래스가 있습니다. 로컬 집계 확인 후 검증하세요.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("변환을 중단했습니다. SUCCESS.json이 없는 출력은 사용하지 마세요.", file=sys.stderr)
        sys.exit(1)
