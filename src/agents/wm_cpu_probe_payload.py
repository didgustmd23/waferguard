"""고정 합성 맵의 CPU 호환성 검사. 실데이터 경로·모델·GPU 옵션을 받지 않는다."""

import json
from pathlib import Path


SCHEMA = "wm_cpu_probe_v1"
CHECKS = frozenset({"torch_cpu", "numpy_tensor", "canonical", "eda", "split", "transform",
                    "cache", "dataset", "roundtrip", "source_unchanged", "development_only"})


def run_synthetic(workspace, diagnostic=lambda stage: None):
    """54개 독립 합성 맵만 생성한다. 기존 전처리 코어를 호출하며 학습하지 않는다."""
    diagnostic("IMPORT_NUMPY")
    import numpy as np
    diagnostic("IMPORT_PANDAS")
    import pandas as pd
    diagnostic("IMPORT_TORCH")
    import torch
    diagnostic("IMPORT_MATPLOTLIB")
    import matplotlib
    matplotlib.use("Agg", force=True)

    diagnostic("IMPORT_WM_CORE")
    from scripts.wm_sandbox.export_canonical import CLASS_NAMES, export_bundle
    from src.agents.wm_canonical_contract import inventory
    from src.wafer_dl.validate_canonical import validate_bundle
    from src.wafer_dl.eda import run_eda
    from src.wafer_dl.split import run_split
    from src.wafer_dl.isolated_transform_check import run_isolated_transform_check
    from src.wafer_dl.isolated_cache import build_isolated_cache
    from src.wafer_dl.transform import transform_map, make_channels
    from src.wafer_dl.dataset import WaferDataset, check_first_batch

    base = Path(workspace).resolve(strict=True)
    if any(base.iterdir()):
        raise ValueError("빈 합성 작업 폴더가 필요합니다.")

    def folder(name):
        path = base / name
        path.mkdir()
        return path

    torch.set_num_threads(1)
    checks = {}
    diagnostic("TORCH_CPU")
    tensor = torch.arange(6, dtype=torch.float32, device="cpu") * 2
    tensor_values = tensor.tolist()
    checks["torch_cpu"] = tensor.device.type == "cpu" and tensor_values == [0, 2, 4, 6, 8, 10]
    checks["numpy_tensor"] = torch.equal(torch.from_numpy(np.arange(6, dtype=np.float32)) * 2, tensor)
    if not checks["torch_cpu"] or not checks["numpy_tensor"]:
        raise ValueError("합성 CPU 텐서 기대값이 일치하지 않습니다.")

    diagnostic("SYNTHETIC_CANONICAL")
    rows = []
    for class_index, label in enumerate(CLASS_NAMES):
        for repeat in range(6):
            number = class_index * 6 + repeat + 1
            array = np.ones((8 + repeat % 3, 8 + repeat % 2), dtype=np.uint8)
            array[-1, :] = 0
            array[3:5, 3:5] = 2
            for bit in range(6):
                array.flat[bit] = 2 if number & (1 << bit) else 1

            # 라벨은 검사 범주 식별자다. 실제 결함 패턴을 재현한 데이터가 아니다.
            rows.append({"waferMap": array, "failureType": label, "lotName": f"cpu_probe_{number}"})

    source = base / "canonical"
    export_bundle(pd.DataFrame(rows), source, "b" * 64, 100,
                  execution_mode="synthetic_demo", max_output_bytes=1024 * 1024)
    diagnostic("CANONICAL_INVENTORY")
    # 이미 고정 scratch 안에서 생성한 파일이다. 기존 WM worker와 같이
    # staged 검사를 사용해 접근 권한이 없는 프로젝트 상위 폴더를 조회하지 않는다.
    # 파일 목록·링크·하드 링크·크기·해시 검사는 inventory 안에서 계속 수행한다.
    before = inventory(source, 1024 * 1024, staged=True)
    diagnostic("CANONICAL_VALIDATION")
    report = validate_bundle(source, workspace=folder("validation_workspace"))
    checks["canonical"] = report["split_assessment"]["status"] == "preconditions_passed"
    if not checks["canonical"]:
        raise ValueError("합성 canonical 사전조건을 만족하지 못했습니다.")

    diagnostic("EDA")
    checks["eda"] = run_eda(source, base / "eda", include_examples=False)["status"] == "eda_complete"

    split_config = {"protocol_version": "wm_group_split_v1", "ratios": [0.7, 0.15, 0.15],
                    "seed": 42, "max_candidates": 8, "max_group_ratio": 0.2, "rare_class_warning_count": 1}
    transform_config = {"transform_version": "wm_resize_v1", "target_size": [32, 32],
                        "resize_backend": "numpy_half_pixel_nearest_v1", "preserve_aspect_ratio": True,
                        "padding_value": 0, "channel_order": ["defect", "valid"], "augmentation": "none"}
    cache_config = {"cache_version": "wm_input_cache_v1", "shard_max_rows": 8, "buffer_max_bytes": 1048576}
    split_path, transform_path = base / "split_config.json", base / "transform_config.json"
    split_path.write_text(json.dumps(split_config), encoding="utf-8")
    transform_path.write_text(json.dumps(transform_config), encoding="utf-8")

    diagnostic("GROUP_SPLIT")
    split = base / "split"
    run_split(source, split, split_path, approved=True, workspace=folder("split_workspace"))
    split_report = json.loads((split / "split.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (split / "split_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    groups = {name: {r["split_group_id"] for r in records if r["split"] == name}
              for name in ("train", "validation", "test")}
    checks["split"] = (sum(split_report["split_counts"].values()) == 54 and
                       all(min(counts) > 0 for counts in split_report["class_counts"].values()) and
                       not (groups["train"] & groups["validation"] or groups["train"] & groups["test"] or
                            groups["validation"] & groups["test"]))

    diagnostic("TRANSFORM")
    check = base / "transform_check"
    result = run_isolated_transform_check(split, check, transform_path, source,
                                         workspace=folder("transform_workspace"), max_output_bytes=16 * 1024 ** 2)
    checks["transform"] = result["status"] == "transform_check_passed"
    if not checks["transform"]:
        raise ValueError("합성 맵 변환의 형태 손실을 검토해야 합니다.")

    diagnostic("CACHE")
    cache = base / "cache"
    # 합성 fixture에만 고정된 검사 승인이다. 실데이터 시각 검토를 대체하지 않는다.
    result = build_isolated_cache(split, check, cache, source, transform_config, cache_config,
                                  workspace=folder("cache_workspace"), max_output_bytes=16 * 1024 ** 2)
    checks["cache"] = result["status"] == "cache_complete"

    diagnostic("DATASET")
    batches = {name: check_first_batch(cache, name, 4, allow_test=False) for name in ("train", "validation")}
    checks["dataset"] = all(item["status"] == "dataset_batch_passed" for item in batches.values())
    checks["development_only"] = (not (cache / "test").exists() and
                                  all(item["test_maps_read"] is False for item in batches.values()))

    # 각 표본을 직접 생성한 합성 배열과 대조한다. 성공 문자열만 확인하지 않는다.
    diagnostic("ROUNDTRIP")
    originals = {r["sample_id"]: rows[r["source_row_position"]]["waferMap"] for r in records}
    labels = {r["sample_id"]: r["class_index"] for r in records}
    checks["roundtrip"] = True
    for name in ("train", "validation"):
        dataset = WaferDataset(cache, name)
        try:
            for index in range(len(dataset)):
                sample = dataset[index]
                transformed, _ = transform_map(originals[sample["sample_id"]], transform_config)
                expected = make_channels(transformed)
                checks["roundtrip"] &= (np.array_equal(sample["image"].numpy(), expected) and
                                         sample["image"].device.type == "cpu" and
                                         sample["target"].item() == labels[sample["sample_id"]])
        finally:
            dataset.close()

    checks["source_unchanged"] = inventory(source, 1024 * 1024, staged=True) == before
    return {"schema_version": SCHEMA, "checks": {key: bool(value) for key, value in checks.items()},
            "tensor_values": tensor_values, "samples": 54, "class_count": 9,
            "split_counts": split_report["split_counts"], "batches": batches,
            "device": "cpu", "torch_version": torch.__version__, "model_trained": False,
            "real_data_read": False, "production_worker_enabled": False,
            "scope": "fixed_synthetic_cpu_preprocessing_only", "gpu_verified": False,
            "notice": "canonical·분할 검증은 전체 합성 맵을 읽습니다. 변환·캐시·배치는 개발 구간만 사용합니다."}
