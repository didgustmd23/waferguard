"""WM 정상 흐름의 작업 준비와 새 배정 기록. 실행·승인 발급은 하지 않는다."""

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sqlite3
from contextlib import closing

from .artifacts import _fingerprint, _root
from .executor import _run_lock
from .isolated_eda import _read_file
from .wm_canonical_contract import digest
from .wm_module_plan import load_wm_modules
from .wm_run_bindings import inspect_bindings, load_bindings
from .wm_run_evidence import WmRunReader, check_run_name, load_run_tool
from .worker_contract import WorkerLimits, _decode
from .worker_paths import _path


MAX_OUTPUT_BYTES = 1024 ** 3
CREATION_MODULES = {
    "wm_canonical_eda": (None, None, 64 * 1024 * 1024, MAX_OUTPUT_BYTES),
    "wm_split_propose": ("wm_canonical_eda", "eda_run", 64 * 1024 * 1024, MAX_OUTPUT_BYTES),
    "wm_split_commit": ("wm_split_propose", "proposal_run", 2 * 1024 ** 3, 2 * 1024 ** 3),
    "wm_transform_check": ("wm_split_commit", "split_run", 256 * 1024 * 1024, MAX_OUTPUT_BYTES),
    "wm_cache_prepare": ("wm_transform_check", "check_run", 8 * 1024 ** 3, 32 * 1024 ** 3),
    "wm_dataset_check": ("wm_cache_prepare", "cache_run", 1024 * 1024, 64 * 1024 * 1024),
}


def _creation_cli(module_id):
    """허용한 종류만 기존 CLI에 연결한다. 설정의 import 경로는 받지 않는다."""
    if module_id == "wm_canonical_eda":
        from . import wm_canonical_cli as cli
    elif module_id == "wm_split_propose":
        from . import wm_split_cli as cli
    elif module_id == "wm_split_commit":
        from . import wm_split_commit_cli as cli
    elif module_id == "wm_transform_check":
        from . import wm_transform_cli as cli
    elif module_id == "wm_cache_prepare":
        from . import wm_cache_cli as cli
    elif module_id == "wm_dataset_check":
        from . import wm_dataset_cli as cli
    else:
        raise ValueError("생성이 지원되지 않는 WM 단계입니다.")

    return cli


def creation_root(*, create=False):
    """작업 생성 기록은 고정 logs 이름 공간에만 둔다. 기존 실행 폴더와 구분한다."""
    root = _root(Path(__file__).resolve().parents[2])

    for part in ("logs", "agents", "wm_creation", "runs"):
        root /= part

        if create:
            root.mkdir(exist_ok=True)

        if root.exists():
            _root(root)

    return root


def _write_new(path, value):
    """고정된 작은 JSON만 배타적으로 쓴다. 기존 파일을 교체하지 않는다."""
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8")

    if len(raw) > 65536:
        raise ValueError("생성 기록은 64 KiB 이하여야 합니다.")

    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


@dataclass(frozen=True)
class CreationPreview:
    """생성 확인 범위다. 실제 실행 승인이나 데이터 처리 성공을 뜻하지 않는다."""

    packet_json: str

    @property
    def packet(self):
        return json.loads(self.packet_json)

    @property
    def signature(self):
        return digest(self.packet)


def preview_creation(pipeline, bindings, step_id, name, split_config, limits,
                     max_output_bytes=None, *, transform_config=None, cache_config=None, dataset_options=None,
                     canonical_dir=None, canonical_options=None):
    """첫 미배정 작업만 준비한다. 분할 확정 조건은 부모 제안에서 상속한다."""
    check_run_name(name)

    if not isinstance(limits, WorkerLimits):
        raise ValueError("검증된 자원 요청이 필요합니다.")

    paths = {key: _path(Path(value).absolute(), directory=False)
             for key, value in (("pipeline", pipeline), ("bindings", bindings))}

    if any(path.stat().st_size > 65536 for path in paths.values()):
        raise ValueError("계획·배정은 각각 64 KiB 이하여야 합니다.")

    stamps = {key: _fingerprint(path) for key, path in paths.items()}
    plan = load_wm_modules(paths["pipeline"])
    runs = load_bindings(paths["bindings"], plan)
    steps = list(plan.steps)
    step = next((item for item in steps if item.step_id == step_id), None)

    if step is None or step.module_id not in CREATION_MODULES or step.step_id in runs:
        raise ValueError("지원하는 첫 미배정 WM 단계만 생성할 수 있습니다.")

    parent_module, _, default_output, maximum = CREATION_MODULES[step.module_id]
    max_output_bytes = default_output if max_output_bytes is None else max_output_bytes

    if type(max_output_bytes) is not int or not 1 <= max_output_bytes <= maximum:
        raise ValueError("출력 상한은 해당 생성 단계의 허용 범위 안의 양의 정수여야 합니다.")

    if transform_config is not None and step.module_id != "wm_transform_check":
        raise ValueError("변환 설정은 변환 검사 단계에서만 지정할 수 있습니다.")

    if split_config is not None and step.module_id != "wm_split_propose":
        raise ValueError("분할 설정은 분할 제안 단계에서만 지정할 수 있습니다.")

    if cache_config is not None and step.module_id != "wm_cache_prepare":
        raise ValueError("캐시 설정은 캐시 준비 단계에서만 지정할 수 있습니다.")

    if dataset_options is not None and step.module_id != "wm_dataset_check":
        raise ValueError("Dataset 설정은 Dataset 검사 단계에서만 지정할 수 있습니다.")

    if (canonical_dir is not None or canonical_options is not None) and step.module_id != "wm_canonical_eda":
        raise ValueError("canonical 입력 선택은 첫 EDA 단계에서만 지정할 수 있습니다.")

    if step.module_id == "wm_split_propose":
        config_path = split_config if split_config is not None else "configs/wm811k/split.json"
        paths["split_config"] = _path(Path(config_path).absolute(), directory=False)

        if paths["split_config"].stat().st_size > 65536:
            raise ValueError("분할 설정은 64 KiB 이하여야 합니다.")

        stamps["split_config"] = _fingerprint(paths["split_config"])
        _decode(_read_file(paths["split_config"], 65536).decode("utf-8"))
    elif step.module_id == "wm_transform_check":
        config_path = transform_config if transform_config is not None else "configs/wm811k/transform.json"
        paths["transform_config"] = _path(Path(config_path).absolute(), directory=False)

        if paths["transform_config"].stat().st_size > 65536:
            raise ValueError("변환 설정은 64 KiB 이하여야 합니다.")

        stamps["transform_config"] = _fingerprint(paths["transform_config"])
        _decode(_read_file(paths["transform_config"], 65536).decode("utf-8"))
    elif step.module_id == "wm_cache_prepare":
        config_path = cache_config if cache_config is not None else "configs/wm811k/cache.json"
        paths["cache_config"] = _path(Path(config_path).absolute(), directory=False)

        if paths["cache_config"].stat().st_size > 65536:
            raise ValueError("캐시 설정은 64 KiB 이하여야 합니다.")

        stamps["cache_config"] = _fingerprint(paths["cache_config"])
        _decode(_read_file(paths["cache_config"], 65536).decode("utf-8"))

    if any(item.step_id in runs for item in steps[steps.index(step) + 1:]):
        raise ValueError("후속 작업이 배정돼 있습니다. 기존 연결을 보존한 새 배정 계획을 준비하세요.")

    reference = dict(step.inputs)["evidence"]

    initial = step.module_id == "wm_canonical_eda"

    if (reference == "source.evidence") != initial:
        raise ValueError("첫 EDA는 원본 증거를, 후속 단계는 계획 안의 완료 선행 단계를 참조해야 합니다.")

    parent_id = reference.split(".", 1)[0] if not initial else None
    parent_step = next((item for item in steps if item.step_id == parent_id), None)
    checked = inspect_bindings(plan, runs)

    if (checked["status"] != "attention_required" or checked["next_action"]["code"] != "RUN_NOT_ASSIGNED"
            or checked["next_action"]["step_id"] != step_id
            or (not initial and (parent_step is None or parent_step.module_id != parent_module))):
        raise ValueError("선행 작업을 먼저 검증하고 현재 첫 미배정 단계만 생성하세요.")

    target = _creation_cli(step.module_id)._directory(name)
    directory = creation_root() / name

    if target.exists() or target.is_symlink() or directory.exists() or directory.is_symlink():
        raise ValueError("기존 작업 또는 생성 기록을 덮어쓰지 않습니다. 새 이름을 사용하세요.")

    reader = WmRunReader()
    parent = reader.read(parent_step.module_id, runs[parent_id]) if not initial else None
    canonical = None

    if initial:
        from .wm_eda_creation import canonical_selection

        canonical = canonical_selection(canonical_dir, canonical_options, limits, max_output_bytes)
    config = None
    transform = None
    cache = None
    estimated_output = None
    dataset = None

    if step.module_id == "wm_split_propose":
        from src.wafer_dl.split import load_split_config
        config = load_split_config(paths["split_config"])

        if config["max_group_ratio"] != parent.tool.contract.max_group_ratio:
            raise ValueError("분할 설정의 그룹 검토 상한이 완료 EDA와 다릅니다.")

    if step.module_id == "wm_transform_check":
        from src.wafer_dl.transform import load_transform_config

        # 설정 형식만 검사한다. 맵 읽기·변환·캐시 생성은 실행 단계에서 수행한다.
        transform = load_transform_config(paths["transform_config"])

    if step.module_id == "wm_cache_prepare":
        from src.wafer_dl.cache import load_cache_config
        from src.wafer_dl.isolated_cache import estimate_cache_bytes

        cache = load_cache_config(paths["cache_config"])
        split = parent.tool.evidence
        # 기존 생성 함수와 같은 보수적 출력 예산을 생성 전에도 확인한다.
        # 캐시 배열이나 Test 맵을 읽지 않고 확정 보고서와 설정만 사용한다.
        estimated_output = estimate_cache_bytes(split["split_report"], split["transform_config"]) + 1024 * 1024
        height, width = split["transform_config"]["target_size"]
        capacity = min(cache["shard_max_rows"], cache["buffer_max_bytes"] // (height * width))

        if capacity < 1 or any(
                (split["split_report"]["split_counts"][part] + capacity - 1) // capacity > 2048
                for part in ("train", "validation")):
            raise ValueError("캐시 shard 수가 구간별 2048개 상한을 초과합니다. 버퍼·shard 행 수를 검토하세요.")

        if estimated_output > max_output_bytes:
            raise ValueError("출력 상한이 보수적 캐시 예상 크기보다 작습니다. 설정과 출력 상한을 검토하세요.")

    if step.module_id == "wm_dataset_check":
        from .wm_dataset_creation import dataset_selection

        dataset = dataset_selection(parent, dataset_options)

    reader.check_unchanged()

    if any(_fingerprint(path) != stamps[key] for key, path in paths.items()):
        raise ValueError("생성 미리보기 중 계획·배정·설정이 변경되었습니다.")

    packet = {
        "version": "wm_run_creation_v6", "pipeline_signature": plan.signature, "module_id": step.module_id,
        "paths": {key: str(value) for key, value in paths.items()}, "files": stamps,
        "step_id": step_id, "run_name": name, "parent_run": parent.name if parent else None,
        "parent_run_id": parent.tool.states.load().run_id if parent else None,
        "parent_plan_id": parent.tool.plan.plan_id if parent else None,
        "parent_files_id": digest(parent.report["files"]) if parent else None, "split_config": config,
        "canonical_options": canonical,
        "transform_config": transform,
        "cache_config": cache, "estimated_output_bytes": estimated_output,
        "dataset_options": dataset,
        "limits": asdict(limits), "max_output_bytes": max_output_bytes,
        "new_bindings_directory": str(directory), "worker_started": False,
        "execution_approval_issued": False,
        "review_required_before_execution": step.module_id in {"wm_split_commit", "wm_cache_prepare"},
    }
    return CreationPreview(json.dumps(packet, sort_keys=True, ensure_ascii=False, allow_nan=False))


def create_and_assign(preview, *, confirmed=False):
    """사용자가 확인한 범위를 다시 확인한 뒤 기존 생성 함수를 한 번만 호출한다."""
    if not confirmed or not isinstance(preview, CreationPreview):
        raise ValueError("작업 생성과 새 배정 기록에 대한 사용자 확인이 필요합니다.")

    saved = preview.packet
    root = creation_root(create=True)

    with _run_lock(root) as acquired:
        if not acquired:
            raise ValueError("다른 통합 생성 작업이 진행 중입니다.")

        current = preview_creation(saved["paths"]["pipeline"], saved["paths"]["bindings"], saved["step_id"],
                                   saved["run_name"], saved["paths"].get("split_config"),
                                   WorkerLimits(**saved["limits"]), saved["max_output_bytes"],
                                   transform_config=saved["paths"].get("transform_config"),
                                   cache_config=saved["paths"].get("cache_config"),
                                   dataset_options=({key: saved["dataset_options"][key] for key in
                                       ("batch_size", "max_input_bytes", "max_runtime_bytes")}
                                       if saved["dataset_options"] is not None else None),
                                   canonical_dir=saved["canonical_options"]["directory"] if saved["canonical_options"] else None,
                                   canonical_options=({key: saved["canonical_options"][key] for key in
                                       ("max_input_bytes", "include_examples", "max_group_ratio")}
                                       if saved["canonical_options"] else None))

        if current != preview:
            raise ValueError("확인한 생성 범위가 변경됐습니다. 새 미리보기와 사용자 확인이 필요합니다.")

        directory = root / saved["run_name"]
        directory.mkdir(exist_ok=False)
        _write_new(directory / "creation_started.json", {"preview_id": preview.signature, "selection": saved})
        # 확인한 설정의 로컬 복사본을 기존 생성 함수에 전달한다.
        # 원래 설정 파일은 수정하지 않고, 실행 승인도 별도로 남겨 둔다.
        cli = _creation_cli(saved["module_id"])
        options = {"confirmed": True}

        if saved["module_id"] == "wm_split_propose":
            _write_new(directory / "split_config.json", saved["split_config"])
            options["config_path"] = directory / "split_config.json"
        elif saved["module_id"] == "wm_transform_check":
            # 확인한 설정의 복사본을 전달하여 원래 설정 파일의 재사용을 피한다.
            _write_new(directory / "transform_config.json", saved["transform_config"])
            options["config_path"] = directory / "transform_config.json"
        elif saved["module_id"] == "wm_cache_prepare":
            # 캐시 설정만 복사하고 기존 변환 조건은 부모 증거에서 상속한다.
            _write_new(directory / "cache_config.json", saved["cache_config"])
            options["config_path"] = directory / "cache_config.json"

        if saved["module_id"] == "wm_canonical_eda":
            selection = saved["canonical_options"]
            result = cli.create(saved["run_name"], selection["directory"], WorkerLimits(**saved["limits"]),
                                selection["max_input_bytes"], saved["max_output_bytes"], confirmed=True,
                                include_examples=selection["include_examples"], max_group_ratio=selection["max_group_ratio"])
        elif saved["module_id"] == "wm_dataset_check":
            # Dataset CLI는 출력 예산을 키워드로 받는다. 모델 학습 옵션은 없다.
            settings = saved["dataset_options"]
            options.update({key: settings[key] for key in ("batch_size", "max_input_bytes", "max_runtime_bytes")})
            result = cli.create(saved["run_name"], saved["parent_run"], WorkerLimits(**saved["limits"]),
                                max_output_bytes=saved["max_output_bytes"], **options)
        else:
            result = cli.create(saved["run_name"], saved["parent_run"], WorkerLimits(**saved["limits"]),
                                saved["max_output_bytes"], **options)
        _write_new(directory / "run_created.json", {"run_name": saved["run_name"], "worker_started": False})
        tool, _, _, _ = load_run_tool(saved["module_id"], saved["run_name"], require_complete=False)
        reader = WmRunReader()
        reader.check_parent(saved["module_id"], tool)
        state = tool.states.load()
        parent_field = CREATION_MODULES[saved["module_id"]][1]

        if saved["module_id"] == "wm_canonical_eda":
            from .wm_eda_creation import check_canonical_source

            selection = saved["canonical_options"]
            check_canonical_source(selection, tool.limits, saved["max_output_bytes"])
            tool.contract.check()

            if (digest(tool.contract.files) != selection["files_id"]
                    or any(getattr(tool.contract, key) != selection[key] for key in
                           ("max_input_bytes", "include_examples", "max_group_ratio"))):
                raise ValueError("생성한 canonical 복사본·설정이 확인한 선택과 다릅니다.")

        if saved["module_id"] == "wm_dataset_check" and any(
                getattr(tool.contract, key) != saved["dataset_options"][key] for key in
                ("batch_size", "max_input_bytes", "max_runtime_bytes", "runtime_sha256", "torch_version")):
            raise ValueError("생성된 Dataset 작업의 배치·복사 예산·런타임이 확인 범위와 다릅니다.")

        if (result.get("status") != "created" or result.get("worker_started") is not False
                or state.status != "created" or state.current_stage != "preflight"
                or (parent_field is not None and getattr(tool, parent_field) != saved["parent_run"])
                or tool.limits != WorkerLimits(**saved["limits"])
                or tool.contract.max_output_bytes != saved["max_output_bytes"]
                or (saved["module_id"] == "wm_cache_prepare"
                    and tool.evidence["cache_config"] != saved["cache_config"])
                or (saved["module_id"] == "wm_transform_check"
                    and tool.evidence["transform_config"] != saved["transform_config"])
                or (saved["module_id"] == "wm_split_propose" and tool.evidence["split_config"] != saved["split_config"])):
            raise ValueError("생성된 작업의 상태·부모·설정이 확인한 범위와 다릅니다.")

        with closing(sqlite3.connect((tool.states.directory / "approvals.sqlite3").as_uri() + "?mode=ro", uri=True)) as connection:
            if any(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                   for table in ("approvals", "reservations", "task_runs")):
                raise ValueError("작업 생성만 수행해야 합니다. 승인·실행 기록이 발견됐습니다.")

        plan = load_wm_modules(saved["paths"]["pipeline"])
        runs = load_bindings(saved["paths"]["bindings"], plan)

        if any(_fingerprint(_path(path, directory=False)) != saved["files"][key]
               for key, path in saved["paths"].items()):
            raise ValueError("생성 중 계획·배정·설정이 변경됐습니다. 새 작업은 보존하되 배정하지 않습니다.")

        reader.check_unchanged()
        runs[saved["step_id"]] = saved["run_name"]
        output = directory / "bindings.json"
        _write_new(output, {"schema_version": "wm_run_bindings_v1", "pipeline_signature": plan.signature, "runs": runs})

        if load_bindings(output, plan) != runs:
            raise ValueError("새 배정 파일 검증에 실패했습니다.")

        _write_new(directory / "CREATION_COMPLETE.json", {"status": "created_and_assigned", "preview_id": preview.signature,
                   "run_id": state.run_id, "bindings": _fingerprint(output), "worker_started": False,
                   "execution_approval_issued": False})

    return {"status": "created_and_assigned", "run_name": saved["run_name"], "bindings": str(output),
            "worker_started": False, "execution_approval_issued": False,
            "review_required_before_execution": saved["review_required_before_execution"],
            "notice": "준비 작업과 새 배정만 생성했습니다. 다음 단계와 실행 승인은 별도입니다."}
