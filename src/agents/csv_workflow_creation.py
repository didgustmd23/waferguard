"""CSV UI 선택을 기존 CLI 생성 계약에 결합한다. worker를 새로 구현하지 않는다."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re

from . import snapshot_split_cli as split_cli
from . import partition_export_cli as export_cli
from . import preparation_pipeline_cli as pipeline_cli
from . import preparation_candidate_cli as candidate_cli
from . import preparation_apply_cli as apply_cli
from .artifacts import _fingerprint, _root
from .cli import _current_actor
from .csv_workflow_schema import creation_schema
from .executor import _run_lock
from .preparation_apply_contract import read_candidate_evidence
from .preparation_candidate_contract import saved_export_scope
from .preparation_module_plan import validate_preparation_modules
from .split_evidence import read_confirmed_split, read_metadata
from .worker_contract import WorkerLimits
from .worker_paths import _path
from src.preprocessing.profiles import profile_from_dict
from src.preprocessing.tabular_bundle import verification_scope


def code_id():
    root = Path(__file__).resolve().parents[2]
    names = ("src/agents/csv_workflow_creation.py", "src/agents/csv_workflow_service.py",
             "src/agents/csv_workflow_schema.py", "src/agents/csv_handoff_workflow.py",
             "src/agents/workflow_service_registry.py", "configs/agents/modules/csv_workflows.json",
             # 작업 조정기는 이 작업들의 승인을 위임 발급하므로 같은 코드 지문에 묶는다.
             # 어댑터·계획 검증·위임 발급 코드는 동의 C·D 발급 전 판정을 맡는다.
             "src/agents/task_orchestrator.py", "src/agents/task_consent.py", "src/agents/task_store.py", "src/agents/protected_store.py",
             "src/agents/llm_task_assist.py", "src/agents/task_adapters.py", "src/agents/tabular_task.py",
             "src/agents/task_plan.py", "src/agents/task_recommend.py", "src/agents/delegated_approval.py",
             "src/agents/strict_json.py")
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class CsvCreationPreview:
    payload_json: str

    @property
    def packet(self):
        return json.loads(self.payload_json)


def typed_run(tool):
    """실행의 고정 정책·계약 버전으로 v2 typed 실행인지 판정한다. 알 수 없는 도구는 거부한다."""
    contract = getattr(tool, "contract", None)
    version = getattr(contract, "schema_version", None)
    if version in ("preparation_candidate_v2", "preparation_apply_v2"):
        return True
    if version in ("preparation_candidate_v1", "preparation_apply_v1"):
        return False
    policy = tool.plan.policy_version
    if policy in ("tabular_split_v2", "partition_export_v2"):
        return True
    if policy in (split_cli.POLICY_VERSION, export_cli.POLICY_VERSION):
        return False
    raise ValueError("작업 버전을 확인할 수 없는 실행입니다.")


def _split(name, *, typed=False):
    tool = split_cli.open_run(name)
    if typed_run(tool) is not typed:
        raise ValueError("이 작업 버전의 확정 분할이 아닙니다.")
    tool._check_arguments(tool.arguments)
    scope = split_cli.executor(tool).scope_for("commit_split")
    evidence = read_confirmed_split(tool.states.directory, scope)
    return tool, scope, evidence


def _pipeline(name, *, applied=False, typed=False):
    # 중첩 검사가 같은 표형 묶음을 반복 검증하지 않도록 이 호출 안에서만 검증 결과를 공유한다.
    with verification_scope():
        return _checked_pipeline(name, applied=applied, typed=typed)


def _checked_pipeline(name, *, applied, typed):
    directory, record, candidate = pipeline_cli.open_run(name)
    if typed_run(candidate) is not typed:
        raise ValueError("이 작업 버전의 전처리 통합 작업이 아닙니다.")
    evidence = read_candidate_evidence(candidate.states.directory, _current_actor())
    application = pipeline_cli._application(directory, record, candidate)
    if (directory / "CANCEL_REQUESTED").exists():
        raise ValueError("취소된 통합 작업입니다.")
    if applied and (application is None or apply_cli.status(application).get("application_status") != "applied"):
        raise ValueError("검증된 적용 완료 작업이 필요합니다.")
    return directory, record, candidate, evidence, application


def checked_values(schema, values, limits):
    """생성 명세의 필드 종류·범위만 검사하고 파일 경로를 정규화한다. 선행 증거 검사는 호출자가 한다."""
    if not isinstance(limits, WorkerLimits):
        raise TypeError("검증된 자원 계약이 필요합니다.")
    if type(values) is not dict or set(values) != {field["name"] for field in schema["fields"]}:
        raise ValueError("생성 명세와 입력 필드가 다릅니다.")

    selected = dict(values)
    for field in schema["fields"]:
        value = selected[field["name"]]
        kind = field["kind"]
        if kind == "boolean":
            if type(value) is not bool:
                raise ValueError("명시적인 bool 설정이 필요합니다.")
        elif kind == "integer":
            if type(value) is not int or not field["minimum"] <= value <= field["maximum"]:
                raise ValueError("출력·복사 상한이 잘못됐습니다.")
        elif type(value) is not str or not value.strip() or len(value) > 4096 or any(ord(c) < 32 for c in value):
            raise ValueError("빈 값·제어 문자를 포함한 생성 입력은 사용할 수 없습니다.")
        elif kind in {"csv_file", "json_file"}:
            selected[field["name"]] = str(_path(Path(value).absolute(), directory=False))
            if kind == "json_file" and Path(selected[field["name"]]).stat().st_size > 65536:
                raise ValueError("설정 JSON은 64 KiB 이하여야 합니다.")
            if kind == "json_file" and Path(selected[field["name"]]).stat().st_size > 65536:
                raise ValueError("설정 JSON은 64 KiB 이하여야 합니다.")
        elif kind == "run_directory":
            selected[field["name"]] = str(_path(Path(value).absolute(), directory=True))
        elif kind == "source_contract_id" and re.fullmatch(r"tabular_(?:time_)?source:[0-9a-f]{64}", value) is None:
            raise ValueError("검증된 표형 출처 ID 형식이 필요합니다.")
    return selected


def preview_creation(module_id, values, limits):
    """현재 파일·부모 증거를 재검증하고 고정한다. 파일 생성·승인 발급은 없다."""
    selected = checked_values(creation_schema(module_id), values, limits)
    name = selected.get("name", selected.get("pipeline_name"))
    files, parents = {}, {}
    if module_id == "csv_split_review":
        target = split_cli._directory(name)
        source, profile_path = Path(selected["source"]), Path(selected["profile"])
        profile = profile_from_dict(read_metadata(profile_path))
        if profile.adapter_id != "tabular_csv_v1" or profile.workflow_id != "split_review_v1":
            raise ValueError("CSV 분할 검토 프로파일이 필요합니다.")
        if source.stat().st_size > selected["max_input_bytes"]:
            raise ValueError("입력 복사 상한을 초과했습니다. 자르거나 일부만 처리하지 않습니다.")
        files = {str(path): _fingerprint(path) for path in (source, profile_path)}
    elif module_id in {"csv_partition_export", "csv_prepare_candidate"}:
        split, scope, evidence = _split(selected["split_name"])
        parents["split"] = evidence.signature
        if module_id == "csv_partition_export":
            from .eda_input_contract import bind_eda_input
            from .partition_export_contract import PartitionExportContract

            source = bind_eda_input(split.contract.snapshot_directory, split.contract.snapshot, split.contract.profile, limits)
            PartitionExportContract(evidence, source, selected["max_partition_bytes"])
            target = export_cli._directory(name)
        else:
            target = pipeline_cli._directory(name)
            export = export_cli.open_run(selected["export_name"])
            if typed_run(export) is not False:
                raise ValueError("CSV v1 내보내기 실행이 필요합니다.")
            export_status = export_cli.status(export)
            if export_status.get("partition_contracts") != "passed":
                raise ValueError("검증된 내보내기 완료 작업이 필요합니다.")
            train = candidate_cli._train(str(split.states.directory), scope, str(export.states.directory),
                                         saved_export_scope(export.states.directory))
            parents["train"] = train.contract_id
            path = Path(selected["pipeline"])
            plan = validate_preparation_modules(read_metadata(path))
            from .preparation_candidate_contract import PreparationCandidateContract
            from src.preprocessing.transform import PreparationRules

            PreparationCandidateContract(train, PreparationRules(**json.loads(plan.rules_json)), limits, selected["max_candidate_bytes"])
            files[str(path)] = _fingerprint(path)
    elif module_id == "csv_prepare_apply":
        directory, record, candidate, evidence, application = _pipeline(name)
        if application is not None or (directory / "review.json").exists():
            raise ValueError("이미 후보 검토와 적용 작업이 준비됐습니다. 기존 작업으로 연결하세요.")
        if asdict(limits) != record["limits"]:
            raise ValueError("적용 작업의 자원은 후보 생성 때 고정한 자원과 같아야 합니다.")
        target = apply_cli._directory(record["apply_name"])
        parents["candidate"] = evidence.signature
        files[str(directory / "selection.json")] = _fingerprint(directory / "selection.json")
    else:
        from .csv_handoff_workflow import directory_for
        from .csv_handoff_cli import _directory

        directory, record, candidate, evidence, application = _pipeline(selected["pipeline_name"], applied=True)
        parents["candidate"] = evidence.signature
        parents["application"] = _fingerprint(application.states.directory / "state.json")
        target = directory_for(name)
        if _directory(name).exists() or _directory(name).is_symlink():
            raise ValueError("기존 전달 묶음을 덮어쓰지 않습니다.")

    if target.exists() or target.is_symlink():
        raise ValueError("기존 작업 또는 생성 기록을 덮어쓰지 않습니다. 새 이름을 사용하세요.")

    return CsvCreationPreview(json.dumps({"module_id": module_id, "run_name": name,
        "values": selected, "limits": asdict(limits), "files": files, "parents": parents,
        "code_id": code_id(), "actor": _current_actor(), "worker_started": False}, ensure_ascii=False, sort_keys=True))


def create_application(name, limits, *, typed=False):
    """검토한 후보에 묶인 적용 작업과 review 기록을 만든다. v1·v2가 같은 후보 검토 계약을 쓴다."""
    directory, record, candidate, evidence, _ = _pipeline(name, typed=typed)
    with _run_lock(directory) as acquired:
        if not acquired:
            raise ValueError("다른 통합 요청이 진행 중입니다.")
        # 기존 CLI 검토와 같은 후보 증거·하위 실행 생성·review 계약을 사용한다.
        directory, record, candidate, current, application = _pipeline(name, typed=typed)
        if current != evidence or application is not None:
            raise ValueError("검토한 후보 상태가 변경됐습니다.")
        evidence.check()
        result = apply_cli.create(record["apply_name"], candidate.states.directory, limits,
                                  record["max_partition_bytes"], confirmed=True)
        application = apply_cli.open_run(record["apply_name"])
        pipeline_cli._save_new(directory / "review.json", {"candidate_id": evidence.signature,
                               "apply_identity": pipeline_cli._child_identity(application)})
    return result


def create(preview, *, confirmed=False):
    if confirmed is not True or not isinstance(preview, CsvCreationPreview):
        raise ValueError("현재 생성 범위를 명시적으로 확인해야 합니다.")
    packet = preview.packet
    module, values, limits = packet["module_id"], packet["values"], WorkerLimits(**packet["limits"])
    if preview_creation(module, values, limits) != preview:
        raise ValueError("입력·선행 결과·코드·자원이 미리보기 이후 변경됐습니다.")

    name = packet["run_name"]
    if module == "csv_split_review":
        result = split_cli.create(name, values["source"], values["profile"], limits, values["max_input_bytes"],
                                  confirmed=True, allow_feature_findings=values["allow_feature_findings"])
    elif module in {"csv_partition_export", "csv_prepare_candidate"}:
        split = split_cli.open_run(values["split_name"])
        exported = split_cli.export_scope(split)
        if module == "csv_partition_export":
            result = export_cli.create(name, exported["split_run_dir"], exported["split_scope"],
                exported["snapshot_dir"], exported["profile"], limits, values["max_partition_bytes"], confirmed=True)
        else:
            export = export_cli.open_run(values["export_name"])
            result = pipeline_cli.create(name, values["pipeline"], exported["split_run_dir"], exported["split_scope"],
                str(export.states.directory), limits, values["max_candidate_bytes"], values["max_partition_bytes"], confirmed=True)
    elif module == "csv_prepare_apply":
        result = create_application(name, limits)
    else:
        from .csv_handoff_workflow import create_run

        result = create_run(name, values["pipeline_name"], values["max_bundle_bytes"], limits)

    return {**result, "run_name": name, "module_id": module, "worker_started": False,
            "notice": "기존 CSV 작업만 준비했습니다. 실행은 별도 승인 후 진행합니다."}
