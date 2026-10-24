"""표형 v2 실행을 기존 승인 서비스 규약에 연결한다. 계산·격리·승인 검증은 기존 도구를 재사용한다."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from . import snapshot_split_cli as split_cli
from . import partition_export_cli as export_cli
from . import preparation_pipeline_cli as pipeline_cli
from .artifacts import _fingerprint
from .cli import _current_actor
from .csv_workflow_creation import (CsvCreationPreview, _pipeline, _split, checked_values, code_id as csv_code_id,
                                    create_application, typed_run)
from .csv_workflow_schema import creation_schema
from .csv_workflow_service import CsvWorkflowService
from .preparation_module_plan import validate_preparation_modules
from .split_evidence import read_committed_artifact, read_metadata
from .worker_contract import WorkerLimits
from src.preprocessing.profiles import profile_from_dict


def code_id():
    """CSV 공통 연결 지문에 v2 생성·전달·출처 검증 코드를 더한다. 같은 지문이 승인 증거에 기록된다."""
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256(csv_code_id().encode())
    # 전달은 worker 없이 부모가 판정하므로 typed 메타데이터 복원·후보/적용 증거 판정 코드도 포함한다.
    for name in ("src/agents/tabular_workflow_service.py", "src/agents/csv_handoff_cli.py",
                 "src/agents/csv_handoff_contract.py", "src/agents/tabular_source_contract.py",
                 "src/agents/tabular_input_service.py", "src/agents/tabular_preparation_worker.py",
                 "src/preprocessing/tabular_bundle.py", "src/agents/preparation_apply_cli.py",
                 "src/agents/preparation_pipeline_cli.py", "src/agents/preparation_apply_contract.py"):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def source_of(module_id, tool):
    """실행 계약에 고정된 표형 출처 ID. 상태 문자열이 아닌 계약 객체에서 읽는다."""
    if module_id == "tabular_split_review":
        return tool.contract.source_id
    if module_id == "tabular_partition_export":
        return tool.contract.source.source_id
    if module_id in ("tabular_prepare_candidate", "tabular_prepare_apply"):
        return tool.contract.train.parent_source_id
    if module_id == "tabular_local_handoff":
        return tool.arguments["source_contract_id"]
    raise ValueError("등록되지 않은 표형 작업입니다.")


def _same_source(found, expected):
    if found != expected:
        raise ValueError("선행 결과의 표형 출처 ID가 선택한 출처와 다릅니다.")


PROJECT = Path(__file__).resolve().parents[2]
INPUT_ROOT = PROJECT / "logs/agents/tabular_input"
TASK_ROOT = PROJECT / "logs/agents/tasks"


def _materialized(run_directory):
    """완료된 입력 생성 실행 폴더에서 승인·커밋을 재검증한 출처 계약을 만든다.

    입력 도구의 기본 루트나 작업 폴더의 input_runs 아래 실행만 받는다. 외부 폴더를 신뢰 루트로 쓰지 않는다.
    """
    from .artifacts import _root
    from .tabular_input_service import TabularInputService

    path = Path(run_directory)
    root = path.parent.parent
    trusted = (root == INPUT_ROOT
               or root.name == "input_runs" and root.parent.parent == TASK_ROOT and root.parent.name != "")
    if path.parent.name != "materialize" or not trusted:
        raise ValueError("관리 루트 아래의 표형 입력 생성(materialize) 실행 폴더가 필요합니다.")
    _root(root)
    service = TabularInputService("materialize", root=root)
    return service, path.name, service.source_contract(path.name)


def _train(split, evidence, export_name, source_id):
    export = export_cli.open_run(export_name)
    if typed_run(export) is not True:
        raise ValueError("v2 내보내기 실행이 필요합니다.")
    from .partition_input_contract import bind_typed_partition_input

    artifact = read_committed_artifact(export.states.directory, export_cli.executor(export).scope_for("ingest"))
    train = bind_typed_partition_input(evidence, artifact, "train", split.contract.profile.roles)
    _same_source(train.parent_source_id, source_id)
    return train


def preview_creation(module_id, values, limits):
    """현재 출처·선행 증거를 재검증해 고정한다. 파일 생성·승인 발급은 없다."""
    selected = checked_values(creation_schema(module_id), values, limits)
    source_id = selected["source_contract_id"]
    name = selected.get("name", selected.get("pipeline_name"))
    files, parents = {}, {"source": source_id}

    if module_id == "tabular_split_review":
        from .tabular_source_contract import TabularEdaInputContract

        _, _, source = _materialized(selected["source_run"])
        _same_source(source.source_id, source_id)
        path = Path(selected["profile"])
        profile = profile_from_dict(read_metadata(path))
        if profile.adapter_id != "tabular_canonical_v2" or profile.workflow_id != "split_review_v1" or profile.split_rules is None:
            raise ValueError("표형 v2 분할 검토 프로파일이 필요합니다.")
        TabularEdaInputContract(source, json.dumps(profile.to_dict(), ensure_ascii=False), limits).check_input()
        parents["materialization"] = source.materialization.marker_sha256
        files[str(path)] = _fingerprint(path)
        target = split_cli._directory(name)
    elif module_id in ("tabular_partition_export", "tabular_prepare_candidate"):
        split, scope, evidence = _split(selected["split_name"], typed=True)
        _same_source(split.contract.source_id, source_id)
        parents["split"] = evidence.signature
        if module_id == "tabular_partition_export":
            from .partition_export_contract import TabularExportContract
            from .tabular_source_contract import TabularEdaInputContract

            contract = TabularEdaInputContract(split.contract.source, split.contract.profile_json, limits)
            TabularExportContract(evidence, contract, selected["max_partition_bytes"]).check()
            target = export_cli._directory(name)
        else:
            from .preparation_candidate_contract import PreparationCandidateContract
            from src.preprocessing.transform import PreparationRules

            train = _train(split, evidence, selected["export_name"], source_id)
            parents["train"] = train.contract_id
            path = Path(selected["pipeline"])
            plan = validate_preparation_modules(read_metadata(path))
            PreparationCandidateContract(train, PreparationRules(**json.loads(plan.rules_json)), limits,
                                         selected["max_candidate_bytes"], schema_version="preparation_candidate_v2").check()
            files[str(path)] = _fingerprint(path)
            target = pipeline_cli._directory(name)
    elif module_id == "tabular_prepare_apply":
        from . import preparation_apply_cli as apply_cli

        directory, record, candidate, evidence, application = _pipeline(name, typed=True)
        _same_source(evidence.derivation.train.parent_source_id, source_id)
        if application is not None or (directory / "review.json").exists():
            raise ValueError("이미 후보 검토와 적용 작업이 준비됐습니다. 기존 작업으로 연결하세요.")
        if asdict(limits) != record["limits"]:
            raise ValueError("적용 작업의 자원은 후보 생성 때 고정한 자원과 같아야 합니다.")
        target = apply_cli._directory(record["apply_name"])
        parents["candidate"] = evidence.signature
        files[str(directory / "selection.json")] = _fingerprint(directory / "selection.json")
    elif module_id == "tabular_local_handoff":
        from .csv_handoff_cli import _directory
        from .csv_handoff_workflow import directory_for

        directory, record, candidate, evidence, application = _pipeline(selected["pipeline_name"], applied=True, typed=True)
        _same_source(evidence.derivation.train.parent_source_id, source_id)
        parents["candidate"] = evidence.signature
        parents["application"] = _fingerprint(application.states.directory / "state.json")
        target = directory_for(name, typed=True)
        if _directory(name).exists() or _directory(name).is_symlink():
            raise ValueError("기존 전달 묶음을 덮어쓰지 않습니다.")
    else:
        raise ValueError("등록되지 않은 표형 작업입니다.")

    if target.exists() or target.is_symlink():
        raise ValueError("기존 작업 또는 생성 기록을 덮어쓰지 않습니다. 새 이름을 사용하세요.")

    return CsvCreationPreview(json.dumps({"module_id": module_id, "run_name": name,
        "values": selected, "limits": asdict(limits), "files": files, "parents": parents,
        "code_id": code_id(), "actor": _current_actor(), "worker_started": False}, ensure_ascii=False, sort_keys=True))


def create(preview, *, confirmed=False):
    if confirmed is not True or not isinstance(preview, CsvCreationPreview):
        raise ValueError("현재 생성 범위를 명시적으로 확인해야 합니다.")
    packet = preview.packet
    module, values, limits = packet["module_id"], packet["values"], WorkerLimits(**packet["limits"])
    if preview_creation(module, values, limits) != preview:
        raise ValueError("입력·선행 결과·코드·자원이 미리보기 이후 변경됐습니다.")

    name = packet["run_name"]
    parents = packet["parents"]
    # 재대조 직후 다시 연 객체도 미리보기에서 고정한 출처·생성·분할 증거와 같아야 한다.
    if module == "tabular_split_review":
        _, _, source = _materialized(values["source_run"])
        if source.source_id != parents["source"] or source.materialization.marker_sha256 != parents["materialization"]:
            raise ValueError("미리보기 이후 표형 생성 출처가 변경됐습니다.")
        profile = profile_from_dict(read_metadata(Path(values["profile"])))
        result = split_cli.create_from_tabular(name, source, profile, limits, confirmed=True)
    elif module == "tabular_partition_export":
        split, _, evidence = _split(values["split_name"], typed=True)
        if evidence.signature != parents["split"] or split.contract.source_id != parents["source"]:
            raise ValueError("미리보기 이후 확정 분할 또는 출처가 변경됐습니다.")
        result = export_cli.create_from_tabular(name, values["split_name"], split.contract.source,
                                                limits, values["max_partition_bytes"], confirmed=True)
    elif module == "tabular_prepare_candidate":
        split, _, evidence = _split(values["split_name"], typed=True)
        train = _train(split, evidence, values["export_name"], values["source_contract_id"])
        if evidence.signature != parents["split"] or train.contract_id != parents["train"]:
            raise ValueError("미리보기 이후 확정 분할 또는 Train 구간이 변경됐습니다.")
        result = pipeline_cli.create_from_tabular(name, values["pipeline"], train, limits, values["max_candidate_bytes"],
                                                  values["max_partition_bytes"], confirmed=True)
    elif module == "tabular_prepare_apply":
        result = create_application(name, limits, typed=True)
    else:
        from .csv_handoff_workflow import create_run

        result = create_run(name, values["pipeline_name"], values["max_bundle_bytes"], limits,
                            source_contract_id=values["source_contract_id"])

    return {**result, "run_name": name, "module_id": module, "worker_started": False,
            "notice": "표형 v2 작업만 준비했습니다. 실행은 별도 승인 후 진행합니다."}


class TabularWorkflowService(CsvWorkflowService):
    """CSV 서비스의 상태·승인·실행·결과 규약을 그대로 쓰고 v2 실행만 연다."""

    connection_code = staticmethod(code_id)
    typed = True

    def __init__(self, module_id):
        from .workflow_service_registry import TABULAR_IDS

        if module_id not in TABULAR_IDS:
            raise ValueError("등록되지 않은 표형 작업입니다.")
        self.module_id = module_id

    def runtime(self):
        local = self.module_id == "tabular_local_handoff"
        supported = CsvWorkflowService("csv_local_handoff" if local else "csv_split_review").runtime()["supported"]
        return {"supported": supported, "code": "TABULAR_V2_RUNTIME_SUPPORTED" if supported else "TABULAR_V2_RUNTIME_UNVERIFIED",
                "notice": "로컬 고정 파일 복사입니다. AppContainer worker를 실행하는 작업이 아닙니다." if local else
                          "Windows Python 3.10·3.12에서 기존 격리 실행기를 사용합니다. 표형 v2 경로는 합성 입력과 모의 OS 운송으로 "
                          "검증했으며 실제 AppContainer 실행은 아직 확인하지 않았습니다. 새 작업과 별도 실행 승인이 필요합니다."}

    def preview_form(self, values, limits):
        return preview_creation(self.module_id, values, limits)

    def create(self, preview, *, confirmed=False):
        if not isinstance(preview, CsvCreationPreview) or preview.packet["module_id"] != self.module_id:
            raise ValueError("현재 모듈의 생성 미리보기가 필요합니다.")
        return {**create(preview, confirmed=confirmed), "runtime": self.runtime()}

    def recorded_results(self, name, recorded_code_id):
        raise ValueError("표형 v2 작업은 과거 코드 지문 기준의 읽기 전용 검증을 지원하지 않습니다.")


def _outputs(result):
    return {item["label"]: Path(item["path"]) for item in result["items"] if item["kind"] == "artifact_list"}


def _committed_json(output, relative, *, marker_sha256=None, file_sha256=None):
    """커밋 폴더의 완료 표식을 한 번 읽고, 그 표식의 크기·해시와 같은 바이트만 한 번 읽어 해석한다.

    검증 뒤 파일만 바꾸거나 바꿨다가 되돌려도 표식과 다른 바이트는 거부된다. 승인 체인에 검증 시점의
    표식 해시(marker_sha256)나 파일 해시(file_sha256)가 묶인 산출물은 그 값과도 대조해 표식 재작성 위조를 거부한다.
    기대 해시가 없는 산출물은 완료 표식의 보장 범위(손상 검사)까지만 확인한다.
    """
    from .artifacts import MARKER_NAME, MAX_MARKER_BYTES, _safe_relative
    from .isolated_eda import _read_file
    from .strict_json import loads
    from .worker_paths import _path

    _safe_relative(relative)
    sealed = _read_file(_path(output / MARKER_NAME, directory=False), MAX_MARKER_BYTES)
    if marker_sha256 is not None and hashlib.sha256(sealed).hexdigest() != marker_sha256:
        raise ValueError("보고 근거의 완료 기록이 검증 시점과 다릅니다.")
    marker = loads(sealed.decode("utf-8"), MAX_MARKER_BYTES)
    files = marker.get("files")
    expected = files.get(relative) if type(files) is dict else None
    if (type(expected) is not dict or type(expected.get("size")) is not int or not 0 < expected["size"] <= 8 * 1024 * 1024
            or type(expected.get("sha256")) is not str or len(expected["sha256"]) != 64):
        raise ValueError("보고 근거 파일이 완료 기록에 없습니다.")
    if file_sha256 is not None and expected["sha256"] != file_sha256:
        raise ValueError("보고 근거 파일이 검증된 결과와 다릅니다.")
    raw = _read_file(_path(output / relative, directory=False), expected["size"])
    if len(raw) != expected["size"] or hashlib.sha256(raw).hexdigest() != expected["sha256"]:
        raise ValueError("보고 근거 파일이 완료 기록과 다릅니다.")
    return loads(raw.decode("utf-8"), expected["size"])


def report_sources(runs, source_contract_id):
    """검증된 v2 결과에서 보고서 근거만 모은다. 계산·추정·표시 조립은 하지 않는다.

    runs는 split·export·candidate·apply·handoff 단계의 실행 이름이다. 모든 단계는 원래 도메인 검증을 통과해야 하고
    같은 출처·분할·내보내기·후보에 이어져 있어야 한다. 반환값은 로컬 전용이며 모델 전송 자료가 아니다.
    """
    from .workflow_service_registry import TABULAR_IDS, TABULAR_VERSION, workflow_service

    steps = dict(zip(("split", "export", "candidate", "apply", "handoff"), TABULAR_IDS))
    if type(runs) is not dict or set(runs) != set(steps) or runs["apply"] != runs["candidate"]:
        raise ValueError("다섯 단계의 실행 이름과 같은 전처리 통합 작업이 필요합니다.")

    verified, tools = {}, {}
    for step, module_id in steps.items():
        service = workflow_service(module_id, TABULAR_VERSION)
        # 결과 조회(results)와 같은 열기·도메인 검증을 한 번만 수행하고 그 도구로 출처를 확인한다.
        _, tool, manager = service._open(runs[step])
        result = service._collect(runs[step], tool, manager)
        if result["status"] != "completed":
            raise ValueError("완료되지 않은 단계가 있어 보고서 근거를 만들지 않습니다.")
        _same_source(source_of(module_id, tool), source_contract_id)
        verified[step], tools[step] = result, tool

    split, scope, evidence = _split(runs["split"], typed=True)
    _, _, _, candidate, application = _pipeline(runs["candidate"], applied=True, typed=True)
    # 보고 근거로 쓸 후보 증거가 승인된 적용 계약의 후보와 같아야 한다. 다른 시점에 읽은 증거를 섞지 않는다.
    if application.contract.candidate != candidate:
        raise ValueError("보고 근거의 후보가 승인된 적용 계약의 후보와 다릅니다.")
    export = read_committed_artifact(tools["export"].states.directory, export_cli.executor(tools["export"]).scope_for("ingest"))
    if (tools["export"].contract.split.signature != evidence.signature
            or candidate.derivation.train.split.signature != evidence.signature
            or candidate.derivation.train.export.marker_sha256 != export.marker_sha256
            or tools["handoff"].arguments["pipeline_name"] != runs["candidate"]):
        raise ValueError("단계 사이의 분할·내보내기·후보 연결이 다릅니다.")

    source = split.contract.source
    source.check_input()
    service, name, _ = _materialized(Path(source.materialization.run_directory))
    _, materialized = service.verified(name)
    inspection = materialized["inspection"]
    split_outputs = _outputs(verified["split"])
    bundle = next(Path(item["path"]) for item in verified["handoff"]["items"] if item["label"] == "handoff")
    # 원본 폴더를 다시 읽지 않고 승인 기록 대조를 통과한 봉인 사본의 완료 기록만 보고 근거로 쓴다.
    manifest = tools["handoff"].verified_bundle(_outputs(verified["handoff"])["ingest"], tools["handoff"].arguments)
    if manifest is None:
        raise ValueError("전달 묶음이 승인된 적용 결과와 다릅니다.")
    return {
        "source": {"source_contract_id": source.source_id, "snapshot_sha256": source.snapshot.sha256,
                   "loader": source.loader_id, "read_spec": json.loads(source.read_spec_json),
                   "inspection_id": source.inspection_id, "header_mapping": inspection["header_mapping"],
                   "columns": inspection["columns"], "rows": inspection["rows"]},
        # 검증을 마친 커밋 폴더라도 원본을 다시 열 때는 완료 표식의 해시에 묶인 바이트만 쓴다.
        # quality는 승인 체인에 묶인 기대 해시가 없어 완료 표식의 손상 검사 수준이다.
        "quality": _committed_json(split_outputs["audit"], "quality/quality.json"),
        # 분할안은 확정 증거가 미리보기 표식과 대조해 보존한 값을 쓴다.
        "proposal": evidence.proposal,
        "candidate": _committed_json(candidate.artifact.directory, "candidate/candidate.json",
                                     marker_sha256=candidate.artifact.marker_sha256),
        "preprocessor": _committed_json(candidate.artifact.directory, "candidate/preprocessor.json",
                                        marker_sha256=candidate.artifact.marker_sha256, file_sha256=candidate.state_sha256),
        "application": _committed_json(_outputs(verified["apply"])["apply_preparation"], "prepared/application.json",
                                       file_sha256=manifest["files"]["prepared/application.json"]["sha256"]),
        "handoff": {"directory": str(bundle), "schema_version": manifest["schema_version"],
                    "files": manifest["files"], "manifest_sha256": manifest["manifest_sha256"],
                    "readme": str(bundle / "README.md")},
        "reports": [item["path"] for result in verified.values() for item in result["items"] if item["kind"] == "report"],
        "local_only": True,
    }
