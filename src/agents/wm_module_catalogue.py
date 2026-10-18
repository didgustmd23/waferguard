"""WM 작업을 공통 모듈 계약으로 표현한다. 기존 worker·승인 코드는 변경하지 않는다."""

from dataclasses import dataclass
import re

from src.preprocessing.module_contracts import ModuleContract, ModulePort


@dataclass(frozen=True)
class WmWorkflowModule:
    """계산 함수가 아닌 승인 작업의 등록 정보다.

    하나의 WM 작업은 preflight·검토·worker 실행 등 여러 로컬 단계를 가진다.
    이를 일반 EDA의 calculate 함수처럼 직접 호출하면 승인 경계를 잃으므로,
    기존 CLI와 실행 관리자에 연결할 고정 경로만 보관한다.
    """

    contract: ModuleContract
    cli_id: str
    worker_stage: str
    review_stages: tuple[str, ...] = ()

    def validate_settings(self, settings=None):
        """실제 설정은 기존 작업 생성 시 고정한다. 계획에서 임의 인자를 주입하지 않는다."""
        settings = {} if settings is None else settings

        if type(settings) is not dict or settings:
            raise ValueError("WM 모듈 계획의 설정은 빈 객체여야 합니다. 실제 설정은 기존 생성 CLI에서 지정하세요.")

        return {}


def wm_modules():
    """품질 검토 분기와 정상 개발 흐름을 서로 다른 증거 계약으로 등록한다."""
    definitions = (
        ("wm_canonical_eda", "analysis", "wm_canonical", "wm_eda_evidence", "wm_canonical_cli", "run_eda", ()),
        ("wm_conflict_candidate", "analysis", "wm_quality_review", "wm_conflict_candidate", "wm_conflict_cli", "derive_preparation", ()),
        ("wm_conflict_apply", "transformation", "wm_conflict_candidate", "wm_canonical", "wm_exclusion_cli", "derive_preparation", ()),
        ("wm_split_propose", "analysis", "wm_eda_evidence", "wm_split_proposal", "wm_split_cli", "derive_preparation", ()),
        ("wm_split_commit", "transformation", "wm_split_proposal", "wm_confirmed_split", "wm_split_commit_cli", "commit_split", ("review_eda",)),
        ("wm_transform_check", "validation", "wm_confirmed_split", "wm_transform_evidence", "wm_transform_cli", "derive_preparation", ()),
        ("wm_cache_prepare", "transformation", "wm_transform_evidence", "wm_development_cache", "wm_cache_cli", "prepare_data", ("review_eda", "commit_split")),
        ("wm_dataset_check", "validation", "wm_development_cache", "wm_dataset_report", "wm_dataset_cli", "validate_contract", ()),
    )
    modules = []

    for identifier, kind, source, result, cli, stage, reviews in definitions:
        contract = ModuleContract(
            module_id=identifier, version="1.0.0", kind=kind,
            inputs=(ModulePort("evidence", source),),
            outputs=(ModulePort("result", result),),
            implementation_id=identifier + "_workflow_v1",
            result_validator_id=identifier + "_evidence_v1",
            preconditions=("committed_input_verified", "explicit_stage_approval"),
        )
        modules.append(WmWorkflowModule(contract, cli, stage, reviews))

    return tuple(modules)


def _cli(identifier):
    """JSON의 import 경로를 실행하지 않는다. 검토된 고정 분기만 지연 import한다."""
    if identifier == "wm_canonical_cli":
        from . import wm_canonical_cli as cli
    elif identifier == "wm_conflict_cli":
        from . import wm_conflict_cli as cli
    elif identifier == "wm_exclusion_cli":
        from . import wm_exclusion_cli as cli
    elif identifier == "wm_split_cli":
        from . import wm_split_cli as cli
    elif identifier == "wm_split_commit_cli":
        from . import wm_split_commit_cli as cli
    elif identifier == "wm_transform_cli":
        from . import wm_transform_cli as cli
    elif identifier == "wm_cache_cli":
        from . import wm_cache_cli as cli
    elif identifier == "wm_dataset_cli":
        from . import wm_dataset_cli as cli
    else:
        raise ValueError("등록되지 않은 WM 실행 경로입니다.")

    return cli


def open_wm_execution(module_id, version, run_name):
    """기존 작업을 재검증해 실행 관리자에 연결한다. 승인·run_next는 호출하지 않는다.

    정적 계획의 포트는 실제 완료 증거가 아니다. 기존 open_run과 입력 검사로
    사용자·소스·코드·선행 증거를 확인한 뒤 기존 승인 저장소를 그대로 사용한다.
    반환 관리자를 호출할 때도 원래 단계별 승인·취소·재실행 방지가 적용된다.
    """
    module = next((item for item in wm_modules()
                   if (item.contract.module_id, item.contract.version) == (module_id, version)), None)

    if module is None:
        raise ValueError("등록되지 않은 WM 모듈 또는 버전입니다.")

    if type(run_name) is not str or re.fullmatch(r"[a-z][a-z0-9_-]{0,47}", run_name) is None:
        raise ValueError("기존 WM 작업 이름 형식이 올바르지 않습니다.")

    cli = _cli(module.cli_id)
    tool = cli.open_run(run_name)
    tool._check_arguments(tool.arguments)
    registered = tool.registered_tool()

    if registered.action_id != module.worker_stage or registered.requires_approval is not True:
        raise ValueError("기존 WM 실행기의 작업 종류 또는 필수 승인 경계가 등록 정보와 다릅니다.")

    return tool, cli.executor(tool)
