"""코드에서 검토한 공통 처리 단계와 프로파일 실행 템플릿."""

from types import MappingProxyType


STAGE_ACTION = MappingProxyType({
    "preflight": "run_preflight",
    "approve_source": "request_source_approval",
    "ingest": "run_ingest",
    "audit": "run_audit",
    "run_eda": "run_eda",
    # 사용자 선택 모듈 파이프라인의 형식 중립 실행 단계.
    "run_pipeline": "run_pipeline",
    "review_eda": "review_eda",
    "propose_split": "propose_split",
    "approve_split": "request_split_approval",
    "commit_split": "commit_split",
    "derive_preparation": "derive_preparation",
    "apply_preparation": "apply_preparation",
    "approve_prepare": "request_prepare_approval",
    "prepare_data": "prepare_data",
    "prepare_cache": "prepare_cache",
    "validate_contract": "validate_contract",
    "publish": "publish_report",
    "render_report": "render_report",
})

STAGE_ORDER = tuple(STAGE_ACTION)

# EDA 전용에는 분할·학습 입력 생성을 강제하지 않는다.
WORKFLOW_TEMPLATES = MappingProxyType({
    "tabular_preparation_v1": (
        "preflight", "audit", "run_eda", "review_eda", "propose_split", "commit_split",
        "prepare_data", "validate_contract", "render_report",
    ),
    "prepare_review_v1": (
        "preflight", "audit", "run_eda", "review_eda", "propose_split", "commit_split", "prepare_data",
    ),
    "split_review_v1": (
        "preflight", "audit", "run_eda", "review_eda", "propose_split", "commit_split",
    ),
    "eda_review_v1": (
        "preflight", "audit", "run_eda", "review_eda",
        "validate_contract", "render_report",
    ),
    "training_preparation_v1": (
        "preflight", "ingest", "audit", "run_eda", "review_eda",
        "propose_split", "commit_split", "prepare_data",
        "validate_contract", "publish", "render_report",
    ),
})
