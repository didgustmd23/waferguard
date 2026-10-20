# WaferGuard 소스 구성

최신 기준(2026-10-20): CSV 5개 작업·WM 정상 개발 6단계의 공통 UI, EDA/CSV workflow LLM 제안·확인 질문, 저장 제안 복원·모델 목록 조회, 별도 정책의 결과 해설을 구현했습니다. 최종 전체 합성 회귀는 958개 실행·945개 통과·13개 생략·실패/오류 0입니다. 실제 모델 품질·OS 격리 재검증과 구분하며 [현재 완료·보류 범위](../docs/agents/GUIDE.md#현재-구현-상태)를 기준으로 읽습니다.

런타임: `agents/wm_dataset_runtime.py`의 `pinned_torch_version`을 Dataset 생성·승인·합성 검사기가 공유합니다. `agents/wm_runtime.py`는 Windows/Python 3.12에서 `2.14.1+cpu` 또는 `2.14.1+cu130`의 WM 정상 6단계를 허용합니다. CUDA 빌드에서도 전처리 worker는 CPU 연산이며 GPU 장치 실행과 충돌 분기 2개는 허용하지 않습니다. CLI와 공통 서비스는 같은 모듈별 정책을 적용하고 실제 환경 격리 호환성 검증은 별도입니다.

공통 등록 모듈 UI: `ui/workflow_view.py`가 기존 5단계 화면을 재사용하고 `ui/workflow_service.py`가 명시적 생성·조회·승인·한 단계 실행만 공통 서비스에 전달합니다. `agents/wm_creation_schema.py`의 필드 명세로 WM 정상 6단계 생성 폼을 표시합니다. 충돌 분기 2개는 기존 작업 연결만 지원하며 Python 3.12 실행은 차단합니다. `ui/workflow_results.py`는 기존 도메인 검증 후 크기가 제한된 로컬 표시 복사본을 만듭니다. 런타임 지원은 위의 현재 정책을 따릅니다. [UI 사용법](../docs/agents/design/METADATA_REGISTRY.md#공통-ui-연결과-생성-설정--2026-10-19)을 참고하세요.

WM 공통 작업 서비스: `agents/wm_workflow_service.py`는 모듈 ID·버전으로 기존 생성·조회·승인·한 단계 실행·결과 검증을 호출합니다. `agents/workflow_approval.py`는 화면과 무관한 승인 미리보기·동일 범위 재검사·저장을 담당하고 WM 8개 CLI가 재사용합니다. 3.10의 기존 정책과 3.12 정상 6단계의 모듈별 지원 정책을 공유하며 설치된 CUDA 빌드와 실제 GPU 장치 사용은 구분합니다. [WM 공통 서비스 안내](../docs/agents/design/METADATA_REGISTRY.md#wm-처리의-공통-서비스-정비--2026-10-19-후속)를 참고하세요.

통합 모듈 명세: `agents/module_specification.py`가 `configs/agents/modules/catalogue.json`을 기존 `agent_module_registry.py`의 계약과 대조해 표형 8개·WM 8개의 공통 설명 명세를 제공합니다. `agents/module_specification_cli.py`는 목록·상세 조회 전용이며 기존 사용자 참조 모듈 CLI는 유지합니다. 설정 기본값·WM 승인 단계는 기존 구현에서 읽고 결과 표시 유형·설명·노출 정책을 결합합니다. 명세 기반 실행 서비스·공통 UI를 연결했으며 명세 등록 자체는 실행 허가가 아닙니다. [명세 사용법](../docs/agents/design/METADATA_REGISTRY.md#통합-모듈-설명-명세-정비--2026-10-19)을 참고하세요.

로컬 UI: `ui/app.py`는 Streamlit 진입점, `ui/navigation.py`는 단계 메뉴, `ui/settings.py`는 세션 설정 폼입니다. `ui/source_selection.py`에서 데이터 유형·경로를 선택하고 CSV 원본은 동의 후 snapshot을 만듭니다. `ui/dialogue_view.py`·`dialogue_service.py`는 기존 LangChain·LangGraph 대화와 제안 검토·저장 계약을 재사용합니다. `ui/input_view.py`는 입력 결합 미리보기, `ui/candidate_service.py`·`candidate_view.py`는 미승인 작업 준비와 영수증 검증을 담당합니다. `ui/model_commands.py`는 원문 없이 명령 ID·지문·상태만 기록해 모델 명령의 중복 전송을 차단합니다.

`ui/execution_service.py`·`execution_view.py`는 CSV 사전 검사·범위 확인·승인·별도 실행과 결과 화면을 연결합니다. `agents/pipeline_approval_cli.py`의 승인 범위 재검사·일회 소비·AppContainer·Job을 유지합니다. `ui/result_service.py`는 완료/검토 중단 결과의 정합성을 확인해 로컬 표시합니다. CSV 분할·변환·전달과 WM canonical 이후 정상 6단계를 공통 UI에 연결했고 `ui/csv_plan_view.py`가 CSV LLM 실행안을 같은 화면으로 연결합니다. `ui/commentary_*.py`는 별도 정책으로 허용한 집계·그래프만 로컬 모델에 전달합니다. EDA 제안은 목록 선택, CSV 실행안·WM 기존 작업은 명시적 이름으로 연결하며 과거 승인·실행 요청·대화 원문은 복원하지 않습니다. 화면 갱신으로 모델·worker를 다시 실행하지 않습니다. pickle 생성 UI·실행 중 취소·백그라운드 실행은 미지원/보류입니다.

저장·모델 확인: `agents/llm_proposal_store.py`의 목록/복원과 `ui/dialogue_service.py`·`dialogue_view.py`를 연결했습니다. `ui/settings.py`의 연결 확인은 `agents/local_ollama.py`로 loopback `/api/tags`를 1회 조회해 설치 모델 이름만 표시하며 추론·GPU 동작 확인은 아닙니다.

프로젝트 루트에서 `python -m pip install -r requirements.txt`로 통합 의존성을 설치한 뒤 `python -m streamlit run src/ui/app.py`로 실행합니다. Windows x64/Python 3.12의 `wm-gpu312` 환경 하나를 사용합니다. [최신 UI 범위·실행 순서](../docs/agents/GUIDE.md#최신-ui-연결-범위--2026-10-19), [데이터 유형·경로 선택](../docs/agents/GUIDE.md#요청-전-데이터-유형경로-선택-보완), [실행 후보 생성](../docs/agents/GUIDE.md#ui-입력-확인에서-실행-후보-생성-연결)을 참고하세요. 소스 변경 후 UI 서버를 재시작합니다.

LangGraph 제안 대화: `agents/llm_dialogue.py`는 메모리 전용 상태·확인 질문 중단·답변 재개·호출/누적 길이 제한을 관리합니다. `agents/llm_dialogue_cli.py`는 매번 전송 내용 확인과 SEND를 받으며 완성 제안만 별도 SAVE로 기존 저장 계약에 연결합니다. 승인·worker를 실행하는 그래프가 아닙니다. [흐름과 명령](../docs/agents/GUIDE.md#langgraph-확인-질문답변-연결)을 참고하세요.

LangChain 선택 연결: `agents/langchain_local.py`는 `BaseChatModel` 인터페이스와 기존 제한형 로컬 HTTP 운송을 연결합니다. `llm_proposal_cli --backend langchain`으로 선택하며 기존 HTTP 기본 경로·제안 검증·승인·worker는 유지합니다. 모델·그래프가 공통 `run_unobserved`로 추적·부모 콜백 비상속 규칙을 적용합니다. [구현 범위와 명령](../docs/agents/GUIDE.md#langchain-모델-연결--기존-운송-보호-재사용)을 참고하세요.

LLM 제안의 실행 후보 연결: `agents/llm_execution_candidate.py`는 저장 제안·snapshot·프로파일·자원을 기존 어댑터 바인더로 검증하고, 확인 후 기존 `pipeline_approval_cli.create`로 미승인 작업만 준비합니다. `agents/llm_execution_candidate_cli.py`는 로컬 미리보기·생성 확인 명령입니다. LLM 재호출·실행 승인 발급·worker 실행은 하지 않습니다. [입력 결합과 사용법](../docs/agents/GUIDE.md#제안과-입력을-결합한-기존-승인-작업-준비)을 참고하세요.

로컬 LLM 제안: `agents/llm_config.py`가 `configs/agents/llm/local_ollama.json`의 모델·문맥·생성량·타임아웃을 검증합니다. `agents/local_ollama.py`는 loopback HTTP 운송, `agents/llm_proposal.py`는 합성/직접 입력 문장 구성·응답 계약 연결, `agents/llm_proposal_cli.py`는 전송 미리보기·사용자 확인·제안 명령입니다. `agents/llm_proposal_store.py`는 별도 확인 후 제안을 로컬에 보관하고 복원 시 현재 계약·지문을 재검사합니다. 데이터 파일을 자동 첨부하거나 승인 DB·worker에 연결하지 않습니다. 직접 입력한 문장 자체는 모델에 전달됩니다. [설정과 명령](../docs/agents/GUIDE.md#로컬-qwen-연결--합성-제안-호출과-질문-지침-보완)을 참고하세요.

WM 공통 승인·실행: `agents/wm_workflow_execution.py`와 `agents/wm_execution_cli.py`는 배정된 현재 작업의 `status`·`approve`·`next`를 제공합니다. 조회 전용 안내는 그대로 분리하고, 승인 화면은 기존 CLI를 재사용하며 실행기는 잠금 안에서도 선택 범위를 다시 확인합니다. 한 번에 한 단계만 처리하고 자동 승인·재시도는 하지 않습니다. [사용법](../docs/agents/design/METADATA_REGISTRY.md#공통-승인한-단계-실행)을 참고하세요.

WM 작업 준비·배정: `agents/wm_run_creation.py`와 `agents/wm_creation_cli.py`는 첫 EDA부터 분할·변환·캐시·Dataset까지 정상 6단계의 범위 미리보기·사용자 확인·기존 생성 함수 호출·새 배정을 제공합니다. `agents/wm_eda_creation.py`는 canonical 바이트 선택·복사 예산을, `agents/wm_dataset_creation.py`는 완료 캐시·배치·런타임 확인을 담당합니다. worker 실행·검토/실행 승인 발급·기존 배정 덮어쓰기는 하지 않습니다. [첫 EDA 사용법](../docs/agents/design/METADATA_REGISTRY.md#첫-eda-작업-준비-통합)을 참고하세요.

WM 다음 작업 안내: `agents/wm_workflow_guidance.py`는 현재 상태·부모 연결·승인 만료·예약을 확인해 행동만 추천하고, `agents/wm_guidance_cli.py`는 안내 전용 CLI입니다. 생성·승인·실행은 호출하지 않습니다. [상태별 안내와 확인 명령](../docs/agents/design/METADATA_REGISTRY.md#다음-작업-안내)을 참고하세요.

WM 실제 작업 연결 조회: `agents/wm_run_evidence.py`는 승인 DB의 읽기 전용 완료 증거, `agents/wm_run_bindings.py`는 단계별 작업 배정·부모 연결, `agents/wm_run_binding_cli.py`는 정상 개발 6단계의 조회를 담당합니다. 승인·worker 실행은 하지 않습니다. [배정 형식과 사용법](../docs/agents/design/METADATA_REGISTRY.md#실제-wm-작업-배정과-읽기-전용-연결-조회)을 참고하세요.

최신 WM 모듈 연결(2026-10-17): `agents/agent_module_registry.py`는 표형 8개·WM 8개 통합 목록, `agents/wm_module_catalogue.py`는 기존 승인 실행기 연결, `agents/wm_module_plan.py`는 증거 포트·순서·버전 검사, `agents/wm_module_cli.py`는 정적 계획 확인을 제공합니다. 기존 worker·완료 승인 코드는 변경하지 않습니다. [현재 계약과 사용법](../docs/agents/design/METADATA_REGISTRY.md#wm-작업의-공통-등록계획-연결)을 참고하세요. 아래 초기 구성 설명과 달리 WM 실데이터의 개발 캐시·Dataset 첫 배치 검사까지 사용자 실행으로 완료했으며, 전체 실행 동선 통합·LLM 연결은 아직 남아 있습니다.

프로젝트 영역은 SECOM 머신러닝, 범용 전처리 에이전트, WM-811K 딥러닝으로 나눕니다. 공통 전처리 입력·EDA·프로파일 계획과 에이전트 실행 관리까지 합성 검증했으며, 실데이터 에이전트 활성화와 WM 모델 학습은 진행 전입니다.

```text
src/
├── modeling_*.py 등   # 재사용 코어·설정·검증·센서 추론
├── sensor_ml/        # 현재 구현된 센서 ML 실행기
│   ├── data_pipeline/
│   ├── experiments/
│   ├── diagnostics/
│   ├── inference/
│   ├── evaluation/
│   └── verification/
├── preprocessing/    # 공통 입력·프로파일·품질 검사·EDA·workflow
├── agents/           # 승인·실행·상태·산출물 및 데이터 어댑터
└── wafer_dl/         # canonical 검증기·맵 처리·딥러닝 영역
```

맵 처리 함수와 딥러닝은 [wm811k](README.md#wm-811k-맵-처리와-딥러닝), 작업 실행·상태·재개 관리는 [agents](README.md#전처리-에이전트)에서 분리합니다. SECOM 코드는 이 두 영역에 의존하지 않습니다. 기존 modeling 코어는 표형 이진 분류 계약이며, 새 preprocessing 코어는 label 개수에 독립적인 입력 검사·EDA를 제공합니다. WM 맵 변환·학습 지표는 도메인 계약을 따릅니다.

## 공통 코어 — src 바로 아래

pickle 변환은 `agents/pickle_import_cli.py`·`pickle_import.py`에서 승인·실행을 관리합니다. `pickle_contract.py`가 선택 계약을, `pickle_converters.py`가 구조 판별과 표·숫자 배열·WM 변환기 연결을 담당합니다. 공통 데이터 코어는 `preprocessing/canonical_data.py`입니다. 격리 실행은 `pickle_import_transport.py`·`pickle_import_worker.py`, 부모 검증은 `pickle_results.py`에 있습니다. 기존 `wm_pickle_*` 파일은 호환 진입점입니다. [실행 방법과 지원 범위](../docs/agents/guides/PICKLE_IMPORT.md)를 참고하세요.

저장 기준 적용은 `agents/preparation_apply_cli.py`·`preparation_apply.py`가 관리합니다. 완료 후보 증거·두 구간 계약은 `preparation_apply_contract.py`, 격리 복원/변환과 운송은 `preparation_apply_worker.py`·`preparation_apply_transport.py`, 부모 검증은 `preparation_apply_results.py`가 담당합니다. [출력 파일·승인·실행 안내](../docs/agents/guides/PREPARATION_APPLY.md)를 참고하세요. fit·Test 적용·모델 학습은 수행하지 않습니다.

Train 후보 기준 산출은 `agents/preparation_candidate_cli.py`·`preparation_candidate.py`가 관리합니다. 입력 증거는 `preparation_candidate_contract.py`, 격리 계산·운송은 `preparation_candidate_worker.py`·`preparation_candidate_transport.py`, 부모 파일 검증은 `preparation_candidate_results.py`가 담당합니다. 기존 `preprocessing/transform.py`를 재사용하며 변환 적용·모델 학습은 하지 않습니다. [구현 범위·사용법](../docs/agents/guides/PREPARATION_CANDIDATE.md)을 참고하세요.

분할 준비는 `agents/snapshot_split_cli.py`·`snapshot_split.py`가 관리하며, `snapshot_split_worker.py`·`snapshot_split_transport.py`가 격리 품질·EDA·제안·확정을 수행합니다. [SECOM Train 기능 검증 절차](../docs/agents/guides/SNAPSHOT_SPLIT.md)를 참고하세요. 기존 `preprocessing/split.py`를 재사용하며 ML 실험 분할·모델은 변경하지 않습니다. 합성 테스트는 통과했고 실제 Windows 실행은 별도 확인 대상입니다.

구간 추출은 `agents/partition_export_contract.py`·`partition_export_worker.py`, 부모 파일 검증은 `partition_export_results.py`, 제한된 CSV 파서는 `partition_csv.py`에 있습니다. 승인·커밋 연결은 `partition_export.py`, AppContainer+Job 운송은 `partition_export_transport.py`, 사용자 명령은 `partition_export_cli.py`가 담당합니다. [구간 내보내기 구현 범위·사용법](../docs/agents/guides/PARTITION_EXPORT.md)을 참고하세요. 합성 연결 테스트는 통과했으며 실제 Windows 격리 실행은 별도 확인이 필요합니다.

변환 준비의 실행 증거는 `agents/split_evidence.py`, 구간 입력 결합은 `agents/partition_input_contract.py`가 검사합니다. [확정 분할·구간 입력 계약](../docs/agents/design/PARTITION_INPUT_CONTRACT.md)을 참고하세요. 기존 분할 결과를 읽기 전용으로 확인하며 실제 내보내기·변환은 실행하지 않습니다.

선택 모듈 실행은 `agents/module_pipeline_cli.py`·`module_pipeline.py`가 관리하며, worker·운송·파일 검증은 각각 `module_pipeline_worker.py`·`module_pipeline_transport.py`·`module_pipeline_results.py`에서 담당합니다. [합성 예제 실행·검증 범위](../docs/agents/guides/MODULE_PIPELINE_EXECUTION.md)를 참고하세요. 기존 고정 EDA·audit CLI는 유지합니다.

`preprocessing/pipeline.py`는 모듈 선택 JSON의 순서·포트·계약·설정을 검증합니다. 예제는 `configs/agents/pipelines/tabular_eda_example.json`이며 실제 실행 명령이나 데이터 경로를 받지 않습니다. [설정 방법·검증 범위](../docs/agents/design/MODULE_PIPELINE.md)를 참고하세요.

모듈형 전처리의 등록 계약은 `preprocessing/module_contracts.py`, 목록 관리·정확한 버전 조회는 `module_registry.py`, 내장 EDA 5개·품질 검사 연결은 `builtin_modules.py`에서 관리합니다. [등록부 사용·경계](../docs/agents/design/MODULE_REGISTRY.md)를 참고하세요. 새 모듈 CLI·worker의 합성 연결을 확인했으며 실제 OS 실행은 별도 검증 대상입니다.

범용 EDA는 `preprocessing/eda.py`의 기능별 계산과 `preprocessing/eda_report.py`의 그림·파일 생성으로 분리합니다. `collect_eda(..., analyses=("missing",))`처럼 필요한 계산만 호출할 수 있습니다. 기존 고정 EDA CLI는 전체 보고서를, 새 모듈 CLI는 선택한 계산의 CSV·품질 보고서를 생성합니다. `quality.py`는 검사/저장 분리를 유지합니다. 과거 보고서·노트북은 재작성하지 않습니다.

여러 실행기가 재사용하는 설정·데이터 구조·전처리·모델·평가·센서 선택·추론 코어만 유지합니다. 실행기는 아래 기능별 폴더에서 관리합니다.

- 코어: `modeling_config`, `dataset_schema`, `modeling_preprocessing`, `modeling_models`, `modeling_metrics`, `split_contract`, `threshold_policy`, `feature_reduction`, `stable_rf_selector`.
- 추론 코어: `sensor_inference`. 저장 모델의 클래스 참조 경로를 보존합니다.

## SECOM 기능별 실행기 — src/sensor_ml

| 폴더 | 역할 | 파일 |
| --- | --- | --- |
| data_pipeline/ | 데이터 병합·품질 점검·분할 | step1_merge_data, step2_data_check, step3_split |
| experiments/ | Baseline·후보·특징 비교·시간 검증·OOF 실험 | step4_baseline~step8_threshold_oof, top20_oof_compare, top20_time_validation, temporal_validation, time_weight_compare, recent_window_compare, feature_time_compare, anomaly_compare |
| experiments/missing_value_compare.py | V2 모델을 유지한 결측 처리 비교 | P0 중앙값 / P1 중앙값+indicator / P2 자체 NaN 처리 |
| experiments/all_sensor_reference.py | 저장 V2와 동일 학습 범위·깊이의 전체 센서 기준 모델 준비 | Train 내부 OOF 문턱 80%·90% 선택, 모델 저장; Test는 별도 평가 |
| evaluation/evaluate_all_sensor_test.py | 동결 전체 센서 모델의 동일 Test 평가 | 확률 한 번 계산, 두 OOF 문턱 적용, 기존 고정 Top-20과 비교표 저장 |
| inference/ | 모델 묶음 저장·복원·시나리오 보존·CSV 예측 | sensor_bundle, save_v2_scenario, predict_cli |
| evaluation/ | 동결 모델 평가·평가 실행 범위 관리 | evaluate_frozen_model, run_evaluation |

실험 폴더는 학습과 후보 비교, 평가 폴더는 동결 모델 평가 및 실행 관리가 중심입니다. `run_evaluation`은 선택된 변경 범위에 따라 OOF 학습을 호출할 수 있습니다. 역할 분류이지 학습 유무만으로 나눈 것은 아닙니다.

## SECOM 진단·보고서 — src/sensor_ml/diagnostics

| 파일 | 역할 |
| --- | --- |
| sensor_importance.py | 전체 Train 진단용 XGBoost gain 중요도·그래프 |
| sensor_stability.py | 기존 시간 Fold별 중요도 순위·Top-K 등장 횟수·공통 센서 |
| sensor_class_distribution.py | Train 정상·불량별 원본 ECDF·시간 구간별 중앙값·결측률 |
| step3_drift_check.py | Train/Validation 입력 분포 변화 |
| selected_sensor_drift.py | 저장된 센서 목록의 분포 변화 |
| sensor_detail.py | 지정 센서의 상세 통계·그림 생성 함수 |
| score_diagnostics.py | 점수 분포·선택 센서 교집합 보고서 생성 함수 |
| step9_error_analysis.py | FN/FP 그룹과 센서 분포 비교 |
| step5_analysis.py | 기존 PCA·모델 탐색 분석과 시각화 |
| step10_random_vs_time.py | 기존 Random/Time 탐색 비교 |
| step11_drift_check.py | 기존 시간 변화 분석 |
| step12_final_analysis.py | 기존 분석 로그 기반 보고서 그림 |

이 분류는 역할 기준입니다. 중요도·오류 분석 등 일부 진단은 모델을 학습하며, 기존 탐색 스크립트가 모두 최종 평가 프로토콜을 준수한다고 보증하지는 않습니다.

## SECOM 구현 정합성 확인 — src/sensor_ml/verification

`check_sensor_inference.py`는 전체 입력과 축소 입력의 확률·판정 일치 확인 및 후보 묶음 저장을 담당합니다. 성능 평가와는 구분합니다. 단위 테스트 파일은 이동하지 않고 프로젝트 루트의 `tests/`에 유지합니다.

## 실행 예시

프로젝트 루트에서 `python -m src.sensor_ml.<기능 폴더>.<모듈명>`으로 실행합니다. 파일 직접 실행 대신 아래 모듈 경로를 사용하세요.

```powershell
# 데이터 분할 실행 (실제 데이터 준비 작업)
python -m src.sensor_ml.data_pipeline.step3_split

# V2 모델·가중치 비교 (실제 학습 실행)
python -m src.sensor_ml.experiments.time_weight_compare --train data/splits/integrated/time_train.csv --output-dir logs/v2_time_weight_new --n-jobs 2

# 시간순 Fold별 gain 안정성: 학습을 수행하지만 성능 평가는 하지 않음
python -m src.sensor_ml.diagnostics.sensor_stability --train data/splits/integrated/time_train.csv --model v2_m0_ratio --top-k 20 --output-dir logs/v2_sensor_stability --n-jobs 2

# 전체 Train 중요도와 그림
python -m src.sensor_ml.diagnostics.sensor_importance --train data/splits/integrated/time_train.csv --model v2_m0_ratio --output-dir logs/v2_sensor_importance_new --figure-dir reports/secom/figures/v2/v2_sensor_importance_new --n-jobs 2

# 추론 정합성 확인 옵션 조회 (실제 검증·학습 실행은 하지 않음)
python -m src.sensor_ml.verification.check_sensor_inference --help

# 안정성 진단 후 선택한 5개 센서의 원본 분포 (모델 학습 없음)
python -m src.sensor_ml.diagnostics.sensor_class_distribution --train data/splits/integrated/time_train.csv --features sensor_64 sensor_46 sensor_426 sensor_65 sensor_59 --output-dir logs/v2_sensor_class_distribution --figure-dir reports/secom/figures/v2/v2_sensor_class_distribution
```

기존 로그·그림·모델 경로와 학습 설정은 폴더 정리로 변경하지 않습니다. 안정성 결과는 `fold_importances.csv`, `stability.csv`, `overlap.csv`, `fold_summary.csv`, `execution.json`, `stability.md`로 저장됩니다. 진단 결과로 최종 센서 목록을 자동 확정하지 않습니다.

노트북 import도 새 경로를 사용합니다. 프로젝트 루트를 `sys.path`에 등록하는 기존 첫 셀은 유지하세요.

```python
from src.sensor_ml.diagnostics.sensor_importance import fit_importance_candidate
from src.sensor_ml.experiments.time_weight_compare import build_weight_candidates
```

공통 모듈은 기존 import를 유지합니다. 예: `from src.modeling_config import load_modeling_config`.

## 경로 호환과 이동 범위

기존 여섯 실행기 폴더를 `src/sensor_ml/`으로 이동하고 코드·테스트·문서·노트북 셀의 import와 실행 명령을 갱신했습니다. 이동으로 깊이가 늘어난 모듈은 프로젝트 루트 계산도 조정했습니다. 데이터·로그·그림·모델 묶음은 이동하거나 재생성하지 않았습니다.

과거 `python -m src.experiments.time_weight_compare`처럼 작성한 명령도 `src/__init__.py`의 패키지 검색 경로를 통해 이동된 파일을 찾습니다. 호환용 소스 복제는 만들지 않았습니다. 새 코드·import·테스트와 명령은 `src.sensor_ml`을 기준으로 사용하며 옛 경로와 새 경로를 한 실행에서 혼용하지 않는 것이 좋습니다.

직전 `src.secom` 패키지 경로도 호환 조회를 유지합니다. 실제 소스는 `src/sensor_ml/`에만 있으며 새 개발에서는 역할 기반 이름을 사용합니다. 데이터셋별 이름은 Profile·실험 기록·문서·데이터·로그·모델 경로에서 유지합니다.

저장 모델의 클래스가 참조하는 `src.sensor_inference`, `src.modeling_models`, `src.modeling_preprocessing`, `src.stable_rf_selector` 경로는 유지합니다. 이 구조 변경은 모델 설정·센서 목록·문턱·과거 실험 수치 변경이 아닙니다.

## 전처리 에이전트

이미 변환된 WM 입력은 `agents/wm_canonical_cli.py`가 승인하고 `wm_canonical.py`가 실행을 관리합니다. `wm_canonical_contract.py`는 전체 묶음의 파일·크기·해시 복사본 계약, `wm_canonical_worker.py`·`wm_canonical_transport.py`는 기존 AppContainer+Job 연결, `wm_canonical_results.py`는 부모 검증을 담당합니다. 기존 `wafer_dl/validate_canonical.py`·`eda.py`를 재사용하며 pickle을 다시 읽지 않습니다. [실행 방법](../docs/agents/guides/PICKLE_IMPORT.md#wm-canonical의-승인격리-품질-검사와-eda)을 참고하세요.

맵 라벨 충돌로 중단한 품질 결과에는 `agents/wm_conflict_cli.py`로 별도 승인 가상 제외 후보를 연결합니다. `wm_conflict_evidence.py`는 이전 승인·품질 증거를 결합하고, `wm_conflict.py`·`wm_conflict_worker.py`·`wm_conflict_transport.py`는 기존 격리 실행을 재사용합니다. `wm_conflict_results.py`가 후보 출력 계약을 검사합니다. 실제 제외·라벨 변경·분할·EDA·학습은 하지 않습니다. [후보 진단 방법](../docs/agents/guides/PICKLE_IMPORT.md#wm-맵-라벨-충돌의-가상-제외-후보)을 참고하세요.

후보 검토 후 실제 제외는 별도 `agents/wm_exclusion_cli.py`에서 승인합니다. `wm_exclusion_evidence.py`로 완료 후보를 결합하고 `wm_exclusion.py`·`wm_exclusion_transport.py`·`wm_exclusion_worker.py`로 기존 AppContainer·Job에서 실행합니다. `wafer_dl/exclude_conflicts.py`는 새 canonical 생성 코어이며 `wm_exclusion_results.py`는 부모 결과 검증입니다. 원본·이전 결과는 보존하고 새 배열 전체를 재검증합니다. [적용 방법](../docs/agents/guides/PICKLE_IMPORT.md#승인한-충돌-후보로-새-canonical-생성)을 참고하세요.

완료된 WM EDA의 그룹 분할 제안은 `agents/wm_split_cli.py`·`wm_split.py`가 관리합니다. `wm_split_evidence.py`는 기존 EDA 완료 증거를 결합하고, `wm_split_worker.py`·`wm_split_transport.py`는 기존 `wafer_dl/split.py`의 그룹 구성·후보 비교·중복 검사를 격리 실행합니다. `wm_split_results.py`는 제안의 파일·집계·그룹 배정 계약을 검사합니다. 분할안을 만들 뿐 확정 manifest는 생성하지 않습니다. [분할안 생성 방법](../docs/agents/guides/PICKLE_IMPORT.md#완료-eda에-연결한-wm-그룹-분할-제안)을 참고하세요.

검토한 그룹 배정의 확정은 `agents/wm_split_commit_cli.py`·`wm_split_commit.py`가 별도 검토·확정 승인으로 관리합니다. `wm_split_commit_evidence.py`는 완료 제안을 결합하고 `wm_split_commit_worker.py`·`wm_split_commit_transport.py`가 고정 배정을 격리 적용합니다. `wafer_dl/commit_split.py`는 기존 `check_transform.index_split`으로 모든 원본 행의 대응을 재검증하며 `wm_split_commit_results.py`는 부모 결과 계약을 검사합니다. 후보 재탐색·배열 파싱·이미지 변환·학습은 하지 않습니다. [확정 방법](../docs/agents/guides/PICKLE_IMPORT.md#검토한-wm-그룹-분할안-확정)을 참고하세요.

확정 Train·Validation의 변환 검사는 `agents/wm_transform_cli.py`·`wm_transform.py`가 승인·검토 중단을 관리합니다. `wm_transform_evidence.py`로 확정 분할을 결합하고 `wm_transform_worker.py`·`wm_transform_transport.py`로 격리 실행하며 `wm_transform_results.py`로 결과를 검사합니다. `wafer_dl/isolated_transform_check.py`는 기존 변환·패턴 지표·그림 함수를 명시적 scratch와 출력 한도에 연결합니다. Test 배열 복원·학습 캐시는 하지 않으며 통과 후에도 사람의 그림 검토가 필요합니다. [검사 방법](../docs/agents/guides/PICKLE_IMPORT.md#확정-분할의-trainvalidation-변환패턴-보존-검사)을 참고하세요.

`agents/reviewed_module_cli.py`는 등록된 참조 모듈 하나를 고정 합성 CSV에 승인·격리 실행합니다. 계약·운송·worker·부모 검증은 `reviewed_module_contract.py`, `reviewed_module_transport.py`, `reviewed_module_worker.py`, `reviewed_module_results.py`로 분리했습니다. [합성 실행 명령과 범위](../docs/agents/guides/REVIEWED_MODULE_EXECUTION.md)를 참고하세요. 실데이터 입력 옵션은 제공하지 않습니다.

`agents/module_catalogue.py`와 `module_catalogue_cli.py`는 사용자 모듈의 고정 명세·코드 목록을 검사합니다. `preprocessing/extensions/missing_row_histogram.py`는 합성 참조 예제이며 실제 worker에는 아직 연결하지 않았습니다. [등록 절차·검증 범위](../docs/agents/design/USER_MODULE_REGISTRATION.md)를 참고하세요.

`agents/csv_handoff_cli.py`는 완료된 CSV 전처리의 전달 묶음을 생성하며, `agents/csv_handoff_contract.py`는 파일·특징 순서·행 연결을 검사합니다. 출력은 `data/processed/agents/handoffs/<name>`에 저장하고 기존 실행·Test·모델은 변경하지 않습니다. [전달 묶음 안내](../docs/agents/guides/CSV_HANDOFF.md)를 참고하세요.

`agents/preparation_pipeline_cli.py`는 변환 모듈 계획을 기존 후보 산출·적용 실행기에 연결합니다. `preprocessing/preparation_modules.py`는 고정 변환 등록부, `agents/preparation_module_plan.py`는 정적 계획 검증을 담당합니다. 후보 완료 후 사용자 검토와 별도 적용 승인을 유지하며, 기존 worker·알고리즘은 재사용합니다. [통합 CLI 사용법](../docs/agents/guides/PREPARATION_PIPELINE_EXECUTION.md)을 참고하세요.

`src/agents/`는 데이터셋 독립 실행·상태·승인·산출물 관리 영역입니다. `src/preprocessing/`에는 에이전트 없이 호출 가능한 공통 입력 검사·EDA·프로파일·실행 템플릿을 둡니다.

프로파일은 코드에 등록한 CSV 또는 WM metadata 어댑터와 EDA/학습 준비 템플릿을 선택합니다. `adapters/tabular.py`가 공통 EDA를 연결하고 `adapters/wm811k_metadata.py`는 WM 읽기·적격 행 선택을 담당합니다. WM 맵·Lot·이미지 변환 특화 기능은 `wafer_dl`에 유지하며 모든 입력에 강제하지 않습니다. 기존 SECOM 학습·문턱·설정은 변경하지 않습니다.

- 라벨·제외 기준·split·변환은 고정 설정을 따릅니다.
- 실패한 부분 산출물을 완료 결과로 등록하지 않습니다.
- 에이전트 없이도 전처리 함수를 실행·검증할 수 있도록 분리합니다.
- 합성 데모 CLI·프로파일 계획 조회를 제공하며 LLM 호출·실데이터 실행은 활성화하지 않았습니다.

구성·프로파일·최초 목표와 구현 차이는 [범용 전처리 설계](../docs/agents/design/GENERIC_PREPROCESSING_DESIGN.md), 품질 기준과 run_audit는 [품질 검사](../docs/agents/history/QUALITY_CHECK.md), Random·Group·Time 제안과 승인 후 확정은 [분할 검사](../docs/agents/history/SPLIT_CHECK.md), 설정형 Pipeline·JSON 저장/복원은 [Train-only 준비](../docs/agents/history/TRAIN_ONLY_PREPARATION.md), 독립 최종 검사와 로컬 보고서는 [최종 계약·보고서](../docs/agents/history/FINAL_CONTRACT_REPORT.md)를 따릅니다. 실행 방법은 [CLI 사용법](../docs/agents/history/CLI_USAGE.md)을 참고하세요. CSV 어댑터의 준비·최종 검사·보고서 연결까지 합성 검증했으며, Test 변환·전체 실데이터 CLI·게시·OS 격리는 후속 범위입니다.

전처리 worker의 로컬 요청·결과·경로 사전 계약은 `agents/worker_contract.py`·`worker_paths.py`에 있습니다. [worker 실행 안내](../docs/agents/security/WORKER_EXECUTION.md)를 참고하세요. 계약 검사와 별도 자원 제한 모듈을 구현했으며 기존 실행기를 격리 worker로 자동 전환하거나 시스템 권한·방화벽을 변경하지 않습니다.

`agents/appcontainer_probe.py`는 합성 격리 호환성 검사이며, `agents/wfp_capture.py`는 별도 동의를 요구하는 선택적 OS 이벤트 진단입니다. [WFP 수집 범위·사용법](../docs/agents/security/WFP_DIAGNOSTIC.md)을 확인하세요. 실제 수집·차단 판정·실데이터 worker 활성화는 서로 다른 단계입니다.

`agents/job_object.py`는 사용자 지정 메모리·프로세스 수·경과 시간 제한과 취소·전체 종료를 제공합니다. `agents/resource_probe.py`로 고정 합성 worker를 검증할 수 있습니다. [자원 제한 사용법·검증 범위](../docs/agents/security/RESOURCE_LIMITS.md)를 참고하세요. 실데이터 실행기로 자동 전환하지 않습니다.

고정 합성 CSV EDA의 승인·예약 → AppContainer+Job → 부모 검증·완료 등록 연결과 실제 정상 실행 확인은 [격리 EDA 사용법](../docs/agents/history/ISOLATED_EDA.md)에 정리했습니다. 실데이터 실행은 비활성 상태입니다.

`agents/input_snapshot.py`는 사용자 지정 파일 크기 상한으로 형식 독립적인 로컬 파일 복사본을 준비합니다. [입력 snapshot 사용법](../docs/agents/design/INPUT_SNAPSHOT.md)을 참고하세요. 내용 해석은 별도 등록 어댑터의 역할이며, 이 준비 명령은 EDA·승인·worker를 실행하지 않습니다.

`agents/eda_input_contract.py`는 snapshot·등록 CSV 프로파일·컬럼 역할·EDA 설정·자원 요청을 실행 계획에 묶습니다. [EDA 입력 계약](../docs/agents/design/EDA_INPUT_CONTRACT.md)을 참고하세요. `snapshot_eda_cli.py`·`snapshot_eda.py`·`snapshot_eda_worker.py`·`snapshot_eda_transport.py`가 승인·예약·AppContainer+Job·부모 결과 검증을 연결합니다. 원본 재파싱 없는 검증은 `eda_result_validation.py`가 담당합니다. [합성 데모의 실제 실행·기록 확인](../docs/agents/guides/SNAPSHOT_EDA.md)은 완료했으며 실데이터 EDA는 별도 검증 대상입니다.

`snapshot_audit_cli.py`·`snapshot_audit.py`·`snapshot_audit_worker.py`·`snapshot_audit_transport.py`는 기존 비파괴 품질 코어를 별도 승인·격리 실행에 연결합니다. `quality_result_validation.py`가 부모의 제한된 결과 검사를 담당합니다. [품질 검사 사용법](../docs/agents/guides/SNAPSHOT_AUDIT.md)을 참고하세요. 검토·차단 보고서는 완료 커밋하지 않으며 EDA로 자동 진행하지 않습니다. 새 audit 경로의 실제 OS 실행 확인은 아직 남아 있습니다.

## WM-811K 맵 처리와 딥러닝

`agents/wm_dataset_cli.py`·`wm_dataset.py`·`wm_dataset_contract.py`는 완료된 개발 캐시의 Dataset·첫 배치 검사를 별도 승인으로 관리합니다. `wm_dataset_runtime.py`는 설치된 PyTorch의 제한 복사 명세를, `wm_dataset_transport.py`·`wm_dataset_worker.py`는 AppContainer·Job의 CPU 검사를, `wm_dataset_results.py`는 부모의 결과 계약 검사를 담당합니다. 기존 `wafer_dl/dataset.py`를 재사용하며 Test·모델 학습·Tensor 외부 전달은 처리하지 않습니다. [Dataset 검사 사용법](../docs/agents/guides/PICKLE_IMPORT.md#완료-캐시의-dataset첫-배치-검사)을 참고하세요.

`agents/wm_cache_cli.py`·`wm_cache.py`·`wm_cache_evidence.py`는 완료된 변환 검사와 확정 분할에 연결해 그림 확인·분할 재확인·캐시 실행 승인을 관리합니다. `wm_cache_worker.py`·`wm_cache_transport.py`는 기존 AppContainer·Job을, `wm_cache_results.py`는 숫자 전용 NPY·색인 행 연결 검증을 담당합니다. `wafer_dl/isolated_cache.py`는 기존 캐시 writer와 변환 함수를 재사용하며 명시적 scratch·출력 예산을 연결합니다. [Train·Validation 캐시 생성](../docs/agents/guides/PICKLE_IMPORT.md#확인한-변환-설정으로-trainvalidation-캐시-생성)을 참고하세요. Test 배열·모델 학습은 처리하지 않습니다.

`src/wafer_dl/validate_canonical.py`는 변환된 NPZ·manifest 정합성, 중복·Lot 연결 그룹의 분할 필요조건을 검사합니다. 원본 pickle을 읽거나 데이터를 수정하지 않습니다. 실행 안내는 [Sandbox 준비·산출물 검증](../docs/WM811K/SANDBOX_SETUP.md#6-호스트에서-변환-산출물-검증)에 있습니다.

`split.py`는 충돌 맵 전체 격리와 Lot·동일 맵 그룹 분할을 담당합니다. [그룹 분할 실행](../docs/WM811K/GROUP_SPLIT.md)을 참고하세요. `transform.py`는 NumPy 기반 결정적 resize·padding·2채널 생성을, `check_transform.py`는 Train/Validation의 패턴 보존 검사를 담당합니다. [변환 검사 실행](../docs/WM811K/TRANSFORM_CHECK.md)을 참고하세요. `cache.py`와 `dataset.py`는 숫자 전용 입력 캐시와 PyTorch 첫 배치 검사를 담당합니다. [캐시·Dataset 실행](../docs/WM811K/CACHE_DATASET.md)은 실제 데이터에서 별도로 수행하며, 딥러닝 학습 모듈은 아직 미구현입니다.

`src/wafer_dl/eda.py`는 기존 WM 노트북·CLI용 도메인 분석입니다. `reports/wm811k/analysis.ipynb`의 결과와 사용법을 유지합니다. 새 에이전트의 기본 분석은 `preprocessing/eda.py`이며 WM metadata 어댑터가 공통 입력을 전달합니다. 맵 예시 등 특화 분석은 별도 확장으로 연결합니다. 집계·그림·경로는 외부 LLM·추적에 전달하지 않습니다.

| 영역 | 예정 모듈 | 담당 |
| --- | --- | --- |
| 맵 데이터 준비 | ingestion, audit, split, dataset, transforms | 양현승 |
| 딥러닝 | models, train, metrics, evaluate, predict | 이종수 |

전처리 에이전트의 작업 관리 코드는 `src/agents/`에서 위 데이터 준비 함수를 재사용합니다. 여기서 다시 에이전트 실행 관리 코드를 중복 구현하지 않습니다.

작은 CNN과 ResNet18, 9종 패턴 분류, Lot 비중첩 분할을 계획합니다. SECOM 이진 분류 설정·지표를 그대로 사용하지 않으며 입력 채널·클래스 매핑·맵 변환·학습 조건은 별도 계약으로 관리합니다. `config.json`의 SECOM Profile을 WM 설정으로 교체하지 않습니다.

구현 전의 예정 모듈은 파일이나 실행 명령으로 제공하지 않습니다. 데이터·모델·전처리 계약과 평가 범위는 [WM-811K 기초 설계](../docs/WM811K/WM811K_BASIC_DESIGN.md)를 참고하세요.
