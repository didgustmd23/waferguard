# WaferGuard 소스 구성

전처리 에이전트는 [사용 안내](../docs/agents/GUIDE.md)와 [단일 설계](../docs/agents/DESIGN.md)를 따릅니다. 구현·검증 상태를 이 파일에 중복 기록하지 않습니다. 아래는 센서 ML·맵 처리 코드를 찾기 위한 안내입니다.

## 공통 코어 — src 바로 아래

pickle 변환은 `agents/pickle_import_cli.py`·`pickle_import.py`에서 승인·실행을 관리합니다. `pickle_contract.py`가 선택 계약을, `pickle_converters.py`가 구조 판별과 표·숫자 배열·WM 변환기 연결을 담당합니다. 공통 데이터 코어는 `preprocessing/canonical_data.py`입니다. 격리 실행은 `pickle_import_transport.py`·`pickle_import_worker.py`, 부모 검증은 `pickle_results.py`에 있습니다. 기존 `wm_pickle_*` 파일은 호환 진입점입니다. [실행 방법과 지원 범위](../docs/agents/guides/PICKLE_IMPORT.md)를 참고하세요.

저장 기준 적용은 `agents/preparation_apply_cli.py`·`preparation_apply.py`가 관리합니다. 완료 후보 증거·두 구간 계약은 `preparation_apply_contract.py`, 격리 복원/변환과 운송은 `preparation_apply_worker.py`·`preparation_apply_transport.py`, 부모 검증은 `preparation_apply_results.py`가 담당합니다. [출력 파일·승인·실행 안내](../docs/agents/guides/PREPARATION_APPLY.md)를 참고하세요. fit·Test 적용·모델 학습은 수행하지 않습니다.

Train 후보 기준 산출은 `agents/preparation_candidate_cli.py`·`preparation_candidate.py`가 관리합니다. 입력 증거는 `preparation_candidate_contract.py`, 격리 계산·운송은 `preparation_candidate_worker.py`·`preparation_candidate_transport.py`, 부모 파일 검증은 `preparation_candidate_results.py`가 담당합니다. 기존 `preprocessing/transform.py`를 재사용하며 변환 적용·모델 학습은 하지 않습니다. [구현 범위·사용법](../docs/agents/guides/PREPARATION_CANDIDATE.md)을 참고하세요.

분할 준비는 `agents/snapshot_split_cli.py`·`snapshot_split.py`가 관리하며, `snapshot_split_worker.py`·`snapshot_split_transport.py`가 격리 품질·EDA·제안·확정을 수행합니다. [SECOM Train 기능 검증 절차](../docs/agents/guides/SNAPSHOT_SPLIT.md)를 참고하세요. 기존 `preprocessing/split.py`를 재사용하며 ML 실험 분할·모델은 변경하지 않습니다. 합성 테스트는 통과했고 실제 Windows 실행은 별도 확인 대상입니다.

구간 추출은 `agents/partition_export_contract.py`·`partition_export_worker.py`, 부모 파일 검증은 `partition_export_results.py`, 제한된 CSV 파서는 `partition_csv.py`에 있습니다. 승인·커밋 연결은 `partition_export.py`, AppContainer+Job 운송은 `partition_export_transport.py`, 사용자 명령은 `partition_export_cli.py`가 담당합니다. [구간 내보내기 구현 범위·사용법](../docs/agents/guides/PARTITION_EXPORT.md)을 참고하세요. 합성 연결 테스트는 통과했으며 실제 Windows 격리 실행은 별도 확인이 필요합니다.

변환 준비의 실행 증거는 `agents/split_evidence.py`, 구간 입력 결합은 `agents/partition_input_contract.py`가 검사합니다. [확정 분할·구간 입력 계약](../docs/agents/DESIGN.md#분할내보내기train-기준)을 참고하세요. 기존 분할 결과를 읽기 전용으로 확인하며 실제 내보내기·변환은 실행하지 않습니다.

선택 모듈 실행은 `agents/module_pipeline_cli.py`·`module_pipeline.py`가 관리하며, worker·운송·파일 검증은 각각 `module_pipeline_worker.py`·`module_pipeline_transport.py`·`module_pipeline_results.py`에서 담당합니다. [합성 예제 실행·검증 범위](../docs/agents/guides/MODULE_PIPELINE_EXECUTION.md)를 참고하세요. 기존 고정 EDA·audit CLI는 유지합니다.

`preprocessing/pipeline.py`는 모듈 선택 JSON의 순서·포트·계약·설정을 검증합니다. 예제는 `configs/agents/pipelines/tabular_eda_example.json`이며 실제 실행 명령이나 데이터 경로를 받지 않습니다. [설정 방법·검증 범위](../docs/agents/DESIGN.md#모듈과-파이프라인)를 참고하세요.

모듈형 전처리의 등록 계약은 `preprocessing/module_contracts.py`, 목록 관리·정확한 버전 조회는 `module_registry.py`, 내장 EDA 5개·품질 검사 연결은 `builtin_modules.py`에서 관리합니다. [등록부 사용·경계](../docs/agents/DESIGN.md#모듈과-파이프라인)를 참고하세요. 새 모듈 CLI·worker의 합성 연결을 확인했으며 실제 OS 실행은 별도 검증 대상입니다.

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

`src/preprocessing/`는 독립 계산·계약, `src/agents/`는 실행·승인·검증·어댑터, `src/ui/`는 공통 화면입니다. 전체 구조와 입력/계획/복구/정보 전달 계약은 [DESIGN](../docs/agents/DESIGN.md), 실제 실행은 [GUIDE](../docs/agents/GUIDE.md)를 참조하세요. 명령과 현재 상태를 여기에 반복하지 않습니다.

## WM-811K 맵 처리와 딥러닝

`src/wafer_dl/`는 canonical 검증·EDA·그룹 분할·변환·캐시·Dataset을 담당하고, `scripts/wm_sandbox/`는 원본 구조 검사/변환 진입점입니다. 에이전트는 `src/agents/`의 기존 서비스를 통해 이 코어를 호출합니다.

현재 WM 처리 코드는 canonical 검증, 그룹 분할, 변환·캐시·Dataset과 CNN 학습·평가로 구성됩니다. 구현과 실행 경로는 [WM-811K 설계](../docs/WM811K/PREPROCESSING_DESIGN.md), 재현 방법·결과·한계는 [캐시·학습·평가 기록](../docs/WM811K/CACHE_DATASET.md)을 참고하세요. 작은 CNN의 Train/Validation 학습과 동결 Test 1회 평가를 완료했으며, ResNet18 비교와 seed 변동은 후속입니다. 모델 가중치와 실행별 보고서는 로컬에만 보관합니다.
