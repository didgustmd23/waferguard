# SECOM 실행 안내

환경 준비·데이터 생성·개별 실험의 실행 방법입니다. 아래 일반 Step 명령은 실행 예시이며 최신 V2 결과를 그대로 재현하는 설정은 아닙니다. V2 실험별 조건과 명령은 [통합 실험 노트](EXPERIMENT_NOTES.md), 고정 센서 모델 보존 절차는 아래 [V2 시나리오 저장·복원](#v2-시나리오-저장복원)을 따릅니다.

설정 파일 작성법은 [Dataset Profile 안내](DATASET_PROFILE.md), 과거 실험 수치는 [보고서](report.md)에 분리했습니다.

## 시작하기

아래 명령은 프로젝트 루트에서 실행합니다. 데이터와 모델 묶음은 저장소에 포함하지 않으므로 원본 준비 또는 별도 재현이 필요합니다.

```bash
git clone <repository-url>
cd secom-waferguard
python -m venv .venv
```

Windows PowerShell에서는 가상환경을 활성화한 뒤 의존성을 설치합니다.

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 자동 테스트

현재 구현된 범용 코어는 아래 명령으로 검증합니다.

```powershell
python -m unittest discover -s tests -v
```

UCI의 원본 데이터 CSV는 저장소에 포함하지 않습니다. [UCI SECOM Data Set](https://archive.ics.uci.edu/dataset/179/secom)에서 `secom.data`, `secom_labels.data`를 받아 `data/raw/`에 저장합니다. 원본 파일은 수정하지 않고, 아래 Script로 재현 가능한 산출물을 생성합니다.

### Day 1 데이터 준비

```powershell
python -m src.sensor_ml.data_pipeline.step1_merge_data
python -m src.sensor_ml.data_pipeline.step2_data_check
```

`step1_merge_data.py`는 Profile의 `ingestion` source 정의로 원본 파일을 병합하여 canonical 입력(`data/processed/secom_merged.csv`)을 생성합니다. 따라서 원본 경로나 읽기 옵션, label/timestamp 열 순서를 코드에 고정하지 않아 같은 수집 형식의 다른 데이터셋 Profile에도 처리 흐름을 적용할 수 있습니다.

`step2_data_check.py`는 결측률·상수 feature·timestamp·label 분포를 `logs/secom/data/dataset_log.csv`에 기록합니다. 이 결과는 전체 데이터의 EDA 후보 정보일 뿐이며, 이를 사용해 `secom_cleaned.csv`를 만들거나 모델 feature를 전역에서 제거하지 않습니다. 실제 결측 처리와 feature 선택은 Train/CV 학습 fold 안에서만 fit하여 데이터 누수를 방지합니다. 두 CSV 산출물은 원본과 Profile만 있으면 재생성 가능하므로 Git에서 추적하지 않습니다.

### 모델링 실험

XGBoost의 전체 센서·gain 중요도 Top-20·RF 중요도 Top-20은 아래 명령으로 비교합니다. 최종 분류기는 모두 같은 XGBoost이며 전처리·선택은 학습 fold 내부에서만 fit합니다. `--experiments`를 생략하면 기존 PCA·L1·LightGBM 비교 목록을 실행합니다. `--details-dir`는 비어 있는 새 폴더를 지정하며 fold별 지표·선택 센서·설정을 저장합니다.

```powershell
python -m src.sensor_ml.experiments.step6_feature_compare --train data/splits/integrated/time_train.csv --experiments xgboost_all xgboost_top_20 xgboost_rf_top_20 --output logs/xgboost_top20_run/feature_compare.csv --details-dir logs/xgboost_top20_run/details --n-jobs 2
```

후속 시간순 검증은 같은 코어에서 전체 XGBoost와 RF Top-20만 지정해 실행할 수 있습니다. 과거 학습→다음 구간 평가 3회와 각 과거 구간의 내부 시간순 OOF 2회를 사용하며, 선택 센서·OOF threshold·정책 미충족을 별도로 기록합니다.

```powershell
python -m src.sensor_ml.experiments.feature_time_compare --train data/splits/integrated/time_train.csv --experiments xgboost_all xgboost_rf_top_20 --output-dir logs/xgboost_top20_time_run --n-jobs 2
```

시간순 센서 축소 비교는 `src/sensor_ml/experiments/feature_time_compare.py`로 실행합니다. 전체 센서·LightGBM 중요도 Top-50/20·RF 중요도 Top-50/20을 동일한 LightGBM 분류기로 비교합니다. 선택기는 각 학습 fold 내부에서만 fit하며 기존 시간 검증 코어의 외부 3구간·내부 시간순 OOF 2구간을 재사용합니다.

```powershell
python -m src.sensor_ml.experiments.feature_time_compare --train data/splits/integrated/time_train.csv --output-dir logs/feature_time_run --n-jobs 2
```

`selected_features.csv`는 각 fit의 실제 특징 목록·선택 중요도·순위, `feature_frequency.csv`는 외부 fit 3회·내부 OOF fit 6회 각각의 선택 빈도입니다. `candidate_features.csv`는 전체 Time Train에서 만든 **최종 미확정 후보**이며 이를 이용해 과거 CV 특징을 전역 고정하지 않습니다. 범주형을 One-Hot으로 확장한 특징은 원본 센서 수와 구분합니다. 실제 결과는 `logs/secom/exploratory/feature_time_20261006/`, 보고서 24절, 분석 노트북 7절에서 확인할 수 있습니다. 외부 Validation·Test는 읽지 않고 기존 실행기 대상도 자동 변경하지 않습니다.

정상만 학습하는 이상 탐지는 `src/sensor_ml/experiments/anomaly_compare.py`로 실행합니다. 센서 품질 필터·대치·표준화와 Isolation Forest 또는 PCA90을 각 학습 구간의 정상 데이터에만 fit합니다. Isolation Forest는 `-score_samples`, PCA는 표준화된 입력의 평균 제곱 복원오차를 사용해 클수록 이상으로 통일합니다. 두 점수는 확률이 아니므로 기본 확률 threshold 0.5를 적용하지 않습니다. Isolation Forest의 점수 방향은 [공식 문서](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)를 따릅니다.

```powershell
python -m src.sensor_ml.experiments.anomaly_compare --train data/splits/integrated/time_train.csv --output-dir logs/anomaly_run --supervised-dir logs/secom/exploratory/temporal_quality_filtered_20261006_final --n-jobs 2
```

출력 폴더는 새 폴더여야 합니다. 기존과 같은 외부 시간 구간 3개·내부 시간순 OOF 구간 2개를 사용하며 초기 미예측 행은 제외합니다. `fold_results.csv`는 OOF F1 최대 진단 후보의 다음 구간 성능, `policy_selection.csv`는 현재 설정 정책의 선택·미충족 결과입니다. 정책 미충족이어도 AP와 F1 진단 비교는 저장하지만 임의 후보를 정책 통과로 취급하지 않습니다. `anomaly_compare.csv`에는 생성 계약·학습 설정·시간 경계가 맞는 기존 지도학습 진단을 연결합니다. `quality_filter.csv`에는 모든 fit의 정상 표본 ID·제거 개수·PCA 성분 수가 기록됩니다. 점수 규모가 학습 구간에 따라 달라질 수 있어 OOF 문턱의 미래 전이는 별도 검증이 필요합니다. 최종 Test와 외부 Validation은 읽지 않습니다.

변경 후 OOF와 시간 검증을 연결할 때는 `src/sensor_ml/evaluation/run_evaluation.py`를 사용합니다. `--change preprocessing`, `model`, `data`, `split`은 Step 8 OOF·threshold 비교표를 재생성한 뒤 Step 7 시간 검증을 실행합니다. `--change threshold`는 이 실행기로 성공한 같은 폴더의 비교표를 재사용하며 `--threshold`가 필수입니다. OOF 생성이 실패하면 후속 평가를 중단하고, 실패한 실행 폴더의 과거 비교표 재사용도 거부합니다. 상태와 설정은 `execution.json`에 기록합니다.

```powershell
# 전처리 변경: OOF 생성 → 후보 threshold 선택 → 시간 검증
python -m src.sensor_ml.evaluation.run_evaluation --change preprocessing --train data/splits/integrated/time_train.csv --validation data/splits/integrated/time_valid.csv --run-dir logs/evaluation_run --n-jobs 2

# threshold 변경: 같은 실행 폴더의 비교표 후보값을 지정
python -m src.sensor_ml.evaluation.run_evaluation --change threshold --threshold <비교표의_반올림하지_않은_값> --train data/splits/integrated/time_train.csv --validation data/splits/integrated/time_valid.csv --run-dir logs/evaluation_run --n-jobs 2

# config.json의 threshold_policy만 변경: 기존 OOF로 정책 후보 재선택
python -m src.sensor_ml.evaluation.run_evaluation --change policy --train data/splits/integrated/time_train.csv --validation data/splits/integrated/time_valid.csv --run-dir logs/evaluation_run --n-jobs 2
```

`--change policy`는 정책만 바뀐 경우 OOF를 재사용하며 `--threshold`는 지정하지 않습니다. 학습 설정·Train 경로가 바뀌면 재사용을 거부합니다. 같은 경로의 파일 내용이나 코드 변경은 자동 감지하지 않으므로 해당 변경 유형으로 OOF부터 재생성해야 합니다. 수동 `--threshold` 지정은 정책을 우회하는 연구용 실행이며 정책 충족을 뜻하지 않습니다. 연결 대상은 `--experiment`로 `lightgbm_all`, `xgboost`, `xgboost_scale_pos_weight` 중에서 지정합니다(기본값 `lightgbm_all`). Baseline·모델 비교·특징 비교·오류 분석은 아래 개별 명령으로 실행합니다. OOF 재사용 시에도 Step 7은 동일 Train으로 모델을 다시 학습합니다. 개별 Step 7 실행에는 실행 상태 검사가 적용되지 않습니다. 최종 Test 평가 전에 정책을 확정하고 Test 결과에 맞춰 바꾸지 않습니다.

`temporal_validation.py`는 기존 F1 최대 진단을 유지하면서 `policy_selection.csv`에 내부 OOF의 정책 충족 여부·선택값과 다음 구간 성능을 별도로 기록합니다. 미충족 구간에는 threshold와 정책 기반 평가값을 만들지 않습니다. 기존 보고서의 수치는 과거 실행 결과이며 새 정책 실험 결과로 간주하지 않습니다.

팀원의 `step3_split.py`를 공식 진입점으로 사용한다. 기존 Random/Time 분할·EDA·클래스 요약·누수 검사 함수를 유지하면서 공통 Profile 검증과 원본 ID·split 계약을 통합했다. 별도 `step3_split_data.py`는 제거했다. 기본 출력은 `data/splits/integrated/`, 점검 로그는 `logs/secom/data/step3_integrated/`, 그림은 `reports/secom/figures/eda/step3/`이며 기존 split CSV는 덮어쓰지 않는다.

```powershell
# 시간 holdout을 먼저 분리 (70/15/15는 목표 비율이며 동일 timestamp·그룹은 분리하지 않음)
python -m src.sensor_ml.data_pipeline.step3_split

# Time Train/Validation baseline
python -m src.sensor_ml.experiments.step4_baseline --train data/splits/integrated/time_train.csv --validation data/splits/integrated/time_valid.csv --split-strategy time --output logs/secom/data/integrated/baseline.csv

# 동일한 Time Train 내부 반복 CV 후보 모델 비교
python -m src.sensor_ml.experiments.step5_model_compare --train data/splits/integrated/time_train.csv --output logs/secom/data/integrated/model_compare.csv

# 위 후보 목록에는 XGBoost 일반·scale_pos_weight 실험도 포함됨

# PCA·L1·LightGBM Top-K 특징 선택 비교
python -m src.sensor_ml.experiments.step6_feature_compare --train data/splits/integrated/time_train.csv --output logs/secom/data/integrated/feature_compare.csv --n-jobs -1

# Train 내부 세 시간 구간: 기본 threshold·계층 OOF·시간순 OOF 전이 비교
python -m src.sensor_ml.experiments.temporal_validation --train data/splits/integrated/time_train.csv --output-dir logs/temporal_validation_run --n-jobs 2

# LightGBM 전체 특성의 OOF 확률·threshold 비교
python -m src.sensor_ml.experiments.step8_threshold_oof --train data/splits/integrated/time_train.csv --oof-output logs/secom/data/integrated/oof_predictions.csv --threshold-output logs/secom/data/integrated/threshold_compare.csv --n-jobs -1

# CV 결과에서 선택한 XGBoost 후보의 OOF threshold 비교
python -m src.sensor_ml.experiments.step8_threshold_oof --train data/splits/integrated/time_train.csv --experiment xgboost --oof-output logs/secom/data/integrated/xgboost_oof_predictions.csv --threshold-output logs/secom/data/integrated/xgboost_threshold_compare.csv --n-jobs -1

# 시간 검증: 생략 시 후보 비교 기본 threshold 사용. 새 Train에서 정한 값만 --threshold로 주입
python -m src.sensor_ml.experiments.step7_time_validation --train data/splits/integrated/time_train.csv --validation data/splits/integrated/time_valid.csv --experiments lightgbm_all --output logs/secom/data/integrated/time_validation.csv --n-jobs -1

# 후보 비교에서 XGBoost를 선택한 경우 같은 시간 분할에서 비교
python -m src.sensor_ml.experiments.step7_time_validation --train data/splits/integrated/time_train.csv --validation data/splits/integrated/time_valid.csv --experiments xgboost --output logs/secom/data/integrated/xgboost_time_validation.csv --n-jobs -1
```

`temporal_validation.py`는 Train 파일만 받으며, 고유 timestamp 단위의 3개 외부 시간 구간에서 지정한 후보 모델을 비교합니다. 기본 목록에는 LightGBM·Random Forest·균형 가중 Logistic Regression·XGBoost가 포함됩니다. 내부 계층 OOF는 설정의 fold 수와 1회 반복, 내부 시간순 OOF는 기본 2개 확장형 fold를 사용합니다. 초기 미예측 행은 제외하고 `oof_coverage.csv`에 기록합니다. `folds.csv`, `fold_results.csv`, `summary.csv`, 예측·OOF·threshold·제거 로그와 `temporal_run.json`을 저장하며 기존 결과를 보존하기 위해 비어 있는 출력 폴더를 요구합니다. 모델·구간별 threshold가 다르므로 합산 Recall·Precision은 단일 최종 모델의 성능이 아닙니다.

Step 3을 재실행할 때는 새로운 `--output-dir`과 `--log-dir`을 지정한다. `--config`, `--input`, `--figures-dir`로 경로를 변경할 수 있고, `--skip-eda`는 분할·검사만 수행한다. 그룹이 없는 Random split은 기존 stratify 방식을 유지한다. 그룹이 선언되면 GroupShuffleSplit을 사용하므로 정확한 행 비율·class 비율은 보장되지 않으며 실제 비율을 요약에서 확인한다. 누수 검사 실패 시 split CSV를 저장하지 않고 오류로 종료한다.

위 LightGBM 명령은 실행 예시이며 최종 후보 확정을 뜻하지 않는다. 기존 `0.000475`를 새 평가 threshold로 그대로 승계하지 않는다. Step 8의 기본 `--score-method single`은 1회 K-fold의 샘플별 OOF 예측을 사용한다. 후보 비교의 5-fold × 5-repeats 정책은 그대로 유지한다. `--score-method repeated_mean`은 반복 평균 분석용이며 결과의 `threshold_use=analysis_only`로 구분한다. 단일 OOF도 fold 학습과 전체 Train 재학습의 확률 척도 일치를 보장하지 않으며, 시간순 내부 검증·확률 보정은 후속 점검 대상이다.

새 OOF 비교표에서 threshold를 선택한 뒤 Step 7에 `--threshold-report logs/secom/data/integrated/threshold_compare.csv --threshold <비교표의_반올림하지_않은_값>`을 전달하면 데이터셋·Train 생성 계약·모델·단일 OOF 방식과 값의 존재 여부를 검사한다. `<…>`는 실제 숫자로 바꿔야 한다. 비교표 없이 실행하는 기존 CLI는 탐색용 호환 경로이며 출처 확인을 보장하지 않는다. 같은 계약이라도 모델 설정을 변경했다면 비교표를 다시 생성해야 한다.

새 split의 `__source_row_id`는 원본 CSV 내 행 위치, `__split_role`은 train/validation/test, `__split_protocol_id`는 원본·Profile·비율의 생성 계약 해시다. 이 컬럼들은 모델 입력에서 제외한다. Baseline·시간 검증은 중복 행과 계약 혼합을 검사하고, 후보 비교·OOF는 Validation/Test 역할의 입력을 거부한다. 기존 metadata 없는 CSV도 사용할 수 있지만 출처까지 보장하지 못하므로 탐색용으로만 취급한다.

모든 모델링 Pipeline은 센서 품질 필터 이후 수치형 median 대치와 선언된 범주형 최빈값 대치/One-Hot Encoding을 공통으로 사용한다. SECOM의 결측률 50% 초과 기준에서는 전부 결측인 컬럼도 제거한다. Profile에서 품질 제거를 비활성화한 경우에는 전부 결측인 수치형 컬럼을 대치 단계에서 0으로 보존한다. Top-K와 특징 수는 변환 후 기준이며 범주형 입력에서는 센서 원본 컬럼 수와 One-Hot 특징 수가 다를 수 있다.

## V2 저장 묶음 검증과 예측

저장 묶음은 로컬 산출물입니다. 먼저 직접 생성한 신뢰한 묶음을 준비해야 합니다. joblib 복원은 코드 실행 위험이 있으므로 출처가 불명확한 파일에 신뢰 옵션을 사용하지 않습니다.

사용 순서는 다음과 같습니다. 로컬 `ml310`에서만 검증했으며 다른 PC·OS의 환경 구성은 검증 범위가 아닙니다.

1. [V2 Model Card 8.1절](model_card_v2.md#81-인계-기준)의 기준 커밋을 사용합니다.
2. 두 묶음 폴더를 비공개 경로로 받아 `models/candidates/` 아래 원래 이름으로 둡니다. 같은 이름의 폴더가 있으면 덮어쓰지 말고 먼저 내용을 확인합니다.
3. 신뢰 옵션을 쓰기 전에 각 파일의 SHA256을 Model Card 표와 대조합니다. 하나라도 다르면 복원하지 않습니다.
4. Model Card 7절의 고정 버전과 정확히 같은 Python 3.10 환경을 사용합니다. 공통 `requirements.txt` 환경(Python 3.12)에서는 버전 검사로 복원이 거부됩니다.
5. 아래 복원 확인에서 `status: passed`를 확인한 뒤 예측을 실행합니다.

```powershell
python -m src.sensor_ml.inference.sensor_bundle --bundle-dir models/candidates/v2_fixed20_recall90 --trusted-local-bundle
python -m src.sensor_ml.inference.predict_cli --bundle-dir models/candidates/v2_fixed20_recall90 --input data/processed/new_sensor20.csv --output logs/v2_new_recall90_predictions.csv --trusted-local-bundle
python -m src.sensor_ml.inference.predict_cli --bundle-dir models/candidates/v2_fixed20_recall80 --input data/processed/new_sensor20.csv --output logs/v2_new_recall80_predictions.csv --trusted-local-bundle
```

`new_sensor20.csv`는 사용자가 준비할 입력 예시입니다. 자동 생성되지 않습니다.

- UTF-8 CSV 첫 행은 `manifest.json`의 `sensors`에 기록된 센서명입니다. 해당 센서만 포함하며 순서는 달라도 됩니다.
- label·timestamp·행 ID·인덱스·추가 센서는 제외합니다. 누락·중복 센서는 오류입니다.
- 값은 숫자여야 합니다. 빈 값·NaN은 저장된 median으로 대치합니다. 무한대·잘못된 타입·모든 필수 센서가 결측인 행은 거부합니다.
- 한 행 이상 필요하며 입력 오류 시 일부 행을 자동 제외하지 않습니다. 기존 출력 파일도 덮어쓰지 않습니다.
- 센서 의미·단위는 학습 데이터와 같아야 합니다. 다른 공장에 모델을 그대로 적용할 수 있다는 뜻은 아닙니다.

출력은 `input_row_index`(0부터 시작하는 입력 데이터 행 번호), `positive_score`(모델 양성 확률 출력), `predicted_positive`(저장된 문턱 이상으로 위험 대상에 선별됐는지), `predicted_label`(원래 label 공간의 모델 예측)입니다. 필드명은 호환성을 위해 유지합니다. 확률 보정이나 실제 불량 확정을 보장하지 않으며 음성 결과도 출하 승인이 아닙니다.

신규 입력이 없다면 Train 유래 데모로 CLI 동작만 확인할 수 있습니다. 성능 평가로 해석하지 않습니다.

```powershell
python -m src.sensor_ml.inference.predict_cli --bundle-dir models/candidates/v2_fixed20_recall90 --export-demo-input data/processed/v2_sensor20_demo.csv --trusted-local-bundle
python -m src.sensor_ml.inference.predict_cli --bundle-dir models/candidates/v2_fixed20_recall90 --input data/processed/v2_sensor20_demo.csv --output logs/v2_recall90_demo_predictions.csv --trusted-local-bundle
```

## 분석 노트북

[analysis.ipynb](../../reports/secom/analysis.ipynb)는 기존 로컬 로그를 읽어 표와 그래프를 표시합니다. 선택 의존성은 `python -m pip install -r requirements-notebook.txt`로 설치합니다. 첫 설정 셀의 실행 폴더를 지정한 뒤 Run All을 실행합니다. 로그가 바뀌면 셀을 다시 실행해야 화면에 반영됩니다. 학습이나 최종 Test 평가는 실행하지 않습니다. 저장된 과거 셀 출력은 당시 기록입니다.

`--n-jobs`를 지원하는 실험은 모델 내부 계산의 병렬 수를 조절합니다. 자원이 제한되면 `1` 또는 `2`를 사용합니다. 출력 보존을 위해 새 출력 폴더와 파일 경로를 지정하세요.

[소스 역할 안내](../../src/README.md) · [문서 목록](../README.md#secom-문서) · [프로젝트 소개](../../README.md)

## V2 시나리오 저장·복원

이미 완료한 저장 작업의 재현 절차입니다. 기존 폴더를 덮어쓰거나 기록 정리를 위해 재학습·재평가하지 마세요. 새 저장은 사용자 요청과 새 출력 폴더가 필요합니다. 현재 모델·문턱·환경·평가 범위·복원 결과는 [V2 Model Card](model_card_v2.md) 한 곳을 기준으로 합니다.

90% 저장은 기존 학습 범위로 재학습하고, 80%는 신뢰한 90% 묶음의 분류기·전처리를 복사해 문턱만 바꿉니다. 원본 OOF 기록과 묶음의 experiment 사본은 보존하며 두 시나리오는 연구용·배포 미승인입니다.

**옛 경로 제약:** 기존 `logs/v2_fixed_sensor20_oof/execution.json`은 저장 당시 위치 `D:\workspace\secom-waferguard`의 Train·후보 폴더 경로를 기록합니다. 90% 저장은 `--train`과 이 기록 경로가 같고 후보 폴더가 존재해야 하므로, 현재 위치에서 기존 OOF 폴더로 아래 명령을 실행하면 중단됩니다. 기록 경로를 수정하거나 옛 폴더를 만들어 맞추지 않습니다. 처음부터 재현할 때는 [실험 노트](EXPERIMENT_NOTES.md)의 고정 센서 재현 명령으로 새 후보·OOF 폴더를 만들고, `--oof-dir`와 `--bundle-dir`에 새 경로를 지정합니다. 이 재현 경로와 결과 일치는 아직 실행으로 확인하지 않았습니다.

### 1. 90% 시나리오 저장

```powershell
python -m src.sensor_ml.inference.save_v2_scenario `
  --train data/splits/integrated/time_train.csv `
  --oof-dir logs/v2_fixed_sensor20_oof `
  --target-recall 0.9 `
  --bundle-dir models/candidates/v2_fixed20_recall90 `
  --n-jobs 2
```

기존 마지막 개발 구간의 확률과 새 학습 확률이 일치해야 저장한다. 문턱은 저장된 CSV의 정확한 값으로 읽으며 출력의 반올림값을 입력하지 않는다. 원래 검증 범위인 824행만 학습하고 마지막 272행을 학습에 추가하지 않는다. 새 독립 Test나 추가 성능 개선을 수행하는 명령이 아니다.

### 2. 별도 프로세스로 90% 복원 확인

```powershell
python -m src.sensor_ml.inference.sensor_bundle `
  --bundle-dir models/candidates/v2_fixed20_recall90 `
  --trusted-local-bundle
```

`status: passed`를 확인한다. joblib은 임의 코드 실행 위험이 있으므로 직접 생성한 신뢰한 파일에만 이 옵션을 사용한다. 저장·복원 정합성과 탐지 성능은 별도다.

### 3. 90% 확인 후 80% 분기 저장

```powershell
python -m src.sensor_ml.inference.save_v2_scenario `
  --oof-dir logs/v2_fixed_sensor20_oof `
  --target-recall 0.8 `
  --base-bundle-dir models/candidates/v2_fixed20_recall90 `
  --trusted-local-bundle `
  --bundle-dir models/candidates/v2_fixed20_recall80
```

80%는 90%를 덮어쓰지 않는다. 같은 OOF 실행 기록과 센서 순서를 확인하고 분류기·통계는 유지한 채 문턱만 바꾼다. 추가 실험은 먼저 이 두 저장 시나리오를 기준으로 비교하고, 모델·전처리 변경이 필요하면 별도 버전으로 진행한다.

### 4. 80% 복원 확인

```powershell
python -m src.sensor_ml.inference.sensor_bundle `
  --bundle-dir models/candidates/v2_fixed20_recall80 `
  --trusted-local-bundle
```


## 기존 V1 동결 평가와 남은 제출 확인

V1 S0·M3는 사용 중단된 연구용 모델입니다. 동결 조건·평가·미사용 여부 한계는 [V1 Model Card](model_card.md), 상세 결과는 [보고서 34절](report.md#34-동결-저장-모델의-후속-시간-구간-평가-2026-10-07)을 따릅니다. 평가 실행기는 `src/sensor_ml/evaluation/evaluate_frozen_model.py`입니다. 완료 기록을 재현하려고 실행 표식을 삭제하거나 다른 폴더로 복사해 다시 평가하지 않습니다. V1 문턱/지표를 V2에 승계하지 않습니다.

남은 제출 확인은 프로젝트 실행 안내의 처음부터 재현 여부, 공개 산출물/데이터 범위, 딥러닝 완료 후 발표자료의 수치·표현 점검입니다. 새 독립 데이터 평가·현장 적용·실제 센서/맵 대응 평가는 별도 확장 범위입니다. 프로젝트 전체 일정은 [PROJECT_SCHEDULE](../PROJECT_SCHEDULE.md)을 따릅니다. 완료된 작업을 새 체크리스트로 반복하지 않습니다.
