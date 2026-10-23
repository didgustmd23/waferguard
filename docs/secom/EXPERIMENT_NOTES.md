# SECOM 통합 실험 노트

Top-20 개선 계획·V2 시간순 개선·고정 센서 검증 노트를 통합한 실행 조건과 판단 이력이다. **실험 완료 기록이며 새 실험 실행 지시가 아니다.** 수치표는 [실험 보고서](report.md)와 [ML 결과 요약](SECOM_ML_SUMMARY.md)에 보존하고 여기서는 설정·범위·근거 경로를 정리한다.

현재 산출물은 고정 센서 20개·XGBoost 깊이 2·규제 1의 90%/80% 시나리오 묶음이다. 모델 계약은 [V2 Model Card](model_card_v2.md), 저장·복원 명령은 [저장·복원 절차](USAGE.md#v2-시나리오-저장복원), 남은 제출 확인은 [실행 안내](USAGE.md#기존-v1-동결-평가와-남은-제출-확인)를 따른다. 문서마다 별도 완료 체크리스트를 반복하지 않는다.

## 1. 공통 비교 기준

- SECOM 센서 20개 이하 조건은 이 과제의 기준이며 범용 코어의 모든 데이터셋에 강제하지 않는다. PCA 성분 20개는 원본 센서 20개와 다르다.
- 품질 제거·대치·스케일링·센서 선택은 각 학습 범위 안에서 fit한다. 결측 비율 50% 초과·관측 값이 하나뿐인 센서를 제거하고 내역을 기록한다.
- 같은 센서 선택 실험에서는 분류기를 유지하고, 분류기 설정 실험에서는 선택기를 유지해 변경 효과를 구분한다.
- 센서 축소 AP 상대 하락률은 `(전체 AP - 축소 AP) / 전체 AP`다. 같은 구간·같은 비교 조건에서 목표 20% 이내, 최대 허용 30% 이내를 사용한다. 전체 AP가 0이거나 평가 불량이 없으면 판정하지 않는다.
- AP·Recall·FN·FP·선별 비율·구간별 저하를 함께 본다. AP 상대 하락은 미검률이 아니다.
- 이미 관찰한 이후 시간 구간은 개발 검증으로 표시한다. 재분할·재실행만으로 미사용 Test가 되지 않는다.
- 전체 Train의 센서 중요도·여러 구간의 선택 빈도를 과거 구간에 소급 적용하지 않는다.

## 2. V1 센서 선택기와 분류기 개선 이력

### S0~S3 RF 선택기

RF 트리 300개와 최종 XGBoost M0를 유지하고 선택기만 비교했다. S0를 기준으로 채택하고 S2를 비교 후보로 보존했다. S3는 시간순 평균 AP 개선을 얻지 못했다. 수치·학습 비용·센서 안정성은 보고서 28~30절을 따른다.

| 선택기 | 최소 leaf | 최대 깊이 | 계층 bootstrap | 조건 |
| --- | ---: | --- | ---: | --- |
| S0 | 1 | 제한 없음 | 0 | 단일 RF 중요도 Top-20 |
| S1 | 2 | 제한 없음 | 0 | leaf 제한 |
| S2 | 2 | 8 | 0 | 깊이·leaf 제한 |
| S3 | 1 | 제한 없음 | 10 | 선택 빈도·평균 순위로 Top-20 |

S0~S3는 문서상의 비교 번호이지 개별 JSON 프리셋 이름이 아니다. `--rf-min-samples-leaf`, `--rf-max-depth`, `--rf-stability-repeats`와 결과 폴더로 구분했다. 깊이 제한 없음은 옵션 생략으로 지정하며 0은 유효한 깊이가 아니다. S3의 10회는 선택기 내부 반복이지 CV 반복 횟수가 아니다.

기록은 `logs/secom/v1/s0_rf_top20_cv/`, `logs/secom/v1/s0_rf_top20_time/`, S1·S3의 동일 이름 폴더, S2의 `logs/secom/v1/s2_rf_top20_cv/`와 `logs/secom/v1/s2_rf_top20_time_user/`에 있다. `feature_run.json`의 실제 설정을 기준으로 해석하고 과거 로그에 없던 필드를 사후 추가하지 않는다. bootstrap 내 빈도와 시간 구간 간 선택 빈도는 구분한다.

### M0~M5 XGBoost 설정

S0 선택기를 유지했다. 기본은 트리 300개·깊이 3·규제 1·학습률 0.05·행/열 샘플 비율 0.8·불량 가중치 1이다.

| 후보 | 기본 대비 변경 | 실행 상태 |
| --- | --- | --- |
| M0 | 변경 없음 | 비교 완료·보존 |
| M1 | 깊이 2 | 비교 완료 |
| M2 | 규제 5 | 비교 완료 |
| M3 | 깊이 2·규제 5 | V1 기준으로 채택했던 후보 |
| M4 | 열 샘플 비율 1.0 | 당시 제안, 이 실험군의 완료 후보 아님 |
| M5 | 학습 범위의 정상/불량 비율 가중치 | 당시 제안, 이후 V2에서 별도 가중치 비교 |

근거는 보고서 31절과 `logs/secom/v1/m1_s0_top20_cv/`~`logs/secom/v1/m3_s0_top20_cv/`, 각 `_time/` 폴더다. M4·M5를 M0~M3와 함께 실행했다고 표시하지 않는다. V2의 깊이 2·규제 1 모델은 V1 M3(규제 5)와 다르다.

### V1 문턱 전이와 시간 진단

M3·M0의 전체 Train 단일 OOF는 `logs/secom/v1/m3_m0_oof_compare/`에 보존했다. 당시 80%·선별 상한 40% 임시 정책으로 선택한 문턱은 이후 시간 검증에서 목표를 충족하지 못했다. 정책 변경·개발 검증은 보고서 32~33절, 저장 V1의 후속 구간 평가는 34절을 따른다. 과거 후보 문턱을 V2에 승계하지 않는다.

| 질문 | 근거 폴더 |
| --- | --- |
| 문턱 전이·확률 분포 | `logs/secom/v1/m3_policy80_40_time_validation/`, `logs/secom/v1/m3_policy80_40_diagnostics/` |
| 동일 M3 전체 센서와 Top-20 | `logs/secom/v1/m3_all_vs_top20_time/` |
| 선택 센서 분포·월별 진단 | `logs/secom/v1/m3_selected_sensor_drift/`, `logs/secom/v1/m3_sensor_detail/` |
| 전체 과거 vs 최근 30/60일 | `logs/secom/v1/m3_recent_window_compare/` |
| 제거 표본·불량과 구간별 AP | `logs/secom/v1/m3_recent_window_diagnostics/` |
| 센서 목록 변화 | `logs/secom/v1/m3_recent_sensor_overlap/` |
| 최근 학습 효과와 센서 재선택 효과 분리 | `logs/secom/v1/m3_fixed_sensor_control/` |

최근 30일은 평균 AP 개선을 얻지 못했고, 60일은 해당 구간에서 제거 행이 없어 전체 과거와 같았다. 센서 목록이 바뀐다는 사실이나 drift 진단값만으로 성능 하락의 원인을 확정하지 않는다. V1 묶음은 `models/retired/v1_m3_sensor20/` 이력이며 현재 후보가 아니다.

## 3. V2 전체 센서·가중치·결측 처리 비교

### 가중치 여섯 후보

`r`은 각 학습 범위의 정상 수/불량 수다. 외부 평가 label로 계산하지 않는다.

| 후보 | 깊이 | 규제 | 불량 가중치 |
| --- | ---: | ---: | --- |
| v2_m0_none | 3 | 1 | 1 |
| v2_m0_sqrt_ratio | 3 | 1 | √r |
| v2_m0_ratio | 3 | 1 | r |
| v2_m3_none | 2 | 5 | 1 |
| v2_m3_sqrt_ratio | 2 | 5 | √r |
| v2_m3_ratio | 2 | 5 | r |

공통은 트리 300개·학습률 0.05·행/열 샘플 비율 0.8이다. 프리셋은 `configs/experiments/time_weight_v2.json`, 결과는 `logs/secom/v2/v2_time_weight_compare/`다. 전체 센서 품질 필터 → median → XGBoost로 비교하며 이 단계에서는 Top-20·PCA·SMOTE를 적용하지 않는다.

같은 Time Train 내부 외부 시간순 3구간과 각 과거 내부 OOF 2구간을 사용했다. 여섯 후보는 총 54회 fit이며 `--n-jobs`는 모델 내부 병렬 수다. `summary.csv`의 `temporal_oof`는 F1 문턱 진단이며 정책 충족은 `policy_selection.csv`로 별도 확인한다.

### P0~P2와 다른 모델 비교

`v2_m0_ratio` 설정을 유지하고 P0 median, P1 median+결측 indicator, P2 XGBoost 자체 NaN 처리를 비교했다. P1·P2의 평균 AP 개선 근거가 없어 P0를 유지했다. indicator 개수를 원본 센서 수로 세지 않는다. P1은 학습에서 결측이 없던 센서의 indicator를 새로 만들지 않는다.

RF·L1, RF 불량 가중치·leaf 비교도 완료했다. 결과·기록 경로는 [ML 요약 3절](SECOM_ML_SUMMARY.md#3-주요-개선-실험-요약)에 한 번만 정리한다. median이 모든 데이터에 최선이라는 결론은 아니다.

## 4. V2 센서 축소·반복 gain·모델 설정

RF Top-20 → 단일 gain → 단일 weighted gain → Top-50 대조 → 반복 weighted gain → 깊이·규제 비교 순으로 진행했다. 반복 선택은 현재 학습 범위 안에서 클래스별 행 수를 유지한 bootstrap 5회를 수행하고 선택 빈도·평균 순위로 목록을 정한다.

선택기는 깊이 3·규제 1·비율 가중치를 유지했다. 최종 분류기의 깊이 2·규제 1을 연구용 기준으로 채택하고 규제 5를 부담 비교로 보존했다. 미검 최소화를 우선한 판단과 동일 목표 Recall에서의 비교는 [ML 요약 4~6절](SECOM_ML_SUMMARY.md#4-주후보-설정과-센서-선택)을 따른다.

이 단계의 3구간 실험은 **구간마다 센서 목록을 다시 선택**했다. 전체 센서 대비 평균 AP 상대 하락 1.96%와 합산 Recall을 뒤의 고정 센서 한 모델 성능으로 표시하지 않는다.

## 5. 고정 센서 선택 시점과 선택 이후 OOF

### 선택 시점 대조

`adaptive_topk`는 매 과거 학습 구간에서 재선택하고 `fixed_topk`는 초기 목록을 유지한다. median·분류기는 각 구간의 과거로 학습하며 센서 목록 고정과 모델 객체 동결을 구분한다.

| 실험 | 센서 선택 범위 | 평가 범위 | 근거 |
| --- | --- | --- | --- |
| 최초 선택 고정 | 최초 272행 | 이후 3개 개발 구간 | `logs/secom/v2/v2_fixed_sensor20_compare/` |
| 선택 시점 대조 | 과거 551행 | 마지막 개발 구간 272행만 | `logs/v2_fixed_sensor20_later_selection/` |

최초 고정의 평균 AP는 0.058049, 재선택은 0.171439였다. 551행 선택의 같은 마지막 구간에서는 고정 AP 0.323619, 재선택 AP 0.227587이었다. 3구간 평균과 단일 구간 AP를 직접 비교하지 않는다. 선택 시점과 표본 수가 함께 바뀌어 개선 원인을 하나로 확정할 수 없고 단일 구간 편차 0도 안정성 증거가 아니다.

최초 고정 실행은 WindowsPath JSON 직렬화 단계에서 오류가 발생한 이력이 있다. 생성된 결과와 완전한 실행 manifest 존재 여부를 구분하며 이후 코드 수정으로 과거 실행 기록이 자동 복구됐다고 주장하지 않는다.

### 선택 이후 시간순 OOF

- 고정 목록은 551행 선택 후보에서 가져온다.
- OOF 1: 과거 551행 학습 → 다음 138행·불량 5행 예측.
- OOF 2: 과거 689행 학습 → 다음 135행·불량 3행 예측.
- 합산 OOF 273행·불량 8행에서 90%/80% 목표별 문턱을 선택한다. 초기 센서 선택 행은 OOF 평가에서 제외한다.
- 과거 824행으로 분류기를 학습하고 마지막 개발 구간 272행·불량 9행에 적용한다.

목표 Recall을 충족하는 문턱 중 선별 비율 최소 → Precision 최대 → 문턱 최대 순으로 선택하고 선별 상한은 적용하지 않았다. 불량 8건에서 90% 목표는 8건 전부, 80% 목표는 7건 검출이 필요하다. 이후 목표 달성 보장은 아니다.

근거는 `logs/v2_fixed_sensor20_oof/`의 `scenario_results.csv`, `threshold_compare.csv`, `oof_predictions.csv`, `inner_folds.csv`, `future_predictions.csv`, `execution.json`이다. 정확한 문턱·이후 성능은 [V2 Model Card](model_card_v2.md#5-이후-개발-구간-성능)에 보존한다.

## 6. 재현용 주요 실행 명령

프로젝트 루트에서 실행한다. 아래는 **재학습을 포함**하며 자동으로 실행하지 않는다. 기존 폴더 대신 비어 있는 새 출력 폴더를 사용하고 당시 설정을 해당 실행 기록과 대조한다. 현재 설정으로 재실행한 결과를 과거 결과로 덮어쓰지 않는다.

```powershell
# 전체 센서·가중치 비교
python -m src.sensor_ml.experiments.time_weight_compare --train data/splits/integrated/time_train.csv --output-dir logs/reproduce_v2_time_weight --n-jobs 2

# 결측 처리 3방식 비교
python -m src.sensor_ml.experiments.missing_value_compare --train data/splits/integrated/time_train.csv --model v2_m0_ratio --modes median median_indicator native_nan --output-dir logs/reproduce_v2_missing --n-jobs 2

# 최초 272행 센서 고정 대조
python -m src.sensor_ml.experiments.fixed_sensor_compare --train data/splits/integrated/time_train.csv --n-splits 3 --repeats 5 --n-estimators 300 --n-jobs 2 --output-dir logs/reproduce_v2_fixed_initial

# 551행에서 선택하고 마지막 개발 구간만 비교
python -m src.sensor_ml.experiments.fixed_sensor_compare --train data/splits/integrated/time_train.csv --selection-fold 2 --evaluate-from-fold 3 --n-splits 3 --repeats 5 --n-estimators 300 --n-jobs 2 --output-dir logs/reproduce_v2_fixed_later

# 재현한 고정 후보로 선택 이후 OOF 생성
python -m src.sensor_ml.experiments.fixed_sensor_oof --train data/splits/integrated/time_train.csv --candidate-dir logs/reproduce_v2_fixed_later --n-jobs 2 --output-dir logs/reproduce_v2_fixed_oof
```

S0~S3·M0~M3의 상세 실행과 조건은 보고서 28~31절에 유지한다. 90%/80% 묶음 저장·복원·CSV 추론 명령은 [저장·복원 절차](USAGE.md#v2-시나리오-저장복원)에만 둔다.

## 7. 진단 그림과 결과 확인

전체 센서 gain 그림은 `reports/secom/figures/v2/v2_sensor_importance/`와 `_run2/`, 시간별 클래스·센서 분포는 `reports/secom/figures/v2/v2_sensor_class_distribution/`에 있다. `reports/secom/analysis.ipynb`에서 기존 로그와 그림을 읽는다. 노트북에서는 통합 문서를 가리키는 안내 문구만 수정하고 실험 계산 로직·저장된 셀 출력·원본 로그는 변경하지 않았다.

```powershell
python -m src.sensor_ml.diagnostics.sensor_importance --train data/splits/integrated/time_train.csv --model v2_m0_ratio --output-dir logs/reproduce_v2_importance --figure-dir reports/secom/figures/v2/reproduce_v2_importance --n-jobs 2
```

이 명령은 전체 Time Train으로 후보 하나를 새로 학습하는 진단이다. gain은 인과적 영향이나 검출률이 아니다. 제거 센서의 빈 중요도와 사용됐지만 gain이 0인 센서를 구분한다. 진단 순위로 센서를 다시 고른다면 학습 구간 내부 선택 실험을 다시 수행해야 한다.

## 8. 20개 추론 경로의 의존성과 해석

과거 전체 입력 Pipeline은 수백 개 센서에 fit한 필터·대치기·선택기를 포함해 20개만 전달할 수 없었다. 현재 저장 경로는 **고정 센서 입력 → 저장된 median → 20개로 학습한 분류기 → 시나리오 문턱**이다. 추론에서 센서 선택·대치·scaling을 다시 fit하지 않는다.

학습 순서와 중요도 순위는 다르다. 센서명·학습 입력 순서를 보존하고, 컬럼 누락과 값 NaN을 구분한다. 모델·전처리·목록·label·문턱을 묶음으로 관리한다. 입력 검증 규칙·버전·모듈 경로·joblib 신뢰 조건은 [V2 Model Card](model_card_v2.md#4-20개-센서-입력-계약)에 정리한다.

수치형 median의 센서별 독립성은 PCA·범주형 인코딩·센서 간 파생 연산으로 자동 확장되지 않는다. 선택기를 추론에서 제외해도 현재 모듈의 LightGBM import 등 라이브러리 의존성이 사라지는 것은 아니다. 저장 환경은 manifest를 따른다. Agent·외부 LLM API는 코어 학습·추론의 필수 의존성이 아니다.

저장 전후 확률·판정 일치는 구현 검증이고 일반화 성능이 아니다. 개발 구간, OOF, Train 유래 데모, 독립 Test의 수치를 서로 대체하지 않는다. WM-811K 확장 설계는 [별도 문서](../WM811K/PREPROCESSING_DESIGN.md)에 유지한다.

## 9. 팀원 LightGBM gain 후보 비교 — 2026-10-09

상세 수치는 [보고서 36절](report.md#36-팀원-lightgbm-gain-top-20-제안-비교--2026-10-09)에 기록한다. 기존 Logistic Regression 분석과 OOF 기본값은 보존하고 팀원 제안을 별도 후보로 추가했다. -1/1 label에 맞게 정상/불량 건수로 가중치를 계산하며, 전체 Train의 센서 JSON을 OOF에 자동 적용하지 않고 각 학습 fold 내부에서 품질 필터·대치·gain 선택을 수행한다.

| 단계 | 조건 | 주요 결과 | 근거 경로 |
| --- | --- | --- | --- |
| 첫 OOF 탐색 | Time Train 1,096행·불량 78행, 단일 무작위 계층 OOF, 최종 깊이 3 | AP 0.166773, ROC-AUC 0.693277 | `logs/secom/exploratory/team_lgbm_gain/` |
| 시간순 대조 | 외부 3구간·내부 시간순 OOF 2구간, 두 Top-20 최종 깊이 2·규제 1·비율 가중치 | 기존 AP 0.171439, 팀원 AP 0.123767 | `logs/secom/exploratory/team_gain_time_compare/` |
| 동일 Recall 목표 | 위 로그만 재사용, OOF 목표 이상에서 선별 비율 최소 후보 선택 | 기존 V2가 80%·90% 모두 검출 건수는 많고 정상 오탐은 적음 | `logs/secom/exploratory/team_gain_recall_compare/` |

80% 목표에서 기존 V2는 TP 35·FP 604·FN 4, 팀원은 TP 33·FP 644·FN 6이었다. 90% 목표에서는 기존 TP 37·FP 723·FN 2, 팀원 TP 36·FP 738·FN 3이었다. 모두 같은 시간순 평가 824행·불량 39행의 합산 결과다. 기존 V2 주후보와 저장 묶음은 변경하지 않고 팀원 방식을 비교 실험으로 보존한다.

랜덤 OOF의 정밀도 최대 후보와 시간순 재분석의 최소 선별 후보는 선택 기준이 다르다. 시간순 `summary.csv`의 F1 문턱 진단과 Recall 목표별 비교도 구분한다. 각 구간에서 센서를 재선택하므로 고정 센서 모델의 Test 결과와 동일하지 않다. 팀원 후보는 Test에서 평가하지 않았다.

설정은 `configs/experiments/team_gain_time_compare.json`에 관리한다. 선택 LightGBM은 gain·200 trees·학습 fold 비율 가중치, 기존 반복 gain은 5회이며 최종 분류기는 300 trees다. 정확한 실행 파라미터는 `candidate_settings.json`, 구간·설정은 `temporal_run.json`, 문턱은 각 원본 CSV를 따른다.

재현 시 기존 결과 폴더를 덮어쓰지 않는다. 아래 첫 두 명령은 학습을 포함하며 마지막 명령은 로그 재분석만 수행한다.

```powershell
python -m src.sensor_ml.experiments.step8_threshold_oof --train data/splits/integrated/time_train.csv --experiment xgboost_lightgbm_gain_top20 --oof-output logs/reproduce_team_gain_oof/oof_predictions.csv --threshold-output logs/reproduce_team_gain_oof/threshold_compare.csv --n-jobs 2

python -m src.sensor_ml.experiments.team_gain_time_compare --train data/splits/integrated/time_train.csv --output-dir logs/reproduce_team_gain_time --n-jobs 2

python -m src.sensor_ml.experiments.team_gain_recall_compare --source-dir logs/reproduce_team_gain_time --output-dir logs/reproduce_team_gain_recall
```

## 10. 저장 V2의 기존 Test 후속 비교 — 2026-10-09

### 팀원 브랜치 통합 기록

`origin/su/branch`의 `1ff1134`, `5129d0b` 변경을 `feature/modeling`에서 통합한다. 메인에는 직접 반영하지 않는다. 이동 전 `src/step*.py`를 되살리지 않고 현재 `src/sensor_ml/` 경로로 충돌을 정리했다.

- Step 5·8: 기존 분석과 보완된 별도 LightGBM gain 후보·Recall 비교를 유지한다. 전체 Train 센서 JSON 자동 적용, label 합계 가중치, 작은 문턱 제외는 채택하지 않는다.
- Step 10: 예측 인덱스를 실제 학습 클래스 label로 복원하는 팀원 수정을 반영한다.
- Step 12: Step 10이 저장한 `validation`, `AP`, `Recall`, `Precision`, `F1`, `ROC-AUC` 컬럼을 읽도록 수정한다.
- CLI: Train·Validation·threshold 명시 입력 계약과 기존 출력 옵션을 유지한다. 특히 Step 5 모델 비교의 `--output`을 삭제하지 않는다.
- 팀원 원본 센서 목록은 `configs/experiments/archive/team_selected_sensors_20.json`에 분석 이력으로 보존한다. 생성 데이터·가중치의 정확성을 보증하거나 추론 입력으로 승인한 목록은 아니다.
- 팀원 그래프 6개는 `reports/secom/figures/exploratory/team_proposal/`에 별도 보존한다. 기존 그림 두 개는 기존 커밋 내용을 유지하며 팀원 원본 그림을 보완 코드의 재실행 결과라고 표시하지 않는다.

### 기존 Test 평가 기록

V1과 동일한 Test 236행·불량 9행에서 저장 V2를 재학습 없이 평가했다. V1 Recall 11.11%(TP 1·FP 30·FN 8)에 대해 V2 90% 목표는 Recall 100%(TP 9·FP 215·FN 0), V2 80% 목표는 Recall 66.67%(TP 6·FP 103·FN 3)였다. V2 AP는 0.075077, ROC-AUC는 0.682330이다. 개발 구간의 88.89%와 구분하며 과거 결과를 참고한 기존 Test 비교이지 완전 미사용 독립 평가가 아니다.

상세 조건은 [보고서 35.5절](report.md#355-저장-v2의-기존-v1-test-구간-비교--2026-10-09), 원본은 `logs/secom/v2/v2_recall90_existing_test/`, `logs/secom/v2/v2_recall80_existing_test/`에 보존한다. 이미 완료한 평가를 문서 작성을 위해 다시 실행하지 않았다.
