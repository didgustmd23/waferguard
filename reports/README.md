# 실험 결과물

공개 집계 결과·그림·분석 노트북을 안내합니다. 상세 문서는 [프로젝트 문서](../docs/README.md)에 있습니다. 데이터·전체 로그·모델 객체는 로컬 산출물로 보존합니다.

## SECOM 공개 결과 요약

GitHub에서 확인할 수 있도록 기존 로컬 결과의 집계값만 보존했습니다. 새 학습·평가는 실행하지 않았습니다. 원본 데이터, 행별 예측, 모델 객체, 개인 절대 경로와 전체 실행 JSON은 포함하지 않습니다.

### 결과 파일과 평가 범위

| 파일 | 내용 | 해석 범위 |
| --- | --- | --- |
| [센서 축소 비교](secom/results/v2_sensor_comparison.csv) | 전체 센서와 반복 weighted gain Top-20의 AP·ROC-AUC | 센서를 각 학습 구간에서 재선택한 시간순 3구간 개발 비교 |
| [고정 센서 90%·80% 시나리오](secom/results/v2_fixed_scenarios.csv) | OOF 문턱, 이후 Recall·선별 비율·혼동행렬 | 고정 센서 20개 후보의 이후 개발 구간 272행·불량 9건 |
| [저장·복원 정합성](secom/results/v2_bundle_verification.csv) | 33행 확률·판정·라벨 일치 여부 | 구현 검증이며 탐지 성능 평가가 아님 |
| [기존 Test 전체 센서·고정 Top-20 비교](secom/results/v2_existing_test_comparison.csv) | 전체 센서 444개와 고정 Top-20의 OOF 목표 80%·90% Recall·선별 비율·혼동행렬·AP·ROC-AUC | 기존 Test 236행·불량 9건. 완전 미사용 독립 Test 아님 |

센서 축소 비교의 AP 상대 하락은 **1.96%**입니다. 전체 모델은 깊이 3, Top-20 모델은 깊이 2이므로 센서 수와 분류기 설정을 함께 변경한 후보 비교입니다. 동일 설정에서 센서 수만 줄인 효과로 해석하지 않습니다. 이 표의 Top-20은 아래 저장 모델의 고정 센서 평가와 다릅니다.

| 고정 모델 OOF 목표 | 이후 Recall | 이후 선별 비율 | TP / FP / FN / TN |
| --- | ---: | ---: | --- |
| 90% — 미검 우선 | 100.00% | 94.49% | 9 / 248 / 0 / 15 |
| 80% — 부담 비교 | 88.89% | 50.37% | 8 / 129 / 1 / 134 |

같은 고정 센서·학습 통계·분류기에 문턱만 다르게 적용했습니다. 두 시나리오의 AP는 0.323619, ROC-AUC는 0.795944입니다. 독립 최종 Test 결과나 미래 검출률 보장이 아닙니다. 저장·복원 검증의 `test_used=false`는 해당 구현 검증에서 Test를 사용하지 않았다는 뜻이며 프로젝트 전체의 데이터 사용 이력을 보증하지 않습니다.

### 기존 Test 비교 — 개발 구간 CSV와 별도

아래는 저장된 V2를 V1과 같은 Test 236행·불량 9건에 적용한 결과입니다. 위 공개 `v2_fixed_scenarios.csv`는 개발 구간 272행의 기존 요약이며 이번 Test 결과로 덮어쓰지 않습니다.

| 후보 | Recall | 선별 비율 | TP / FP / FN / TN | AP | ROC-AUC |
| --- | ---: | ---: | --- | ---: | ---: |
| V1 | 11.11% | 13.14% | 1 / 30 / 8 / 197 | 0.040461 | 0.488008 |
| V2 — OOF 목표 90% | 100.00% | 94.92% | 9 / 215 / 0 / 12 | 0.075077 | 0.682330 |
| V2 — OOF 목표 80% | 66.67% | 46.19% | 6 / 103 / 3 / 124 | 0.075077 | 0.682330 |

V2 원본은 `logs/secom/v2/v2_recall90_existing_test/`, `logs/secom/v2/v2_recall80_existing_test/`, V1 원본은 `logs/secom/v1/m3_frozen_time_evaluation/`의 평가 JSON·CSV입니다. 재학습·문턱 변경 없이 평가했으나 과거 V1 결과를 참고한 개발 이력이 있어 완전 미사용 독립 Test는 아닙니다. [V2 Model Card 5.1절](../docs/secom/model_card_v2.md#51-기존-v1-test-구간의-후속-평가)을 참고하세요.

### CSV 필드 읽는 방법

- 비율은 0~1입니다. `0.9`는 90%이며 `target_recall`은 목표, `recall`은 관측한 값입니다.
- `v2_fixed_scenarios.csv`에서 `oof_*`는 문턱을 선택한 과거 OOF 결과이고, 접두사가 없는 성능 지표는 이후 개발 구간 272행 결과입니다.
- `v2_existing_test_comparison.csv`의 성능 지표는 기존 Test 236행·불량 9건 결과입니다. `recall_target_met`은 이 Test에서 목표 Recall을 달성했는지를 뜻합니다.
- `alarm_ratio`(`v2_fixed_scenarios.csv`)와 `reinspection_ratio`(`v2_existing_test_comparison.csv`)는 모두 `(TP + FP) / support`인 위험 대상 선별 비율입니다. 실제 불량 비율이나 검사 수행량이 아닙니다.
- TP는 검출한 불량, FP는 선별한 정상, FN은 놓친 불량, TN은 선별하지 않은 정상입니다.
- `ap_mean`·`ap_std`는 3구간 AP의 평균·표준편차입니다. `ap_loss_ratio = 1 - candidate_ap_mean / baseline_ap_mean`입니다.
- 정밀한 문턱과 집계 수치는 원본 값을 유지했습니다. CSV에는 수식이 없으며 기존 실험의 정적 요약입니다.

### 로컬 원본 출처와 갱신

원본은 Git에서 제외된 로컬 산출물입니다. 아래 경로는 출처 기록이며 GitHub 다운로드 링크가 아닙니다.

| 공개 파일 | 로컬 원본 |
| --- | --- |
| `v2_sensor_comparison.csv` | `logs/secom/v2/v2_p0_stable_gain_top20_depth2/summary.csv`, `reduction_assessment.csv`, 같은 실험의 설정 기록 |
| `v2_fixed_scenarios.csv` | `logs/v2_fixed_sensor20_oof/scenario_results.csv` |
| `v2_bundle_verification.csv` | `models/candidates/v2_fixed20_recall90/verification.json`, `models/candidates/v2_fixed20_recall80/verification.json` |
| `v2_existing_test_comparison.csv` | `logs/secom/v2/v2_all_vs_fixed20_existing_test/comparison.csv` |

실험을 다시 실행해도 이 공개 요약은 자동 갱신되지 않습니다. 새 결과의 설정·평가 범위를 확인한 뒤 필요한 집계값만 갱신합니다. `.gitignore`는 `reports/secom/results/`의 위 CSV 네 파일만 예외로 허용합니다.

상세 해석은 [V2 Model Card](../docs/secom/model_card_v2.md), [실험 요약](../docs/secom/SECOM_ML_SUMMARY.md), [상세 보고서](../docs/secom/report.md), 전체 로컬 이력은 [로그 목록](../docs/secom/LOG_INDEX.md)을 참고하세요.

## SECOM 노트북과 그림

### 분석 노트북

[analysis.ipynb](secom/analysis.ipynb)는 기존 로그를 읽어 표와 그림을 확인하는 분석 노트북이다. 프로젝트 루트를 상위 폴더에서 찾으므로 이 폴더에서도 실행할 수 있다. 노트북의 과거 셀 출력은 당시 기록이며, 현재 설정·로그를 확인하려면 해당 셀을 다시 실행한다. 이번 정리에서는 셀 출력을 재생성하지 않았다.

### 그림 분류

| 위치 | 내용 | 구분 |
| --- | --- | --- |
| `secom/figures/eda/step3/` | 클래스 비율·결측률·시간 분포 | 데이터·split 점검 |
| `secom/figures/exploratory/` | Day 3 PCA·중요도·센서 수/AP, 분석 미리보기 | 초기 탐색 기록 |
| `secom/figures/exploratory/team_proposal/` | 팀원 브랜치의 원본 그림 6개 | 제안 당시 이력, 보완 코드·최종 V2 평가 결과 아님 |
| `secom/figures/v1/m3_policy80_40_diagnostics/` | OOF와 시간 검증 확률 분포 | V1 문턱 진단 |
| `secom/figures/v1/m3_selected_sensor_drift/` | 선택 센서 drift | V1 분포 진단 |
| `secom/figures/v1/m3_sensor_detail/` | 센서 상세·월별 변화·drift | V1 상세 진단 |
| `secom/figures/v2/v2_sensor_importance/` | 전체 센서 gain·상위 20개 | V2 중요도 진단 |
| `secom/figures/v2/v2_sensor_importance_run2/` | 별도 실행의 전체 gain·상위 20개 | 실행 이력 보존 |
| `secom/figures/v2/v2_sensor_class_distribution/` | 정상·불량 및 기간별 분포 | V2 분포 진단 |

#### 주요 그림 바로 보기

- [데이터 클래스 분포](secom/figures/eda/step3/class_distribution.png)
- [PCA 누적 설명분산](secom/figures/exploratory/day3_pca_cumulative_variance.png)
- [초기 중요도](secom/figures/exploratory/day3_feature_importance.png)
- [센서 수와 AP](secom/figures/exploratory/day3_sensor_count_vs_ap.png)
- [V1 확률 분포](secom/figures/v1/m3_policy80_40_diagnostics/score_distribution.png)
- [V1 센서 상세](secom/figures/v1/m3_sensor_detail/sensor_detail.png)
- [V2 전체 센서 gain](secom/figures/v2/v2_sensor_importance/feature_importance_all.png)
- [V2 gain 상위 20개](secom/figures/v2/v2_sensor_importance/feature_importance_top20.png)
- [V2 정상·불량 분포](secom/figures/v2/v2_sensor_class_distribution/class_distribution.png)
- [V2 기간별 분포](secom/figures/v2/v2_sensor_class_distribution/class_distribution_by_period.png)

중요도 상위 20개 진단 그림은 저장 모델의 고정 센서 목록과 같다고 가정하지 않는다. 저장 모델 목록은 [V2 Model Card](../docs/secom/model_card_v2.md)를 따른다. 탐색·V1·V2 그림을 같은 모델의 최종 성능 근거로 혼용하지 않는다.

### 최종 그림·발표 점검 — 2026-10-09

저장 V2의 기존 Test 예측 로그로 그림을 생성했습니다. 평가 기록의 지표와 236행의 정답·판정 및 두 시나리오의 공통 확률을 검증했으며 재학습·Test 재예측·문턱 변경은 하지 않았습니다.

- [90%·80% 시나리오 혼동행렬](secom/figures/v2/existing_test/confusion_matrices.png)
- [공통 PR 곡선 및 두 문턱의 위치](secom/figures/v2/existing_test/precision_recall_curve.png)

PR 곡선은 같은 모델의 곡선 하나이며 전체 범위와 정밀도 0~20% 확대를 함께 표시합니다. AP는 0.075077이고 기본 불량 비율은 9/236=3.81%입니다. 발표 자료는 딥러닝 완료 후 작성합니다.

그림 재생성 명령 (기존 그림 보존을 위해 새 출력 폴더 지정):

```powershell
python -m src.sensor_ml.diagnostics.plot_v2_test `
  --recall90-dir logs/secom/v2/v2_recall90_existing_test `
  --recall80-dir logs/secom/v2/v2_recall80_existing_test `
  --figure-dir reports/secom/figures/v2/existing_test_run2
```

발표에서는 평가 범위(구간별 재선택 824행 / 고정 모델 개발 272행 / 기존 Test 236행)를 반드시 구분합니다. 100% Recall만 강조하지 않고 선별 94.92%와 불량 9건의 소표본 한계를 함께 표기합니다. 개발 Recall 88.89%를 Test 성능으로 표시하거나, 팀원 초기 그림을 저장 V2 성능으로 사용하지 않습니다.

실제 이미지 점검에서 `team_proposal/day3_feature_importance.png`의 x축이 `|Logistic Regression Coefficient|`로 남아 있는 것을 확인했습니다. LightGBM gain 제안의 중요도 그림으로 설명할 때 축 의미가 맞지 않으므로, 원본 이력으로만 보존하고 발표에 그대로 쓰지 않습니다. `v2_sensor_importance/feature_importance_top20.png`도 전체 센서 진단 모델의 중요도이며 저장된 고정 센서 20개 목록을 뜻하지 않습니다.

### 전체 센서·고정 Top-20의 동일 Test 비교

같은 학습 824행·XGBoost 깊이 2·규제 1·300 trees·클래스 비율 가중치로 준비한 전체 센서 기준과 저장 고정 Top-20을 비교합니다. 문턱은 각 모델의 동일 Train 내부 OOF 구간에서 따로 선택했습니다. Test는 동일한 236행·불량 9행입니다.

- [OOF 목표 80%의 혼동행렬 비교](secom/figures/v2/all_vs_fixed20_existing_test/confusion_recall80.png)
- [OOF 목표 90%의 혼동행렬 비교](secom/figures/v2/all_vs_fixed20_existing_test/confusion_recall90.png)
- [전체 센서·Top-20 PR 곡선 비교](secom/figures/v2/all_vs_fixed20_existing_test/precision_recall_comparison.png)
- [공개용 비교 수치](secom/results/v2_existing_test_comparison.csv)

전체 센서 AP는 0.065251, Top-20 AP는 0.075077입니다. Top-20은 검출도 많고 선별 비율도 높습니다. 같은 OOF 목표를 적용했지만 실제 Test Recall은 달라 동일 Recall에서의 선별 부담 비교로 주장하지 않습니다. 기존 Test 재사용 비교이며 그림 생성은 저장 예측 로그만 재사용했습니다. 근거 로그는 `logs/secom/v2/v2_all_vs_fixed20_existing_test/`입니다.

재생성할 때는 다음 명령의 그림 폴더를 새 이름으로 지정합니다.

```powershell
python -m src.sensor_ml.diagnostics.plot_sensor_test_compare `
  --all-result-dir logs/secom/v2/v2_all_vs_fixed20_existing_test `
  --top20-recall80-dir logs/secom/v2/v2_recall80_existing_test `
  --top20-recall90-dir logs/secom/v2/v2_recall90_existing_test `
  --figure-dir reports/secom/figures/v2/all_vs_fixed20_existing_test_run2
```

### 결과 생성 규칙

그림 생성 명령은 프로젝트 루트에서 실행하고 `--figure-dir` 또는 `--figures-dir`에 위 분류의 새 결과 폴더를 지정한다. 기존 실행 그림은 덮어쓰지 않는다. 실행 설정과 CSV 근거는 원래 로그 폴더에 유지한다.

분할 점검의 기본 경로는 `reports/secom/figures/eda/step3/`, V1 문턱 진단은 `reports/secom/figures/v1/m3_policy80_40_diagnostics/`, 초기 분석 스크립트는 `reports/secom/figures/exploratory/`다. 실제 이동된 파일만 새 위치에 있으며 저장된 과거 실행 JSON·노트북 출력의 옛 경로는 당시 기록일 수 있다.
