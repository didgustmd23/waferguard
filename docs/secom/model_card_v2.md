# V2 Model Card — 고정 센서 20개 위험 선별 후보

작성일: 2026-10-09. 상태: **연구용 후보 저장·복원·추론 및 기존 Test 비교 완료**. 완전 미사용 독립 평가 결과는 아니다.

## 1. 목적과 산출물

SECOM 공정 센서로 후속 확인이 필요한 위험 대상을 선별하는 이진 분류 프로토타입이다. 양성 label은 `1`, 음성 label은 `-1`이다. 모델의 양성 판정은 실제 불량 확정이 아니며 음성 판정도 출하 승인이 아니다.

| 시나리오 | 저장 폴더 | 정확한 문턱 |
| --- | --- | ---: |
| 과거 OOF Recall 목표 90% — 미검 우선 | `models/candidates/v2_fixed20_recall90` | 0.0029468848183751106 |
| 과거 OOF Recall 목표 80% — 부담 비교 | `models/candidates/v2_fixed20_recall80` | 0.0339033380150795 |

두 묶음은 센서·median·분류기를 공유하며 문턱만 다르다. 90% 저장은 기존 검증과 같은 범위로 재학습하고 기존 개발 구간 확률과 정합성을 확인했다. 80%는 재학습 없이 90% 객체를 독립 복사해 문턱만 변경했다. 각 묶음의 `is_final_model=false`와 `deployment_approved=false`를 유지한다.

각 폴더는 `candidate.joblib`, `manifest.json`, `verification.json`, `experiment/`를 포함한다. manifest에는 저장 환경·센서 순서·문턱·실험 출처가 있고 experiment에는 원본 OOF 기록 사본이 있다. 원래 실험 기록의 `model_saved=false`는 당시 상태를 보존한 것이며 이후 저장 완료 상태와 구분한다.

## 2. 데이터와 학습·문턱 선택 범위

- 데이터 출처·다운로드 안내: [README](../../README.md). 원본 SECOM은 1,567행·익명 센서 590개다.
- 개발 입력: `data/splits/integrated/time_train.csv`, 1,096행·불량 78행.
- 센서 선택: 과거 551행에서 반복 weighted gain으로 20개를 한 번 선정했다. 이후 평가 구간의 정답은 이 선택에 사용하지 않았다.
- 최종 저장 분류기 학습: 과거 824행·불량 69행. 마지막 개발 구간 272행은 학습에 추가하지 않았다.
- 문턱 선택: 센서 선택 이후의 과거 273행·불량 8행을 시간순 2구간 OOF로 예측했다. 각 OOF 모델은 해당 평가 행보다 과거인 데이터로만 학습한다. 초기 센서 선택 551행은 OOF 평가에서 제외한다.
- 선택 규칙: 목표 Recall을 충족하는 저장된 문턱 후보 중 선별 비율 최소 → Precision 최대 → 문턱 최대. 선별 상한은 적용하지 않았다.
- 이후 개발 평가: 마지막 272행·불량 9행. 개발 중 관찰한 결과이며 독립 미사용 Test가 아니다.

OOF 불량 8건에서 한 건은 Recall 12.5%p다. 따라서 90% 목표 문턱은 OOF 불량 8건 모두를 선별해야 했고, 80% 목표는 7건을 선별했다. 목표값은 미래 Recall 보장이 아니다. 기존 데이터 사용 이력 때문에 새 분할만으로 독립성을 주장하지 않는다. 이번 저장·복원·데모에는 Test를 사용하지 않았다.

## 3. 전처리와 분류기

센서 선택 실험은 학습 범위 내부에서 결측 비율 50% 초과·상수 센서를 제거하고 중앙값 대치 후 반복 weighted gain을 계산했다. 선택기는 XGBoost 깊이 3·L2 규제 1·클래스 비율 가중치, 반복 횟수는 5다.

저장된 고정 입력 경로는 **고정 센서 20개 → 학습 median 대치 → XGBoost**다. 추론 시 품질 필터·센서 선택·median을 다시 fit하지 않고 다른 센서를 요구하지 않는다. scaling과 PCA는 이 후보에 사용하지 않는다.

최종 분류기는 트리 300개·깊이 2·L2 규제 1이며 `class_weight_mode="ratio"`로 각 학습 범위의 정상/불량 비율을 사용한다. 나머지 설정과 학습 출처는 묶음의 manifest에 보존된 실행 설정을 따른다. 현재 프로젝트 설정을 변경해도 기존 저장 묶음의 모델은 자동 변경되지 않는다.

## 4. 20개 센서 입력 계약

manifest 기준 학습 순서는 다음과 같다.

```text
sensor_7, sensor_12, sensor_21, sensor_59, sensor_65,
sensor_121, sensor_126, sensor_147, sensor_247, sensor_267,
sensor_298, sensor_351, sensor_367, sensor_446, sensor_456,
sensor_457, sensor_477, sensor_519, sensor_562, sensor_568
```

- UTF-8 CSV에 위 센서 컬럼만 제공한다. 입력 열 순서는 달라도 센서명으로 학습 순서에 맞춘다.
- label·timestamp·행 ID·인덱스·추가 센서는 제외한다. 필수 컬럼 누락·중복은 오류다.
- 값은 숫자여야 한다. 존재하는 컬럼의 빈 값·NaN은 저장된 학습 median으로 대치한다.
- 모든 필수 센서가 결측인 행, 무한대, 잘못된 타입은 거부한다. 입력 오류 시 일부 행을 조용히 제외하지 않는다.
- 센서 의미·단위·수집 조건이 학습 데이터와 같아야 한다. 익명 센서를 타 공장의 동명 센서로 해석하지 않는다.

## 5. 이후 개발 구간 성능

정밀한 문턱과 집계값은 [공개 시나리오 CSV](../../reports/secom/results/v2_fixed_scenarios.csv)에 보존했습니다. [공개 결과 안내](../../reports/README.md#secom-공개-결과-요약)에서 다른 평가 범위와 구분해 확인할 수 있습니다.

평가 표본은 272행(정상 263·불량 9)이다. 아래 수치는 저장된 `development_scenario_result`에 기록된 기존 개발 평가이며 새 평가를 실행한 것이 아니다.

| OOF 목표 | OOF Recall | OOF 선별 비율 | 이후 Recall | 이후 Precision | 이후 선별 비율 | TP | FP | FN | TN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 90% | 100.00% | 95.97% | 100.00% | 3.50% | 94.49% | 9 | 248 | 0 | 15 |
| 80% | 87.50% | 54.95% | 88.89% | 5.84% | 50.37% | 8 | 129 | 1 | 134 |

두 시나리오의 AP는 0.323619, ROC-AUC는 0.795944다. 같은 확률 점수에 문턱만 다르게 적용하므로 순위 지표는 같다. 90%는 불량 1건을 더 검출하는 대신 정상 119건을 추가 선별했다. 이는 실제 비용·인력·장비 용량을 측정한 운영 판단이 아니다.

이 표는 한 개발 구간의 결과다. 이후 기존 Test 평가도 완료했으며 아래 5.1절에 별도로 기록한다. 불량 9건으로 얻은 100% Recall을 미래 데이터의 안정적인 100% 검출로 주장하지 않는다. 센서를 매 구간 재선택한 3구간 실험의 평균 AP 0.171439와 합산 Recall 94.87%는 **다른 평가 범위**이며 이 저장 모델의 지표로 표시하지 않는다.

### 5.1 기존 V1 Test 구간의 후속 평가

평가 파일은 `data/splits/integrated/time_test.csv`, 236행·정상 227행·불량 9행이다. V1 평가의 원본 ID·label과 대조한 같은 구간에 저장 모델을 적용했다. 재학습·센서 재선택·문턱 변경은 없으며, OOF 목표는 Test의 검출률 보장이 아니다.

| 후보 | Recall | Precision | F1 | 선별 비율 | TP | FP | FN | TN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V1 비교 기준 | 11.11% | 3.23% | 0.050000 | 13.14% | 1 | 30 | 8 | 197 |
| V2 — OOF 목표 90% | 100.00% | 4.02% | 0.077253 | 94.92% | 9 | 215 | 0 | 12 |
| V2 — OOF 목표 80% | 66.67% | 5.50% | 0.101695 | 46.19% | 6 | 103 | 3 | 124 |

V2의 AP는 0.075077, ROC-AUC는 0.682330이다. 두 시나리오의 확률 점수가 같으므로 두 지표도 같다.

V1의 AP 0.040461·ROC-AUC 0.488008보다 개선됐지만, 90% 시나리오는 236행 중 224행을 선별한다.

80% 시나리오는 목표에 미달했다. 개발 구간의 88.89%를 이 Test의 수치로 사용하지 않는다. V1·V2는 센서·모델·문턱이 함께 달라져 개선을 한 요소만의 효과로 해석하지 않는다. V1 Test 결과를 참고한 개발 이력이 있으므로 완전 미사용 독립 평가가 아니다. 이는 Test 정답을 학습했다는 뜻은 아니다.

근거는 `logs/secom/v2/v2_recall90_existing_test/`, `logs/secom/v2/v2_recall80_existing_test/`의 `evaluation.json`, `test_result.csv`, `test_predictions.csv`다. 각 묶음의 `evaluation_started.json`에도 완료 기록을 남겼다. 기존 manifest의 `test_used=false`는 묶음 생성 당시 상태이며 이번 후속 평가를 부정하는 뜻이 아니다. 기록 목적의 평가 재실행·manifest 변경은 하지 않았다.

### 5.2 동일 조건의 전체 센서 기준 비교

동일 학습 824행·XGBoost 깊이 2·규제 1·300 trees·클래스 비율 가중치의 전체 센서 모델을 별도 준비했다. 품질 필터 후 444개 센서를 사용하며, 같은 Train 내부 OOF 구간에서 전체 센서용 문턱을 선택했다. 기존 Top-20 묶음은 변경하지 않았다.

같은 Test 236행에서 전체 센서의 AP는 0.065251, ROC-AUC는 0.681351이다. 80% 목표의 Test Recall은 33.33%·선별 25.00%, 90% 목표는 Recall 44.44%·선별 31.36%다. Top-20은 AP와 검출 건수가 높지만 선별 부담도 높았다. 같은 OOF 목표 조건이지 같은 실제 Test Recall 또는 선별 예산 조건은 아니다. 소표본·기존 Test 재사용 한계를 유지하며 문턱을 재조정하지 않는다.

[비교 그림·수치](../../reports/README.md#전체-센서고정-top-20의-동일-test-비교)와 [상세 보고서 37절](report.md#37-전체-센서고정-top-20의-동일-기존-test-비교--2026-10-09)을 참고한다.

## 6. 구현 정합성과 추론 데모

두 묶음의 검증 집계값은 [공개 저장·복원 검증 CSV](../../reports/secom/results/v2_bundle_verification.csv)를 참고합니다.

각 저장 묶음의 검증 입력 33행에서 별도 프로세스 복원 결과는 `passed`였다. 최대 확률 차이 0.0, 확률·판정·라벨이 저장 전 기대값과 일치했다. 저장 과정에서는 기존 Pipeline과 20개 입력 추론 경로도 비교했다. 이는 구현 정합성 검증이지 탐지 성능 평가가 아니다.

**로컬 복원 검증 완료, 다른 PC·OS 재현은 검증 범위 제외.** 2026-10-26에 저장 환경과 버전이 정확히 같은 로컬 conda 환경 `ml310`에서 커밋 `9961eb6`의 소스로 두 묶음을 다시 복원했다. 두 실행 모두 종료 코드 0·`passed`이며 결과값은 위와 같았다. 복원 객체의 센서 이름·순서, 양성 label `1`, median 20개, 분류기 주요 파라미터, XGBoost booster 원본 바이트가 두 묶음에서 같았고 문턱만 달랐다. 실행 전후 묶음 파일의 크기·SHA256은 변하지 않았다. 객체 비교는 센서·median·주요 파라미터·booster 바이트·문턱으로 한정했으며 모든 내부 상태를 비교한 것은 아니다. Test·`predict_cli`·데모는 이 확인에서 다시 실행하지 않았고 처음부터의 재학습 재현도 범위에서 제외했다.

Train 유래 공통 데모 `data/processed/v2_sensor20_demo.csv`의 33행에서 90%는 33행, 80%는 30행을 선별했다. 출력은 각각 `logs/secom/demos/v2/v2_recall90_demo_predictions.csv`, `logs/secom/demos/v2/v2_recall80_demo_predictions.csv`에 저장했다. 데모에는 정답 평가를 적용하지 않으며 추가 선별 3건의 정상·불량 여부를 판단하지 않는다.

실행 명령은 [V2 저장·복원 절차](USAGE.md#v2-시나리오-저장복원)를 따른다. 저장 상태·검증 결과는 이 카드의 계약과 근거를 기준으로 한다.

## 7. 보안·재현성과 사용 제한

- joblib은 임의 코드 실행 위험이 있다. 직접 생성한 신뢰한 로컬 파일만 `--trusted-local-bundle`로 읽는다.
- 고정 환경: Python 3.10.22, NumPy 1.26.4, pandas 2.2.3, scikit-learn 1.5.2, XGBoost 2.1.3, LightGBM 4.5.0, joblib 1.6.0. 복원기는 7개 버전 중 하나라도 다르면 거부한다. 프로젝트 공통 `requirements.txt`는 Python 3.12·범위 지정 환경이므로 이 묶음의 복원 환경이 아니다. 검증된 환경은 로컬 `ml310` 하나다.
- 묶음에는 로컬 검증용 실제 센서 값과 로컬 경로가 포함된다. 공개 저장소·외부 서비스에 업로드하지 않는다. Git 저장소 복제만으로 모델 객체가 제공되지 않는다.
- 높은 선별 비율은 후속 확인 부담이다. 검사 비용 절감·현장 배포·자동 생산 제어·출하 승인에 사용하지 않는다. `positive_score`의 실제 불량 확률 보정도 입증하지 않았다.
- SECOM과 WM-811K는 대응되는 동일 웨이퍼 데이터가 없다. 별도의 맵 패턴 진단을 가상 연결할 수 있지만 통합 Recall·정상 복구·불량 유출 감소 성능을 주장하지 않는다. WM `None` 패턴으로 센서 경고를 자동 해제하지 않는다.
- 현장 적용에는 센서 매핑·측정 시점·단위 확인, 새 독립 데이터 검증, 비용·처리 용량 합의, drift 감시와 재학습 기준이 필요하다.

## 8. 근거와 보존 이력

- 센서 선정·개발 대조: `logs/v2_fixed_sensor20_later_selection/`.
- 선택 이후 OOF·문턱·개발 평가: `logs/v2_fixed_sensor20_oof/` 및 각 묶음 `experiment/`.
- 모델 계약·개발 성능·환경: 각 묶음 `manifest.json`.
- 저장·복원 정합성: 각 묶음 `verification.json`과 사용자 별도 프로세스 실행 출력.
- 기존 V1 Test 비교: `logs/secom/v2/v2_recall90_existing_test/`, `logs/secom/v2/v2_recall80_existing_test/`. 완료 상태·재학습 없음·문턱·집계값 확인.
- 종합 결론: [ML 요약](SECOM_ML_SUMMARY.md), [보고서 35.4절](report.md#354-후속-고정-센서-후보와-저장추론-데모).

### 8.1 인계 기준

소스 기준은 커밋 `9961eb6c267ec9d8dd4aa152ccca354ed4b05aa9`(`origin/main` 포함) 전체다. 개별 파일만 골라 전달하지 않는다. 복원 클래스 경로 `src.sensor_inference`·`src.modeling_models`는 이동하지 않는다. 모델 묶음은 Git 밖에서 비공개로 전달하고, 받는 쪽은 신뢰 옵션을 쓰기 전에 아래 SHA256으로 변경 여부를 확인한다. SHA256은 변경 확인용이며 안전한 역직렬화를 보증하지 않는다.

| 파일 | 크기 (bytes) | SHA256 |
| --- | ---: | --- |
| `v2_fixed20_recall90/candidate.joblib` | 278,908 | `3870ea77ad5e2fc2710d225bf1c4a62980a867bd54b0633c2731d3f09f132d3e` |
| `v2_fixed20_recall90/manifest.json` | 10,285 | `f356327c5e9aed4b7f1ae4b6ba2e66d059fbe598343916e9e5ac7d661f9bfabe` |
| `v2_fixed20_recall90/verification.json` | 254 | `036989c2a16b3420096ba81372ffb6da41ebfeb5d4bbdf5e7107a891b22393fe` |
| `v2_fixed20_recall90/evaluation_started.json` | 1,830 | `1f137bf860f7b4643b162e6d9795c404ff9bbeee1517278292d0fa2603446348` |
| `v2_fixed20_recall80/candidate.joblib` | 279,020 | `699be6ca67f904d773e60642ea37831dd302638212f6f780c885c20adcddf580` |
| `v2_fixed20_recall80/manifest.json` | 9,810 | `b9c58736dd8b2b7fc05f3a2aa1c70a62c967b3be47350c6114377c361f9bb28e` |
| `v2_fixed20_recall80/verification.json` | 254 | `036989c2a16b3420096ba81372ffb6da41ebfeb5d4bbdf5e7107a891b22393fe` |
| `v2_fixed20_recall80/evaluation_started.json` | 1,839 | `79dad6fd39d343c846c445bbba3d5ec3e882e06675829f3c6acf3c1c446001d3` |

경로는 모두 `models/candidates/` 기준이다. 목적별 전달 범위는 다음과 같다.

- 모델 사용: 위 커밋, 두 묶음 폴더 전체, 고정 환경.
- 결과 설명·감사: 위에 더해 이 절의 근거 로그(`logs/v2_fixed_sensor20_*`, `logs/secom/v2/v2_recall*_existing_test/`, `logs/secom/v2/v2_all_vs_fixed20_existing_test/`, `logs/secom/v1/m3_frozen_time_evaluation/`)와 `data/splits/integrated/manifest.json`.
- 처음부터 재현: 위에 더해 원본 `data/raw/secom.data`·`secom_labels.data`와 `data/splits/integrated/` 전체. [재현 절차](USAGE.md#v2-시나리오-저장복원)의 옛 경로 제약을 따른다.

manifest·`execution.json`의 절대 경로는 저장 당시 저장소 위치 `D:\workspace\secom-waferguard`를 기록한 이력이며 현재 존재하지 않는다. 복원·예측은 이 경로를 사용하지 않는다. 기록 보존을 위해 경로를 수정하지 않는다.

V1 S0·M3 후보는 retired 이력이다. [V1 Model Card](model_card.md)의 성능·문턱·검증 결과를 V2에 승계하지 않는다. 이 문서는 새 학습·Test 평가·배포 승인을 수행하지 않는다.
