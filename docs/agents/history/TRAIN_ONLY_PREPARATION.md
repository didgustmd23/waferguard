# Train-only 범용 표형 전처리

> 초기 구현·실행 검증 이력입니다. 현재 상태와 권장 진입점은 [에이전트 문서 안내](../GUIDE.md)를 따릅니다. 당시 명령과 결과는 이력 보존용이며 현재 계약으로 자동 재개된다는 뜻이 아닙니다.

## 처리 흐름

```text
프로파일·입력 계약·품질 검사·EDA 검토
    ↓
분할 제안 → 승인 → 분할 확정
    ↓
학습 입력 준비 승인
    ↓
Train에서만 특징 제거 목록·대치·변환 기준 fit
    ↓
Train·Validation transform
    ↓
JSON 저장·복원 → 재학습 없이 출력·행 계약 검사
```

`src/preprocessing/transform.py`는 에이전트 없이도 실행하는 독립 코어다. `build_preparation()`은 sklearn `Pipeline`을 반환하며 내부 `TabularPreprocessor`는 clone 가능하다. 교차검증에서는 Pipeline을 각 학습 Fold 안에서 fit해야 한다. Train 전체에서 fit한 변환기를 먼저 적용한 뒤 교차검증하면 누수가 발생하므로 그렇게 사용하지 않는다.

`prepare_splits()`는 분할 계약을 검증한 후 Train만 fit하고 Train·Validation 특징을 반환한다. Test 특징은 생성하지 않는다. 분할 계약 검증 과정에서 전체 입력·분할 메타데이터는 확인하므로, Test 파일 자체를 전혀 읽지 않는 격리 경로라는 뜻은 아니다.

## 설정

[설정 예시](../../../configs/agents/profiles/tabular_prepare_example.json)는 Group 분할·Train-only 준비·독립 최종 검사·보고서를 선택하는 `tabular_preparation_v1` 템플릿이다. `prepare_review_v1`는 준비까지만 수행하는 기존 템플릿으로 유지한다. 실제 입력·경로는 예시에 포함하지 않으며 자동 실행하지 않는다. `training_preparation_v1`에도 `split_rules`와 `preparation_rules`를 명시해야 한다. 실데이터 진입점·승인 화면·게시 연결은 후속 범위다. [최종 계약·보고서](FINAL_CONTRACT_REPORT.md)를 참고한다.

| 항목 | 지원 값·동작 | 기본값 |
| --- | --- | --- |
| missing_ratio_threshold | Train 결측률이 기준을 **초과**하는 특징 제거. null은 미적용 | null |
| drop_constant | Train에서 관측한 고유값이 1개 이하인 특징 제거 | false |
| numeric_imputation | median / mean / none | median |
| scaling | none / standard / minmax / robust | none |
| outlier | none / iqr_clip. 행 삭제가 아니라 값 clipping | none |
| iqr_multiplier | Train Q1·Q3에서 계산한 IQR 배수 | 1.5 |
| binning | none / quantile / uniform | none |
| n_bins | 목표 구간 수. 동점 경계를 합쳐 실제 구간 수는 줄 수 있음 | 5 |
| bin_encoding | ordinal / onehot | ordinal |
| categorical_imputation | missing_category / most_frequent | missing_category |
| unknown_category | error / ignore | error |
| max_categories | 범주형 특징 하나당 원핫 범주 수 상한 | 256 |
| max_output_features | 최종 출력 열 상한 | 4096 |
| max_output_cells | 한 transform 호출의 행×열 상한 | 10000000 |

기본값을 모든 데이터·모델에 강제하지 않는다. 예를 들어 트리 모델용 NaN 보존은 `numeric_imputation: "none"`을 선택할 수 있다. 스케일링은 별도 선택이며, 범주화는 결측 대치를 요구하고 스케일링과 함께 적용하지 않는다. 이상치 처리는 기본 미적용이다.

수치형 처리 순서는 **특징 제거 → 결측 대치 → 선택적 IQR clip → 스케일링 또는 범주화**다. 표준화는 대치·clip 후 Train 평균과 모집단 표준편차를 사용한다. MinMax는 Train 최솟값·범위, Robust는 Train 중앙값·IQR을 사용한다. 범위가 0이면 나누는 값을 1로 둔다. 미래 값은 MinMax에서 0~1 밖으로 나올 수 있으며 자동 clipping하지 않는다.

범주형은 **특징 제거 → 결측 처리 → 원핫 인코딩**이다. 결측 전용 토큰은 실제 문자열과 구분한다. Train에 결측이 없어도 결측 전용 범주는 준비한다. 최빈값 동점은 정규화한 범주 토큰 순서로 결정한다. `ignore`는 새 범주를 해당 특징의 전부 0으로 표현하며 발생 건수는 로컬 진단에 남긴다. ID·그룹·시간·정답 라벨은 출력 특징에서 제외한다.

전체 결측 특징에 median/mean·최빈값 등을 적용할 수 없으면 임의로 0을 채우지 않고 중단한다. 제거 기준을 명시하거나 정책을 검토해야 한다. 기본 품질 검사에서 review_required로 중단된 입력은 전처리 옵션을 지정했다고 자동 통과하지 않는다. 설정 변경·검토 후 새 실행을 준비해야 한다.

## 저장·복원과 출력

`save_preparation()`은 처리 기준·출력 컬럼 순서를 JSON으로 저장하며 기존 파일을 덮어쓰지 않는다. `load_preparation()`은 허용된 필드·유한 통계·경계·역할·출력 폭을 검사하고 fit 없이 복원한다. pickle/joblib·사용자 모듈 import를 사용하지 않는다. JSON도 인증 서명은 아니므로 승인·산출물 정합성 검사와 추후 ACL·인증 무결성 통제가 필요하다.

```python
from src.preprocessing.transform import build_preparation, save_preparation, load_preparation

# train_frame과 validation_frame은 이미 확정된 분할의 DataFrame이다.
pipeline = build_preparation(profile.roles, profile.preparation_rules)
train_features = pipeline.fit_transform(train_frame)
validation_features = pipeline.transform(validation_frame)

save_preparation(pipeline, "preprocessor.json")  # 존재하지 않는 새 파일
restored = load_preparation("preprocessor.json")
same_validation = restored.transform(validation_frame)
```

출력은 고정 순서의 float64 DataFrame이다. 원핫도 현재는 dense 출력이며 크기 상한을 넘으면 중단한다. 대규모 sparse 출력·정교한 물리 규격 범주화·지도학습 특징 선택은 별도 확장 대상이다. `none` 대치에서는 수치형 NaN이 남을 수 있으므로 후속 모델의 결측 지원을 확인해야 한다.

에이전트의 `prepare_data` 도구는 확정 분할의 완료 표식 해시까지 승인 범위에 결합한다. 실행기가 DB 커밋 이력과 파일 정합성을 확인한 선행 폴더만 별도 인자로 전달한다. 승인 후 다음 산출물을 새 attempt 폴더에 저장한다.

- `preprocessor.json`: Train 기준 처리 통계·범주·출력 컬럼 계약.
- `split_contract.json`: 독립 최종 검사에 사용하는 확정 분할 복사본.
- `train_features.csv`, `validation_features.csv`: 모델 입력 특징.
- `row_contract.json`: 각 출력 행에 대응하는 원본 위치·선택적 정답. Test 정답은 복사하지 않음.
- `preparation.json`, `preparation.md`: 원본·설정·분할 결합, 제거 이유, 결측·새 범주 진단, 처리 범위.

저장 후 JSON 복원·transform만으로 특징 CSV와 행 연결을 검사한다. 검증 중 다시 fit하지 않는다. 상세 통계·범주·행 계약·경로는 로컬 전용이며 LLM에 전달하지 않는다. 실제 데이터 CLI·실제 승인 화면·OS 격리 worker는 아직 활성화하지 않았다.

## 기존 코드와 관계

SECOM의 완료한 전처리·모델·실험 결과는 변경하지 않는다. WM 이미지 변환·연결 그룹 분할·캐시는 `wafer_dl`에 유지한다. 이번 표형 준비 도구는 CSV 어댑터만 연결하며 이미지에 표준화·원핫을 강제하지 않는다. 기존 기능의 공통 코어 이관은 별도 정합성 검증 후 진행한다.

## 검증

```powershell
python -m unittest tests.agents.test_generic_transform -v
python -m unittest discover -s tests/agents -q
```

2026-10-15 합성 검증: 신규 17개 통과, 전체 104개 중 103개 통과·Windows 링크 권한에 따른 1개 건너뜀. Train 통계 고정, 50% 제거 경계, 스케일링, clipping, 범주화, 새 범주 차단·발생 진단, 범주 타입·결측 구분, 출력 상한, clone 독립성, JSON 무재학습 복원, 승인·원본 변경 차단, 행 연결과 Test 미변환을 포함한다. 실제 데이터를 학습·재처리한 결과가 아니다.
