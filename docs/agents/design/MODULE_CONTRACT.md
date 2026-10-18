# 모듈형 전처리 파이프라인 — 모듈 계약

작성일: 2026-10-16. 아래는 목표 계약이며 타입·내장 목록은 [등록부](MODULE_REGISTRY.md), 정적 검증은 [파이프라인](MODULE_PIPELINE.md), CSV 분석·검사의 실행 연결은 [승인·worker 문서](../guides/MODULE_PIPELINE_EXECUTION.md)를 따른다. 사용자 코드·변환 실행 연결은 아직 구현하지 않았다. 기존 고정 CLI와 과거 산출물은 유지한다.

## 1. 목적과 범위

각 구성 요소의 위치와 실행 관계는 [에이전트 구성도](../ARCHITECTURE.md)를 먼저 참고한다.

여러 파일에 있는 EDA·품질 검사·전처리를 기능별 모듈로 등록하고, 사용자가 선택한 모듈과 설정으로 순차 파이프라인을 구성한다. 한 파일로 코드를 합치는 것이 아니라 하나의 등록 체계로 관리한다. LLM 없이도 구성·검증·실행할 수 있어야 한다.

모델 학습·하이퍼파라미터 튜닝·예측·문턱 선택은 범위에서 제외한다. 모델을 학습해야 하는 센서 중요도 분석도 초기 전처리 모듈 목록에 포함하지 않는다. 기존 머신러닝·딥러닝 코드는 별도 영역으로 보존한다.

전처리의 중앙값·스케일링 계수·범주 목록 산출은 분류 모델 학습과 다르다. 공개 계약에서는 이를 `fit_required` 대신 **전처리 기준 산출 여부·산출 구간**으로 표현한다. 내부 구현에서 사용하는 scikit-learn의 `fit()`까지 금지하는 뜻은 아니다.

## 2. 모듈 종류

| 종류 | 책임 | 허용하지 않는 동작 |
| --- | --- | --- |
| `analysis` | 분포·결측률·상관관계 등 EDA | 입력 수정·자동 제거·모델 학습 |
| `validation` | 입력·품질·산출물 검사와 진행 여부 판단 | 검토 결과 자동 해제·자동 재라벨링 |
| `transformation` | 승인된 규칙으로 새로운 전처리 산출물 생성 | 원본 덮어쓰기·평가 구간에서 기준 재산출 |
| `reporting` | 검증된 결과를 로컬 표·그림·문서로 구성 | 실행 권한 변경·LLM으로 자동 전송 |

분할 생성은 별도 입력·출력 계약과 승인 절차를 가진 후속 모듈로 연결한다. 분석 모듈이 데이터를 임의로 분할하지 않는다. 첫 버전은 순차 실행이며 임의 DAG·병렬 스케줄링은 별도 범위다.

## 3. 등록 명세의 필수 항목

| 항목 | 내용 |
| --- | --- |
| `schema_version` | 계약 문서 형식. 초기 제안값 `module_contract_v1` |
| `module_id` | 기능의 고정 ID. `[a-z][a-z0-9_]{0,63}` |
| `version` | 모듈 버전. 초기 규칙은 `MAJOR.MINOR.PATCH`, 실행 시 정확한 버전 지정 |
| `kind` | 2절의 종류 중 하나 |
| `inputs`, `outputs` | 이름 있는 입출력 포트, 계약 ID·버전·필수 여부 |
| `settings_schema` | 허용 설정명·타입·범위·기본값·조합 제약 |
| `preconditions` | 필요한 검증 결과·승인·입력 역할·이전 산출물 |
| `original_policy` | 원본 보존 여부와 파생 데이터의 행·특징·역할 변경 범위 |
| `criteria_policy` | 전처리 기준 산출 여부·산출 구간·저장 기준의 적용 방식 |
| `execution_policy` | 승인 종류·읽기/쓰기 범위·네트워크·자원 요구 |
| `result_validator_id` | 등록된 부모 결과 검증기 ID와 버전 |
| `implementation_id` | 검토된 고정 구현의 등록 ID. 임의 import 경로가 아님 |

알 수 없는 필드·중복 키·미등록 ID·미지원 버전·유효하지 않은 설정은 실행 전에 거부한다. 버전 이름만 같아도 코드가 변경되면 기존 승인으로 실행하지 않도록 코드 식별값을 별도 결합한다.

### 3.1 입출력 계약

각 포트는 `name`, `contract_id`, `contract_version`, `required`를 선언한다. 이전 모듈의 출력 포트와 다음 모듈의 입력 포트를 명시적으로 연결하며, 파일 확장자가 같다는 이유로 호환된다고 보지 않는다.

초기 논리 계약은 다음과 같이 구분한다. 아래 이름은 새 등록부를 위한 제안 이름이며 기존 파일의 `schema_version`을 바꾸지 않는다.

| 논리 계약 | 의미·현재 구현과의 연결 |
| --- | --- |
| `tabular_data` v1 | 검증된 어댑터가 구성한 `TabularInput`·`ColumnRoles`. DataFrame은 worker 내부에서만 사용 |
| `eda_report` v1 | 공통 EDA의 CSV·PNG·JSON·Markdown 묶음 |
| `quality_report` v1 | `quality.json`·`quality.md`, 품질 상태와 고정 finding 코드 |
| `approved_split` v1 | 승인·확정된 분할과 행/그룹 역할 증거. 기존 분할 계약에 매핑 |
| `preparation_state` v1 | Train에서 산출한 전처리 기준·특징 순서·범주 매핑의 JSON 상태 |
| `prepared_data` v1 | 저장 기준을 적용한 파생 특징과 원래 행·분할 역할의 연결 |

실행 전달에는 기존 snapshot·프로파일 계약을 재사용한다. 원본 경로를 다음 모듈이 자유롭게 읽게 하거나 DataFrame을 pickle로 전달하지 않는다. 입력·출력의 해시, 출처, 분할 역할, 설정·코드 식별값은 로컬 실행 기록에 보존하며 LLM DTO에 넣지 않는다.

### 3.2 설정과 선행 조건

설정은 허용 목록으로 검증하고 추가 필드를 거부한다. 사용자 설정에 함수 경로·셸 명령·임의 URL·자동 설치 명령을 허용하지 않는다.

선행 조건은 단순히 앞 단계가 실행됐는지가 아니라 **현재 입력·설정에 대응하는 검증된 산출물인지** 확인한다. 품질 상태가 `review_required` 또는 `blocked_input`이면 자동 진행하지 않는다. 제거 정책 승인은 별도 변경 제안에 묶고, 승인 후 생성한 파생 데이터는 다시 검사한다. 기존 보고서의 상태만 `passed`로 바꾸지 않는다.

### 3.3 원본 보존과 전처리 기준

초기 모든 모듈의 원본 보존 정책은 `preserve`다. 변환은 새 출력에만 기록한다. 특징 제거·행 제외·라벨 변경은 서로 다른 효과이므로 명시적으로 구분한다. 초기 표형 전처리는 특징만 변환하고 라벨·ID·그룹·timestamp의 의미나 행 연결을 임의로 변경하지 않는다.

`criteria_policy`는 다음 세 가지 실행 방식을 구분한다.

- `none`: EDA·검사·보고서. 통계를 계산하더라도 재사용할 변환 기준은 만들지 않는다.
- `derive_on_train`: 확정된 Train에서 제거 목록·대치 값·스케일링 계수·범주 매핑 등을 산출하고 상태를 저장한다.
- `apply_saved`: 저장 상태를 변경하지 않고 적용한다. Validation·Test는 이 방식만 허용한다.

분할 전 전체 데이터의 결측률·상수 특징 확인은 진단이다. 모델 입력용 제거 기준은 Train에서 산출한다. 전체 데이터 EDA 결과를 자동으로 Train용 제거 목록으로 재사용하지 않는다. 교차검증에 연결할 때도 각 학습 fold에서 기준을 새로 산출해야 한다.

이미 저장한 기준의 재사용은 입력 계약·특징 순서·설정·알고리즘 버전의 일치를 확인한다. 누락 컬럼과 존재하는 컬럼의 NaN은 구분한다. 알 수 없는 범주 처리도 명시한 규칙만 따른다.

### 3.4 승인·권한·자원

모듈 **등록 승인**과 실제 데이터에 대한 **실행 승인**은 별개다. 등록됐다고 자동 실행하거나 다른 데이터에 대한 승인을 재사용하지 않는다. 변환 승인은 제거 대상·규칙·새 출력·입력 식별값을 포함한다.

실행 승인은 정확한 모듈 버전·코드·입력·설정·계획·자원 한도에 결합한다. 변경 시 새 계획과 승인을 요구한다. 승인 만료·단일 소비·취소·보안 잠금은 기존 실행 관리자를 따른다.

worker는 허용된 복사본·작업 폴더만 사용하고 원본 폴더·승인 DB에 접근하지 않는다. 네트워크는 기존 `deny_all`을 유지한다. 메모리·프로세스 수·시간·입출력 크기를 선언하되 요청은 보안 상한을 넘을 수 없다. 선언만으로 OS 권한이나 제한이 적용됐다고 보지 않고 실제 격리·Job 종료 증거를 확인한다. 디스크 크기 검사는 OS 디스크 quota 보장과 구분한다.

### 3.5 결과 검증

worker의 종료 성공과 데이터 품질 통과를 분리한다. 검사 실행이 성공해도 `quality_status=review_required`일 수 있다. 모듈의 품질/검토 상태는 기존 실행 상태 체계에 매핑하며 새 상태 이름으로 기존 프로토콜을 교체하지 않는다.

부모 검증기는 허용된 파일 목록·크기·해시·형식·필드·계약 일치·범위·완료 증거를 검사한다. 임의 pickle 역직렬화나 결과에 포함된 코드 실행은 금지한다. 검증기가 없으면 모듈을 등록하지 않는다.

일반 데이터의 계산 정답까지 보안 검증만으로 보증하지 않는다. 별도로 정답이 알려진 합성 데이터 테스트에서 결측률·상관계수·변환 결과를 확인한다. 검토/차단·부분 실패에는 완료 표식을 부여하거나 다음 단계로 자동 전달하지 않고 로컬 진단만 보존한다.

## 4. 기존 기능의 계약 매핑

| 제안 모듈 ID / 버전 | 종류·입출력 | 설정·선행 조건 | 원본·기준 | 기존 구현 |
| --- | --- | --- | --- | --- |
| `tabular_eda` / `1.0.0` | analysis, tabular_data → eda_report | `max_plot_features` 1~30. snapshot·역할 계약·실행 승인 확인 | preserve / none | `src/preprocessing/eda.py`: 계산 기능·`collect_eda`; `eda_report.py`: `generate_eda_report` |
| `tabular_quality` / `1.0.0` | validation, tabular_data → quality_report | `QualityRules` 허용 필드. 안전한 로더·프로파일·실행 승인 확인 | preserve / none | `src/preprocessing/quality.py`: `inspect_quality`, `write_quality_report` |
| `tabular_prepare` / `1.0.0` | transformation, tabular_data + approved_split → preparation_state + prepared_data | `PreparationRules` 허용 필드. 분할 확정·필요한 검토/변환 승인 확인 | preserve / derive_on_train 또는 apply_saved | `src/preprocessing/transform.py`: `TabularPreprocessor`, `prepare_splits` |

품질 검사는 데이터 내용의 이상을 보고해야 하므로 EDA와 동일한 엄격 입력 검증을 앞에 강제해 보고서 생성을 막지 않는다. 안전한 파일 읽기·구조 검증과 품질 규칙에 의한 내용 진단을 분리한다.

EDA 파일 결과 검증은 `src/agents/eda_result_validation.py`, 품질 파일 검증은 `src/agents/quality_result_validation.py`를 재사용한다. 현재 snapshot EDA·audit CLI는 각각 고정 도구에 연결돼 있다. 위 표는 목표 복합 기능이며 실제 초기 등록 목록은 EDA 5개 계산 기능과 `tabular_quality`다. 등록부의 메모리 결과 검증기는 기존 부모 파일 검증을 대체하지 않는다.

초기 전처리 연결은 기존 복합 `TabularPreprocessor`를 하나의 모듈로 감싼다. 제거·대치·스케일링·범주화를 즉시 별도 상태 모듈로 분해하지 않는다. 해당 변환의 격리 worker 연결과 부모 검증 계약은 후속 구현에서 보완한다.

## 5. EDA 등록 명세 예시

다음은 설계 검토용 예시이며 현재 CLI 입력 파일이 아니다. 설정 스키마의 구체 문법은 통합 등록부 구현 시 확정한다.

```json
{
  "schema_version": "module_contract_v1",
  "module_id": "tabular_eda",
  "version": "1.0.0",
  "kind": "analysis",
  "inputs": [{"name": "data", "contract_id": "tabular_data", "contract_version": 1, "required": true}],
  "outputs": [{"name": "report", "contract_id": "eda_report", "contract_version": 1, "required": true}],
  "settings_schema": {
    "additional_properties": false,
    "properties": {"max_plot_features": {"type": "integer", "minimum": 1, "maximum": 30, "default": 12}}
  },
  "preconditions": ["snapshot_verified", "tabular_roles_validated"],
  "original_policy": {"mode": "preserve", "changes_rows": false, "changes_features": false},
  "criteria_policy": {"mode": "none", "derivation_partition": null},
  "execution_policy": {
    "approval": "explicit_execution",
    "read_scope": "approved_staged_inputs",
    "write_scope": "task_outputs_and_scratch",
    "network": "deny_all",
    "resources": "explicit_request_within_security_caps"
  },
  "result_validator_id": "tabular_eda_result_v1",
  "implementation_id": "builtin_tabular_eda_v1"
}
```

CSV 로더 선택은 `adapter_id`, 분석 내용 선택은 `module_id`로 분리한다. 등록 ID는 후속 구현에서 고정 구현에 연결하며 문자열을 그대로 import하지 않는다.

## 6. 사용자 작성 모듈 등록 조건

- 소스·입출력 계약·설정·의존 라이브러리·부작용을 검토한다.
- import 시 데이터 읽기·모델 학습·파일 쓰기를 시작하지 않는다.
- 고정 구현 ID·코드 식별값·worker 복사 허용 목록을 등록한다. 임의 `.py`의 자동 검색·자동 실행은 하지 않는다.
- 결과 검증기와 합성 테스트를 필수로 두며 취소·자원 초과·변조 시 완료 등록하지 않는지 확인한다.
- 새 의존성을 자동 설치하거나 권한을 확대하지 않는다. 기존 모듈의 승인을 다른 모듈에 재사용하지 않는다.

## 7. 1번 완료 기준과 다음 작업

- [x] ID·버전·종류·입출력·설정·선행 조건 정의.
- [x] 원본 보존과 전처리 기준 산출·적용 구분. 모델 학습 제외.
- [x] 승인·권한·자원·결과 검증을 계약 항목에 포함.
- [x] 기존 EDA·품질 검사·복합 전처리와의 대응 정리.
- [x] 2번: 계약 타입과 고정 내장 6개 기능의 등록부 구현. 실제 실행 연결은 별도.
- [x] 파이프라인 설정·순서/입출력/의존성 정적 검증. [구현 범위](MODULE_PIPELINE.md)
- [x] CSV 분석·검사에 대한 기존 실행 관리 연결과 합성 검증. [실행 범위](../guides/MODULE_PIPELINE_EXECUTION.md)
- [x] 새 진입점의 합성 정상·검토 중단 실제 OS 실행 기록 확인. [검증 기록](../guides/MODULE_PIPELINE_EXECUTION.md)
- [ ] 변환 실행 연결. [선행 계약·구현 계획](MODULE_TRANSFORM_PLAN.md)
- [ ] 사용자 모듈 등록.

LLM 해설·선택형 보안 검토는 [모듈 개발 이후 검토 안건](LLM_COMMENTARY_REVIEW.md)으로 분리하며 이번 계약으로 정보 전송을 허용하지 않는다.
