# 범용 전처리 에이전트 설계

에이전트의 단일 설계 기준이다. 현재 사용·검증 상태는 [GUIDE](GUIDE.md), 명령별 절차는 `guides/`, OS 설정·진단은 `security/`에서 확인한다. 기본·상세 설계, 구성도와 기능별 설계에 분산되었던 유효한 계약을 이 문서로 대체한다. 과거 설계·변경 과정은 Git 이력으로 확인한다.

## 읽는 방법

작업에 해당하는 절만 읽고 수정한다. 구현 이력·테스트 수치를 이 문서에 누적하지 않는다. 필드·상한·허용값의 정확한 정의는 아래에 연결한 계약 코드가 기준이며, 문서 변경만으로 기능을 활성화하거나 승인·검증을 완화하지 않는다. 사용자와 새로 확정한 정책은 구현 상태와 구분한다.

| 변경 영역 | 기준 절 |
| --- | --- |
| 공통 구조·유형 연결 | [책임 구분](#책임-구분), [내부 구조](#내부-구조), [배열·이미지 연결](#배열이미지-통합-작업-연결) |
| 입력·프로파일·헤더 정비 | [입력 계약](#입력-계약) |
| 권장안·질문·설정 확인 | [계획과 질문](#계획과-질문) |
| 실행·동의·위임 | [승인과 실행](#승인과-실행) |
| 등록·바인딩·기준 산출 | [모듈과 파이프라인](#모듈과-파이프라인) |
| 결과·복구·과거 실행 | [검증과 저장](#검증과-저장) |
| LLM·개인정보·격리 | [정보 전달](#정보-전달), [보안 기준](#보안-기준) |
| 다음 구현 | [후속 범위](#후속-범위) |

## 목표와 범위

데이터와 목적을 받으면 진단 → 권장안·필요한 질문 → 계획 확인 → 분할·전처리 → 적용 확인 → 결과까지 연결한다. 사용자에게 내부 실행 이름·경로·선행 결과를 수동 연결시키지 않는다. 실행 성공뿐 아니라 문제·실제 처리·전후 변화·미처리 사항·산출물이 이해되는 것을 완료 기준으로 삼는다.

현재 제품 통합 작업은 CSV 로더를 쓰는 표형 어댑터와 `.npz` 배열 어댑터다. 기존 등록 모듈·CLI의 WM 처리와 일반 pickle canonical 변환은 별도 경로이며, 웨이퍼 통합 프로파일을 의미하지 않는다. 모델 학습·Test 성능 평가·임의 코드 생성 실행은 수행하지 않는다. 코어는 에이전트 없이도 호출할 수 있다.

## 책임 구분

> 구성도 이미지를 누르면 인터랙티브 버전으로 이동한다(GitHub Pages 배포 후 활성화). 배포 전에는 각 로컬 HTML을 내려받아 브라우저에서 열 수 있다. 그림은 커밋 `4fd209c` 기준이며 진행 중인 작업 8 변경은 포함하지 않는다.

[![공통 책임과 유형별 처리 구성도](../architecture/responsibilities.svg)](https://didgustmd23.github.io/waferguard/responsibilities.html)

[인터랙티브 보기](https://didgustmd23.github.io/waferguard/responsibilities.html) · [로컬 HTML](../architecture/responsibilities.html) · [구성도 원본](../architecture/responsibilities.json)

| 계층 | 책임·의존 방향 |
| --- | --- |
| 공통 핵심 | 조정·상태·잠금·작업 기록·질문·동의·진행·문제 처리·보고서·화면 틀. 유형별 모듈이나 WM 필드를 직접 알지 않는다 |
| 데이터 형태 어댑터 | canonical 입력·진단·허용 모델 집계·결정 스키마·기본 계획·검사·질문·서비스 연결·검증된 결과 요약. 현재 표형(CSV)과 배열(`.npz`) |
| 도메인 프로파일 | 형태 위에 분야 의미·라벨·그룹·전용 검사·모듈 카탈로그·계보 규칙을 더한다. 웨이퍼는 배열 위의 선택적 프로파일이지 데이터 형태가 아니다 |
| 로더·입력 매핑 | CSV, 구조 판별 pickle, WM-811K 필드 매핑. 형식 판별이 모호하면 질문하며 임의 추정으로 실행하지 않는다 |
| 계산 모듈 | `src/preprocessing/`와 기존 도메인 계산. 공통 코어는 `agents`·`sensor_ml`·`wafer_dl`에 의존하지 않는다 |

### 내부 구조

[![어댑터 등록·위임 승인·실행·부모 검증·보고서 연결 구성도](../architecture/internal.svg)](https://didgustmd23.github.io/waferguard/internal.html)

[인터랙티브 보기](https://didgustmd23.github.io/waferguard/internal.html) · [로컬 HTML](../architecture/internal.html) · [구성도 원본](../architecture/internal.json)

| 결정 | 화면에서 하는 일 | 조정기가 이어서 하는 일 |
| --- | --- | --- |
| A | 파일·목표 입력, 진단·모델 전달 범위 동의 | snapshot 복사 → 격리 진단 → 허용 목표·이름/구조·집계·계획/선택지로 모델 권장안 → 계획 확인 화면 |
| C | 역할·매핑·분할·전처리 수정, 질문 답변, 최종 계획 미리보기 후 동의 | 분할 → (CSV) 내보내기 → Train 기준 후보 → 예상 영향 표시 |
| D | 후보·경고 확인 후 적용 동의(또는 계획 수정으로 새 시도) | Train·Validation 적용 → 결과 묶음 → 공통 보고서 |

각 단계의 1회 승인은 해당 동의 범위·입력·계획·코드 지문과 맞을 때만 위임 발급한다. worker 출력은 부모 검증기를 통과한 것만 커밋하고, 화면·보고서는 검증된 결과만 다시 읽는다.

등록부 `task_adapters.py`는 유형별 어댑터를 import하는 의도된 연결 경계다. 조정·계획·동의·저장·보고서의 공통 규칙이 유형별 계산에 직접 의존하지 않는 것과 구분한다. 구성도의 연결은 주요 호출/처리 관계이며 모든 함수 호출·반환을 나타내지는 않는다. 표형 진단은 `pipeline_approval_cli.record_delegated_approval`에서 동의를 대조해 자체 위임 기록을 남기고 실행기를 거친다. CSV·배열 서비스의 공통 `delegated_approval` 경로와 구분하되, 모두 승인 소비 뒤 격리 계산하고 부모가 출력을 검증한다. `task_report`는 어댑터의 `report_facts`를 호출해 검증된 결과 사실을 받아 조립하며, `report_facts` 연결선은 그 반환 자료를 나타낸다.

주요 구현은 `task_orchestrator.py`, `task_plan.py`, `task_consent.py`, `task_store.py`, `task_adapters.py`, `task_report.py`, `tabular_task.py`, `array_task.py`와 `src/ui/task_view.py`다. 등록은 고정 ID 목록을 사용하고 폴더 탐색·동적 import·새 에이전트 프레임워크를 도입하지 않는다.

### 배열·이미지 통합 작업 연결

**상태: 구현, 합성 검증 및 합성 입력의 실제 격리 실행·모델 요청·사용자 화면 확인 이력 있음.** 아래 확정 결정대로 `array_task*.py`·`array_options.py`·`src/preprocessing/array_dataset.py`·`src/ui/array_task_form.py`로 구현하고 통합 작업에 등록했다. 최신 bool 라벨 보완은 대상 테스트·전체 회귀와 합성 입력의 실제 AppContainer 실행까지 확인했다. 배열 메모리의 기본 용량·옵션·dtype·제한 초과 시험도 확인한 범위 안에서 완료했다. 검증 근거·모델 품질 한계·네트워크 검증 범위 등 현재 범위는 [운영 안내](GUIDE.md#검증-상태와-남은-제한)를 따른다. 웨이퍼 전용 규칙(WM-811K 필드·결함 맵 채널·lot 규칙)과 시간 처리는 이 범위가 아니다. WM-811K 매핑의 `waferMap`·`failureType`·`lotName`은 이후 웨이퍼 프로파일 안에만 둔다.

**재사용 판정(코드 확인)**

| 대상 | 판정 |
| --- | --- |
| `task_orchestrator`·`task_plan`·`task_consent`·`task_store`·`task_recommend`·`task_report`, `task_forms.generic_form` | 그대로 재사용. 어댑터 함수 집합은 테스트 전용 `tests/agents/array_task_fixture.py`로 이미 끝까지 검증됨 |
| `input_snapshot`(단일 파일 복사), 승인 저장소·`RulesExecutor`·`delegated_approval.record_delegated`·`workflow_approval`, AppContainer·Job 실행·`worker_contract` | 그대로 재사용 |
| `split._partition`(random·group·계층) | 표본별 메타데이터 표(`sample_id`·`label`·`group`)를 만들어 재사용. 배열 값은 분할에 쓰지 않음 |
| `canonical_data` 숫자 배열 계약 | 일부 재사용. `canonical_array_v1`은 배열 하나만 담고 표본 ID·라벨·그룹이 없어 표본 메타데이터를 더한 묶음 계약이 필요 |
| `transform.py`(표형 전처리)·`impact` 계산 방식 | 계산 코드는 재사용 불가(컬럼 단위). 예상/실측 영향 구분, 후보 v2·적용 v2 검증 방식은 같은 원칙으로 따름 |
| `module_pipeline_worker` | 재사용 불가. CSV 로드·`tabular_csv_v1`·중복 헤더 분기가 고정돼 있고, 수정하면 CSV 파이프라인 코드 지문이 바뀜 |
| `wm_*` 실행 체인·`src/wafer_dl` | 재사용 불가. WM 필드·결함/유효 채널·lot 분할·범주형 맵 resize에 고정. 서비스 API 모양만 같음 |
| `pickle_import`의 `numeric_array_v1` | 첫 범위에서 미사용. 단일 배열만 변환하고 라벨·그룹 매핑이 없음 |
| 요청 화면·파일 선택기 | 공통 보완 필요. `task_view`가 등록 첫 유형의 입력 안내(`specs[0]`)만 쓰고, 선택기 종류가 `csv_file`·`json_file`·`directory`뿐 |

**첫 지원 입력과 매핑**
- 형식: NumPy `.npz` 파일 하나를 `np.load(allow_pickle=False)`로 읽는다. pickle·이미지 폴더·크기가 다른 표본 묶음은 첫 범위가 아니며, 선택하면 지원 형식과 다음 행동을 안내하고 실행하지 않는다.
- 판별: 어댑터 선택은 기존 `task_adapters.select`의 확장자·로더 판별을 따른다. 로더는 zip 항목 수·선언된 압축 해제 크기·각 배열 헤더(dtype·shape)를 읽기 전에 상한과 대조한다. object dtype은 거부한다.
- 매핑(결정 항목, 계획의 `mapping.*` 경로): `samples`는 숫자·bool 배열이며 첫 축이 표본이다(N×H×W 또는 N×H×W×C 등, 2~5차원). 나머지 축의 의미(`channel_axis`: 없음·마지막·두 번째)는 사용자가 정한다. `label`은 길이 N의 1차원 bool·정수·실수·유니코드 문자열 배열이거나 없음이며, bool 라벨은 결과 JSON에서도 true/false 타입·값·표본 순서를 보존한다. `group`·`sample_id`는 길이 N의 1차원 bool·정수·유니코드 문자열 배열이거나 없음이다. `sample_id`가 없으면 위치 번호를 쓴다.
- 기본 계획은 이름 단서 없이 형상으로만 정한다. N을 공유하는 다차원 배열이 하나뿐이면 표본 후보로 둔다. 후보가 여럿이거나 없으면 필수 질문이다. 모델은 키 이름·형상·자료형으로 매핑 변경을 권할 수 있지만 같은 계획 검증을 통과해야 한다.

**진단(격리 worker, 결정 A)**
- 구조: 키별 dtype·shape, 표본 수, 매핑 후보, 형상 일치.
- 값(채널별): 비유한 값 수, 0 비율, 평균·표준편차, 고유값 수 구간(정수형이고 고유값이 적으면 범주형 맵 가능성 안내).
- 표본: 상수 표본 수, 완전 동일 표본 수(표본 바이트 해시 기준, 근사 중복은 범위 밖).
- 라벨·그룹: 클래스 수·클래스별 표본 수·라벨 결측, 그룹 수·그룹당 표본 수.
- 최솟값·최댓값·분위수와 표본 값은 로컬 표시용이며 모델 전송 집계에 넣지 않는다(표형과 같은 원칙).

**사용자가 결정할 전처리(모두 none 선택 가능, 자동 변경 없음)**
- 비유한 값: 차단(none) 또는 Train 채널 중앙값 대치.
- 값 clipping: none 또는 Train 채널 분위수(하한·상한 % 선택) 경계.
- 크기 조정: none, Train 채널 평균·표준편차 표준화, Train 채널 최솟값·최댓값 정규화, 학습 없는 고정 배율(나누는 값 입력).
- 결과는 모든 옵션이 none이어도 float32로 저장한다. float64·큰 정수(2^24 초과) 입력은 반올림되거나 서로 다른 값이 같아질 수 있으며, 옵션 설명·결정 D 화면·보고서(정보 손실)에 표시한다. 입력 자료형 보존 기능은 없다. 기존 상수 채널 경고와 별개다.
- 라벨 결측 표본: 라벨을 매핑했고 결측이 있으면 필수 질문(분할 전 제외 / 라벨 없음으로 유지, 유지하면 계층 분할 불가).
- 완전 동일 표본: 기본은 유지하고 경고한다. "누수 방지: 같은 표본을 한 구간에 모으기" 선택지를 제공하며, 사용자가 확정한 뒤에만 같은 해시끼리 분할 그룹으로 묶는다. 매핑한 group이 있으면 두 기준을 함께 만족하도록 합친다. 표본 삭제는 제공하지 않는다.
- 분할: random(라벨 계층 선택 가능) 또는 group(그룹 매핑 필요). 역할×분할 교차 검사는 표형과 같은 규칙. time은 이번 범위가 아니다.
- 옵션 설명(처리 내용·적용 대상·근거·주의·none의 의미), 출처 표시, 양식·답변 충돌 선택, 최종 계획 미리보기 결합은 표형과 같은 공통 규칙을 따른다.

**기준 산출·적용 구간과 누수 방지**
- 결정 C 동의 범위: 품질 검사 → 분할 제안·확정 → Train·Validation 배열 내보내기 → Train 기준 후보. 결정 D 동의 범위: Train·Validation 적용 → 로컬 전달 묶음. 분할 확정 증거·단계별 1회 승인·위임 발급 규칙은 표형과 같다.
- 분할은 배열 값이 아닌 메타데이터로만 계산한다. 대치값·분위수 경계·평균·표준편차·최솟값·최댓값은 Train에서만 채널별로 산출하고, Validation은 저장 기준만 적용한다. Test 배열은 내보내지 않고 통계·적용에 쓰지 않는다.
- 분할 뒤 구간 사이 완전 동일 표본 수와 group 매핑 시 구간 사이 그룹 겹침 0건을 검사해 보고한다. 동일 표본이 구간을 넘으면 누수 경고로 표시한다(자동 이동·삭제 없음).
- 라벨·그룹·`sample_id`는 특징 배열에 넣지 않고 표본 메타데이터로 보존한다.

**모델 전달 집계**
- 결정 A 범위에서 목표 문장, 키 이름·dtype 종류·shape, 위 진단의 허용 집계, 기본 계획·허용값만 보낸다. 표본 값·이미지·경로·최솟값·최댓값·분위수는 보내지 않는다.
- 표본 배열의 평균·표준편차는 값이 한 종류이거나 유한값이 2개 이하이거나, 값이 정확히 두 종류이고 그중 하나가 0이면 모델 요청에서 함께 뺀다(0 비율과 평균으로 나머지 값이 복원됨). 로컬 진단·계산에는 그대로 쓴다. 0이 없는 두 값 배열(예: {1, 2})은 표형과 같은 수준으로 평균·표준편차를 보낸다.
- 집계 이름에 건수·분모·비율 기준을 적는다(예: `nonfinite_or_missing_ratio_of_values`, `zero_ratio_of_finite_values`). 지시문은 수치 전처리가 표본 배열에만 적용되고 그룹 배열은 라벨이 아니며 0 비율과 비유한 값 비율이 다름을 명시한다.
- 모델이 라벨·그룹 매핑을 바꿔 권하면 '확인 필요'로 표시하고, 표시한 계획과 최종 계획이 그 값을 쓰는 동안 필수 확인 질문에 답하거나 양식에서 값을 바꿔야 확정할 수 있다. 사용자가 양식에서 직접 고른 값은 다시 묻지 않는다.
- 라벨은 클래스 이름 대신 순번(`class_1`…)과 개수만 보낸다. 화면과 보고서에 "모델은 클래스 의미를 해석하지 못하며 권장안은 분포만 근거로 한다"는 한계를 표시한다. 요청 예산·coverage·응답 계약 검사·전체 거부 규칙은 기존 `task_recommend`를 그대로 쓴다.
- 표본 미리보기 이미지는 어떤 경우에도 모델 요청에 넣지 않는다.

**결과 보고서(공통 틀, 어댑터 사실)**
- 핵심 결과: 표본 수 전→후, 구간별 표본·클래스·그룹 수, 매핑.
- 처리: 옵션별 설정과 출처, Train 채널 기준값(로컬 표시).
- 전후 비교: 예상(Train 후보 계산)과 실측(적용 결과 검증), 비유한 값 대치 수, clipping 변경 값 수·비율, 변환 후 상수 채널, 구간별 값 결측.
- 경고: 정보 손실(clipping·상수 채널·제외 표본), 미처리(동일 표본 유지, Test 미적용), 미검증(worker 집계 항목, 모델 학습 없음).
- 활용: `train_samples.npy`·`validation_samples.npy`, 표본 메타데이터 JSON, 저장 기준 JSON의 용도와 Test 적용 방법.
- 메모리: 입력 파일 크기·압축 해제 크기·원소 수 상한은 입력 검사 기준이며 실제 계산 메모리 요구량과 다르다. worker는 표본 배열을 한 번에 읽고 float64로 기준을 계산하며, 부모 검증도 출력 배열을 한 번에 읽는다. 작업 메모리 상한(Job)을 넘으면 중단하고 상한을 자동으로 늘리지 않는다. 진단 중 실행 실패는 원인을 단정하지 않는 고정 문구로 안내한다. 대용량 스트리밍 처리는 지원하지 않는다.
  - 실제 합성 검증: 32·128 MiB float32 입력은 기본 Job 2 GiB에서 진단·후보·적용과 부모 결과 검증을 완료했다. 32 MiB 입력에 시험 전용 256 MiB 제한을 적용하면 메모리 초과 통지 후 중단·정리되고 미완성 결과는 커밋되지 않았다. 부모 메모리는 별도로 측정했으며 worker Job 제한의 대상은 아니다. 측정 조건·수치·남은 범위는 [자원 제한](security/RESOURCE_LIMITS.md#배열-통합-작업-메모리-실측)을 따른다. 입력 상한 2 GiB 전체의 처리 가능성을 입증한 것은 아니다.
- 그래프: 구간별 클래스 분포·채널 값 분포 등 집계 그림을 둔다.
- 표본 미리보기
  - 진단 worker가 기본 매핑의 표본 배열에서 표본 순서 앞쪽 최대 12개를 PNG로 만든다. 진단 시점에는 라벨 매핑이 정해지지 않아 클래스별로 고르지 않는다. 표시용 밝기 조정만 하고 데이터에는 적용하지 않는다. 이미지 형태(2차원, 또는 채널 1·3)가 아니면 만들지 않는다.
  - 부모는 개수·크기·PNG 형식·선택 규칙 기록을 검증한다. 화면은 표시할 때마다 검증된 파일만 연다.
  - 로컬 화면과 로컬 보고서 목록에서만 보이며, 모델 전달·해설 패킷·전달 묶음에는 넣지 않는다. 미리보기는 처리 결과가 아니라 입력 확인용이라고 표시한다.
- 부모 검증: 배열 헤더·shape·dtype·파일 해시·표본 수·메타데이터 순서·분할 계보를 확인하고, 상수 채널·결측 수처럼 출력에서 다시 셀 수 있는 실측은 대조한다.

**연결 구조와 기존 CSV 영향 차단**
- 새 유형 어댑터 `array_npz`를 `task_adapters.ADAPTERS`에 등록한다. 계산은 유형별 격리 도구 3개로 기존 모듈 형태(contract·cli·transport·worker·results)를 따른다. 진단(결정 A), 분할·내보내기·기준 후보(결정 C, 단계형 한 도구), 적용·전달(결정 D)이다. 새 조정기·화면·승인 체계를 만들지 않는다.
- 공통 보완은 두 가지다. 요청 화면은 등록된 모든 입력 형식을 안내하고 선택 파일로 어댑터를 고른다. 파일 선택기에는 어댑터가 정한 확장자 종류를 추가한다.
- 표형 어댑터·CSV 서비스·`module_pipeline_worker`·`transform.py`는 수정하지 않는다. `task_adapters.py`는 CSV `code_id` 대상이라 등록 시 CSV 코드 지문이 한 번 바뀐다. 적절한 구조를 우선해 이를 수용한다. 미실행 CSV 승인과 진행 중 통합 작업은 다시 준비해야 하며, 구현 후 사용 안내에 이 영향을 알린다.
- 검증
  - 공통 경계 테스트(공통 코드에 유형 이름 없음)를 유지한다.
  - 기존 표형·CSV 대상 테스트를 기대값 변경 없이 통과해야 한다.
  - `.csv`는 표형, `.npz`는 배열로만 선택되는지 확인한다.
  - 합성 npz로 정상 흐름과 거부 사례(object 배열, 압축 해제 상한, 길이 불일치, 매핑 누락, 비유한 값 차단, group 겹침, 변조 출력, v1 다운그레이드형 우회)를 검사한다.
  - 미리보기 개수·형식 위반과 변조를 거부하고, 미리보기가 모델 요청·해설 패킷·전달 묶음에 없는지 확인한다.
  - 동일 표본 묶기는 확정 전 계획을 바꾸지 않고, 확정 후 구간을 넘는 동일 표본이 0건인지 확인한다. 미선택 시에는 누수 경고만 내는지 확인한다.
  - 모델 요청에 라벨 원래 값이 없고 순번만 있는지 확인한다.
  - 독립 보안 검토를 받는다. 실제 AppContainer 선택 검사와 실제 화면 1건은 합성 검증과 구분해 보고한다.

**확정 결정**
1. 첫 입력은 `.npz` 하나다. pickle·이미지 폴더 입력은 후속 작업이다.
2. 어댑터 등록에 따른 CSV 코드 지문 1회 변경을 수용한다. 지문 유지보다 적절한 구조를 우선한다.
3. 모델에는 라벨 순번·개수만 보내고, 클래스 의미 해석 불가 한계를 표시한다.
4. 제한된 개수의 검증된 표본 미리보기를 로컬에만 표시한다. 모델에는 보내지 않는다.
5. 동일 표본 그룹 묶기를 누수 방지 선택지로 제공한다. 사용자가 확정한 뒤에만 적용하며, 삭제는 하지 않는다.

### 공통 표형 입력·처리 확장

**상태: T1 구현 중·작업 1~7(저장/출처, CSV 검사/생성, 입력 승인·해석 재개, v2 진단·분할·내보내기, typed 후보/적용, v2 서비스·handoff·보고서 근거, v2 어댑터·통합 화면 연결)의 구현·합성 검증 완료.** 통합 화면의 새 CSV 작업은 v2 표형 작업으로 진행한다. 작업 8의 지문·독립 검토·합성 입력의 실제 격리·관리자 UI 확인을 마쳤고, 기준 커밋의 전체 회귀로 작업 1~7 관련 합성 검증을 확인했다. 실제 모델 권장안 반영 성공·실데이터·대용량 성능은 미검증이며, 실제 격리·네트워크 시험과 생략 범위는 GUIDE를 따른다. 표형 입력 형식은 여전히 CSV뿐이다. 작성했던 웨이퍼 통합 연결 설계안은 폐기하고 기존 WM 계산·개별 실행·도메인 문서는 유지한다. 아래 T2~T4와 연결 전 계약/파일은 설계 기준이며 지원 완료를 뜻하지 않는다.

**지원 목표**

| 입력 형식 | 필요한 해석·선택 | 제한 |
| --- | --- | --- |
| CSV·TSV·구분자 기반 TXT/DATA/DAT | 인코딩·구분자·헤더·결측 표현 | 확장자만으로 구분자나 자료형을 확정하지 않음 |
| XLSX·XLS | 시트·헤더 범위·표 영역·수식 값 정책 | 복수 시트 자동 병합·매크로/수식 실행 안 함 |
| Parquet·Feather | 컬럼·자료형·파일/테이블 구조 | 기존 자료형을 설명 없이 문자열로 손실 변환하지 않음 |
| JSON·JSONL | 레코드 경로·중첩 필드·배열 펼침 정책 | 임의 깊이 flatten·행 증식·다중 테이블 자동 병합 안 함 |
| DataFrame Pickle | 구조 선택·출처/신뢰 승인·격리 역직렬화 | 임의 객체를 표로 강제 변환하거나 일반 파일 읽기로 처리하지 않음 |

**공통 흐름**

형식별 로더 → 격리 입력 진단·필요한 해석 질문 → 승인된 공통 표형 작업 복사본 → 역할·타깃·그룹·처리 계획 → 기존 승인형 분할·Train 기준 전처리·적용 → 공통 보고서로 이어진다. 형식마다 조정기·승인 체계·전체 화면·전처리 엔진을 만들지 않는다. 어떤 형식도 파일 열기만으로 지원 완료로 보지 않으며 결과와 전달까지 확인한다.

- 공통 표형 계약은 컬럼 이름/순서·논리 자료형·결측 표현·행 위치/식별·입력 해석 설정·원본 및 정규화 지문을 보존한다. 아래 v2 계약은 기존 표형 canonical의 안전한 태그/검증 원칙을 재사용한다. 내부 입력을 무조건 CSV로 내보내 재읽지 않는다.
- 현재 표형 어댑터뿐 아니라 진단·분할·후보·적용 worker의 CSV 의존부를 조사한다. 로더 하나만 추가하면 모든 후속 단계가 호환된다고 가정하지 않는다. 기존 계산 코어와 결과 검증을 재사용하고 입력 로딩 경계를 공통화하는 범위를 제시한다.
- 원본은 보존하고 새 작업 복사본만 사용한다. 헤더 정비·자료형/결측 표현 정규화는 변경 전후 매핑·근거·손실 가능성을 표시하고 동의 범위에 결합한다. 파일 표현 정규화와 Train에서 기준을 학습하는 대치·인코딩·스케일링을 분리해 누수를 방지한다.
- 숫자처럼 보이는 문자열 등 공통 입력 정비는 대상·실패 값·결측으로 바뀔 영향과 사용자 선택을 설계한다. TotalCharges 같은 특정 컬럼명 예외나 조용한 변환을 넣지 않는다. 표현 정규화만으로 역할·target을 자동 확정하지 않는다.
- 모호한 구분자·헤더·시트·레코드 경로는 필요한 질문으로 해결한다. 지원하지 않는 구조·암호화·라이브러리 부재·용량/시간 초과는 원인과 다음 행동을 표시한다. 모든 임의 파일을 무제한 수용한다는 의미의 '전부 지원'은 약속하지 않는다.
- 값 파싱은 격리 worker에서 수행하고 부모는 byte 복사·내용 지문·허용 경로/결과 검증 경계를 유지한다. 원본 행·경로·전체 파일·예외 원문을 모델에 자동 전달하지 않는다. 요청·응답 계약과 승인·코드 지문 결합을 형식 확장으로 약화하지 않는다.
- 형식별 합성 정상·모호성·손상·자원 초과·악성/위장 입력, 자료형/정밀도·결측·행/열 순서 보존, 원본 불변·변조 거부, 동일 표의 형식 간 결과 동등성을 검증한다. 기존 CSV/배열 회귀와 실제 AppContainer/UI 확인을 분리한다.

#### 조사 근거와 선택한 구조

| 영역 | 확인한 현재 코드 | 설계에 반영할 점 |
| --- | --- | --- |
| 공통 흐름 | `task_orchestrator.py`·`task_adapters.py`·`task_plan.py`가 어댑터의 진단·계획·단계를 호출 | 조정기·동의·질문·작업 기록·보고서 틀 재사용 |
| 입력 선택 | `tabular_task.accepts`는 `.csv`만 허용하고 `task_orchestrator.start_task`는 파일 하나를 받음 | 표형 유형과 형식 로더를 분리하고 입력 해석 상태 추가 |
| 진단·분할 | `module_pipeline_worker.py`·`snapshot_split_worker.py`가 UTF-8 CSV를 직접 읽음 | 공통 표형 reader 뒤에서 기존 진단·품질·분할 계산 호출 |
| 구간·전처리 | `partition_export_worker.py`는 CSV로 구간을 내보내고 후보·적용 worker가 CSV로 다시 읽음 | Train·Validation 중간 결과도 공통 표형 계약으로 유지 |
| 계산 | `input_profile.py`·`eda.py`·`quality.py`·`split.py`·`transform.py`는 DataFrame 또는 `TabularInput`을 받음 | 계산 코어 재사용, 역할별 자료형 적합성은 실행 전 검사 |
| 기존 저장 | `canonical_data.py`는 schema·태그형 JSONL·인덱스·자료형을 보존하고 `load_table`로 복원 | 기존 v1을 덮어쓰지 않고 후속 v2 계약을 추가 |
| Pickle | `pickle_import_worker.py`에서 역직렬화, 부모 `pickle_results.py`에서 제한된 결과 검사 | 승인·격리 재사용. `pickle_converters.py`의 WM 조립 의존은 공통 로더에 끌어오지 않음 |
| SECOM 원본 | `sensor_ml/data_pipeline/step1_merge_data.py`에 특징/메타데이터 두 파일의 행 위치 결합이 있음 | 기존 실험은 유지. 통합 작업의 복수 입력 결합은 별도 단계로 명시 |

선택안은 **형식별 격리 로더 → 버전 있는 공통 표형 묶음 → 기존 계산 코어**다. 모든 입력을 CSV로 변환하는 안은 nullable 정수·불리언·범주·시간·빈 문자열의 의미가 재읽기 때 바뀔 수 있어 채택하지 않는다. 형식마다 조정기·도구 전체를 복제하는 안도 채택하지 않는다. 기존 canonical v1만 그대로 연결하는 안은 범주·시간대·해석 설정·정비 계보가 부족하므로 확장한다. 파일 분할·함수명은 구현자가 조정할 수 있지만 책임과 계약 경계는 유지한다.

#### 형식별 로더와 지원 경계

확장자는 후보 선택에만 사용한다. 승인된 snapshot의 서명·구조를 격리 worker에서 검사하고, 위장 파일을 다른 파서로 무제한 재시도하지 않는다. 사용자 파이썬 표현식·converter 함수·import 경로·URL·원격 파일시스템 옵션을 받지 않는다. 결과는 하나의 유한한 표이며 자료형별 지원과 계산 가능 여부를 구분한다.

| 로더 ID(제안) | 처리할 입력과 선택 | 첫 구현의 명확한 경계 |
| --- | --- | --- |
| `delimited_v1` | `.csv`·`.tsv`·`.txt`·`.data`·`.dat`; 인코딩, 고정 구분자 또는 공백 구분, 따옴표, 헤더 유무, 결측 토큰, 소수점/천 단위 | UTF-8/BOM·CP949·UTF-16의 명시 선택. 추정이 모호하면 질문. 고정폭·다중 표·주석/행 건너뛰기 규칙은 지원 설정에 없으면 거부. 헤더 없는 경우 위치 기반 컬럼명을 제안하고 확인 후 생성 |
| `excel_v1` | `.xlsx`·`.xls`; 시트 하나, 연속 표 영역, 헤더 행 | 복수 시트 병합·매크로·수식 계산·외부 링크 갱신 없음. 캐시된 수식 결과만 쓰는 정책을 명시하고 캐시 없음/오래됨 가능성을 경고. 병합 셀 자동 채우기·색/서식의 의미 추정 없음. 암호화·지원 밖 구조는 다음 행동과 함께 거부 |
| `parquet_v1` / `feather_v1` | 단일 로컬 `.parquet`·`.feather`; 스키마와 컬럼 선택 | 디렉터리 dataset·원격 참조·암호화·임의 extension type 제외. 중첩 list/struct/map·binary 등 공통 계약 밖 컬럼은 명시적으로 제외하거나 지원 불가로 안내. 문자열로 조용히 변환하지 않음 |
| `json_records_v1` / `jsonl_records_v1` | `.json`의 레코드 배열 또는 명시 경로, `.jsonl`/`.ndjson`의 행별 객체 | 중복 키·비표준 숫자·과도한 깊이/긴 레코드 거부. 객체 경로는 키/인덱스 목록으로만 선택. 선택한 스칼라 필드는 한 행을 유지하며 배열 자동 explode·임의 flatten·테이블 자동 결합 없음. 없는 키와 명시 null의 차이는 해석 기록에 남김 |
| `dataframe_pickle_v1` | `.pkl`·`.pickle`의 DataFrame 하나 또는 명시 구조 경로 | 격리 전에 출처/신뢰 확인과 역직렬화 허용을 동의에 결합. 자동 형식 판별을 위해 Pickle을 열지 않음. 안전한 Pickle이라는 보장은 하지 않으며 dict/list 등 임의 객체의 강제 표 변환·압축 묶음은 제외 |

빈/중복/비문자열 컬럼명은 공통 계약으로 가져오기 전에 위치와 원래 이름을 보존한 매핑을 제시한다. 기존 CSV 중복 헤더 정비의 원본 불변·충돌 없는 번호·동의·이력을 유지하며, 다른 형식에서는 '헤더 뒤 바이트 동일' 대신 선택 표의 값·순서 보존 범위를 표시한다. 숫자 문자열과 큰 정수 ID는 자동 float 변환하지 않는다.

**CSV v2 기본 해석(이 설계의 제안값):** 최초 A 화면에 UTF-8/BOM·쉼표·첫 행 헤더·빈 필드의 결측 처리·제한된 자료형 판별 정책을 요약한다. 빈 필드는 따옴표 유무와 관계없이 기본 결측으로 해석하며, 빈 문자열 보존을 고르면 문자열로 유지한다. `NA`·`NULL` 등 추가 결측 토큰은 자동 적용하지 않고 명시 선택한다. 따라서 문자열/결측 보존은 원본 바이트의 모든 구분을 자동 유지한다는 뜻이 아니라 **확정한 해석 이후**의 의미 보존이다. 날짜 문자열·문자 true/false·선행 0이 있는 숫자 문자열은 기본 문자열로 유지한다. 정수는 범위 안에서 정수/nullable 정수로 보존하고 float를 경유하지 않는다. 실수는 허용 문법·범위·십진 표현의 round-trip을 검사하며 정밀도 손실 우려가 있으면 문자열로 유지하고 정비 후보로 알린다. 파일 전체의 타입/범위 검사를 마치기 전에 표본 일부의 추정만으로 확정하지 않는다. 구분자·인코딩 등이 기본과 다르면 후보를 제시하고 사용자 선택을 받는다. 기존 `read_csv` 기본 추론과 항상 같다는 호환 보장은 하지 않으며, 동등성 테스트는 동일 해석이 성립하는 입력에서 수행한다.

#### 공통 표형 묶음과 입력 계약

**저장:** 기존 `canonical_table_v1`은 그대로 읽고, 새 `canonical_table_v2`를 명시적으로 선택한다. 새 writer·reader·validator는 `canonical_data.py`의 기존 v1 동작을 바꾸지 않도록 별도 `tabular_bundle.py`에 둔다. v1으로 낮춰 v2의 해석/영향 검증을 우회할 수 없게 실행 계약이 저장 버전을 고정한다. 기본 구성은 다음과 같다.

- `schema.json`: 버전·행/열 수·컬럼 순서·논리/물리 자료형·인덱스 설명·스칼라 인코딩 버전. 컬럼에는 위치 기반 내부 ID와 표시 이름을 함께 둔다.
- `data.jsonl`: 명시 태그로 표현한 셀과 인덱스. 정수·불리언·빈 문자열·null·NaN·±inf·날짜를 구별한다. 범주 사전/순서와 시간대·시간 단위를 스키마로 보존한다. 임의 클래스 복원이나 동적 import는 없다.
- `provenance.json`: 원본 snapshot 지문, 로더/정책 버전, 해석 설정 서명, 선택 범위, 정비 전후 매핑, 손실/제외 내역, 행 위치 계보. 원본 경로는 로컬 출처 정보로만 관리하며 모델 DTO에는 넣지 않는다.
- `manifest.json`·완료 표식: 고정 허용 파일 목록과 각 파일의 해시·크기. 상위 작업/승인에 보존한 기대 묶음 ID와 대조하며 묶음 자체의 manifest만 신뢰하지 않는다. 부분 출력은 완료 결과로 등록하지 않는다.

**자료형:** 초기 필수 범위는 signed/unsigned 정수·float·nullable 정수/실수·bool/nullable bool·문자열·지원 스칼라만 든 object, 범주형, 날짜/시각/시간대 포함 timestamp다. 범주 사전과 timestamp 단위는 손실 없이 표현 가능한 경우에만 지원한다. Decimal·기간·복합 인덱스·중첩/임의 객체 등은 첫 계약의 필수 범위 밖으로 표시하고, 제외/명시 변환이 없으면 거부한다. 미래 확장을 이유로 기존 제한을 조용히 넓히지 않는다. 저장 가능한 자료형이라도 기존 변환기가 받지 못하면 계획 단계에서 해당 역할·처리 조합을 차단한다. 시각 자료형 보존이 통합 작업의 시간 분할 지원을 뜻하지 않는다.

**식별:** `TabularSourceContract`(제안)는 원본 snapshot, 로더 ID/버전, 해석 설정/정비 서명, canonical manifest 지문, 저장 계약 버전, 자원 한도를 결합한다. 동일 원본이라도 시트·헤더·결측 규칙·정비가 다르면 다른 입력이다. 후속 source ID는 원본 파일 해시만이 아닌 이 계약의 서명에서 만들고, 분할·내보내기·후보·적용·전달이 같은 계보를 대조한다. 로컬 원본 경로의 변경만으로 승인된 snapshot이 무효가 되지는 않는다.

**행 계보:** 파싱 후 최초의 행 위치를 기본 식별자로 두고 인덱스와 구분한다. 원본의 시트/레코드 위치와 정비 후 행 위치를 연결한다. 분할의 행 위치 목록으로 canonical 구간을 만들며 정렬·index 재설정·행 삭제가 숨겨지지 않게 한다. 임의 중복 행 제거와 자동 행 필터링은 입력 로더의 역할이 아니다.

**자원:** 기존 canonical 상한(2백만 행·4096열·1억 셀·JSON 레코드 1MiB)을 초기 상한으로 유지하되, 실제 허용치는 작업의 입력/출력/Job 메모리·시간 상한과 함께 더 작은 쪽을 적용한다. JSONL은 순차 입출력하고 계산 직전에 필요한 표를 복원한다. 현재 v1처럼 전체 행의 파이썬 목록과 DataFrame을 동시에 유지하는 복원은 피한다. 파서 선언 크기·ZIP 해제 크기·배열/문자열 길이·레코드 수를 검사하고 읽는 중에도 제한한다. 일부 라이브러리의 선할당은 Job 상한과 실패 후 정리로 방어하며 파일 크기만으로 처리 가능성을 보장하지 않는다.

#### 사용자 흐름·입력 정비·모델 경계

1. 기존 통합 작업 화면에서 파일과 목표를 받는다. 최초 동의 A는 로더 후보/버전·허용된 구조 검사·기본 해석 설정·'검증된 검사 결과가 이 설정과 정확히 같고 모호성/정비가 없을 때만 생성·진단까지 진행'하는 정책 버전·자원/코드 범위를 포함한다. snapshot 생성 후 inspect의 1회 실행 승인을 해당 byte 지문에 결합한다. Pickle은 별도 신뢰 확인 없이는 구조 검사도 하지 않는다.
2. inspect 결과가 A의 기본 해석과 정확히 맞으면, 서버가 결과 커밋/지문·스키마·허용 설정·정책·입력 계보를 검사하고 **확정 해석 서명과 inspect 증거를 포함한 별개의 materialize 승인**을 A에서 위임 발급한다. 같은 승인을 두 번 쓰지 않으며 사용자 확인 버튼은 늘리지 않는다. 기본과 다르거나 구분자/헤더/시트/레코드 경로가 모호하면 같은 화면의 `입력 해석 확인 필요`에서 필요한 항목만 묻는다. 제출 시 해석/정비 영향에 대한 새 A 개정을 기록한 뒤 새 승인을 발급한다. 정책에 없는 추정값으로 자동 생성하지 않는다. 이전 승인·실행은 보존하며 화면 rerun만으로 읽기·쓰기·모델 호출·동의 발급하지 않는다.
3. 숫자 문자열 등 정비 후보에는 대상 컬럼·성공/실패/빈 문자열 건수·변환 뒤 결측 증가·정밀도 손실 가능성을 로컬로 보여 준다. 그대로 유지, 엄격 변환(실패 시 중단), 실패를 결측으로 처리 중 지원된 선택을 명시한다. 변환 실패를 결측으로 만드는 선택은 사용자가 영향에 동의한 경우에만 새 복사본에 적용한다. ID의 선행 0·큰 정수·지역별 숫자 표기·빈 문자열을 임의로 해석하지 않는다.
4. 정비 결과를 검증한 뒤 해당 입력으로 진단을 갱신하고 기존 권장안·필수 질문·역할/분할/전처리 확인 화면으로 진행한다. 정비 변경은 이후 C·D의 입력과 계획을 무효화하므로 D 이후 정비로 되돌리지 않는다. Train 중앙값·범주 사전·clipping·스케일링 등 학습 기준은 입력 정비에 포함하지 않는다.
5. C에서 확정한 입력/역할/계획으로 분할 → Train·Validation 내보내기 → Train 기준 후보를 수행하고 D에서 확인한 후보만 적용·전달한다. Test는 기존 정책대로 fit·적용·전달하지 않고 분할 정보만 보존한다. 결과에는 원본 형식·해석/정비·역할·실제 처리·전후 비교·미처리·활용법을 공통 틀로 표시한다.

구조 검사는 로컬 전용이며 파서 선택·시트/경로 선택·정비 동의를 LLM에 맡기지 않는다. 모델에는 확정 입력의 기존 허용 컬럼명·집계만 보내며 원본 행·개별 셀·수식·경로·Pickle 객체·파일 내용은 추가하지 않는다. 넓은 표의 기존 상세도/토큰 예산과 전체 응답 검증을 유지한다. 모델 실패로 입력 로딩이나 규칙 기반 계획이 불가능해지는 구조를 만들지 않는다.

inspect는 완성 canonical을 만들지 않고 제한된 구조 후보·해석 후보·컬럼별 타입/변환 성공/실패/결측 집계·변환 영향만 검증 가능한 로컬 결과로 낸다. 지원 밖 타입은 값 자체를 직렬화하지 않고 타입/위치와 거부 코드를 알린다. 정비 영향을 정확히 세려면 선택 범위를 자원 한도 안에서 모두 검사하며, 표본 검사에 그쳤으면 추정임을 표시하고 실제 생성 단계에서 다시 전수 검증한다. 설정이 바뀌면 inspect를 새 해석으로 다시 수행한다. materialize는 같은 검증 snapshot을 다시 읽고 확정 선택/제외/정비만 적용해 v2를 생성한다. 정비 전 임시 표나 지원 밖 객체를 부모/작업 기록에 보관하지 않는다. 결정/제출은 작업 revision·해석 시도 번호·snapshot·inspect 결과 지문에 결합하며 중복 제출과 오래된 화면은 새 실행 없이 거부한다.

#### 실행 경로 전환·호환성·변경 위치

새 형식마다 contract/CLI/transport/worker/results 전체를 복제하지 않는다. 로더와 canonical 생성은 표형 입력 도구 하나의 고정 action(`inspect`·`materialize`)으로 처리하고, 기존 승인 저장소·상태·AppContainer/Job 운송을 재사용한다. 후속 작업은 기존 서비스의 계산/증거 흐름에 버전 있는 표형 입력을 연결한다. 기존 CSV 공개 ID·기록을 새 의미로 재해석하지 않으며 새 통합 작업은 `tabular_v2` 어댑터와 정적 등록된 서비스 버전으로 연결한다. 모듈마다 버전 전환은 필요하지만 조정기나 전처리 계산의 두 번째 구현을 만들지는 않는다.

프로파일 없는 진단은 v2의 고정 진단 계획에만 허용한다. 기존 일반 EDA 계약의 profile 필수 검사를 완화하지 않고, worker는 새 계획 전체·서명·입력 계약 버전을 재대조한다. 결과의 컬럼 ID/순서는 승인한 canonical 스키마에 대조하며 CSV 헤더를 열어 대조하던 검사를 생략하지 않는다. v2의 typed 입력을 읽는 분기와 구 CSV 분기를 계약 버전으로 명시 구분한다.

| 변경 영역 | 구현 방향 |
| --- | --- |
| 신규 `tabular_bundle.py`, `tabular_loaders.py`, `tabular_source_contract.py` | v2 저장/복원/검증, 고정 로더 등록부와 허용 설정, 승인·출처 결합 |
| 표형 입력 도구(service/worker/results와 transport 연결) | inspect/materialize·모호성 결과·제한/정리. 기존 운송/승인 확장 지점을 재사용하고 공유 코드는 적합할 때만 추출 |
| `tabular_task.py`, `tabular_options.py`, 프로파일/계획 연결 | 로더/정비와 표형 계산 분리, v2 어댑터/프로파일 ID 연결. 구기록은 기존 ID로 읽음 |
| 진단·분할·내보내기·후보·적용의 contract/worker/results | CSV reader를 버전별 표형 reader 경계로 정리. 내부 canonical 구간 유지, 기존 v1의 값/순서 규칙 보존. 결과 계약에 출처·입력 버전 추가 |
| `csv_workflow_*`, handoff/binding/서비스 등록부 | 기존 CSV 호출의 호환 wrapper를 유지하고 새 표형 서비스 버전 등록. `csv:` 고정 source ID·필드 검사·파일명 가정을 명시 분리 |
| `task_orchestrator.py`, `task_store.py`, `task_adapters.py` | 공통 입력 해석 대기/재개와 계약 참조. 파일 형식별 분기는 어댑터 내부. 기존 작업 기록은 버전별로 검증하고 읽을 때 자동 변환하지 않음 |
| `task_view.py`, `task_forms.py`, 표형 form, `path_picker.py`·`path_dialog.py` | 확장자 안내·동일 화면 해석 질문·정비 영향·지원 불가 안내. 유형마다 새 메뉴/전체 화면은 만들지 않음 |
| requirements·런타임 복사 목록 | 아래 엔진과 전이 의존·DLL·메타데이터를 검증된 조합으로 추가. 설치만으로 운영 지원 표시하지 않음 |
| 결과 보고/로컬 전달 | 전처리된 수치 특징 CSV는 기존 소비자를 위해 유지 가능. 인덱스·라벨·그룹·출처의 자료형은 검증된 부속 계약으로 보존하고 로드 예제 제공. 내부 입력을 CSV로 되돌려 계산하지 않음 |

원본 snapshot은 계속 단일 파일이다. canonical은 외부 경로를 사용자가 직접 신뢰시키는 입력이 아니라 worker가 만든 검증된 결과 묶음이며, transport가 허용 파일만 stage한다. 기존 snapshot 계약에 디렉터리를 억지로 넣지 않는다. 부모는 링크·허용 파일·크기·해시·스키마·행/열 수·타입·계보·승인/계획 서명을 검사한다. worker가 주장한 원본 대비 통계·값 동등성을 부모 검증이 독립 증명했다고 표시하지 않는다.

기존 CSV와 배열 테스트는 기존 경로에서 유지한다. 새 CSV v2 경로는 기존 합성 입력과 역할/계획을 고정해 결과 동등성을 검사하며, 자료형 보존으로 의미가 달라지는 사례는 자동 호환으로 처리하지 않고 차이를 기록한다. 기존 실행을 읽기만 해 자동 마이그레이션하거나 재실행하지 않는다. 코드 지문 대상의 조정기·등록부·worker·런타임을 변경하므로 CSV·모듈 파이프라인과 배열/WM 일부에도 영향 가능성이 있다. 구현 전후 각 code manifest를 실제 비교해 범위를 보고하고, 미실행 승인 갱신과 완료 결과 조회 영향을 구분한다. '문서/로더 변경이라 CSV만 영향'으로 가정하지 않는다.

#### 라이브러리·복수 입력·제한

- 조사한 `wm-gpu312` 환경에는 pandas 2.3.3·PyArrow 26.0.0이 있고 `openpyxl`·`xlrd`는 없었다. 현재 requirements와 격리 `runtime_files`의 고정 패키지 목록에는 Excel 엔진과 PyArrow가 선언돼 있지 않다. 제안 엔진은 XLSX=`openpyxl`, XLS=`xlrd`, Parquet/Feather=`pyarrow`다. 호환 버전은 구현 시 설치 승인 후 환경·DLL·라이선스·격리 import 및 합성 round-trip으로 확정한다. 현재 런타임 복사 상한 512MiB를 포함해 패키지 추가의 크기/시작 시간 영향을 측정하며 상한을 몰래 확대하지 않는다. Arrow의 임의 extension type·사용자 extension 복원·Python 객체 역직렬화를 비활성/거부하고 파일 metadata로 사용자 코드를 복원하지 않는다. 라이브러리 부재는 설치 안내로 표시하고 비격리 fallback·자동 다운로드하지 않는다.
- Excel의 시트·헤더·자료형 옵션과 Arrow/Pandas 간 nullable·범주·시간 변환은 [pandas 2.3 read_excel](https://pandas.pydata.org/pandas-docs/version/2.3/reference/api/pandas.read_excel.html), [Arrow Pandas integration](https://arrow.apache.org/docs/python/pandas.html)을 기준으로 검증한다. 원본 엔진의 모든 자료형을 지원한다고 약속하지 않는다.
- SECOM의 특징 `.data`와 라벨/시각 파일처럼 두 파일인 입력은 단일 `.data` 파싱과 별개다. 확장 마지막 단계에서 선택적 `feature_metadata_pair`를 제공한다. 두 파일을 각각 snapshot으로 고정하고 이름·해시·행 수·컬럼 매핑·결합 방식(명시적인 행 위치 일대일)을 계약에 묶는다. 행 수 일치는 의미 정렬의 증명이 아니므로 같은 행 순서라는 사용자 확인과 보고서 경고를 남긴다. 임의 key join·폴더 수집·행 증식·자동 라벨 추정은 범위 밖이다.
- 복수 입력을 구현할 때만 공통 source descriptor를 단일/명시 파일 목록으로 확장한다. 기존 단일 snapshot 읽기는 호환 경로로 유지한다. 입력을 추가하면 A를 새로 받고 C·D를 재사용하지 않는다. 기존 SECOM 독립 실행 코드를 통합 조정기에 직접 연결하거나 변경하지 않는다.
- 입력 형식 지원과 시간 기준 분할은 별도 기능이다. 시각 데이터의 보존·제외·메타데이터 사용과 지원하지 않는 timestamp 역할/분할 조합을 명확히 안내한다. SQL/HDF5/HTML/XML·ZIP 입력·클라우드·이미지·웨이퍼 연결·모델 학습은 이번 범위 밖이다.

#### 구현 순서·완료 조건

| 단계 | 범위 | 해당 단계의 완료 근거 |
| --- | --- | --- |
| T1 공통 기반과 CSV 전환 | v2 저장/입력 계약, CSV 로더, 진단부터 분할·후보·적용·전달까지 연결 | 기존 CSV 경로의 테스트 유지, 새 CSV v2의 입력→최종 결과, 자료형/계보/승인·변조 거부 |
| T2 구분자·JSON | TSV/TXT/DATA/DAT·JSON/JSONL, 입력 해석 질문, 숫자 문자열 등 명시 정비 | 헤더 없는 표·모호한 구분자·중첩 선택·TotalCharges형 합성 사례, 정비 영향과 손실 확인 |
| T3 Excel·열 지향·Pickle | XLSX/XLS·Parquet/Feather·DataFrame Pickle, 엔진/격리 런타임 확장 | 각 형식의 정상 끝단 흐름·지원 밖 타입/악성 입력 거부, 실제 AppContainer에서 파서 import·계산·정리 |
| T4 복수 입력 | 특징/메타데이터 두 파일의 명시 일대일 결합 | 합성 SECOM형 두 파일, 행 수/변조/매핑 오류 거부, 라벨·시각·행 계보 보존 및 최종 전달 |

T1만 끝내고 모든 형식이 지원된 것으로 표시하지 않는다. T2~T4는 후속 업데이트로 보류하며 현재 마감 범위에 포함하지 않는다. 재개는 별도 사용자 지시를 따른다. 특정 엔진 설치와 실제 데이터/모델/OS 실행은 각각 허용 범위를 확인한다.

공통 검증은 다음을 포함한다.

- 고정 합성 표의 형식별 값·자료형·결측·빈 문자열·큰 정수·행/열 순서·범주·시각 보존. 입력 표현이 같은 의미인 범위에서 동일 분할과 Train 기준/출력의 동등성을 대조하고 손실 있는 변환은 별도로 검증한다.
- 정상 흐름뿐 아니라 모호성·헤더 충돌·잘못된 인코딩·수식/캐시·손상/과대 압축·과대 행·악성/위장 입력·라이브러리 없음·변조/링크·다른 actor·만료/동시 제출·오래된 해석/계획/후보의 거부. 실패 시 자동 파서 전환·부분 결과 등록·무단 재실행 없음.
- 동의 A(해석/정비), C(입력·역할·분할~후보), D(후보·적용·전달)의 서명 결합과 원본 불변. 파서 설정의 임의 옵션·사용자 코드·외부 참조를 차단한다. 보안 영향은 SEC-01·02·04·05·06·07·08·09·10·11·12·13·14·15·16을 실제 변경 경로로 대조한다.
- 기존 CSV·배열 회귀와 공통 계층의 유형 비의존 검사. 실제 OS 격리/egress는 모의 테스트와 분리해 해당 실행의 증거 범위로 보고한다. 독립 보안 검토와 실제 관리자 UI 정상 흐름을 통과해야 사용자 흐름 완료로 표시한다. 전체 회귀는 사용자 요청/동의 후 실행한다.
- 보고서에 입력 해석·정비 영향·문제·실제 처리·예상/실측·제외/미처리·파일 활용법이 있으며, 차단 시 원인과 같은 화면의 수정 경로가 있어야 한다. 모델 권장안 실패와 전처리 완료를 구분한다.

**현재 마무리 범위:** 작업 8의 실제 사례·선별 검증과 작업 9의 문서 정리를 기준으로 안정화한다. 작업 8의 관련 합성 검증은 기준 커밋 전체 회귀로 확인했으며, 개별 명령을 중복 실행하지 않는다. 실제 선택 검사와 모델·실데이터·대용량의 미검증 범위는 별도로 유지한다. 작업 1~7은 공통 기반·v2 서비스·v2 어댑터·통합 화면 연결까지 구현했고 기존 운영 경로·라이브러리·실제 데이터·운영 승인은 변경하지 않았다. T2~T4와 웨이퍼 통합 연결을 T1 구현에 섞지 않는다.

#### T1 공통 표형 기반·CSV 전환 구현 계획

> **구현 담당자:** 설치된 `superpowers:executing-plans`로 아래 체크 항목을 순서대로 수행한다. 기본은 한 구현자가 진행하며 경계 변경의 독립 보안 검토를 별도로 적용한다. 사용자가 지정한 대로 각 작업 완료 후 결과를 보고한다. 병렬 구현·작업마다 커밋은 이 계획에 포함하지 않는다. 체크되지 않은 항목은 미수행이다.

**목표:** UTF-8 CSV 하나를 기존 통합 화면에서 받아 승인된 v2 표형 묶음으로 고정하고 진단 → 계획 확인 → 분할/내보내기 → Train 후보 → 적용/전달 → 내용 있는 보고서까지 연결한다.

**구조:** 기존 계산과 실행 서비스를 계약 버전으로 확장한다. 원본은 `file_snapshot_v1`, 내부 표는 `canonical_table_v2`, 작업 유형은 `tabular_v2`로 구분한다. 기존 `tabular_csv`·CSV v1·배열 기록은 기존 해석으로 검증한다. 새로운 형식의 엔진이나 다른 전처리 계산 엔진은 추가하지 않는다.

**기술:** Windows Python 3.12·현재 numpy/pandas·unittest·Streamlit·기존 RulesExecutor/ApprovalStore/AppContainer/Job. T1은 새 패키지 설치가 필요 없다. **기준:** 바로 위 공통 표형 상세 설계와 [현재 검증 상태](GUIDE.md).

**공통 제약**

- T1 로더는 UTF-8/BOM·쉼표·첫 행 헤더의 CSV만 활성화한다. 인코딩/구분자 자동 전환·헤더 없는 입력·숫자 문자열의 사용자 변환·다른 확장자는 T2 이후다. 기본 빈 필드 결측/문자열 보존 선택과 기존 중복 이름 번호 정비는 T1에서 유지한다.
- 저장 상한은 2백만 행·4096열·1억 셀·JSON 레코드 1MiB이며 실제 실행은 더 엄격한 단계별 한도를 따른다. 현재 `split_evidence.MAX_SPLIT_ROWS`는 1백만 행이므로 T1의 분할 가능한 전체 흐름도 1백만 행 이하로 제한해 진단/계획에서 안내한다. 저장 가능한 행 수와 처리 가능한 행 수를 혼동하거나 기존 한도를 몰래 늘리지 않는다.
- 정수는 부호/범위를 보존하고 bool과 구분한다. 모든 결측인 컬럼은 임의 수치형으로 확정하지 않는다. 빈 문자열 보존 모드의 문자열은 수치 역할을 자동 허용하지 않는다. 날짜 문자열은 자동 timestamp로 바꾸지 않는다.
- 구조 검사와 생성은 별도 1회 승인이다. A의 정확한 기본 정책과 일치할 때만 materialize를 위임 발급한다. 수정은 새 A 개정, 분할/후보는 C, 적용/전달은 D에 결합한다. 반복 제출·만료·다른 사용자·취소·실행 중 코드 변경 시 기존 동의로 진행하지 않는다.
- Test는 기준 산출·변환·전달에 쓰지 않는다. 모델 전송 계약은 확대하지 않는다. 데이터 내용 파싱은 격리 worker에서 하고 부모는 검증된 결과만 검사한다. 기존 보안·회귀 기대값을 느슨하게 바꾸지 않는다.
- 테스트 실행 중 대상 파일을 수정하지 않는다. 각 작업의 실패 재현 → 구현 → 통과 확인 뒤 다음 작업으로 이동한다. 새 코드와 연결되기 전 신규 테스트의 import 실패는 초기 RED로만 기록하고, 최종 검증에는 실제 동작/거부 단언이 있어야 한다.

**중점 검토 사례와 담당 작업**

| 입력/실패 상황 | 기대 동작 | 작업 |
| --- | --- | --- |
| 선행 0 ID·2^53 초과 정수·bool·문자열/빈 필드 혼합 | 임의 float/string 변환 없이 확정 해석대로 보존 | 1·2·5 |
| 중복 헤더와 원래 `name_1`이 이미 있는 CSV | 충돌 없는 번호와 원래 위치 매핑, 동의 없는 변경 없음 | 2·3 |
| 같은 파일에서 결측 정책 변경·오래된 화면 재제출 | 원본 해시가 같아도 다른 입력으로 판정, 이전 A/C/D 재사용 거부 | 3·7 |
| 분할 후 불연속 행 위치·중복 원본 인덱스 | 행 위치와 인덱스를 혼동하지 않고 라벨/그룹/ID 정렬 보존 | 4·5·6 |
| materialize 뒤 저장 실패·단계 timeout·출력 파일 바꿔치기 | 부분 결과 미등록, 승인 자동 재사용/worker 재실행 없음 | 3·6·8 |

**파일 배치와 공통 인터페이스(아래 신규 이름은 T1에서 정의)**

| 위치 | 책임/변경 |
| --- | --- |
| 신규 `src/preprocessing/tabular_bundle.py` | 순수 자료형 `BundleRef`, v2 스키마·태그·행 스트림·분할 부분집합·복원. 기존 `canonical_data.py` v1은 유지 |
| 신규 `src/agents/tabular_source_contract.py` | `TabularSourceContract`·`TabularEdaInputContract`·`TabularPartitionInputContract`; 기존 CSV 계약을 느슨하게 바꾸지 않는 명시 타입. `BundleRef`와 `CsvReadSpec`는 preprocessing의 순수 타입을 import |
| 신규 `src/preprocessing/tabular_loaders.py` | 순수 자료형 `CsvReadSpec`, 고정 CSV 로더·허용된 해석 설정·전수 타입/결측/중복 검사 |
| 신규 `src/agents/tabular_input_{service,worker,results,transport}.py` | inspect/materialize 도구. 기존 실행기·운송 구성요소 재사용, 별도 승인/작업 엔진 없음 |
| 신규 `src/agents/tabular_v2_task.py`, `tabular_workflow_service.py` | 새 계약을 기존 어댑터/서비스 API에 연결. `tabular_task.py`의 공통 표시/요약만 필요한 만큼 추출·공유하며 파일 전체 복제 금지 |
| 기존 진단·분할·구간·후보·적용·handoff 계약/worker/results/CLI/transport | 계약 버전별 읽기/staging과 부모 검증. 아래 작업에 변경 대상을 구분 |
| 기존 공통 조정기·저장소·등록부·UI | 입력 해석 대기/재개, 새로운 작업 생성만 v2 선택, 구기록 재해석 방지 |

타입은 불변 dataclass로 정의하고 strict JSON 왕복을 제공한다. `BundleRef`는 저장 버전·manifest 지문·전체 바이트·행 수·컬럼 수를 갖는다. preprocessing은 agents를 import하지 않는다. `TabularSourceContract`는 원본 snapshot 기대값·로컬 묶음 위치·`BundleRef`·로더/정책 버전·확정 `CsvReadSpec` 서명·inspect 증거 ID·생성 커밋 증거·한도를 갖는다. `source_id`는 이 내용 중 경로를 제외한 서명이며 `tabular_source:<sha256>` 형식이다. 로컬 위치를 worker/모델에 그대로 전달하지 않고 승인 후 고정 폴더로 stage한다. `BundleRef`만으로 커밋 증거나 승인을 대신할 수 없다.

`TabularEdaInputContract(source, profile_json, limits)`는 일반 EDA/분할용이며 프로파일 필수다. 입력 도구와 고정 진단은 별도 계약으로 프로파일 없이 실행한다. `TabularPartitionInputContract(split, export, partition, bundle, roles_json, parent_source_id)`는 `train`/`validation`만 허용하며 확정 분할·커밋된 export·원래 행 위치에 결합한다. v2 후보/적용은 이 타입을 명시 허용하고 기존 v1 타입의 필수 검사를 그대로 유지한다.

**작업 1 — v2 저장과 출처 계약**

파일: 신규 `tabular_bundle.py`·`tabular_source_contract.py`, 신규 `tests/agents/test_tabular_bundle.py`·`test_tabular_source_contract.py`.

인터페이스: `write_bundle(frame, directory, *, provenance, max_output_bytes, row_positions=None) -> BundleRef`, `inspect_bundle(directory, *, expected, max_bytes) -> BundleRef`, `iter_bundle_rows(directory, *, expected)` → `(source_position, original_index, values)` 반복자, `load_bundle(directory, *, expected, columns=None) -> DataFrame`, `write_subset(directory, positions, destination, *, expected, provenance, max_output_bytes) -> BundleRef`. `expected`는 생산 중 신규 결과를 검사할 때만 None을 허용하고, 후속 소비/승인 경로에서는 필수다. 저장 버전·날짜/시간 단위·시간대·범주 사전·순서·결측 태그는 위 v2 계약을 따른다.

- [x] nullable 정수/boolean·`9007199254740993`·`'0012'`·빈 문자열·None/pd.NA/NaN·±inf·순서 있는 category·시간대 timestamp·중복 인덱스의 복원 단언을 먼저 작성한다. `pd.testing.assert_frame_equal(restored, original)`과 원래 행 위치 `[0, 1, 2]`를 대조한다. 잘못된 태그·미지원 Decimal·자료형/행 수 불일치·v1을 v2로 위장·초과 레코드/출력 크기는 거부한다.
- [x] `python -X utf8 -B -m unittest -v tests.agents.test_tabular_bundle tests.agents.test_tabular_source_contract`로 실패를 확인한다.
- [x] 위 인터페이스와 원자적 완료 기록을 구현한다. manifest는 `schema.json`·`data.jsonl`·`provenance.json`의 해시/크기를 담고 완료 표식은 manifest 지문을 담는다. BundleRef가 완료 표식/manifest를 외부에서 고정해 순환 해시를 만들지 않는다. 작은 청크의 컬럼별 버퍼로 복원해 전체 행 객체 목록과 표를 동시에 유지하지 않는다.
- [x] 같은 명령으로 통과를 확인한다. 생성 결과의 자기 일관성 검사를 원본과의 독립 동등성 증명으로 표현하지 않는다.

현재 구현은 `tabular_bundle.py`·`tabular_source_contract.py`와 해당 테스트다. 소비할 데이터는 비공개 임시 스트림에 순차 복사해 기대 해시/크기를 확인한 뒤 최초 행을 반환한다. 경로의 읽기 전후 해시만 비교해 실제 소비 바이트가 달라지는 경우를 허용하지 않는다. Python 저장 문자열·스칼라 object·명시 범주/시각을 지원하며 Arrow 저장 문자열과 미지원 extension/Decimal/다중 인덱스는 명시 거부한다. 실제 OS 격리·대용량 성능·운영 실행 연결은 후속 작업에서 검증한다.

**작업 2 — CSV inspect/materialize 계산**

파일: 신규 `tabular_loaders.py`와 `tests/agents/test_tabular_loaders.py`, `tabular_source_contract.py`의 해석 검사 위임. 기존 `input_profile.py`의 `dedupe_names`를 import해 재사용하며 해당 파일 자체는 변경하지 않는다.

인터페이스: `inspect_csv(snapshot_file, spec: CsvReadSpec, *, limits, max_input_bytes) -> dict`, `materialize_csv(snapshot_file, spec, directory, *, expected_snapshot, expected_inspection, max_output_bytes) -> BundleRef`. `CsvReadSpec`의 T1 허용값은 loader=`delimited_v1`, encoding=`utf-8-sig`, delimiter=`,`, header=`first_row`, empty_field=`missing|preserve`, header_repair=`reject|number_duplicates`; 나머지 키는 거부한다. 타입 추론 정책 ID를 고정하고 개인 converter·정규식·표현식·임의 pandas 옵션은 받지 않는다.

- [x] `test_default_and_preserve_empty_fields`에서 `a,b\n,NA\n1,0012\n`의 기본 첫 셀은 결측, `NA`·`0012`는 문자열임을 확인하고 preserve 모드의 첫 셀은 `''`임을 확인한다. 전체가 빈 컬럼·아주 큰 정수·긴 소수·혼합 타입·잘못된 UTF-8·행 너비 변화·CSV 따옴표/줄바꿈도 고정 정답으로 검사한다.
- [x] `test_duplicate_repair_requires_exact_spec`에서 `zero,zero,zero_1`의 미승인 정비 거부와 승인된 충돌 없는 이름/위치 매핑을 검사한다. 모호한 입력은 고정 코드와 후보만 반환하고 다른 파서로 자동 재시도하지 않는다. inspect에는 원본 행/실패 값이 포함되지 않아야 한다.
- [x] `python -X utf8 -B -m unittest -v tests.agents.test_tabular_loaders tests.agents.test_input_profile`로 RED를 확인한다.
- [x] inspect는 snapshot 전체를 제한 안에서 검사해 컬럼별 타입·성공/실패/결측 건수와 `inspection_id=해시(snapshot+spec+policy+결과)`를 생성한다. materialize는 동일 snapshot을 다시 읽고 inspect ID·해석 서명·형상·타입을 재대조해 작업 1 writer로 기록한다. 수치 변환 사용자 정비와 임의 행 삭제는 T1에 추가하지 않는다.
- [x] 같은 명령으로 통과를 확인하고 원본 바이트 불변·두 번째 실행의 덮어쓰기 거부를 검사한다.

현재 구현의 `CsvReadSpec`는 순수 불변 타입이며 부모 계약도 같은 타입으로 해석을 검사한다. 검사 결과 `csv_inspection_v1`에는 확정 spec·자원 요청·실제 읽은 snapshot 지문·타입 추론 정책·행 수·컬럼별 dtype/건수·변경된 이름 매핑·고정 상태/코드·inspection ID를 담는다. 원본 행·실패 값·경로는 넣지 않는다. numeric 성공/실패는 셀별 수치 표현 가능 건수이며 컬럼 전체가 수치형으로 확정됐다는 뜻은 아니다. 큰 정수와 실수의 혼합으로 정밀도가 손실되거나 공통 정수 범위가 맞지 않으면 컬럼을 문자열로 유지한다.

생성 단계는 비공개 임시 스트림에 고정한 실제 소비 바이트를 기대 snapshot과 대조하고, 전체 재검사 결과의 canonical JSON을 기대 결과와 비교해 bool/int/float의 묵시적 동등성을 허용하지 않는다. 중복 이름 번호 충돌은 `CSV_HEADER_REPAIR_CONFLICT`, 입력 I/O 실패는 경로 없는 고정 코드로 처리한다. 다른 구분자를 추정해 파서를 자동 전환하지 않으며 명시한 쉼표 해석의 정상 컬럼명 안에 다른 구분자 문자가 있다는 이유만으로 차단하지 않는다. 파일/레코드/행/열/셀 제한과 파싱 중 deadline은 계산 모듈에서 확인하지만 OS 메모리·프로세스 강제와 실제 격리 실행은 작업 3 이후 범위다.

**작업 3 — 입력 도구와 A 승인·입력 해석 재개**

파일: 신규 `tabular_input_service.py`·`tabular_input_worker.py`·`tabular_input_results.py`·`tabular_input_transport.py`·`tabular_input_flow.py`, 수정 `task_orchestrator.py`·`task_store.py`와 개발 중인 v2 출처/저장/로더 계약; 신규 `tests/agents/test_tabular_input.py`·`test_task_input_interpretation.py`.

인터페이스: `TabularInputService(action)`은 기존 서비스의 `create/status/preview_approval/record_delegated_approval/next/results` 규약을 따른다. action은 `inspect|materialize`만 허용하고 stage는 둘 다 기존 증거 도구가 다룰 수 있는 `ingest`로 하되 module ID는 `tabular_inspect|tabular_materialize`로 분리한다. `read_inspection(directory, expected_request)`와 `read_materialization(directory, expected_request)`는 부모에서 요청/출력 결합을 검사한다. 공통 조정기에 `resume_input(task_id, selection, *, expected_revision, expected_inspection_id, root=None, requester=None)`를 추가한다. 선택 객체는 어댑터가 검사하며 공통 조정기는 CSV 필드를 해석하지 않는다.

- [x] 입력 해석 대기에서는 모델 호출·materialize·C/D 발급이 0임을 검사한다. 정상 기본 입력은 A 한 번의 사용자 확인으로 서로 다른 inspect/materialize 실행 승인 두 개를 소비한다. 다른 spec·다른 snapshot·취소·만료·중복 제출·stale revision/inspect ID는 새 동의/실행을 만들기 전에 거부한다.
- [x] materialize 실행 성공 뒤 작업 기록 저장 실패를 주입해 자동 재실행이 없고 원본·구실행·승인 소비 기록이 보존됨을 검사한다. 링크/추가 파일/완료 표식/해시 변조와 불완전 출력은 등록하지 않는다.
- [x] `python -X utf8 -B -m unittest -v tests.agents.test_tabular_input tests.agents.test_task_input_interpretation`로 RED를 확인한다.
- [x] 기존 ApprovalStore·RulesExecutor·상태·위임 기록·AppContainer/Job을 재사용한다. worker는 `-I -S -B` 복사 런타임에서 작업 2를 호출하고 부모 모듈/승인 DB에 의존하지 않는다. 새 transport는 기존 고정 복사/ACL/Job 구성요소를 이용하며 새로운 비격리 fallback은 만들지 않는다.
- [x] `awaiting_input` phase와 `input_selection` 기록(시도·inspect 참조·해석 서명·동의 이력)을 추가한다. 어댑터가 이 상태로 돌아오면 공통 `diagnose/resume_input`은 `_recommend`를 호출하지 않고 반환하며 materialize/진단 커밋 뒤에만 모델 단계로 간다. A 기본 정책의 정확한 일치 조건으로만 materialize를 승인하고, selection 변경은 새 inspect와 A 개정으로 수행한다. 큰 inspect 본문은 기존 parts 저장을 이용하고 새 구역을 분리하면 `PART_SECTIONS`·`PART_NAME`·크기 검사를 함께 확장한다. 64KiB IPC에는 해시/파일 참조만 넣는다. 원본 snapshot 검사는 유지하고 소비 직전 canonical 참조도 재검사하는 어댑터 hook을 추가한다.
- [x] 위 대상과 기존 `tests.agents.test_task_orchestrator`·`tests.agents.test_array_task`를 실행해 구경로가 유지되는지 확인한다.

입력 준비 정책은 `tabular_input_flow`가 맡고 공통 조정기는 유형 이름 없이 대기/재개 hook만 호출한다. 생성의 설정 지문은 worker packet 전체를 기준으로 하며 snapshot·확정 spec·검사 ID/파일 해시·코드·정책·자원·출력 상한을 포함한다. 개발 중인 `TabularSourceContract`에도 `inspection_sha256`·`max_output_bytes`를 필수로 추가했고, provenance의 검사 파일 해시를 대조한다. 입력 준비 흐름의 기본 총 출력 상한은 64MiB, 검사 sidecar 상한은 16MiB다. `workflow_approval.py`는 부모 코드 지문에 포함하되 worker에는 복사하지 않는다.

합성 어댑터에서 입력 준비 흐름을 검증했으며 후속 진단·모델 단계는 대역이다. 별도 Python `-I -S -B` 프로세스에 허용 소스만 복사해 계산 성공과 부모 서비스/승인 DB 모듈 미사용을 확인했다. 작업 3의 단독 검증은 합성·모의 기준이며, 이후 실제 AppContainer·ACL·Job·입력 UI의 확인 범위는 작업 8과 GUIDE를 따른다. worker는 부모가 검증하고 권한을 준 폴더 안의 경로를 조회하며, 부모의 전체 경로/링크 검사는 유지한다. v2 어댑터·화면 연결은 작업 7에서 구현했다. 독립 보안 재검토에서 지문 누락 보완 후 추가 확정 결함은 없었으며 검증 수치는 [GUIDE](GUIDE.md)에 둔다.

**작업 4 — v2 고정 진단·분할·canonical 구간 내보내기**

파일: 수정 `input_profile_pipeline.py`·`module_pipeline_worker.py`·`module_pipeline_results.py`·`pipeline_binding_registry.py`, `snapshot_split_cli.py`·`snapshot_split_worker.py`·`split_evidence.py`, `partition_export_cli.py`·`partition_export_contract.py`·`partition_export_worker.py`·`partition_export_results.py`와 해당 transport 목록; 신규 `tabular_worker_input.py`·`tabular_pipeline.py`·`tabular_split.py`·`tests/agents/test_tabular_split_export.py`. 기존 `snapshot_split.py`·`partition_export.py`의 승인·실행·커밋을 v2 도구가 상속하며 해당 두 본문은 변경하지 않았다.

인터페이스: `bind_tabular_diagnostic(source: TabularSourceContract, limits)`는 v2 고정 계획만 결합한다. `snapshot_split_cli.create_from_tabular(name, source, profile, limits, *, confirmed=False)`와 `partition_export_cli.create_from_tabular(name, split_run, source, limits, max_partition_bytes, *, confirmed=False)`를 추가하고 기존 create 인자는 유지한다. `read_confirmed_split`은 같은 `ConfirmedSplit` API를 유지하되 계획 버전에 맞는 source ID와 packet을 검사한다. v2 export 결과는 구간별 BundleRef·원본 행 위치·부모 source ID를 담는다.

- [x] 고정 합성 표에서 원래 인덱스가 `[8,8,3,2,...]`여도 분할은 행 위치로 수행되고 구간을 다시 읽으면 정확한 라벨/그룹/ID가 따라오는지 검사한다. Test canonical/값 파일이 생성되지 않는지, group/stratify·품질 review_required·빈 구간·1백만 행 초과 사전 차단을 검사한다.
- [x] 일반 계획의 profile 없음·v2 진단에 단계 추가·v1 CSV 파일을 canonical로 위장·원본 해석 서명 변경·분할과 export의 source 불일치를 거부한다. worker가 고정 계획 전체와 서명을 재계산하는지 검사한다.
- [x] `python -X utf8 -B -m unittest -v tests.agents.test_tabular_split_export tests.agents.test_snapshot_split_execution tests.agents.test_partition_export_execution tests.agents.test_input_profile_pipeline`로 RED를 확인한다.
- [x] v2 분기에서만 작업 1 reader와 기존 `profile_input`·`inspect_quality`·EDA·`propose_split/validate_proposal`을 호출한다. 구분자가 아닌 계약 버전으로 dispatch한다. `csv:` source ID, UTF-8 헤더/텍스트 강제 dtype, CSV row writer는 v1 경로에서 유지하고 v2에는 적용하지 않는다.
- [x] 부모 검증은 CSV 헤더 대신 승인된 canonical 스키마의 ID/이름/순서·타입·행 수에 대조한다. 구간 파일은 `write_subset`으로 만들고 분할 증거·행 위치·출력 커밋과 결합한다. 기존 v1 공개 계약의 필수 조건을 제거하지 않는다.
- [x] 같은 명령과 `tests.agents.test_generic_split`·`tests.agents.test_generic_preprocessing`을 실행해 통과를 확인한다.

v2 입력 helper만 worker에 복사하며 생성 승인 DB·부모 출처 계약·도구는 복사하지 않는다. 진단은 기존 단일 고정 계획, 분할은 기존 품질/EDA/제안/확정, 내보내기는 기존 ingest 실행기로 진행한다. `TabularExportContract`가 동일 생성 출처·프로파일·v2 확정 분할·상한을 결합한다. 확정 증거를 다시 읽을 때도 선행 제안의 실제 승인·커밋·완료 표식과 확정 승인 인자를 대조한다. 이 선행 제안 읽기는 `tabular_split_v2` 정책에만 허용하며 기존 v1 증거 검사는 유지한다.

부모는 진단의 컬럼 순서와 스키마에서 확정 가능한 분류, 진단·EDA·제안·확정의 행 수를 승인 묶음과 대조한다. category 등 분류가 모호한 경우 추정하지 않는다. 구간은 컬럼 스키마·행 위치·provenance·기대 해시를 검증하며 원본 대비 모든 통계·변환 값의 독립 정답 인증을 뜻하지 않는다. 정상·변조 합성 검사와 허용 파일만 둔 별도 Python `-I -S -B`의 진단/분할/내보내기를 확인했다. 작업 4 자체 검증과 이후의 실제 AppContainer·UI·모델 요청 확인은 구분하며 현재 상태는 GUIDE를 따른다. 정식 서비스·어댑터 연결은 작업 6~7에서 구현했다. 독립 보안 재검토에서 복사 의존 누락과 기대 행 수 대조 누락을 보완한 뒤 추가 확정 결함은 없었다. 검증 수치는 [GUIDE](GUIDE.md)에 둔다.

**작업 5 — typed 구간의 Train 후보·Validation 적용**

파일: 수정 `partition_input_contract.py`, `preparation_candidate_contract.py`·`preparation_candidate.py`·`preparation_candidate_cli.py`·`preparation_candidate_worker.py`·`preparation_candidate_results.py`, `preparation_apply_contract.py`·`preparation_apply.py`·`preparation_apply_cli.py`·`preparation_apply_worker.py`·`preparation_apply_results.py`, `preparation_pipeline_cli.py`; 신규 `tests/agents/test_tabular_preparation_v2.py`.

인터페이스: 작업 1의 `TabularPartitionInputContract`를 명시적으로 허용하는 `PreparationCandidateContract`/`PreparationApplyContract` 버전 `v2`를 추가한다. 기존 `.packet(code_id)`·`.check()`·후보 증거 `.signature` API를 유지하되 packet의 입력 버전/BundleRef/부모 source ID·분할 증거를 포함한다. 직렬화 복원은 허용된 두 타입만 분기하며 임의 dict/duck typing을 허용하지 않는다. `.snapshot`에 canonical을 가짜로 넣지 않고 `.source_id`·구간의 고정 입력 참조를 사용한다.

- [x] Train 값 `[1, NaN, 9]`, Validation `[100, NaN]`의 중앙값 기준이 5이며 Validation 결과 `[100,5]`이고 refit=0임을 검사한다. 임의 source index와 bool/큰 정수 target·문자열 group/ID의 값/타입/행 순서를 검사한다. 큰 정수 **특징**의 기존 float64 계산 한계는 조용히 무손실 처리했다고 하지 않고 승인 전 영향 안내 대상으로 고정한다.
- [x] typed 범주/nullable 입력이 기존 변환기의 지원 범위인지 사전 검사하고, 지원되지 않는 조합을 후보 worker 실패까지 미루지 않는다. 후보 교체·다른 split/partition·v1 downgrade·D 이후 입력 변경·원래 행 위치 변조는 거부한다.
- [x] 대상 테스트에서 미구현 binder/연결 API와 복사 worker·지문 누락의 RED를 확인한 뒤 수정했다.
- [x] typed reader에서 가져온 표에 기존 `build_preparation`·저장/복원·영향 집계를 적용한다. 기존 transform 계산을 재작성하지 않는다. v2 row metadata는 작업 1의 안전한 스칼라 인코딩과 schema로 보존하고 임의 `str()` 변환하지 않는다. 후보/적용 결과 버전과 검증기는 같은 encoding을 고정한다.
- [x] 위 대상 모듈과 `tests.agents.test_generic_transform`·분할/내보내기 관련 검사를 실행해 후보 예상/적용 실측 구분·Test 미사용·잘못된 입력의 승인 전 거부를 확인했다. 검증 수치와 미검증 범위는 [GUIDE](GUIDE.md)에 둔다.

v2 전처리는 `bind_typed_partition_input`으로 승인·커밋된 Train/Validation을 결합하고 기존 후보·적용 실행기에 연결한다. 부모의 `tabular_preparation`은 타입/계보 사전 검사와 staging, worker의 `tabular_preparation_worker`는 고정 입력 참조·typed 로딩·안전한 메타데이터를 맡는다. 변환 계산은 기존 `build_preparation`·저장/복원·영향 집계를 재사용한다. 범주 사전 dtype은 값이 같은 object 뷰로 특징 계산에만 전달하고 메타데이터의 원래 dtype은 유지한다. bool·문자열·시각의 수치 특징 지정과 지원되지 않는 범주 스칼라는 후보 생성 전에 거부한다. int64/uint64 특징은 기존 float64 계산의 정밀도 한계를 승인 안내와 후보 보고서에 표시한다.

기존 영향 보고서 v2와 구별해 typed 후보는 `preparation_candidate_report_v3`, 적용은 `preparation_application_v3`만 허용한다. typed 결과를 과거 결과 형식으로 낮추는 것은 거부한다. `*_rows.json`의 `typed_row_metadata_v2`는 안전한 scalar encoding·원래 컬럼/인덱스 schema·원래 행 위치를 담는다. 부모는 승인 구간에서 복원한 메타데이터와 대조하며 `load_row_metadata`로 원래 dtype을 읽을 수 있다. 숫자 특징의 변환 값 전체를 부모가 재계산해 인증한다는 의미는 아니다. 후보 선택 기록은 쓰기/읽기 모두 8 MiB 상한, IPC는 기존 64 KiB 상한을 유지한다. 적용은 해시를 확인한 실제 후보 바이트를 worker scratch에 고정해 복원하며 fit하지 않는다. 후보/적용 worker에는 부모 승인 모듈을 복사하지 않고 부모 출처 검증 의존성은 코드 지문에 포함한다. 정식 서비스·handoff·통합 UI 연결은 작업 6~7이다.

**작업 6 — 표형 서비스 버전·전달·보고서 사실 연결**

파일: 신규 `tabular_workflow_service.py`, 수정 `workflow_service_registry.py`·`csv_workflow_service.py`·`csv_workflow_creation.py`·`csv_workflow_schema.py`·`csv_handoff_workflow.py`·`csv_handoff_contract.py`·`csv_handoff_cli.py`, `llm_workflow_binding.py`·`csv_workflow_binding_evidence.py`·`csv_workflow_proposal_catalogue.py`·`llm_workflow_contract.py`; 신규 `tests/agents/test_tabular_workflow_v2.py`·`test_tabular_handoff_v2.py`.

인터페이스: `workflow_service('tabular_split_review'|'tabular_partition_export'|'tabular_prepare_candidate'|'tabular_prepare_apply'|'tabular_local_handoff', '2.0.0')`를 고정 등록한다. 서비스는 기존 `create/status/preview_approval/record_delegated_approval/next/results` 규약을 유지한다. 새 service의 생성 명세에는 임의 원본 파일 대신 검증된 `source_contract_id`/선행 커밋 참조를 쓴다. 기존 CSV ID `1.0.0`은 바꾸지 않는다. workflow 제안·step binding은 정적 명세의 v2 ID와 선행 결과를 지원하고 검증 없는 우회 실행으로 대체하지 않는다.

- [x] v2 계획의 각 단계가 같은 source/시도/동의에 묶이고 `source_contract_id`가 다른 선행 결과나 지원하지 않는 서비스 버전은 거부되는지 검사한다. v1 서비스는 기존 생성 명세와 결과 그대로 검증한다.
- [x] handoff에서 출력 숫자 특징 CSV와 타입/결측/행 위치를 보존한 metadata 부속 파일을 대조한다. 라벨 `True`와 `1`, `'0012'`와 `12`가 혼동되지 않아야 하며, 저장/복원한 fitted 기준과 출력 컬럼 순서가 같아야 한다. 자기 manifest만 다시 계산한 위조·추가 파일·누락·다른 D 후보·다른 시도는 거부한다.
- [x] `python -X utf8 -B -m unittest -v tests.agents.test_tabular_workflow_v2 tests.agents.test_tabular_handoff_v2 tests.agents.test_csv_workflow_service tests.agents.test_csv_handoff tests.agents.test_llm_workflow_binding`로 RED를 확인한다.
- [x] 필요한 서비스/생성/검증 공통 함수만 추출하고 CSV wrapper는 유지한다. 표형 모듈을 공통 UI 카탈로그에 중복 노출하지 않는다. 읽기 전용 과거 결과 검증은 기존 허용 범위만 유지하고 v2를 이유로 확대하지 않는다. metadata에 값이 있는 파일은 모델 전송 대상이 아니며 사용 예제에서도 입력 타입 보존 reader를 제공한다.
- [x] 같은 명령으로 통과를 확인하고 v2 결과의 보고 근거(`report_sources`)가 검증된 단계 결과에서만 입력 해석·품질·제안·후보·전처리기·적용·전달 파일을 모으는지 검사한다. 이 근거를 `report_facts`의 문제·처리·전후/미집계·해석·정비·실제 적용 구간·파일 활용 설명으로 화면에 연결하는 것은 작업 7의 완료 조건이며 아직 구현하지 않았다.

v2 5개 서비스는 `2.0.0`으로 정적 등록하고 UI 카탈로그와 기본 LLM 제안 명세(`workflow_proposal_specifications()`)에는 노출하지 않는다. v2 제안 명세는 `include_tabular=True`로만 만들며 한 제안에 CSV/표형 단계를 섞거나 다른 계열의 선행 참조를 쓰면 거부한다. 각 단계는 같은 `source_contract_id`에 묶이고 materialize 실행은 관리 입력 루트(`logs/agents/tabular_input`, 작업별 `input_runs`) 아래만 허용한다. `create()`는 preview 이후 원본·분할 서명·Train 계약을 다시 대조한다. v2 연결 지문은 `tabular_workflow_service.code_id`(CSV 지문 + v2 의존 파일)이며 v1은 기존 `csv_workflow_creation.code_id`를 쓴다. v1 단계는 v2 선행 결과를, v2 단계는 v1 선행 결과를 거부한다.

v2 handoff는 `tabular_preparation_handoff_v2`로 숫자 특징 CSV와 `typed_row_metadata_v2` 행 파일, `preparation_application_v3`, `source_contract_id`·파이프라인 서명·적용 계획 ID를 담은 provenance, 고정 README를 둔다. 결과 확인은 이 값을 승인 기록과 대조하므로 자기 manifest만 다시 계산한 위조나 다른 D 시도의 묶음은 거부된다. 사용자용 reader `load_handoff_partition`과 결과 확인(`HandoffTool.verified_bundle`)은 완료 기록(manifest) 바이트를 한 번 읽어 봉인하고, 그 기록의 크기·해시와 같은 바이트만 비공개 임시 사본에 복사한 뒤 사본 하나에서 묶음 검사·승인 기록 대조·해석을 모두 수행한다.

검사 뒤 원본을 다시 읽지 않으므로 검사 중 manifest·파일을 바꾸거나 되돌려도 검증하지 않은 값이 반환되지 않는다. 특징 CSV는 `%.17g`로 쓰고 `float_precision="round_trip"`으로 읽어 float64 값이 추가로 반올림되지 않는다. 라벨·그룹·식별자 결측은 분할 단계가 거부하므로 메타데이터의 bool·nullable 결측 왕복은 reader 계층의 합성 사본으로만 검사했다.

reader는 묶음 자체 정합성만 확인하며 승인 이력과의 결합은 `results()`가 담당한다. 임시 사본은 `%TEMP%`에 생기며 강제 종료 시 남을 수 있다.

`report_sources`는 단계별 결과를 다시 검증하고 단계 간 연결을 대조한 뒤 보고 근거를 로컬 전용 사실로 돌려준다. 원본을 다시 열 때는 커밋 폴더 완료 표식의 해시에 묶인 바이트만 쓰고, 승인 체인에 묶인 값(분할안은 확정 증거, 후보·전처리기는 검증 시점 표식 해시, 적용 기록은 봉인 manifest 해시)과 대조한다. quality는 승인 체인에 묶인 기대 해시가 없어 완료 표식의 손상 검사 수준이다. 반환하는 전달 폴더·README·보고서 경로는 원본 경로이므로 작업 7에서 화면에 표시할 때는 검증된 해시에 묶인 바이트만 써야 한다. 작업 6 자체 검증은 합성 입력·모의 OS 전송 기준이다. 이후 실제 AppContainer·UI·모델 요청의 확인 범위는 작업 8과 GUIDE를 따른다. 검증 비용 개선 A로 한 호출 안에서 같은 하위 증거를 반복 결합·검사하던 구조를 정리했다(Validation 구간 1회 결합, packet·자료형 대조·후보 읽기가 같은 호출의 검사 결과를 공유). 이후 `verification_scope`를 후보 증거 읽기·`_pipeline`·`report_sources`·v2 단계 생성·서비스 실행 열기에 적용해 같은 공개 호출 안의 묶음 검증과 작은 검사 결과를 공유한다. 같은 경로·외부 기대 지문·상한 및 검사별 조건에 묶고 재사용할 때도 경로·파일 구성·크기·해시를 다시 확인한다. 실제 행 소비 바이트의 무결성 검사는 유지하고, 범위는 반환·예외 시 해제하며 worker 안에서는 재사용하지 않는다. 승인 DB·취소·상태·코드 지문 확인과 입력 복사·결과 등록 경계의 검증은 유지한다. 호출 사이·전역·디스크 캐시와 별도 B안 프레임워크는 도입하지 않는다. 검증 수치와 실제 처리 시간은 [GUIDE](GUIDE.md)에 둔다.

**작업 7 — 새 어댑터·기존 화면의 끝단 연결**

파일: 신규 `tabular_v2_task.py`, 수정 `tabular_task.py`·`tabular_options.py`·`profile_store.py`·`task_adapters.py`·`task_orchestrator.py`·`task_plan.py`, `src/ui/task_view.py`·`task_forms.py`·`tabular_task_form.py`; 신규 `tests/agents/test_tabular_task_v2.py`·`test_ui_tabular_v2.py`, 기존 `test_task_array_adapter.py`의 공통 경계 검사 확장.

인터페이스: v2 어댑터는 기존 어댑터와 같은 `diagnose/base_plan/plan_fields/validate_plan/questions/prepare/create_step/service/check_lineage/review/report_facts`를 제공하고 작업 3 입력 선택을 추가한다. `task_adapters.select(path, *, generation='legacy')`와 `input_specs(*, generation='legacy')`는 legacy/current 후보 집합을 각각 검증한다. `start_task`에 keyword-only `input_generation='legacy'`를 추가해 기존 호출/테스트를 유지하고, 통합 UI의 신규 생성은 고정 `current`를 요청한다. current는 CSV→`tabular_v2`, NPZ→기존 `array_npz` 한 개씩만 고르며 외부 입력이 임의 kind/import를 지정할 수 없다. 구기록은 저장된 kind로 읽는다.

- [x] v2 어댑터 `report_facts`가 작업 6의 `report_sources` 근거로 문제·처리·전후/미집계·해석·정비·실제 적용 구간·파일 활용 설명을 화면 보고서에 연결하는지 검사한다. 화면에 보일 파일 내용은 검증된 완료 표식·manifest 해시에 묶인 바이트만 쓰고 반환 경로를 다시 열어 신뢰하지 않는다.
- [x] AppTest로 정상 CSV의 A→진단→권장안/질문→최종 계획→C→후보→D→전달/보고서를 확인한다. mock 모델 거부 시 기본 계획으로 끝까지 진행해야 한다. 한 화면에서 중복 헤더 동의·해석 수정·재제출 거부·원인/다음 행동도 검사한다.
- [x] 새 current 선택이 v2로 가고 기존 legacy 선택과 저장된 CSV/배열 기록이 그대로 열리는지 검사한다. 공통 조정기/화면에 CSV 파서·WM 필드·형식별 계산 분기를 넣지 않는다. 새 메뉴·T2 미구현 선택지는 노출하지 않는다.
- [ ] `python -X utf8 -B -m unittest -v tests.agents.test_tabular_task_v2 tests.agents.test_ui_tabular_v2 tests.agents.test_task_orchestrator tests.agents.test_ui_task tests.agents.test_task_array_adapter tests.agents.test_array_task`로 RED를 확인한다.
- [x] v2 프로파일 ID `tabular_canonical_v2`를 새 `TabularEdaInputContract`의 정적 허용 목록에 등록한다. 기존 `eda_input_contract.REGISTERED_EDA_ADAPTERS`의 CSV v1 검사를 완화하지 않는다. `profile_store`의 기존 CSV 기본값은 유지하고 v2 어댑터만 새 ID를 명시한다. `source_id`·선택 revision·계획 미리보기 digest·후보 signature가 최종 실행까지 같은지 검사한다. 같은 코드의 옵션 설명·권장안 값 명세·역할 검사·보고서 조립은 공유하되 CSV 파일명/경로를 공통 결과로 가정하지 않는다.
- [x] 같은 명령으로 통과를 확인한다. T2 숫자 문자열 정비를 구현하지 않은 상태에서 수치 역할 부적합을 자동 수정하지 않고 지원 범위와 다음 행동을 안내한다.

RED 항목은 신규 테스트를 구현과 함께 작성해 실패 단계 기록을 남기지 않았다(미수행으로 둔다). 통과 확인은 목록의 모듈과 영향 모듈을 함께 실행했다.

새 작업은 `task_adapters`의 current 세대에서 CSV→`tabular_v2`, NPZ→`array_npz`로만 정하고, legacy 세대와 기본 호출은 v2 도입 전과 같은 후보다. 저장된 작업은 기록의 kind로 연다. `tabular_v2` 어댑터는 결정 A로 입력 검사·생성(`tabular_input_flow`)과 생성된 표의 고정 진단(`pipeline_approval_cli.create_tabular_diagnostic`, 선택 기록 v2)을 위임 발급한다.

같은 컬럼명이 있으면 진단·모델 단계 전에 `awaiting_input`으로 멈추고, 화면은 고정 코드·개수와 처리 선택지, 동의 범위를 보인 뒤 직전 화면의 revision·검사 ID로만 재개한다. 오래된 화면·해석 변경 없음·만료된 동의는 새 동의·실행 전에 거부하고 원인과 다음 행동을 같은 화면에 보인다. 권장안·질문·옵션 설명·역할 검사·보고서 조립은 CSV v1 어댑터와 같은 코드(`review_facts`·`report_from_sources`)를 쓰고, 생성된 표에서 수치 자료형이 아닌 컬럼의 수치형 역할은 자동 수정 없이 계획 단계에서 거부·안내한다(T2 숫자 문자열 정비 미구현). v2 계획 항목 명세(`plan_fields`)는 같은 판정으로 이 컬럼들의 역할 선택지에서 수치형을 빼므로, 권장안 요청의 `allowed_values`·응답 형식·병합 검사·질문 답변 검사가 실행 계약과 같은 허용 역할을 쓴다. v2 지시문은 숫자처럼 보이는 문자열도 자동 변환하지 않음을 알린다. CSV v1·배열 명세와 계획 화면의 역할 표 선택지는 바뀌지 않는다.

진단 표·결정 D 후보·그래프·보고 근거는 완료 표식·후보 증거·봉인 manifest 해시에 묶인 바이트만 쓴다. 품질 검토로 멈춘 분할의 지적은 완료 표식이 없어 한 번 읽은 사본을 검증한 값만 보인다(표시 전용). 진단 표식·EDA 그래프 표식은 승인 체인에 기대 해시가 없어 표식 재작성은 손상 검사 수준까지만 막는다. 결과 보고서의 '처리 전 데이터와 기본 EDA' 절은 같은 분할 실행의 `run_eda` 문맥 표식에 묶인 고정 EDA 파일 전체를 한 번 읽고, 비공개 임시 사본에서 기존 EDA 결과 검증기로 검사한 같은 바이트만 문자열 보존 파서로 해석한다(범주값 미표시, 같은 손상 검사 수준). 표시 기준(상위 20·희소·|r| 0.8)은 표시용이며 처리·제외·차단에 쓰지 않는다. 보고서 그래프(`charts`: 요청·확정 분할 비율, target 분포)는 분할 결과 표·target 표와 같은 검증된 값으로 만든 사실이며 그림 파일을 새로 쓰지 않는다. 화면은 이 사실로 그리고 저장 Markdown은 같은 값의 표·텍스트 막대로 남긴다.

합성 입력·모의 OS 전송·모의 모델로 검증했으며 실제 AppContainer·실제 모델 요청·실제 관리자 UI 흐름의 확인 범위와 한계는 작업 8과 [GUIDE](GUIDE.md)를 따른다.

**작업 7 마무리 명세**

- 완료 범위는 신규 CSV의 v2 입력 검사·해석 선택·표 생성부터 진단, 권장안·질문, 계획 확인, 분할·후보, 적용·전달, `report_facts` 보고서 연결까지의 구현과 합성 검증이다. 기존 CSV v1·배열 기록은 저장된 유형으로 유지한다.
- CSV에는 원래 자료형 선언이 없으므로, 자료형 보존은 **입력 해석에서 확정한 자료형과 의미를 지원 계약 범위에서 보존**한다는 뜻이다. 숫자 문자열의 수치 변환·추가 입력 형식 지원은 포함하지 않는다.
- 진단 표·EDA 그래프의 완료 표식은 승인 체인에 묶인 기대 해시가 없으므로 손상 검사 수준의 근거다. 승인 체인에 결합된 후보·적용 근거와 동일한 보장으로 표현하지 않는다. 실제 화면의 구분 표시는 작업 8에서 확인한다.
- 검증 수치와 수정 전후 범위는 [GUIDE](GUIDE.md)의 표형 입력 확장 행을 따른다. RED 기록은 미수행이며 실제 AppContainer·모델·관리자 UI·실데이터 검증은 완료에 포함하지 않는다.
- 이후 수행한 작업 8의 검증 근거와 미수행 항목은 아래 체크 목록과 GUIDE를 따른다. B안·T2~T4·전체 회귀·기존 작업 재실행·외부 전송은 자동으로 진행하지 않는다.

**작업 8 — 승인 지문·독립 검토·실제 환경 확인 준비**

파일: 앞 작업의 worker/등록부와 `module_pipeline_transport.py`·`snapshot_split_transport.py`·`partition_export_transport.py`·`preparation_candidate_transport.py`·`preparation_apply_transport.py`·신규 `tabular_input_transport.py`의 복사/지문 목록, `appcontainer_probe.py`의 실제 영향 경로 확인, 신규 `tests/agents/test_tabular_v2_appcontainer.py`; 필요 시 기존 `scripts/run_agent_regression.py`의 자동 발견 여부만 확인한다. CLI/UI 실제 확인용 고정 합성 fixture는 테스트 helper로 두며 새 날짜별 보고서/스크립트를 양산하지 않는다.

- [x] 구현 착수 직전 전체 기존 code manifest와 작업 트리 기준을 읽기 전용으로 보존한다. 모든 신규 worker 의존 파일·출처/정책/위임 판정 코드가 해당 worker 또는 제어 지문 목록에 있는지 검사한다. 런타임에는 T1의 새 엔진을 추가하지 않으며 기존 복사 상한·network capability·핸들·ACL·Job 정책을 유지한다. — 기준은 로컬 `_workspace`의 작업 8 시작 지문·해시. 검토에서 확인된 누락(`tabular_v2_task`·`llm_workflow_store`·`preparation_module_plan`·`tabular_input_flow`)을 해당 지문에 넣었다. ACL·Job·복사 정책은 바꾸지 않았고, worker의 bundle 경로 검사만 부모가 권한을 준 폴더(input·output/result·scratch/task)로 좁혔다.
- [x] 독립 보안 검토에서 A 조건부 materialize·C/D 출처/후보 결합·신규 파일·legacy downgrade·원본/결과 경로 혼동·코드 지문 누락을 SEC 기준으로 검토한다. 지적 반영은 실행 중인 테스트가 끝난 뒤 수행하고 영향 테스트를 다시 실행한다. 보안 PASS와 제품 완료를 구분한다. — 확정 지적(실제 격리 테스트 판별력, 다른 작업 출처 수용, 다른 서명 실행안 수용, 동의 C와 다른 역할 프로파일 수용)을 재현·수정했고 시도 번호 대조를 더했다. 마지막 판정 경계 변경 뒤 재검토 PASS. 이후 계획 화면의 질문 표시 수정은 판정식을 바꾸지 않아 독립 검토 없이 화면·조정기 직접 테스트로 확인했다.
- [x] 기준 커밋의 전체 회귀로 작업 1~7 관련 합성 테스트를 확인하고 실행/통과/실패/오류/생략·해시 시점을 기록했다. DESIGN에 명시된 테스트 모듈 25개가 로그에 포함됐으며, 실행 전후 소스·테스트·설정·실행기와 HEAD가 같았다. 개별 명령은 따로 반복하지 않았다. 실제 AppContainer 선택 검사는 생략됐고 간헐 실패는 원인 해결로 표시하지 않는다. 수치·원본 로그는 [GUIDE](GUIDE.md#마지막-전체-회귀와-적용-범위)를 따른다. 추가 전체 회귀는 별도 사용자 요청/동의 후 수행한다.
- [x] 실제 환경 허용 후 관리자 UI에서 고정 합성 CSV 한 건을 새 v2 경로로 끝까지 실행한다. 선택 검사 `WAFERGUARD_TEST_APPCONTAINER=1`과 `python -X utf8 -B -m unittest -v tests.agents.test_tabular_v2_appcontainer`로 inspect/materialize·진단·후속 단계의 종료·Job 정리·active_processes=0·원본 불변을 확인한다. 원본 수치 정답과 NumPy/pandas 독립 계산을 대조한다. 새 egress 시험을 하지 않았다면 기존 증거를 새 실행의 전면 보장으로 재사용하지 않는다. — 선택 검사는 worker 경계 수정 뒤 코드에서 통과했다(이후의 출처·서명·분할 정지 수정 뒤 재실행 없음). 관리자 UI 흐름은 두 건이며 코드 시점이 다르다. 검증 근거와 한계는 [GUIDE](GUIDE.md)를 따른다. 이후 별도 합성 probe의 외부·loopback WFP 폐기 증거를 확인했다. 운영 worker의 모든 접속과 SEC-12 전체를 검증한 것은 아니며, 기존·신규 시험은 GUIDE에서 구분한다.
- [x] 실제 manifest 변경 목록으로 CSV·모듈 파이프라인·배열·WM의 미실행 승인/완료 결과 조회 영향을 보고한다. 사용자 실데이터·기존 작업 재실행·실제 모델 호출·커밋/푸시는 이 계획의 자동 실행 항목이 아니다. — 작업 8 전 대비 23종 중 10종 변경(CSV v1·binding·tabular·배열·모듈 파이프라인·입력·분할·내보내기·후보·적용), WM 9종·EDA·audit·pickle·reviewed module 13종 동일. 바뀐 지문의 미실행 승인은 다시 준비해야 하며, 열 수 없는 저장 실행안은 결정 D 전에 `PLAN_STALE`로 멈춰 계획 수정으로 이어 간다.

**작업 9 — 현재 지원 상태와 마무리**

파일: `DESIGN.md`의 이 절과 `GUIDE.md`의 입력 지원/시작 순서/한계만 갱신한다. 일정표는 실제 단계 완료 상태가 바뀐 경우에만 해당 항목을 갱신한다.

- [x] T1 구현·합성 검증·독립 검토·실제 OS/UI 검증을 코드 시점과 근거별로 구분해 GUIDE에 정리했다. 사례 A·B의 기본 계획 사용과 이후 모델 권장안 채택 확인을 구분한다. 관련 합성 명령 범위는 기준 커밋 전체 회귀로 확인했으며, 이후 변경·실제 선택 검사·모델·실데이터의 검증 범위는 GUIDE를 따른다.
- [x] GUIDE에 v1/v2 입력 해석·CSV 지원 범위·빈 문자열/큰 정수의 보존과 계산 한계·원본 보존·동의 재발급 영향을 안내했다. T2~T4는 보류로 유지한다.
- [x] 문서 링크·`git diff --check`·변경 파일 범위를 확인한다. 최종 검증 수치는 GUIDE, 커밋은 사용자 지시에 따른다. 문서 정리는 코드 검증을 대체하지 않는다.

**계획 자체 점검:** 공통 저장/출처는 작업 1, 기본 해석/정비는 2·3, 중간 표형 계약은 4·5, 서비스/보고서는 6·7, 독립 검토/실제 확인·정리는 8·9에 대응한다. 체크는 해당 근거 범위의 완료만 나타내며 미체크 항목은 미수행이다. 후속 로더·엔진·다중 파일·수치 문자열 변환은 이번 마감 범위에서 제외한다.

## 입력 계약

### Snapshot과 프로파일

- `input_snapshot.py`는 파일을 새 작업 폴더로 바이트 복사하고 `input.bin`·`snapshot.json`을 생성한다. 내용 식별값은 `file_snapshot:<SHA-256>`, 계약은 `file_snapshot_v1`이다. 확장자로 로더를 추정하지 않는다.
- 복사 동의는 분석·역직렬화·외부 전달 승인이 아니다. 일반 파일·허용 경로·크기·링크·원본/복사본 바이트 정합성을 검사한다. 크기 상한을 자동 확대하거나 데이터를 잘라 맞추지 않는다. 부분 파일은 완료 입력이 아니다.
- manifest 자체에서 기대 snapshot을 신뢰하지 않고 외부에 보존한 기대 계약과 재대조한다. 출처 경로와 작업 snapshot을 구분한다. 작업 시작 뒤 원본이 이동·변경돼도 승인한 snapshot이 온전하면 이후 처리는 그 복사본 기준이다.
- `profiles.py`의 `dataset_profile_v1`은 역할·품질·분할·EDA 규칙을 검증한다. 내부 역할은 numeric·categorical·target·group·timestamp·identifiers이며, UI의 제외는 역할 목록에 넣지 않는 방식이다. 역할 중복·없는 컬럼·잘못된 설정을 거부한다.
- `EdaInputContract`는 snapshot·어댑터·프로파일·자원 한도를 결합한다. `contract_id`와 `source_id`를 계획·승인 scope에 결합한다. 계약 생성/JSON 복원은 실행이나 승인이 아니다. 계약 경로·해시는 로컬 전용이다.

### 프로파일 없는 진단과 작성

`input_profile_pipeline.py`의 진단 입력은 snapshot·자원 한도만 가지며 일반 `EdaInputContract`의 profile 필수 검사를 우회하지 않는다. 프로파일 없는 실행은 고정 진단 계획에만 허용한다. worker가 계획 전체·서명을 재검사하고 일반 계획과 진단 계약의 혼합을 거부한다.

격리 worker는 기존 EDA 계산을 재사용한다. 결측·고유값·중복은 원래 값 기준, 숫자 통계·IQR은 유한값 기준으로 계산하고 ±무한대·공백 문자열을 따로 집계한다. 자료형 구분은 역할 확정이 아니다. 부모는 컬럼명·순서를 승인된 헤더와 대조하고 형식·범위·내부 항등식을 검사한다. 원본 대비 통계 정답의 독립 증명은 아니다.

진단 결과는 로컬 표시/프로파일 작성용이며 해설 패킷·자동 해설·일반 해설 화면으로 전달하지 않는다. 역할은 사용자에게 보이고 수정 가능해야 한다. 명시 버튼으로만 일괄 지정하며 결측률·이상치 후보만으로 자동 삭제하지 않는다. 프로파일은 새 파일로 저장하고 동명 덮어쓰기·예약 이름·링크를 거부한다. 같은 입력과 저장 당시 해시가 맞을 때만 후속 기본값으로 연결한다. 저장은 실행 승인이 아니다.

### 중복 헤더 정비

일반 로더는 중복/빈 헤더를 거부한다. 통합 작업은 중복 헤더일 때만 사용자 동의를 받고 격리 정비를 제공한다. 작업 snapshot의 중복 이름에 충돌 없는 번호를 붙인 새 복사본을 만들고, 컬럼 순서·헤더 뒤 데이터 바이트·줄 끝이 보존되는지 검증한다. 빈 이름·파싱 실패를 자동 수정하지 않는다.

해결 가능 여부·단계·이미 정비했는지·snapshot 해시를 검사한 뒤 새 동의를 발급한다. 정비 전 진단·이전/새 동의·정비·재진단 실행·새 snapshot 해시를 history에 연결하고 이전 기록과 소비된 승인은 보존한다. 실패한 정비는 자동 반복하지 않는다. 이후 동의·실행은 새 복사본에 결합하며 원본용 승인을 재사용하지 않는다. 계획·보고서에 이름 변경과 복사본 사용을 알린다.

## 계획과 질문

계획은 어댑터의 고정 모듈 템플릿과 역할·분할·전처리의 유효 설정으로 구성한다. 기본 계획 → 모델 권장 변경 → 사용자 수정/답변 → 전체 검증 → 최종 계획 미리보기 → 동의 C 순서다. LLM은 모듈 순서나 도구를 임의 추가하지 않는다.

모델 변경은 `{path, value, reason}` 형식으로 허용 경로·값·타입·범위를 검사한다. 기존 기본값·허용값은 계약 코드에서 가져온다. 권장 변경을 합친 계획에 하나라도 위반이 있으면 변경 전체를 거부하고 사유 코드와 기본 계획을 표시한다. 허용되지 않은 모델 질문을 제외하는 것과 계획 변경의 부분 채택은 다르다.

질문은 고정 ID·출처(adapter/model)·필수/안내 수준·대상 path·선택지·권장값·문구·관련 항목을 가진다. 필수 질문은 어댑터만 생성한다. 답변 하나는 해당 path 하나만 바꾸며, 필수 질문은 답변 후 최종 계획에서 다시 판정한다. 계획 변경형 질문은 실제 조건이 해소돼야 해결이고, 확인형 답만 답변 사실로 해결한다. 모델 문장·컬럼명은 일반 텍스트로 표시한다. 화면은 질문마다 출처(진단 계산 기준 질문 / 모델 질문·확인된 사실 아님)를 표시하고, 어댑터 질문이 있는 항목의 모델 질문은 보이지 않는다. 모델 권장안은 형식·허용값 검사를 통과해도 내용의 타당성은 확인되지 않았다고 표시하며, 권장 변경마다 근거(모델 의견)와 어댑터의 확인 안내를 붙인다. 기본 계획과 같은 값의 '변경'은 권장 변경 목록에서 뺀다.

### 현재 모델의 품질 한계와 확장 원칙

현재 로컬 모델의 확인 사례에서는 진단 수치 해석, 라벨·그룹 구분, 전처리 권장 이유에 의미 오류가 관찰됐다. 응답 형식·허용값 검사 통과는 내용의 타당성을 보장하지 않는다. 계획 확인·완료 화면은 이 한계를 알리고 계산 사실과 모델 의견을 구분한다. 이 관찰은 현재 모델과 확인한 사례의 범위이며, 작은 로컬 모델에 맞춰 공통 권장 기능을 영구 축소하지 않는다.

공통 계획·권장안 계약과 승인·격리·결과 검증은 유지한다. 모델 교체 시 같은 사례와 기준으로 형식 적합성과 내용 타당성을 따로 평가한다. 로컬/API라는 연결 방식만으로 모델 능력을 판단하지 않는다. 향후 API 모델도 공통 계약 안에서 권장·질문·설명을 제공할 수 있도록 하되, API 연결 자체는 아직 미구현이며 별도 전송 동의·자료 범위·비용·시간 제한·실패 정책을 설계해야 한다. 현재 로컬 동의를 외부 전송 동의로 확대하지 않는다.

### 표형 검사와 옵션

- target 미지정은 필수 질문이다. 후보를 고르거나 역할 표에서 지정하고, 비예측 목적이면 명시적으로 예측 대상 없음에 답한다. 이 답은 동의 C와 보고서 경고에 보존한다. 요약 문장에 target이 있다는 것만으로 실제 역할이 바뀌지 않는다.
- group 역할과 분할 방식, 계층 분할과 target, 미선택 방식의 교차 검사는 화면·서버·권장안에서 같은 함수를 사용한다. 통합 작업의 time 분할·시간 형식은 미지원이다. timestamp를 실행 가능한 역할처럼 권하지 않고 필수 확인으로 처리한다. random으로 조용히 대체하지 않는다.
- 고유값 수와 이름/연속 정수 단서의 식별자 의심은 안내 질문이며 자동 역할 변경이 아니다. Train에서 범주 상한 초과가 확정되는 경우는 필수 질문이고, group 분할 등 불확실한 경우는 안내로 구분한다.
- 모든 전처리 옵션을 모델 질문과 무관하게 수정/해제할 수 있다. `PreparationRules`의 허용값·조합 검사를 유지한다. 구간화와 스케일링 동시 선택 등 충돌에는 질문 번호·선택값·다음 행동을 안내하고 자동 수정하지 않는다.
- 양식과 답변이 같은 항목에서 충돌하면 사용자가 선택한다. 최종 미리보기의 계획·시도 번호·revision을 결합하고 서버 확정 시 대조한다. 거부된 확정은 동의·시도·worker 실행을 만들지 않는다. 동일 요청 재전송과 오래된 적용 화면을 검사한다.
- 전처리 옵션은 처리 내용·적용 대상·선택 근거·주의·none의 의미를 설명한다. 기준 출처를 기본·모델 의견·사용자 수정·질문 답변으로 구분한다.

계획의 기준 코드는 `task_plan.py`, `llm_task_assist.py`, `task_recommend.py`, `tabular_task.py`, `tabular_options.py`와 `src/preprocessing/transform.py`다. 문서에서 별도 기본값/선택지 복사본을 유지하지 않는다.

## 승인과 실행

### 동의와 단계

| 결정 | 범위 |
| --- | --- |
| A | 선택한 snapshot의 격리 진단과 화면에 고지된 모델 권장안 요청. 정비 시 새 범위의 동의 필요 |
| C | 최종 계획과 입력·코드·시도에 결합한 분할 검토/확정 → 개발 구간 내보내기 → Train 기준 후보. 분할 EDA 검토 위임 포함 |
| D | 검증된 정확한 후보/적용 대상에 대한 Train·Validation 적용 → 로컬 전달 묶음 |

동의는 개별 승인 자체가 아니다. 확정 계획의 모듈·단계에서 허용 위임 범위를 계산하고, 매 발급 직전 동일 입력·계획·actor·코드·시도·취소 상태와 허용 범위를 대조한다. `DELEGATED_APPROVAL_RECORDED`로 기록하며 실제 사용자 검토 이벤트로 위장하지 않는다. 단계별 승인은 여전히 한 번만 소비한다. 위임 기능은 CSV 연결부와 배열 서비스(`delegated_approval.record_delegated`)에서 기존 승인 실행기를 사용하며 웨이퍼 통합 위임은 미구현이다.

승인은 actor·run·stage·tool·정규화 대상/인자·source/config/plan/code에 결합하며, 만료·다른 사용자·취소·변조·보안 잠금이면 거부한다. 승인 검증·소비·작업 예약은 DB 트랜잭션으로 수행한다. 소비 직후 장애가 나도 자동으로 되돌리거나 재실행하지 않는다. 값이 같은 것과 실행 증거/승인이 있는 것은 다르다.

### 실행 경계

worker는 허용 목록의 복사된 소스와 작업 입력만 사용한다. AppContainer capability·Job 소속·자원 한도·종료/정리 증거를 검사한다. 원본 폴더·승인 DB·다른 작업에 접근시키지 않는다. 부모의 모델 요청과 계산 worker의 격리는 별개이며 별도 Python 환경만으로 OS 격리라고 하지 않는다.

#### 공유 worker 런타임 (표형 입력 검사·생성에만 적용)

`experiment/worker-runtime-reuse` 실험 브랜치에서 표형 입력의 검사(inspect)와 작업용 표 생성(materialize) worker에만 적용한다. 나머지 worker는 기존처럼 실행마다 런타임을 작업 폴더로 복사한다. 기준 코드는 `worker_runtime.py`다.

- **위치와 관리 주체**
  - 부모 앱이 `logs/agents/worker_runtime/<runtime_id>/`를 만든다. 이 폴더에는 `manifest.json`(완료 기록)과 `runtime/`(`sys.prefix` 기준 상대 경로)이 있다.
  - 위치는 코드 상수다. 요청·모델·입력 데이터가 고르지 않는다.
  - 저장소 작성 주체는 런타임을 만든 현재 사용자, SYSTEM, Administrators뿐이다(신뢰 주체).
  - 상위 폴더(logs·프로젝트·드라이브)의 ACL은 바꾸지 않는다.
- **식별값**
  - `runtime_files()` 허용 목록의 각 파일에 대해 [상대 경로, 크기, SHA-256]을 모으고, Python 버전·cache_tag와 함께 정규 JSON으로 만들어 SHA-256을 계산한다.
  - 원본 전체 내용을 해시한다. 폴더 존재·버전 문자열·크기만으로 같은 환경이라고 판단하지 않는다.
- **승인 결합**
  - 입력 준비(`tabular_input_flow.prepare_input`)는 검사·생성 실행 기록을 만들기 전에 현재 원본 구성 ID(`current_runtime_id`)를 계산해 작업 기록에 고정한다.
  - 검사·생성 packet의 `runtime_id`에 같은 값을 넣는다. 그래서 다음이 모두 이 ID에 결합된다.
    - `config_id`·요청 식별
    - 승인 범위·실행 계획
    - 승인 미리보기 요약(`runtime_id`)
  - 위임 발급은 미리보기 요약의 `runtime_id`가 작업 기록의 값과 같을 때만 진행한다.
  - 생성 실행은 선행 검사와 같은 `runtime_id`여야 한다.
  - 생성 출처 계약(`TabularSourceContract.execution_runtime_id`)도 이 ID로 승인 `config_id`를 다시 계산해 대조한다.
  - 동의 A의 내용에는 런타임 ID가 없다. 동의 뒤 입력 준비 시점의 환경으로 ID가 정해진다.
  - 승인 1회 소비·actor·입력·계획·정책 결합은 그대로다.
- **실행 시 선택**
  - transport는 승인 packet의 ID로만 런타임을 고른다.
  - 공개본이 있으면 그 완료 기록으로 검증한다. 완료 기록 내용의 해시가 승인 ID와 같아야 기준으로 쓴다.
  - 공개본이 없으면 현재 원본 구성 ID가 승인 ID와 같을 때만 준비한다. 다르면 `RUNTIME_CHANGED`로 멈춘다.
  - 같은 승인으로 다른 ID를 고르지 않는다. 승인 뒤 원본 환경이 바뀌어도 승인한 공개본이 검증을 통과하면 그 공개본을 쓴다.
- **보호 권한**
  - 저장소 루트와 각 버전·staging 폴더는 생성 순간부터 상속을 끊은 보호 DACL을 갖는다(`CreateDirectoryW`와 보안 서술자, `D:P`).
  - 루트: 신뢰 주체 FILE_ALL_ACCESS만 둔다.
  - 버전·staging: 신뢰 주체 FILE_ALL_ACCESS에 더해 capability 읽기·실행(0x1200A9)만 두고, 하위 폴더·파일에 상속한다.
  - 생성 뒤 권한을 바꾸는 단계가 없다.
  - RX를 받는 capability는 고정 이름 capability SID(`DeriveCapabilitySidsFromName("waferguardWorkerRuntime")`) 하나다. worker는 이 capability 하나만 켜고 실행한다. `launch`는 이 값 외의 capability를 거부하며, 네트워크 capability는 없다.
  - 작업별 package SID에는 기존처럼 input·tools RX와 output/result·scratch/task Modify·Low만 준다. 작업마다 공유 런타임 ACL을 바꾸지 않는다.
  - ALL APPLICATION PACKAGES·사용자 그룹·일반 로컬 사용자에게는 부여하지 않는다.
  - 같은 이름의 capability로 AppContainer를 만드는 같은 사용자의 다른 프로그램은 공유 런타임을 읽고 실행할 수 있지만 쓸 수 없다. 다른 사용자 토큰은 사용자 DACL 판정에서 거부된다.
- **재사용 검증(실행마다)**
  - 다음 내용을 확인한다.
    - 폴더 이름·완료 기록의 ID·완료 기록 내용의 해시가 승인 ID와 같다.
    - 실제 파일·폴더 집합이 기록과 정확히 같다(추가·누락·빈 폴더 없음).
    - 링크·재분석 지점·하드 링크가 없다.
    - 각 파일의 크기와 SHA-256이 같다.
  - 권한은 저장소 루트, 버전 폴더, 모든 하위 폴더·파일과 완료 기록에 대해 확인한다.
    - 소유자가 신뢰 주체다.
    - DACL이 있다(NULL DACL 거부).
    - 루트·버전 폴더는 상속이 끊겨 있다.
    - 허용 ACE의 주체는 신뢰 주체, OWNER RIGHTS(소유자가 신뢰 주체일 때), 읽기·실행 이하 마스크의 capability뿐이다.
    - 허용·거부 외 ACE 형식이 없다.
    - Medium 미만 무결성 레이블이 없다.
  - 파일 권한은 해시를 읽는 같은 핸들로 조회한다.
  - 내용이 다르면 `RUNTIME_VERIFY_FAILED`, 소유자·권한이 다르면 `RUNTIME_PERMISSION_INVALID`로 멈춘다.
  - 권한을 복구하거나 공개본을 덮어쓰거나 원본 환경으로 실행하지 않는다.
- **경로 교체 방지(lease)**
  - 저장소 루트는 목록·보안 읽기 권한과 삭제 공유 없음으로 먼저 열어 보유하고, 그 핸들로 소유자·권한을 검사한다.
  - 버전 폴더도 같은 방식으로 보유한 뒤 그 핸들로 검사한다.
  - 두 보유는 준비(원본 해시·복사·공개)부터 검증·worker 종료까지 유지한다. 보유 중에는 루트·버전 폴더와 그 상위 경로의 이름 변경·삭제가 막힌다(`os.rename`으로 확인, 다른 API로 일반화하지 않음).
  - 버전 폴더가 보유한 루트 객체의 바로 아래인지는 객체 수준으로 결합하지 않는다. 보유 전에 상위를 우회시킨 경우에도 버전 폴더는 보호 SD·내용 해시·보유로 쓰기·교체가 막혀, 영향은 실행 실패에 그친다(정적 검토 판단).
- **잔여 경계**
  - 해시·완료 기록·ACL은 서명된 무결성이 아니다.
  - 신뢰 주체(같은 사용자·SYSTEM·Administrators)의 변경은 막지 않는다.
  - 같은 capability를 가진 프로세스가 공유 거부 모드로 파일을 열면 다른 실행이 실패할 수 있다(가용성, 미확인).
  - 작업 폴더(input·tools·output)·실행 기록·승인 DB는 아래 [입력 실행 저장소 보호](#입력-실행-저장소-보호-표형-입력-검사생성에만-적용)가 담당한다. 동의 A·C·D가 있는 작업 기록은 아래 [공통 작업 기록 보호](#공통-작업-기록-보호와-위임-근거-모든-작업-유형)가 담당한다.
- **실행 연결**
  - 실행 파일은 `<id>/runtime/python.exe`이고 `-I -S -B`로 실행한다.
  - PATH는 해당 runtime뿐이다. 부모의 PATH·PYTHONPATH는 넘기지 않는다. TEMP·TMP·MPLCONFIGDIR는 작업 scratch다.
  - worker는 실행 조건을 대조한다.
    - `sys.executable`의 위치·폴더 이름·완료 기록의 ID가 인자의 ID와 같아야 한다.
    - 그 ID가 packet의 `runtime_id`와 같아야 한다.
  - 결과 binding의 packet에도 같은 ID가 남는다.
  - 부모 결과 검증은 `transport.json`의 runtime ID가 승인 packet의 ID와 다르거나 없으면 결과를 등록하지 않는다. 형식만 맞는 다른 ID도 거부한다.
  - 공유 런타임과 작업 폴더가 겹치거나 링크를 지나면 실행 전에 거부한다.
- **실패**
  - 실패 코드는 다음과 같다.
    - 원본 목록 실패: `RUNTIME_SOURCE_INVALID`
    - 복사·보호 생성·rename 실패: `RUNTIME_PREPARE_FAILED`
    - 공개본 내용 불일치: `RUNTIME_VERIFY_FAILED`
    - 소유자·권한 이상(다른 계정이 만든 저장소 포함): `RUNTIME_PERMISSION_INVALID`
    - 승인 런타임 미준비 + 환경 변경: `RUNTIME_CHANGED`
  - 준비 중 취소는 취소로 처리한다.
  - 원본 환경으로 대신 실행하거나 검사를 생략하지 않는다.
  - 화면에는 확인된 원인과 다음 행동(관리자의 폴더 확인·이동, 새 실행·승인 준비)을 고정 문구로 보이고, 자동 재시도하지 않는다.
- **호환과 코드 지문**
  - `runtime_id`가 없는 이전 packet·출처 계약은 열지 않는다. 이전 기록은 고치지 않는다.
  - 네이티브 실행기(`appcontainer_native.py`)·출처 계약이 공통 지문 체인에 있어서, 이 변경으로 CSV·배열·WM·pickle 실행기의 code_id도 바뀐다. 해당 경로의 미실행 승인은 새로 준비해야 한다.
  - 미적용 경로의 실행 동작(네트워크 capability 없음·capability 0개, 작업별 복사)은 바뀌지 않는다.

#### 입력 실행 저장소 보호 (표형 입력 검사·생성에만 적용)

같은 실험 브랜치에서 표형 입력 검사·생성의 실행 저장소에만 적용한다. 기준 코드는 `protected_store.py`, `tabular_input_service.py`, `tabular_input_transport.py`다. 보호 생성·보안 조회·의미상 권한 판정·보유는 공유 런타임과 같은 공통 코드(`protected_store.py`)를 쓴다.

- **보호 단위**
  - 입력 실행 저장소 루트 하나를 보호한다. 흐름에서는 `<작업 폴더>/input_runs`다.
  - 루트는 생성 순간부터 신뢰 주체(현재 사용자·SYSTEM·Administrators) FILE_ALL_ACCESS만 있는 보호 DACL을 갖는다. 보호 SD로 만든 임시 폴더(`.store-*`)를 rename으로 공개한다. 상위 경로에 링크가 있으면 루트를 만들지 않는다. 다만 그 전에 빈 상위 폴더가 생길 수 있다(아래 잔여 경계). 비정상 종료로 남은 임시 폴더는 자동으로 정리하지 않는다.
  - 다음은 모두 이 ACE만 상속한다.
    - 실행 기록: `selection.json`, 상태, `events.jsonl`, 완료 결과 `attempts`
    - 승인 DB: `approvals.sqlite3`
    - worker 작업 폴더: `<action>/workers/<task>`와 `transport.json`
  - 실행 이름 `workers`는 예약어로 쓰지 않는다.
- **사용 구간 보유와 재사용 검사**
  - 서비스 공개 연산 9개는 저장소 루트를 보유한 뒤 시작한다. 대상은 create·status·승인 미리보기·위임 발급·next(실행 포함)·verified·source_contract·results·runtime_failure다.
    - 보유 방식: 목록·보안 읽기, 삭제 공유 없음
    - 검사: 보유한 핸들로 소유자·보호·ACE를 확인하고, 보유 뒤 경로에 링크가 없는지 다시 확인한다.
    - 해제: 연산이 끝나면(예외 포함) 놓는다.
  - 실행 기록을 열 때는 다음을 검사한다.
    - action 폴더 객체
    - 실행 기록 폴더 트리 전체: 소유자, 신뢰 주체 외 허용 ACE, 알 수 없는 ACE, Medium 미만 레이블, 링크·하드 링크, 목록 오류
  - 결과 검증과 실패 분류는 `transport.json` 객체도 검사한다.
- **worker 권한**
  - 작업 SID에는 기존처럼 input·tools 읽기·실행, output/result·scratch/task Modify·Low만 준다.
  - 실행 기록·승인 DB·`transport.json`·다른 작업 폴더에는 작업 SID ACE가 없다.
- **실행 직전 검사**
  - transport는 worker 파일 복사와 코드 지문 재확인 뒤, 실행 직전에 작업 폴더를 의미상 검사한다.
    - 작업 폴더 루트, output·scratch 상위: 신뢰 주체만
    - input·tools: 작업 SID 읽기·실행(0x1200A9) 이하
    - output/result·scratch/task: 작업 SID Modify(0x1301BF) 이하와 Low 레이블
  - 다른 주체나 넓은 권한이 있으면 실행하지 않는다. 권한을 고치지 않는다.
- **실패**
  - 소유자·권한·경로 이상은 `INPUT_STORE_PERMISSION_INVALID`, 보호 생성 실패(상위 링크·토큰 조회 실패 포함)는 `INPUT_STORE_PREPARE_FAILED`다.
  - 둘 다 입력 데이터 오류와 구분해 고정 문구와 다음 행동을 보인다. 다음 행동은 관리자의 폴더 확인 후 새 작업 시작이다.
  - 실행 기록 루트나 실행 폴더가 없을 때도 `INPUT_STORE_PERMISSION_INVALID`로 분류된다.
  - 자동 복구·비격리 실행·자동 재시도는 없다.
- **기존 영역**
  - 이전에 만든 보호되지 않은 `input_runs`와 단독 기본 루트(`logs/agents/tabular_input`)는 고치지 않는다. 새 코드로 열면 `INPUT_STORE_PERMISSION_INVALID`로 멈춘다.
  - 이전 작업은 code_id·출처 계약 변경으로 이미 현재 코드에서 진행·재검증할 수 없다. 파일은 보존한다.
- **검사 범위와 잔여 경계**
  - 입력 저장소 검사는 입력 서비스 연산과 생성 결과의 후속 소비(`TabularSourceContract.check_input`, 아래 절)에서 수행한다.
  - 다른 사용자가 작업 폴더에 `input_runs`를 먼저 만들면 실행이 멈춘다(가용성). 새 작업 폴더는 보호되므로 이 위험은 이전 작업 폴더에 남는다.
  - 조상 경로가 junction이면 저장소 생성 중 다른 위치에 빈 폴더가 생길 수 있다(쓰기·데이터 변경 없음).
  - 신뢰 주체의 변경은 막지 않는다.

#### 공통 작업 기록 보호와 위임 근거 (모든 작업 유형)

같은 실험 브랜치에서 공통 작업 저장소(`task_store`, `logs/agents/tasks/<task>/`)에 적용한다. CSV v1·v2와 배열 작업이 함께 쓴다. 각 유형의 계산·transport는 바꾸지 않는다.

- **보호 단위**
  - 새 작업 폴더를 생성 순간부터 보호 DACL로 만든다(신뢰 주체 FILE_ALL_ACCESS, 상속 차단). 실패하면 `RECORD_PROTECT_FAILED`이며 작업을 만들지 않는다.
  - 다음은 모두 이 ACE를 상속한다.
    - `task.json`: 동의 A·C·D, 확정 계획, 단계·위임, 시도
    - `parts`: 진단·초안·후보·보고서·시도 이력·입력 선택
    - `plans`, `snapshot`·`derived_snapshot`
    - 작업 폴더 안의 실행 저장소(`input_runs`·`workflow`·`profiles`·`report`)
  - 저장소 루트(`logs/agents/tasks`)와 이전 작업 폴더의 권한은 바꾸지 않는다.
- **잠금 구간 보유와 검사**
  - `task_lock`은 잠금을 얻은 뒤 작업 폴더를 보유하고(목록·보안 읽기, 삭제 공유 없음), 보유 핸들로 소유자·상속 차단·신뢰 주체 ACE를 검사한다. 보유 뒤 링크도 다시 확인한다.
  - 검사 대상은 직속 파일(`task.json`·잠금 등, 하드 링크 거부)과 `parts`·`plans`·`snapshot`·`derived_snapshot` 트리(의미상 권한)다. `workflow`·`profiles`·`report`·`input_runs`는 상속 보호에 맡기며 여기서 의미상 검사하지 않는다. `input_runs`는 입력 서비스가 검사한다.
  - 위임 발급·worker 실행·기록 저장은 모두 같은 잠금 블록 안에서 진행한다. 해당 연산은 진단·문제 해결·입력 재개·계획 확정·적용 승인·완료 복구다.
  - 보유는 연산이 끝날 때(예외 포함) 놓는다. 보유 중에는 작업 폴더와 상위 경로의 이름 변경·삭제가 막힌다.
  - 보호 밖이면 `RECORD_UNPROTECTED`로 멈추며, 권한을 고치거나 동의를 그대로 신뢰하지 않는다.
- **이전 작업**
  - 보호되지 않은 이전 작업 폴더도 읽기 전용 조회(`load_task`)는 기존대로 된다.
  - 승인 없는 보고서 재조립(`rebuild_report`)은 잠금까지만 허용한다. 재조립은 보유 뒤 경로를 다시 확인한다.
    - 성공 여부는 현재 코드의 결과 검증 통과에 달려 있다.
    - 이전 작업은 보호되지 않은 `input_runs` 등으로 대개 실패하고, 기존 보고서를 그대로 둔다.
  - 진행·위임 발급·완료 복구는 `RECORD_UNPROTECTED`로 열지 않는다.
  - 이전 작업은 code_id 변경으로 이미 새 실행·재검증이 불가했다. 파일은 이동·변경하지 않는다.
- **위임 근거 기록**
  - 동의 판정기(`task_consent.authorizer`)는 사용한 동의의 근거를 돌려준다. 근거는 결정, `consent_id`, 동의 내용 해시, 작업 ID, `attempt`(작업 기록의 계획 시도 번호이며 입력 준비 시도와 다르다)다.
  - 위임 발급은 이 근거와 승인 범위 해시를 실행 저장소의 `DELEGATED_APPROVAL_RECORDED` 이벤트에 남긴 뒤 승인 DB에 발급한다. 해당 경로는 공통 `delegated_approval`과 표형 진단 `pipeline_approval_cli`다.
    - 이벤트 기록이 실패하면 발급하지 않는다.
    - 발급 실패 뒤 다시 시도하면 이벤트가 2건일 수 있으므로, 이 이벤트는 "발급 시도"로 해석한다. 자동 재시도·중복 발급은 없다.
    - 근거 형식은 고정이며(원문·경로·승인 ID 없음) 서명이 아니다. 동의의 진본 인증이 아니라 추적 근거다.
  - 근거가 남는 실행 저장소의 보호 여부는 저장소마다 다르다. 현재 보호되는 곳은 입력 검사·생성(`input_runs`)뿐이다. 작업 폴더 안 `workflow`·`profiles`에는 위임 이벤트가 없다. 표형 진단(`logs/agents/pipeline_approval/runs`)처럼 작업 폴더 밖 전역 저장소의 이벤트는 다른 사용자가 바꿀 수 있다.
- **후속 출처 소비**
  - `TabularSourceContract.check_input`은 생성 실행의 입력 저장소 루트를 보유한 채 다음을 검사한 뒤 기존 대조(승인 소비·완료 표식·snapshot·묶음·해석)를 한다.
    - action 폴더(`materialize`) 객체와 실행 기록 트리 권한
  - 흐름의 `check_source`는 작업 기록의 출처 참조(실행 이름·위치·snapshot)를 작업 ID로 다시 계산해 대조한다. 저장소 오류는 고정 문제(`INPUT_STORE_PERMISSION_INVALID`)로 보인다.
  - 공통 `CommittedArtifact`에는 CSV 정책을 넣지 않았다.
  - 사용 구간 전체의 상위 경로 고정은 오케스트레이터의 작업 폴더 보유가 담당한다. 오케스트레이터 밖에서 서비스·CLI를 직접 호출하면 검사 뒤 사용 구간의 상위 경로는 고정되지 않는다.
- **잔여 경계(전역 실행 저장소는 서버 전환 때 보완 예정)**
  - 2026-10-29 결정: 현재는 개인 PC 단일 사용자 운영이 전제다. 아래 전역 실행 저장소 보호는 이번 범위에서 검토하지 않고 서버 전환 때 수정한다.
  - 진단·분할·내보내기·후보·적용·전달과 배열의 실행 저장소 중 작업 폴더 밖 전역 위치(`logs/agents/...`)에 있는 것은 보호하지 않는다.
  - 결정 C·D 위임은 실행 인자를 작업 기록과 별도로 대조(bound)하지 않는다.
  - 따라서 다른 사용자가 전역 실행 저장소의 실행 선택·인자를 바꿔 동의하지 않은 설정에 위임을 유도할 가능성이 남아 있다(독립 검토 의심 위험, 미재현).
  - 완료 복구에서 저장소 권한 오류는 `RECOVERY_RECORD_MISMATCH`로 표시된다.
  - 신뢰 주체의 변경, 다른 실제 계정의 위조 시도는 검증하지 않았다.

#### AppContainer 허용·차단 구성도

아래는 **에이전트의 격리 worker 경계**다. 기존 SECOM CLI·WM CLI 전체를 자동으로 이 경계 안에서 실행한다는 뜻은 아니다. 실선은 허용된 처리 경로, 점선의 `차단 대상`은 권한을 부여하지 않거나 검증에서 거부하는 경로다. 차단 대상 표시는 모든 실제 환경에서 차단 실측을 마쳤다는 뜻이 아니다.

[![AppContainer 허용 경로·부모 검증·차단 대상 구성도](../architecture/isolation.svg)](https://didgustmd23.github.io/waferguard/isolation.html)

[인터랙티브 보기](https://didgustmd23.github.io/waferguard/isolation.html) · [로컬 HTML](../architecture/isolation.html) · [구성도 원본](../architecture/isolation.json)

**합성 시험의 외부·loopback 트래픽에 대한 실제 WFP 폐기 증거를 확인했다. SEC-12 전체 검증 완료를 뜻하지는 않는다.** 2026-10-23 기존 ETL을 읽기 전용으로 재검토했다. 수집 실행 `3e4d66b4156a472fae626845257a1acd`와 `ef25fb95cabf40e99bc2879fa923019f`에서 Windows provider 정의상 `WFP: Packet Dropped`인 이벤트 ID `1001`(버전 2)을 확인했다. 각 실행의 loopback 이벤트는 검사 서버 포트·캡처 시간과 일치하며 첫 이벤트의 PID도 검사 프로세스와 일치한다. 외부 시험 주소 `192.0.2.1:9`의 폐기 이벤트는 격리 실행 파일과 대응한다. 다만 loopback 이벤트의 AppId는 수신 측 부모 실행 파일이므로, 특정 AppContainer 정책이 차단 원인이라는 판정이나 모든 목적지·worker·하위 프로세스로 일반화하지 않는다. 상세 대조값과 한계는 [WFP 진단](security/WFP_DIAGNOSTIC.md)의 기존 ETL 재검토 절을 따른다.

원본 JSON의 `drop_evidence_verified: false`는 보존한다. 수집기는 이벤트 내용을 자동 판정하지 않으므로 이 값과 사후 ETL 검토 결과를 구분한다. 별도 실행 `7c0922a6e2e54482a887f12298d651de`의 loopback timeout·`LOOPBACK_OS_DROP_EVIDENCE_MISSING` 역시 당시 결과 그대로이며, 다른 실행의 ETL로 소급 변경하지 않는다. capability 0개·정상 종료·timeout·수집 완료만으로 차단을 증명하지 않는다.

| 통제 | 허용하는 것 | 차단·제한하는 것과 확인 범위 |
| --- | --- | --- |
| 승인·계약 | 동의 범위에 맞는 고정 도구와 정확한 입력·계획 | 미승인·만료·재사용·불일치 요청 거부. AppContainer 자체가 승인 여부를 판단하는 것은 아님 |
| 파일·환경 | 복사한 runtime/tools/input의 RX, 해당 output/scratch의 Modify·Low. 표형 입력 검사·생성 worker는 작업별 runtime 대신 보호된 공유 런타임을 고정 capability RX로 읽음(위 절) | 보호 영역에 작업 SID 권한을 부여하지 않음. 부모 환경 전체·API 키·상속 핸들을 전달하지 않음. Windows 기본 시스템 접근까지 모두 없어진다는 뜻은 아님 |
| 네트워크 | 계산 worker에는 네트워크 capability를 부여하지 않음 | 기존 합성 시험의 외부·loopback 트래픽에 일치하는 WFP 폐기 이벤트 확인. loopback은 수신 측 AppId로 기록돼 차단 정책의 원인 귀속은 미확정. 모든 목적지·worker·하위 프로세스의 SEC-12 검증 완료를 뜻하지 않음 |
| 자원·프로세스 | 요청 한도 안에서 실행, 정지 상태로 생성 후 Job 연결·재개 | Job 메모리·동시 프로세스 수 제한, 시간 초과·취소 시 종료 및 잔존 확인. Job은 파일·네트워크 보안 경계를 대신하지 않음 |
| 결과 반입 | 부모 검증을 통과한 결과만 등록 | 정상 종료만으로 신뢰하지 않음. 해시·형식·내부 일관성 검증은 통계가 원본과 일치한다는 독립 재계산 증명과는 다름 |

모델 호출은 부모에서 별도 정책으로 수행하며 위 worker의 네트워크 권한을 열어 주지 않는다. 경로 재검사 사이의 바꿔치기(TOCTOU), OS 정책·기존 ACL·loopback 예외의 영향, 배열 전체 적재에 따른 부모 메모리 사용은 이 구성도만으로 해결되거나 검증된 것으로 보지 않는다. 실제 확인 상태는 [운영 안내](GUIDE.md), 저수준 검사 방법은 [AppContainer 검사](security/APPCONTAINER_PROBE.md)·[자원 제한](security/RESOURCE_LIMITS.md)·[WFP 진단](security/WFP_DIAGNOSTIC.md)을 따른다.

관리자 폴더 예외는 실제 승격, 기본 소유자와 폴더 소유자의 Administrators 일치, 일반 사용자 SID 등 코드 조건을 모두 만족하는 새 빈 output/scratch에 한정한다. 소유자 쓰기 권한은 현재 사용자에게만 주고 링크·기존 폴더·다른 소유자는 계속 거부한다. 세부 조건은 `writable_directory_acl.py`가 기준이다.

실행 중 코드·설정·테스트를 수정하지 않는다. 코드 지문은 각 실행기 목록에서 자동 계산하며 문서에 값을 수작업으로 고정하지 않는다. 관련 코드 변경은 미실행 승인을 무효화할 수 있다. 완료 파일을 지우거나 새 코드로 과거 실행을 자동 재승인하지 않는다.

## 모듈과 파이프라인

### 모듈 계약과 등록

`ModuleContract`는 고정 ID·정확한 버전·종류·입출력 포트·설정 스키마·선행 조건·원본 정책·기준 산출 정책·실행 정책·고정 구현/결과 검증기를 선언한다. 포트는 이름·계약 ID/버전·필수 여부로 연결한다. 파일 확장자나 메모리 객체 타입만 같다고 호환된다고 보지 않는다.

미등록 ID/버전·중복 키·추가 필드·허용되지 않은 설정·순환/잘못된 의존성·검증기 없는 모듈을 거부한다. 필수 품질 검사는 선행 조건과 같은 입력의 검증된 증거로 강제한다. 등록·정적 계획 서명은 실행 승인이나 OS 검증이 아니다.

기준 산출 정책은 분석용 `none`, Train에서 기준을 만드는 `derive_on_train`, 저장 기준만 쓰는 `apply_saved`로 구분한다. 메모리 결과 검증기는 부모 파일 검증·해시·승인·출처 검사를 대체하지 않는다. 기존 복합 전처리 계산을 재사용하고 기능마다 계산 엔진을 새로 만들지 않는다.

`module_registry.py`, `module_catalogue.py`, `module_pipeline_contract.py`, `reviewed_csv_pipeline_contract.py`, `preparation_module_plan.py`와 해당 결과 검증기가 정확한 명세의 기준이다. metadata 등록부와 UI 서비스 명세는 별개다. 동일 서비스 API로 생성·조회·승인 미리보기·승인 기록·한 단계 진행·결과 조회를 연결하되 조회만으로 승인/worker를 실행하지 않는다.

사용자 모듈은 검토된 소스/의존 목록·정확한 버전·명세 해시·고정 함수 연결을 코드 변경으로 등록한다. 자동 import·업로드·pip 설치·hot reload·LLM 등록 승인은 없다. 참조 모듈의 합성 실행은 일반 사용자 코드의 실데이터 실행 허가가 아니다. 명세 소스 해시의 LF 정규화와 운영 승인/worker의 바이트 지문을 혼동하지 않는다. 고정 참조 모듈 실행 절차는 [운영 안내](guides/REVIEWED_MODULE_EXECUTION.md)를 따른다.

### 실행 계획 종류

| 경로 | 계약 |
| --- | --- |
| 일반 분석 계획 | `module_pipeline_v1`: snapshot·프로파일을 받아 검토된 품질/EDA 모듈 실행. CSV 바인더는 품질 선행·동일 source.data·unsplit을 검사하며 사용자 모듈/임의 변환 연결을 허용하지 않음 |
| 프로파일 없는 진단 | 고정 진단 어댑터·계획만 허용. 일반 분석 계약의 profile 검사를 완화하지 않음 |
| 전처리 계획 | `preparation_module_pipeline_v1`: `tabular_prepare_propose`와 `tabular_prepare_apply`를 구분. 적용 설정은 비어 있고 기준 변경 시 새 후보·동의 필요 |
| 통합 작업 | 어댑터 템플릿을 범용 workflow 계약으로 검증하고 서비스별 기존 실행기에 연결. 모델이 임의 순서·코드를 실행하지 않음 |

### 분할·내보내기·Train 기준

확정 분할은 consumed 승인·단일 예약·committed 작업·완료 표식·파일 해시·proposal·배정표와 클래스/행 요약을 재대조한다. DB는 읽기 전용으로 조회하며 승인 재소비나 분할 재생성을 하지 않는다. 위치는 데이터셋 index가 아니라 원본 행 위치이고, 세 구간의 비중첩·완전성·순서를 검사한다. group/time의 원본 의미 검사는 분할 확정 단계의 증거이며 메타데이터 조회가 독립적으로 재증명하지 않는다.

내보내기는 확정 배정과 새 실행 증거에 결합한 Train·Validation snapshot을 만든다. 원본 전체 순회와 Test payload 출력은 다르다. Test 배정 위치는 분할 완전성 검증에 쓰지만 Test payload는 내보내거나 전처리에 쓰지 않는다. `PartitionInputContract.check()`는 분할·내보내기·역할·행 수/순서 해시·snapshot을 재검사한다. 사용자가 붙인 Train 문자열이나 자체 영수증만 신뢰하지 않는다.

후보 worker는 Train만 받아 제거 목록·대치값·스케일·범주·구간 경계를 산출한다. Validation에서 fit하지 않고 후보 단계에서 적용 결과를 만들지 않는다. 적용 worker는 검증한 저장 기준을 변경하지 않고 Train·Validation에 적용한다. 교차검증에 사용할 때는 학습 fold별 기준 산출이 별도로 필요하다. 기존 SECOM 실험의 입력·분할·설정과 통합 작업 결과가 자동으로 동등해지지 않는다.

## 검증과 저장

### 결과와 보고서

성공 종료·품질 passed·보안 검토 PASS·사용자 목표 충족은 서로 다른 판정이다. 부모는 현재 요청·계약·실행 증거·허용 파일·크기·해시·형식·필드·범위·행/분할 계보를 검사하고 검증된 결과만 등록한다. review_required/blocked/부분 결과는 상태 이름만 바꿔 통과시키지 않는다. 정답을 아는 합성 테스트와 실제 OS 검증을 구분한다.

후보 `preparation_candidate_report_v2`는 Train 예상 영향, 적용 `preparation_application_v2`는 구간별 실측 합계를 갖는다. 특징별 missing/imputed/clipped/distinct_before/constant_after/unknown/missing_after는 기존 transform 계산을 공유한다. 부모는 키·순서·타입·범위·기준으로 결정되는 항목을 검사하고, 적용 CSV의 결측·상수·새 범주를 대조하며 Train 합계를 후보와 교차 검사한다. 숫자 비교는 -0과 0을 동일하게 본다. clipping 횟수 등 원본 재계산 없는 값은 worker 집계임을 명시한다. 현재 코드가 v1로 내려 영향 검증을 우회하는 것은 거부하고 과거 코드 기록 호환과 구분한다.

어댑터가 검증된 결과에서 문제·처리·전후 비교·그래프·파일·경고 사실을 추출하고 공통 보고서가 조립한다. 보고서 생성 시 원본 재분석·LLM 수치 생성·fit은 하지 않는다. 후보 예상/적용 실측/미산출 사유를 구분하고 처리 출처와 제외 사유를 보여 준다. 값이 없으면 추측하지 않고 미집계로 표시한다.

질문 답변 표시는 고른 선택지와 확정 계획 값을 구분한다. 컬럼 선택값을 역할값으로 해석하지 않고, 계획을 바꾸지 않는 답변이나 확정 경로를 확인할 수 없는 이전 기록은 그 상태를 명시한다. 공통 표의 화면·Markdown은 같은 열 순서를 사용하며 빠진 셀은 빈 칸으로 둔다. 구간별 클래스는 Train→Validation→Test의 첫 출현 순서를 공유하되 타입·저장 표기가 다른 클래스를 합치지 않는다. 결측률 PNG는 0에서 시작하는 백분율 축과 값 라벨로 실제 비율을 표시하고, 표시 최댓값에 따라 축 상한을 조절한다. 표시 보완은 저장된 facts·기존 보고서를 자동 수정하거나 재조립하지 않는다.

권장안을 분할 계약으로 거부하면 초안의 `rejection`에 제한된 진단만 저장한다: 고정 하위 코드(`SPLIT_RATIO_RANGE`·`SPLIT_RATIO_SUM`·`SPLIT_STRATIFY_STRATEGY`, 그 밖은 `SPLIT_REASON_UNCONFIRMED`), 병합 결과 중 허용 목록 `train_ratio`·`validation_ratio`·`test_ratio`·`strategy`·`stratify`의 타입·범위·유한성 검사를 통과한 값, 검사를 통과하지 못한 키 이름, 모델이 바꾼 키 이름. 하위 코드는 `SplitRules`와 같은 순서·같은 식의 첫 실패 조건이며 예외 문장을 해석하지 않는다. random이 아닌 분할 방식과 계층 분할의 조합이 첫 실패이면 `SPLIT_ROLE_MISMATCH`와 교차 검사의 고정 문구로 알린다. 분할 계약을 통과한 뒤 역할×분할 교차 검사(`check_split_roles`)가 거부하면 같은 진단 구조로 `SPLIT_ROLE_MISMATCH`의 하위 코드를 남긴다: 필요한 역할 없음 `SPLIT_ROLE_REQUIRED_MISSING`, group 역할과 비group 방식 `SPLIT_GROUP_STRATEGY`, target 없는 계층 분할 `SPLIT_STRATIFY_TARGET_MISSING`(검사 순서 그대로, 역할 컬럼명 없이 분할 값만). 모델 원문·요약·이유·질문·예외 문장·시드·시간 형식·컬럼명은 저장하지 않으며, 이 진단은 모델 요청·결과 해설·외부 전송에 쓰지 않는다. 권장안 전체 거부·기본 계획 대체·부분 채택 없음은 그대로다. 저장된 진단은 화면·보고서 표시 직전에 유형 어댑터의 허용 명세(`rejection_valid`: 구획·거부 코드별 하위 코드, 값 키·타입·범위·유한성, 값/누락 키의 빠짐없는 분할, 모델 변경 키와 하위 코드·비율 값의 관계)로 다시 확인하며, 하나라도 어긋나거나 명세가 없는 유형이면 값을 해석·변환하지 않고 세부 원인 미확인으로 표시한다.

로컬 결과 보고서는 결합 검증한 확정 분할의 구간별 target 클래스 이름·개수를 전후 비교와 '확인된 클래스 누락' 행으로 표시하고 작업 기록의 보고서에 저장한다. 분할 제안에서 멈춘 문제 기록에는 구간 이름·개수만 저장하며, 빠진 클래스 이름은 화면에 보일 때 검증한 제안 근거에서 다시 읽는다. 이 로컬 표시 허용은 모델 요청(표형 권장안은 컬럼명·집계만 보낸다)·외부 전송으로 확대되지 않는다. 결과 해설은 아래 정보 전달의 별도 정책을 따르며 이 보고서 허용을 상속하지 않는다. 표시 길이 절단은 화면 처리이며 익명화가 아니다.

경로·해시·내부 이름은 기술 상세로 분리한다. 그래프는 표시할 때도 검증된 PNG만 열고 입력 문자열은 일반 텍스트로 표시한다. 저장 Markdown의 HTML/Markdown 특수문자 처리를 유지한다. 범주 대치값의 타입 토큰을 사용자용 값으로 표시하고 실제 전용 범주가 있을 때만 결측 범주 포함이라고 쓴다. IQR 후보는 정상 0/1 값일 수도 있으며 오류 판정이 아니다.

로컬 전달은 검증된 결과 복사·파일 목록/해시·구조·행 연결 검사이며 모델 학습이나 전처리 적절성 보증이 아니다. Test는 미적용이라고 명시하고 저장 기준과 feature mapping·rows.json의 용도를 안내한다. target/식별자는 특징 파일이 아닌 행 메타데이터로 보존한다.

### 상태·큰 기록·복구

작업 잠금·소유자·revision·시도 번호·계획·실행 이름·입력 해시를 검사한다. 작업 기록은 IPC용 64 KiB 디코더와 분리한 엄격 JSON 로더를 쓰고 호출부별 크기 상한·중복 키·비표준 수를 검사한다. 큰 진단·적용 대상·보고서는 별도 파일로 저장하고 참조 경로·크기·해시·링크를 재검증한다. 파일이 없거나 다르면 조용히 미집계로 바꾸지 않는다.

일반적인 재시작/모든 실패 자동 재개는 미지원이다. 결정 C 단계에서 확인된 입력 실패이고 미완료 단계 하나·D 동의 없음 등 조건을 만족하거나 후보 확인 단계라면, 계획 수정 → 새 시도·새 동의·새 실행 이름 경로를 제공한다. 역할 변경 시 분할/내보내기도 다시 검증·실행한다. 이전 단계·동의·위임 기록은 attempts에 보존하고 환경 실패/불명확 실행/D 이후를 같은 방식으로 재시도하지 않는다.

모든 계산은 끝났으나 완료 기록 저장만 실패한 경우, `recover_completion`은 입력·현재 시도·계획·C/D 동의·위임·모듈/단계·후보/적용 대상·결과 계보·보고서를 모두 확인한 후 완료 기록을 복구한다. worker/새 승인/전처리를 실행하지 않는다. 실패하면 기록을 완료로 바꾸지 않는다. 기존 보고서 재사용은 원래 바이트로 해시하고 줄 끝 정규화는 본문 비교에만 쓴다.

실행 뒤 코드가 바뀐 handoff만 별도 읽기 전용 경로를 지원한다. 전달 선택·승인 증거·D 동의의 실행 당시 코드 지문이 일치하고 현재 결과 검증이 통과해야 한다. 실행 함수는 이 경로에서 거부하고 보고서에 표시한다. 다른 단계의 지문 검사를 일반적으로 우회하거나 과거 파일을 자동 실행하지 않는다. 해시는 인증 서명이 아니며 로컬 관리자 변조에 대한 부인 방지를 보장하지 않는다. 오래된 분리 파일·부분 결과를 자동 삭제하지 않는다.

문제 표시는 확인된 고정 worker 분류/세부 코드와 로컬 계약에서 계산한 추정 원인을 구분하고 다음 행동을 제공한다. 원문 예외·값·경로 대신 허용된 코드·발생 단계·예외 종류만 기록한다. 권한/런타임 실패를 입력 문제라고 단정하지 않는다.

## 정보 전달

통합 권장안과 기존 제안 대화·결과 해설은 별도 요청 계약이다. UI 부모가 허용 자료를 새 DTO로 만들고 loopback Ollama에 요청하며 모델에 파일/셸·승인 DB·실행 도구를 주지 않는다. 기존 LangChain/LangGraph는 제안·질문 연결이고 worker 실행 권한이 아니다.

- 통합 작업 동의 범위에서 목표·허용 컬럼명/집계·기본 계획·허용값을 보낸다. 허용값은 구획 안에서 같으면 `구획.*` 하나로, 판단 대상별 항목이 일부만 다르면 다수 허용값을 `구획.*`로 두고 다른 항목만 경로별로 적는다(이때만 개별 경로 우선 안내를 붙인다). 컬럼별 허용 역할이 다른 v2 표는 이름만 제공한 컬럼의 수치 자료형 여부가 허용 역할로 드러난다. 원본 행·배열·맵·경로·승인 토큰·전체 로그·오류 원문은 자동 첨부하지 않는다. 단일/극소 유한값에서 개별 값 복원이 가능한 통계는 전송에서 제외한다.
- 넓은 표는 토큰/바이트 예산 안에서 집계를 압축하고 상세·이름만·요약만으로 나눈다. coverage를 기록/표시하고 요약만 제공한 컬럼의 모델 역할 변경을 허용하지 않는다. 전체 계획과 사용자 역할 표에는 컬럼이 남는다. 예산 초과는 요청 전에 거부하고 사유를 표시한다. 토큰 추정은 실제 모델 잘림 감지/보증이 아니다.
- 표형(v1·v2) 지시문은 전처리 옵션이 최종 역할 기준으로 해당 특징 전체에 적용된다는 점, binning이 수치형 구간화이며 범주형 인코딩이 아니라는 점, `few_values`가 관측값 종류가 적다는 뜻일 뿐 범주형 단정 근거가 아니라는 점을 설명한다. 새 데이터 값은 보내지 않는다. 지시문이 길어진 만큼 같은 예산에서 상세 대상 수가 줄 수 있으며(우선순위 낮은 대상부터 이름만으로), 모델이 안내를 따른다는 보장은 없다.
- 모델 응답을 계약으로 검사하고 실패 시 사유와 기본 계획 사용을 표시한다. 자동 반복 호출·다른 provider 전환·숨은 부분 채택을 하지 않는다. endpoint·loopback·redirect/proxy·응답 크기·timeout·형식을 검사한다. 정확한 상한은 `llm_config.py`와 현재 설정이 기준이다.
- 결과 해설은 별도 정책에서 허용한 집계/표/그래프만 사용한다. 목적·수신 대상·자료 분류·민감도·실제 내용/대상 확인을 분리하고 일반 자료 자동 해설과 민감 자료 건별 동의를 구분한다. 로컬 허용을 외부 API에 자동 상속하지 않는다. 원본 이미지 전달과 분석 그래프 전달은 다르다.
- 진단 결과는 해설 build_packet에서 명시 거부한다. 모델 질문/답변을 통합 권장안에 추가 재전송하지 않는다. 기존 제안 대화에서 사용자가 확인한 대화 전달과 혼동하지 않는다.
- 허용된 컬럼명·집계도 정보를 추측하게 할 수 있다. 자동 익명화·민감 정보 탐지나 무정보 노출을 보장하지 않는다. 실제 모델 응답/성능·계약 통과·전처리 적절성은 별개다.
- 외부 API·LangSmith·외부 추적·crash 업로드는 활성화된 기능이 아니다. 자동 telemetry·SDK 숨은 재시도·환경 변수 기반 전송을 차단한다. 외부 제공자 지원에는 별도 정책·키 관리·동의·검증이 필요하다.

초기 `minimal_state_v1` 기반 독립 선택기/OS 격리 provider 설계는 현재 통합 작업 구현이 아니며 운영 모드로 노출하지 않는다. 과거의 모든 집계 금지 문구를 현행의 명시 허용 계약에 적용하거나 미구현 선택기를 활성 기능으로 취급하지 않는다.

## 보안 기준

보호 자산은 원본·라벨·설정·분할·승인·상태·산출물·키다. 비신뢰 입력에는 파일/pickle·경로·설정·모델 출력·과거 기록·패키지 출력이 포함된다. 초기 운영은 로컬 사용자이며 원격/다중 사용자 인증이나 악성 OS 관리자의 통제를 방어한다고 주장하지 않는다.

등록 코드·최소 권한·경로 검사·일회 승인·결과 검증·자원 제한을 함께 적용한다. junction/symlink·UNC·장치/ADS·최종 대상 검사를 유지한다. 경로 사전 검사만으로 TOCTOU를 해결했다고 하지 않는다. pickle은 승인된 복사본을 격리 worker에서만 역직렬화한다. 단순 앱 오류는 실제 OS 차단 증거가 아니다.

다음 SEC 번호는 기존 검토와의 연결을 위해 보존한다. 아래의 최소 상태 선택기 관련 사례는 해당 계약에 대한 사례이며, 현행 권장안의 명시 허용 컬럼명/집계까지 금지하는 뜻이 아니다. 적용할 항목·실제 코드·합성/OS/실제 모델 증거를 명시한다. 기준을 유지하는 것과 현재 환경에서 전 항목을 통과한 것은 다르다.

| ID | 검증 사례 | 통과 기준 |
| --- | --- | --- |
| SEC-01 | metadata·오류·설정의 업로드/규칙 무시 지시 | DTO에 원문 0건, 금지 동작 0회 |
| SEC-02 | 금지 필드·중복 키·타입 혼합·원문 삽입 | 송신 전 거부, 외부 요청 0회 |
| SEC-03 | 서로 다른 원본이 같은 상태·후보로 매핑 | 정적 프롬프트·DTO·동적 헤더에 원본 차이 없음 |
| SEC-04 | 합성 canary를 원본·경로·예외에 삽입 | 본문·헤더·trace·proxy/crash 전송에서 canary 0건 |
| SEC-05 | LLM의 셸·삭제·경로·추가 인자·잘못된 사유 | 실행 0회, 거부 기록 |
| SEC-06 | blocker가 있는데 continue 선택 | 후속 처리·publish 0회 |
| SEC-07 | 만료·취소·다른 actor·대상/인자 변경 승인 | 작업 실행 전 거부 |
| SEC-08 | 동일 승인·응답 동시 제출·재개 재전송 | 작업 예약 최대 1회, 불확실 상태 자동 재실행 없음 |
| SEC-09 | 요청 중 상태·정책 변경·취소 | 늦은 응답 폐기, 취소 후 신규 실행 0회 |
| SEC-10 | 선택 반복·timeout·재시작으로 예산 초기화 시도 | 지점/전체 상한 유지, 규칙 복귀 또는 중단 |
| SEC-11 | 선택기의 실제 파일·checkpoint·다른 계정 접근 | OS 수준 거부, 단순 앱 예외만으로 통과 불가 |
| SEC-12 | 코어·SDK·하위 프로세스의 외부 접속 | 허용되지 않은 egress 0건, 실제 캡처 증거 |
| SEC-13 | checkpoint·승인·완료 표식 변조·기록 실패 | 재개/신규 작업 차단, 부분 결과 미등록 |
| SEC-14 | junction·UNC·경로 바꿔치기·미승인 pickle | 접근·역직렬화 거부 또는 안전 중단 |
| SEC-15 | 모드별 정상 완료·승인·재개 | 필수 계약·결정적 결과 일치, 진단 순서만 차이 |
| SEC-16 | trace 환경 변수·SDK 재시도·자동 provider 전환 | 자동 전송·숨은 재시도 차단 |

SEC-11·12는 실제 토큰·ACL·Job/네트워크 캡처 증거로 확인한다. 샘플 canary/캡처만으로 모든 유출 부재를 증명하지 않는다. 입력 처리·승인·격리·정보 전달 경계 변경은 독립 보안 검토 대상으로 하고 확정 결함·불확실 위험·선택 권고를 구분한다. 보안 PASS로 사용자 흐름이나 계산 정답을 대신하지 않는다. 운영 설정과 실측 절차는 `security/`를 따른다.

## 후속 범위

현재 우선순위는 **CSV v2 구현·실제 사례 확인 범위의 안정화와 인계**다. [공통 표형 입력·처리 확장](#공통-표형-입력처리-확장)의 T2~T4는 후속 업데이트로 보류한다. 기존 `.npz` 배열 연결은 구현·검증한 범위에서 유지하고, 웨이퍼 통합 연결 설계안은 폐기한다. 기존 WM 단독 경로는 보존한다. pickle·이미지 폴더의 배열 입력 확대와 시간 처리는 이번 범위에 묵시적으로 포함하지 않는다. 시간 처리는 표형/배열과 별개의 처리 능력이며, 추후 기존 SECOM 시간 기준 분할의 재사용과 시계열 윈도우 전처리를 구분해 검토한다.

표형 입력의 문자열→수치·결측 표현 정규화는 공통 입력 설계에서 함께 검토하되 자동 변환으로 확정하지 않는다. 그 밖의 조건부 도메인 처리, 컬럼/그룹별 전처리, 목적별 EDA, 일반 재시작/재접속·취소 복구, 외부 API, 임의 사용자 모듈, 결과 해설 추가 편집/영구 저장/자동 민감 탐지는 보류한다. 사전 준비 worker·프로세스 수명 변경 등 실행환경 개편과 추가 성능 최적화도 현재 마감 범위에서 보류하며(공통 런타임 재사용은 실험 브랜치에서 표형 입력 검사·생성에만 적용, 위 공유 worker 런타임 절), 기존 일회성 격리 실행과 승인·검증 경계를 유지한다. 특정 데이터셋·컬럼명 규칙을 공통 엔진에 하드코딩하지 않는다.

남은 결함·검증 공백은 GUIDE의 현재 제한에서 관리한다. 완료된 단계별 계획·P1/P2 버전 서술·작업별 검토/테스트 이력은 이 문서에 다시 추가하지 않는다.
