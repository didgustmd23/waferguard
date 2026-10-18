# 사용자 모듈의 초기 등록 체계

2026-10-17. 검토한 코드를 고정 목록에 연결하는 명세 검사·참조 등록부·합성 예제를 구현했다. **임의 Python 업로드·자동 import·실데이터 worker 실행 기능은 아니다.**

## 구현 구성

| 파일 | 역할 |
| --- | --- |
| `src/agents/module_catalogue.py` | 고정 ID·정확한 버전·명세 해시·선언한 소스 목록·보안 계약 검사 |
| `src/agents/module_catalogue_cli.py` | list/check. 확장 코드 import·통계 계산·자동 등록 없음 |
| `src/preprocessing/extensions/missing_row_histogram.py` | 설정 검증·계산·결과 검증·ModuleContract를 갖춘 참조 구현 |
| `configs/agents/modules/missing_row_histogram.json` | 명세·코드/의존 소스 해시·검토 범위 |
| `configs/agents/pipelines/user_module_example.json` | 품질 검사 후 예제 모듈을 배치한 정적 계획 예시 |
| `tests/agents/test_module_catalogue.py` | import 경계·변조·설정·결과·정적 계획·실행 거부 검사 |

기존 내장 모듈 등록부와 worker 파일 목록은 바꾸지 않았다. `build_reviewed_registry()`는 내장 6개와 참조 모듈 1개로 별도 봉인 등록부를 만든다. 이를 명시적으로 전달한 정적 계획에서만 새 모듈을 조회할 수 있다.

## 참조 모듈의 기능

- ID: `user_missing_row_histogram`, 버전: `1.0.0`, 종류: analysis.
- 입력: `tabular_data` v1. 출력: `missing_row_histogram_result` v1.
- 설정: `include_complete`는 bool이며 기본값은 true. 결측 특징이 0개인 행도 빈도표에 포함할지 결정한다.
- 수치/범주 특징만 분석한다. 라벨·ID·그룹·시간은 제외한다.
- 출력: `missing_feature_count`와 `row_count` 두 정수 컬럼. 실제 값 예시는 테스트의 합성 입력만 사용한다.
- 원본 보존, 기준 산출 없음, 네트워크 차단, 명시 실행 승인, 작업별 자원 상한을 선언한다.

`review.kind: reference_example`은 프로젝트에서 작성한 합성 참조 구현이라는 구분이다. 운영 보안 감사를 받았다거나 사용자 실데이터 실행 승인이 있다는 뜻이 아니다. `test_suite`는 검증 대상의 고정 이름이며 check 명령이 테스트를 자동 실행하지 않는다.

## 검사 방법

```powershell
python -m src.agents.module_catalogue_cli list

python -m src.agents.module_catalogue_cli check `
  --module-id user_missing_row_histogram `
  --version 1.0.0

python -m unittest tests.agents.test_module_catalogue -v
```

정상 검사는 `module_spec_verified`다. `module_imported: false`, `data_read: false`, `native_execution_enabled: false`가 표시된다. 소스 파일은 검사하지만 데이터셋은 읽지 않는다. 실제 모듈은 이 CLI에서 import하지 않는다.

개발용 신뢰된 초기화 코드에서 `build_reviewed_registry()`를 호출하면 참조 구현을 고정 Python import로 연결한다. 이는 일반 로컬 코어 호출이며 격리 실행이 아니다. 알 수 없는 사용자 코드를 이 경로로 시험하지 않는다. `validate_pipeline(config, registry)`로 계획을 만들 수 있지만, 위 예제를 기존 module_pipeline_cli로 실행하면 거부한다.

## 새 사용자 모듈을 추가하는 절차

1. 별도 개발 작업에서 모듈 코드와 입력·출력·허용 설정·선행 조건·결과 검증 명세를 작성한다.
2. 코드와 직접/간접 의존 파일을 검토한다. import 시 부작용·파일/네트워크 접근·원본 변경·무제한 자원·모델 학습이 없는지 확인한다. 필요한 의존 코드는 목록에서 누락하지 않는다.
3. 정답을 아는 합성 데이터로 정상·예외·원본 보존·설정/결과 검증을 테스트한다. 미검토 코드를 호스트에서 실행하지 않는다.
4. 검토 담당자가 고정 목록·정확한 버전·명세 해시·고정 함수 연결을 **코드 변경으로** 추가한다. JSON에 이름이나 Python 경로를 적는 것만으로 등록되지 않는다. 현재 목록은 참조 예제 1개이며 새 항목은 개발 작업이 필요하다.
5. 실제 실행이 필요하면 worker 고정 복사 목록·요청 인자·파일 결과 계약·부모 검증기·승인 scope를 별도로 연결하고 합성 격리 검증을 한다.
6. 그 이후 사용자가 입력·설정·자원 한도를 확인하고 새 실행 승인을 받아야 실데이터 실행할 수 있다.

동작이 바뀌면 버전을 올리고 의존 목록·명세·테스트·고정 등록을 재검토한다. 아직 hot reload·플러그인 검색·pip 자동 설치·LLM 등록 승인·웹 업로드는 구현하지 않았다.

## 해시와 신뢰 범위

명세와 소스의 해시는 `utf8_lf_sha256` 방식이다. UTF-8 코드의 CRLF만 LF로 정규화하므로 Git 체크아웃의 줄바꿈 차이로만 등록이 실패하지 않는다. 공백·주석을 포함한 나머지 내용은 변경으로 감지한다. 운영 worker의 원시 바이트 기반 코드·승인 해시는 변경하지 않았다.

사용자 JSON에서 source_files를 추가하거나 소스 해시를 바꿔도, 고정 명세 해시와 고정 파일 목록이 다르면 거부한다. 해시·검토 목록을 자동으로 갱신하는 승인 우회 명령은 제공하지 않는다. 기준을 바꾸려면 명시적인 코드 검토와 고정 목록 변경이 필요하다.

이 검사는 코드의 안전성·수학적 정답·외부 라이브러리 전체를 인증하지 않는다. 이미 실행 중인 신뢰된 프로젝트 Python 환경을 전제로 하며, 로컬 관리자가 등록 코드까지 바꾸는 공격이나 import 시 파일 경합을 막는 OS 보안 경계가 아니다. 따라서 사용자 코드의 실제 실행을 호스트의 등록 검사로 대체하지 않는다. 이후 실제 worker 연결에서도 AppContainer·Job·고정 복사/검증을 유지해야 한다.

## 진행 상태

- 완료: 명세 형식·고정 등록 검사·기존 계약 재사용·참조 계산/검증·정적 계획 합성 예제.
- 미구현: 일반 사용자 파일의 등록 신청/검토 이력 저장, 사용자 모듈의 실제 worker 실행 연결.
- 검증: 신규 8개를 포함한 관련 단위 테스트 27개 통과. 실제 데이터·AppContainer는 실행하지 않았다.

후속 갱신: [고정 합성 참조 모듈 실행](../guides/REVIEWED_MODULE_EXECUTION.md)을 별도 승인 CLI에 연결했다. `module_catalogue_cli`는 여전히 검사만 수행하며 확장을 import하지 않는다. 합성 worker 연결이 사용자 모듈의 일반 실데이터 실행 허용을 뜻하지 않는다.

추가 설계: [검토된 CSV 모듈의 범용 파이프라인 연결](REVIEWED_CSV_PIPELINE_DESIGN.md)에 메타데이터 등록부·신뢰된 설정 스키마·일반 결과 검사와 실제 CSV 연결의 승인 조건을 정의했다. 현재 [메타데이터 등록기](METADATA_REGISTRY.md)와 [CSV 실행 계약 바인더](REVIEWED_CSV_PIPELINE_BINDING.md)를 추가했다. 아직 격리 worker를 실행하는 기능은 없다.
