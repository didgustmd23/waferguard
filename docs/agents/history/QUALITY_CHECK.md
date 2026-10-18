# 범용 품질 검사와 프로파일 기준

> 초기 구현·실행 검증 이력입니다. 현재 상태와 권장 진입점은 [에이전트 문서 안내](../GUIDE.md)를 따릅니다. 당시 명령과 결과는 이력 보존용이며 현재 계약으로 자동 재개된다는 뜻이 아닙니다.

갱신일: 2026-10-15. `src/preprocessing/quality.py`는 원본을 변경하지 않고 품질 상태를 계산한다. `src/agents/adapters/quality.py`는 같은 입력 어댑터를 사용하여 `run_audit`에 연결한다. 에이전트 없이 `inspect_quality(TabularInput, QualityRules)`를 호출할 수도 있다.

2026-10-16 후속 연결: [snapshot 품질 검사](../guides/SNAPSHOT_AUDIT.md)에 별도 승인 CLI·AppContainer+Job worker·부모 결과 검사·검토/차단 보고서 보존을 구현했다. 기존 코어와 EDA 소스는 변경하지 않았다. 합성 테스트를 통과했으며 실제 audit OS 경로 확인은 남아 있다.

## 수행 흐름

~~~text
입력·프로파일 변경 확인
          ↓
로더 → 공통 표형 입력
          ↓
구조 → 특징 → target → 복합 ID → group·timestamp 검사
          ↓
로컬 quality.json·quality.md 저장
          ↓
passed / review_required / blocked_input 반환
~~~

검사는 컬럼 역할과 데이터 구조를 먼저 확인한다. 구조가 잘못되면 후속 컬럼 검사는 중단한다. CSV 읽기 계약 오류도 원문 예외를 복사하지 않고 고정 코드로 보고서에 남긴다. 파일 권한·I/O 등 실행 오류는 실행기의 실패 처리로 전달한다.

## 검사와 기본 판정

| 검사 | 기본 처리 | 비고 |
| --- | --- | --- |
| 빈 표·중복 컬럼·필수 컬럼 누락 | 차단 | 통지 수준으로 우회 불가 |
| numeric 자료형 오류·무한대 | 차단 | bool을 수치 특징으로 취급하지 않음 |
| 결측률이 기준을 초과 | 검토 필요 | 기본 기준 0.5, 정확히 50%는 초과 아님 |
| 관측 값 1개·전체 결측 특징 | 검토 필요 | 삭제하지 않고 후보로 기록 |
| target 결측 | 검토 필요 | target 역할이 없으면 검사 생략 |
| 지정한 허용 라벨 이외 값 | 차단 | 클래스 수·양성 라벨은 고정하지 않음 |
| 복합 ID 중복·ID 결측 | 검토 필요 | 중복은 해당 행 전체를 계산, 결측 ID는 중복 계산에서 제외 |
| group 결측 | 검토 필요 | 그룹 유효성 검사이며 분할 가능성 검사는 후속 범위 |
| timestamp 결측 | 검토 필요 | 원본 값 유지 |
| 문자열 시간의 파싱 형식 미지정 | 검토 필요 | 날짜 형식을 임의 추측하지 않음 |
| 명시한 시간 형식에 맞지 않는 값 | 차단 | 파싱은 검사에만 사용하고 컬럼은 변경하지 않음 |

숫자 라벨 1과 1.0은 동일하게 취급하고 bool True와 숫자 1은 구분한다. 공백 문자열도 결측으로 검사한다. 원본 행·미지 라벨의 실제 값은 findings에 저장하지 않는다. 컬럼명·건수는 로컬 근거이므로 LLM에 전달하지 않는다.

## 프로파일 설정

기존 프로파일에 선택 항목 `quality_rules`를 추가한다. 생략하면 기본 기준을 사용하며 기존 프로파일 식별값은 보존한다. 기본값과 다른 기준은 프로파일 서명·실행 계획·승인 대상에 반영한다.

~~~json
"quality_rules": {
  "missing_ratio_threshold": 0.5,
  "high_missing": "review",
  "constant_feature": "review",
  "all_missing_feature": "review",
  "allowed_labels": ["normal", "review", "reject"],
  "target_missing": "review",
  "duplicate_id": "block",
  "identifier_missing": "review",
  "group_missing": "review",
  "timestamp_missing": "review",
  "timestamp_format": "%Y-%m-%d"
}
~~~

`ignore/review/block`은 해당 품질 기준의 통지 수준이다. 허용 라벨은 JSON 배열 또는 null이며 null은 특정 라벨 목록으로 제한하지 않는다는 뜻이다. target이 없는 프로파일에 라벨 목록을 지정하면 설정 오류로 거부한다. ID 여러 컬럼은 하나의 복합 키로 취급한다. 시간 역할이 없으면 시간 검사도 생략한다.

WM EDA 프로파일의 `target_missing=ignore`는 accepted 미라벨 행도 metadata EDA에 포함하는 기존 도메인 정책을 표현한다. 학습용 라벨 보충이나 미라벨 학습 허용을 뜻하지 않는다. WM 라벨 목록은 해당 프로파일에만 있고 공통 코어에는 없다.

## 실행기 연결과 중단 의미

입력 어댑터의 `audit_tool()`이 `run_audit`를 등록한다. `registered_tool()`은 기존 `run_eda` 등록이다. 두 도구는 같은 로더·원본 식별값·프로파일을 사용한다.

~~~python
adapter = create_eda_adapter(profile, input_path)
tools = [adapter.audit_tool(), adapter.registered_tool()]
# 실제 실행 계획에는 preflight·검토·계약·보고서 도구도 등록해야 한다.
~~~

| 품질 결과 | 실행기 처리 |
| --- | --- |
| passed | 도구 내용 검증·완료 표식·다음 단계 |
| review_required | `TOOL_REVIEW_REQUIRED`, blocked_input으로 중단 |
| blocked_input | `TOOL_INPUT_BLOCKED`, blocked_input으로 중단 |

검토 필요도 자동 승인하거나 EDA로 진행하지 않는다. 검사 근거를 확인하고 입력·기준 변경 또는 별도 검토 재개 정책을 결정해야 한다. 현재 불명확한 작업·품질 중단을 자동 해제하는 기능은 없다. 중단 보고서는 attempt 폴더에 보존하지만 완료 표식은 생성하지 않는다. 설정을 바꾸면 새 계획을 사용하며 기존 승인·완료 결과로 진행하지 않는다.

이 단계의 결측률·상수 검사는 데이터 품질 진단이다. 센서 제거·대치·스케일링·특징 선택의 학습은 수행하지 않는다. 학습 입력 변환은 분할 후 Train 또는 각 CV 학습 fold 내부에서 fit하도록 후속 구현한다.

## 검증과 남은 범위

~~~powershell
python -m unittest tests.agents.test_quality -v
python -m unittest discover -s tests/agents -v
~~~

독립 합성 입력으로 50% 경계·상수·전체 결측·미지 라벨·결측 정책·숫자/bool 라벨 구분·복합 ID·그룹·시간 형식·설정 변경·원본 미변경·CSV 읽기 계약 오류·품질 중단 후 EDA 차단을 검증한다. 기존 범용 CSV 실행 테스트도 실제 audit 도구를 사용하도록 바꿨다.

2026-10-15 실행 결과: 품질 테스트 11개를 포함한 전체 73개 중 72개 통과, OS 심볼릭 링크 생성 권한 부족으로 1개 건너뜀. 실제 사용자 데이터는 처리하지 않았다.

시계열 정렬·그룹별 클래스 지원·분할 비중첩·도메인별 중복 연결 그룹·학습용 fit/transform은 후속 범위다. 실데이터 CLI·OS 격리·DB 인증 무결성·LLM 최소 정보 게이트의 완료 상태는 바뀌지 않는다.
