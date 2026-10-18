# 모듈 파이프라인 설정·정적 검증

구현일: 2026-10-16. 모듈 개발 3~4번의 설정 형식·순서·입출력 연결 검증이다. 이 파일의 정적 검증은 데이터를 실행하지 않는다. 별도 [승인·worker 연결](../guides/MODULE_PIPELINE_EXECUTION.md)은 CSV 분석·검사에 한해 구현·합성 검증했으며 새 진입점의 실제 OS 실행은 확인 전이다.

## 파일 구성

- `src/preprocessing/pipeline.py`: 엄격한 JSON 읽기·등록부 조회·설정 검증·순차 계획 생성.
- `configs/agents/pipelines/tabular_eda_example.json`: 품질 검사 → 결측 분석 → 상관분석 예제. 데이터 경로·컬럼명·원본 값은 포함하지 않는다.
- `tests/agents/test_module_pipeline.py`: 정상 설정·미등록 ID·순환·계약 불일치·파일 제한·계획 식별값 테스트.

## 설정 구조

```json
{
  "schema_version": "module_pipeline_v1",
  "pipeline_id": "missing_only",
  "input": {
    "contract_id": "tabular_data",
    "contract_version": 1,
    "partition": "unsplit"
  },
  "steps": [
    {
      "step_id": "missing",
      "module_id": "eda_missing",
      "version": "1.0.0",
      "inputs": {"data": "source.data"},
      "settings": {},
      "depends_on": []
    }
  ]
}
```

| 항목 | 의미 |
| --- | --- |
| `pipeline_id` | 로컬 계획 이름. 파일 경로·명령이 아님 |
| `input` | 공통 입력의 논리 계약과 선언한 구간. 초기에는 tabular_data v1만 지원 |
| `step_id` | 이 계획 안에서 고유한 단계 이름. 같은 모듈을 여러 번 선택해도 단계 ID는 달라야 함 |
| `module_id`, `version` | 등록된 기능과 정확한 버전. 임의 함수 경로·latest 금지 |
| `inputs` | 입력 포트와 출처 연결. 최초 입력은 `source.data`, 앞 단계 출력은 `<step_id>.<port>` |
| `settings` | 모듈에서 허용한 설정. 생략 시 등록된 기본값 검증 |
| `depends_on` | 앞서 완료되어야 하는 단계 ID. 생략 가능하며 출력 참조도 의존성에 자동 포함 |

단계는 `steps`에 적은 순서 그대로 실행하도록 계획한다. 자동 재정렬하지 않는다. 모든 EDA는 원본을 바꾸지 않으므로 결측 분석 다음의 상관분석도 같은 `source.data`를 읽는다. `missing.result`는 결측 통계 표이지 변환된 원본이 아니므로 다음 EDA의 데이터 입력에 연결하면 계약 불일치로 거부한다.

품질 검사는 필요에 따라 포함할 수 있다. 이를 선택하지 않아도 EDA 코어의 필수 입력 구조 검사는 생략되지 않는다. 배포 예제는 품질 검사를 먼저 배치했다. 품질 결과가 검토 필요·차단이면 이후 실행을 중단해야 한다는 정책을 계획에 고정하며 설정으로 해제할 수 없다. 실제 상태 확인·중단 처리는 후속 실행 관리자 연결에서 강제한다.

## 검증 규칙

- 설정은 64 KiB 이하, 단계는 1~32개. 알 수 없는 필드·중복 JSON 키·NaN·Infinity는 거부한다.
- 단계 ID 중복·예약어 `source`·미등록 모듈·미지원 버전·허용되지 않은 설정을 거부한다.
- 필수 입력 포트 누락·미등록 포트·존재하지 않는 출력·입출력 계약 ID/버전 불일치를 거부한다.
- 자기 참조·뒤 단계 참조·순환·중복 의존성을 거부한다. 임의 DAG 실행이나 병렬 스케줄링은 하지 않는다.
- 모듈 선행 조건은 지원하는 고정 검사 코드만 허용하고 계획의 계약 정보에 보존한다. 설정만 보고 실제 검사가 통과했다고 간주하지 않는다.
- Train 전용 기준 산출 모듈이 연결되는 경우 `partition=train`이 아니면 거부한다. 현재 내장 목록에는 변환 모듈이 없으며, 선언만으로 실제 행의 Train 역할을 증명하지 않는다.

## 결과와 식별값

`PipelinePlan`에는 순서·연결·입력 구간·모듈 계약·사용자 설정·기본값을 적용한 유효 설정을 보존한다. 문자열 JSON과 tuple로 저장해 원래 설정 dict가 바뀌어도 기존 계획은 바뀌지 않는다. `signature`는 이 정보를 SHA-256으로 식별하며 키 순서에는 영향을 받지 않는다.

이 식별값은 **설정 계획 식별값**이다. 데이터 출처·snapshot·코드 해시·OS 격리·승인에 결합된 실행 ID가 아니며 보안 서명도 아니다. 실행 연결 단계에서 기존 실행 관리자의 증거·승인 범위에 함께 결합해야 한다. 입력 파일 경로·원본 데이터·API 키는 설정에 추가하지 않는다.

## 확인 방법

프로젝트 루트에서 실행한다. 아래 명령은 설정만 읽고 데이터나 worker는 실행하지 않는다.

```powershell
python -c "from src.preprocessing.pipeline import load_pipeline; p = load_pipeline('configs/agents/pipelines/tabular_eda_example.json'); print('설정 검증 완료:', p.pipeline_id); print('단계:', [s.step_id for s in p.steps]); print('계획 식별값:', p.signature)"
python -m unittest tests.agents.test_module_pipeline -v
```

검증된 계획의 승인·상태·취소·격리 실행 연결은 [별도 실행 문서](../guides/MODULE_PIPELINE_EXECUTION.md)를 따른다. 새 실행 경로의 실제 OS 검증은 확인 전이다. 과거 보고서·노트북·데이터와 현재 고정 CLI는 변경하지 않는다.
