# 모듈 계획의 승인·격리 실행 연결

구현일: 2026-10-16. 계약·등록부·정적 계획을 기존 실행 관리자에 연결했다. 자동 테스트는 OS transport를 대체하며, 별도 사용자 실행의 합성 정상·검토 중단 경로도 로컬 기록으로 교차 확인했다. 실데이터·변환 실행·모든 보안 실패 경로를 검증했다는 뜻은 아니다.

## 실제 실행 확인 기록

2026-10-16 사용자 실행 후 읽기 전용으로 확인했다. worker 재실행·승인 변경·DB 쓰기는 하지 않았다. 상세 통계 대신 고정 상태·무결성·종료 증거만 기록한다.

| 항목 | 정상 예제 | 검토 중단 예제 |
| --- | --- | --- |
| 실행 이름 | `module-demo01` | `module-review01` |
| 작업 ID | `2edf0078268f4026a89af9f0c5e64a88` | `07a188a504114f9e9fb1cf23b21268bb` |
| 실행 상태 / 품질 상태 | completed / passed | blocked_input / review_required |
| 실제 결과 단계 | quality → missing → correlations | quality만 |
| 승인·예약 | 승인 1회 소비·예약 1건 | 승인 1회 소비·예약 1건 |
| 작업 기록 | preflight·실행 모두 committed | 실행 failed, 완료 커밋 없음 |
| 완료 표식 | 존재, ARTIFACT_REUSABLE | 없음 |
| 후속 분석 결과 | 선택한 표만 존재 | 부모·worker 모두 missing·correlations 폴더 없음 |
| 종료·정리 | 종료 코드 0·활성 프로세스 0·Job/프로필 정리 기록 정상 | 동일 |
| 기록된 실행 시간 | 13.141초 | 12.734초 |
| 기록된 최대 Job 메모리 | 94.01 MiB | 93.93 MiB |
| 요청 자원 | 전체 작업 60초·2048 MiB·동시 프로세스 2개 | 동일 |
| 교차 검사 | 26개 모두 통과 | 24개 모두 통과 |

정상 실행은 현재 코드·계획·snapshot·요청/응답·부모 결과·완료 해시·worker 결과 복사본이 일치했다. 검토 예제는 `HIGH_MISSING_RATIO`가 원인이며 품질 보고서와 중단 기록이 일치했다. 품질 미통과로 작업을 failed로 남긴 것은 OS 실행 장애가 아니라 의도한 처리다. 이력의 누적 프로세스 생성 수는 동시 프로세스 제한과 구분한다.

로컬 근거는 `logs/agents/module_pipeline/runs/<실행명>/`의 상태·선택·승인 DB·attempt 결과와 `logs/agents/module_pipeline/workers/<작업ID>/transport.json`이다. 승인 DB는 `mode=ro`로 조회했다. 이 기록은 당시 로컬 증거 확인이며 DB 변조 방지 인증이나 통계 정답 인증이 아니다. 코드 변경 후 재사용 판단은 현재 코드 식별값으로 다시 확인해야 한다.

## 현재 실행 구조

```text
snapshot + 프로파일 + 모듈 설정 + 자원 한도
  → 계약 결합·계획 검증
  → preflight
  → 전체 목록에 대한 사용자 승인
  → SQLite 승인 단일 소비·작업 예약
  → 새 AppContainer + Job
      → CSV 복사본 검증·읽기
      → 선택 모듈 순차 실행·메모리 결과 검사
      → 품질 검토 필요/차단이면 나머지 모듈 중단
  → 프로세스 종료·Job·프로필 정리 증거 확인
  → 부모의 제한된 결과 파일 검증
  → 통과 시 완료 커밋 / 그 외 진단 보존·중단
```

기존 실행기의 고정 단계 정책을 바꾸지 않도록 전체 목록을 하나의 `run_eda` 작업으로 감싼다. 이 단계 이름은 기존 실행 관리자용 식별자이며 내부 모듈 목록에는 품질 검사도 들어갈 수 있다. 목록 안의 각 모듈에 별도 승인을 소비하거나 중간 상태를 개별 커밋하지 않는다. 시간·메모리·프로세스 한도는 **전체 worker 작업**에 적용한다.

현재 연결 범위는 내장 EDA 5개와 품질 검사, CSV snapshot, `source.data` 입력이다. 다른 모듈의 출력 데이터를 다음 모듈 입력으로 전달하는 실행은 아직 지원하지 않으며 계약 결합 단계에서 거부한다. 정적 계획의 연결 검증 기능과 실제 지원 범위를 구분한다. 변환·분할·사용자 코드·LLM은 실행하지 않는다.

## 소스 역할

| 파일 | 역할 |
| --- | --- |
| `src/agents/module_pipeline_contract.py` | 실행 지원 범위 확인·프로파일 품질 규칙 결합 |
| `src/agents/module_pipeline.py` | 기존 상태·승인·예약·취소 연결, 부모 검증 후 커밋 |
| `src/agents/module_pipeline_cli.py` | 생성·다음 단계·토큰 승인·상태·취소 |
| `src/agents/module_pipeline_transport.py` | 고정 코드 복사·해시 확인·AppContainer·Job·프로필 정리 |
| `src/agents/module_pipeline_worker.py` | 승인된 CSV 읽기·선택 모듈 실행·중단·파일 결과 생성 |
| `src/agents/module_pipeline_results.py` | 표준 파서로 결과 파일·헤더·역할·범위·품질 상태·순서 검사 |
| `tests/agents/test_module_pipeline_execution.py` | 합성 실행·승인·취소·자원·변조·독립 복사 패키지 검증 |

품질 규칙은 프로파일 값을 기본으로 하고 모듈 설정에서 명시한 항목만 덮어쓴다. 결합된 규칙 전체가 계획 식별값·승인 미리보기에 포함된다. JSON 배열/tuple 표현도 정규화해 저장·복원·IPC가 같은 계약으로 비교되게 한다.

## 결과와 검증의 의미

선택한 모듈만 실행하므로 결측 분석을 선택했다고 전체 상관분석·그림을 만들지 않는다. 이번 실행 경로는 CSV 통계와 품질 JSON/Markdown만 생성한다. 그림·종합 보고서는 후속 reporting 모듈로 연결할 예정이며 기존 고정 EDA CLI의 그림 생성은 그대로 유지한다.

결과는 `logs/agents/module_pipeline/runs/<name>/attempts/<task_id>/pipeline/`에 기록한다. `pipeline_result.json`에서 실행된 단계 목록·상태를 확인하고, 단계명 하위 폴더에서 표·품질 보고서를 확인한다. worker 작업 폴더는 `logs/agents/module_pipeline/workers/<task_id>/`다.

부모는 원본 CSV를 다시 파싱하지 않는다. 승인된 계획의 단계 순서와 정확한 파일 목록·크기·CSV 헤더·역할·일부 범위·품질 상태를 검사한다. 실제 통계 정답이나 악성 worker가 보고한 모든 의미적 사실까지 독립적으로 인증하는 것은 아니다. LLM 검증은 사용하지 않는다.

검토/차단 보고서는 검증한 뒤 로컬에 보존하지만 `AGENT_COMPLETE.json`을 만들지 않는다. 품질 검토 뒤 생성된 후속 결과나 일부 단계만 실행하고 전체 통과를 주장하는 결과는 거부한다. 자원 초과·취소·종료 증거 누락·요청 불일치도 완료 처리하지 않는다.

입력·프로파일·전체 모듈 순서·유효 설정·코드·자원 한도가 승인 범위에 결합된다. 변경 시 새 이름으로 새 실행·승인을 준비한다. 부분 실패를 자동 재시도하거나 기존 승인을 초기화하지 않는다. 네트워크·원본 폴더·승인 DB 권한을 넓히지 않는다.

## 합성 예제 실행

프로젝트 루트의 **관리자 PowerShell**, Python 3.10 환경에서 실행한다. 기존 실행 이름이 있으면 새 이름으로 바꾼다. 아래 생성 명령은 합성 CSV·snapshot을 만들 뿐 worker를 시작하지 않는다.

```powershell
python -m src.agents.module_pipeline_cli create-demo `
  --name module-demo01 `
  --pipeline configs/agents/pipelines/tabular_eda_example.json `
  --memory-mib 2048 --processes 2 --wall-seconds 60

python -m src.agents.module_pipeline_cli next --name module-demo01
python -m src.agents.module_pipeline_cli next --name module-demo01
python -m src.agents.module_pipeline_cli approve --name module-demo01
```

표시된 `PIPELINE-...`를 정확히 입력한다. 발급된 승인 ID를 아래 자리표시자에 넣는다. 승인 화면의 역할·규칙·경로는 로컬 확인용이며 그대로 LLM에 전송하지 않는다.

```powershell
python -m src.agents.module_pipeline_cli next --name module-demo01 --approval-id <발급된_승인_ID>
python -m src.agents.module_pipeline_cli status --name module-demo01
```

실제 snapshot 선택 명령에는 `create --name <새이름> --snapshot-dir <검증된복사본폴더> --profile <로컬프로파일> --pipeline <모듈설정> --confirm-snapshot-selection`을 지정한다. 먼저 합성 예제의 실제 OS 실행을 확인하며 이번 작업에서는 실데이터를 자동 실행하지 않는다.

취소는 `python -m src.agents.module_pipeline_cli cancel --name module-demo01`을 사용한다. 실패한 실행을 재사용하지 말고 진단 후 새 이름과 승인을 사용한다.

## 테스트

### 실제 OS 중단 경로 확인용 설정

`configs/agents/pipelines/tabular_review_example.json`은 합성 데모의 결측률 기준을 `0.0`, 통지 수준을 `review`로 지정한다. 합성 CSV에 결측값이 있으므로 품질 검사가 검토 필요를 반환해야 한다. 실데이터 권장 기준이 아니라 중단 동작을 확인하는 전용 설정이다. 기존 정상 예제·데이터·실행 코드는 변경하지 않는다.

새 실행명 `module-review01`과 위 설정으로 `create-demo → next → next → approve → next --approval-id ... → status`를 진행한다. 기대 결과는 실행 상태 `blocked_input`, 코드 `TOOL_REVIEW_REQUIRED`, `pipeline_status=review_required`다. 로컬 결과의 `completed_steps`는 `["quality"]`여야 하며 후속 `missing`·`correlations` 폴더와 `AGENT_COMPLETE.json`은 없어야 한다. 검토 필요를 성공으로 바꾸거나 동일 승인을 재사용하지 않는다. 실제 OS 실행 결과는 사용자 확인 후 별도로 판정한다.

### 합성 자동 테스트

```powershell
python -m unittest tests.agents.test_module_pipeline_execution -v
python -m unittest discover -s tests/agents -q
```

복사 허용 목록만으로 별도 Python 프로세스에서 worker를 호출하는 테스트도 포함한다. 다만 이는 AppContainer·ACL·WFP의 실제 환경 검증이 아니다. 실제 OS 실행 결과는 별도로 기록한다.
