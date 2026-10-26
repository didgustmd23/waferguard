# snapshot 품질 검사 — run_audit 격리 연결

기존 `preprocessing.quality.inspect_quality()`를 snapshot·사용자 승인·SQLite 예약·AppContainer+Job·부모 보고서 검사에 연결한 단독 CLI다. 데이터를 삭제·대치·스케일링·재라벨링하거나 모델 학습·분할·EDA를 실행하지 않는다. 이 CLI는 preflight → audit 계획만 수행한다. EDA·분할·전처리를 이어서 수행하는 통합 작업은 [GUIDE](../GUIDE.md#통합-작업)의 별도 서비스 연결을 따른다.

## 소스 경로와 역할

| 파일 | 역할 |
| --- | --- |
| src/agents/snapshot_audit_cli.py | 생성·명시 승인·상태·취소·로컬 보고서 위치 안내 |
| src/agents/snapshot_audit.py | 기존 snapshot 기반 재사용, audit 승인 범위·전달·부모 검증 |
| src/agents/snapshot_audit_worker.py | 입력/설정 검사와 비파괴 품질 코어 호출 |
| src/agents/snapshot_audit_transport.py | 고정 코드·런타임 복사와 AppContainer+Job 실행·정리 |
| src/agents/quality_result_validation.py | 제한된 JSON/Markdown의 구조·역할·정책·판정 일관성 검사 |
| tests/agents/test_snapshot_audit.py | 합성 통과·검토·차단·변조·승인·취소·독립 복사 코드 검증 |

기존 EDA 소스·코어·완료 로그와 모델 설정은 변경하지 않았다. 공통 snapshot 경로·계약·취소 신호·승인 저장소를 재사용하지만 audit의 정책/계획/코드 식별값은 별도로 결합한다. 일반 CSV에만 연결하며 pickle 로더·임의 import·비격리 fallback·외부 전송은 추가하지 않는다.

## 판정과 정지

| 품질 결과 | 실행기 출력 | 보고서·완료 처리 |
| --- | --- | --- |
| passed | completed / STEP_COMPLETED | 부모 검증 후 완료 표식·committed |
| review_required | blocked_input / TOOL_REVIEW_REQUIRED | 검증한 보고서는 attempt에 보존, 완료 표식·커밋 없음 |
| blocked_input | blocked_input / TOOL_INPUT_BLOCKED | 차단 보고서 보존, 완료 표식·커밋 없음 |
| 실행·자원 장애 | failed_runtime / 고정 실패 코드 | 정상 품질 결과로 취급하지 않음 |

worker 응답 `succeeded`와 진단 `COMPLETE / PASSED`는 **검사 프로세스가 정상 종료**했다는 뜻이다. 데이터 품질 통과는 `quality.json`의 상태와 부모 판정으로 구분한다. 검토·차단 때 작업 DB의 `failed`는 통과 완료 커밋을 하지 않았다는 뜻이며 반드시 프로세스 장애를 의미하지 않는다. 검토 필요를 자동 승인하거나 설정을 완화하지 않는다.

부모는 원본을 재파싱하지 않고 파일 목록·일반 파일/링크·JSON 중복 키·버전 없는 고정 필드·허용 코드·프로파일 역할·통지 수준·판정 일관성·고정 Markdown 표현을 검사한다. JSON/Markdown은 각각 1MiB 상한이며 findings 개수도 역할 수에 따라 제한한다. 이 검사는 원본 대비 모든 품질 건수를 독립 재계산한 인증이 아니다.

## 먼저 합성 AppContainer 실행 확인

코드 테스트의 OS transport는 모의 처리했다. 기존 EDA 성공이 새로운 audit worker의 OS 실행 성공을 대신하지 않으므로 아래 예제를 먼저 실행한다. 관리자 PowerShell, 프로젝트 루트, ml310 환경을 사용한다.

```powershell
python -m src.agents.snapshot_audit_cli create-demo --name audit-demo01 --memory-mib 2048 --processes 2 --wall-seconds 60
python -m src.agents.snapshot_audit_cli next --name audit-demo01
python -m src.agents.snapshot_audit_cli next --name audit-demo01
python -m src.agents.snapshot_audit_cli approve --name audit-demo01

# 표시된 AUDIT-문구를 직접 입력한 후 발급된 approval_id로 교체
python -m src.agents.snapshot_audit_cli next --name audit-demo01 --approval-id <approval_id>
python -m src.agents.snapshot_audit_cli status --name audit-demo01
```

상태 출력의 `quality_status`로 데이터 품질 판정을 구분하고 `local_quality_report`를 로컬에서 확인한다. 상세 findings·컬럼명·건수를 외부 LLM이나 추적 서비스로 보내지 않는다. 실패·검토·차단 작업은 자동 재실행하지 않는다.

실행 중 취소는 같은 OS 계정의 별도 터미널에서 요청한다.

```powershell
python -m src.agents.snapshot_audit_cli cancel --name audit-demo01
```

네트워크 capability·loopback 예외·시스템 방화벽·감사 정책·기존 자원 기본값은 변경하지 않았다. worker는 작업 input/tools/runtime만 읽고 output/scratch에만 쓰기 권한을 추가한다. 부모는 전체 경로를 검사하며 worker는 허용된 input 내부만 검사한다.

## 실제 Train 준비 — 아직 실행 전

기존 `secom-train-eda01` snapshot을 재검증해 새 품질 검사 실행 `secom-train-audit01`을 만들었다. 원본·Test·Validation은 다시 읽거나 변경하지 않았다. 프로파일은 `configs/agents/profiles/secom_train_audit.json`이며 기존 EDA 프로파일과 별도로 관리한다.

- 결측률 0.5 초과·상수·전체 결측 특징: review.
- 허용 라벨: -1, 1. 미지 라벨은 공통 코어의 block.
- 라벨 결측·ID 중복·ID 결측·시간 결측: block.
- 시간 형식: 기존 SECOM 데이터 계약의 `%d/%m/%Y %H:%M:%S`. 값을 추측하거나 자동 변환해 원본에 쓰지 않는다.
- 메모리 2048MiB·동시 프로세스 2개·실행 60초. 앞선 요청값을 재사용했으며 충분성을 보장하는 운영 상한은 아니다.

현재 실제 Train 실행은 `audit / APPROVAL_REQUIRED`다. 승인은 발급·소비하지 않았고 worker는 시작하지 않았다. 합성 경로를 실제 OS에서 확인한 뒤 다음 명령으로 별도 승인한다.

```powershell
python -m src.agents.snapshot_audit_cli approve --name secom-train-audit01
python -m src.agents.snapshot_audit_cli next --name secom-train-audit01 --approval-id <approval_id>
python -m src.agents.snapshot_audit_cli status --name secom-train-audit01
```

SECOM의 결측·상수 특징 때문에 review_required가 나올 수 있다. 이는 자동 수정하거나 기준을 무시할 이유가 아니다. 로컬 근거를 보고 다음 정책을 사용자가 결정한다. EDA 성공과 품질 통과도 별개다.

저장 경로는 `logs/agents/snapshot_audit/runs/<name>/`, `examples/<name>/`, `workers/<task_id>/`다. 진단은 worker scratch의 `diagnostic.json`, 품질 결과는 attempt의 `quality/quality.json`·`quality.md`에 보존한다. 부분 결과·기존 로그는 자동 삭제하지 않는다.

## 검증 기록

```powershell
python -m unittest tests.agents.test_snapshot_audit -v
python -m unittest discover -s tests/agents -q
```

신규 합성 검사 13개 통과. 전체 225개 중 224개 통과, 기존 Windows 링크 권한 관련 1개 건너뜀. worker 코어·보고서 쓰기·승인/예약·부모 검증을 실제 호출했으며 단위 테스트의 AppContainer/Job 운송 부분만 모의 처리했다. 별도 프로세스의 복사 목록·고정 진입점 실행도 통과했다. 이후 실제 합성 audit 실행 확인은 아래와 같으며 품질 검토의 자동 해제 기능은 구현하지 않았다.

## 실제 합성 실행 확인 — audit-demo01

2026-10-16 사용자 관리자 터미널 실행 후 로컬 상태·부모 검증·완료 표식·승인 DB·transport를 대조했다. worker 재실행이나 승인 초기화 없이 기존 산출물을 확인했다.

| 항목 | 결과 |
| --- | --- |
| 실행·품질 상태 | completed / passed |
| 승인·작업 DB | consumed, preflight·audit 모두 committed |
| worker 종료 | exited, 종료 코드 0 |
| Job 실행 시간 | 12.078초, 부모 준비 시간 제외 |
| 최고 Job 메모리 | 98660352 bytes ≈ 94.09MiB, 상한 2048MiB |
| 프로세스 | 동시 상한 2개, 누적 생성 3개, 정리 후 활성 0개 |
| 자원 초과 알림 | 없음 |
| Job·AppContainer 프로필 정리 | 모두 확인 |
| 부모 결과 검증 | 통과 |
| 완료 표식·계획 대조 | ARTIFACT_REUSABLE |

작업 ID는 `0f0fd941849041c9ba3dd7fa7e96006c`다. 실행 기록은 `logs/agents/snapshot_audit/workers/<task_id>/transport.json`, 완료 보고서는 `logs/agents/snapshot_audit/runs/audit-demo01/attempts/<task_id>/quality/quality.md`에서 로컬 확인한다.

이는 합성 품질 검사 한 건의 실제 정상 경로 확인이다. 실제 Train 품질 통과·실제 OS에서의 모든 차단/취소 경로·통계의 독립 정확성·전체 전처리 workflow 완료를 의미하지 않는다. 다음은 이미 승인 대기 상태로 준비한 secom-train-audit01의 별도 사용자 승인·실행이다.
