# 확정 분할의 Train·Validation 내보내기

구현일: 2026-10-16. 요청 결합·worker·부모 파일 검증에 이어 **별도 사용자 승인 CLI → AppContainer+Job 운송 → 부모 검증 → 기존 실행기 커밋** 연결을 구현했다. 합성 테스트에서는 실제 worker 추출과 승인·커밋을 검사하고 OS 운송만 모의했다. 실제 Windows 격리 실행은 아직 확인하지 않았다. 함수 직접 호출은 승인된 격리 실행이 아니다.

## 구현 파일

| 파일 | 역할 |
| --- | --- |
| `src/agents/partition_export_contract.py` | 확정 분할·동일 원본 snapshot·동일 프로파일·출력 크기 상한 결합 |
| `src/agents/partition_export_worker.py` | 고정 요청 검증·CSV 스트리밍 추출·바이트 확인·내보내기 기록 생성 |
| `src/agents/partition_export_results.py` | 부모의 출처·배정·snapshot·CSV 구조·행 수 검증 |
| `src/agents/partition_csv.py` | 레코드 1 MiB·컬럼 1024개 상한의 공통 스트리밍 파서 |
| `src/agents/worker_contract.py` | ingest/run_ingest 요청 형식 추가. 도구 실행을 자동 허용하는 기능은 아님 |
| `tests/agents/test_partition_export.py` | 문자열·행 순서·출력 상한·변조·기존 구간 계약 연결 합성 검증 |
| `src/agents/partition_export.py` | 별도 ingest 계획·승인 후 복사·Job/응답 확인·부모 검증·커밋 연결 |
| `src/agents/partition_export_transport.py` | 고정 소스 복사·작업 전용 ACL·새 AppContainer·Job 한도·종료 정리 |
| `src/agents/partition_export_cli.py` | create/next/approve/status/cancel, 정확한 분할 승인 범위와 현재 OS 계정 확인 |
| `tests/agents/test_partition_export_execution.py` | 승인 전 차단·1회 실행·취소·한도 초과·변조·구간 계약 연결 검증 |

기존 EDA·audit worker는 자신의 고정 action/stage를 별도로 확인하므로 새 ingest 요청을 대신 실행하지 않는다. 공통 요청 코드 변경은 코드 식별값에 반영된다. 과거 승인·선택 기록을 수정하거나 재사용하지 말고 후속 실제 실행은 새 이름·계획·승인을 사용한다.

## 요청 결합

기존 CSV 분할의 출처 ID는 `csv:<원본 SHA-256>`, 형식 독립 snapshot은 `file_snapshot:<SHA-256>`이다. 접두사를 임의로 없애지 않고 검토된 CSV 어댑터에 한해 실제 바이트 해시가 같은지 명시적으로 대조한다. 프로파일 서명도 기존 확정 분할의 config_id와 같아야 한다.

요청에는 snapshot 계약·프로파일·분할 증거 ID·확정 split.json의 파일 해시·구간별 출력 바이트 상한·자원 한도·검토된 코드 식별값을 결합한다. 승인 DB·원본 디렉터리 경로는 worker 설정에 넣지 않는다. 실행 관리자가 승인 소비·예약 후 복사하며 운송 계층이 고정 코드 복사와 식별값 확인을 수행한다. 변경된 분할·설정·코드는 기존 승인으로 재개하지 않는다.

worker는 ingest/run_ingest·고정 입력/출력 경로·요청 설정 해시를 확인한 뒤에만 처리한다. 분할 파일은 별도 크기 제한으로 읽고 원본·프로파일·proposal_id·행 위치 완전성을 다시 대조한다. 새로운 분할·다른 시드 탐색은 하지 않는다.

## 추출 방식과 Test 접근 범위

CSV를 값 변환 없이 문자열로 읽어 확정된 원본 행 위치의 Train·Validation만 출력한다. ID의 앞자리 0·소수 문자열·한글·인용·인용된 줄바꿈은 보존한다. CSV 인용과 줄 끝 표현은 표준 writer로 정규화되므로 원본의 연속 바이트 복사와는 다르다. 각 파생 파일에는 별도 SHA-256을 기록한다.

**내보내기는 원본 CSV 전체를 순회하므로 Test로 배정된 행의 문자열도 읽는다.** 다만 Test 파일을 생성하거나 그 값으로 제거 목록·중앙값·스케일링 기준·모델을 계산하지 않는다. 이를 Test 미접근 실행이라고 표현하지 않는다. 이후 Train 기준 산출 worker에는 Train 구간 복사본만 전달해야 한다.

빈 행·중복/빈 헤더·컬럼 수 불일치·행 수 불일치는 자동 보정하거나 건너뛰지 않고 중단한다. 행 위치는 물리 텍스트 줄이 아니라 헤더 다음의 CSV 레코드 순서다. 따라서 인용된 줄바꿈 때문에 위치가 바뀌지 않는다.

## 결과와 제한

worker 결과 루트에는 `exports/`와 종료 응답을 두며, `exports/`에는 다음 계약만 생성한다.

```text
exports/
├─ partition_exports.json
├─ train/input.bin, snapshot.json
└─ validation/input.bin, snapshot.json
```

구간별 출력 상한은 `max_partition_bytes`로 명시하며 최대 두 구간을 생성한다. 상한은 디스크 quota나 worker 메모리 상한과 다르다. 각 기록 전에 바이트 크기를 확인하고 초과하면 잘라서 성공 처리하지 않는다. 실패 시 부분 파일이 남을 수 있지만 완성된 내보내기 기록·완료 참조는 제공하지 않는다.

원본 해시는 처리 전후에 확인한다. 구간 파일도 실제 저장 바이트를 다시 읽어 검증한 다음 snapshot manifest와 내보내기 기록을 만든다. worker는 AGENT_COMPLETE 표식을 생성하지 않는다. 부모가 Job 종료·활성 프로세스 0·한도 일치·프로필 정리·요청 응답·파일 검증을 통과한 뒤 기존 실행 관리자가 커밋한다. 취소·실패·부분 출력은 커밋하지 않는다.

## 부모 검증의 의미

부모는 원본 CSV를 파싱하지 않고 내보낸 두 CSV만 제한된 스트리밍 파서로 확인한다. 기대 분할 증거·원본 ID·proposal_id·역할·정확한 행 위치 목록의 해시·구간 행 수·파일 해시·상한·공통 헤더·컬럼 수를 대조한다. Test 또는 미선언 파일이 추가되면 거부한다.

이는 파일·계약의 정합성 확인이다. 악성 worker가 모든 기록을 일관되게 위조했을 때 원본 대비 모든 필드의 정답까지 독립적으로 인증하지 않는다. 추출의 실제 정답은 합성 입력과 기대 위치의 필드를 비교하는 테스트로 확인한다. 일반 데이터의 의미적 파생 관계는 검토된 내보내기 구현과 승인된 실행 경계를 전제로 한다.

기존 [구간 입력 계약](../DESIGN.md#분할내보내기train-기준)에 필요한 출력 형식을 그대로 유지한다. 완료 attempt에는 partition_exports.json·train·validation·AGENT_COMPLETE.json만 둔다. 요청·응답·운송 증거는 별도 workers 폴더에 보존하고 커밋 직전 및 완료 상태 조회 시 재확인한다. 로컬 관리자에 의한 DB·증거 동시 변조를 인증하는 기능은 아니다.

계산 테스트의 커밋 이력은 fixture이며, 연결 테스트에서는 선행 분할 이력만 fixture로 준비하고 내보내기 승인의 소비·예약·검증·커밋은 실제 RulesExecutor로 수행한다. 실제 AppContainer 실행과 OS 제한 강제는 별도 확인이 필요하다.

## CLI 사용 흐름

선행 분할 기록이 없으면 [격리 분할 CLI](SNAPSHOT_SPLIT.md)의 생성·단계별 승인·확정·export-scope 절차로 준비한다. SECOM Train 기능 검증용 명령과 동일 프로파일 연결 방법도 해당 문서에 있다.

먼저 승인·커밋된 기존 범용 CSV 분할, 그 실행 관리자가 보존한 정확한 commit_split ApprovalScope JSON, 같은 원본의 snapshot, 같은 프로파일이 필요하다. scope 파일은 승인 증거 자체가 아니다. 기존 DB의 소비·예약·커밋 기록과 완료 해시를 다시 확인한다. 예전 SECOM 분할 CSV나 수동으로 작성한 배정표만으로는 실행할 수 없다. 가짜 분할 승인·커밋을 만들어 주는 create-demo 명령도 제공하지 않는다.

다음은 경로를 실제 승인 산출물로 교체해야 하는 템플릿이다. 실데이터 실행 전에 합성 입력으로 네이티브 운송을 먼저 확인한다.

```powershell
python -m src.agents.partition_export_cli create `
  --name partition-export01 `
  --split-run-dir <확정_분할_실행폴더> `
  --split-scope <실행관리자가_보존한_commit_split_scope.json> `
  --snapshot-dir <동일_원본_snapshot_폴더> `
  --profile <동일_프로파일.json> `
  --max-partition-bytes 8388608 `
  --memory-mib 2048 --processes 2 --wall-seconds 60 `
  --confirm-selection

python -m src.agents.partition_export_cli next --name partition-export01
python -m src.agents.partition_export_cli next --name partition-export01
python -m src.agents.partition_export_cli approve --name partition-export01
# 화면의 EXPORT-... 문자열을 정확히 입력한 후 반환된 승인 ID를 사용한다.
python -m src.agents.partition_export_cli next --name partition-export01 --approval-id <승인_ID>
python -m src.agents.partition_export_cli status --name partition-export01
```

순서는 created → preflight → awaiting_approval → ingest → completed이다. create는 worker를 시작하지 않는다. 승인 화면은 원본 전체 순회·Test 출력 없음·출력 파일별 상한·자원 한도·유효기간 30분을 알린다. 한 승인은 한 작업 예약에만 소비된다. 실패 시 같은 승인을 재사용하거나 자동 재실행하지 않는다. cancel 명령은 미사용 승인을 폐기하고 실행 중 worker에 취소 신호를 전달한다.

성공 상태 조회는 완료 증거와 두 구간 계약을 재검사하여 partition_contracts: passed 및 로컬 결과 폴더만 출력한다. 아직 대치·스케일링·모델 학습은 수행하지 않는다. workers 폴더는 운송 증거이므로 완료 확인 전에 삭제하지 않는다.

## 테스트와 다음 단계

```powershell
python -m unittest tests.agents.test_partition_export -v
python -m unittest tests.agents.test_partition_export_execution -v
python -m unittest discover -s tests/agents -q
```

다음은 승인·커밋된 합성 분할을 준비하여 실제 Windows AppContainer 실행을 확인하는 것이다. 그 이후 Train 전용 전처리 기준 후보 산출과 별도 적용 승인을 연결한다. 이번 구현에서는 원본·실데이터·모델·과거 보고서에 변경을 적용하지 않았다.
