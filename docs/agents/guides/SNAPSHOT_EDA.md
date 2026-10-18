# snapshot CSV EDA — 승인 CLI와 격리 worker 연결

형식 독립 복사본과 EDA 입력 계약을 사용자 승인·SQLite 예약·AppContainer+Job·부모 결과 검증에 연결했다. 현재 분석 어댑터는 등록된 `tabular_csv_v1`만 허용한다. pickle·임의 모듈·셸 명령을 분석 옵션으로 받지 않는다.

기존 `isolated_eda_cli`와 완료 로그는 변경하지 않았다. 단위·연결 테스트의 **OS transport는 모의 처리**했으며, 별도로 사용자 관리자 실행 `snapshot-demo03`의 실제 AppContainer 정상 완료와 로컬 기록 재검증을 확인했다. 초기 두 번의 입력 차단과 보완·성공 기록은 아래에 보존한다. 실데이터 EDA 검증 완료나 모든 보안 위협에 대한 인증은 아니다.

## 파일별 역할

| 경로 | 역할 |
| --- | --- |
| src/agents/snapshot_eda_cli.py | 선택 저장·생성·승인·다음 단계·상태·취소 |
| src/agents/snapshot_eda.py | 승인 범위·코드 결합, 예약 후 staging·요청·결과 검증·등록 |
| src/agents/snapshot_eda_transport.py | 고정 소스·런타임 복사, AppContainer 권한·Job 실행·정리 |
| src/agents/snapshot_eda_worker.py | 요청/입력/설정 대조, CSV 헤더·역할 검사, 공통 EDA 호출 |
| src/agents/eda_result_validation.py | 원본 재파싱 없는 부모의 결과 형식·범위·내부 일관성 검사 |
| tests/agents/test_snapshot_eda.py | 합성 연결·변조·실패·취소·CLI 승인·복사 의존성 검증 |

## 합성 예제 실행

프로젝트 루트, 관리자 PowerShell의 Python 3.10 `ml310` 환경을 사용한다. 프로세스 2개는 이번 예제의 사용자 요청값이며 기존 기본값 1개를 변경하지 않았다.

```powershell
# 고정 합성 파일·프로파일·snapshot 생성. 아직 worker/승인 발급 없음
python -m src.agents.snapshot_eda_cli create-demo --name snapshot-demo01 --memory-mib 2048 --processes 2 --wall-seconds 60

# 환경 검사
python -m src.agents.snapshot_eda_cli next --name snapshot-demo01

# 승인 대기 (APPROVAL_REQUIRED)
python -m src.agents.snapshot_eda_cli next --name snapshot-demo01

# 로컬 컬럼 역할·파일 크기·설정·권한·자원 요청을 확인하고 표시된 EDA-문구 입력
python -m src.agents.snapshot_eda_cli approve --name snapshot-demo01

# 직전 출력의 approval_id로 교체
python -m src.agents.snapshot_eda_cli next --name snapshot-demo01 --approval-id <approval_id>

python -m src.agents.snapshot_eda_cli status --name snapshot-demo01
```

성공 상태는 `completed`다. 실패한 작업은 자동 재실행하지 않는다. 원인 확인 후 새 이름과 새 승인을 사용한다. 실행 중 취소는 같은 OS 계정의 별도 터미널에서 요청한다.

```powershell
python -m src.agents.snapshot_eda_cli cancel --name snapshot-demo01
```

런타임 복사 중 취소는 준비 단계의 반환 시점에 확인한다. worker 실행 중에는 Job 감시기로 전체 종료한다. 미사용 승인은 폐기하며 완료 커밋은 취소로 되돌리지 않는다.

## 사용자 준비 복사본 선택

합성 예제의 새 OS 경로 확인 후 진행한다. 원본 파일이 아니라 `input_snapshot`으로 준비한 폴더와 로컬 JSON 프로파일을 명시한다.

```powershell
python -m src.agents.snapshot_eda_cli create --name csv-eda01 --snapshot-dir logs/agents/input_snapshots/csv-input01 --profile configs/agents/profiles/example.json --confirm-snapshot-selection --memory-mib 2048 --processes 2 --wall-seconds 60
```

경로는 예시이며 기존 파일/폴더로 교체한다. `--confirm-snapshot-selection`은 사용자가 **현재 선택한 복사본·manifest를 새 기준으로 채택**한다는 뜻이다. manifest를 위변조 불가능한 과거 승인 증명으로 취급하지 않는다. 선택 시 실제 해시·크기를 대조한 뒤 계약을 저장하고, 이후 변경은 같은 계약/승인으로 실행하지 않는다. 실제 실행 승인은 별도의 `approve`에서 필요하다.

프로파일은 기존 DatasetProfile JSON 형식을 사용한다. worker는 역할에 맞는 필수 컬럼·수치형 입력·중복 헤더를 검사하며 미지원 형식을 자동 추정하지 않는다. 이 실행은 EDA만 수행하며 프로파일에 포함된 분할·학습 준비 workflow 전체를 실행하지 않는다.

## 접근·출력 검증

승인 소비·예약 이후에만 새로운 작업 staging을 만든다. 선택 snapshot을 다시 제한 복사하고 해시·크기·상한이 같은지 확인한다. worker에는 작업용 input/tools/runtime의 읽기, 작업 output/scratch의 쓰기 권한만 추가한다. 선택 snapshot 원래 폴더·원본 데이터 폴더·승인 DB에는 권한을 추가하지 않는다. 네트워크 capability·loopback 예외·방화벽·감사 정책은 변경하지 않는다. 실패 시 비격리 fallback은 없다.

부모는 원본을 다시 CSV로 파싱하거나 자식의 pickle을 역직렬화하지 않는다. 결과는 정해진 CSV 7개·PNG 3개·JSON 2개·Markdown 1개만 허용한다. 파일당 8MiB, 총 64MiB, CSV 표당 100,000행·1,025열의 검증 상한을 적용한다. 초과 시 완료 등록을 거부하며 결과를 자동 자르지 않는다. 이 상한은 **생성 중 OS 디스크 quota가 아니다.** 누적 저장 공간 정책은 별도 과제다.

부모 검사는 요청/응답 일치·Job 한도와 종료/정리·프로필 정리·결과 목록·링크 거부·JSON 필드와 타입·CSV 헤더/역할·통계 범위·분포 합계·결측률 내부 일관성·PNG 크기/청크/CRC·고정 Markdown 문구를 다룬다. `SUCCESS.json`만으로 완료하지 않는다. PNG는 부모에서 렌더링하지 않는다.

이 검사는 원본 대비 통계의 독립적 정확성이나 이미지 내용의 완전한 안전성을 인증하지 않는다. CSV의 데이터 유래 문자열은 로컬 전용이며 스프레드시트 수식으로 실행하거나 외부 LLM/추적 서비스에 전송하지 않는다. 입력 해시·코드 해시는 변경 탐지용이며 인증 서명이 아니다.

## 저장 위치와 검증

- `logs/agents/snapshot_eda/runs/<name>/`: 선택 계약·상태·승인 DB·검증 완료 attempts.
- `logs/agents/snapshot_eda/examples/<name>/`: create-demo의 고정 합성 입력·프로파일·snapshot.
- `logs/agents/snapshot_eda/workers/<task_id>/`: 작업 staging·런타임·출력·OS 실행 기록.

코드·계약 변경 시 기존 실행으로 재개하지 않는다. 부분 결과는 완료로 등록하거나 자동 삭제하지 않는다. 복사 런타임은 매번 새로 준비하므로 디스크 공간과 준비 시간이 필요하다.

```powershell
python -m unittest tests.agents.test_snapshot_eda -v
python -m unittest discover -s tests/agents -q
```

2026-10-16 첫 실제 실행 전: 신규 연결 검사 12개 포함 전체 208개 중 207개 통과, 기존 Windows 링크 권한 관련 1개 건너뜀. CLI 선택/대화형 승인 모의 입력, 실제 공통 EDA, 서로 다른 컬럼 구조·범주형 전용 입력, 변조·프로파일 변경·추가 pickle 결과·자원 장애·취소·정리 실패 차단과 복사 의존성 import를 확인했다. 이후 실제 실행 결과는 아래와 같다.

## 실제 실행 실패 기록 — snapshot-demo01

2026-10-16 사용자 관리자 실행에서 승인 소비·worker 시작까지 진행했으나 완료하지 못했다. 작업 `fc6f37088f28473da63135f26b89e4ed`의 transport 기록은 종료 코드 2, 실행 13.532초, 자원 초과 알림 없음, 활성 프로세스 0개, Job·프로필 정리 확인이었다. worker_result는 `blocked_input / WORKER_INPUT_BLOCKED`였으나 부모가 일반 `WORKER_FAILED`로 표시했다. EDA 완료 산출물은 등록되지 않았다.

당시 worker에 세부 단계 기록이 없으므로 어느 검사에서 차단됐는지는 아직 확정할 수 없다. 파일 경로/권한 확인과 내용 계약 검사를 구분하기 위해 scratch의 `diagnostic.json`에 고정 단계·분류만 기록하도록 보완했다. `ACCESS_DENIED`·`INPUT_INVALID`·`IO_FAILED`·`RUNTIME_FAILED`로 구분하며 원문 예외·입력 값·경로는 기록하지 않는다. 진단은 worker가 쓴 비신뢰 관측값이며 보안 증거나 승인 근거로 승격하지 않는다.

부모는 종료 코드 2에서도 요청 일치와 Job·프로필 정리가 확인된 입력 차단 응답을 `TOOL_INPUT_BLOCKED`로 구분한다. 성공으로 승격하거나 부분 결과를 커밋하지 않는다. 실패한 승인·예약·로그는 유지하고 새 이름 `snapshot-demo02`로 새 승인 후 확인한다. 메모리·프로세스·시간 상한이나 격리 권한은 이 보완으로 변경하지 않았다.

## snapshot-demo02 접근 거부와 worker 검사 범위 보완

두 번째 실제 실행의 진단은 `SNAPSHOT / ACCESS_DENIED`였다. 실행 12.859초, 자원 초과 알림 없음, 종료 코드 2, Job·프로필 정리 확인이었다. EDA 완료 결과는 등록되지 않았다.

소스 검토에서 worker가 부모용 `verify_snapshot()`을 호출하며 `_root()`·`_path()`로 입력 밖의 모든 조상까지 조회하는 구조를 확인했다. 이는 input에만 읽기 권한을 추가한 AppContainer 구성과 충돌할 수 있다. 기존 진단은 거부된 개별 경로를 기록하지 않으므로 정확한 OS 거부 지점 자체를 특정한 것은 아니며, 해당 구조를 우선 보완했다.

`snapshot_eda_worker.py`의 `_verify_staged_snapshot()`은 부모가 경로 검증·제한 복사·ACL 적용한 고정 input 경계 아래만 검사한다. 디렉터리/파일 유형·재분석 지점·하드 링크·파일 목록·manifest 기대값·크기·해시 검사는 유지한다. 부모의 전체 경로 검사와 기존 공통 `verify_snapshot()`은 변경하지 않았고 상위 폴더 읽기 권한도 추가하지 않았다.

상위 조상 조회가 거부되는 조건을 합성 테스트로 재현하고 worker 내부 검사는 성공하며 부모 검사는 계속 거부되는 것을 확인했다. 전체 212개 중 211개 통과, 기존 링크 권한 관련 1개 건너뜀. 실제 OS 해결 여부는 새 실행 `snapshot-demo03`으로 확인해야 한다. 기존 두 실행·승인·로그는 초기화하거나 자동 재실행하지 않는다.

## 실제 정상 실행과 기록 확인 — snapshot-demo03

2026-10-16 사용자 관리자 실행 후 상태·승인 DB·transport·요청/응답·완료 표식·선택/staging 복사본을 읽기 전용으로 대조했다. worker 재실행·승인 초기화·기존 로그 수정은 하지 않았다.

| 확인 항목 | 결과 |
| --- | --- |
| 실행 상태 | completed, 취소 요청 없음 |
| 승인 | consumed |
| 작업 DB | preflight·run_eda 모두 committed |
| worker 종료 | exited, 종료 코드 0 |
| worker 단계 진단 | COMPLETE / PASSED (비신뢰 보조 관측값) |
| Job 실행 시간 | 14.937초, 부모의 런타임 복사 등 준비 시간 제외 |
| 메모리 | 상한 2048MiB, 최고 Job 기록 181452800 bytes ≈ 173.05MiB |
| 프로세스 | 동시 상한 2개, 누적 생성 3개, 정리 후 활성 0개 |
| 자원 초과 알림 | 없음 |
| Job·AppContainer 프로필 정리 | 모두 확인 |
| 선택 snapshot·작업용 snapshot | 기대 크기·해시·파일 상한과 일치 |
| worker 응답·부모 receipt | 요청 ID 일치 |
| 선택 코드·계획 | 현재 코드 및 재구성 계획과 일치 |
| 완료 산출물 | 계획 기준 ARTIFACT_REUSABLE, 부모 결과 계약 재검증 통과 |

누적 프로세스 3개는 실행 중 만들어진 전체 개수이며 동시 상한 2개 초과를 뜻하지 않는다. 부모의 결과 계약 검사는 원본 대비 통계값을 재계산한 독립 정확성 검증이 아니다.

로컬 근거: `runs/snapshot-demo03/state.json`, `selection.json`, `approvals.sqlite3`, `attempts/3b4b656f69204d82b4d65a2cfad52c67/`, `workers/3b4b656f69204d82b4d65a2cfad52c67/transport.json` (모두 `logs/agents/snapshot_eda/` 아래).

상위 폴더 권한 확대 없이 worker 내부 검사 범위를 분리한 뒤 정상 경로가 통과했다. 이 실행은 합성 CSV 한 건의 정상 경로 검증이며, 이번 실행만으로 네트워크 차단·자원 초과·취소·다른 파일 형식·실데이터 전체를 새로 실측했다고 해석하지 않는다.

다음은 사용자가 선택한 실제 CSV의 컬럼 역할·프로파일과 입력 크기/자원 요청을 준비하고 별도로 승인해 EDA를 확인하는 단계다. 사용자 지정 EDA의 분석 ID·출력 계약 확장은 [범용 설계의 후속 검토 항목](../design/GENERIC_PREPROCESSING_DESIGN.md#41-사용자-지정-eda-추가--후속-검토-항목)으로 보존하며 이번에 구현하지 않았다.

첫 실제 CSV의 준비 및 승인 후 정상 실행·기록 재검증은 [SECOM Train EDA 기록](../history/SECOM_TRAIN_EDA.md)에 정리했다. 해당 CSV의 EDA 정상 경로 확인이며 전체 전처리·학습 workflow나 모든 보안 검증 완료를 주장하지 않는다.
