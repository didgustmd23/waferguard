# 합성 EDA의 승인 기반 격리 실행

> 초기 구현·실행 검증 이력입니다. 현재 상태와 권장 진입점은 [에이전트 문서 안내](../GUIDE.md)를 따릅니다. 당시 명령과 결과는 이력 보존용이며 현재 계약으로 자동 재개된다는 뜻이 아닙니다.

갱신일: 2026-10-16.

고정 합성 CSV 8행을 공통 EDA 코어로 분석하는 등록 도구를 기존 실행기에 연결했다. 실데이터·임의 입력 경로·명령·모듈 인자를 받지 않으며 LLM/API를 호출하지 않는다. 일반 CSV·WM 어댑터를 이 경로로 자동 전환하지 않는다.

## 실행 흐름

환경 검사 → 승인 대기 → 사람이 승인 → SQLite 승인 소비·작업 예약 → 새 AppContainer 및 Job 실행 → 부모의 결과 재검증 → 완료 표식·DB 커밋.

승인 범위에는 실행 주체·원본 식별값·계획·정책·관련 코드·자원 요청이 결합된다. WorkerRequest는 예약 후 생성된 작업 ID와 정확한 staging 경로까지 식별한다. 부모가 요청 ID·작업 ID를 대조하며 작업 JSON의 주체 문자열을 별도 인증으로 신뢰하지 않는다.

worker에는 고정 코드·런타임·합성 입력의 읽기 권한과 해당 작업의 출력·임시 폴더 쓰기 권한만 추가한다. 승인 DB와 부모 상태 폴더를 worker 허용 영역에 넣지 않는다. 네트워크 capability와 loopback 예외를 추가하지 않으며 시스템 방화벽·감사 정책도 변경하지 않는다.

## 실행 명령

프로젝트 루트의 **관리자 PowerShell**, `ml310` 환경에서 실행한다. 아래 프로세스 2개는 앞선 합성 호환성 검사 결과에 따른 이번 요청값이며 기본값 변경이 아니다. 기본 요청은 300초·2048MiB·1개다.

```powershell
python -m src.agents.isolated_eda_cli create --name eda-worker01 --memory-mib 2048 --processes 2 --wall-seconds 60

# 환경 검사: 프로필·worker를 만들지 않음
python -m src.agents.isolated_eda_cli next --name eda-worker01

# 승인 대기: 아직 worker를 실행하지 않음
python -m src.agents.isolated_eda_cli next --name eda-worker01

# 표시된 한도·효과를 확인하고 일회성 EDA-문구를 직접 입력
python -m src.agents.isolated_eda_cli approve --name eda-worker01

# 직전 명령이 출력한 approval_id로 교체
python -m src.agents.isolated_eda_cli next --name eda-worker01 --approval-id <approval_id>

python -m src.agents.isolated_eda_cli status --name eda-worker01
```

승인은 30분간 유효하며 실행 예약 때 한 번만 소비한다. 코드·한도·계획 변경 후에는 기존 승인을 재사용하지 않는다. 실패·시간 초과·자원 초과 작업은 자동 재실행하지 않는다. 원인을 검토한 뒤 새 이름으로 명시적으로 시작한다.

실행 중 취소는 같은 OS 계정의 별도 터미널에서 요청한다.

```powershell
python -m src.agents.isolated_eda_cli cancel --name eda-worker01
```

부모 전용 취소 신호를 Job 감시기가 확인하고 전체 작업을 종료한다. 런타임 복사 중에는 준비 단계가 반환된 뒤 확인할 수 있다. 이미 완료 커밋된 작업은 취소로 되돌리지 않는다.

## 결과 확인

- 실행 상태·승인 DB·검증 완료 결과: `logs/agents/isolated_eda/demos/<name>/`.
- 작업별 runtime/input/tools/output/scratch 및 OS 실행 기록: `logs/agents/isolated_eda/workers/<task_id>/`.
- 정상 완료는 상태 `completed`, EDA attempt의 `AGENT_COMPLETE.json`, 검증된 `eda/` 산출물로 확인한다.
- 실패 작업의 부분 결과는 완료 결과가 아니며 자동 삭제하지 않는다. 새 런타임을 매번 복사하므로 준비 시간과 디스크 공간이 필요하다.

부모는 Job 한도·전체 프로세스 정리·프로필 정리, 요청과 응답 결합을 확인한다. CSV 통계는 합성 입력으로 다시 계산하고 JSON·보고서 문구·파일 목록도 검사한다. PNG는 크기·청크·CRC만 검사하며 이미지의 완전한 안전성을 인증하지 않는다. 소스 해시는 변경 탐지용이며 인증 서명이 아니다.

## 검증 범위와 남은 일

```powershell
python -m unittest tests.agents.test_isolated_eda -v
python -m unittest discover -s tests/agents -q
```

연결 단위 테스트는 실제 공통 EDA와 승인·예약·부모 검증을 실행하지만 **OS transport는 모의 처리**한다. 별도로 아래 사용자 실행의 로컬 기록에서 새 등록 도구의 실제 AppContainer+Job 경로 완료를 확인했다.

2026-10-16 검증: 신규 10개 포함 에이전트 테스트 184개 중 183개 통과, 기존 Windows 링크 권한 관련 1개 건너뜀. 복사 허용 목록만 사용하는 별도 Python 프로세스의 패키지 import도 통과했다. Matplotlib 의존성의 기존 deprecated 경고는 실행 실패가 아니다.

이번 구현은 실데이터 worker 활성화 승인이 아니다. 임의 입력의 안전한 staging·바꿔치기 방어, 런타임 신뢰 관리, 추가 프로파일 연결과 각 도구의 격리 검증은 별도 단계다. OS 격리 실패 시 비격리 실행으로 우회하지 않는다.

## 실제 실행 확인 — eda-worker01

2026-10-16 사용자 관리자 터미널 실행 후 로컬 상태·transport·승인 DB·완료 산출물을 읽기 전용으로 확인했다. 다시 실행하거나 기존 승인·결과를 변경하지 않았다.

| 확인 항목 | 결과 |
| --- | --- |
| 실행 상태 | completed, 취소 요청 없음 |
| 승인 상태 | consumed |
| 작업 DB 상태 | preflight·run_eda 모두 committed |
| worker 종료 | exited, 종료 코드 0 |
| Job 실행 시간 | 7.39초 (런타임 복사 등 부모 준비 시간 제외) |
| 메모리 | 상한 2048MiB, Job 최고 기록 약 132.51MiB |
| 프로세스 | 상한 2개, 전체 생성 2개, 정리 후 활성 0개 |
| 자원 초과 알림 | 없음 |
| Job·AppContainer 프로필 정리 | 모두 확인 |
| 요청·응답 식별값 | transport와 부모 receipt 일치 |
| 완료 산출물 | ARTIFACT_REUSABLE, EDA 내용 재검증 통과 |

로컬 근거: `demos/eda-worker01/state.json`, `approvals.sqlite3`, `attempts/c5da4774762f45dfb683a140efbb6cf4/`, `workers/c5da4774762f45dfb683a140efbb6cf4/transport.json` (모두 `logs/agents/isolated_eda/` 아래).

이 실행은 고정 합성 EDA의 정상 경로 검증이다. 이번 실행만으로 네트워크 차단·자원 초과·취소·침해 방어 전체를 새로 실측했다고 해석하지 않는다.

## 다음 구현 순서

1. [형식 독립 snapshot 준비](../design/INPUT_SNAPSHOT.md): 단일 파일의 복사·내용 식별·크기 제한·링크 거부·변경 탐지를 구현했다. 형식 해석은 별도 어댑터가 맡으며 프로파일·승인·worker 전달 결합은 아직 남아 있다. 원본 폴더 전체 권한은 주지 않는다.
2. 동일 EDA worker에 등록 프로파일의 컬럼 역할·설정을 전달하고 합성 CSV로 검증한다. 현재 고정 fixture 도구는 보안 회귀용으로 유지한다.
3. EDA 결과를 원본 없이 부모에서 검증하는 규칙과 로컬 보고서 경로를 정비한다. LLM에 상세 통계·경로·그래프를 전달하지 않는다.
4. 승인 후 실데이터 EDA를 별도로 검증한다. 품질 검사·분할·Train-only 준비 도구는 각각 이후에 연결한다.
