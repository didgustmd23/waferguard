# worker 자원 제한 구현·검증

갱신일: 2026-10-16. 사용자 지정 한도를 Windows Job Object와 부모 감시기로 강제한다. 실제 데이터 전처리 실행기로 자동 전환하거나 LLM을 활성화하지 않는다.

## 구현 범위

| 파일 | 역할 |
| --- | --- |
| `src/agents/job_object.py` | Job 설정·연결·통지·통계·경과 시간/취소 감시·전체 종료 |
| `src/agents/appcontainer_native.py` | 선택적으로 Job 연결 후 AppContainer 프로세스 재개 |
| `src/agents/resource_probe.py` | 사용자 옵션을 받는 고정 합성 자원 검사 CLI |
| `src/agents/resource_child.py` | 정상·메모리·자식 생성·대기·잔존 자식 검사 payload |
| `tests/agents/test_job_object.py` | 구조·실행 순서 및 실제 Windows 합성 프로세스 검증 |

기존 WorkerLimits의 기본값과 허용 범위를 유지했다. 자동 용량 측정·프로파일 추천·상한 자동 확대는 구현하지 않는다. 사용자 설정은 기존 계약의 양의 정수 범위 검사를 통과해야 한다. 관리자가 정한 별도 maximum은 WorkerRequest.check_policy에서 비교하는 구조를 유지한다. 합성 CLI는 WorkerRequest 실행기가 아니므로 승인·원본·경로 정책 검사를 대신하지 않는다.

## 제한 방식

1. 지정한 값으로 이름 없는 비상속 Job을 만들고 제한 설정을 완료한다.
2. worker를 정지 상태로 생성한다. 프로세스 핸들은 자식에게 상속하지 않는다.
3. Job 연결에 성공해야 콜백과 프로세스 재개를 수행한다. 연결 실패 시 정지된 프로세스를 종료하고 제한 없는 실행으로 우회하지 않는다.
4. 실제 실행 직전부터 monotonic 시계로 경과 시간을 감시한다. 취소 Event와 OS 자원 초과 통지도 함께 확인한다.
5. 시간 초과·취소·자원 초과·예외에서 Job 전체를 종료한다. 루트가 정상 종료해도 남은 Job 자식을 정리한다.
6. 최대 5초 동안 활성 프로세스 수 0을 확인한다. 확인 실패는 성공으로 등록하지 않는다. 마지막 Job 핸들 닫힘에는 KILL_ON_JOB_CLOSE가 적용된다.

| 옵션 | 의미 |
| --- | --- |
| `--memory-mib` | Job 전체의 커밋 메모리 한도. GPU VRAM·작업 관리자 RAM 표시와 다름 |
| `--processes` | 루트 포함 활성 프로세스 수. 스레드 수 제한이 아님 |
| `--wall-seconds` | 실제 실행 경과 시간. CPU 사용 시간과 다름 |

시간 초과는 약 50ms 단위로 감시하며 종료 확인 시간이 추가될 수 있다. 런타임 복사·프로필 준비 시간은 실행 한도에 포함하지 않는다. Job에는 BREAKAWAY_OK·SILENT_BREAKAWAY_OK를 부여하지 않는다.

## 사용자 실행 방법

프로젝트 루트의 Python 환경에서 실행한다. 다음은 합성 검사 예시이며 운영 권장 한도가 아니다. 관리자 권한·감사 활성화·방화벽 변경은 필요하지 않다. 다만 현재 환경의 부모 Job 정책 등에 의해 연결이 거부되면 실행을 중단한다.

```powershell
# 정상 실행
python -m src.agents.resource_probe --confirm-resource-probe --case normal --memory-mib 128 --processes 1 --wall-seconds 5

# 메모리 상한: 최대 256MiB 요청을 64MiB Job에서 시도
python -m src.agents.resource_probe --confirm-resource-probe --case memory --memory-mib 64 --processes 1 --wall-seconds 5

# 루트 외 자식 생성 제한
python -m src.agents.resource_probe --confirm-resource-probe --case spawn --memory-mib 128 --processes 1 --wall-seconds 5

# 실행 시간 제한
python -m src.agents.resource_probe --confirm-resource-probe --case sleep --memory-mib 128 --processes 1 --wall-seconds 1

# 0.5초 뒤 합성 취소 신호
python -m src.agents.resource_probe --confirm-resource-probe --case cancel --memory-mib 128 --processes 1 --wall-seconds 5

# 루트 종료 뒤 잔존 자식 정리
python -m src.agents.resource_probe --confirm-resource-probe --case descendant --memory-mib 128 --processes 2 --wall-seconds 5
```

매 실행은 `logs/agents/resource_probes/<새 ID>/result.json`에 기록한다. 원본 데이터·API 키·네트워크 연결을 사용하지 않으며 임의 명령·모듈 입력을 받지 않는다. 이 합성 CLI는 AppContainer가 아니므로 실행 파일·네트워크 격리 검증으로 해석하지 않는다.

AppContainer 합성 검사와 결합할 때는 사용자가 연 관리자 PowerShell에서 다음처럼 실행한다. 이 명령은 기존 검사와 동일하게 새 프로필·새 검사 폴더 ACL을 사용한다. 결합 코드의 순서는 모의 검증했지만 실제 AppContainer+Job 조합은 별도 사용자 OS 실행이 남아 있다.

```powershell
python -m src.agents.appcontainer_probe --confirm-isolation-probe --memory-mib 2048 --processes 1 --wall-seconds 60
```

AppContainer CLI의 기존 60초 기본 경과 시간과 2048MiB·프로세스 1 기본 요청을 사용한다. 저수준 AppContainer.launch의 limits 인자가 None이면 기존 검사 경로를 유지한다. 운영용 worker에서는 이 선택 경로를 그대로 허용하지 않고 WorkerRequest와 신뢰 정책으로 제한을 필수 결합해야 한다. 현재 실데이터 실행은 비활성이다.

## 결과 해석과 한계

### 자식 프로세스 생성 시도 진단

AppContainer 합성 payload에는 단계별 START/DONE/FAILED와 Python 감사 이벤트 기록을 추가했다. `output/progress.jsonl`을 최대 64개·64KiB 범위에서 검증하고 `result.json`의 `progress`에 요약한다. 매 기록을 즉시 쓰고 fsync하므로 최종 결과의 finally에만 의존하지 않는다. 마지막 줄이 끊어졌다면 완성된 앞부분만 읽고 partial로 표시한다.

기록은 고정 단계 코드, 감사 이벤트 종류, 실행 파일 분류 코드, 승인 패키지 분류·고정 호출 위치 코드·줄 번호뿐이다. 명령 인자·환경 변수·원본 내용·절대 경로·지역 변수·소스 원문은 저장하지 않는다. Python hook은 생성 시도를 관찰할 뿐 허용하거나 거부하지 않는다. 보안 제한은 계속 Job Object가 담당한다. 네이티브 DLL의 직접 생성까지 모든 경로를 포착한다고 보장하지 않는다.

진단 파일은 worker 보고이므로 보안 인증 증거가 아니다. 마지막 단계는 가장 최근 체크포인트이며 백그라운드 스레드의 정확한 원인 단계와 다를 수 있다. 프로세스 시도 하나에 Popen·CreateProcess 이벤트가 둘 다 남을 수 있으므로 기록 건수를 실제 자식 개수로 해석하지 않는다. 기록 부재도 생성 시도 없음을 증명하지 않는다. 기록 상한 도달·디스크 오류·강제 종료 시 일부 진단이 누락될 수 있다.

`PROBE_RESOURCE_EXCEEDED`이면 자원 제한에 따른 종료이고, 최종 child_result.json 미생성은 후속 현상이다. 프로세스 한도를 자동으로 높이지 않는다. 다음 명령으로 같은 1개 한도에서 원인을 확인한다. 감사 활성화·WFP 수집은 필요하지 않다.

```powershell
python -m src.agents.appcontainer_probe --confirm-isolation-probe --memory-mib 2048 --processes 1 --wall-seconds 60
```

새 실제 AppContainer 진단 실행은 사용자가 연 관리자 터미널에서 수행해야 한다. 진단 추가 후 자동으로 재실행하거나 새 프로필을 생성하지 않았다.

- `succeeded`: 루트 정상 종료와 Job 잔존 프로세스 정리 확인. 실제 전처리 산출물 검증 성공을 뜻하지 않는다.
- `resource_exceeded`: OS의 메모리 또는 프로세스 수 초과 통지를 관찰했다.
- `timed_out` / `cancelled`: 부모의 경과 시간/취소 정책으로 Job 종료.
- `failed_runtime`: 비정상 종료 또는 정리 확인 실패. 모든 비정상 종료를 자원 초과라고 추측하지 않는다.
- `cleanup_confirmed`: Job 활성 프로세스 0 확인 여부. 실패하면 성공 결과로 인정하지 않는다.

Job 통지 전달은 항상 보장되는 것이 아니다. 통지가 없어도 OS 한도는 설정되어 있지만, 원인 분류는 failed_runtime으로 남을 수 있다. `configured_job_memory_bytes`·`configured_process_limit`은 OS에서 다시 조회한 설정이다.

실제 검사에서 누적 프로세스 수가 시작한 payload 개수와 다르게 보고되고 peak 메모리 통계가 설정 한도보다 크게 보고된 실행도 있었다. 따라서 누적 total_processes를 활성 수로 해석하거나 peak를 상한 준수의 단독 증명으로 쓰지 않는다. 메모리 설정 재조회·OS 초과 통지·합성 할당 실패를 함께 확인했다. 반대로 이 통계를 바탕으로 RAM·VRAM까지 동일하게 제한됐다고 주장하지 않는다.

Job Object는 보안 sandbox가 아니다. 일반 CreateProcess 자식이 Job에 묶이는 범위와 외부 서비스·WMI 등의 broker를 통한 프로세스 생성은 다르다. AppContainer·고정 도구 등록·네트워크/파일 통제와 함께 사용해야 한다. 부모 강제 종료의 전체 에이전트 복구·승인 DB·산출물 정합성까지 검증한 것은 아니다.

## 검증 기록

2026-10-16 신규 10개 테스트 통과. 3개는 계약·실행 순서 모의 검증이며 7개는 실제 Windows 합성 프로세스로 정상 종료·메모리 제한·프로세스 제한·시간 초과·취소·잔존 자식 정리·마지막 Job 핸들 닫힘 시 정지된 worker 종료를 확인했다. 실제 AppContainer 프로필이나 감사 설정은 변경하지 않았다.

전체 에이전트 회귀 검증은 166개 중 165개 통과·기존 Windows 링크 생성 권한 테스트 1개 건너뜀이다.

진행·생성 시도 진단 추가 후 신규 테스트 8개 통과. 정보 최소화·고정 코드 분류·부분 줄 복구·기록 상한·쓰기 실패·잘못된 입력·별도 Python의 실제 감사 hook을 검증했다. 전체는 174개 중 173개 통과·기존 1개 건너뜀이다. 전역 테스트 실행기에는 hook을 설치하지 않았다.

```powershell
python -m unittest tests.agents.test_job_object -v
```

Microsoft 참조: [Job Objects](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects), [Extended Limit Information](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_extended_limit_information).
