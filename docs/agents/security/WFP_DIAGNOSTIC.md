# WFP 로컬 차단 증거 수집 — 합성 검사 전용

갱신일: 2026-10-23. 실제 합성 수집 기록과 기존 ETL의 읽기 전용 재검토 결과를 포함한다. 수집 명령은 실데이터 worker와 LLM을 활성화하지 않는다.

## 목적과 범위

기존 검사에서 부모의 전후 로컬 연결은 성공했지만 AppContainer 연결은 timeout이었다. 이것만으로 Windows가 차단했다고 확정하지 않기 때문에 추가 OS 증거를 수집한다.

`src/agents/wfp_capture.py`는 고유 이름 `waferguard.wfp.<UUID>`의 ETW 세션에 `Microsoft-Windows-WFP` provider를 연결한다. 전역 `netsh wfp capture`나 `pktmon` 세션을 시작·종료하지 않는다. 다른 이름의 세션도 종료하지 않는다.

패킷 캡처 도구를 쓰지 않지만 **다른 프로세스의 네트워크 메타데이터가 ETL에 포함될 수 있다.** 수집 당시 다른 네트워크 작업을 줄이고 파일은 로컬에만 보관한다. LLM·외부 API·LangSmith·GitHub·채팅 첨부로 전송하지 않는다. 수집기의 출력도 제한된 상태·로컬 경로뿐이며 명령 출력 원문을 저장하지 않는다.

방화벽·감사 정책·통신 예외·기존 프로젝트 ACL은 변경하지 않는다. 합성 AppContainer 검사 자체의 새 프로필·검사 전용 폴더 ACL 작업은 기존 검사와 동일하다. 명시적인 시스템 메타데이터 수집 동의가 별도로 필요하다.

## 실행

사용자가 연 관리자 PowerShell에서 프로젝트 루트로 이동하고 Python 3.10 환경을 활성화한다.

```powershell
# 먼저 본인 PC의 프로젝트 루트에서 PowerShell을 연다.
conda activate ml310
python -m src.agents.wfp_capture `
  --confirm-isolation-probe `
  --confirm-system-metadata-capture `
  --seconds 20
```

기본 `appcontainer_probe` 명령은 WFP를 수집하지 않는다. 위 명령은 원본 데이터 없이 새 합성 검사 한 번을 실행한다. 런타임 복사가 끝난 뒤 프로세스를 시작하기 직전에 수집을 시작하므로 복사 시간은 수집 시간에 포함하지 않는다.

수집 시간은 10~60초다. 성공적인 시작 후 Python 타이머와 정상·예외 종료의 `finally`에서 고유 세션 종료를 시도한다. 직접 ETW 실행인 `-ets`에는 예약 수집 옵션 `-rf`를 섞지 않는다. 순환 ETL은 32MiB로 설정하고 로컬 provider 조회에서 확인한 `ut:Dropped` 키워드 `0x0000010000000000`을 명시한다. 이는 이벤트 범위를 줄이려는 설정이지 민감 정보 제거를 보장하는 필터가 아니다. 아래 실제 실행에서는 시작·중지·세션 부재를 확인했지만, 타이머 단독의 시간 제한 동작을 독립적으로 입증한 것은 아니다. 종료 명령과 종료 후 조회는 각각 최대 10초가 추가될 수 있다.

**강제 프로세스 종료·전원 종료까지 자동 정리를 보장하지 않는다.** Python 타이머는 프로세스가 살아 있어야 작동하며 독립적인 OS 종료 보장은 구현하지 않았다. 이 한계는 운영 worker의 OS Job Object·별도 감시 체계와 구분한다.

## 결과

새 검사 폴더에 다음 파일이 생성된다.

| 경로 | 역할 |
| --- | --- |
| `result.json` | 기존 격리 검사 결과와 수집 종료 확인 여부 |
| `wfp/capture.json` | 고유 세션명·SID·검사 PID·예상 실행 파일·로컬 포트·UTC 시각·시작/종료 코드 |
| `wfp/events.etl` 또는 OS가 부여한 변형 파일명 | 제한된 시간의 WFP 이벤트. 내용은 로컬에서만 확인 |

검사 PID는 정지 상태 프로세스를 재개하기 직전에 기록한다. 기록이 실패하거나 이미 수집이 끝났다면 프로세스를 재개하지 않고 종료한다. 같은 이름의 세션이 이미 조회되거나 존재 여부 조회가 실패하면 시작하지 않는다. 기존 세션을 종료하지 않는다.

`capture.json` v2는 시작 성공·종료 명령 성공·활성 세션 부재 확인을 분리한다. 조회의 `0x80300002`만 세션 없음으로 처리하고, 권한 오류·timeout 등 다른 실패는 unknown이다. 최초 시작 오류는 정리 결과로 덮어쓰지 않는다.

| 상태·필드 | 의미 |
| --- | --- |
| `start_failed_cleanup_complete` | 시작 실패 후 세션 없음 확인. 종료할 세션이 없어 stop은 생략할 수 있음 |
| `capture_complete_review_required` | 시작 성공 후 세션 없음 확인. 차단 이벤트 내용 검토는 별도 |
| `stop_unconfirmed` | 시작 시도 후 활성 세션 부재를 확인하지 못함. 고유 이름으로 수동 확인 필요 |
| `existing_session_not_touched` | 같은 이름의 세션 발견. 시작·종료하지 않음 |
| `preflight_unconfirmed` | 시작 전 세션 조회 실패. 시작하지 않음 |
| `stop_confirmed` | stop 명령의 반환 코드가 0인지. 명령을 생략하면 false일 수 있음 |
| `cleanup_confirmed` | 조회로 활성 세션 없음을 확인했는지. 최종 정리 판정은 이 필드를 사용 |

`result.json`의 최초 `failure_code`는 보존하고 정리 문제가 있으면 `cleanup_warning`을 추가한다. 시작 실패를 종료 실패로 덮어쓰지 않는다.

`capture_complete_review_required`는 수집 세션 정리 확인이지 차단 증명이나 전체 검사 통과가 아니다. `drop_evidence_verified`는 항상 false로 유지한다. 이 CLI의 종료 코드 2는 검토 필요를 뜻하며 자동 승인 경로가 없다.

## 로컬 검토 기준

Windows 이벤트 분석 도구로 ETL을 열고 다음을 함께 확인한다. 현재 단계에서는 자동 이벤트 파싱·차단 판정은 구현하지 않았다.

- 기록된 UTC 시각 범위와 일치하는 실제 차단 이벤트인지.
- 검사 실행 파일·AppContainer SID·PID 중 이벤트가 제공하는 식별 정보가 해당 검사와 일치하는지. PID만 같다는 이유로 확정하지 않는다.
- 목적지 `127.0.0.1`과 기록된 임시 포트, TCP 연결 방향이 일치하는지.
- 이벤트 종류가 접근 거부/패킷 드롭인지. 일반 WFP 이벤트나 외부 주소 접근 거부를 로컬 차단 증거로 대체하지 않는다.
- 이벤트 손실·파일 잘림·수집 시간 밖 연결 시도가 없었는지. 순환 로그는 이전 이벤트를 덮을 수 있다.

일치하는 이벤트가 없거나 필요한 필드가 없으면 계속 `review_required`다. 단순히 Drop이라는 문구가 있거나 ETL 파일이 생겼다는 이유로 통과시키지 않는다. 더 상세한 수집이나 정책 변경이 필요하면 범위를 다시 승인받는다.

## 종료 확인 실패 대처

화면 또는 `capture.json`에 기록된 **해당 고유 이름만** 사용한다.

```powershell
logman query waferguard.wfp.<해당UUID> -ets
logman stop waferguard.wfp.<해당UUID> -ets
```

이미 종료된 세션이라면 stop이 실패할 수 있다. 실패를 숨기거나 다른 세션을 일괄 종료하지 않는다. 파일은 자동 삭제하지 않으며, 로컬 검토 후 사용자가 보존·삭제를 결정한다.

## 구현 검증과 참조

동의·시간 범위·고유 세션·시간 초과 정리·기존 세션 보호·종료 실패·중복 종료를 모의 명령으로 검사한다. OS 이벤트 생성·자동 종료·차단 식별 성공을 대신하지 않는다.

2026-10-16 합성 검증: 신규 7개 통과. 전체 에이전트 테스트 152개 중 151개 통과, Windows 링크 생성 권한에 따른 기존 1개 건너뜀. 실제 WFP 세션은 생성하지 않았다.

후속 실제 실행 `1f7bdc1ef1194e4ea5c13377d01fa13a`는 시작 코드 `0x80070057`(잘못된 인자), 종료 코드 `0x80300002`(세션 없음)으로 중단했고 ETL이 생성되지 않았다. 이때 초기 수집기는 시작 오류를 종료 경고로 덮어썼다. 후속 읽기 전용 조회에서도 해당 세션 없음과 WFP provider 등록을 확인했다. 위 v2 명령·상태 보완은 이 이력을 바탕으로 작성했으며, `-rf`가 단독 원인인지는 확정하지 않았다. 당시에는 수정 후 실제 수집 전이었으며, 이후 실제 수집과 ETL 재검토 결과는 아래에 구분한다.

보완 후 합성 검증: WFP 테스트 11개, 전체 에이전트 테스트 156개 중 155개 통과·기존 Windows 링크 권한 테스트 1개 건너뜀. 시작 인자 오류·조회 실패·종료 후 부재 확인·타이머 콜백·HRESULT 부호 정규화를 검증했다. 이 검증에서 실제 수집 세션은 시작하지 않았다.

## 기존 ETL 재검토 — 2026-10-23

새 실행·접속·수집·정책 변경 없이 기존 `wfp/capture.json`·`input/probe.json`·`wfp/events.etl`을 대조했다. `Get-WinEvent -Path <events.etl> -Oldest`로 원본을 읽고, 로컬 `Microsoft-Windows-WFP` provider 정의에서 이벤트 ID `1001` 버전 2가 `WFP: Packet Dropped`임을 확인했다.

| 수집 실행 | 캡처 시각(UTC) | 검사 PID | loopback 서버 포트 | 확인한 이벤트 |
| --- | --- | --- | --- | --- |
| `3e4d66b4156a472fae626845257a1acd` | 2026-10-16 00:46:23.776~00:46:41.340 | 22208 | 64596 | loopback 폐기 2건·외부 시험 폐기 1건 |
| `ef25fb95cabf40e99bc2879fa923019f` | 2026-10-15 23:50:23.273~23:50:41.714 | 18980 | 62777 | loopback 폐기 2건·외부 시험 폐기 1건 |

- 두 수집 모두 시작·중지 반환 코드 0, 종료 후 세션 부재와 정리 완료가 기록돼 있다.
- loopback: `Loopback=true`, 로컬 주소 `127.0.0.1`과 서버 포트가 검사 입력과 일치한다. 첫 이벤트는 검사 PID·캡처 시각과 일치하고, 두 번째는 같은 주소·포트의 후속 폐기로 PID가 0이다. 필터 ID 70073·레이어 ID 44가 기록돼 있다.
- 외부 시험: `192.0.2.1:9`에 대한 폐기 이벤트의 PID와 AppId가 격리 검사 실행 파일에 대응한다. 필터 ID 71469·레이어 ID 48이 기록돼 있다. 문서용 예약 주소에 대한 해당 시험의 결과이지 모든 외부 목적지의 검증은 아니다.
- loopback AppId는 수신 측 부모 Python 실행 파일이다. 따라서 해당 시험 트래픽의 WFP 폐기는 확인하지만, 자식의 송신 측 AppId 일치나 AppContainer SID에 대응하는 특정 차단 정책의 원인 귀속까지 확인한 것은 아니다. 필터 ID만으로 정책의 의미를 추정하지 않는다.
- 이벤트 손실·파일 잘림이 전혀 없음을 별도 입증한 것은 아니다. 관찰된 폐기 이벤트를 근거로 해당 시험 범위만 판정한다.

**판정: 해당 합성 시험의 외부·loopback 트래픽에 대한 실제 WFP 폐기 증거 확인. SEC-12 전체 검증 완료는 아님.** 모든 목적지·worker·하위 프로세스·프로토콜로 일반화하지 않는다.

원본 JSON의 `drop_evidence_verified: false`와 `review_required`는 바꾸지 않았다. 수집기는 내용을 자동 판정하지 않고 항상 false를 기록하므로, 이 필드는 사후 ETL 검토 결과와 별개다. 별도 검사 실행 `7c0922a6e2e54482a887f12298d651de`의 timeout 판정도 이 ETL로 소급 변경하지 않는다.

```powershell
python -m unittest tests.agents.test_wfp_capture -v
```

명령 구성: Microsoft [logman create trace](https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/logman-create-trace), [logman start/stop](https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/logman-start-stop). 일반 WFP 진단 범위: [netsh wfp](https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/netsh-wfp).
