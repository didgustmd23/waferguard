# worker 자원 제한 구현·검증

갱신일: 2026-10-23. 사용자 지정 한도를 Windows Job Object와 부모 감시기로 강제한다. 합성 자원 검사와 배열 통합 작업의 실제 AppContainer+Job 메모리 실측을 구분해 기록한다. 검사 도구가 실데이터 처리나 LLM 호출을 자동 활성화하지 않는다.

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

AppContainer 합성 검사와 결합할 때는 사용자가 연 관리자 PowerShell에서 다음처럼 실행한다. 이 명령은 기존 검사와 동일하게 새 프로필·새 검사 폴더 ACL을 사용한다. 아래 명령의 작성 당시에는 결합 순서의 모의 검증까지였으며, 이후 배열 통합 작업의 실제 AppContainer+Job 확인은 아래 실측 절에 구분한다.

```powershell
python -m src.agents.appcontainer_probe --confirm-isolation-probe --memory-mib 2048 --processes 1 --wall-seconds 60
```

AppContainer CLI의 기존 60초 기본 경과 시간과 2048MiB·프로세스 1 기본 요청을 사용한다. 저수준 AppContainer.launch의 limits 인자가 None이면 기존 검사 경로를 유지한다. 운영용 worker에서는 이 선택 경로를 그대로 허용하지 않고 WorkerRequest와 신뢰 정책으로 제한을 필수 결합해야 한다. 이 합성 검사 CLI는 실데이터 worker를 활성화하지 않는다.

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

## 배열 통합 작업 메모리 실측

2026-10-23 사용자 관리자 PowerShell에서 실제 배열 worker·AppContainer·Job 경로로 수행했다. 기본 용량 시험의 합성 float32 표본은 `(N, 256, 256, 1)`이고 라벨은 bool, 채널 축은 마지막, random 분할 0.70/0.15/0.15·시드 42다. 아래 기본 용량 시험은 대치·clipping·스케일링을 사용하지 않았으며, 옵션·dtype 시험은 별도로 구분한다. 모델 호출 대신 빈 합성 권장 응답을 사용했으며 승인·계산·부모 결과 검증은 기존 코드를 거쳤다. 제품 코드와 기본 Job 2 GiB 제한은 바꾸지 않았다.

| 측정 항목 | 32 MiB 정상 시험 | 128 MiB 정상 시험 |
| --- | --- | --- |
| 표본 수 / Train·Validation·Test | 128 / 89·19·20 | 512 / 358·77·77 |
| 진단 최대 Job 메모리(MiB) | 349.02 | 1024.56 |
| 후보 최대 Job 메모리(MiB) | 372.29 | 1159.50 |
| 적용 최대 Job 메모리(MiB) | 372.25 | 1159.55 |
| 부모 private 최대 증가량(MiB) | 78.47 | 104.43 |
| 부모 RSS 최대 관측값(MiB) | 235.98 | 264.61 |
| 시스템 여유 메모리 최소 관측값(GiB) | 9.81 | 9.21 |
| 전체 경과 시간(초) | 65.812 | 67.797 |

두 정상 시험 모두 진단·후보·적용 종료 코드 0, 활성 프로세스 0, Job 정리·프로필 정리 완료다. 커밋·해시·도메인 검증을 다시 통과했고, Train·Validation 배열과 bool 라벨을 합성 입력의 해당 표본 위치·값·타입·순서에 대조했다. 전체 표본 위치는 중복 없이 분할됐으며 Test 배열은 출력하지 않았다. 보고서 해시도 일치했다.

부모 private/RSS·시스템 여유 메모리는 50ms 간격 표본 측정이다. 연속 측정의 정확한 최대값이나 worker Job 통계와 같은 지표로 해석하지 않는다. 부모 private 증가량은 입력 생성 후 기준 대비이며, 부모는 worker Job 밖에 있다. 시험 감시기는 여유 RAM 6 GiB 미만 또는 부모 private 기준 대비 2 GiB 초과를 관찰하면 worker 취소 신호를 보내도록 구성했다. 이 시험에서 감시 조건은 발생하지 않았고, 이 장치는 부모 프로세스 자체의 OS 메모리 상한을 설정한 것이 아니다.

### 옵션·dtype 비교

동일 형상 `(128, 256, 256, 1)`·분할·시드로 대치 → clipping → 표준화 → 조합 → uint8 → float64를 순차 실행했고 6건 모두 통과했다. float32 기준 입력은 32 MiB이며, dtype 비교는 원소 수를 고정하므로 uint8은 8 MiB, float64는 64 MiB다. 비유한 값 대치 시험에는 NaN·무한대를 넣었고, clipping은 1·99 분위수, 표준화는 standard를 사용했다. dtype 비교는 처리 옵션을 끄고 float32 출력 변환을 확인했다.

| 시험 | 입력(MiB) | worker 최대 Job 메모리(MiB, 세 단계 중 최대) | 부모 private 최대 증가량(MiB) | 전체 시간(초, 별도 대조 포함) |
| --- | --- | --- | --- | --- |
| Train 중앙값 대치 | 32 | 372.85 | 101.01 | 69.828 |
| 분위수 clipping | 32 | 372.84 | 101.74 | 69.062 |
| standard 표준화 | 32 | 372.69 | 101.97 | 68.829 |
| 대치 + clipping + standard | 32 | 372.27 | 105.57 | 71.687 |
| uint8 | 8 | 336.71 | 107.67 | 70.390 |
| float64 | 64 | 427.63 | 79.40 | 71.546 |

총 18개 격리 단계는 종료 코드 0·활성 프로세스 0·Job 및 프로필 정리 완료였고, OS 자원 초과 통지·부모 감시 조건·오류 출력은 없었다. 시스템 여유 메모리 최소 관측값은 약 9.62 GiB였다. 원본·snapshot·보고서 해시, 커밋·도메인 검증과 요청 식별값도 다시 대조했다.

Train에서 산출한 중앙값·clip 경계·center·scale과 Train·Validation 출력 전체를 별도 NumPy 계산으로 확인했다. 기준값 비교 허용 오차는 `rtol=1e-12, atol=1e-12`, float32 출력은 `rtol=1e-6, atol=1e-6`이다. 표본은 Train 89·Validation 19·Test 20으로 빠짐없이 나뉘었고, bool 라벨 타입·값·순서가 보존됐으며 Test 배열은 출력하지 않았다. float64 시험의 일치는 float32로 변환한 기대값과의 일치이지 원래 정밀도 보존을 뜻하지 않는다.

별도 NumPy 대조는 native 처리·부모 검증의 메모리 측정을 종료한 뒤 수행했으며 위 메모리 수치에 포함하지 않았다. 실제 처리 시간은 각 기록의 `pipeline_elapsed_seconds`, 별도 대조를 포함한 시간은 `elapsed_seconds`로 구분한다. 이번 표본 측정은 대용량 입력의 최대 메모리 보장이나 성능 벤치마크가 아니다.

근거: `logs/agents/memory_validation/suite_20261023_142221_768120/summary.json` 및 여기에 연결된 6개 시험 폴더의 `measurement.json`·단계별 `transport.json`·작업/결과 파일. 검토 때 원본 기록을 바꾸거나 재실행·모델 호출을 하지 않았다.

### 제한 초과 대응

동일한 32 MiB 합성 입력에 시험 전용 Job 256 MiB·동시 프로세스 2·시간 300초 한도를 적용했다. `resource_messages: [10]`과 `resource_exceeded`가 확인됐고, 활성 프로세스 0·Job 정리·AppContainer 프로필 정리가 완료됐다. 작업은 `needs_attention`, 예약 실행은 `failed`로 남았다. 진단 결과는 커밋되지 않았고 서비스의 결과 조회도 거부됐다. 결정 C·D, 후보·적용·최종 보고서는 생성되지 않았다. 판정은 **메모리 제한 초과 시 중단·정리·미완성 결과 등록 차단 정상**이다.

이때 최대 Job 메모리는 280.58 MiB로 기록됐다. 초과 감지 후 중단하는 시험의 기록이며 이 수치만으로 결함이라고 판정하지 않는다. 사용량이 항상 설정값 이내라는 증명으로도 쓰지 않는다. 전체 시험은 20.016초였고 부모 감시기의 별도 중단 조건은 발생하지 않았다.

근거는 다음 폴더의 `measurement.json`과 해당 파일에 연결된 단계별 `transport.json`·작업 기록·커밋 산출물이다. 원본 기록은 수정하지 않았다.

- 정상 32 MiB: `logs/agents/memory_validation/20261023_140202_595929` / 작업 `956c1ad216e84491b620cd4bfd10e0f9`
- 정상 128 MiB: `logs/agents/memory_validation/20261023_140516_565690` / 작업 `71cab63bfe324a9096c5d4741a8201eb`
- 제한 초과: `logs/agents/memory_validation/20261023_140930_434158` / 작업 `01bcc7cd70bd436ca49301e7034dd1b0`

비승격 셸의 앞선 준비 실패는 Low Integrity 설정 오류 5로 계산 전에 중단된 별도 기록이며, 메모리 시험 실패로 합산하지 않는다. 256 MiB 이상 입력·2 GiB 근처 입력, 여러 채널·다른 형상·그 밖의 dtype/처리 설정의 부하는 미확인이다. 이번 수치로 지원 최대 용량을 추정하거나 보장하지 않는다. 이후 최신 코드 전체 회귀도 통과했으며, 집계·생략 범위는 [운영 안내](../GUIDE.md#검증-상태와-남은-제한)를 따른다. 전체 회귀와 위 실제 메모리 실측은 별개의 검증이다.

## 검증 기록

2026-10-16 신규 10개 테스트 통과. 3개는 계약·실행 순서 모의 검증이며 7개는 실제 Windows 합성 프로세스로 정상 종료·메모리 제한·프로세스 제한·시간 초과·취소·잔존 자식 정리·마지막 Job 핸들 닫힘 시 정지된 worker 종료를 확인했다. 실제 AppContainer 프로필이나 감사 설정은 변경하지 않았다.

전체 에이전트 회귀 검증은 166개 중 165개 통과·기존 Windows 링크 생성 권한 테스트 1개 건너뜀이다.

진행·생성 시도 진단 추가 후 신규 테스트 8개 통과. 정보 최소화·고정 코드 분류·부분 줄 복구·기록 상한·쓰기 실패·잘못된 입력·별도 Python의 실제 감사 hook을 검증했다. 전체는 174개 중 173개 통과·기존 1개 건너뜀이다. 전역 테스트 실행기에는 hook을 설치하지 않았다.

```powershell
python -m unittest tests.agents.test_job_object -v
```

Microsoft 참조: [Job Objects](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects), [Extended Limit Information](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_extended_limit_information).
