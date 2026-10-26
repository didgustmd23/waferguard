# AppContainer 호환성 검사 — 2단계

고정 합성 probe의 런타임·파일 접근·토큰 검사와 외부·loopback WFP 폐기 확인 이력이 있다. 최신 근거와 한계는 [GUIDE](../GUIDE.md#네트워크-시험)·[WFP 진단](WFP_DIAGNOSTIC.md#etl-대조-근거와-확인-범위)을 따른다. 아래 초기 권한 준비 실패와 후속 실제 실행을 구분하며, 운영 worker 전체나 SEC-12 전체 검증 완료를 뜻하지 않는다.

## 검사 범위

기존 ml310 환경은 읽기만 하며 Python·표준 라이브러리·고정된 NumPy/pandas/SciPy/sklearn/Matplotlib 및 의존 패키지를 새 검사 폴더에 복사한다. 임의 설치 스크립트·.pth·프로젝트·실데이터를 복사하지 않는다. 신규 런타임은 최대 512MiB로 제한한다. 이번 환경의 복사 예정 파일 크기는 약 344MiB이며 실제 복사 이전에 권한 준비가 실패했다.

일회성 AppContainer 프로필에는 네트워크 capability를 주지 않는다. 새 폴더만 대상으로 runtime/input/tools는 RX, output은 Modify와 Low Integrity를 설정한다. 별도 protected 폴더에는 AppContainer 권한을 주지 않는다. 기존 Python·프로젝트 루트·데이터·시스템 방화벽 ACL은 변경하지 않는다.

실행할 고정 합성 payload는 다음을 검사한다.

- 실제 프로세스 토큰이 AppContainer인지 확인.
- 허용한 합성 파일 읽기·출력 폴더 쓰기.
- 합성 보호 파일 읽기와 입력 파일 쓰기의 접근 거부.
- NumPy·pandas·sklearn·Matplotlib import와 작은 Pipeline·CSV·Agg 그림 생성.
- 부모에서 연결을 확인한 로컬 서버와 문서용 예약 주소 192.0.2.1에 대한 TCP 연결 차단.

네트워크 검사는 합성 연결만 시도하고 원본·키·상세 데이터는 전송하지 않는다. 접근 거부 오류만 차단 성공으로 인정하며, 시간 초과·연결 거부·파일 없음은 통과로 처리하지 않는다. 허용 동작·라이브러리·토큰·거부 검사를 모두 통과해야 compatibility_passed로 기록한다.

## 실행

프로젝트 루트에서 Python 3.10 환경을 활성화한 뒤 실행한다. Windows의 AppContainer 프로필 생성 및 새 출력 폴더 무결성 레벨 설정 권한이 필요하다.

```powershell
python -m src.agents.appcontainer_probe --confirm-isolation-probe
```

기존 폴더를 재사용하거나 덮어쓰지 않고 `logs/agents/appcontainer_probes/<새 ID>/`를 생성한다. 승인 스위치는 검사 전용 복사·프로필·새 폴더 ACL 조정의 범위를 확인하는 장치이며 운영용 에이전트 승인 API가 아니다.

compatibility_passed일 때 종료 코드 0, 검토 필요·검사 미완료일 때 2를 반환한다.

도구는 명시한 인터프리터의 sys.prefix에서만 런타임을 선택한다. Microsoft Store의 실행 별칭이 아니라 설치된 Python 3.10을 사용해야 한다. 기존 ml310을 다시 설치하거나 패키지를 다운로드하는 과정은 없다.

무결성 설정에서 Access is denied가 발생하면 관리자가 해당 폴더의 소유권·WRITE_OWNER 등 권한을 확인해야 한다. 필요하면 사용자가 허용한 관리자 PowerShell에서 위 명령을 새 실행으로 재검증한다. 이 문서가 다른 사용자 파일의 소유권 변경이나 프로젝트 전체 권한 완화를 허용하는 것은 아니다. 관리자 실행이 모든 보안 정책 제한을 해결한다고 보장하지 않는다.

## 결과와 정리

- result.json: 단계 상태·고정 실패 코드·숫자 오류·프로필 정리 여부.
- output/child_result.json: payload가 실행된 경우의 고정 검사 결과.
- output/: 합성 CSV·그림·쓰기 검사 파일과 Matplotlib 캐시.
- runtime/: 검사 전용 복사본. 추후 재사용은 별도 검증·정책 필요.

부모는 환경 변수를 최소화하고 API 키·사용자 PYTHONPATH·부모 PATH·상속 핸들을 전달하지 않는다. Windows 프로필 조회용 SystemRoot·LOCALAPPDATA·USERPROFILE 및 존재하는 APPDATA는 명시적 allowlist로 보존한다. 이 경로 전달이 해당 사용자 폴더의 읽기 권한을 부여하는 것은 아니며 AppContainer token·ACL 제한을 유지한다. python -I -S -B로 실행하며 복사한 site-packages만 payload가 명시적으로 등록한다. 프로세스는 창 없이 실행하고 실험용 60초 제한 후 종료한다. 일반 worker의 Job Object·프로세스 트리 자원 제한은 다음 구현 범위다.

매 실행에서 자신이 새로 만든 AppContainer 프로필만 finally에서 삭제한다. 삭제가 실패하면 상태를 review_required로 유지하고 로컬 result.json에 프로필명을 남긴다. 검사 폴더와 부분 결과는 자동 삭제하지 않으며 사용자가 확인 후 정리한다.

## 실제 실행 이력

2026-10-16 두 번의 합성 실험에서 프로필 생성 및 SID 기반 RX/Modify ACL 부여까지 진행했으나, output 폴더의 Low Integrity 설정에서 Access is denied가 발생했다. 해당 설정을 검사 폴더 한 곳에서 별도로 확인했을 때도 동일하게 거부됐다. 두 실험 모두 프로필 정리는 성공했고 런타임 복사·payload 실행 이전에 중단했다.

검사 기록은 다음에 남겼다.

- logs/agents/appcontainer_probes/096f6657f7054674b2df1618f751b2fb/result.json
- logs/agents/appcontainer_probes/e4e4f5ddf1874d60aaba15eff5234982/result.json

첫 기록은 일반 실패 코드, 두 번째 기록은 acl_setup 단계 코드까지 포함한다. 이후 도구에는 PROBE_INTEGRITY_SETUP_FAILED와 숫자 종료 코드 기록을 보완했다. 두 실행의 검사 폴더는 보존했고 원본 환경·방화벽을 변경하지 않았다. **해당 두 실행의 실제 OS 결과는 review_required이며 당시 실데이터 worker는 비활성이었다.** 이후 실행과 현재 검증 범위는 아래 이력과 GUIDE를 따른다.

후속 사용자 실행 `logs/agents/appcontainer_probes/5d0304f6e4ba44ed9bed3c3a092f2ba2/result.json`에서는 ACL 설정과 런타임 복사를 통과했고 process_launch에서 APPCONTAINER_PROCESS_CREATE_FAILED / 오류 203으로 중단했다. 203은 ERROR_ENVVAR_NOT_FOUND이며, 최소 환경 구성에서 Windows 프로필 조회 변수가 빠져 있던 점을 확인해 allowlist로 보완했다. 환경 블록 정렬도 대소문자 무시 순서로 변경했다. 이 수정으로 실제 프로세스 생성 문제가 해결됐는지는 사용자 재실행으로 확인해야 한다. 해당 실행의 AppContainer 프로필 정리는 성공했다.

다음 사용자 실행 `logs/agents/appcontainer_probes/4da64a637e954e2489bd264e503a7dcf/result.json`에서는 프로세스 생성·종료(코드 0), AppContainer 토큰, 합성 입력 읽기·출력 쓰기, 보호 파일 읽기·입력 쓰기 접근 거부, 외부 연결 오류 10013을 확인했다. NumPy·pandas·Matplotlib import는 통과했으나 sklearn import와 로컬 연결 접근 거부 확인은 미완료였다. 따라서 전체 결과는 review_required다.

동일 런타임 복사본을 -I -S -B로 분리 검사한 결과 joblib에서 요구하는 cloudpickle 패키지가 빠져 있었다. 복사 allowlist에 cloudpickle과 해당 배포 메타데이터를 추가했다. 원본 pickle을 읽거나 cloudpickle.loads 등을 실행하는 변경이 아니다. 로컬 연결의 기존 결과는 denied=false·error_code=null이므로 성공 연결로 단정하지 않으며, 후속 payload에 timeout/access_denied/connection_refused/connected_or_accessible 구분을 추가했다. import 실패에도 원문 대신 제한된 누락 모듈 코드만 남긴다. 추가 수정의 실제 호환성 결과는 다음 사용자 실행으로 확인해야 한다.

다음 사용자 실행 `logs/agents/appcontainer_probes/4df19725a52b49738943181745a6ac24/result.json`에서는 네 라이브러리 import·합성 Pipeline·CSV·Agg 그림 생성까지 모두 통과했다. 파일 허용/거부 검사는 통과했고 외부 연결은 오류 10013으로 거부됐다. 로컬 연결은 timeout이므로 전체 상태는 여전히 review_required다. 이 실행은 아래 보완 전의 결과이며 새 토큰 capability·전후 대조·예외 조회까지 통과했다는 뜻이 아니다.

## 로컬 통신 진단 보완

후속 실행 `032b4ac3d74d469e810d483577eec305`에서 전후 부모 연결 각 1건·child 연결 0건, capability 0개, 전후 loopback 예외 없음, 런타임·파일·외부 연결 검사 통과를 확인했다. child 로컬 연결은 timeout이어서 `LOOPBACK_OS_DROP_EVIDENCE_MISSING`을 유지했다. 프로필 정리는 성공했다. 당시에는 선택적 WFP 도구의 실제 수집 전이었으며, 이후 실제 수집과 재검토 결과는 [WFP 진단](WFP_DIAGNOSTIC.md)의 기존 ETL 재검토 절에 정리한다.

2026-10-23 기존 수집 실행 `3e4d66b4156a472fae626845257a1acd`·`ef25fb95cabf40e99bc2879fa923019f`의 ETL에서 시험 포트·시각·검사 PID에 대응하는 loopback 폐기 이벤트와, 격리 실행 파일에 대응하는 외부 시험 폐기 이벤트(ID 1001)를 확인했다. loopback AppId는 수신 측 부모 실행 파일로 기록돼 특정 차단 정책의 원인 귀속은 미확정이다. 해당 시험 트래픽의 WFP 폐기 확인이지 SEC-12 전체 검증 완료가 아니며, 원본 JSON의 timeout·검토 필요·`drop_evidence_verified: false`는 보존한다. 이번 재검토에서는 새 접속·수집을 하지 않았다.

`src/agents/loopback_probe.py`는 다음 증거를 수집한다. 실데이터나 LLM은 사용하지 않는다.

1. 부모가 127.0.0.1 임시 포트 서버를 열고 검사 직전·직후 동일 서버에 연결하여 응답을 확인한다.
2. 서버는 전후 대조와 child의 32바이트 합성 식별 토큰을 구분해 연결 건수만 기록한다. 토큰·수신 원문·상대 주소는 결과에 기록하지 않는다.
3. child는 자신의 AppContainer 토큰과 TokenCapabilities를 조회한다. 조회 성공과 capability 개수 0을 별도로 확인하며, 조회 실패를 0으로 처리하지 않는다.
4. 부모는 `CheckNetIsolation.exe LoopbackExempt -s`로 검사 프로필 SID의 예외 여부를 전후 조회한다. 다른 앱 이름·SID 목록은 결과에 보존하지 않는다. 조회 실패는 `unknown`이며 예외 없음으로 해석하지 않는다.
5. 런타임·파일 접근·토큰·외부 통신·로컬 통신의 상태를 `component_status`에 분리한다. 일부 통과는 전체 통과가 아니다.

child의 연결 성공 또는 서버의 child 토큰 수신이 확인되면 `LOOPBACK_CONNECTION_OBSERVED`로 실패한다. 전후 대조·토큰·예외 검사가 모두 정상이고 명시적 접근 거부가 있을 때만 로컬 통신 검사를 통과한다.

전후 대조가 정상이어도 timeout은 `LOOPBACK_OS_DROP_EVIDENCE_MISSING`으로 검토 필요를 유지한다. 서버에서 연결을 관찰하지 못한 사실만으로 OS가 패킷을 차단했다고 증명할 수 없기 때문이다. 필요하면 다음 단계에서 해당 프로세스·목적지와 일치하는 Windows Filtering Platform 차단 증거를 별도로 확인한다. 시스템 감사 정책 변경이나 광범위 트래픽 캡처는 이번 구현에 포함하지 않으며 별도 승인 없이 실행하지 않는다.

방화벽·loopback 예외는 변경하지 않는다. 일회성 프로필과 새 검사 폴더 권한만 기존 검사 범위대로 설정한다. 새 코드의 OS 검증은 관리자 PowerShell의 Python 3.10 환경에서 위 명령으로 별도 실행해야 한다. 이전 결과를 새 검사 통과로 소급하지 않으며 실데이터 worker는 계속 비활성이다.

## 검증·제한

```powershell
python -m unittest tests.agents.test_appcontainer_probe -v
python -m unittest tests.agents.test_loopback_probe -v
python -m unittest discover -s tests/agents -q
```

합성 테스트는 판정의 누락/실패/시간 초과 거부, 복사 allowlist, 무 capability 구조 초기화를 검사한다. 네이티브 API·ACL·실제 Python 시작의 성공을 대체하지 않는다. AppContainer의 기본 시스템 접근과 프로필 저장소도 있으므로, 향후 운영 worker에서는 정확한 런타임·입출력 권한 및 공격 표면을 더 검증해야 한다.

2026-10-16: 전후 연결·child 수신·timeout 보수적 판정·증거 누락·읽기 전용 예외 조회·UTF-16 출력 처리 테스트를 추가했다. 실제 AppContainer 차단의 증명은 합성 단위 테스트와 구분한다.

당시 전체 에이전트 합성 테스트 145개 중 144개 통과, Windows 링크 생성 권한에 따른 기존 1개 건너뜀을 확인했다. 당시에는 새 진단 코드의 실제 AppContainer 재실행 전이었다. 이후 실행·ETL 재검토는 위 기록과 WFP 진단을 참조한다.

Microsoft 참조: [AppContainer 실행](https://learn.microsoft.com/en-us/windows/win32/secauthz/implementing-an-appcontainer), [프로필 생성](https://learn.microsoft.com/en-us/windows/win32/api/userenv/nf-userenv-createappcontainerprofile). 이 실험은 Microsoft 인증이나 격리 보안 검증 완료 선언이 아니다.
