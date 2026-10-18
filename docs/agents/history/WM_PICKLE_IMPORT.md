# WM pickle의 AppContainer 변환

> 초기 구현·실행 검증 이력입니다. 현재 상태와 권장 진입점은 [에이전트 문서 안내](../GUIDE.md)를 따릅니다. 당시 명령과 결과는 이력 보존용이며 현재 계약으로 자동 재개된다는 뜻이 아닙니다.

> 이 문서는 최초 WM 전용 연결 기록이다. 현재 구현·명령은 [구조 판별형 pickle 변환](../guides/PICKLE_IMPORT.md)을 따른다. `wm_pickle_*`는 호환 진입점이며 새 실행은 `pickle_import` 로그에 기록한다. 아래 합성 예제를 WM 구조로 실행하려면 `create-demo`에 `--demo-kind wm`을 추가한다. 과거 실행은 새 코드·계약으로 자동 재개하지 않는다.

`wm_pickle_import_cli`는 기존 승인 DB·규칙 실행기·AppContainer·Job Object에 WM-811K 원본 변환을 연결한다. 현재 실행 계획은 `preflight → ingest`다. 부모는 원본을 바이트 복사하고, 승인 후 생성한 AppContainer worker만 pickle을 역직렬화한다. 기존 WM canonical 변환 함수와 숫자 배열 검증기를 재사용한다.

worker 진입점은 실제 AppContainer 토큰·capability 0개·Job 소속을 확인한 후에만 원본을 읽는다. 일반 Python에서 worker 파일을 직접 실행하면 파일 읽기 전에 종료한다.

## 실행과 결과

```text
파일 선택 → 바이트 snapshot·해시 고정 → preflight → 사용자 승인
  → 기존 AppContainer + Job에서 pd.read_pickle
  → manifest·숫자 NPZ·provenance·SUCCESS 생성
  → Job 종료·프로필 정리 확인
  → 부모의 출처·파일 목록·크기·manifest·숫자 배열 검증
  → 승인 실행기에 canonical 결과 커밋
```

승인은 입력 해시·변환 코드·출력 상한·shard 크기·자원 요청·계획에 결합된다. 설정이나 코드가 변경되면 새 실행을 생성해야 한다. 취소·실행 실패·정리 증거 누락·결과 변조는 완료 등록하지 않는다. 부분 결과는 작업 폴더에 보존한다.

`status`가 반환하는 `local_canonical_directory`를 후속 WM canonical 검증·분할·EDA 입력으로 사용할 수 있다. 기존에 변환한 canonical 데이터는 다시 pickle에서 생성할 필요가 없다. 이 CLI는 원본 변환 단계이며, CSV용 `pipeline_approval_cli`의 분석 어댑터 등록을 변경하지 않는다. WM canonical 입력을 그 모듈형 분석 경로에 직접 연결하는 것은 별도 작업이다.

## 합성 입력으로 연결 확인

프로젝트 루트의 `ml310` 환경에서 새 실행 이름으로 진행한다.

```powershell
python -m src.agents.wm_pickle_import_cli create-demo `
  --name wm-pickle-demo01 `
  --memory-mib 2048 --processes 2 --wall-seconds 60

python -m src.agents.wm_pickle_import_cli next --name wm-pickle-demo01
python -m src.agents.wm_pickle_import_cli next --name wm-pickle-demo01
python -m src.agents.wm_pickle_import_cli approve --name wm-pickle-demo01
```

승인 화면에서 파일 해시와 자원 한도를 확인하고 표시된 `PICKLE-...` 문구를 입력한다. 반환된 승인 ID로 실행한다.

```powershell
python -m src.agents.wm_pickle_import_cli next `
  --name wm-pickle-demo01 --approval-id 발급된_ID

python -m src.agents.wm_pickle_import_cli status --name wm-pickle-demo01
```

정상 결과는 `status=completed`, `canonical_integrity=passed`다. `training_ready=false`는 변환·정합성 검사 다음에 품질 검토와 분할을 수행해야 한다는 뜻이다. `worker_started`는 해당 `next` 호출에서 새 프로세스를 시작했는지 나타내며, 작업 생성과 승인 대기에서는 false다.

## 사용자가 선택한 원본

아래 경로와 한도는 예시이며, 입력 파일 크기·전체 pickle 로딩 메모리·예상 출력량에 맞춰 지정한다. `max-input-bytes`는 복사 파일 상한이고 `memory-mib`는 실행 프로세스들의 메모리 상한이다.

```powershell
python -m src.agents.wm_pickle_import_cli create `
  --name wm-pickle-import01 `
  --source data/raw/wm811k/LSWMD.pkl `
  --max-input-bytes 4294967296 --max-output-bytes 8589934592 `
  --memory-mib 16384 --processes 2 --wall-seconds 1800 `
  --confirm-pickle-selection
```

이후 절차는 합성 예제의 `next → next → approve → next --approval-id → status`와 같다. 생성 명령은 역직렬화하지 않으며, 복사본을 만든다. 실제 원본은 출처와 승인 화면의 SHA-256을 확인한 후 실행한다.

## 출력 상한과 검증 범위

`max-output-bytes`는 변환기가 기록하는 manifest·NPZ·conversion·provenance·SUCCESS의 실제 바이트 합계다. UTF-8 바이너리 기록으로 Windows 줄바꿈 증가를 없애고, ZIP 헤더를 다시 쓰는 경우에는 파일 길이가 증가한 만큼만 센다. 상한을 넘는 쓰기는 기록 전에 거부한다. 부모도 실제 파일 크기 합계를 독립적으로 확인한다.

이 제한은 등록된 변환 함수의 출력 제한이며, 임의 pickle 코드의 모든 디스크 쓰기에 대한 OS 디스크 할당량은 아니다. AppContainer의 파일·네트워크 접근 제한과 Job의 메모리·프로세스·시간 제한을 함께 적용한다. canonical 검증은 `allow_pickle=False`로 숫자 배열의 구조·해시·다이 통계를 검사한다. 원본 pickle과 결과의 의미적 동등성이나 pickle의 무해성을 인증하지 않는다.

진단은 worker의 `scratch/task/diagnostic.json`에 단계와 고정 오류 코드만 남긴다. 원본 값·원문 예외·LLM 전송은 없다.

## 소스와 테스트

2026-10-17 확인: 아래 변환·검증·승인 연결 테스트와 기존 WM EDA·분할·어댑터 회귀 검사, 실제 AppContainer 합성 왕복을 합쳐 50개 테스트가 통과했다. 실제 원본 WM pickle은 재변환하지 않았다.

| 파일 | 역할 |
| --- | --- |
| `src/agents/wm_pickle_contract.py` | snapshot·출력·자원·설정 계약 |
| `src/agents/wm_pickle_import_cli.py` | 생성·승인·단계 진행·상태·취소 |
| `src/agents/wm_pickle_import.py` | 승인 실행기와 worker·부모 검증·커밋 연결 |
| `src/agents/wm_pickle_import_transport.py` | 기존 AppContainer·ACL·Job·런타임 복사 |
| `src/agents/wm_pickle_import_worker.py` | 격리된 pickle 역직렬화·canonical 변환 |
| `src/agents/wm_pickle_results.py` | 출력 바이트·출처·manifest·숫자 배열 검증 |
| `scripts/wm_sandbox/export_canonical.py` | 공통 변환 및 전체 묶음 출력 제한 |

```powershell
python -m unittest tests.test_wm_canonical_export tests.test_validate_canonical tests.agents.test_wm_pickle_import -v
```

실제 AppContainer 검사는 명시적으로 활성화하며 테스트가 만든 합성 pickle만 처리한다. 임시 프로필·폴더를 생성하고 종료 후 정리한다.

```powershell
$env:WAFERGUARD_TEST_APPCONTAINER = '1'
python -m unittest tests.agents.test_wm_pickle_appcontainer -v
Remove-Item Env:WAFERGUARD_TEST_APPCONTAINER
```
