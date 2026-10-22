# 메타데이터 전용 모듈 등록기

현재 기준(2026-10-20): 명세 기반 공통 UI에 CSV 5개 작업·WM 정상 개발 6단계를 연결했다. EDA/CSV 자연어 제안·로컬 결과 해설과 B의 저장 제안 목록 복원·설치 모델 조회는 [GUIDE](../GUIDE.md#현재-구현-상태)를 따른다. 아래 날짜별 구현 기록의 미연결·수동 진행 안내보다 현재 공통 UI의 승인 지점 중심 진행을 우선한다. 승인 불필요 단계는 진행하되 승인 대기·실패·품질 검토 필요에서는 멈추며 자동 승인하지 않는다.

## Dataset 버전 공통화·3.12 CPU 정책·공통 UI 통합 — 2026-10-19

현재 런타임은 `src/agents/wm_runtime.py`를 기준으로 한다. 3.12는 아래 두 빌드에서 정상 6단계만 지원하며, CUDA 빌드에서도 worker는 CPU를 사용한다. 아래의 이전 ‘3.12 미지원/CPU 빌드만 지원’ 기록은 해당 시점 이력이다.

### 1. 실제 Dataset 경로의 빌드 버전 검사

합성 검사기에 있던 버전 판독을 `wm_dataset_runtime.pinned_torch_version`으로 공통화했습니다. `runtime_manifest`는 배포 메타데이터 문자열이 아니라 해시로 고정한 `torch/version.py`의 전체 빌드 버전을 반환합니다. 따라서 실제 Dataset 생성·승인 계약에는 `2.14.1+cpu`가 저장되고 worker의 기존 정확 비교를 유지합니다. 공개 버전이 다르거나 metadata의 local 표기와 빌드가 다르면 계속 거부합니다. 버전 파일은 실행하지 않으며 Conda 설치의 하드 링크는 기존 복사 정책과 동일하게 허용하고 심볼릭 링크/재분석 지점·파일 변경은 거부합니다. `wm_cpu_probe`도 같은 함수를 사용하며 기존 import 호환 별칭을 유지합니다.

### 2. Python 3.12 지원 범위

| 환경 | WM 에이전트 지원 범위 |
| --- | --- |
| Windows/Python 3.10 | 기존 정책 유지 |
| Windows/Python 3.12 + `2.14.1+cpu` 또는 `2.14.1+cu130` | canonical EDA·분할 제안·분할 확정·변환 검사·개발 캐시·Dataset 검사 6단계, CPU 연산 |
| Python 3.12의 다른 빌드·충돌 후보/제외 분기·모듈 미지정, 실제 GPU 장치 실행 | 미지원, 실행 차단 |

8개 CLI preflight와 공통 서비스는 모듈 ID를 같은 `wm_runtime.runtime_status(module_id)`에 전달합니다. 정책 조회는 Torch를 import하거나 GPU를 초기화하지 않고 현재 환경의 작은 버전 파일과 배포 정보를 읽습니다. 3.12는 기존 native CPU 합성 검증 범위에 한정합니다. 입력 검증·코드/런타임 해시·단계별 승인·AppContainer·Job·결과 검증·취소/중복 방지는 유지합니다. 비격리 fallback이나 품질 검토 자동 해제는 추가하지 않았습니다.

실제 설치 환경을 읽기 전용으로 확인한 결과 `dl312`의 `2.14.1+cpu`는 정상 6단계 지원, `wm-gpu312`의 `2.14.1+cu130`은 WM 격리 경로 미지원입니다. 후자는 GPU 학습 환경을 사용할 수 없다는 뜻이 아니라, GPU 빌드를 복사한 AppContainer 실행을 아직 허용하지 않는다는 뜻입니다. 학습 환경·캐시·모델은 변경하지 않았습니다.

### 3. 공통 UI 통합 검사와 실제 실행 경계

`tests/agents/test_ui_wm_integration.py`는 고정 합성 canonical과 실제 등록부·생성 서비스·승인 DB·실행기·WM 계산 코어·결과 검증기를 사용합니다. OS 운송만 모의합니다. UI에서 모듈 선택→검토→입력/생성 미리보기→새 작업 준비→사전 검사→별도 승인→별도 실행→표/그래프/보고서 표시를 확인합니다. 품질 검토 필요이면 EDA를 중단하고 로컬 품질 보고서만 표시하는 경로도 검사합니다. 재렌더링은 worker를 다시 호출하지 않습니다. 테스트 DB/생성물은 임시 합성 영역에만 만들며 사용자의 실제 승인·실데이터 작업은 만들지 않습니다.

이 통합 검사는 실제 GUI에서 사용자 승인 후 native worker를 실행한 결과가 아닙니다. 다음은 `dl312`로 UI 서버를 재시작하고 새 합성 작업을 선택해 사용자가 승인/실행을 확인하는 단계입니다. 기존 코드 해시의 실행·승인은 재사용하지 않으며 로그/산출물은 보존합니다. 관리 데이터/기존 숫자 캐시를 다시 원본 pickle에서 변환할 필요는 없습니다. 새 소스와 새 승인 범위의 작업만 별도로 준비합니다.

최종 회귀: 공통 버전/정책·Dataset·합성 CPU 검사·모듈 명세·CSV/WM UI·실제 서비스 기반 UI 정상/품질 중단 통합을 합쳐 **144개 통과**했습니다. 실제 UI에서의 사용자 승인·native 실행은 별도 확인 대상으로 남겼습니다. 이 검증에서는 실제 데이터·GPU 학습·사용자 승인 DB를 사용하지 않았습니다.

## PyTorch·합성 WM CPU 검사 준비 — 2026-10-19

최신 실제 검사 `16444caf34434afab4ca9e3162f71cd0`: Python 3.12.15의 AppContainer에서 11개 합성 검사·worker 정상 종료·Job/프로필 정리가 모두 성공했습니다. Train 36·Validation 9·Test 9개로 배정했고 개발 배치는 `[4,2,32,32]`, float32 입력·int64 라벨입니다. 원래 parent는 배포 메타데이터 `2.14.1`과 worker의 `2.14.1+cpu`를 문자열로 직접 비교해 최종 상태를 `review_required`로 기록했습니다.

parent를 보완해 고정 런타임 명세의 `torch/version.py`를 크기·SHA-256으로 확인하고 AST의 단일 버전 문자열만 읽도록 했습니다. 파일을 실행하거나 `+cpu`를 단순히 무시하지 않습니다. 공개 버전은 배포 메타데이터와 비교하고, worker 버전은 해시로 고정한 빌드 문자열과 정확히 대조합니다. metadata에 local 표기가 있으면 그것도 일치해야 합니다. 부모 판정 항목을 결과와 콘솔에 구분해 표시합니다.

버전 표기·파일 변조·모호한 정의 거부 등 신규 4개를 포함한 관련 회귀 **29개 통과**. 이어서 저장된 원래 런타임 manifest 지문·복사 코드 지문·버전 파일·child 원문·고유 probe ID·종료/자원/정리 증거를 읽기 전용으로 재검토해 모든 부모 판정 조건이 참임을 확인했습니다. 원래 `result.json`은 변경하지 않았고 worker도 재실행하지 않았습니다. 이는 해당 고정 CPU 합성 처리 경로의 native 성공 증거이며 일반 WM 실데이터·GPU·모델·전체 보안 인증은 아닙니다. WM 3.12 지원 정책은 그대로 제한합니다.

실제 검사 `c489a7bdeb414a87a6b64cec00e9467a`는 AppContainer에서 PyTorch/WM 코어 import와 CPU 연산 단계를 지나 `SYNTHETIC_CANONICAL`에서 중단됐습니다. synthetic canonical 파일은 생성됐지만 validation workspace는 생성되지 않았습니다. 자원 초과는 없었고 프로세스·프로필 정리를 확인했습니다. 당시 오류가 일반 코드로만 기록돼 정확한 OS 예외는 보존되지 않았습니다.

생성 직후의 `inventory`가 부모용 `_root`를 호출해 허용되지 않은 프로젝트 상위 경로를 조회하는 부분을 발견했습니다. 기존 WM worker와 동일한 `staged=True`로 두 파일 지문 검사를 변경했으며 파일 목록·링크/하드 링크·바이트 상한·해시 검사와 부모 측 검증은 유지합니다. `CANONICAL_INVENTORY`/`CANONICAL_VALIDATION` 단계 진단과 정제된 접근 거부·OS 코드·검증 코드 기록도 추가했습니다. 상위 경로 검사를 PermissionError로 모의한 조건을 포함해 합성 CPU·분할·캐시·Dataset 회귀 25개가 통과했습니다. 수정 후 실제 AppContainer 재검사는 아직 남아 있으며 이 기록을 native 통과로 해석하지 않습니다.

`src/agents/wm_cpu_probe.py`는 Windows/Python 3.12에서 고정 합성 검사만 실행하는 별도 CLI입니다. 기존 `AppContainer`·ACL·Job·PyTorch 런타임 허용 목록/바이트 복사 검증기를 재사용합니다. 실데이터 입력 경로·pickle·GPU·모델·일반 작업 선택 인자를 받지 않습니다. WM 런타임의 3.12 지원 정책이나 기존 작업 승인은 변경하지 않습니다.

고정 payload는 9개 라벨 범주별 6개, 총 54개의 독립 Lot·맵을 생성합니다. 서로 다른 크기와 숫자 패턴을 사용하되 실제 결함 클래스의 형태를 재현한 데이터로 주장하지 않습니다. PyTorch CPU 텐서 `[0,2,4,6,8,10]`와 NumPy 변환을 확인한 뒤 기존 canonical 검증·EDA·그룹 분할·격리 변환 검사·격리 캐시·Dataset 코어를 호출합니다. 검사 크기는 32×32, 배치는 4개, DataLoader는 기존 단일 CPU 프로세스 경로입니다. 모든 개발 표본의 이미지·라벨을 직접 생성한 합성 원본과 대조하며 canonical 파일 지문이 그대로인지 검사합니다.

canonical 정합성·분할 검증은 전체 합성 맵을 읽습니다. 이후 변환·캐시·Dataset은 Train·Validation만 사용하며 Test 캐시를 만들지 않습니다. 이 구분을 전체 과정의 `test_maps_read: false`로 과장하지 않습니다. 합성 입력만을 위한 고정 검사 확인은 실데이터 시각 검토 승인이나 모델 학습 승인이 아닙니다.

분할 코어 `wafer_dl/split.py`에 선택적 `workspace`를 추가해 격리 scratch의 상속 ACL을 사용하도록 했습니다. 비어 있고 canonical·출력과 겹치지 않는 폴더만 받으며 기존 CLI는 인자를 생략해 이전 임시 폴더 동작을 유지합니다. 알고리즘·분할 목적함수는 바꾸지 않았습니다.

worker는 실제 AppContainer 토큰·네트워크 capability 부재·Job 소속이 아니면 거부합니다. parent는 복사 코드/런타임 지문과 버전·고유 probe ID·모든 필수 검사·배치 shape/dtype·구간 프로토콜·정상 종료·Job 정리·설정 한도·프로필 정리를 대조합니다. worker의 성공 문자열만으로 통과하지 않습니다. 별도 네트워크 차단을 재수집하는 도구는 아니며 `network_evidence_retested: false`로 표시합니다. `cpu_compatibility_passed`는 해당 고정 CPU 처리 경로의 호환성 결과이지 전체 격리 보안·GPU·성능·실데이터 지원 인증이 아닙니다.

최종 검증: 신규 검사 8개를 포함한 합성 CPU 코어·복사 코드 독립성·비격리 진입 거부·분할/캐시/Dataset·공통 서비스·명세·기본 probe 회귀 **64개 통과**. 이 검증에서 실제 CPU 텐서와 WM 코어는 호스트에서 실행했으며 AppContainer 실행을 대신하지 않습니다. 실제 native 검사는 관리자 PowerShell에서 아래 명령으로 수행해야 합니다.

```powershell
Set-Location D:\workspace\waferguard
conda activate dl312
python -m src.agents.wm_cpu_probe --confirm-synthetic-wm-cpu-probe --memory-mib 8192 --processes 2 --wall-seconds 300
```

기본 런타임 복사 예산은 16 GiB입니다. 설치된 CUDA 포함 PyTorch는 CPU 검사라도 DLL 의존성 때문에 복사량이 클 수 있으며, 디스크 복사 예산과 8 GiB Job RAM 상한은 다릅니다. 실제 크기를 먼저 검사하고 예산 초과는 거부합니다. 자동 재시도·패키지 설치·다운로드·감사 정책 변경을 하지 않습니다. 결과는 새 `logs/agents/wm_cpu_probes/<고유 ID>/result.json`에, 단계 진단은 `scratch/task/diagnostic.json`에 남습니다. 실패 시 부분 결과를 통과로 표시하지 않고 새 프로필 정리를 시도합니다.

## Python 3.12 격리 검사 실행 기록 — 2026-10-19

`appcontainer_probe.py`에 `--experimental-python312`를 추가했습니다. 기존 고정 합성 payload만 사용하며 외부 데이터 경로를 받지 않습니다. 기본 검사와 WM 실데이터 preflight의 지원 정책은 바꾸지 않았습니다. 검사 결과에 Python 버전·실험 여부·검사 범위를 기록하고 `wm_pipeline_verified: false`, `torch_verified: false`, `production_worker_enabled: false`를 유지합니다.

Python 3.12.15에서 실제 합성 AppContainer 검사를 요청했습니다. 결과는 **검토 필요**, `acl_setup / PROBE_INTEGRITY_SETUP_FAILED / native_error_code: 5`입니다. 출력 폴더의 낮은 무결성 수준을 설정하는 단계에서 접근이 거부됐고 worker는 시작하지 않았습니다. 현재 실행 토큰은 관리자 권한이 아님을 확인했습니다. Python 3.12 비호환으로 확정한 결과가 아닙니다. `profile_cleanup: true`, `runtime_copied: false`, `real_data_read: false`이며, 결과는 `logs/agents/appcontainer_probes/c7cd3ee881374ec792626479b0370c1f/result.json`에 남겼습니다. 시스템 감사·방화벽 정책은 변경하지 않았습니다.

독립적인 실제 Windows Job 합성 검사 7개(정상·메모리·프로세스 수·시간·취소·자식 정리·마지막 핸들 종료)는 Python 3.12에서 통과했습니다. probe·Job·진행 기록·loopback 판정·WM 모의 서비스 회귀를 합쳐 **49개 통과**했습니다. Job 검사 통과는 AppContainer 파일/통신 격리, PyTorch 격리 로딩 또는 WM 종단간 검증 통과가 아닙니다.

다음은 관리자 PowerShell에서 프로젝트 루트로 이동한 뒤 같은 합성 검사를 다시 수행합니다. 별도 `--experimental-python312` 옵션이 없는 3.12 실행은 계속 거부됩니다.

```powershell
Set-Location D:\workspace\waferguard
conda activate dl312
python -m src.agents.appcontainer_probe --confirm-isolation-probe --experimental-python312 --memory-mib 2048 --processes 2 --wall-seconds 60
```

이 명령도 통신 차단의 OS 증거가 부족하면 `review_required`를 유지합니다. 감사 정책 자동 변경이나 검토 상태 자동 해제를 하지 않습니다. 기본 격리 검사 이후 PyTorch·WM 합성 처리·UI 실제 실행의 추가 검증이 남아 있으므로 WM의 Python 3.12 지원을 활성화하지 않았습니다.

## 공통 UI 연결과 생성 설정 — 2026-10-19

기존 5단계 UI의 사이드바에서 `작업 시작 방식 → 등록 모듈`을 선택합니다(2026-10-22 메뉴 정리 이후: 사이드바 `고급 도구 표시`를 켠 뒤 `작업 도구 → 등록 모듈`). WM 전용 화면이나 메뉴를 추가하지 않고 `ui/workflow_view.py`의 공통 표시기에서 등록된 서비스·생성 필드 자료형·결과 종류를 사용합니다. 기존 `LLM 제안` 경로의 CSV UI는 유지합니다. WM을 LLM이 제안하거나 자동 실행하는 연결은 이번 범위가 아닙니다.

1. **요청·대화**: 공통 등록 목록에서 모듈을 선택합니다. 기존 작업은 모듈 선택 후 이름으로 명시적으로 연결할 수 있습니다.
2. **제안 검토**: 서비스 지원 범위·런타임 제약을 확인하고 검토 완료를 누릅니다.
3. **입력 선택**: 계획·배정 JSON·단계 ID·새 작업 이름·입출력 예산·자원 상한을 지정합니다. EDA는 canonical 폴더·예시 그림 여부·그룹 상한, 분할/변환/캐시는 각 설정 JSON, Dataset은 배치·런타임 복사 예산을 받습니다. 생성 범위 검사 후 별도 확인으로 작업만 준비합니다.
4. **실행 승인**: 명시적 상태 확인 → 현재 단계 한 번 진행 → 승인 범위 검사 → 이번 단계 승인 → 별도 실행 확인 순서입니다. 분할 확정·캐시 등 기존 여러 승인 단계는 그대로 유지합니다. 후속 단계를 자동 연쇄 실행하지 않습니다.
5. **결과 확인**: 버튼을 누른 경우만 도메인 결과 계약을 검사하고 표·그림·텍스트를 표시합니다. 표시 복사본은 파일당 8 MiB, 합계 32 MiB, 참조 128개로 제한합니다. 결과를 읽기 전후에 파일 지문·서비스 결과를 재확인하며 캐시 폴더 전체를 브라우저로 보내지 않습니다.

`agents/wm_creation_schema.py`는 정상 개발 6단계의 **작업 생성** 설정 명세와 인자 변환기입니다. `WmWorkflowService.creation_schema()`와 `preview_form()`을 통해 조회·검증합니다. UI는 임의 import나 Python 코드를 받지 않습니다. 모듈 계획의 `settings`는 여전히 `{}`이며 그 `form_ready: false`와 별개로, 생성 지원 6개 서비스의 `capabilities().creation_form_ready`는 이제 `true`입니다. 충돌 후보·충돌 제외 적용 2개는 기존 작업 연결·조회·승인·실행만 지원하고 새 생성은 기존 CLI를 유지합니다.

계획과 배정은 동일 서명이어야 합니다. 최초 단계도 해당 계획의 `runs: {}`인 유효한 배정 JSON이 필요하며 UI가 임의로 만들거나 기존 배정을 덮어쓰지 않습니다. 생성 성공 시 표시하는 **새 bindings 경로**를 다음 모듈의 생성 입력에 사용합니다. 계획/배정 자동 편집기와 후속 단계 자동 준비는 미구현입니다.

`ui/workflow_service.py`는 세션에 마지막 표시 상태·불변 생성/승인 미리보기·일회 승인 ID를 보관합니다. 렌더링·메뉴 이동만으로 `status` 정밀 검사나 worker·모델 호출을 하지 않습니다. 상태·결과는 사용자가 명시적으로 검사하며 이후 자동 감시하지 않습니다. 실행 예외 후에는 승인 ID를 자동 재사용하지 않고 현재 상태 재확인이 필요합니다. 이전 상태를 재확인한 경우에도 최종 입력·코드·revision·승인 검증은 기존 서비스/실행기가 담당합니다.

Windows/Python 3.10 native 정책은 유지합니다. **Python 3.12에서 화면을 확인할 수 있다는 사실은 WM 격리 실행을 지원한다는 뜻이 아닙니다.** 지원되지 않은 환경에서는 UI 실행 버튼을 비활성화하고 기존 preflight도 차단합니다. 새 생성 스키마를 관리 코드 지문에 추가했으므로 변경 전 소스 해시로 저장된 작업/승인을 그대로 재실행하지 않습니다. 실제 데이터·AppContainer·LLM은 이번 검증에서 실행하지 않았습니다.

최종 검증: Python 3.12에서 신규 생성 명세·UI 세션·결과 표시·Streamlit AppTest 13개를 포함해 CSV UI·WM 공통 서비스·명세 회귀 **116개 통과**. 모의 승인/실행 분리, 재렌더링 무호출, 변경된 생성 입력·중복 생성 거부, 실행 예외 후 자동 재시도 거부, 품질 검토 중단, 결과 변경·크기 상한을 검사했습니다. 실제 3.12 native 지원을 증명하는 검사가 아닙니다.

## WM 처리의 공통 서비스 정비 — 2026-10-19 후속

WM 8개 CLI의 반복 승인 처리를 `src/agents/workflow_approval.py`로 모았습니다. 각 CLI의 `approval_summary`는 기존 작업별 영향·설정·검토 자료 설명을 제공하고, `preview_approval`·`record_approval`이 같은 공통 저장 함수를 사용합니다. 기존 `approve`는 터미널 확인 문자열만 담당합니다. UI에서 `input()`·isatty를 흉내 내거나 승인 DB에 직접 INSERT할 필요가 없습니다.

`src/agents/wm_workflow_service.py`는 등록된 WM 모듈이 공유하는 하나의 서비스입니다. **WM용 화면을 추가하지 않았으며 기존 계산·worker·도메인 결과 검증기는 유지했습니다.** 기존 8개 CLI 명령과 확인 문자열도 유지합니다. 서비스는 `build_agent_registry().service(module_id, version)`으로 조회합니다. CSV 계산 모듈의 서비스 전환은 아직 남아 있으므로 해당 조회는 미지원 오류를 반환합니다.

| 공통 서비스 메서드 | 재사용 범위 |
| --- | --- |
| `capabilities()` | 지원 작업·생성 가능 여부·런타임 제약 조회. 실행하지 않음 |
| `preview_creation(...)` / `create(...)` | 기존 `wm_run_creation`의 정상 개발 6단계 미리보기·생성·배정 검증에 위임 |
| `status(name)` | 기존 상태/결과 검사와 현재 범위의 미사용 승인·작업 이력을 대조해 `actions` 반환 |
| `preview_approval(name)` / `record_approval(name, preview, confirmed=True)` | 도메인별 승인 설명과 공통 잠금·상태·입력·코드 재검사·저장 재사용 |
| `next(name, expected_stage=..., expected_revision=..., approval_id=...)` | 기존 실행 잠금 안에서도 사용자가 본 단계·revision을 재검사해 한 단계만 실행 |
| `results(name)` | 기존 부모 검증·완료 커밋을 확인한 로컬 결과를 표·이미지·보고서·묶음 참조로 정규화 |

명세의 WM 항목은 `execution.connection_status: service_registered`로 변경했습니다. 이는 백엔드 서비스 등록이지 런타임 호환성 승인이 아닙니다. UI 연결과 생성 폼의 최신 범위는 위 절을 참고하세요. 충돌 후보·충돌 제외 적용 2개는 상태·승인·한 단계 실행·결과 조회를 공유하지만, 생성은 기존 전용 CLI를 유지합니다. 기능 조회에서 이를 `creation_supported: false`로 명시합니다.

```python
from src.agents.agent_module_registry import build_agent_registry

registry = build_agent_registry()
service = registry.service("wm_canonical_eda", "1.0.0")
capabilities = service.capabilities()  # 작업 생성·승인·worker 실행 없음

# 아래는 기존 작업 이름을 사용자가 선택한 뒤 명시적으로 호출할 정밀 조회 예시다.
# current = service.status("사용자가_선택한_작업")
# preview = service.preview_approval("사용자가_선택한_작업")
# 사용자에게 preview.to_dict()["summary"]를 보여 주고 확인받은 뒤에만:
# approval = service.record_approval("사용자가_선택한_작업", preview, confirmed=True)
```

상태·결과 조회는 기존 입력·코드·선행 증거의 정밀 검사를 포함하며 큰 canonical 해시를 다시 읽을 수 있습니다. 기존 실행 관리자 생성/SQLite 초기화 경로도 재사용하므로 파일/DB를 전혀 건드리지 않는 가벼운 렌더링 함수로 해석하지 않습니다. 공통 UI 연결 시 명시적 검사 버튼에서 호출하고, 현재 CSV UI 렌더링 경로에 자동 추가하지 않았습니다. 서비스가 반환하는 행동 목록은 안내이며 실행 권한을 대신하지 않습니다.

승인 저장 전 동일 범위를 잠금 안에서 다시 확인합니다. 유효한 미사용 승인이나 예약/실행 이력이 있으면 중복 발급하지 않습니다. 검토 필요·취소·실패·보안 차단 상태는 다음 행동을 주지 않으며, 원래 검토 단계들을 하나의 승인으로 축약하지 않습니다. 결과 참조는 검사 시점의 로컬 경로이며 표시 시에도 경로·파일 검증을 유지해야 합니다. LLM·LangSmith로 자동 전송하지 않습니다.

런타임 판단은 `src/agents/wm_runtime.py`에서 공통 관리합니다. **현재 native 지원은 Windows/Python 3.10뿐**입니다. Python 3.12에서는 서비스 명세 조회·모의 코어 테스트와 실제 격리 지원을 구분하며, 8개 CLI의 preflight는 지원되지 않은 환경을 계속 차단합니다. 버전 조건을 삭제하거나 비격리 실행으로 우회하지 않았습니다.

공통 승인·런타임·서비스 소스를 `wm_canonical_transport.CONTROL_FILES`에 포함했고 후속 WM 운송 계층이 이를 상속합니다. worker에 이 파일들의 읽기 권한을 추가한 것은 아닙니다. 변경 전 코드 해시로 저장된 실행 선택/승인은 새 코드에서 재사용할 수 없습니다. 기존 로그·산출물·소비 승인은 그대로 보존하고, 테스트는 새 합성 작업으로 수행합니다.

검증 기록: 새 WM 서비스 검사 14개를 포함해 Python 3.12의 서비스·명세·UI 회귀 **38개가 통과**했습니다. 이 중 WM 처리 코어는 모의 런타임·모의 격리 운송으로 검사했고 실제 3.12 native 실행은 계속 차단되는 것을 확인했습니다. Python 3.10의 WM 기존 8단계·생성·배정·실행·명세 회귀 106개에서는 105개 통과, 대용량 충돌 제외 1개가 검사 도중 소스 변경에 따른 해시 불일치로 실패했습니다. 최종 소스를 고정한 뒤 해당 1개를 단독 재실행해 통과했습니다. 실제 데이터·Ollama·AppContainer 실행은 하지 않았습니다. 공통 UI 자동 연결·CSV 서비스 전환·WM 생성 폼 스키마·3.12 native 검증은 후속 범위입니다.

## 통합 모듈 설명 명세 정비 — 2026-10-19

공통 UI를 유지하기 위한 구조 정비의 첫 단계입니다. 기존 `ModuleContract`·계산 등록부·WM 처리/승인/worker는 보존하고, **별도 설명 파일과 기존 계약을 결합한 조회 명세**를 추가했습니다. 새 UI나 WM 전용 실행기를 추가한 작업이 아닙니다.

- 설명 파일: `configs/agents/modules/catalogue.json` — 표형 8개·WM 8개의 ID·버전·한글 이름·설명·설정 스키마 ID·실행 프로파일·결과 표시 유형·노출 정책.
- 결합·검사: `src/agents/module_specification.py` — 등록된 구현과 설명 목록을 대조하고 데이터 전용 명세를 생성.
- 기존 등록부: `AgentModuleRegistry.specifications()`·`specification()`·`catalogue()` — 봉인 후 명세 조회. 기존 `contracts()`·`resolve()`는 호환 유지.
- 공통 조회 CLI: `src/agents/module_specification_cli.py` — 목록·상세를 조회할 뿐 작업 준비·실행·승인은 하지 않음. 기존 사용자 참조 모듈의 `module_catalogue_cli list/check`는 그대로 보존.

### 공통 명세의 구성과 기준 정보

| 정보 | 기준 정보와 조회 방식 |
| --- | --- |
| ID·버전·입출력 포트·선행 조건·원본 보존·네트워크·승인·자원 정책 | 기존 `ModuleContract`를 그대로 결합. JSON에 중복 복사하지 않음 |
| 이름·기능 설명·공통 결과 유형·노출 정책 | `catalogue.json`에서 조회 |
| 표형 설정의 허용 필드·기본값 | 등록 검증기의 기존 `QualityRules`·`PreparationRules` 객체에서 생성 |
| 설정 자료형·범위·선택지 | 표시용 스키마에 제공. 조합 제약·최종 유효성은 기존 검증기로 다시 검사 |
| 계산과 다단계 승인 작업의 구분 | `execution.kind`의 `calculation`·`workflow` |
| WM 실제 처리·검토 단계 | 기존 `WmWorkflowModule.worker_stage`·`review_stages`에서 상속 |
| 서비스·격리 코드 목록 연결 | `service_id`·`runtime_manifest_id`라는 논리 참조만 선언. 임의 import 경로를 받지 않음 |

결과 표시 유형은 `table`·`chart`·`image`·`report`·`artifact_list`입니다. 모듈별 화면 이름이나 UI Python 함수를 명세에 넣지 않습니다. 이 목록은 사용할 공통 표현 종류이며 실제 결과 파일의 존재·정합성 검증을 대신하지 않습니다.

WM의 계획 `settings`는 여전히 빈 객체입니다. 실제 분할·변환·캐시·자원 설정은 기존 작업 생성 계약을 사용하므로 `settings.scope: workflow_creation`, `form_ready: false`로 표시합니다. 작업 생성 설정 폼은 다음 공통 서비스 분리 단계에서 명세화합니다. 아직 없는 설정·실행 기능을 준비 완료로 표시하지 않습니다. 표형 `form_ready: true`도 표시용 필드 스키마의 준비 상태이지 UI 자동 폼이 이미 연결되었다는 뜻은 아닙니다.

### 조회 방법

```powershell
python -m src.agents.module_specification_cli list
python -m src.agents.module_specification_cli list --audience llm
python -m src.agents.module_specification_cli show --module wm_cache_prepare
python -m src.agents.module_specification_cli show --module tabular_quality
```

`list --details`로 전체 명세를 조회할 수 있습니다. 출력의 `declaration_only: true`·`worker_started: false`·`approval_issued: false`는 명세 검사와 실제 실행을 구분합니다.

```python
from src.agents.agent_module_registry import build_agent_registry

registry = build_agent_registry()
modules = registry.specifications(audience="ui")
cache = registry.specification("wm_cache_prepare", "1.0.0")
catalogue_signature = registry.catalogue().signature
```

반환 데이터는 복사본이며 변경해도 등록 정보가 바뀌지 않습니다. 등록부 객체별 최초 조회 시 파일을 읽고 이후에는 검증한 불변 명세를 사용합니다. 명세 파일을 바꾸면 새 등록부를 생성해야 합니다. signature는 변경 식별값이며 인증 서명·실행 승인으로 사용하지 않습니다.

### 등록 검증과 남은 연결

구현만 등록하고 설명 명세를 빠뜨린 경우, 중복 ID·잘못된 버전·미등록 구현·알 수 없는 필드·잘못된 설정 스키마·계산/WM 실행 방식 혼용은 거부합니다. JSON 중복 키·비표준 숫자·링크·과대 파일·설명 제어 문자와 임의 HTML 표시 유형도 거부합니다. 등록 조회는 원본 데이터·승인 DB를 읽거나 함수 실행·모델 호출·worker 시작을 하지 않습니다.

`ui_visible`은 공통 목록 표시 후보이고 실행 허가가 아닙니다. `llm_proposable`은 향후 목록 공급용 정책입니다. 현재 값은 기존 허용 3개(품질·결측·상관)와 맞췄으며 WM 제안을 자동 허용하지 않습니다. 실제 LLM 전달은 공개 계약·설명만 별도로 선별해야 하며 명세 전체를 자동 전송하지 않습니다.

**이번 단계에서는 기존 UI·LLM·CSV/WM 실행 목록을 새 명세로 전환하지 않았습니다.** `execution.connection_status`는 모두 `declaration_only`입니다. 다음 단계는 공통 서비스 인터페이스를 분리하고 기존 실행기 연결을 검증한 후, UI와 LLM의 분산 목록이 이 명세를 참조하게 바꾸는 것입니다. 기존 실행 허용 목록·승인 코드 지문은 아직 유지하며 새 명세가 검증을 우회하지 않습니다. 명세 파일 하나만 편집해 임의 모듈이 실행되는 구조가 아닙니다.

사용자 참조 모듈 `user_missing_row_histogram`은 별도 검토·합성 실행 계약을 유지합니다. 현재 통합 실행 등록부의 16개에 자동 편입하지 않습니다. 추가 모듈도 구현·검증기·보안 검토를 먼저 등록하고 이 설명 명세를 함께 제공해야 합니다.

검증: 신규 명세 검사와 기존 사용자 모듈 CLI 호환성 14개, 기존 등록부·WM 계획·표형 준비·메타데이터·모듈 파이프라인·LLM 계약·UI 회귀를 합친 **129개 모두 통과**했습니다. 실제 데이터·모델·AppContainer 실행은 하지 않았습니다. 기존 승인 경계·실행 결과를 바꾸는 단계와 구분합니다.

## 목적

파이프라인을 미리 검사할 때 사용자 모듈의 Python 계산 코드를 import하지 않고, 고정된 모듈 계약과 JSON 설정 스키마만 확인합니다. 모듈 계산은 별도 실행 단계에서 처리합니다.

## 현재 등록된 예시

- 모듈: `user_missing_row_histogram` 1.0.0
- 계약: 기존 `missing_row_histogram.json` 명세의 고정 해시로 확인
- 설정 스키마: `missing_row_histogram_settings.json`의 SHA-256으로 고정
- `include_complete`: 불리언이며 기본값은 `true`
- 알 수 없는 설정 키, 잘못된 자료형, 비표준 JSON 숫자, 스키마 해시 불일치는 거부

스키마는 제한된 자료형만 지원합니다. 불리언, 정수, 유한 숫자, 제한된 길이의 문자열, 고정 enum을 사용할 수 있습니다. 설정값으로 Python 함수·모듈 경로·표현식·임의 중첩 객체를 지정할 수 없습니다.

## 사용

2026-10-17: 공통 코어에서 에이전트 등록부를 import하던 역참조를 제거했습니다. `build_metadata_registry()`는 내장 모듈만 반환합니다. 사용자 참조 모듈은 에이전트 조립 계층에서 명세·소스 해시를 검사한 뒤 `reviewed_contract`로 전달합니다. 아래 코드는 이 조립 계층의 사용 예입니다.

```python
import json
from pathlib import Path

from src.preprocessing.metadata_registry import build_metadata_registry
from src.preprocessing.pipeline import validate_pipeline
from src.agents.module_catalogue import MODULE_ID, VERSION, inspect_module

config = json.loads(Path("configs/agents/pipelines/user_module_example.json").read_text(encoding="utf-8"))
contract = inspect_module(MODULE_ID, VERSION)
registry = build_metadata_registry(reviewed_contract=contract)
plan = validate_pipeline(config, registry)
```

명세·소스 해시 검사 후 파이프라인 설정만 검증합니다. 입력 데이터를 읽거나 모듈을 실행하지 않습니다. 설정에 `include_complete`가 생략되면 계획에는 기본값 `true`가 반영됩니다. 다른 ID·버전·설정 필드의 계약이나 계약이 아닌 객체를 전달하면 거부합니다. 공통 등록부에 계약을 전달하는 행위 자체는 worker 실행 권한이나 사용자 승인이 아닙니다.

## 경계와 다음 작업

등록기는 현재 참조 모듈 하나의 메타데이터를 제공합니다. 사용자 모듈을 임의로 추가하거나 실제 CSV 실행을 허용하지 않습니다. 내장 모듈은 기존의 신뢰된 구현을 통해 계획 검증에 사용됩니다.

CSV snapshot·프로파일과 결합하는 실행 계약과 내장 CSV 모듈의 격리 실행은 [형식 중립 승인 CLI](../guides/PIPELINE_APPROVAL.md)에 연결되어 있습니다. 사용자 참조 모듈의 메타데이터 조회와 실제 실행 허용 범위는 구분합니다. 부모의 등록부·명세 검사·CSV 바인더·설정 스키마 파일도 승인 코드 지문에 포함하며, 변경하면 기존 실행을 재사용하지 않습니다.

## 다음 연결 단계

검증 기록(2026-10-17): 관련 회귀 41개 통과. 전체 에이전트 검사도 403개 중 399개 통과·선택 실행 4개 생략으로 완료했습니다. 이전 공통 코어 의존성 방향 검사 실패는 해결되었습니다. 실제 WM 원본 재처리·모델 학습·LLM 전송은 하지 않았습니다.

이번 단계는 공통 코어 의존성 문제를 해결하는 선행 작업입니다. `src/agents/adapters/wm811k_metadata.py`에 기존 WM metadata 어댑터가 있지만 형식 중립 승인 CLI의 격리 worker에는 아직 연결되지 않았습니다. 후속 작업에서는 canonical 파일 목록·해시·크기를 승인 범위에 묶고, 기존 WM 검증·EDA를 재사용하는 전용 어댑터를 연결합니다. 기존 pickle을 다시 읽거나 새로운 WM 분석 알고리즘을 작성하지 않습니다.

후속 구현(2026-10-17): [WM canonical 품질·EDA](../guides/PICKLE_IMPORT.md#wm-canonical의-승인격리-품질-검사와-eda)를 별도 승인 실행 도구에 연결하고 실제 합성 정상·검토 중단을 확인했습니다. 위 문장은 그 이전의 계획 이력입니다. 현재 CSV용 공통 모듈 조합 CLI의 어댑터 목록을 늘리거나 기존 metadata 어댑터를 비격리 fallback으로 사용하지는 않습니다.

## WM 작업의 공통 등록·계획 연결

2026-10-17 추가. `src/agents/agent_module_registry.py`에서 기존 표형 계산 모듈 8개와 WM 승인 작업 8개를 함께 조회합니다. 코어 등록부와 기존 승인 대상 소스는 변경하지 않았습니다. 코어가 에이전트를 import하지 않는 방향도 유지합니다.

WM 작업은 여러 승인 단계를 가진 실행기이므로 일반 EDA의 `calculate` 함수로 위장하지 않습니다. 같은 `ModuleContract`·`ModulePort`를 재사용하지만 에이전트 계층의 `WmWorkflowModule`로 구분합니다. 기존 CSV worker의 실행 허용 목록을 확장한 것은 아닙니다.

| 모듈 ID (버전 1.0.0) | 입력 → 출력 계약 | 기존 CLI |
| --- | --- | --- |
| `wm_canonical_eda` | canonical → 통과한 품질·EDA 증거 | `wm_canonical_cli` |
| `wm_conflict_candidate` | 품질 검토 증거 → 충돌 제외 후보 | `wm_conflict_cli` |
| `wm_conflict_apply` | 제외 후보 → 새 canonical | `wm_exclusion_cli` |
| `wm_split_propose` | 통과한 품질·EDA 증거 → 분할 제안 | `wm_split_cli` |
| `wm_split_commit` | 분할 제안 → 확정 분할 | `wm_split_commit_cli` |
| `wm_transform_check` | 확정 분할 → 변환 검사 증거 | `wm_transform_cli` |
| `wm_cache_prepare` | 변환 검사 증거 → 개발 구간 캐시 | `wm_cache_cli` |
| `wm_dataset_check` | 개발 구간 캐시 → Dataset 검사 보고서 | `wm_dataset_cli` |

### 정적 계획 확인

```powershell
python -m src.agents.wm_module_cli list

python -m src.agents.wm_module_cli validate `
  --pipeline configs/agents/pipelines/wm_development_example.json
```

이 명령은 설정만 읽고 `plan_validated`, `plan_only: true`, `worker_started: false`를 반환합니다. 작업 생성·데이터 처리·승인 소비·모델 학습은 하지 않습니다. 예제는 `canonical → EDA → 분할 제안 → 분할 확정 → 변환 검사 → 캐시 → Dataset 검사`의 정상 경로입니다. 실제 데이터를 다시 실행하라는 명령이 아닙니다.

계획의 `inputs.evidence`는 `source.evidence` 또는 앞 단계의 `<step_id>.result`를 참조합니다. 입력·출력 계약 v1, 정확한 모듈 버전, 구간, 선행 단계, 미등록 설정을 검사합니다. Test 맵, 표형 모듈 혼합, 미래 참조, 단계 건너뛰기, 승인 생략 필드는 거부합니다. 현재 `settings`는 빈 객체만 지원하며, 실제 변환·분할·자원 설정은 기존 작업 생성 CLI에서 고정하고 그 증거로 검증합니다.

품질 검토 중단을 정상 통과로 연결하지 않습니다. 충돌 처리는 `wm_quality_review → 후보 → 승인 적용 → 새 canonical → 품질·EDA 재검사`의 별도 계획으로 구성합니다. 자동 분기·자동 승인·검토 해제는 하지 않습니다.

### 기존 실행기 연결과 남은 범위

`wm_module_catalogue.open_wm_execution(module_id, version, run_name)`은 고정 ID로 기존 CLI를 선택하고 `open_run`·입력 계약 검사를 거쳐 기존 도구와 실행 관리자를 반환합니다. 작업 종류와 필수 승인 선언도 확인합니다. 함수 자체는 승인 생성·`run_next`·worker 시작을 호출하지 않습니다. 실제 실행은 기존 관리자의 승인·예약·취소·결과 검증을 그대로 따릅니다.

정적 계획의 연결만으로 두 실제 실행이 같은 데이터에서 유래했다고 증명하지 않습니다. 각 실제 작업의 선행 증거 연결은 기존 실행기가 검사하며, 현재 계획의 각 단계에 작업 이름을 배정하고 전체 연결을 검증하는 통합 실행 동선은 후속 작업입니다. 계획 서명은 설정 식별값이지 소스 해시·데이터·승인에 결합된 실행 서명은 아닙니다.

이번 단계에서 구현한 파일:

- `src/agents/agent_module_registry.py`: 에이전트 계층 통합 조회·봉인.
- `src/agents/wm_module_catalogue.py`: WM 계약·고정 실행 경로·기존 실행기 연결.
- `src/agents/wm_module_plan.py`: 증거 포트·순서·버전·승인 경계의 정적 계획.
- `src/agents/wm_module_cli.py`: 목록·설정 검증 CLI.
- `configs/agents/pipelines/wm_development_example.json`: 정상 개발 경로 예제.
- `tests/agents/test_wm_module_plan.py`: 등록·계약·거부 사례·실행 경로 검사.

검증: 새 테스트 10개를 포함해 기존 등록부·계획·메타데이터·표형 변환 관련 테스트 총 38개가 통과했습니다. 새 테스트 10개를 다시 포함한 WM Dataset·캐시 회귀 25개도 통과해 중복을 제외한 검사 수는 53개입니다. 실제 데이터 재실행·실제 승인 소비·새 네이티브 worker 실행은 하지 않았습니다. 합성 테스트의 승인 저장소와 모의 운송만 사용했습니다. 전체 에이전트 테스트 전체집합은 이번 단계에서 실행하지 않았습니다.

## 실제 WM 작업 배정과 읽기 전용 연결 조회

2026-10-17 후속 구현. 정적 계획의 각 단계에 기존 작업 이름을 배정하고, 저장된 실제 선행 증거와 대조하는 기능입니다. 정상 개발 6단계만 지원하며 충돌 후보·제외 분기는 기존 CLI로 별도 확인합니다. 계획에 충돌 분기가 포함되면 이 조회기는 명시적으로 거부합니다.

소스는 `wm_run_evidence.py`(기존 완료 증거 조회), `wm_run_bindings.py`(단계 배정·연결·중단), `wm_run_binding_cli.py`(조회 CLI)입니다. 기존 WM worker·승인 대상 소스는 변경하지 않았습니다.

### 작업 배정 파일

`configs/agents/pipelines/wm_completed_runs_example.json`에 현재 완료된 작업 6개를 배정했습니다. `pipeline_signature`는 앞 단계의 계획 검증 결과와 일치해야 합니다. 이 예제 서명은 해당 계획의 식별값이지 범용 고정값이 아닙니다. 계획을 의도적으로 수정했다면 먼저 정적 검증을 수행하고 배정도 검토해야 합니다.

**공유 예제의 한계:** 이 파일과 `wm_common_native_runs_example.json`, `wm_common_review_runs_example.json`은 작성 환경에서 검증한 작업 이름을 보존하는 배정 형식 예제입니다. 대응하는 `logs/agents/` 실행 기록·승인 DB·산출물은 Git으로 공유하지 않으므로 새 PC에서 아래 조회가 성공한다고 보장하지 않습니다. 이 JSON이나 계획 지문만으로 완료·승인 상태가 만들어지지 않습니다. 다른 사용자는 본인의 작업을 별도 생성하고, 생성 CLI가 반환한 로컬 `bindings.json`을 사용합니다. 예제 작업 이름을 기존 승인 기록에 덮어쓰거나 누락 증거를 자동 생성하지 않습니다. 아래 당시 명령·결과는 로컬 검증 이력입니다.

```json
{
  "schema_version": "wm_run_bindings_v1",
  "pipeline_signature": "계획 검증으로 얻은 64자리 서명",
  "runs": {
    "eda": "wm-canonical-eda02",
    "proposal": "wm-group-proposal01",
    "split": "wm-group-commit01",
    "transform": "wm-transform-check01",
    "cache": "wm-development-cache02",
    "dataset": "wm-dataset-check01"
  }
}
```

미배정 단계를 `runs`에서 생략하면 첫 미배정 단계에서 멈추고 기존 생성 CLI를 안내합니다. 알 수 없는 단계·임의 경로·버전 또는 서명 불일치·중복 JSON 키는 거부합니다. 파일에는 원본 데이터·집계·이미지·API 키를 넣지 않습니다.

### 확인 명령과 결과

```powershell
python -m src.agents.wm_run_binding_cli `
  --pipeline configs/agents/pipelines/wm_development_example.json `
  --bindings configs/agents/pipelines/wm_completed_runs_example.json
```

- `bindings_verified`: 배정한 모든 작업의 완료 증거와 부모 연결을 확인함. `next_action`은 `null`.
- `attention_required`: 첫 미배정·미완료·불일치에서 중단. 뒤 단계는 `not_checked`이며 자동 실행하지 않음.
- `verification_failed`: 조회 중 기록·결과 변경을 탐지함. 이전 확인을 전체 완료로 사용하지 않음.
- 정상 전체 검증일 때만 종료 코드 0. 그 외에는 종료 코드 2.

이 명령은 승인 DB를 `mode=ro`로 열고 승인 소비·예약·커밋을 대조합니다. `ApprovalStore` 초기화·`executor` 생성·`run_next`·승인 발급은 호출하지 않습니다. 기존 계약·도구 생성자로 코드/계획 식별값과 입력 지문을 재구성하고, 기존 부모 결과 검증 함수로 요청·OS 종료·Job 정리·응답·영수증을 확인합니다. 분할 검토와 변환 그림 확인의 기존 승인 증거도 확인합니다.

자식에 저장된 부모 증거를 현재 부모 결과로 다시 구성해 비교합니다. 단순히 작업 이름·행 수가 같다는 이유로 연결하지 않습니다. 실제 선행 작업이 계획에서 선택한 부모와 달라도 거부합니다. 한 번의 조회 안에서만 결과를 재사용하고, 마지막에 입력과 실행 기록·결과의 변경을 다시 확인합니다.

대용량 입력·캐시의 해시와 행 연결을 확인하므로 시간이 걸릴 수 있습니다. 캐시 검증에는 기존 검증기의 임시 SQLite가 사용되지만 원래 승인 DB와 데이터는 변경하지 않습니다. 캐시 바이트·색인·저장 보고서를 확인하며 원본 Test 맵을 복원하거나 전처리·통계를 재산출하지 않습니다.

실제 완료 기록 확인: 위 예제의 EDA·제안·확정·변환·캐시·Dataset 6단계가 모두 `verified`, 전체 `bindings_verified`로 확인됐습니다. `worker_started: false`, `approval_consumed: false`입니다. 이는 기존 결과 연결 검증이며 새 성능 평가·모델 학습 결과가 아닙니다.

검증 기록: 새 배정 테스트 4개·기존 코어를 사용한 합성 6단계 읽기 전용 통합 테스트 1개·기존 모듈 계획 테스트 10개, 총 15개가 통과했습니다. 통합 테스트는 조회 시 승인 DB 생성·실행 API를 차단하고 모든 fixture 파일의 지문과 파일 목록이 그대로인지 확인했습니다. 결과 파일 변경과 DB 커밋 실패 상태도 거부했습니다. 전체 에이전트 테스트 전체집합과 새로운 네이티브 AppContainer 테스트는 실행하지 않았습니다.

남은 범위는 새 작업 생성과 승인 필요 단계의 안내·실행 동선 통합입니다. 이번 CLI에는 생성·승인·실행 명령을 넣지 않았습니다. 조회 결과를 새로운 승인이나 데이터 처리 성공으로 기록하지도 않습니다. 전달 묶음 범용화는 계속 후순위로 보류합니다.

## 다음 작업 안내

2026-10-18 추가. `src/agents/wm_workflow_guidance.py`와 `src/agents/wm_guidance_cli.py`는 기존 연결 검증 결과를 바탕으로 다음 행동만 안내합니다. 생성·승인 발급·실행 API를 호출하지 않으며, 정상 개발 6단계만 지원합니다. 출력은 로컬 전용이며 LLM 전송용 DTO가 아닙니다.

```powershell
python -m src.agents.wm_guidance_cli `
  --pipeline configs/agents/pipelines/wm_development_example.json `
  --bindings configs/agents/pipelines/wm_completed_runs_example.json
```

| 안내 action | 의미 |
| --- | --- |
| `finish` | 모든 단계의 기존 완료·부모 연결을 확인함. 재실행 명령 없음 |
| `select_or_create_run` | 작업을 선택·생성한 뒤 배정 필요. 고정된 생성 CLI의 도움말만 안내 |
| `continue_local_stage` | 승인 불필요한 현재 로컬 단계의 `next` 호출 안내 |
| `request_approval` | 기존 CLI의 `next`로 승인 대기 전환 요청. 사용자 승인은 별도 |
| `approve_stage` | 현재 범위의 유효한 미사용 승인이 없어 사용자 `approve` 필요 |
| `execute_approved_stage` | 현재 범위의 유효한 미사용 승인이 있음. 사용자가 해당 승인 ID를 직접 넣어 `next` 호출 |
| `wait_or_review_task` | 예약·실행·종료 이력이 있음. 자동 재시도 없이 상태 확인 |
| `review_local_report` | 검토 필요·실패·blocker가 있어 로컬 보고서 확인 필요 |
| `stop_cancelled` / `security_review` | 취소·보안 잠금. 재개·우회 명령 없음 |
| `review_local_evidence` / `repeat_verification` | 코드·입력·부모·완료 증거 불일치 또는 조회 중 변경. 재검증 전 진행 금지 |

출력은 `status`, `guidance`, 단계별 확인 상태, `advisory_only: true`, `local_only: true`, `worker_started: false`, `approval_consumed: false`로 구성합니다. `guidance`에는 고정 CLI·하위 명령·검증한 작업 이름만 넣고 셸을 실행하지 않습니다. 승인 ID 자체는 출력하지 않으며 승인 ID가 필요한지만 표시합니다.

미완료 작업도 기존 선택 기록·코드/계획·입력·부모 연결을 확인합니다. 현재 단계 앞의 커밋·완료 표식과 해당 검토 승인을 확인하고, 현재 범위에 해당하는 pending 승인만 읽기 전용으로 검사합니다. 만료·취소·소비·다른 범위의 승인은 사용 가능으로 안내하지 않습니다. 만료된 pending 승인의 DB 상태를 조회 과정에서 변경하지도 않습니다.

이미 예약되었거나 실행 결과가 불명확한 경우에는 `next` 재시도를 제안하지 않습니다. 취소·보안 잠금·검토 중단·실패를 일반 실행 가능 상태로 바꾸지 않습니다. 안내 후 다른 프로세스가 상태를 변경할 수 있으므로, 이 안내는 실행 허가·잠금 획득·성공 보증이 아닙니다. 실제 명령 수행 시 기존 실행기가 상태·잠금·승인·입력을 다시 검증합니다.

정상 안내(`completed`·`needs_user_action`)는 종료 코드 0, 중단·증거 오류(`blocked`)는 종료 코드 2입니다. `completed`는 현재 배정한 기존 작업이 끝났다는 뜻이지 LLM 연결을 포함한 프로젝트 전체 완성을 뜻하지 않습니다.

검증: 새 안내 테스트 7개와 기존 모듈 계획·완료 연결 테스트 15개, 총 22개가 통과했습니다. 합성 작업의 준비→승인 요청→승인→만료→예약 상태를 구분하고, 조회 중 파일 지문·파일 목록이 바뀌지 않으며 생성·승인·실행 API가 호출되지 않는지 확인했습니다. 전체 에이전트 테스트 전체집합이나 새로운 네이티브 worker 검사는 실행하지 않았습니다.

실제 완료 작업 6개도 위 명령으로 읽기 전용 조회했습니다. 모두 `verified`이며 전체 `status: completed`, `guidance.action: finish`, `worker_started: false`, `approval_consumed: false`를 확인했습니다. 완료된 전처리를 다시 수행하지 않았습니다.

이 단계의 남은 범위는 사용자가 명시적으로 요청했을 때 기존 생성·승인·실행 함수를 통합 CLI에서 연결하는 작업입니다. 현재 안내 명령에 해당 권한을 추가한 것은 아닙니다. 전달 범용화·LLM·충돌 제외 분기 통합은 이번 변경 범위가 아닙니다.

### 추후 개선 검토: 조회 응답 시간

기록일: 2026-10-18. 사용자 요청으로 후순위 검토 항목에 보존합니다. 구현·기본 동작·검증 강도는 이번 기록으로 변경하지 않습니다.

현재 안내 명령은 연결 검증을 먼저 수행하므로 대용량 canonical·캐시의 해시, 완료 파일, 부모·자식 증거를 반복 확인합니다. 같은 입력을 공유하는 단계의 중복 읽기도 지연 요인으로 검토할 필요가 있습니다. 기능별 소요 시간과 읽은 바이트를 계측하지 않았으므로 병목 비중이나 개선 효과를 확정하지 않습니다.

검토 방향은 **빠른 참고 조회와 정밀 검증 분리**입니다.

| 구분 | 검토할 동작 | 보안·표현 경계 |
| --- | --- | --- |
| 빠른 상태·행동 안내 | 상태·배정·승인 메타데이터 중심 조회 | 전체 파일 무결성을 재검증하지 않은 참고 상태임을 표시. 실행 허가로 사용하지 않음 |
| 정밀 검증 | 파일 해시·산출물·선행 결과 연결 확인 | 명시적으로 선택한 전체 검증으로 유지 |
| 실제 승인·실행 직전 | 기존 실행기의 필수 검증 | 빠른 조회 결과로 검증을 생략하거나 승인을 대신하지 않음 |

재개 시 확인할 항목:

- [ ] 단계별 소요 시간·읽은 바이트·중복 검증 횟수를 로컬에서 측정한다. 상세 경로·통계는 LLM·외부 추적으로 보내지 않는다.
- [ ] 빠른 조회와 정밀 검증의 명령·기본 모드·출력 상태를 구분한다. 참고 상태에 `verified`·전체 완료 검증 등의 표현을 재사용하지 않는다.
- [ ] 빠른 안내는 실제 실행 가능성을 확정하지 않는다. 생성·승인·실행 연결 시 정확한 입력·코드·부모 증거를 다시 확인한다.
- [ ] 같은 조회 안에서 동일 입력을 읽는 중복을 줄일 수 있는지 검토한다. 변경 탐지와 마지막 재확인 범위를 함께 정의한다.
- [ ] 검증 결과를 조회 간 캐시할 경우 만료·무효화·정책/코드/입력 변경 기준을 먼저 정한다. 크기·수정 시각만 같다는 이유로 해시 검증 통과를 보증하지 않는다.
- [ ] 만료 승인·취소·예약·보안 잠금·조회 중 변경이 빠른 모드에서도 실행 안내로 잘못 승격되지 않는지 검사한다.
- [ ] 대표 크기의 합성/로컬 입력으로 개선 전후 응답 시간을 비교한다. 현재 완료된 WM 작업을 성능 측정 목적으로 재처리하지 않는다.

현재 CLI에는 빠른 모드나 영속 검증 캐시를 추가하지 않았습니다. 이 항목은 기능 구현의 선행 조건이 아니라 추후 성능·사용성 개선 검토입니다.

## 분할 제안 작업 생성·배정

2026-10-18 추가. `src/agents/wm_run_creation.py`와 `src/agents/wm_creation_cli.py`는 사용자 확인 후 기존 생성 함수를 호출하고 **새 작업 배정 파일**을 생성합니다. 초기에는 완료 EDA를 부모로 가진 첫 미배정 `wm_split_propose`만 지원했으며, 아래 확장으로 완료 제안을 부모로 가진 `wm_split_commit`도 지원합니다. 나머지 단계 생성, 충돌 분기, 자동 승인·worker 실행은 아직 연결하지 않았습니다.

### 생성 전 미리보기

```powershell
python -m src.agents.wm_creation_cli preview `
  --pipeline configs/agents/pipelines/wm_development_example.json `
  --bindings configs/agents/pipelines/wm_proposal_creation_example.json `
  --step proposal --name wm-module-proposal01 `
  --split-config configs/wm811k/split.json `
  --memory-mib 2048 --processes 2 --wall-seconds 300 `
  --max-output-bytes 67108864
```

`wm_proposal_creation_example.json`은 완료 EDA만 배정한 별도 예제입니다. 기존 6단계 완료 배정은 수정하지 않았습니다. 현재 실제 WM 전처리는 이미 끝났으므로 이 예제는 새 작업 생성 기능을 사용하려는 경우에만 선택하며 완료 결과를 재생성할 필요는 없습니다.

`preview`는 계획·배정·설정과 부모 증거를 확인하고 내용·영향만 표시합니다. 생성 함수·작업 폴더·승인 기록을 만들지 않습니다. 출력은 로컬 전용이며 경로·설정·출처 식별값을 LLM이나 외부 추적으로 보내지 않습니다.

### 사용자 확인 후 생성

위 명령의 `preview`만 `create`로 바꾸면 같은 미리보기 뒤에 `CREATE-…` 확인 문구를 직접 입력해야 합니다. 비대화형 입력이나 불일치는 거부합니다. 이 확인은 **작업 준비와 새 배정 기록에 대한 확인**이며, 이후 분할 제안 worker의 실행 승인과 다릅니다.

확인 후 입력·계획·코드·부모·설정 범위를 다시 검증합니다. 고정 생성 이름 공간의 잠금으로 통합 생성 명령을 직렬화하고, 기존 작업 이름·배정된 단계·후속 배정이 있는 계획은 덮어쓰지 않습니다. 제안의 출력 요청은 1 byte~1 GiB(기본 64 MiB), 확정은 1 byte~2 GiB(기본 2 GiB), 시간·메모리·프로세스 수는 기존 `WorkerLimits` 상한을 적용합니다.

생성 시 확인한 분할 설정의 복사본을 보존하고 기존 생성 함수에 전달합니다. 새 작업이 `created/preflight`이며 부모·설정·자원이 일치하고 승인·예약·실행 기록이 없는지 확인한 뒤 새 배정을 기록합니다. 데이터의 분할 제안·확정이나 이미지 변환은 아직 수행하지 않습니다.

```text
logs/agents/wm_creation/runs/<새 작업 이름>/
  creation_started.json   # 확인한 범위
  split_config.json       # 확인한 설정 복사본
  run_created.json        # 기존 생성 함수의 작업 준비 완료 기록
  bindings.json           # 기존 배정에 새 작업 하나를 추가한 별도 파일
  CREATION_COMPLETE.json  # 새 배정 검증 후에만 생성하는 완료 기록
```

정상 결과는 `created_and_assigned`, `worker_started: false`, `execution_approval_issued: false`입니다. 다음 조회에는 결과에 표시된 **새 bindings 경로**를 사용합니다. 기존 입력 배정 파일·완료된 WM 실행·원본 데이터는 보존합니다.

중간 실패 시 새 작업과 부분 생성 기록을 자동 삭제하지 않습니다. 특히 `run_created.json` 이후 실패하면 작업은 이미 존재하지만 배정은 완료되지 않았을 수 있습니다. `CREATION_COMPLETE.json`이 없는 폴더를 통합 생성 완료로 사용하지 말고, 원인과 기존 작업을 검토하세요. 같은 이름으로 자동 재시도·재생성하지 않습니다. 개별 파일의 존재만으로 전체 생성을 완료 처리하지 않습니다.

검증 기록: 새 생성 테스트 4개가 통과했습니다. 합성 완료 EDA에서 미리보기의 무변경, 확인 필요, 생성·새 배정 성공, 실행/승인 발급 금지, 입력 변경 거부, 기존 결과 보존과 중간 실패 후 미배정 보존을 확인했습니다. 실제 WM 작업을 새로 생성하지 않았으며 새로운 네이티브 worker 검사도 실행하지 않았습니다.

최종 관련 회귀는 기존 계획·완료 연결·안내 테스트 22개와 새 생성 테스트 4개, 총 26개가 통과했습니다. 전체 에이전트 테스트 전체집합은 이번 변경 후 실행하지 않았습니다.

### 분할 확정 준비 확장

2026-10-18 후속 구현. 첫 미배정 단계가 `wm_split_commit`이면 기존 `wm_split_commit_cli.create()`로 **확정 작업의 준비만** 생성합니다. 제안의 고정 그룹 배정·설정·산출물 지문을 부모 증거로 확인하며 새 분할 조건이나 그룹 배정을 다시 받지 않습니다. `--split-config`를 확정 단계에 지정하면 무시하지 않고 오류로 거부합니다.

```powershell
python -m src.agents.wm_creation_cli preview `
  --pipeline configs/agents/pipelines/wm_development_example.json `
  --bindings configs/agents/pipelines/wm_commit_creation_example.json `
  --step split --name wm-module-commit01 `
  --memory-mib 2048 --processes 2 --wall-seconds 300 `
  --max-output-bytes 2147483648
```

예제 배정에는 완료 EDA와 제안만 연결했습니다. 기존 6단계 완료 배정을 수정하지 않았습니다. 미리보기의 `review_required_before_execution: true`는 이후 사용자 검토가 남아 있다는 뜻입니다. 이 플래그는 기존 검토 승인을 대신하지 않습니다.

생성을 원하면 동일 명령에서 `preview`를 `create`로 바꾸고 대화형 확인 문구를 직접 입력합니다. 생성 결과는 `created/preflight`이며, 이후 `review_eda` 사용자 검토 승인과 `commit_split` 확정 실행 승인을 각각 받아야 합니다. 통합 생성 시 이 단계들을 진행하거나 승인을 발급하지 않습니다. 실제 분할 파일도 아직 생성되지 않습니다.

확정 준비에서는 별도의 `split_config.json` 복사본을 만들지 않습니다. `creation_started.json`의 부모 식별값·산출물 지문과 실제 작업에 저장된 제안 증거가 범위를 고정합니다. 이후 `bindings.json`과 `CREATION_COMPLETE.json` 생성·부분 실패 보존 규칙은 제안 단계와 같습니다. 확인 중 또는 생성 전 부모 제안이 변경되면 생성을 거부합니다.

새 미리보기의 형식은 `wm_run_creation_v2`이며 모듈 ID와 검토 필요 여부를 포함합니다. 과거 생성 기록을 덮어쓰거나 자동 재개하지 않습니다. 변경 전 미리보기 객체를 확인 근거로 그대로 재사용할 수도 없습니다.

검증: 새 합성 테스트 2개가 통과했습니다. 제안 배정 상속·새 배정 생성·원래 파일 보존·검토/실행 승인 유지와 설정 재지정·과대 출력·부모 변조·기존 배정 거부를 확인했습니다. 현재 실제 WM 작업은 재생성하지 않았습니다.

최종 관련 회귀는 기존 계획·연결·안내·제안 생성 26개와 새 확정 준비 2개, 총 28개가 통과했습니다. 변조 테스트는 파일 열거 순서 대신 실제 제안 결과 폴더의 완료 표식을 지정하도록 보완했습니다. 전체 에이전트 테스트 전체집합과 새로운 네이티브 worker 검사는 실행하지 않았습니다.

### 변환 검사 준비 확장

2026-10-18 후속 구현. 첫 미배정 단계가 `wm_transform_check`이면 기존 `wm_transform_cli.create()`로 검사 작업만 준비합니다. 완료 분할의 증거를 검증하고 변환 설정의 형식·지문을 확인합니다. 생성 확인 후 같은 범위를 재검증하고 `transform_config.json` 복사본을 기존 생성 함수에 전달합니다. 기존 승인된 worker·CLI·변환 코어는 수정하지 않았습니다.

```powershell
python -m src.agents.wm_creation_cli preview `
  --pipeline configs/agents/pipelines/wm_development_example.json `
  --bindings configs/agents/pipelines/wm_transform_creation_example.json `
  --step transform --name wm-module-transform01 `
  --transform-config configs/wm811k/transform.json `
  --memory-mib 2048 --processes 2 --wall-seconds 300 `
  --max-output-bytes 268435456
```

미리보기는 파일을 생성하지 않습니다. 새 작업 준비가 필요할 때만 `preview`를 `create`로 바꾸고 표시된 확인 문구를 직접 입력합니다. 이미 완료한 실제 WM 검사를 다시 생성할 필요는 없습니다. 출력 상한은 기본 256 MiB, 허용 범위는 1 byte~1 GiB입니다. 실제 출력 크기를 보장하는 값이 아니라 실행 시 사용할 제한입니다.

변환 검사에서는 `--split-config`를 거부하고 다른 단계에서는 `--transform-config`를 거부합니다. 설정은 기본 WM 변환 계약(크기·최근접 보간·종횡비·두 채널·증강 없음)을 따릅니다. 설정 형식의 유효성은 실제 형태 보존의 시각 검토를 대신하지 않습니다.

새 미리보기 형식은 `wm_run_creation_v3`입니다. 이전 생성 기록은 보존하며 과거 미리보기 객체를 재사용하지 않습니다. 생성 결과는 `created/preflight`이고 새 배정만 저장합니다. 실행 승인·이미지 변환·캐시 생성·Test 맵 읽기·모델 학습은 하지 않습니다. 변환 검사의 별도 실행 승인과 검사 후 캐시 단계의 시각 검토는 여전히 필요합니다. 중간 실패 보존 규칙은 앞 절과 같습니다.

검증: 새 합성 테스트 2개와 기존 계획·연결·안내·제안/확정 생성 회귀를 합친 30개가 모두 통과했습니다. 설정 복사본·새 배정·원래 파일 보존과 실행/승인 발급 금지, 설정 재지정·과대 출력·확인 후 설정 변경·부모 분할 변조 거부를 확인했습니다. 실제 WM 작업 생성·네이티브 worker 실행·전체 에이전트 테스트 전체집합 실행은 하지 않았습니다.

### 캐시 생성 준비 확장

2026-10-18 후속 구현. 첫 미배정 단계가 `wm_cache_prepare`이면 기존 `wm_cache_cli.create()`로 **캐시 작업의 준비만** 생성합니다. 완료 변환 검사와 그 부모 분할의 증거를 확인하며 기존 변환 크기·채널·분할을 새 설정으로 교체하지 않습니다. 기존 승인된 worker·CLI·전처리 코어는 수정하지 않았습니다.

```powershell
python -m src.agents.wm_creation_cli preview `
  --pipeline configs/agents/pipelines/wm_development_example.json `
  --bindings configs/agents/pipelines/wm_cache_creation_example.json `
  --step cache --name wm-module-cache01 `
  --cache-config configs/wm811k/cache.json `
  --memory-mib 2048 --processes 2 --wall-seconds 1800 `
  --max-output-bytes 8589934592
```

예제 배정은 기존 완료 변환 검사까지만 참조합니다. 미리보기 자체는 완료된 캐시를 재생성하지 않습니다.

`--cache-config`는 캐시 단계에만 허용합니다. 생략하면 `configs/wm811k/cache.json`을 사용합니다. `cache_version`·shard 최대 행 수·버퍼 바이트를 검증하고, 기존 생성 함수와 같은 기준으로 예상 출력 예산과 구간별 shard 2048개 상한을 미리 확인합니다. 예상 크기는 확정 보고서와 변환 해상도로 계산하며 맵 배열을 읽지 않습니다.

통합 생성 계층의 출력 상한은 기본 8 GiB, 허용 최대 32 GiB입니다. 이는 실제 크기나 RAM 사용량이 아니라 출력 파일 예산입니다. 예산이 예상 크기보다 작으면 생성 전에 거부하며 자동 확대·부분 생성하지 않습니다. 필요량을 확인해 사용자가 다시 설정합니다. 이미 완료한 실제 캐시는 재생성할 필요가 없습니다.

준비가 필요할 때만 `preview`를 `create`로 바꾸고 확인 문구를 직접 입력합니다. 확인 후 범위를 다시 검증하고 `cache_config.json` 복사본을 보존합니다. 원래 설정과 배정은 변경하지 않으며 `created/preflight` 작업·새 `bindings.json`·생성 완료 표식만 저장합니다. 출력 크기 예상값·설정·경로 등 미리보기는 로컬 전용이며 LLM 전송용 정보가 아닙니다.

`review_required_before_execution: true`는 시각 검토 등이 남아 있다는 뜻입니다. 이후 `review_eda` 시각 검토, `commit_split` 분할 재확인, `prepare_data` 캐시 실행은 기존 별도 승인을 유지합니다. 준비 확인만으로 해당 승인을 발급하거나 단계를 실행하지 않습니다. Test 맵 읽기·모델 학습도 하지 않습니다.

새 미리보기 형식은 `wm_run_creation_v4`이며 캐시 설정과 예상 출력 예산을 포함합니다. 과거 기록은 보존하고 이전 미리보기 객체를 재사용하지 않습니다. 부분 실패 기록 보존 규칙은 앞 절과 같습니다.

검증: 새 캐시 생성 합성 테스트 2개와 기존 계획·연결·안내·제안/확정/변환 생성 회귀를 합친 32개가 모두 통과했습니다. 설정 복사·새 배정·원본 보존·검토 및 실행 승인 유지와 출력 부족/초과·잘못된 설정·확인 후 설정 변경·부모 변환 결과 변조 거부를 확인했습니다. 실제 WM 작업 생성·네이티브 worker 실행·전체 에이전트 테스트 전체집합 실행은 하지 않았습니다.

### Dataset 검사 준비 확장

2026-10-18 후속 구현. 첫 미배정 단계가 `wm_dataset_check`이면 기존 `wm_dataset_cli.create()`로 CPU Dataset·첫 배치 **검사 작업만 준비**합니다. 완료 캐시의 출처·파일·부모 연결과 설정을 확인하고, 설치된 PyTorch 런타임의 기존 허용 목록 수집기를 재사용합니다. 기존 승인된 worker·CLI·전처리 코어는 수정하지 않습니다.

```powershell
python -m src.agents.wm_creation_cli preview `
  --pipeline configs/agents/pipelines/wm_development_example.json `
  --bindings configs/agents/pipelines/wm_dataset_creation_example.json `
  --step dataset --name wm-module-dataset01 `
  --batch-size 32 `
  --max-input-bytes 8589934592 `
  --max-runtime-bytes 8589934592 `
  --memory-mib 8192 --processes 2 --wall-seconds 300 `
  --max-output-bytes 1048576
```

예제는 이미 완료된 개발 캐시까지만 배정한 별도 파일입니다. 현재 완료한 Dataset 검사를 재생성할 필요는 없습니다. 필요할 때만 `preview`를 `create`로 바꾸고 확인 문구를 직접 입력합니다. 미리보기는 작업 폴더·승인·배치를 만들지 않습니다.

- `--batch-size`: Dataset 전용, 기본 32, 허용 1~256. 모델 학습의 배치 설정이 아니라 입력 검사용 배치 크기입니다.
- `--max-input-bytes`: Dataset 전용 캐시 복사 예산. 기본 8 GiB, 통합 생성 계층 허용 1 byte~32 GiB. 완료 캐시보다 작은 상한은 생성 전에 거부합니다.
- `--max-runtime-bytes`: Dataset 전용 설치 라이브러리 복사 예산. 기본 8 GiB, 허용 1 byte~32 GiB. 기존 수집기가 실제 설치 파일의 합계·지문을 검사하며 초과 시 자동 확대하거나 일부만 선택하지 않습니다.
- `--max-output-bytes`: 검사 보고서 출력 예산. Dataset 기본 1 MiB, 통합 생성 계층 최대 64 MiB. 캐시 복사 예산이나 RAM 상한과 구분합니다.

시간·RAM·프로세스 상한은 기존 `WorkerLimits`를 사용합니다. Dataset 전용 옵션을 다른 단계에 지정하거나 Dataset 단계에 분할·변환·캐시 설정 파일을 지정하면 무시하지 않고 거부합니다. 설치·다운로드·모델 파일 로드는 하지 않습니다.

새 미리보기는 `wm_run_creation_v5`입니다. 배치·복사 예산·런타임 지문·PyTorch 버전을 `creation_started.json`의 확인 범위에 포함합니다. 생성 직전 다시 수집한 범위가 다르거나 기존 생성 함수가 저장한 계약이 다르면 새 배정을 완료하지 않습니다. 런타임 파일 목록 자체는 기존 Dataset 생성 함수가 `runtime_files.json`으로 저장하고 기존 검증기로 확인합니다. 부분 실패는 앞 절의 보존 규칙을 따릅니다.

정상 결과는 `created_and_assigned`, 실제 작업 상태는 `created/preflight`입니다. 이후 `validate_contract` 실행 승인은 별도로 필요합니다. 첫 배치 검사·Test 맵 읽기·학습·성능 평가는 수행하지 않습니다. 다른 단계의 시각 검토를 대신하는 기능도 아닙니다. 확인 범위와 상세 경로는 로컬 전용이며 LLM/외부 추적에 보내지 않습니다.

검증: 새 Dataset 생성 합성 테스트 2개와 기존 계획·연결·안내·후속 작업 생성 회귀를 합친 34개가 모두 통과했습니다. 설치 런타임 수집은 작은 명세 대역을 쓰고 부모 캐시까지 합성 완료한 뒤, 새 작업·배정·원래 파일 보존·실행 승인 유지, 잘못된 배치/복사 상한·다른 단계 설정·확인 후 런타임 변경·부모 캐시 변조 거부를 확인했습니다. 실제 WM 작업 생성·새 네이티브 worker 실행·전체 에이전트 테스트 전체집합 실행은 하지 않았습니다.

### 공통 승인·한 단계 실행

2026-10-18 후속 구현. `wm_execution_cli`는 정상 WM 6단계에 배정된 기존 작업을 대상으로 `status`·`approve`·`next`를 제공합니다. 충돌 제외 분기·초기 EDA 생성·전체 자동 진행·승인 대행은 지원하지 않습니다. `wm_guidance_cli`는 여전히 조회 전용이며 이번 변경으로 실행 권한을 추가하지 않았습니다.

실제 실행 전에는 생성 결과의 새 `bindings.json` 경로를 사용합니다. 아래는 새 Dataset 작업을 준비했을 경우의 예제이며 이미 완료한 실데이터 작업을 재실행하는 명령이 아닙니다.

```powershell
$pipelinePath = "configs/agents/pipelines/wm_development_example.json"
$bindingPath = "logs/agents/wm_creation/runs/wm-module-dataset01/bindings.json"

# 조회만 수행하며 승인 저장·소비·worker 실행을 하지 않습니다.
python -m src.agents.wm_execution_cli status --pipeline $pipelinePath --bindings $bindingPath

# 명시한 작업의 preflight 한 단계만 진행합니다.
python -m src.agents.wm_execution_cli next --pipeline $pipelinePath --bindings $bindingPath `
  --step dataset --name wm-module-dataset01 --stage preflight

# 검사 승인을 요청합니다. 이 호출 자체는 승인·검사를 수행하지 않습니다.
python -m src.agents.wm_execution_cli next --pipeline $pipelinePath --bindings $bindingPath `
  --step dataset --name wm-module-dataset01 --stage validate_contract

# 기존 승인 화면에서 설정·영향을 확인하고 표시된 문구를 직접 입력합니다.
python -m src.agents.wm_execution_cli approve --pipeline $pipelinePath --bindings $bindingPath `
  --step dataset --name wm-module-dataset01 --stage validate_contract

# 아래 자리표시는 바로 앞 화면에서 받은 실제 승인 ID로 교체합니다.
python -m src.agents.wm_execution_cli next --pipeline $pipelinePath --bindings $bindingPath `
  --step dataset --name wm-module-dataset01 --stage validate_contract --approval-id "<발급된 승인 ID>"
```

`--step`은 모듈 계획 단계, `--name`은 그 단계에 배정된 작업 이름, `--stage`는 기존 작업 내부의 현재 단계입니다. 셋이 맞지 않으면 다른 작업이나 다음 내부 단계로 자동 이동하지 않습니다. 현재 첫 미완료 단계만 호출하며 미배정·완료·검토 필요·실패·취소·보안 잠금·이미 예약된 작업에서는 자동 실행하지 않습니다.

`approve`는 기존 CLI의 대화형 확인·범위 재검사·OS 실행 잠금·승인 저장 절차를 재사용합니다. 거절이나 비대화형 입력으로는 승인하지 않습니다. 유효한 미사용 승인이 이미 있으면 중복 발급하지 않고 그 ID를 직접 지정해야 합니다. 캐시의 시각 검토·분할 재확인·캐시 실행 승인도 여전히 각각 필요합니다.

`next`는 호출당 내부 단계 하나만 처리합니다. 사용자 지정 승인 ID의 정확한 범위·만료·미사용 상태를 읽기 전용으로 사전 확인하고, 기존 실행기가 소비·예약을 원자적으로 재검사합니다. 다른 유효 승인이 존재해도 잘못 지정한 ID를 대신 골라 쓰지 않습니다. 실행 직전 잠금 안에서 상태·계획·배정 변경을 다시 확인하며 바뀌면 `WM_SELECTION_CHANGED`로 중단합니다. 기존 실행·승인 소스는 수정하지 않고 새 하위 실행기에서 선택 검사를 추가했습니다.

사용자 확인 도중 배정이나 상태가 바뀌면 승인 저장 전에 거부합니다. 실행 도중 외부에서 계획 파일을 편집해도 실행 대상을 새 작업으로 갈아타지 않습니다. 이미 시작한 외부 효과를 되돌리는 보장은 아니며 다음 호출에서 변경을 재검증합니다. 상세 경로·설정·표·그림·승인 ID는 로컬 전용이며 LLM/추적 서비스에 보내지 않습니다. 이 CLI는 모델 학습 기능을 제공하지 않습니다.

검증: 새 합성 테스트 5개와 기존 계획·연결·읽기 전용 안내 회귀를 합친 27개가 모두 통과했습니다. 합성 EDA의 승인 요청·사용자 확인·승인 1회 소비·완료 후 재실행 금지, 거절/비대화형 승인 금지, 다른/만료 승인·실패/취소/보안 잠금 거부와 확인 중 배정·실행 잠금 전 상태 변경 중단을 확인했습니다. 합성 캐시에서도 시각 검토·분할 재확인·캐시 실행의 세 승인을 각각 소비하며 단계별로 진행했습니다. 초기 검증에서 확인한 상태 저장소 타입 요구사항은 기존 저장소를 상속하는 승인 확인용 읽기 계층으로 보완했습니다. 기존 승인·실행 소스를 변경하지 않았습니다. 실제 데이터·새 네이티브 AppContainer worker·전체 에이전트 테스트 전체집합은 이번 변경에서 실행하지 않았습니다.

### 공통 CLI의 실제 격리 합성 검사 준비

2026-10-18. 공통 실행 진입점을 실제 명령으로 확인할 별도 단일 EDA 계획 `wm_common_native_example.json`과 배정 `wm_common_native_runs_example.json`을 추가했습니다. 고정 합성 맵 27개의 `wm-common-native01` 작업을 기존 `create-demo`로 준비했습니다. 실데이터나 기존 완료 WM 결과를 참조하지 않습니다. 준비 명령을 같은 이름으로 다시 실행하지 마세요.

실제 확인에서 공통 실행 API가 상대경로를 worker의 절대경로 검증기에 그대로 전달하는 오류를 발견했습니다. 사용자 측 계획·배정 경로만 현재 폴더 기준으로 절대경로화하도록 보완했습니다. worker의 엄격한 경로 검사·링크 검사·승인 소스는 변경하지 않았습니다. Windows의 실제 작업 폴더 변경을 포함한 기본 회귀 테스트 4개가 통과했습니다. 다른 생성 CLI의 상대경로 지원까지 검증한 기록은 아닙니다.

준비 시 공통 CLI를 통해 `preflight`의 `STEP_COMPLETED`와 `run_eda`의 `APPROVAL_REQUIRED`를 확인했습니다. 이후 사용자가 직접 승인하고 실행해 `completed/STEP_COMPLETED`, `worker_started: true`를 확인했습니다. 이어서 읽기 전용 `status`로 `COMPLETED_RUN_VERIFIED`와 `completed/finish`를 확인했습니다. 따라서 **공통 CLI를 통한 합성 EDA의 실제 AppContainer 실행·완료 검증이 통과**한 기록입니다. 승인 ID는 문서에 저장하지 않습니다.

아래는 완료된 검사의 실행 절차 기록입니다. 같은 완료 작업을 재실행하지 않으며, 새 검사가 필요하면 별도 작업 이름과 배정 파일을 준비합니다. 승인·실행은 사용자의 대화형 터미널에서 진행합니다.

```powershell
$pipelinePath = "configs/agents/pipelines/wm_common_native_example.json"
$bindingPath = "configs/agents/pipelines/wm_common_native_runs_example.json"

python -m src.agents.wm_execution_cli approve `
  --pipeline $pipelinePath --bindings $bindingPath `
  --step eda --name wm-common-native01 --stage run_eda

# 위 화면의 승인 ID는 로컬에서 입력하며 LLM/추적 서비스로 보내지 않습니다.
$approvalId = Read-Host "방금 발급된 승인 ID"
python -m src.agents.wm_execution_cli next `
  --pipeline $pipelinePath --bindings $bindingPath `
  --step eda --name wm-common-native01 --stage run_eda --approval-id $approvalId

python -m src.agents.wm_execution_cli status --pipeline $pipelinePath --bindings $bindingPath
```

성공 판단은 실제 실행 응답의 `completed/STEP_COMPLETED`, `worker_started: true`와 이후 읽기 전용 조회의 `completed/finish`를 함께 확인합니다. 후자의 `worker_started: false`는 조회 명령이 worker를 시작하지 않았다는 뜻이며 앞선 실행의 실패를 뜻하지 않습니다. 조회는 기존 결과 검증기로 승인 소비·산출물·OS 종료·Job 정리·부모 연결을 확인합니다. 실패하면 같은 승인으로 자동 재시도하지 말고 로컬 실패 기록을 검토합니다. 이 검사는 합성 EDA의 공통 진입점 검사이며 WM 전체 개발 6단계의 새 네이티브 종단간 통과나 일반 보안 인증을 의미하지 않습니다.

### 공통 CLI의 검토 필요 중단 검사 준비

2026-10-18. 정상 합성 실행 `wm-common-native01`은 보존하고, 기존 `create-demo --review-demo`로 별도 작업 `wm-common-review01`을 준비했습니다. 단일 EDA 계획은 그대로 재사용하고 배정만 `wm_common_review_runs_example.json`으로 분리했습니다. 합성 맵은 클래스마다 독립 그룹이 하나뿐이므로 품질 검토가 필요하도록 구성합니다. 실제 WM 데이터를 읽거나 품질 기준을 완화하지 않았습니다.

준비 시 공통 CLI의 `preflight` 통과와 `run_eda`의 `awaiting_approval/APPROVAL_REQUIRED`를 확인했습니다. 이후 사용자가 직접 승인·실행해 `blocked_input/TOOL_REVIEW_REQUIRED`, `worker_started: true`를 확인했습니다. 로컬 증거를 읽기 전용으로 재검증한 결과 `review_stop_verified`로 **의도한 품질 검토 중단 검사가 통과**했습니다. 같은 이름이나 소비된 승인으로 재실행하지 않습니다.

아래 명령은 완료한 검사의 절차 기록입니다. 승인 화면은 분석 실행 범위에 대한 동의이며 품질 검토 결과를 자동 해제하는 승인이 아닙니다.

```powershell
$pipelinePath = "configs/agents/pipelines/wm_common_native_example.json"
$bindingPath = "configs/agents/pipelines/wm_common_review_runs_example.json"

python -m src.agents.wm_execution_cli approve `
  --pipeline $pipelinePath --bindings $bindingPath `
  --step eda --name wm-common-review01 --stage run_eda

$approvalId = Read-Host "방금 발급된 승인 ID"
python -m src.agents.wm_execution_cli next `
  --pipeline $pipelinePath --bindings $bindingPath `
  --step eda --name wm-common-review01 --stage run_eda --approval-id $approvalId

python -m src.agents.wm_execution_cli status --pipeline $pipelinePath --bindings $bindingPath
```

예상 결과는 품질 검토 필요에 따른 중단입니다. 실행 응답은 `blocked_input/TOOL_REVIEW_REQUIRED`, `worker_started: true`로 예상합니다.

확인할 항목은 품질 보고서의 `review_required`, EDA 미생성, 정상 완료 산출물 미등록과 다음 단계 미진행입니다. `blocked_input`은 이 검사의 의도한 중단 결과이며 일반 런타임 실패와 구분합니다. 실제 출력과 로컬 증거를 대조한 뒤에만 중단 검사 통과로 기록합니다. 같은 승인으로 재시도하거나 품질 조건을 자동 변경하지 않습니다. 승인 ID는 로컬에서만 사용합니다.

검증 기록: 품질 보고서의 `review_required`, EDA 폴더 미생성, `run_eda` 완료 표식·산출물 참조 미등록, 현재 단계 유지, 정확한 승인 소비·미커밋 작업 상태를 확인했습니다. 기존 `inspect_attempt()`로 요청·응답 일치, 결과 파일 지문·영수증·OS 종료·Job 정리 증거도 재검증했습니다. 조회 전후 상태·선택·승인 DB 지문은 동일했습니다. 실행기의 비통과 작업 기록이 `failed`인 것은 정상 완료로 커밋하지 않았다는 뜻이며, 이 사례는 런타임 실패가 아닌 품질 검토 중단입니다. 기존 정상 성공 작업과 실데이터는 변경하지 않았습니다. 전체 WM 6단계의 새 네이티브 종단간 검증이나 일반 보안 인증으로 확대 해석하지 않습니다.

### 첫 EDA 작업 준비 통합

2026-10-18. 공통 `wm_creation_cli`에서 첫 `wm_canonical_eda` 작업도 기존 `wm_canonical_cli.create()`로 준비합니다. 새 EDA 알고리즘을 만든 것이 아니라 기존 바이트 복사·입력 계약을 공통 생성·배정 절차에 연결한 것입니다. 정상 6단계의 생성과 승인·한 단계 실행을 이제 공통 진입점으로 관리할 수 있습니다. 충돌 제외 분기는 아직 별도입니다.

```powershell
python -m src.agents.wm_creation_cli preview `
  --pipeline configs/agents/pipelines/wm_development_example.json `
  --bindings configs/agents/pipelines/wm_eda_creation_example.json `
  --step eda --name wm-module-eda01 `
  --canonical-dir data/processed/wm811k/canonical `
  --max-input-bytes 4294967296 `
  --memory-mib 8192 --processes 2 --wall-seconds 1800 `
  --max-output-bytes 67108864
```

canonical 폴더는 예시이며 실제로 사용할 묶음의 경로를 직접 확인합니다. 기존 완료 실데이터 흐름을 재생성할 필요는 없습니다. `wm_eda_creation_example.json`은 아무 작업도 배정하지 않은 별도 예제입니다. 생성이 필요할 때만 `preview`를 `create`로 바꾸고 확인 문구를 직접 입력합니다.

- `--canonical-dir`: 첫 EDA 전용. 이미 변환된 canonical 묶음만 받으며 raw pickle·일반 이미지·임의 폴더를 직접 분석하지 않습니다.
- `--max-input-bytes`: 첫 EDA에서는 명시적으로 지정하는 바이트 복사 상한입니다. 허용 1 byte~32 GiB이며 RAM 상한과 다릅니다. Dataset에서는 기존 캐시 복사 상한 의미를 유지합니다.
- `--include-examples`: 첫 EDA 전용. 기본 꺼짐이며 이후 승인 실행에서 예시 맵 그림을 요청하는 옵션입니다. 준비 중에는 그림을 만들지 않습니다.
- `--max-group-ratio`: 첫 EDA 전용. 기본 0.2, 기존 계약의 0 초과~1 이하 검토 상한입니다. 품질 검토 결과를 자동 해제하지 않습니다.
- 출력 예산: 첫 EDA 기본 64 MiB, 통합 생성 계층 최대 1 GiB. 이후 분석 worker가 사용하는 출력 상한이며 준비 중 품질·EDA를 실행하지 않습니다.

입력 선택 검사 `wm_eda_creation.py`는 기존 `inventory()`를 재사용해 허용 파일 이름·일반 파일·링크·파일 개수·합계 크기·지문을 확인합니다. NPZ 배열·pickle·관리 JSON의 데이터 의미는 파싱하지 않습니다. 미리보기는 파일 목록 전체 대신 원본 경로·묶음 식별값·파일 수·총 바이트·설정만 로컬에 표시하며 복사본·작업·승인 DB를 만들지 않습니다. 이는 품질 통과나 안전성 인증이 아닙니다.

사용자 생성 확인 후 선택을 재검증하고 기존 생성 함수가 새 canonical 복사본·`created/preflight` 작업을 준비합니다. 원본과 복사본의 지문·입력/출력 예산·예시/그룹 설정을 다시 확인한 뒤에만 새 배정과 `CREATION_COMPLETE.json`을 기록합니다. 생성 중 달라지면 부분 작업은 보존하되 배정 완료로 등록하지 않습니다. 실제 분석은 이후 별도의 `run_eda` 사용자 승인이 필요합니다.

새 미리보기 형식은 `wm_run_creation_v6`이며 첫 EDA는 부모 작업 대신 `canonical_options`에 원본 선택을 기록합니다. 과거 기록을 덮어쓰거나 이전 미리보기 객체를 자동 재사용하지 않습니다. 후속 단계의 canonical 전용 옵션·첫 EDA의 Dataset/분할/변환/캐시 옵션 혼용은 거부합니다. 공통 생성의 계획·배정·설정 경로도 현재 폴더 기준 절대경로로 변환한 뒤 엄격한 경로 검사를 적용하도록 보완했습니다. 기존 worker·승인 CLI·canonical 계약은 변경하지 않았습니다.

검증: 새 첫 EDA 생성 합성 테스트 4개와 기존 계획·연결·안내·생성·공통 승인/실행 회귀를 합친 43개가 모두 통과했습니다. 미리보기 무변경, 사용자 생성 확인 필요, 새 복사본·배정·별도 실행 승인 유지, 원본 보존, 배열/pickle 파싱·worker/승인 발급 금지, 잘못된 설정·복사 상한 부족·확인 후 원본 변경·생성 후 복사본 변조 거부와 상대경로/CLI 옵션 전달을 확인했습니다. 생성 후 변조 사례는 부분 작업을 남기되 배정·완료 표식을 만들지 않았습니다. 실제 WM 작업이나 새 네이티브 worker는 생성·실행하지 않았으며 전체 에이전트 테스트 전체집합도 이번 변경에서 실행하지 않았습니다.

### 공통 생성 CLI의 실제 합성 작업 준비 확인

2026-10-18. 기존 정상 검사에서 만든 고정 합성 canonical 입력을 재사용해 새 이름 `wm-common-create01`의 `wm_creation_cli preview`를 실제 명령으로 확인했습니다. 결과는 `creation_preview`, 모듈은 `wm_canonical_eda`, `worker_started: false`, `execution_approval_issued: false`입니다. 미리보기 후 새 EDA 실행 폴더와 생성·배정 기록 폴더가 모두 없는 것도 확인했습니다. 기존 성공/검토 중단 작업과 실데이터는 변경하지 않았습니다.

이후 사용자가 아래 명령을 직접 확인해 실행했으며 `created_and_assigned`를 확인했습니다. 로컬 기록도 읽기 전용으로 재검증해 `creation_assignment_verified`를 확인했습니다. 아래는 완료한 생성 절차 기록이며 같은 이름으로 다시 실행하지 않습니다. 생성 확인은 바이트 복사·작업 준비·새 배정에 대한 확인이며 분석 실행 승인이 아닙니다.

```powershell
python -m src.agents.wm_creation_cli create `
  --pipeline configs/agents/pipelines/wm_development_example.json `
  --bindings configs/agents/pipelines/wm_eda_creation_example.json `
  --step eda --name wm-common-create01 `
  --canonical-dir logs/agents/wm_canonical/examples/wm-common-native01/canonical `
  --max-input-bytes 1048576 `
  --memory-mib 2048 --processes 2 --wall-seconds 300 `
  --max-output-bytes 33554432
```

정상 기대값은 `created_and_assigned`, `worker_started: false`, `execution_approval_issued: false`이며 새 배정 경로는 `logs/agents/wm_creation/runs/wm-common-create01/bindings.json`입니다. 생성 출력과 저장 계약·복사본·생성 완료 표식을 확인한 뒤에만 실제 생성·배정 통과로 기록합니다. 생성 완료 후 같은 명령을 반복하지 않습니다. 실패 시 부분 작업을 삭제하거나 자동 재시도하지 말고 로컬 기록을 검토합니다. 이후 승인·실행은 원래 미배정 예제가 아닌 새 배정 파일을 사용합니다.

검증 기록: 새 작업의 `created/preflight`, 새 배정과 원래 빈 배정의 보존, 생성 완료 표식·확인 범위 식별값·작업 ID·배정 파일 지문의 일치, 원본과 별도 복사본의 바이트 지문 및 설정 일치를 확인했습니다. 승인·예약·실행 테이블은 모두 비어 있었고 worker/승인 발급 주장은 거짓으로 유지됐습니다. 조회 전후 상태·선택·승인 DB·생성 기록·배정 파일 지문도 동일했습니다. 따라서 공통 CLI의 실제 합성 입력 복사·작업 생성·배정 확인은 통과했습니다. 이 작업의 품질·EDA 실행과 후속 5단계의 새 네이티브 종단간 검증은 아직 수행하지 않았습니다.

### 공통 생성 결과에서 승인·실행으로 연결

2026-10-18. 새 배정 `logs/agents/wm_creation/runs/wm-common-create01/bindings.json`으로 `preflight`의 `STEP_COMPLETED`와 `run_eda`의 `APPROVAL_REQUIRED`를 확인했습니다. 이후 사용자가 직접 승인·실행해 `completed/STEP_COMPLETED`, `worker_started: true`를 확인했습니다. 읽기 전용 공통 조회에서도 EDA의 `COMPLETED_RUN_VERIFIED`와 다음 `proposal`의 `RUN_NOT_ASSIGNED`를 확인했습니다. 따라서 **공통 생성·복사·새 배정 → 사전검사 → 사용자 승인 → 실제 AppContainer EDA 실행 → 완료 검증·다음 단계 안내 연결이 통과**했습니다. 후속 5단계는 아직 생성·실행하지 않았습니다.

아래는 완료된 검사의 절차 기록입니다. 같은 작업·소비 승인을 다시 실행하지 않습니다. 생성 확인과 별도로 분석 범위·설정을 보고 승인했으며 승인 ID는 문서에 기록하지 않습니다.

```powershell
$pipelinePath = "configs/agents/pipelines/wm_development_example.json"
$bindingPath = "logs/agents/wm_creation/runs/wm-common-create01/bindings.json"

python -m src.agents.wm_execution_cli approve `
  --pipeline $pipelinePath --bindings $bindingPath `
  --step eda --name wm-common-create01 --stage run_eda

$approvalId = Read-Host "방금 발급된 승인 ID"
python -m src.agents.wm_execution_cli next `
  --pipeline $pipelinePath --bindings $bindingPath `
  --step eda --name wm-common-create01 --stage run_eda --approval-id $approvalId

python -m src.agents.wm_execution_cli status --pipeline $pipelinePath --bindings $bindingPath
```

이번 계획에는 후속 5단계도 포함되므로 EDA 성공 후 전체 계획의 상태가 `completed/finish`일 필요는 없습니다. EDA 단계의 `COMPLETED_RUN_VERIFIED`와 다음 `proposal` 단계의 `RUN_NOT_ASSIGNED` 안내가 예상됩니다. 이를 전체 WM 흐름 실패와 혼동하지 않습니다. 실제 실행·결과 재검증 뒤에만 생성부터 실행까지의 연결 통과로 기록합니다. 완료 작업이나 소비 승인은 다시 실행하지 않습니다.

### 새 합성 EDA에서 분할 제안 준비 연결

2026-10-18. 완료된 합성 EDA `wm-common-create01`의 새 배정을 입력으로 `wm-common-proposal01` 분할 제안 생성 미리보기를 확인했습니다. 공통 생성 결과는 `creation_preview`, 모듈은 `wm_split_propose`이며 부모가 해당 EDA와 일치했습니다. 설정은 기존 `configs/wm811k/split.json`을 그대로 사용합니다. 완료 EDA의 품질·그룹 검토 상한과 분할 설정의 정합성도 기존 검증 경로로 확인했습니다.

미리보기 후에는 새 제안 작업·생성 배정 폴더가 없었고 worker 실행·승인 발급도 없었습니다. 이후 사용자가 아래 명령의 확인 문구를 직접 입력해 `created_and_assigned`를 확인했습니다. 저장 기록을 읽기 전용으로 재검증한 결과도 `creation_assignment_verified`입니다. 아래는 완료된 생성 절차 기록이며 같은 이름으로 다시 실행하지 않습니다. 아직 그룹 배정 계산은 수행하지 않았습니다.

```powershell
python -m src.agents.wm_creation_cli create `
  --pipeline configs/agents/pipelines/wm_development_example.json `
  --bindings logs/agents/wm_creation/runs/wm-common-create01/bindings.json `
  --step proposal --name wm-common-proposal01 `
  --split-config configs/wm811k/split.json `
  --memory-mib 2048 --processes 2 --wall-seconds 300 `
  --max-output-bytes 67108864
```

예상 결과는 `created_and_assigned`, `worker_started: false`, `execution_approval_issued: false`이며 새 배정은 `logs/agents/wm_creation/runs/wm-common-proposal01/bindings.json`입니다. 원래 EDA 배정과 성공 결과는 보존합니다. 생성 후에는 새 배정을 사용하며 제안 계산은 사전검사와 별도 사용자 실행 승인 뒤 진행합니다. 분할 제안은 실제 분할 확정이나 학습 준비 완료를 의미하지 않습니다.

검증 기록: 제안 작업의 `created/preflight`, 새 배정의 EDA·제안 연결과 원래 EDA 배정 보존, 부모 EDA의 완료·작업/계획/산출물 식별값 일치, 생성 완료 표식·확인 범위·새 배정 지문 일치를 확인했습니다. 보존된 분할 설정과 제안 도구의 설정도 일치했습니다. 승인·예약·실행 테이블은 모두 비어 있었으며 조회 전후 상태·선택·DB·생성 기록·설정·배정 파일은 변경되지 않았습니다. 따라서 분할 제안의 실제 작업 생성·배정은 통과했지만 제안 실행·분할 확정은 아직 확인하지 않은 상태입니다.

### 새 분할 제안의 실행 승인 연결

2026-10-18. 새 배정을 공통 실행 CLI에서 조회해 현재 작업이 `proposal/wm-common-proposal01`임을 확인했습니다. `preflight`의 `STEP_COMPLETED`와 제안 계산 단계 `derive_preparation`의 `awaiting_approval/APPROVAL_REQUIRED`를 확인했습니다. 이후 사용자가 직접 승인을 발급하고 실행 명령을 전달했습니다. 현재 범위의 실행 가능 상태를 조회한 뒤 해당 명령을 실행했으나 `failed_runtime/TOOL_RUNTIME_FAILED`, `worker_started: false`로 종료됐습니다. 아래 명령은 시도 기록이며 동일 작업·소비 승인을 재실행하지 않습니다.

사용자가 아래 명령으로 제안 설정·영향을 확인하고 직접 승인합니다. 생성 확인과 별개의 실행 승인입니다.

```powershell
$pipelinePath = "configs/agents/pipelines/wm_development_example.json"
$bindingPath = "logs/agents/wm_creation/runs/wm-common-proposal01/bindings.json"

python -m src.agents.wm_execution_cli approve `
  --pipeline $pipelinePath --bindings $bindingPath `
  --step proposal --name wm-common-proposal01 --stage derive_preparation

$approvalId = Read-Host "방금 발급된 승인 ID"
python -m src.agents.wm_execution_cli next `
  --pipeline $pipelinePath --bindings $bindingPath `
  --step proposal --name wm-common-proposal01 --stage derive_preparation --approval-id $approvalId

python -m src.agents.wm_execution_cli status --pipeline $pipelinePath --bindings $bindingPath
```

성공 시 기대값은 제안 단계의 `COMPLETED_RUN_VERIFIED`와 다음 `split` 단계의 `RUN_NOT_ASSIGNED`이지만, 이번 시도는 실패했으므로 통과로 기록하지 않습니다. 제안 계산 성공도 분할 확정 승인은 아닙니다. 승인 ID는 문서에 저장하지 않으며 실패 시 같은 승인으로 자동 재시도하지 않습니다.

실패 증거의 읽기 전용 확인 결과: 승인 상태는 `consumed`, 작업은 `failed`이며 정상 완료 표식은 없습니다. worker 준비 요청과 `transport.json`은 있으나 `launch` 기록·worker 응답은 없고 런타임/도구 복사 파일 수도 0입니다. `profile_cleanup: true`가 기록돼 있지만 프로필 생성이 실제 성공했다는 증거는 아닙니다. 따라서 전송 준비에 진입한 뒤 실제 프로세스 시작 전의 초기 준비 단계에서 예외가 발생한 것으로 범위를 좁혔습니다. 별도 읽기 전용 조회에서 현재 런타임 목록과 512 MiB 제한은 통과했습니다. 다만 예외의 단계·OS 오류 코드가 저장돼 있지 않아 프로필 생성·ACL 설정 중 어디에서 실패했는지는 단정할 수 없습니다. 코드나 설정을 변경하거나 새 승인·작업을 자동 생성하지 않았습니다. 다음 대처는 제한된 준비 단계 진단을 보완한 후 별도 작업·새 사용자 승인으로 확인하는 것입니다.

### 사용자 터미널의 새 제안 성공과 분할 확정 준비

2026-10-18 후속 확인. 코드 수정 대신 사용자가 직접 새 작업 `wm-common-proposal02`를 생성·승인하고 터미널에서 실행했습니다. 실행은 `completed/STEP_COMPLETED`, `worker_started: true`였고, 새 배정으로 읽기 전용 재검증한 결과 제안이 `COMPLETED_RUN_VERIFIED`, 다음 `split`이 `RUN_NOT_ASSIGNED`로 확인됐습니다. 실패한 `wm-common-proposal01`과 소비 승인은 그대로 보존합니다.

두 작업의 저장 기록을 비교해 코드 식별값·기존 실행 계획 식별값·부모 EDA·입력 파일 명세가 동일한 것도 확인했습니다. 따라서 기존 실패를 알고리즘·코드 결함으로 단정하지 않습니다. 실행 환경의 권한 차이 또는 일시적 준비 문제를 의심할 근거는 있지만, 최초 시도의 정확한 OS 오류가 없으므로 원인을 확정한 것은 아닙니다. 이후 실제 생성 확인·승인·격리 실행은 사용자 터미널에서 직접 진행하며 도구는 읽기 전용 확인과 미리보기를 우선합니다.

새 제안 배정 `logs/agents/wm_creation/runs/wm-common-proposal02/bindings.json`으로 분할 확정 작업 `wm-common-commit01`의 생성 미리보기를 확인했습니다. 결과는 `creation_preview`, 모듈 `wm_split_commit`, 부모 제안 일치, `review_required_before_execution: true`, worker/실행 승인 발급 없음이었습니다. 새 확정 작업·생성 배정 폴더도 아직 없었습니다.

사용자가 아래 명령을 실행해 표시된 생성 확인 문구를 직접 입력합니다. 제안의 그룹 배정·설정을 상속하므로 새 분할 설정 파일은 지정하지 않습니다.

```powershell
python -m src.agents.wm_creation_cli create `
  --pipeline configs/agents/pipelines/wm_development_example.json `
  --bindings logs/agents/wm_creation/runs/wm-common-proposal02/bindings.json `
  --step split --name wm-common-commit01 `
  --memory-mib 2048 --processes 2 --wall-seconds 300 `
  --max-output-bytes 33554432
```

이후 사용자가 생성 확인을 직접 수행해 `created_and_assigned`, `worker_started: false`, `execution_approval_issued: false`, `review_required_before_execution: true`를 확인했습니다. 새 배정 경로는 `logs/agents/wm_creation/runs/wm-common-commit01/bindings.json`입니다. 위 생성 명령은 완료 절차 기록이며 같은 이름으로 반복하지 않습니다. 실제 분할 확정은 아직 수행하지 않았습니다. 생성 후에도 기존 제안의 사용자 검토 승인과 확정 실행 승인은 별도로 필요합니다. 소규모 합성 입력 확인용으로 출력 상한은 32 MiB를 선택했으며 실데이터 출력 예산의 권장값을 의미하지 않습니다.

확정 준비의 읽기 전용 검증도 `creation_assignment_verified`로 통과했습니다. `created/preflight`, 원래 EDA·제안 연결 보존과 새 확정 배정, 부모 제안 완료·식별값·산출물 지문, 생성 완료 표식·자원·출력 예산 일치, 검토/확정 실행 승인 요구 유지, 승인·예약·실행 기록 없음과 조회 전후 기록 무변경을 확인했습니다. `review_required_before_execution: true`는 준비 실패나 품질 오류가 아니라 이후 `review_eda` 사용자 검토가 남았다는 뜻입니다. 실제 실행은 `preflight → run_eda(기존 증거 연결) → review_eda(사용자 승인) → propose_split(기존 제안 연결) → commit_split(별도 실행 승인)` 순서이며 기존 EDA·제안을 다시 계산하지 않습니다.

### 새 합성 분할 확정 실행 검증 완료

2026-10-18. 사용자가 `wm-common-commit01`의 사전검사·기존 EDA 증거 연결 후 `review_eda` 검토 승인을 발급했습니다. 최초 실행 요청 차단 시 읽기 전용 조회로 해당 승인이 유효·미사용 상태이며 검토 단계의 예약·작업 기록이 없음을 확인했습니다. 원인은 전달된 명령 인자가 기록되지 않아 단정하지 않았습니다. 사용자가 승인 ID를 다시 입력해 검토를 완료한 뒤 기존 제안을 연결하고 `commit_split`의 별도 실행 승인을 발급·실행했습니다. 실제 결과는 `completed/STEP_COMPLETED`, `worker_started: true`였습니다.

새 배정 파일을 기준으로 읽기 전용 재검증한 결과 분할 확정은 `COMPLETED_RUN_VERIFIED`, 다음 `transform`은 `RUN_NOT_ASSIGNED`로 확인됐습니다. 기존 검증기로 분할 결과·요청/worker 응답·OS 종료/Job 정리·부모 제안 연결을 확인하고, `review_eda`와 `commit_split` 각각의 정확한 승인 소비·작업 커밋·산출물 완료 기록도 대조했습니다. 따라서 공통 생성부터 사용자 검토·별도 확정 승인·실제 격리 확정 실행·완료 검증까지의 합성 연결은 통과했습니다. 승인 ID는 기록하지 않습니다. 변환 검사·캐시·Dataset의 새 네이티브 연결 검증은 아직 남아 있으며 모델 학습·실데이터 분할 변경은 하지 않았습니다.

### 새 합성 분할에서 변환 검사 준비

2026-10-18. 완료 분할 `wm-common-commit01`의 새 배정으로 변환 검사 `wm-common-transform01`의 공통 생성 미리보기를 확인했습니다. 결과는 `creation_preview`, 모듈 `wm_transform_check`, 부모 확정 분할·기존 변환 설정 일치, worker/실행 승인 발급 없음입니다. 새 검사 작업·생성 배정 폴더도 아직 없었습니다. 실제 생성·변환 검사는 수행하지 않았습니다.

설정은 기존 `configs/wm811k/transform.json`을 사용합니다. 목표 크기 128×128, 고정 최근접 보간·종횡비 보존·0 패딩·defect/valid 채널 순서·증강 없음을 유지하며 이번 검증을 위해 설정을 변경하지 않습니다. 사용자가 아래 생성 확인 문구를 직접 입력합니다.

```powershell
python -m src.agents.wm_creation_cli create `
  --pipeline configs/agents/pipelines/wm_development_example.json `
  --bindings logs/agents/wm_creation/runs/wm-common-commit01/bindings.json `
  --step transform --name wm-common-transform01 `
  --transform-config configs/wm811k/transform.json `
  --memory-mib 2048 --processes 2 --wall-seconds 300 `
  --max-output-bytes 33554432
```

이후 사용자가 생성 확인을 직접 수행해 `created_and_assigned`, worker/실행 승인 발급 없음을 확인했습니다. 새 배정은 `logs/agents/wm_creation/runs/wm-common-transform01/bindings.json`입니다. 위 생성 명령은 완료 절차 기록이며 같은 이름으로 반복하지 않습니다. 생성은 설정 복사본·입력 계약·작업만 고정합니다. 실제 Train·Validation 변환 검사는 이후 사전검사와 별도 실행 승인이 필요하며 Test 맵 읽기·캐시 생성·모델 학습은 포함하지 않습니다. 변환 검사 통과 후에도 캐시 생성 전 사용자의 형태 보존 시각 검토가 남습니다. 32 MiB 출력 예산은 작은 합성 입력 검사용이며 실데이터 권장 예산을 의미하지 않습니다.

읽기 전용 검증 결과도 `creation_assignment_verified`로 통과했습니다. `created/preflight`, 기존 EDA·제안·확정 배정 보존과 새 변환 검사 연결, 부모 분할 완료 증거·작업/계획/산출물 식별값 일치, 생성 완료 표식·배정 지문 일치, 저장 변환 설정·자원·출력 예산 일치와 실행 승인 필수 유지, 승인·예약·실행 기록 없음 및 조회 전후 기록 무변경을 확인했습니다. 아직 실제 변환 검사 실행은 하지 않았습니다.

### 새 합성 변환 검사 실행 검증 완료

2026-10-18 후속 확인. 사용자가 `wm-common-transform01`의 사전검사 후 `derive_preparation` 실행 승인을 직접 발급·실행했습니다. 결과는 `completed/STEP_COMPLETED`, `worker_started: true`였으며 새 배정에 대한 읽기 전용 조회에서도 변환 검사가 `COMPLETED_RUN_VERIFIED`, 다음 `cache`가 `RUN_NOT_ASSIGNED`로 확인됐습니다.

기존 결과 검증기로 재검증한 보고서는 `transform_check_passed`, `test_maps_read: false`, `training_ready: false`, `visual_review_required: true`입니다. 요청·결과·부모 분할·산출물·승인 소비·OS 종료/Job 정리 증거도 완료 조회 경로에서 대조했습니다. 따라서 공통 생성부터 실제 격리 변환 검사·완료 검증까지의 합성 연결이 통과했습니다. 승인 ID는 기록하지 않습니다.

비교 그림은 해당 작업의 검증된 시도 폴더 아래 `check/examples.png`에 있습니다. 자동 검사 통과는 사용자의 시각 검토 완료를 뜻하지 않습니다. 캐시 준비 전 원본과 변환 후의 유효 영역·결함 위치·종횡비·패딩을 직접 확인하며, 이후 캐시 작업의 별도 시각 검토 승인도 유지합니다. 캐시·Dataset의 새 네이티브 연결 검증과 모델 학습은 아직 수행하지 않았습니다.

### 새 합성 변환 결과에서 캐시 작업 준비

2026-10-18. 사용자가 비교 그림의 합성 데이터 용도를 확인하고 후속 진행을 요청했습니다. 완료 변환 검사 `wm-common-transform01`을 부모로 캐시 작업 `wm-common-cache01`의 공통 생성 미리보기를 확인했습니다. 결과는 `creation_preview`, 모듈 `wm_cache_prepare`, 부모 연결 일치, 보수적 예상 출력이 32 MiB 요청 상한 이내, `review_required_before_execution: true`였습니다. 새 작업/생성 배정 폴더·worker 실행·실행 승인 발급은 아직 없습니다. 대화상 진행 동의를 시각 검토 승인 DB에 자동 기록하지 않았습니다.

기존 `configs/wm811k/cache.json`의 uint8 캐시 설정·최대 shard 행 수·버퍼 상한을 그대로 사용합니다. 사용자 터미널에서 생성 확인 문구를 직접 입력합니다.

```powershell
python -m src.agents.wm_creation_cli create `
  --pipeline configs/agents/pipelines/wm_development_example.json `
  --bindings logs/agents/wm_creation/runs/wm-common-transform01/bindings.json `
  --step cache --name wm-common-cache01 `
  --cache-config configs/wm811k/cache.json `
  --memory-mib 2048 --processes 2 --wall-seconds 300 `
  --max-output-bytes 33554432
```

기대값은 `created_and_assigned`, worker/실행 승인 발급 없음, `review_required_before_execution: true`입니다. 새 배정은 `logs/agents/wm_creation/runs/wm-common-cache01/bindings.json`입니다. 생성은 설정과 입력 증거·작업을 준비하는 단계이며 실제 캐시 파일 생성은 하지 않습니다. 이후 기존 EDA/분할 증거 연결, 사용자 시각 검토·분할 재확인·캐시 실행의 별도 승인을 유지합니다. Train·Validation만 캐시 대상으로 하며 Test 맵 복원·모델 학습은 하지 않습니다. 출력 상한 32 MiB와 실행 자원은 작은 합성 입력 확인용이며 실데이터 필요량과 구분합니다.

이후 사용자가 `wm-common-cache01` 생성을 직접 확인해 `created_and_assigned`, worker/승인 발급 없음, 검토 필요 플래그를 확인했습니다. 위 명령은 완료 생성 절차 기록이며 같은 이름으로 반복하지 않습니다. 저장 기록의 읽기 전용 검증도 `creation_assignment_verified`로 통과했습니다. `created/preflight`, 부모 변환 검사 완료 증거·식별값·파일 지문, 기존 배정 보존과 새 캐시 배정, 생성 완료 표식·설정 복사본·자원·출력 예산 일치를 확인했습니다. 시각 검토·분할 재확인·캐시 실행의 각각 승인 요구가 유지됐고 승인·예약·실행 테이블은 비어 있었습니다. 조회 전후 관련 기록도 동일했습니다. 현재는 작업 준비 완료이며 실제 캐시 생성·시각 검토 승인 완료가 아닙니다.

### 새 합성 캐시 실행 검증 완료

2026-10-18 후속 확인. 사용자가 `wm-common-cache01`의 사전검사·기존 EDA 연결 후 시각 검토(`review_eda`), 기존 분할 재확인(`commit_split`), 실제 캐시 생성(`prepare_data`)을 각각 별도 승인하고 터미널에서 실행했습니다. 실제 캐시 생성은 `completed/STEP_COMPLETED`, `worker_started: true`였습니다.

최초 완료 조회는 도구 실행 환경에서 검증용 임시 SQLite 파일을 만들지 못해 차단됐습니다. 이는 실제 캐시 worker 실패나 데이터 손상으로 확인된 결과가 아니었습니다. worker 재실행 없이 검증용 임시 파일 생성 권한을 허용한 별도 조회에서 기존 검증기를 다시 실행했고, 결과는 `cache_complete`, `test_maps_read: false`, `training_ready: false`였습니다. 세 승인 단계 각각의 정확한 승인 소비·작업 커밋·완료 표식과 부모/캐시 파일·행 연결·OS 종료/Job 정리 증거도 대조했습니다.

같은 권한 조건에서 새 배정의 전체 진행 상태를 재조회해 캐시의 `COMPLETED_RUN_VERIFIED`와 다음 `dataset`의 `RUN_NOT_ASSIGNED`를 확인했습니다. 따라서 공통 생성부터 사용자 검토·분할 재확인·별도 실행 승인·실제 격리 캐시 생성·완료 검증까지의 합성 연결은 통과했습니다. 승인 ID는 기록하지 않습니다. Dataset·첫 배치의 새 네이티브 연결 검증은 아직 남아 있으며 Test 처리·모델 학습은 하지 않았습니다.

### 새 합성 캐시에서 Dataset 검사 작업 준비

2026-10-18. 완료 캐시 `wm-common-cache01`의 새 배정으로 Dataset·첫 배치 검사 작업 `wm-common-dataset01`의 공통 생성 미리보기를 확인했습니다. 캐시 행 연결 검증의 임시 SQLite에 필요한 권한을 허용해 조회만 수행했으며 실제 worker는 시작하지 않았습니다. 결과는 `creation_preview`, 모듈 `wm_dataset_check`, 부모 캐시 일치·캐시 복사 예산 충족·설치 런타임 지문 고정·worker/실행 승인 발급 없음입니다. 새 작업 및 생성 배정 폴더도 아직 없습니다.

배치 크기는 32, 작은 합성 캐시 복사 상한은 32 MiB, 설치 PyTorch 런타임 복사 상한은 8 GiB, worker RAM은 8 GiB, 프로세스는 2개, 실행 시간은 300초, 검사 보고서 출력은 1 MiB로 준비합니다. 캐시·런타임 복사 예산과 worker RAM은 서로 다른 제한입니다. 런타임 목록·파일 지문 수집은 설치 환경 읽기이며 패키지 설치·모델 학습·캐시 재생성이 아닙니다.

사용자 터미널에서 아래 생성 확인 문구를 직접 입력합니다.

```powershell
python -m src.agents.wm_creation_cli create `
  --pipeline configs/agents/pipelines/wm_development_example.json `
  --bindings logs/agents/wm_creation/runs/wm-common-cache01/bindings.json `
  --step dataset --name wm-common-dataset01 `
  --batch-size 32 `
  --max-input-bytes 33554432 --max-runtime-bytes 8589934592 `
  --memory-mib 8192 --processes 2 --wall-seconds 300 `
  --max-output-bytes 1048576
```

기대값은 `created_and_assigned`, worker/실행 승인 발급 없음이며 새 배정은 `logs/agents/wm_creation/runs/wm-common-dataset01/bindings.json`입니다. 실제 검사는 이후 사전검사와 별도 `validate_contract` 실행 승인이 필요합니다. 합성 표본 수가 설정 배치 크기보다 작으면 실제 첫 배치 행 수가 32보다 작은 것은 정상일 수 있습니다. 값은 실제 검증 결과와 구간 표본 수를 대조해 해석합니다. 아직 실제 작업 생성·Dataset 검사·전체 새 6단계 완료 검증을 통과로 기록하지 않습니다. Test 맵 읽기·모델 학습은 포함하지 않습니다.

이후 사용자가 `wm-common-dataset01`을 직접 생성해 `created_and_assigned`, worker/승인 발급 없음을 확인했습니다. 위 생성 명령은 완료 절차 기록이며 같은 이름으로 반복하지 않습니다. 검증용 임시 SQLite 권한을 허용한 저장 기록 대조도 `creation_assignment_verified`로 통과했습니다. `created/preflight`, 기존 5단계 배정 보존·새 Dataset 배정, 부모 캐시 완료 증거·식별값·산출물 지문 일치, 생성 완료 표식·자원·배치·복사/출력 예산·PyTorch 버전·런타임 명세 지문 일치를 확인했습니다. 실행 승인 요구는 유지되고 승인·예약·실행 테이블은 비어 있었으며 조회 전후 관련 기록도 동일했습니다. 실제 첫 배치 검사와 새 6단계 전체 완료 검증은 아직 수행하지 않았습니다.

### 새 합성 6단계 전체 완료 검증

2026-10-18 후속 확인. 사용자가 Dataset 작업의 사전검사·`validate_contract` 실행 승인을 직접 진행하고 터미널에서 실제 AppContainer worker를 실행했습니다. 결과는 `completed/STEP_COMPLETED`, `worker_started: true`였습니다. 읽기 전용 결과 검증에서도 `dataset_batch_passed`, `development_inputs_ready: true`, `test_maps_read: false`, `model_trained: false`를 확인했습니다.

최종 배정 `logs/agents/wm_creation/runs/wm-common-dataset01/bindings.json`을 기준으로 공통 상태 조회를 수행한 결과 아래 6단계가 모두 `COMPLETED_RUN_VERIFIED`였습니다. 전체 상태는 `completed`, 다음 안내는 `finish`입니다.

| 계획 단계 | 확인한 작업 |
| --- | --- |
| EDA | `wm-common-create01` |
| 분할 제안 | `wm-common-proposal02` |
| 분할 확정 | `wm-common-commit01` |
| 변환 검사 | `wm-common-transform01` |
| 캐시 생성 | `wm-common-cache01` |
| Dataset·첫 배치 | `wm-common-dataset01` |

각 단계의 현재 입력·코드·계획, 부모 결과 연결, 정확한 승인 소비·작업 커밋·완료 표식, 산출물과 요청/응답 일치, 실제 worker 종료·Job 정리 증거를 기존 검증기로 대조했습니다. 검증용 임시 SQLite에 필요한 권한만 허용해 조회했고 worker 실행·승인 소비를 다시 수행하지 않았습니다. 따라서 조회 응답의 `worker_started: false`, `approval_consumed: false`는 이번 조회의 무실행을 뜻하며 앞선 사용자 실행이 없었다는 뜻이 아닙니다.

이번 결과는 **고정 합성 canonical 입력에서 공통 작업 생성·배정 → 사용자 단계별 승인 → 실제 격리 실행 → 정상 6단계 완료·부모 연결 검증**을 통과한 기록입니다. 기존 검토 필요 중단 검사도 별도로 보존합니다. 실패한 `wm-common-proposal01`과 소비 승인 기록은 덮어쓰거나 삭제하지 않았습니다. 전체 흐름을 사용자 승인 없이 자동 수행한 것은 아니며 실데이터 전처리 재실행·Test 맵 처리·모델 학습·분류 성능 평가·일반 보안 인증을 뜻하지 않습니다.

위 합성 작업들은 다시 실행할 필요가 없습니다. 충돌 제외 분기의 공통 생성/실행 통합, 다른 형식의 후속 연결, 범용 전달 묶음, 사용자 모듈 확대와 LLM 제안 연결 등은 별도 후속 범위입니다. 이번 완료 확인에서는 소스 코드를 변경하거나 전체 에이전트 테스트 전체집합을 재실행하지 않았습니다.
