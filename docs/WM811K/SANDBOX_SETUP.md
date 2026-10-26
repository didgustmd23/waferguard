# WM-811K 최초 읽기용 Sandbox 준비

> **실제 실행 상태:** 이 실행 환경에서는 Windows Sandbox를 사용할 수 없어, 사용자의 승인 후 `--trusted-pickle --allow-host` 경로로 원본 pickle을 호스트에서 변환했습니다. 이 문서의 Sandbox 절차는 실제 처리에 사용하지 않았습니다. 호스트 실행은 pickle의 코드 실행 위험을 제거하지 않으며 Sandbox 검증으로 간주하지 않습니다.

## 승인된 로컬 실행과 라벨 수정 재변환 (2026-10-10)

사용자가 확보한 pickle을 신뢰하고 격리 없이 실행하기로 명시적으로 결정한 경우의 선택 경로다. pickle의 코드 실행 위험이 사라지는 것은 아니며, 도구가 안전성을 인증한 것이 아니다. 로컬 실행은 `--trusted-pickle --allow-host --output-root`를 모두 지정해야 가능하다. 기본 Sandbox 실행은 유지한다. 아래 경로는 실제 파일명으로 바꾼다. 파일명·해시·내용을 LLM에 전달할 필요는 없다.

```powershell
python scripts/wm_sandbox/export_canonical.py `
  --input "data/raw/wm811k/실제파일명.pkl" `
  --output-root data/processed/wm811k/canonical `
  --trusted-pickle --allow-host
```

`wm_label_v2`는 원본 표기 `Near-full`을 canonical `Near-Full`로 정규화하고 정상 `none`→`None` 규칙도 유지한다. 9개 클래스와 중첩 배열의 합성 회귀 테스트를 수행한다. 새 UUID 폴더에 저장해 기존 결과를 덮어쓰지 않는다. 메타데이터에는 라벨 규칙 버전·별칭·실행 모드를 기록하며 원본 해시 기반 sample ID 계약은 유지한다. 같은 원본이라도 라벨 규칙 버전이 다른 결과는 서로 다른 변환 묶음으로 취급한다.

원본 로드 완료 뒤에는 변환 진행률을 정수 퍼센트로 표시한다. 9개 중 미검출 클래스가 있으면 경고하지만 정답을 생성하거나 예상 건수에 맞춰 보충하지 않는다. `canonical_export_complete`는 형식 변환 완료이며 모든 클래스의 품질 승인·분할 완료를 뜻하지 않는다. 새 묶음으로 `validate_canonical`과 `diagnose_split`을 다시 수행하고, 이전 보고서는 이전 묶음의 결과로 보존한다. 과거 잘못된 라벨의 원문이 manifest에 없으므로 기존 격리 행을 임의 재라벨링하지 않는다.

2026-10-10. 오프라인 환경 준비와 명시적으로 실행하는 최초 구조 검사 스크립트를 제공한다. 실제 원본은 작성 과정에서 읽지 않았다. canonical 변환·전체 audit·출력 재검증은 후속 작업이며, 기존 SECOM·Conda·학습 환경을 바꾸지 않는다.

## 1. 호스트에서 다운로드

프로젝트 루트의 PowerShell에서 다음을 실행한다. 패키지를 설치하지 않고 Windows x64 CPython 3.13용 wheel과 의존성만 내려받는다. 활성 Python은 다운로드 실행용이며 Sandbox Python과 같을 필요는 없다.

```powershell
New-Item -ItemType Directory -Force .sandbox_offline/wheels
python -m pip --isolated download --index-url https://pypi.org/simple `
  --only-binary=:all: --platform win_amd64 --python-version 313 `
  --implementation cp --abi cp313 `
  --dest .sandbox_offline/wheels -r scripts/wm_sandbox/requirements.txt
```

공식 [Python 3.13.16 embeddable ZIP (64-bit)](https://www.python.org/ftp/python/3.13.16/python-3.13.16-embed-amd64.zip)을 내려받아 `.sandbox_offline/python-3.13.16-embed-amd64.zip`에 둔다. 현재 기본 경로는 설치 프로그램을 사용하지 않는다. ARM64·32-bit 배포본과 혼동하지 않는다.

호스트에서는 원본 pickle을 열지 않는다. Python ZIP과 wheel은 공식 배포 경로에서 확보한다. 다운로드 출처와 로컬 파일 해시는 로컬에서 기록한다. 직접 계산한 해시만으로 안전성을 보장하지 않는다.

wheel 확보 방식은 [pip download 공식 문서](https://pip.pypa.io/en/stable/cli/pip_download/)를 따른다. 버전 고정은 재현성 기준이며 취약점 부재 증명이 아니다. 최신 환경에서 과거 pickle이 호환되지 않을 수 있으며, 실패 시 임의 모듈 주입 대신 격리된 호환 환경을 별도 검토한다.

## 2. 새 설정으로 실행

공유 설정은 `configs/wm811k/sandbox_inspect.example.wsb` 템플릿이다. 직접 실행하지 않고 같은 폴더의 `sandbox_inspect.local.wsb`로 복사한다. 복사본에서 `__PROJECT_ROOT__` 네 곳을 현재 프로젝트의 절대 경로로 바꾸고 저장한다. XML 특수문자가 있는 경로는 XML 형식에 맞게 이스케이프한다. `.local.wsb`는 Git에서 제외하며 개인 경로가 들어간 복사본은 공유하지 않는다.

전용 호스트 출력 폴더 `.sandbox_output`을 준비한 뒤 현재 Sandbox를 닫고 수정한 `sandbox_inspect.local.wsb`를 더블클릭한다. 네 HostFolder는 실행 전에 존재해야 한다. 입력·설치 파일·스크립트는 읽기 전용이고, 출력 폴더 하나만 `C:\wm_output`에 쓰기 가능으로 연결한다. 이전 `sandbox_inspect.wsb`를 사용했다면 새 로컬 복사본을 준비한다. 입력 데이터나 기존 Sandbox 출력은 이 변경으로 삭제하지 않는다.

이미 위 네 폴더가 연결된 Sandbox가 실행 중이면 닫지 않는다. 공유 폴더에 추가한 ZIP과 스크립트를 현재 세션에서 사용한다. 새 `.wsb`에서는 설치 프로그램 자동 실행을 제거했다. 이 변경은 이미 실행 중인 설치 프로세스를 종료하지 않는다.

출력 공유는 악성 pickle도 쓸 수 있는 경로다. 중요한 파일·프로그램·다른 프로젝트를 두지 않고 결과를 비신뢰 파일로 취급한다. 결과를 자동 실행·import하거나 LLM에 보내지 않는다. 파일 공유의 위험을 없앴다고 주장하지 않는다. 처음 실행 전 출력 폴더가 비어 있는지 로컬에서 확인한다. 반복 검사 결과는 덮어쓰지 않고 새 하위 폴더로 남는다.

네트워크·클립보드 차단과 공유 폴더 쓰기 거부를 다시 확인한다. 설정 파일만으로 검증 완료를 선언하지 않는다. [Microsoft 설정 안내](https://learn.microsoft.com/en-us/windows/security/application-security/application-isolation/windows-sandbox/windows-sandbox-configure-using-wsb-file).

## 3. Sandbox 내부에서 설치 없이 준비

Sandbox에서 PowerShell을 열고 다음을 실행한다. Windows 설치 프로그램이나 pip을 실행하지 않고 공식 ZIP과 고정 wheel을 별도 실행 폴더에 해제한다. embeddable 배포에 라이브러리를 함께 배치하는 방식은 [Python 공식 문서](https://docs.python.org/3.13/using/windows.html#the-embeddable-package)를 따른다. 클립보드·네트워크 차단은 유지한다.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\wm_tools\prepare_portable.ps1
```

ExecutionPolicy 설정은 이번 PowerShell 프로세스에만 적용하며 시스템 정책을 변경하지 않는다. 실행 전 호스트에서 스크립트를 검토한다. 환경은 `C:\wm_portable\<임의ID>`에 준비하고 원본 경로는 열지 않는다. 압축 경로와 해제 크기를 검사하며, import·버전·64비트 확인 성공 후에만 활성 실행 경로를 기록한다. 실패 시 기존 시도 폴더는 삭제하거나 덮어쓰지 않는다. 실제 Sandbox에서의 실행과 pickle 호환성은 별도 확인이 필요하다.

완료 메시지: `준비 완료. 원본은 읽지 않았습니다. 이제 run_inspection.ps1을 실행하세요.`

기존 `install_offline.ps1`은 이전 설치 방식으로 남겨 두었지만 자동 호출하지 않는다. 설치가 오래 걸린다고 반복 실행하거나 모든 `msiexec.exe`를 일괄 종료하지 않는다. 포터블 방식은 기존 설치 완료를 기다리지 않는다.

Sandbox를 닫으면 내부 설치는 사라지지만 출력 공유의 결과는 호스트에 남는다. 설치 성공은 전체 보안 준수·원본 안전성·실데이터 호환성 통과를 뜻하지 않는다.

## 4. 최초 구조 검사

설치 완료 메시지를 확인하고 네트워크·클립보드 차단, 입력 쓰기 거부, 출력 전용 폴더 연결을 다시 확인한다. 출처를 신뢰하기로 결정한 경우에만 Sandbox 안에서 다음 명령을 직접 입력한다.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\wm_tools\run_inspection.ps1 -TrustedPickle
```

입력 폴더 바로 아래 `.pkl`/`.pickle`이 하나일 때만 자동 선택한다. 여러 파일이면 `-InputFile C:\wm_input\실제파일명.pkl`을 추가한다. 압축 파일은 자동 해제하지 않는다. 명시적 승인은 악성 파일이 무해하다는 증명이 아니다.

전체 pickle을 메모리에 읽은 후 앞부분 최대 1,000행의 맵·라벨 구조를 검사한다. 샘플은 대표 표본이 아니며 전체 품질·중복·Lot·클래스 audit 완료를 뜻하지 않는다. 파일 크기 4 GiB 상한은 보조 제한일 뿐 16 GiB RAM 내 실행을 보장하지 않는다. 메모리 부족·호환성 실패 시 원본을 호스트에서 재시도하거나 임의 모듈을 주입하지 않는다.

결과는 호스트 `.sandbox_output/inspection_<임의ID>/inspection.json`과 `inspection.md`에 남는다. 상태가 `inspection_complete`인지 사용자가 로컬에서 확인한다. 상세 집계·컬럼·해시·환경 정보는 LLM에 붙여 넣지 않는다. `blocked_schema`나 `inspection_failed`이면 후속 처리는 진행하지 않는다. 현재 코드는 정형 실패 코드만 저장하고 원문 예외는 보존하지 않는다.

테스트는 실제 pickle을 읽지 않는 합성 입력 검사다.

```powershell
python -m unittest tests.test_wm_sandbox_inspect -v
```

## 5. 숫자 전용 canonical 변환

최초 구조 검사 성공 후 Sandbox에서 실행한다. 호스트에서 원본을 열지 않는다.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\wm_tools\run_inspection.ps1 -TrustedPickle -Canonical
```

`.sandbox_output/canonical_<임의ID>/maps/*.npz`에 원본 크기의 uint8 맵을 저장한다. 로드 시 `allow_pickle=False`를 사용한다. `manifest.jsonl`은 모든 행의 라벨·Lot·품질 상태·배열 참조를 보존한다. 미라벨을 정상으로 바꾸지 않고 무효 행은 사유와 함께 격리한다. 배열 한도는 1,000,000칸, shard는 최대 1,000개 맵이다. 최초 배포본 변환용 독립 도구이며 CNN 전처리 코어 전체 구현이 아니다.

`SUCCESS.json`이 있어야 변환 완료 후보로 취급한다. 중복·동일 ID 충돌·Lot 연결 그룹 검사, 호스트의 비신뢰 산출물 검증, 분할·resize·증강은 아직 미완료다. `training_ready=false`를 유지한다. 출력은 비신뢰 파일이며 자동 실행·import하지 않는다. 상세 내용·집계·식별값은 LLM에 보내지 않는다. 실제 원본 실행은 사용자가 Sandbox에서 수행한다.

```powershell
python -m unittest tests.test_wm_canonical_export tests.test_wm_sandbox_inspect -v
```

## 6. 호스트에서 변환 산출물 검증

이미 검증을 완료했다면 아래 명령으로 기존 보고서의 분할 검토 사유만 확인할 수 있다. 실제 보고서 경로로 바꾸며, NPZ·manifest를 다시 읽거나 기존 보고서를 수정하지 않는다.

```powershell
python -m src.wafer_dl.validate_canonical --show-report logs/wm811k/canonical_validation_03/validation.json
```

라벨 충돌 없음, 웨이퍼 ID 충돌 없음, 유효 라벨 그룹 존재, 클래스별 독립 그룹 확보, 최대 연결 그룹 비율 조건의 통과·검토 필요 여부를 각각 표시한다. 콘솔은 실제 집계 수치·Lot·맵·샘플 식별자를 출력하지 않는다. 새로 생성하는 Markdown 보고서에도 같은 조건별 진단을 기록한다. 기존 보고서는 `--show-report`만으로 수정되지 않는다.

`canonical_<생성ID>` 폴더 전체를 `data/processed/wm811k/canonical/` 아래에 보존한 뒤 실제 PC의 프로젝트 루트에서 실행한다. 원본 pickle은 읽지 않는다. 입력은 상대 경로 기반 묶음 전체여야 하며 보고서는 입력 바깥의 새 폴더에 기록한다.

```powershell
python -m src.wafer_dl.validate_canonical `
  --input-dir data/processed/wm811k/canonical/canonical_<생성ID> `
  --output-dir logs/wm811k/canonical_validation_01
```

`<생성ID>`는 실제 폴더명으로 바꾼다. 기존 보고서 폴더가 있으면 덮어쓰지 않으므로 다른 새 이름을 사용한다. 검증기는 manifest를 줄 단위로 임시 SQLite에 색인화하고 shard별로 배열을 읽는다. 임시 DB는 OS 임시 폴더에 생성 후 정리하며, 충분한 임시 디스크 공간이 필요하다. 메모리는 전체 행의 JSON 객체 대신 shard와 Lot 연결 정보 위주로 사용하지만 자원 사용량이 0이 되는 것은 아니다.

검사 범위는 완료 계약·행 위치/ID·고정 클래스 매핑·미라벨/격리 사유·집계·참조 경로, 배열 구성원·dtype·shape·허용값·내용 해시·다이 수·비율이다. JSON 중복 키·비유한 숫자, 폴더 밖 참조, object 배열, 과대 배열·shard는 거부한다. NPZ는 디스크에 풀지 않고 `allow_pickle=False`로 숫자 배열만 읽는다. 제한은 현재 exporter 계약(행 200만 이하, 배열 100만 칸 이하, shard당 1,000개 이하)에 맞춘다.

`validation.json`과 `validation.md`를 저장한다. `integrity_passed`는 저장 정합성이 통과했다는 뜻이다. 중복 맵의 라벨 충돌·Lot+wafer ID 충돌은 별도 집계하며 라벨을 자동 수정하거나 행을 삭제하지 않는다. 미라벨을 포함한 Lot·동일 맵 연결 그룹에서 클래스별 독립 그룹 3개 이상과 최대 그룹 비율을 진단한다. 최대 그룹 비율은 상세 설계의 임시 검토 기준 20%이며 `--max-group-ratio`로 명시할 수 있다. 모델 성능 기준이 아니므로 통과시키려고 임의 완화하지 않는다.

`split_assessment.status`는 `preconditions_passed` 또는 `review_required`다. 전자는 필요조건만 통과한 상태이며 실제 분할 가능성을 확정하지 않아 `split_feasible=null`이다. 두 상태 모두 `training_ready=false`를 유지한다. 저장 계약 실패 시 종료 코드는 1, 정합성 통과 시 0이며 분할 검토 필요 여부는 별도로 확인한다. 원본 동등성·무해성 인증이 아니며 원본/검증 집계/식별정보를 LLM으로 보내지 않는다.

```powershell
python -m unittest tests.test_validate_canonical tests.test_wm_canonical_export tests.test_wm_sandbox_inspect -v
```

## 7. 라벨 충돌 그룹 제외 영향의 가상 비교

정합성 검증은 통과했으나 `MAP_LABEL_CONFLICT`·`INSUFFICIENT_CLASS_GROUPS`가 나온 경우, 같은 canonical 묶음을 입력으로 아래 진단을 실행한다. 실제 폴더명으로 바꾸고 새 출력 폴더를 지정한다.

```powershell
python -m src.wafer_dl.diagnose_split `
  --input-dir "data/processed/wm811k/canonical/canonical_<생성ID>" `
  --output-dir logs/wm811k/split_diagnosis_01
```

manifest·관리 문서만 읽고 배열 내용은 다시 검사하지 않는다. 직전에 검증한 동일 묶음을 대상으로 사용하며 이 진단을 NPZ 정합성 검사의 대체로 취급하지 않는다. 임시 SQLite에서 명시적 라벨 충돌 hash의 모든 행(미라벨·격리 행 포함)을 가상 제외한 뒤 Lot·동일 맵 연결 관계와 클래스별 표본·Lot·독립 그룹을 재계산한다. 원본 파일·manifest·라벨·실제 분할은 변경하지 않는다. 실제 제외 정책은 미승인 상태로 유지한다.

결과는 `diagnosis.md`, `diagnosis.json`, `class_comparison.csv`다. 클래스별 전후 표와 부족 그룹은 로컬에서 확인한다. 보고서에도 Lot·맵 해시·샘플 ID는 출력하지 않으며 집계는 LLM으로 보내지 않는다. 충돌 맵 제거로 연결이 끊기면 그룹 수가 늘 수 있고, 표본 손실로 줄 수도 있다. `preconditions_passed`여도 실제 분할 성공을 보장하지 않는다. 가상 제외 후에도 부족 그룹·ID 충돌·큰 그룹이 남으면 평가·제외 정책을 별도 결정하며 자동으로 기준을 완화하지 않는다.

## 8. 분할 전 EDA — 노트북·CLI·에이전트 공통 코어

`reports/wm811k/analysis.ipynb`를 열고 프로젝트 환경의 Python 커널을 선택한다. 원본 pickle은 읽지 않으며, 먼저 정합성 검증을 통과한 `wm_label_v2` canonical 묶음을 사용한다. 후보가 하나일 때만 자동 선택하고, 여러 후보면 `CANONICAL_DIR`를 명시한다. 전체 metadata 집계에는 RAM이 필요하며 전체 맵을 동시에 적재하지 않는다. 예시 셀에서만 클래스별 첫 유효 맵을 숫자 전용으로 읽는다. 예시는 대표성을 보장하지 않는다. Notebook 셀은 배포 시 미실행·출력 없는 상태이며, 실제 사용 후 Git 공유 전 출력 제거가 필요하다.

공통 코어 `src.wafer_dl.eda.run_eda()`는 로컬 표·그림·보고서를 생성한다. 에이전트 실행기 구현 시 같은 함수를 정해진 로컬 승인 노드에서 호출하며, 노트북 코드를 실행 도구로 사용하지 않는다. 현재 LangChain/LangGraph 도구 등록·LLM 연결은 하지 않았으며 기존 외부 상태/동작 열거값을 확장하지 않았다. 반환 `{"status":"eda_complete"}`는 로컬 실행 결과일 뿐 provider DTO가 아니다. 이미지·집계·경로·오류 상세는 LLM·LangSmith로 전송하지 않고 새 상태가 필요하면 기존 고정 매핑과 승인을 거친다. EDA 결과만으로 라벨·분할·해상도를 자동 변경하지 않는다.

```powershell
python -m src.wafer_dl.eda `
  --input-dir "data/processed/wm811k/canonical/canonical_<생성ID>" `
  --output-dir reports/wm811k/figures/eda_01
```

예시 NPZ 없이 metadata 분포만 보려면 `--metadata-only`를 추가한다. 출력은 새 폴더에만 기록하고 마지막 `SUCCESS.json`이 있어야 완료로 취급한다. 이전 부분 결과는 자동 삭제하지 않는다. 분할 전 EDA는 클래스·결측·맵 크기·Lot·중복 같은 구조/품질 확인용이며 모델·해상도·증강 선택은 Train/Validation으로 제한한다.
