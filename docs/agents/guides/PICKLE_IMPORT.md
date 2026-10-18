# 구조 판별형 pickle 입력 변환

## 변경 방향

pickle 처리의 진입점을 WM 전용에서 **공통 승인·격리 실행 + 등록 변환기** 구조로 변경했다. 사용자는 기본적으로 파일과 자원·출력 상한을 선택한다. 승인 후 기존 AppContainer worker가 객체 구조를 확인하고, 지원 객체가 하나로 명확할 때만 변환한다. 데이터 의미를 LLM이나 휴리스틱으로 추측하지 않는다.

```text
사용자 파일 선택 → 바이트 snapshot 고정 → preflight → 사용자 승인
  → AppContainer + Job에서 pickle 역직렬화
  → 제한된 구조 검사 → 등록 변환기 선택
      ├─ 단일 지원 객체: canonical 생성 → 부모 검증 → 완료 등록
      └─ 모호함·미지원·검사 상한: 로컬 구조 보고서 → 검토 중단
```

구조를 확인하기 위해서도 pickle을 먼저 읽어야 한다. 타입 검사는 역직렬화 전의 안전성 검사나 무해성 인증이 아니다. 부모·승인 CLI는 pickle을 읽지 않으며, 격리 토큰·capability 부재·Job 소속을 확인한 worker 진입점만 역직렬화한다. 승인 1회 소비, 입력·코드 결합, 취소, 자원 한도, 종료·정리 증거 확인을 유지한다.

## 현재 지원 구조

| 변환기 | 입력 | 결과 |
| --- | --- | --- |
| `tabular_v1` | 일반 스칼라 셀의 pandas DataFrame | 자료형·인덱스 스키마 + 태그 기반 JSONL |
| `numeric_array_v1` | object가 아닌 NumPy 숫자·bool·복소 배열 | dtype·shape 스키마 + `allow_pickle=False` NPY |
| `wm811k_v1` | `waferMap`·`failureType`·`lotName`이 있는 기존 WM 구조 | 기존 manifest·숫자 NPZ·출처·완료 기록 |

기본 `--converter auto`는 등록된 구조 서명을 사용한다. 일반 `object` 컬럼도 문자열·숫자·결측 같은 허용 스칼라이면 처리한다. 임의 사용자 클래스, object 배열, 일반 표의 중첩 배열 셀, MultiIndex, 범주형 dtype·시간대 dtype 등 미지원 구조는 검토가 필요하다. 정확한 지원 자료형은 `src/preprocessing/canonical_data.py`의 `TABLE_DTYPES`를 따른다.

기본 dict·list·tuple 내부도 검사한다. 지원 객체가 하나이고 나머지가 허용 스칼라이면 해당 객체를 선택한다. 여러 객체, 미지원 구성 요소 또는 검사 상한 초과가 있으면 자동 확정하지 않는다. 구조 깊이 4, 노드 2,048, 후보 64, dict 항목 256의 검사 상한을 둔다. 일반 표는 최대 2,000,000행·4,096열·100,000,000셀, 배열은 최대 8차원·100,000,000원소이며 실제 실행에는 별도 메모리·시간·출력 상한도 적용된다.

선택은 **선택 객체의 변환**이다. dict의 다른 필드까지 canonical에 보존한다는 뜻이 아니다. 선택 경로와 범위를 출처에 기록하고 전체 원본은 읽기 전용 전달 snapshot에 남긴다. label·group·시간·채널 의미는 추측하지 않으며 후속 프로파일에서 명시한다.

## 합성 실행

프로젝트 루트와 `ml310` 환경에서 새 이름으로 진행한다.

```powershell
python -m src.agents.pickle_import_cli create-demo `
  --name pickle-table01 --demo-kind tabular `
  --memory-mib 2048 --processes 2 --wall-seconds 60

python -m src.agents.pickle_import_cli next --name pickle-table01
python -m src.agents.pickle_import_cli next --name pickle-table01
python -m src.agents.pickle_import_cli approve --name pickle-table01

# 화면에서 승인한 뒤 실제 발급 ID로 바꿉니다.
python -m src.agents.pickle_import_cli next `
  --name pickle-table01 --approval-id 발급된_ID

python -m src.agents.pickle_import_cli status --name pickle-table01
```

`--demo-kind array`, `wm`, `ambiguous`로 다른 고정 합성 입력을 확인할 수 있다. `ambiguous`는 완료 대신 `TOOL_REVIEW_REQUIRED`로 중단하는 것이 정상이다. `status`의 `local_structure_report` 옆 `structure.json`에서 후보 경로와 변환기를 로컬로 확인한다.

## 사용자 파일과 명시적 선택

```powershell
python -m src.agents.pickle_import_cli create `
  --name pickle-selected01 --source data/raw/example.pkl `
  --converter numeric_array_v1 --select-path '["features", 0]' `
  --max-input-bytes 8388608 --max-output-bytes 33554432 `
  --memory-mib 2048 --processes 2 --wall-seconds 300 `
  --confirm-pickle-selection
```

`--select-path`는 dict의 문자열 키와 list/tuple의 음수가 아닌 인덱스로 된 JSON 배열이다. 위 예시는 `root["features"][0]`을 선택한다. 속성 접근·Python 식·사용자 함수·동적 import 경로는 허용하지 않는다. 생략하면 전체 구조를 검사한다. `--converter`는 등록된 ID 중 선택한다.

검토 중단 후에는 경로·변환기를 정해 **새 이름의 실행과 새 승인**을 만든다. 기존 승인을 재사용하거나 중단된 작업을 자동 해제하지 않는다. 코드가 변경된 기존 실행도 새 승인이 필요하다.

## 결과와 후속 연결 범위

일반 표·배열 결과는 `schema.json`, `source.json`, `manifest.json`, `SUCCESS.json`과 `data.jsonl` 또는 `array.npy`다. 표의 NaN·pd.NA·NaT·무한대·날짜·인덱스를 명시적으로 기록하여 단순 CSV 저장에서 생길 타입 혼동을 피한다. 부모는 파일 목록·링크 금지·전체 바이트·해시·출처·요청 ID·자료형·행 수·NPY 헤더와 크기를 검증한다. 결과의 통계적 타당성이나 원본과 의미적 동등성을 인증하지 않는다.

등록 변환기의 전체 결과 바이트는 `--max-output-bytes`에 포함한다. 이 상한은 악성 pickle 코드의 임의 디스크 쓰기 전체를 제한하는 OS 디스크 할당량은 아니다. 격리의 권한·네트워크 제한과 Job 자원 제한을 함께 유지한다.

표 복원 코어는 `src.preprocessing.canonical_data.load_table(directory)`다. canonical에서 복원된 표를 현재 CSV 전용 승인·EDA 경로에 자동 연결하지는 않는다. 일반 표·배열의 후속 입력 어댑터와 역할 프로파일 연결은 후속 범위다. WM canonical은 아래 별도 승인 CLI로 기존 WM 품질·EDA 코어에 연결한다. 변환 자체는 결측 대치·스케일링·모델 학습이 아니다.

기존 `wm_pickle_*` 경로는 호환 진입점으로 남기고 실제 구현은 `pickle_*`에서 관리한다. 새 실행 로그는 `logs/agents/pickle_import/`에 저장한다. 과거 `wm_pickle_import` 실행 기록은 자동 이동·재개하지 않는다. 기존 WM 전용 예제에서 새 기본값으로 WM 합성을 원하면 `create-demo --demo-kind wm`을 사용한다.

## 구현 위치와 검증

- `src/preprocessing/canonical_data.py`: 에이전트·WM에 의존하지 않는 표·배열 변환·검증·복원 코어.
- `src/agents/pickle_converters.py`: 구조 검사와 등록 변환기 조립, WM 기존 변환기 연결.
- `src/agents/pickle_contract.py`: 자동 선택·명시적 경로·자원·출력 계약.
- `src/agents/pickle_import_cli.py`, `pickle_import.py`: 생성·승인·실행 관리·검토 중단·커밋.
- `src/agents/pickle_import_worker.py`, `pickle_import_transport.py`: 기존 AppContainer·Job 실행.
- `src/agents/pickle_results.py`: 부모 측 제한된 결과 검증.

```powershell
python -m unittest tests.agents.test_pickle_conversion tests.agents.test_wm_pickle_import tests.test_wm_canonical_export tests.test_validate_canonical -v

$env:WAFERGUARD_TEST_APPCONTAINER = '1'
python -m unittest tests.agents.test_wm_pickle_appcontainer -v
Remove-Item Env:WAFERGUARD_TEST_APPCONTAINER
```

테스트는 고정 합성 입력만 사용한다. 실제 WM 원본은 이 변경 검증에서 다시 읽거나 변환하지 않는다. 데이터 값·구조 보고서·상세 결과는 로컬 전용이며 LLM으로 보내지 않는다.

### 2026-10-17 검증 기록

- 관련 합성 회귀 테스트 44개 통과: 표 자료형·결측·인덱스 보존, 숫자 배열, 자동/명시적 선택, 모호한 구조 중단, 승인 변경 거부, 변조·출력 상한, 기존 WM 변환 포함.
- 실제 Windows AppContainer 검사 3개 통과: 합성 표·숫자 배열·WM 맵의 변환·부모 검증·커밋·종료 정리 확인.
- 전체 에이전트 검사 실행에서는 397개 중 실패 1건·선택 실행 생략 4건을 확인했다. 실패는 이번 변경 대상이 아닌 `src/preprocessing/metadata_registry.py`의 `src.agents.module_catalogue` 참조로 인한 의존성 방향 검사다. 공통 코어→에이전트 역참조 정리는 별도 후속 점검이 필요하며 전체 통과로 기록하지 않는다. 위 관련 44개는 이후 추가한 명시적 선택·검토 보고서 변조 검사를 포함해 별도로 다시 실행했다.

후속 수정: 같은 날짜에 [등록부 의존성 방향](../design/METADATA_REGISTRY.md)을 정리했다. 명세·소스 검사는 에이전트 바인더에서 수행하고 공통 코어는 검토 계약만 전달받는다. 관련 회귀 41개에서 기존 의존성 실패가 해결되었음을 확인했다. 위 397개 기록은 수정 전 검사 이력이다. 이 수정 자체가 WM canonical의 격리 EDA 연결 완료를 뜻하지는 않는다.

수정 후 전체 에이전트 검사: 403개 중 399개 통과·선택 실행 4개 생략. 실패는 없으며 위의 의존성 방향 실패는 해소되었다.

## WM canonical의 승인·격리 품질 검사와 EDA

2026-10-17 후속 구현: `wm_canonical_cli`는 이미 변환된 WM canonical을 읽는다. pickle 재변환 없이 기존 `wafer_dl.validate_canonical.validate_bundle`과 `wafer_dl.eda.run_eda`를 재사용한다. 기존 CSV용 `pipeline_approval_cli`의 지원 형식을 변경하거나 WM 구조를 CSV로 임의 변환하지 않는다. 현재는 별도 WM 실행 도구이며 공통 모듈 조합 CLI로 통합하는 것은 후속 범위다.

```text
canonical 선택 → 전체 파일의 바이트 snapshot·지문 고정
  → preflight → 사용자 승인 → AppContainer + Job
  → 기존 canonical 정합성·중복·분할 사전조건 검사
      ├─ 통과: 기존 WM EDA → 부모 결과 검증 → 완료 등록
      └─ 검토 필요: 품질 보고서 보존 → 중단, EDA 미실행
```

### 합성 입력 확인

```powershell
python -m src.agents.wm_canonical_cli create-demo `
  --name wm-canonical-demo01 --memory-mib 2048 --processes 2 --wall-seconds 120

python -m src.agents.wm_canonical_cli next --name wm-canonical-demo01
python -m src.agents.wm_canonical_cli next --name wm-canonical-demo01
python -m src.agents.wm_canonical_cli approve --name wm-canonical-demo01

# 표시된 WM-문구를 직접 입력한 후 발급된 ID를 사용합니다.
python -m src.agents.wm_canonical_cli next `
  --name wm-canonical-demo01 --approval-id 발급된_ID

python -m src.agents.wm_canonical_cli status --name wm-canonical-demo01
```

`status`의 `local_quality_report`와 `local_eda_report`를 로컬에서 연다. 정상 합성 예제는 클래스마다 독립 연결 그룹 3개를 가진다. `create-demo --review-demo`를 새 이름으로 실행하면 독립 그룹 부족으로 `TOOL_REVIEW_REQUIRED` 중단을 확인할 수 있다. 보고서가 존재한다는 것과 작업 완료·학습 준비 완료는 구분한다.

### 기존 canonical 선택

아래 경로·한도는 예시다. 실제 canonical 폴더를 확인하고 전체 파일 크기·필요 메모리·시간에 맞춰 지정한다. 원본 pickle 파일이나 분할 후 출력 폴더를 지정하지 않는다.

```powershell
python -m src.agents.wm_canonical_cli create `
  --name wm-canonical-eda01 `
  --canonical-dir data/processed/wm811k/canonical `
  --max-input-bytes 8589934592 --max-output-bytes 67108864 `
  --memory-mib 8192 --processes 2 --wall-seconds 1800 `
  --confirm-selection
```

이후 `next → next → approve → next --approval-id → status`는 합성 예제와 같다. 원본과 별도로 실행용 snapshot과 worker용 복사본을 만들므로 임시 디스크 여유도 필요하다. 생성 단계는 파일을 바이트로만 복사·해싱하며 NPZ 배열·manifest 내용의 품질 분석은 승인 후에 한다.

기본은 통계 EDA만 생성한다. 클래스별 예시 맵 그림이 필요하면 생성 시 `--include-examples`를 명시한다. 승인에는 이 설정, 파일 지문, 코드, 자원, 입출력 상한, `--max-group-ratio`(기본 0.2)가 함께 결합된다. 정합성 검증 자체는 예시 옵션과 무관하게 모든 승인 NPZ를 숫자 배열로 검사한다. 예시 옵션은 EDA 그림을 위한 추가 맵 읽기를 허용하는 설정이지 NPZ 검사를 생략하는 옵션이 아니다.

기존 canonical에 맵 라벨 충돌·부족한 클래스 그룹이 있으면 파일 정합성이 정상이어도 검토 중단할 수 있다. `validation.json`·`validation.md`를 확인하고 별도 정책을 결정한다. 자동 제외·라벨 수정·검토 해제·EDA 강제 진행은 제공하지 않는다.

### 결과·권한·한도

- 품질 결과: `analysis/quality/validation.json`, `validation.md`.
- 통과 시 EDA: 클래스·상태·크기 집계, 다이 통계, Pearson/Spearman·클래스별 상관 표와 그림, 선택한 경우 예시 맵.
- 부모 검증: 요청·승인 결합, 파일 목록·전체 크기·해시, JSON/CSV 구조, PNG 청크·CRC·크기, Job 종료·정리 증거. 통계의 수학적 정확성·원본 동등성 인증은 아니다.
- worker에는 승인된 복사본·고정 코드·출력·scratch만 전달한다. 원본 폴더·승인 DB 권한이나 네트워크 capability를 추가하지 않는다.
- 검증용 SQLite는 worker의 빈 scratch 폴더를 사용한다. Python 임시 폴더의 권한 재설정 대신 기존 상속 ACL을 유지한다. 기존 CLI·노트북 검증기의 기본 임시 폴더 동작은 유지한다.
- 그래프는 GUI 없는 Agg 백엔드로 생성한다. 분할·입력 변환·캐시·모델 학습·LLM 전송은 수행하지 않는다.
- `max-output-bytes`는 결과의 크기를 검사하고 초과·부분 결과의 완료 등록을 막는다. EDA 쓰기 전체를 OS에서 중간 차단하는 디스크 할당량은 아니다. 실패 시 worker scratch·부분 결과는 로컬 진단용으로 남을 수 있다.

### 소스와 검증

`src/agents/wm_canonical_contract.py`는 복사본·파일·승인 계약, `wm_canonical.py`는 실행 관리, `wm_canonical_cli.py`는 사용자 명령, `wm_canonical_worker.py`·`wm_canonical_transport.py`는 기존 격리 실행, `wm_canonical_results.py`는 부모 검증을 담당한다. `src/wafer_dl/validate_canonical.py`에 빈 호출자 작업 폴더를 받는 선택 인자 `workspace`를 추가했다. 입력 데이터·분석 알고리즘·클래스 정의는 변경하지 않았다.

관련 회귀 검사 74개 통과, 실제 AppContainer 합성 정상·검토 중단 검사 2개 통과. 전체 에이전트 검사도 418개 중 412개 통과·선택 실행 6개 생략으로 완료했고 실패는 없었다. 실제 WM 원본은 이번 검증에서 다시 읽지 않았다.

```powershell
python -m unittest tests.agents.test_wm_canonical tests.test_validate_canonical tests.test_wafer_eda -v

$env:WAFERGUARD_TEST_APPCONTAINER = '1'
python -m unittest tests.agents.test_wm_canonical_appcontainer -v
Remove-Item Env:WAFERGUARD_TEST_APPCONTAINER
```

## WM 맵 라벨 충돌의 가상 제외 후보

`wm_conflict_cli`는 기존 WM 품질 검사에서 `MAP_LABEL_CONFLICT`로 검토 중단한 결과에 연결한다. 기존 결과를 통과로 바꾸지 않으며, 실제 데이터 제외 정책도 승인하지 않는다. 충돌 맵 해시와 같은 모든 행을 가상 제외했을 때 클래스 표본·Lot·연결 그룹이 어떻게 달라지는지 보고서만 생성한다. 동일 해시의 미라벨 행도 후보에 포함한다.

```powershell
python -m src.agents.wm_conflict_cli create `
  --name wm-conflict-candidate01 `
  --quality-run wm-canonical-eda01 `
  --memory-mib 8192 --processes 2 --wall-seconds 1800 `
  --max-output-bytes 67108864 `
  --confirm-candidate-only

python -m src.agents.wm_conflict_cli next --name wm-conflict-candidate01
python -m src.agents.wm_conflict_cli next --name wm-conflict-candidate01
python -m src.agents.wm_conflict_cli approve --name wm-conflict-candidate01

# 표시된 CONFLICT-문구를 직접 입력한 뒤, 이번에 발급된 승인 ID로 실행한다.
python -m src.agents.wm_conflict_cli next `
  --name wm-conflict-candidate01 --approval-id 발급된_ID

python -m src.agents.wm_conflict_cli status --name wm-conflict-candidate01
```

이전 품질 검사 승인 ID는 소비된 것이므로 재사용하지 않는다. `--quality-run`은 canonical 경로가 아니라 기존 실행 이름이다. 입력·품질 결과·코드·설정이 달라졌다면 재개하지 않고 새 검사와 승인이 필요하다. 원본 검사 기록을 삭제하거나 수정해서 연결 검사를 우회하지 않는다.

### 실행과 결과의 의미

- 생성 시 기존 품질 실행의 snapshot을 참조한다. 새 worker 실행 시에는 별도 복사본을 전달하므로 snapshot 전체 크기 이상의 추가 디스크 여유가 필요하다.
- 실행 전후 전체 파일의 바이트 해시를 확인하지만, NPZ 배열을 다시 파싱하거나 pickle을 열지 않는다. 새 scratch SQLite에서 manifest만 진단한다.
- 이전 승인 소비·종료 증거·품질 보고서 해시와 현재 입력을 결합한다. 부모는 출력 목록·해시·행 위치 범위·전후 개수·기존 기준 집계와의 일관성을 검증한다. 통계의 수학적 정확성이나 원본 동등성을 인증하는 것은 아니다.
- 완료 상태의 `candidate_status: candidate_ready`는 **후보 보고서가 준비되었다**는 뜻이다. `data_modified: false`, `exclusion_approved: false`를 유지한다. 가상 제외 후에도 부족한 클래스나 큰 그룹 문제가 남을 수 있다.
- 로컬 결과: `candidate/diagnosis/diagnosis.md`, `diagnosis.json`, `class_comparison.csv`, `conflict_hashes.jsonl`, `excluded_rows.csv`, `candidate.json`.
- `excluded_rows.csv`의 `source_row_position`은 기존 manifest의 0부터 시작하는 행 위치다. 원본 pickle의 인덱스나 새 분할 ID를 뜻하지 않는다.
- 해시 목록·행 위치·상세 집계는 로컬 전용이다. LLM·외부 서비스로 전송하지 않는다.
- 실제 제외·라벨 변경·분할·EDA·캐시·모델 학습은 하지 않는다. 이후 정책을 결정해 별도 구현·승인 작업으로 진행해야 한다.

### 소스와 검사

- `src/agents/wm_conflict_cli.py`: 생성·승인·실행·상태 조회.
- `src/agents/wm_conflict_evidence.py`: 기존 품질 실행의 승인 소비·결과 증거 결합.
- `src/agents/wm_conflict.py`: 기존 실행 관리자·파일 복사·종료 증거 확인 재사용.
- `src/agents/wm_conflict_worker.py`, `wm_conflict_transport.py`: 기존 AppContainer·Job 기반 manifest 진단.
- `src/agents/wm_conflict_results.py`: 부모 측 제한된 출력 계약 검사.
- `src/wafer_dl/diagnose_split.py`: 기존 가상 제외 비교 알고리즘 재사용. 이번 연결에서 기존 WM 품질 검사 소스는 변경하지 않았다.

합성 기능 검사 8개와 실제 AppContainer 후보 검사 1개가 통과했다. 실제 격리 검사에서는 이전 품질 기록만 모의 OS 전송으로 준비하고, 새 후보 worker의 진단·정리·커밋은 실제 AppContainer에서 확인했다. 실데이터의 후보 진단은 자동 승인하거나 실행하지 않았다.

후속 전체 에이전트 회귀 검사는 427개 중 420개 통과·선택 실행 7개 생략, 실패 없음으로 완료했다. 기존 가상 제외 알고리즘 검사 3개도 별도로 통과했다. `wm-canonical-eda01`의 기존 품질 증거가 새 후보 기능에 연결 가능함을 읽기 전용으로 확인했다. Matplotlib·pyparsing의 폐기 예정 API 경고는 남아 있으며 테스트 실패는 아니다.

```powershell
python -m unittest tests.agents.test_wm_conflict -v

$env:WAFERGUARD_TEST_APPCONTAINER = '1'
python -m unittest tests.agents.test_wm_conflict_appcontainer -v
Remove-Item Env:WAFERGUARD_TEST_APPCONTAINER
```

## 승인한 충돌 후보로 새 canonical 생성

`wm_exclusion_cli`는 가상 제외 후 모든 분할 사전조건을 통과한 완료 후보만 받는다. 후보 진단 승인은 실제 제외 승인이 아니므로 **새 작업과 별도 사용자 승인**이 필요하다. 원본과 기존 품질·후보 실행은 보존하며, 새 결과 안에서만 충돌 맵 해시와 같은 모든 행을 제외한다. 라벨을 수정하거나 다수결로 정답을 만들지 않는다.

```powershell
python -m src.agents.wm_exclusion_cli create `
  --name wm-conflict-apply01 `
  --candidate-run wm-conflict-candidate01 `
  --memory-mib 8192 --processes 2 --wall-seconds 1800 `
  --max-output-bytes 4294967296 `
  --confirm-exclusion

python -m src.agents.wm_exclusion_cli next --name wm-conflict-apply01
python -m src.agents.wm_exclusion_cli next --name wm-conflict-apply01
python -m src.agents.wm_exclusion_cli approve --name wm-conflict-apply01

# 이번에 표시된 EXCLUDE-문구를 입력한 후 새 승인 ID를 사용한다.
python -m src.agents.wm_exclusion_cli next `
  --name wm-conflict-apply01 --approval-id 새로_발급된_ID

python -m src.agents.wm_exclusion_cli status --name wm-conflict-apply01
```

### 생성·재검증·출처

- 기존 snapshot을 새 worker 입력으로 복사하고, AppContainer·읽기 전용 입력 ACL·Job 한도를 그대로 적용한다. 원본·승인 DB 권한과 네트워크 capability는 추가하지 않는다.
- manifest에서 재계산한 충돌 해시·행 위치 목록의 바이트 지문을 완료 후보와 대조한다. 다른 행을 제외하거나 일부 충돌만 골라 제외하지 않는다.
- 남은 행의 위치·샘플 ID·NPZ 배열 키를 새 연속 번호로 갱신한다. `parent_source_row_position`·`parent_sample_id`로 바로 이전 canonical 행과의 대응을 보존한다. 새 행 번호로 과거 분할을 재사용하지 않는다.
- 남은 NPY 배열의 본문 바이트는 그대로 복사하고 NPZ 키만 갱신한다. 제외된 맵은 새 shard에서 제거하며 빈 shard는 만들지 않는다. 라벨·Lot·웨이퍼 메타데이터·맵 해시는 유지한다.
- 직접 입력은 부모 manifest이므로 새 provenance의 `source_kind`는 `parent_canonical_manifest`, `source_sha256`·`source_bytes`는 부모 manifest의 실제 지문·크기다. 원래 출처 해시·부모 전체 묶음 ID·후보 증거 ID도 별도 보존한다. pickle을 다시 읽지 않는다.
- 기존 `validate_bundle`로 새 manifest·모든 숫자 배열·맵 통계·해시·중복·분할 사전조건을 재검증한다. 검증 집계가 승인한 가상 제외 결과와 일치해야 완료 등록한다.
- 부모는 요청·승인 증거·종료·Job 정리·결과 목록·전체 크기·해시·검증 집계·출처 계약을 검사한다. 부모가 배열을 다시 파싱하거나 수학적 정확성·원본 동등성을 인증하는 것은 아니다.

### 완료 결과와 사용 범위

`status`가 `completed`이고 `derived_status`가 `derived_ready`인 경우에만 `local_canonical`을 다음 입력으로 사용한다. `local_report`는 새 canonical의 `validation.json`이다. 실패한 worker 폴더나 부분 결과는 사용하지 않는다. 내부 `SUCCESS.json` 하나만 보고 에이전트 작업 완료를 판단하지 않는다.

새 canonical은 완료 실행의 `attempts/<작업 ID>/derived/canonical`에 보존된다. 기존 경로를 덮어쓰거나 자동으로 다른 폴더에 게시하지 않는다. 결과 경로를 다음 `wm_canonical_cli create --canonical-dir`에 지정하면 별도 승인 후 품질·EDA를 수행할 수 있다. 이번 작업은 EDA·분할·캐시·모델 학습을 실행하지 않으며 `training_ready: false`를 유지한다.

출력 상한은 새 canonical과 검증·결과 JSON을 포함한다. canonical 쓰기는 기존 제한 writer로 증가 바이트를 검사하고, 전체 결과도 완료 전 다시 크기를 검사한다. scratch DB·runtime 복사본에 대한 OS 디스크 할당량은 아니다. worker 입력·파생 결과·부모 완료 결과 복사본이 각각 필요하므로 원본 크기의 수 배와 runtime을 보관할 디스크 여유를 확보한다.

### 소스와 검사

- `src/wafer_dl/exclude_conflicts.py`: 검토한 선택 지문 대조·원본 보존·새 행/키 구성·NPY 바이트 복사·출처 기록.
- `src/agents/wm_exclusion_evidence.py`: 완료 후보와 기존 증거 연결.
- `src/agents/wm_exclusion_cli.py`, `wm_exclusion.py`: 별도 승인·상태·실행 관리.
- `src/agents/wm_exclusion_worker.py`, `wm_exclusion_transport.py`: 기존 격리 실행과 전체 재검증.
- `src/agents/wm_exclusion_results.py`: 부모 측 출력 계약 검사.

실제 AppContainer 합성 제외 적용 검사 1개가 통과했다. 이전 품질·후보 기록만 모의 OS 전송으로 준비하며, 새 canonical 생성·전체 숫자 배열 재검증·Job 정리·부모 검증·커밋은 실제 AppContainer에서 수행했다. 실제 WM 데이터의 제외 작업은 자동 승인하거나 실행하지 않았다.

제외 적용·기존 후보·품질 검사·진단·검증·exporter 관련 합성 회귀 검사 57개가 통과했다. 이 중 새 제외 적용 검사는 9개이며, 원본 배열·라벨·행 대응 보존, 별도 승인과 재실행 방지, 후보·출력 변조, 크기 초과, 배열 손상 재검증, 취소·Job 정리 실패, 기존 출력 덮어쓰기 거부, 복사 허용 목록의 부트스트랩을 확인했다. 이번 변경 후 전체 에이전트 전체집합은 다시 실행하지 않았으며 앞 절의 427개 기록과 구분한다. 기존 실데이터 후보의 연결 가능성은 읽기 전용으로 확인했다.

```powershell
python -m unittest tests.agents.test_wm_exclusion -v

$env:WAFERGUARD_TEST_APPCONTAINER = '1'
python -m unittest tests.agents.test_wm_exclusion_appcontainer -v
Remove-Item Env:WAFERGUARD_TEST_APPCONTAINER
```

### 2026-10-17 다중 shard 결과 목록 크기 수정

실데이터 적용 실행 `wm-conflict-apply01`은 worker 생성·전체 재검증·Job 정리에는 성공했지만 부모 결과 검사에서 실패했다. 결과 지문 목록 `result.json`이 107,394바이트로, 작은 IPC용 64KiB 판독 상한을 넘은 것이 원인이다. 합성 소수 shard 검사에서는 이 경계가 재현되지 않았다.

`wm_exclusion_results.read_result_record`에서 결과 파일 목록·영수증만 별도의 1MiB 제한으로 읽도록 수정했다. 요청·worker 응답·검증 집계 등 기존 IPC의 64KiB 제한은 그대로 유지한다. 중복 JSON 키·비표준 숫자·객체 이외 타입·1MiB 초과를 거부하며 전체 결과 크기·허용 파일 수·지문 검사는 유지한다. 영수증도 파일 목록을 포함하므로 같은 제한을 사용한다.

기존 실패 worker의 결과는 수정된 부모 검증기로 읽기 전용 확인을 통과했다. 이는 실패 실행의 완료 승격·출력 게시·새 승인 발급이 아니다. 실패 상태와 소비된 승인은 보존한다. 소스 지문이 달라졌으므로 `wm-conflict-apply02`처럼 새 이름으로 생성하고 **새 승인**을 받아 실행한다. 이전 승인 ID를 재사용하거나 실패 폴더의 파일을 수동으로 완료 결과에 옮기지 않는다.

회귀 검사에는 작은 합성 맵을 다중 shard로 내보내 64KiB를 넘는 결과 목록과 영수증을 만들고, 생성·부모 검증·커밋·상태 재조회까지 확인하는 경우를 추가했다. 별도 상한·엄격한 JSON·기존 IPC 제한 유지 검사도 추가했다.

수정 후 제외 적용 합성 검사 11개와 실제 AppContainer 합성 검사 1개가 통과했다. 전체 에이전트 전체집합은 이번 수정 후 다시 실행하지 않았다.

## 완료 EDA에 연결한 WM 그룹 분할 제안

`wm_split_cli`는 품질·EDA가 완료된 동일 canonical에서 **분할안만** 생성한다. 기존 `wafer_dl.split.build_groups`, `choose_assignment`, `verify_assignments`를 재사용한다. 기존 WM 분할 코어와 과거 결과는 변경하지 않는다. 제안 이후의 확정·분할 manifest 게시 기능은 이번 연결 범위가 아니다.

기본 설정은 `configs/wm811k/split.json`의 Train/Validation/Test 70%/15%/15%, seed 42, 후보 32개, 연결 그룹 검토 상한 20%다. 그룹을 쪼개지 않으므로 실제 표본·클래스 비율은 목표와 다를 수 있다. 클래스별 지원 그룹과 거대 그룹 조건을 통과한 동일 EDA 입력만 사용한다.

```powershell
python -m src.agents.wm_split_cli create `
  --name wm-group-proposal01 --eda-run wm-canonical-eda02 `
  --config configs/wm811k/split.json `
  --max-output-bytes 67108864 `
  --memory-mib 8192 --processes 2 --wall-seconds 1800 `
  --confirm-proposal-only

python -m src.agents.wm_split_cli next --name wm-group-proposal01
python -m src.agents.wm_split_cli next --name wm-group-proposal01
python -m src.agents.wm_split_cli approve --name wm-group-proposal01

# WM-SPLIT-문구를 직접 입력한 뒤 새 승인 ID로 실행한다.
python -m src.agents.wm_split_cli next `
  --name wm-group-proposal01 --approval-id 새로_발급된_ID
python -m src.agents.wm_split_cli status --name wm-group-proposal01
```

`wm-group-proposal01`을 이미 준비했다면 create·사전 검사를 반복하지 않고 approve부터 시작한다. 새 이름으로 만들 때만 위 전체 명령을 사용한다. 입력·EDA 증거·코드·분할 설정·자원 한도를 승인에 결합하고 변경 시 재개하지 않는다.

- AppContainer·읽기 전용 입력 ACL·Job 한도를 재사용한다. 부모는 원본 맵을 파싱하지 않는다. worker도 이 단계에서는 전체 파일을 바이트 해싱하고 manifest만 분석하며 NPZ 배열은 파싱하지 않는다. 배열 검사는 앞선 완료 EDA의 품질 증거와 같은 파일 지문에 결합된다.
- 지도학습 표본뿐 아니라 남아 있는 미라벨·격리 행의 Lot·동일 맵 연결도 그룹 구성에 반영한다. 충돌이 다시 발견되면 중단하며 추가 자동 제외는 하지 않는다.
- 고정 후보의 표본·클래스 비율 목적함수로만 비교한다. 모델·성능·Test 예측을 이용해 후보를 선택하지 않는다.
- 결과 `proposal.json`은 설정·후보 비교·선택 결과·클래스 집계, `groups.csv`는 그룹별 단일 배정과 클래스 수, `proposal.md`는 로컬 검토 표다. 미라벨 전용 그룹의 빈 배정은 학습 사용 허가가 아니다.
- 부모는 요청·EDA 결합·파일 목록·상한·해시·그룹 ID 중복·집계·후보 선택과 목적함수 일관성을 확인한다. Lot·맵 연결은 고정 worker의 기존 코어가 검사하며 부모가 독립적으로 전체 원본을 재분석하는 것은 아니다.
- 실행 관리 단계명은 공통 후보 단계인 `derive_preparation`이다. 작업 상태가 `completed`여도 **제안 생성 완료**만 뜻하며 `proposal_status: proposal_ready`, `split_complete: false`, `training_ready: false`를 유지한다. `split_manifest.jsonl`은 생성하지 않는다.
- 제안 보고서를 사람이 검토한 뒤 확정에는 별도 계약·승인·구현 연결이 필요하다. 이미지 변환·캐시·학습·LLM 전송은 하지 않는다. 상세 집계·그룹 목록은 로컬 전용이다.

소스: `src/agents/wm_split_cli.py`(명령·승인), `wm_split.py`(실행 관리), `wm_split_evidence.py`(완료 EDA 결합), `wm_split_worker.py`·`wm_split_transport.py`(격리 실행), `wm_split_results.py`(부모 검사). 기존 코어 `src/wafer_dl/split.py`를 재사용한다.

합성 제안 검사 6개, 기존 WM 분할 코어·EDA 회귀 검사 20개, 실제 AppContainer 합성 제안 검사 1개가 통과했다. 실제 격리 검사에서는 이전 EDA만 모의 OS로 준비하고 새 제안은 실제 AppContainer에서 실행했다. 전체 에이전트 전체집합은 이번 연결 후 재실행하지 않았다. 실데이터 작업은 `wm-group-proposal01`로 준비했으며 사용자 승인·실제 제안 실행은 아직 하지 않았다.

## 검토한 WM 그룹 분할안 확정

`wm_split_commit_cli`는 완료된 제안의 `groups.csv`를 그대로 적용한다. 새 후보·seed 탐색, 추가 제외, 모델 학습은 하지 않는다. 원본과 제안 기록도 변경하지 않는다. 앞 절의 제안 전용 기능과 별도인 후속 확정 기능이다.

기존 실행기의 분할 보호 규칙을 유지하기 위해 `preflight → run_eda → review_eda → propose_split → commit_split` 계획을 사용한다. 이 계획의 `run_eda`·`propose_split`은 **완료된 EDA·제안 증거 참조**이고 재계산이 아니다. 각 참조 단계에서 선행 승인 소비·커밋·파일 지문·현재 코드와 입력을 다시 확인한다. `review_eda`는 보고서 검토 확인 승인, `commit_split`은 실제 manifest 생성 승인으로 각각 새 승인이 필요하다.

```powershell
python -m src.agents.wm_split_commit_cli create `
  --name wm-group-commit01 --proposal-run wm-group-proposal01 `
  --memory-mib 8192 --processes 2 --wall-seconds 1800 `
  --max-output-bytes 2147483648 --confirm-commit

python -m src.agents.wm_split_commit_cli next --name wm-group-commit01
python -m src.agents.wm_split_commit_cli next --name wm-group-commit01
python -m src.agents.wm_split_commit_cli next --name wm-group-commit01

# review_eda: 로컬 EDA·제안을 검토한 뒤 WM-REVIEW-문구를 직접 입력한다.
python -m src.agents.wm_split_commit_cli approve --name wm-group-commit01
python -m src.agents.wm_split_commit_cli next `
  --name wm-group-commit01 --approval-id 검토_승인_ID

# 기존 고정 제안 참조 후 commit_split 승인 대기로 이동한다.
python -m src.agents.wm_split_commit_cli next --name wm-group-commit01
python -m src.agents.wm_split_commit_cli next --name wm-group-commit01

# commit_split: WM-COMMIT-문구를 직접 입력해 별도 승인한다.
python -m src.agents.wm_split_commit_cli approve --name wm-group-commit01
python -m src.agents.wm_split_commit_cli next `
  --name wm-group-commit01 --approval-id 확정_승인_ID
python -m src.agents.wm_split_commit_cli status --name wm-group-commit01
```

이미 `wm-group-commit01`이 `review_eda` 승인 대기라면 create·사전 검사를 반복하지 않고 첫 approve부터 진행한다. 검토 승인과 확정 승인 ID는 서로 재사용하지 않는다. 두 승인 중 어느 것도 에이전트가 자동 발급하지 않는다.

### 저장·검증과 후속 입력

- 새 AppContainer·읽기 전용 입력 ACL·Job 한도에서 canonical과 검토한 제안의 복사본만 사용한다. 추가된 제안 폴더도 현재 허용 루트·지문에 결합한다. 원본·승인 DB의 권한을 추가하지 않는다.
- 그룹을 다시 구성해 `groups.csv`의 그룹 ID·클래스별 표본 수를 재대조하고 저장된 배정만 적용한다. 후보 비교 함수는 호출하지 않는다. Lot·맵의 중복 분할을 검사한다.
- 새 manifest에서 원본 행 순서·ID·라벨·배열 참조·원래 메타데이터는 유지하고 `split_protocol_id`, `split_group_id`, `group_split`, `exclusion_reasons`, `split`만 기존 분할 계약에 맞춰 기록한다. 미라벨·격리 행도 누락하지 않는다.
- 별도 SQLite와 기존 `wafer_dl.check_transform.index_split`으로 원본 모든 행과 다시 대조한다. 행 누락·추가·중복·라벨/배열 참조 변경·그룹 중복·부분 미배정·클래스 집계·프로토콜 불일치를 거부한다. 기존 입력 변환·캐시의 분할 형식과 프로토콜 공식을 유지한다.
- 이 단계에서 NPZ 배열은 파싱하지 않는다. 앞선 품질·EDA 완료 증거와 동일한 전체 입력 지문에 연결하며 `arrays_revalidated: false`, `resampled: false`로 기록한다. `arrays_revalidated: false`는 manifest 행 대응 재검증을 생략했다는 뜻이 아니다.
- 부모는 요청·승인·Job 정리·결과 목록·전체 크기·해시·고정 제안 집계·완료 표식을 확인한다. 큰 manifest는 스트리밍 해싱하고 제한된 작은 관리 JSON만 파싱한다. 원본 행 내용 재분석은 worker에 둔다.
- 결과는 완료 실행의 `attempts/<작업 ID>/split`에 보존한다. `split_manifest.jsonl`, `split.json`, `split_summary.csv`, `split_summary.md`, 내부 `SUCCESS.json`, `validation.json`, 결과 지문 `result.json`을 생성한다. 에이전트의 `AGENT_COMPLETE.json`은 모든 검증 뒤 부모가 별도로 생성한다. 내부 SUCCESS 하나만 보고 사용하지 않는다.
- `split.json`의 `canonical_root`는 보존된 EDA snapshot을 가리킨다. worker용 임시 경로를 가리키지 않는다. 해당 snapshot을 정리하거나 묶음을 이동하면 연결이 깨질 수 있으므로 삭제·이동하지 않는다. `data/splits`로 게시·전달하는 별도 기능은 아직 연결하지 않았다.
- `split_complete: true`는 분할 manifest 확정만 뜻한다. `training_ready: false`를 유지하며 이미지 크기 통일·채널 변환·캐시·학습은 수행하지 않는다. Test 배열도 읽지 않는다.
- 출력 상한 기본값은 2GiB다. 실데이터 원본 manifest가 약 708MB이므로 제안 보고서용 64MiB 제한을 확정 manifest에 재사용하지 않는다. 쓰기 증가 바이트를 제한하고 관리 JSON 공간도 예약한다. scratch DB·runtime에 대한 OS 디스크 할당량은 아니다.

소스: `src/wafer_dl/commit_split.py`(고정 배정 적용·기존 검증기 재사용), `src/agents/wm_split_commit_evidence.py`(완료 제안 결합), `wm_split_commit.py`·`wm_split_commit_cli.py`(두 승인·상태 관리), `wm_split_commit_worker.py`·`wm_split_commit_transport.py`(격리 실행), `wm_split_commit_results.py`(부모 결과 계약).

새 확정 합성 검사 7개와 실제 AppContainer 합성 확정 검사 1개가 통과했다. 기존 그룹 분할·이미지 변환 회귀 검사 16개도 통과했다. 실제 격리 검사에서는 이전 EDA·제안만 모의 OS로 준비하며 실제 확정 worker와 Job 정리·부모 커밋을 검사했다. 전체 에이전트 전체집합은 이번 연결 후 다시 실행하지 않았다. 실데이터 확정은 자동 승인·실행하지 않았으며 `wm-group-commit01`은 검토 승인 대기까지 준비한다.

## 확정 분할의 Train·Validation 변환·패턴 보존 검사

`wm_transform_cli`는 완료된 확정 분할에 연결해 Train·Validation의 지도학습 대상 맵만 검사한다. 기본 설정은 `configs/wm811k/transform.json`의 128×128, 종횡비 유지·0 패딩, 결정적 최근접 좌표 규칙, `defect/valid` 2채널, 증강 없음이다. 변환값은 맵별로 메모리에서 검사하고 지표·비교 그림만 저장한다. 전체 변환 캐시·Dataset·모델은 만들지 않는다.

```powershell
python -m src.agents.wm_transform_cli create `
  --name wm-transform-check01 --split-run wm-group-commit01 `
  --config configs/wm811k/transform.json `
  --memory-mib 8192 --processes 2 --wall-seconds 1800 `
  --max-output-bytes 268435456 --confirm-check-only

python -m src.agents.wm_transform_cli next --name wm-transform-check01
python -m src.agents.wm_transform_cli next --name wm-transform-check01
python -m src.agents.wm_transform_cli approve --name wm-transform-check01

# 표시된 WM-TRANSFORM-문구를 직접 입력한 후 이번 승인 ID를 사용한다.
python -m src.agents.wm_transform_cli next `
  --name wm-transform-check01 --approval-id 새로_발급된_ID
python -m src.agents.wm_transform_cli status --name wm-transform-check01
```

이미 승인 대기로 준비했다면 create·사전 검사를 반복하지 않고 approve부터 진행한다. 기존 분할 확정 승인 ID를 재사용하지 않는다. 실행 관리의 공통 단계명은 검사 후보를 뜻하는 `derive_preparation`이며 학습 입력 생성 완료를 뜻하지 않는다.

### 재사용·격리·Test 경계

- 기존 `transform_map`, `make_channels`, `preservation_metrics`, `index_split`, `iter_development_maps`, `save_examples`, `write_check_report`를 재사용한다. 과거 승인 기록의 코드 지문을 보존하기 위해 기존 `check_transform.py`는 수정하지 않았다.
- `wafer_dl/isolated_transform_check.py`는 기존 검사 순회 구조를 명시적 scratch·제한 CSV writer에 연결한다. Python 임시 폴더가 AppContainer의 상속 ACL을 바꾸는 문제를 피한다. 새 wrapper와 기존 코어의 행별 CSV 바이트·클래스 집계가 같은지 합성 회귀 검사로 확인했다.
- 확정 분할·canonical·변환 설정·코드·한도를 한 승인에 결합한다. canonical과 확정 분할은 읽기 전용 작업 복사본으로 전달한다. 분할 manifest는 검사 출력보다 클 수 있으므로 그 복사본 검사는 **승인된 입력 파일 크기 합계**를 사용하며 출력 256MiB 제한으로 입력을 자르지 않는다.
- 전체 파일 지문 확인을 위해 NPZ 파일 바이트는 해싱한다. Test는 분할 연결 검증에 필요한 메타데이터만 확인하고 **Test 배열을 복원하거나 변환하지 않는다**. 같은 shard 안에 Test가 있어도 해당 NPY 구성원은 읽지 않는다. 파일 바이트 해싱과 이미지 배열 해석을 구분한다.
- 작업은 기존 AppContainer·Job 자원 한도·네트워크 capability 제한을 사용한다. 부모는 원본 이미지·pickle을 복원하지 않으며 실행·결과 집계는 LLM으로 전송하지 않는다.

### 결과와 검토 필요

- `preservation.csv`: Train·Validation의 행별 크기·패딩 위치·불량/유효 픽셀 수·비율 변화·전체 소실 지표.
- `transform_check.json`, `transform_check.md`: 클래스별 집계와 설명.
- `examples.png`: 기존 함수의 클래스별 첫 사례와 크기·종횡비 극단 사례를 비교한 로컬 원본/변환 그림. 대표 표본·형태 보존 보증이 아니다. 이 비교 그림은 본 작업의 고정 출력이며 공통 계약의 선택형 EDA 예시 옵션과 별개다.
- 내부 `SUCCESS.json`, 요청에 결합된 `result.json`과 부모 영수증. 부모는 파일 목록·크기·해시·요청 결합·18개 클래스/구간 집계·행별 CSV 중복/구간/소실 개수·PNG 청크/CRC/크기를 확인한다. 원본 지표를 독립 재계산하거나 통계의 수학적 정확성·시각 형태 보존을 인증하지 않는다.
- 기존 비교 그림은 최대 높이 4,680px다. 다른 EDA의 4,096px 제한을 전역 변경하지 않고 이 출력에만 2,048×5,120px·8MiB 제한을 사용한다. 그래프는 Agg 파일 백엔드로 생성한다.
- 불량·유효 영역 전체 소실이 발견되면 `TOOL_REVIEW_REQUIRED`·`transform_review_required`로 중단한다. 보고서는 로컬에 보존하지만 `AGENT_COMPLETE.json`을 만들거나 자동 해상도 변경·재실행을 하지 않는다.
- 기계적 검사 통과 시 `completed`, `check_status: transform_check_passed`다. 그래도 `visual_review_required: true`, `training_ready: false`는 유지한다. **원본/변환 그림을 사람이 검토한 뒤 별도 승인을 받아야 캐시 생성으로 진행할 수 있다.** 이번 승인은 시각 검토 완료·캐시 생성 승인이 아니다.
- CSV는 쓰기 증가 전에 크기를 검사한다. 그림·고정 보고서는 저장 후 전체 크기를 검사하여 초과·부분 결과의 완료 등록을 막는다. scratch·그림 저장 전체에 대한 OS 디스크 할당량은 아니다. 실패한 worker 폴더나 내부 SUCCESS만을 학습 입력으로 사용하지 않는다.

소스: `src/agents/wm_transform_cli.py`(명령·승인), `wm_transform.py`(실행 관리·검토 중단), `wm_transform_evidence.py`(확정 분할 결합), `wm_transform_worker.py`·`wm_transform_transport.py`(격리 실행), `wm_transform_results.py`(부모 검사), `src/wafer_dl/isolated_transform_check.py`(기존 변환 코어의 격리용 연결).

새 합성 검사 7개와 기존 변환·분할 회귀 16개가 통과했다. 실제 AppContainer 합성 검사 1개도 통과했다. 같은 shard의 Test 배열 미복원, 소실 검토 중단, 원본 보존, 출력 변조, Job 정리 실패, 복사 허용 목록, 기존 코어와의 지표 동등성을 확인했다. 실제 격리 검사는 이전 확정 기록만 모의 OS로 준비하고 새 변환 검사·그림·Job 정리는 실제 실행했다. 전체 에이전트 전체집합은 이번 연결 후 다시 실행하지 않았다. 이후 사용자가 실행한 `wm-transform-check01`은 `transform_check_passed`로 완료했다. 그림의 시각 확인은 이 기계적 통과와 별개다.

## 확인한 변환 설정으로 Train·Validation 캐시 생성

`wm_cache_cli`는 완료된 `wm_transform_cli` 검사에 연결한다. `examples.png`의 형태 보존을 직접 확인한 뒤 승인한다. 기계적 통과만으로 사람이 그림을 확인한 것으로 처리하지 않는다.

캐시는 **128×128 범주형 uint8 맵**을 숫자 전용 `.npy` shard에 저장한다. 0/1/2 값을 그대로 보존하며 픽셀을 일반 연속값으로 스케일링하지 않는다. float32의 `defect/valid` 두 채널은 기존 `WaferDataset`이 읽을 때 생성한다. `cache.json` 형식은 기존 Dataset과 동일하다. 원본·기존 캐시는 덮어쓰지 않으며 Test 캐시·모델 학습은 이번 범위가 아니다.

```powershell
python -m src.agents.wm_cache_cli create `
  --name wm-development-cache01 --check-run wm-transform-check01 `
  --config configs/wm811k/cache.json `
  --memory-mib 8192 --processes 2 --wall-seconds 1800 `
  --max-output-bytes 8589934592 --confirm-development-cache

# preflight → 완료된 EDA·분할 증거 참조 → 그림 검토 승인 대기
python -m src.agents.wm_cache_cli next --name wm-development-cache01
python -m src.agents.wm_cache_cli next --name wm-development-cache01
python -m src.agents.wm_cache_cli next --name wm-development-cache01

# review_eda: 표시된 로컬 그림과 보고서를 직접 확인한 뒤 승인한다.
python -m src.agents.wm_cache_cli approve --name wm-development-cache01
python -m src.agents.wm_cache_cli next `
  --name wm-development-cache01 --approval-id 그림_확인_승인_ID

# 기존 분할 제안 참조 → 기존 분할 확정 증거의 재확인 승인 대기
python -m src.agents.wm_cache_cli next --name wm-development-cache01
python -m src.agents.wm_cache_cli next --name wm-development-cache01
python -m src.agents.wm_cache_cli approve --name wm-development-cache01
python -m src.agents.wm_cache_cli next `
  --name wm-development-cache01 --approval-id 분할_재확인_승인_ID

# prepare_data: 실제 캐시 생성은 별도 승인 후에만 실행한다.
python -m src.agents.wm_cache_cli next --name wm-development-cache01
python -m src.agents.wm_cache_cli approve --name wm-development-cache01
python -m src.agents.wm_cache_cli next `
  --name wm-development-cache01 --approval-id 캐시_생성_승인_ID
python -m src.agents.wm_cache_cli status --name wm-development-cache01
```

`run_eda`·`propose_split`·`commit_split`은 기존 완료 증거의 참조·재확인만 한다. EDA·그룹 배정·split manifest를 다시 생성하지 않는다. 기존 실행기의 `입력 준비 전 분할 확정` 보호 규칙과 worker의 등록 작업 목록을 변경하지 않고 `prepare_data`에서 캐시를 생성한다. 각 승인 ID는 그 단계의 새 승인에만 사용한다. 이미 작업이 만들어졌다면 create를 반복하지 않는다.

### 저장·한도·검증 경계

- `configs/wm811k/cache.json`의 shard 행 수·버퍼 바이트를 사용한다. 기존 `CacheWriter`, `validate_check`, 변환·패턴 소실 검사, 캐시 보고서 생성을 재사용하며 기존 승인에 묶인 코어 파일은 수정하지 않는다.
- 입력 복사·코드 해시·한도·확정 분할·변환 검사 파일을 승인에 결합한다. 작업 전용 canonical·split·검사 복사본만 AppContainer로 전달하고 원본·승인 DB 권한은 부여하지 않는다.
- 보수적 출력 예상 바이트가 사용자 상한보다 크면 새 작업 생성 전에 거부한다. 상한은 메모리 제한과 별개이며 사용자가 정확한 파일 크기를 맞출 필요는 없다. 기본 출력 상한은 여유를 둔 8GiB다. 실제 픽셀·색인·보고서의 총 크기도 확인한다. runtime·scratch·복사본·결과의 전체 디스크 OS quota를 의미하지 않는다.
- 기존 `CacheWriter`는 각 shard를 `allow_pickle=False`로 저장한 뒤 숫자 dtype·shape·범주·유효 영역을 재검증한다. 격리 wrapper는 명시적 빈 scratch를 사용한다. 버퍼·shard 크기 한도를 지키며 전체 float 채널을 RAM이나 디스크에 만들지 않는다.
- 부모는 pickle이나 원본 이미지를 복원하지 않는다. 결과의 제한 NPY 헤더를 파싱하여 uint8·C-order·고정 shape를 확인하고 픽셀 바이트가 0/1/2인지 검사한다. 색인은 확정 manifest와 SQLite로 대조하여 Train·Validation ID·라벨·행 위치·누락·중복·추가를 확인한다. 변환의 수학을 독립 재계산하거나 원본과의 시각 동등성을 보증하지는 않는다.
- 부모 검증 후 별도 시도 폴더에 복사·재검사·영수증·`AGENT_COMPLETE.json`으로 커밋한다. 부분 생성·Job 정리 실패·변조 결과는 학습 입력으로 승격하지 않는다.
- 완료 상태는 `cache_status: cache_complete`, `test_maps_read: false`, `training_ready: false`다. 다음 단계는 이 새 캐시의 Train·Validation Dataset 첫 배치 검사다. 이전 독립 전처리 캐시의 배치 통과 기록을 새 캐시에 자동 승계하지 않는다.
- Test 구간의 metadata와 전체 NPZ 바이트 지문은 연결 검증에 사용하지만 Test 배열은 복원하지 않고 Test `.npy`도 생성하지 않는다. 모든 파일·집계는 로컬에 보존하며 외부 LLM에 전달하지 않는다.

소스: `src/agents/wm_cache_cli.py`(명령·사용자 승인), `wm_cache.py`(계획·증거 참조·실행 관리), `wm_cache_evidence.py`(완료 변환 검사 결합), `wm_cache_worker.py`·`wm_cache_transport.py`(기존 AppContainer·Job 연결), `wm_cache_results.py`(숫자 저장 형식·행 연결 검증), `src/wafer_dl/isolated_cache.py`(기존 캐시 코어의 격리용 연결).

검증 기록: 새 캐시 승인·실행 검사 7개, 기존 캐시·Dataset 회귀 7개, 격리 코어 출력 동등성·예산·숫자 판독 검사 3개로 총 17개가 통과했다. 실제 AppContainer 합성 캐시 검사 1개도 별도로 통과했다. 모든 실데이터 실행을 검증했다는 뜻은 아니며 전체 에이전트 전체집합은 이번 변경 후 재실행하지 않았다.

실제 격리 검사에서 실행 관리자까지 불러오던 불필요한 import와 결과 검사기의 호스트 상위 경로 탐색을 제거했다. worker는 고정 출력 폴더 내부만 검사하며 원본·호스트 상위 폴더 권한을 넓히지 않았다. 부모의 전체 경로 검증은 유지한다.

실데이터 작업 `wm-development-cache01`은 `wm-transform-check01`에 연결하여 create만 완료했다. worker는 시작하지 않았으며 그림 확인·분할 재확인·캐시 생성 승인을 발급하거나 소비하지 않았다. 위 명령에서 create를 반복하지 않고 첫 next부터 진행한다.

### 시작 전 실패 후 재준비

진단 보완 재검증: 준비 단계 오류·정리·취소·정보 노출 방지 모의 검사 6개, 실제 AppContainer 합성 캐시 검사 1개, 실패 상태 조회·소비 승인 재실행 차단 합성 검사 1개가 통과했다. 앞서 수행한 전체 캐시 회귀 17개와 별도 기록이며 이번 진단 보완 후 전체 테스트 전체집합을 재실행한 것은 아니다.

사용자가 이후 실행한 `wm-development-cache02`는 `prepare_data / cache_complete`로 완료했다. Test 배열은 읽지 않았으며 새 캐시의 첫 배치 검사는 아래 별도 단계로 진행한다.

위 문단은 최초 준비 시점의 기록이다. 이후 `wm-development-cache01`은 사용자 승인 후 `prepare_data` 준비 과정에서 실패했고 승인도 소비되었다. `transport.json`에 launch가 없고 runtime·tools가 비어 있어 캐시 worker 시작 전 실패로 확인했지만, 당시 기록만으로 정확한 OS 예외는 확정하지 않았다. 중복 요청의 `EXECUTOR_BUSY`는 잠금 차단 결과이며 그 자체가 원래 작업의 실패 원인은 아니다.

후속 보완은 `wm_cache_transport.py`의 준비 단계별 고정 진단과 `wm_cache_cli.py status`의 제한된 진단 표시다. `RUNTIME_INVENTORY`, `PROFILE_CREATE`, `INPUT_ACL`, `RUNTIME_COPY` 등 발생 단계, `SETUP_ACCESS_DENIED` 등 고정 분류, 숫자 OS 코드만 저장한다. 예외 원문·데이터·원본 경로는 추가 기록하지 않는다. Job·ACL·네트워크 제한을 완화하거나 실패 시 자동 재실행하지 않는다.

기존 작업·소비 승인·부분 결과를 삭제하거나 완료로 승격하지 않는다. 코드 지문이 달라졌으므로 `wm-development-cache01`을 새 코드로 재개하지 않고 `wm-development-cache02`라는 새 작업으로 준비한다. 선행 `wm-transform-check01`과 확정 분할 소스는 변경하지 않았다. 새 작업에서도 그림 확인·분할 증거 재확인·캐시 생성은 각각 새 승인이 필요하다. 앞 절 명령의 작업 이름을 `wm-development-cache02`로 바꾸고, create가 이미 완료된 경우 첫 next부터 진행한다. 실제 캐시 실행은 사용자의 PowerShell에서 한 번만 시작하며 같은 작업을 다른 터미널이나 에이전트에서 동시에 시작하지 않는다.

## 완료 캐시의 Dataset·첫 배치 검사

`wm_dataset_cli`는 완료된 캐시의 승인 소비·커밋·OS 종료 증거와 파일/행 연결을 부모에서 확인하고, 캐시 복사본만 새 worker에 전달한다. 기존 `src/wafer_dl/dataset.py`의 `check_first_batch`·`WaferDataset`·DataLoader를 재사용한다. 기존 캐시·변환·분할 소스는 수정하지 않으므로 선행 승인 기록과 연결이 유지된다.

```powershell
python -m src.agents.wm_dataset_cli create `
  --name wm-dataset-check01 --cache-run wm-development-cache02 `
  --batch-size 32 --memory-mib 8192 --processes 2 --wall-seconds 900 `
  --max-input-bytes 8589934592 --max-runtime-bytes 8589934592 `
  --max-output-bytes 1048576 --confirm-development-only

python -m src.agents.wm_dataset_cli next --name wm-dataset-check01
python -m src.agents.wm_dataset_cli next --name wm-dataset-check01
python -m src.agents.wm_dataset_cli approve --name wm-dataset-check01

# WM-DATASET-문구를 직접 입력해 새 승인 ID를 받은 뒤 사용자 터미널에서 실행한다.
python -m src.agents.wm_dataset_cli next `
  --name wm-dataset-check01 --approval-id 새로_발급된_ID
python -m src.agents.wm_dataset_cli status --name wm-dataset-check01
```

작업이 이미 created 상태면 create를 반복하지 않고 첫 next부터 진행한다. 계획은 `preflight → validate_contract`이며 실제 첫 배치 검사는 별도 사용자 승인 후에만 실행한다. 완료된 캐시 증거를 참조할 뿐 EDA·분할·캐시를 다시 만들지 않는다. 캐시 생성용 승인은 재사용하지 않는다.

### 검사 범위와 결과

- Train·Validation 두 구간만 각각 검사한다. worker에는 원본 canonical, 원래 split manifest, Test 배열, 승인 DB를 전달하지 않는다. 캐시·원본을 수정하거나 재변환·학습하지 않는다.
- 각 구간의 전체 캐시 파일 해시·숫자 헤더·색인·행/클래스/분할 식별값을 확인한 뒤 순서대로 첫 배치만 읽는다. 정상 입력은 기본 배치 기준 `[32, 2, 128, 128]`, 이미지 `torch.float32`, 라벨 `torch.int64`다. 구간 표본 수가 배치 크기보다 적으면 그 표본 수만 사용한다.
- 기존 코어가 픽셀의 0/1 값, 불량 채널이 유효 채널 안에 포함되는지, 라벨 0~8 범위를 확인한다. CPU·shuffle=False·num_workers=0으로 고정하며 증강·모델·역전파·GPU 연산은 사용하지 않는다. Job의 프로세스 상한 2는 DataLoader worker 2개를 사용한다는 뜻이 아니다.
- 결과는 `dataset_check.json`, `dataset_check.md`, 내부 `SUCCESS.json`, 요청과 결합한 `result.json`이다. Tensor·샘플 ID·픽셀 배열·모델 파일은 출력하지 않는다. 부모는 정확한 구간별 shape·dtype·범위·요청/코드/입력 결합·파일 지문과 Job 정리를 확인하고 별도 영수증·완료 표식으로 커밋한다. Tensor 연산을 부모에서 독립 재실행한 것은 아니다.
- 성공 상태는 `dataset_status: dataset_batch_passed`, `development_inputs_ready: true`, `test_maps_read: false`, `model_trained: false`다. 이는 두 개발 구간의 입력 정합성·첫 배치 계약 확인 범위다. 전체 표본의 Tensor 내용·분류 성능·Test 준비 완료를 보장하지 않는다. 과거 캐시 보고서의 `training_ready: false`는 해당 생성 시점 기록이므로 덮어쓰지 않고 새 검사 결과에 연결한다.
- 실패 시 준비 오류는 고정 단계·분류·숫자 OS 코드, worker 오류는 `IMPORT_TORCH`, `CHECK_TRAIN`, `CHECK_VALIDATION` 등 고정 진단으로만 표시한다. 상세 데이터·예외 원문은 자동 노출하지 않으며 부분 결과를 완료 처리하거나 소비 승인을 재사용하지 않는다.

### PyTorch 런타임과 한도

- 이 검사 전용 고정 패키지 목록에 설치된 PyTorch·필수 의존성을 포함한다. 설치·다운로드·사용자 Python 모듈 실행·호스트 실행 fallback은 제공하지 않는다. CPU 검사이지만 현재 환경의 CUDA 포함 PyTorch 패키지 자체는 약 4.4GB, 기존 표준 런타임과 의존성을 합친 복사 목록은 약 4.8GB다. GPU 모델·가중치를 다운로드하거나 로드하는 것은 아니다.
- `max-runtime-bytes`는 설치 라이브러리의 복사 합계, `max-input-bytes`는 캐시 복사 합계, `memory-mib`는 실행 Job 메모리, `max-output-bytes`는 보고서 합계다. 서로 다른 제한이며 원본 크기에 정확히 맞출 필요는 없다. runtime·캐시 작업 복사본을 둘 디스크 여유도 필요하다. Job 시간 제한은 실행한 프로세스에 적용되며 부모의 준비·복사 시간을 포함한 CLI 전체 경과 시간 제한은 아니다.
- 런타임 파일을 크기·SHA-256 명세로 고정하여 승인에 결합한다. 명세는 최대 8MiB의 별도 파일로 제한하고 IPC JSON 64KiB 상한을 전역 확대하지 않는다. worker 요청에는 명세의 지문을 넣는다. 복사 중 성장·변경은 거부하고 출력 상한 초과·부분 복사는 완료로 등록하지 않는다.
- Conda 설치 라이브러리의 하드 링크는 신뢰하는 설치 환경의 읽기 전용 원본으로만 취급하고 별도 바이트 복사·해시 확인으로 분리한다. 사용자 데이터의 하드 링크 거부 규칙은 완화하지 않는다. 설치 파일명 `C++17.h`·공백 이름은 런타임 전용 경로 검사에서만 허용하며 상위 이동·절대 경로·예약 이름은 계속 거부한다.
- 새 AppContainer·입력/런타임 RX ACL·Job 한도·capability 없는 네트워크 정책·전체 프로세스 정리를 재사용한다. `CUDA_VISIBLE_DEVICES=-1`, 외부 장치 backend 자동 로드 비활성화와 CPU 데이터 경로를 사용한다. 입력·집계·보고서·Tensor를 외부 LLM이나 추적 서비스에 전달하지 않는다.

소스: `src/agents/wm_dataset_cli.py`(명령·승인·상태), `wm_dataset.py`(실행 관리), `wm_dataset_contract.py`(캐시 완료 증거·입력 계약), `wm_dataset_runtime.py`(고정 런타임·제한 복사), `wm_dataset_transport.py`·`wm_dataset_worker.py`(격리 실행), `wm_dataset_results.py`(부모 결과 검사). 기존 `src/wafer_dl/dataset.py`·`cache.py`는 변경하지 않았다.

검증 기록: 새 Dataset 승인·출력·실패 표시 검사 7개, 런타임 경로·명세·제한 복사 검사 3개, 기존 캐시·Dataset 회귀 7개로 총 17개가 통과했다. 별도로 실제 AppContainer·Job에서 설치된 PyTorch 런타임을 복사하고 두 개발 구간 첫 배치를 읽는 합성 검사 1개도 최종 코드로 통과했다. 해당 native 검사의 선행 완료 캐시 증거는 합성 대역이다. 실데이터 검사 통과나 에이전트 전체 테스트 전체집합 재실행을 뜻하지 않는다.

실데이터 작업 `wm-dataset-check01`은 `wm-development-cache02`에 연결하여 created 상태까지 준비했다. 승인 발급·소비·실데이터 worker 실행은 하지 않았다. 위 명령에서 create를 반복하지 않고 첫 next부터 진행한다. 준비 중 실행 도구의 임시 SQLite 폴더 권한 제한이 있었지만, 제한 밖에서 완료 증거를 재확인하여 작업 설정 생성만 마쳤다. 사용자 터미널의 승인·실제 검사와는 별개이며 기존 캐시·분할·변환 파일은 변경하지 않았다.
