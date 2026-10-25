# 저장 기준의 Train·Validation 적용

구현일: 2026-10-17. 완료된 전처리 후보의 정확한 상태를 별도 승인한 뒤 두 구간에 적용하는 CLI·AppContainer 운송·worker·부모 검증·커밋을 연결했다. 합성 테스트에서는 실제 복원·변환·승인 소비·커밋을 수행하고 OS 운송만 모의한다. 사용자 Windows 실행과 저장 결과의 계약 재검증까지 확인했다.

## 소스 구성

| 파일 | 역할 |
| --- | --- |
| `src/agents/preparation_apply_contract.py` | 과거 완료 후보 증거 복원·상태 해시·같은 분할의 두 구간 결합 |
| `src/agents/preparation_apply.py` | 승인 후 복사·격리 실행·부모 검증·결과 커밋 연결 |
| `src/agents/preparation_apply_worker.py` | 저장 JSON 복원·transform만 호출·행/메타데이터 보존 |
| `src/agents/preparation_apply_results.py` | 출력 CSV·행 계약·해시·정책·형식 검사, 원본 재파싱 없음 |
| `src/agents/preparation_apply_io.py` | 파일 상한을 지키는 JSON 저장·고정 보고서 안내 |
| `src/agents/preparation_apply_transport.py` | 고정 소스 복사·새 AppContainer·Job 자원 제한·정리 |
| `src/agents/preparation_apply_cli.py` | create/next/approve/status/cancel |
| `tests/agents/test_preparation_apply.py` | 승인 분리·재산출 금지·결과 일치·행 보존·실패/취소 검증 |

공통 정책에 `apply_preparation` 단계·작업을 추가하고 필수 승인 대상으로 등록했다. 기존 prepare_data의 선행 분할 조건을 삭제하지 않았다. 별도 계획은 `preflight → apply_preparation`이며, 완료된 후보·분할·내보내기 증거를 입력 계약으로 확인한다.

## 입력과 승인 범위

사용자는 완료된 후보 실행 폴더만 선택한다. 해당 후보의 저장 계획·선택 기록·정확한 derive_preparation scope·DB 승인 소비·예약·커밋·완료 파일을 재검증한다. 현재 코드로 후보를 재산출하거나 과거 도구를 재실행하지 않는다. 저장 범위 복원은 새 승인을 발급하는 기능이 아니다.

후보가 산출된 Train과 동일한 분할·내보내기에서 나온 Validation만 적용한다. 다른 Validation이나 Test 파일을 임의 경로로 받지 않는다. 적용 규칙 변경 옵션도 제공하지 않는다. 규칙을 바꾸려면 새 후보 산출과 검토를 거쳐야 한다.

적용 승인은 후보 증거 ID·정확한 preprocessor.json SHA-256·Train/Validation snapshot/행 위치·자원 제한·출력 파일 상한·코드 식별값에 결합된다. 후보 산출 승인은 적용 승인으로 재사용하지 않는다. 상태·입력·한도가 바뀌면 새 실행·새 승인이 필요하다.

## 격리 실행과 재산출 방지

worker에는 두 구간의 input.bin·snapshot.json, 승인된 preprocessor.json, 행 위치 기록과 고정 설정만 전달한다. 원본 전체 CSV·Test·승인 DB·원본 폴더 접근 권한을 추가하지 않는다.

기존 `load_preparation`으로 검증된 JSON 상태를 복원한 뒤 `pipeline.transform`만 호출한다. fit·fit_transform·prepare_splits는 호출하지 않는다. Validation의 분포·결측·범주를 이용해 중앙값·제거 목록·스케일·범주를 다시 계산하지 않는다.

새 범주가 나타나면 후보의 unknown_category 정책을 그대로 사용한다. error라면 중단하고 ignore라면 코어의 명시된 처리를 따른다. 적용 시 자동으로 정책을 완화하지 않는다. Train 파일이 만들어진 뒤 Validation에서 실패해도 두 구간 전체를 완료로 등록하지 않는다.

구간 snapshot 검사는 기존 scoped helper에 snapshot/train/validation의 고정 이름만 허용하도록 확장했다. 사용자 상대 경로를 받거나 허용 밖의 조상 경로를 worker에서 검사하지 않는다. 공통 CSV 파서는 입력 기본 1024컬럼 제한을 유지하며, 변환 출력 검증에서만 승인된 출력 폭을 상한으로 명시한다. 레코드 1 MiB 제한은 유지한다.

## 출력 계약

```text
prepared/
├─ train_features.csv
├─ validation_features.csv
├─ train_rows.json
├─ validation_rows.json
├─ application.json
└─ application.md
```

특징 CSV는 후보에 저장한 출력 특징 순서대로 수치 값만 담는다. 라벨·ID·그룹·시간은 특징에 섞지 않고 각 rows.json에 보존한다.

```json
{
  "row_positions": [0, 2],
  "metadata_columns": ["label", "id"],
  "metadata": [["-1", "0001"], ["1", "0003"]]
}
```

위 값은 형식 설명용 합성 예시다. features.csv의 i번째 행과 rows.json의 i번째 row_position/metadata가 대응한다. 행 삭제·재정렬을 하지 않는다. 메타데이터는 CSV에서 읽은 **원문 문자열**로 저장해 앞자리 0·인용·줄바꿈 표현을 보존한다. 라벨을 수치로 사용하려면 이후 학습 입력 로더에서 명시적으로 변환해야 한다. 이 단계에서 라벨 매핑·학습을 자동 수행하지 않는다. 역할이 없으면 metadata_columns는 빈 목록이고 각 행의 metadata도 빈 목록이다.

원본 구간의 내부 snapshot 이름은 여전히 input.bin이다. 전달할 변환 결과는 CSV 확장자를 사용한다. 결과는 새 logs 실행의 attempts 아래에 저장하며 원본·후보를 덮어쓰지 않는다. data/processed 게시나 Git 업로드는 자동으로 하지 않는다.

## 부모 검증과 한계

부모는 승인 후보·구간·상태 해시·파일 목록·크기·해시·출력 특징 순서·행 수·원본 위치 목록·메타데이터 구조·숫자 유한성·정리 증거를 검사한다. 원본 CSV 파싱·fit·transform·pickle/joblib 로드는 하지 않는다. numeric_imputation none인 해당 수치 출력만 빈 NaN 셀을 허용한다. 해시가 일치해도 무한대·행/컬럼 불일치는 거부한다.

이는 파일·형식·승인 계약의 정합성 검사이며 원본 대비 모든 변환 수치·라벨 값의 독립 인증은 아니다. 수치/행/라벨/ID 보존은 정답을 아는 합성 데이터와 저장 코어의 결과를 비교해 검증한다. 로컬 관리자가 DB와 파일을 함께 변조하는 위협에 대한 인증 보장은 별도 검토다. 결과·통계·라벨·식별자는 로컬 전용이고 LLM에 보내지 않는다.

출력 파일 상한은 CSV와 각 행 JSON에 개별 적용한다. 행 JSON은 추가로 8 MiB 이하, application.json은 64 KiB 이하로 제한한다. 원핫 출력은 특징/셀 상한과 Job 메모리 상한도 적용한다. 상한 초과 시 자르거나 일부 행만 적용하지 않는다. 실패·취소·자원 초과·검증 실패 시 부분 파일을 커밋하지 않는다.

## SECOM 실행 명령

기존 완료 후보 secom-prep-candidate01을 그대로 사용한다. 이전 분할·내보내기·후보 산출 단계를 재실행할 필요는 없다. 관리자 PowerShell의 프로젝트 루트에서 실행한다.

```powershell
python -m src.agents.preparation_apply_cli create `
  --name secom-prep-apply01 `
  --candidate-run-dir logs/agents/preparation_candidate/runs/secom-prep-candidate01 `
  --max-partition-bytes 33554432 `
  --memory-mib 2048 --processes 2 --wall-seconds 300 `
  --confirm-application-selection

python -m src.agents.preparation_apply_cli next --name secom-prep-apply01
python -m src.agents.preparation_apply_cli next --name secom-prep-apply01
python -m src.agents.preparation_apply_cli approve --name secom-prep-apply01
```

후보 보고서를 확인하고 화면의 APPLY-... 문자열을 입력한다. 반환된 **새 적용 승인 ID**로 실행한다.

```powershell
python -m src.agents.preparation_apply_cli next `
  --name secom-prep-apply01 `
  --approval-id "새로_반환된_approval_id"

python -m src.agents.preparation_apply_cli status --name secom-prep-apply01
```

정상 상태는 completed, application_status applied, transformed true, refit false, test_used false다. 이 결과는 저장 기준 적용 완료이지 모델 학습·성능 평가·배포 승인이 아니다.

```powershell
python -m unittest tests.agents.test_preparation_apply -v
python -m unittest discover -s tests/agents -q
```

## 실제 실행 확인: 2026-10-17

- 실행: `secom-prep-apply01`, 후보: `secom-prep-candidate01`.
- 사용자 실행 결과: `completed / apply_preparation`, `application_status: applied`.
- `transformed: true`, `refit: false`, `test_used: false`.
- 결과 위치: `logs/agents/preparation_apply/runs/secom-prep-apply01/attempts/d780afc7a7c14fcfae0344a5721d1b4d/prepared`.
- 저장 결과에 기존 `PreparationApplyTool.validate`를 다시 적용한 결과: `True`. worker를 다시 실행하거나 기준을 재산출하지 않았다.
- 출력은 위 출력 계약의 6개 파일과 일치한다. 특징 순서·행 연결·수치 형식·파일 해시·요청 및 운송 증거는 기존 검증기로 확인했다. 원본 대비 모든 수치의 독립 인증이나 모델 성능 평가로 확대 해석하지 않는다.

이번 입력은 기존 SECOM ML의 Train에서 기능 확인용으로 새 분할한 데이터다. 기존 ML 실험 결과·모델·Test를 교체하지 않는다. 제거 특징·사유·처리 기준은 후보 실행의 `candidate.md`, 적용 파일별 기록은 `application.json`에서 로컬로 확인한다.

## 통합 작업과의 관계

이 문서는 별도 승인 CLI의 적용 계약과 실행 방법을 설명한다. 후보·적용의 모듈 계획 연결과 통합 화면은 이후 구현됐으며, 현재 사용 순서는 [GUIDE](../GUIDE.md#통합-작업)를 따른다. 서비스 등록이나 화면 연결이 동의·적용 승인 경계를 생략하는 근거는 아니다.

결과 전달 묶음과 파일 활용은 [CSV_HANDOFF](CSV_HANDOFF.md)·[GUIDE](../GUIDE.md#결과-파일-사용)를 참고한다. 기존 실험 로그는 보존하며 자동 게시하지 않는다. 임의 사용자 코드 등록이나 전체 학습 기능까지 지원한다는 뜻은 아니다.
