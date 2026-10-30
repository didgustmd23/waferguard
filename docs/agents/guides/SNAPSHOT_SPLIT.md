# 복사본 기반 격리 분할 검증

기존 범용 분할 코어를 사용자 승인·격리 worker·확정 증거 저장에 연결한 CLI다. 합성 테스트는 실제 계산·승인 소비·예약·커밋·후속 내보내기 계약을 검사하며 OS 운송만 모의한다. 이 문서의 SECOM 예제와 새 통합 v2의 실제 확인 근거는 구분하며, 현재 검증 범위는 [GUIDE](../GUIDE.md#검증-상태와-남은-제한)를 따른다.

## 구성과 경계

| 소스 | 역할 |
| --- | --- |
| `src/agents/snapshot_split_cli.py` | 원본 선택·상한·복사본 생성·단계별 승인·상태·확정 범위 저장 |
| `src/agents/snapshot_split.py` | 고정 계획·선행 단계·승인·작업 복사·부모 검증·실행기 연결 |
| `src/agents/snapshot_split_worker.py` | 격리 품질 검사·EDA·분할 제안·동일 조건 재계산 후 확정 |
| `src/agents/snapshot_split_transport.py` | 고정 코드 복사·AppContainer·Job 자원 한도·종료 정리 |
| `src/agents/split_evidence.py` | 제안/배정 파일 구조 검사와 승인·커밋된 확정 증거 검사 |
| `configs/agents/profiles/secom_train_split_check.json` | 기존 SECOM Train 안의 기능 검증용 시간순 분할 프로파일 |
| `tests/agents/test_snapshot_split_execution.py` | 승인·검토 중단·변경 차단·범위 저장·내보내기까지 합성 검증 |

부모는 원본 바이트 복사·해시 확인만 수행하고 CSV 분석은 worker 안에서 한다. worker에 원본 폴더·승인 DB 접근 권한은 추가하지 않는다. 제한된 CSV 파서로 레코드/열/행 수를 검사한 뒤 표형 로더를 사용한다. 식별자·그룹·범주·시간은 문자열로 읽어 앞자리 0 등이 수치 추정으로 합쳐지지 않게 한다. 라벨 타입은 기존 프로파일 기준으로 검사한다.

실행 순서:

```text
preflight → audit → run_eda → review_eda → propose_split → commit_split
```

preflight 외에는 모두 별도 사용자 승인을 받는다. review_eda는 사람이 로컬 EDA를 확인했다는 승인 기록이며, LLM이나 코드가 해석의 적절성을 대신 승인하는 기능이 아니다. 확인 토큰은 현재 한 단계에만 적용된다. 제안이 review_required이거나 품질이 review/block이면 중단하고 자동 해제·삭제·재시드하지 않는다.

분할 알고리즘은 기존 `preprocessing.split.propose_split`을 재사용한다. 확정 worker는 승인된 제안 JSON의 해시를 확인하고 같은 원본·설정으로 다시 계산해 정확히 일치해야 저장한다. 부모의 구조·파일 검사는 원본 대비 수학적 정답의 독립 인증이 아니다. 검토된 계산 코드·격리 경계·합성 정답 테스트를 함께 전제로 한다.

## SECOM 실행 준비

관리자 PowerShell의 프로젝트 루트에서 `ml310` 환경을 활성화한다. 기존 ML Test가 아닌 `time_train.csv`만 입력으로 선택한다. 그 내부에 새로 생기는 train/validation/test는 **에이전트 기능 검증용 배정**이며 기존 ML 평가 구간을 대체하지 않는다. 모델 재학습·문턱 변경·결측 대치·특징 삭제·스케일링은 하지 않는다.

SECOM 프로파일은 원래 결측·상수 센서에 review 정책을 유지한다. 아래 명령의 `--allow-feature-quality-findings`는 **분할 연결 검증에 한해** 결측률 초과·전체 결측·상수 센서를 삭제하지 않고 ignore 수준의 발견 사항으로 보고하게 하는 명시적 선택이다. 자동으로 review 결과를 통과로 바꾸는 기능이 아니며, 생성 시 변경된 정책을 별도 저장 프로파일·계획·승인에 결합한다. 라벨·ID·시간의 block 기준과 구조 오류는 유지한다. 이 옵션을 원하지 않으면 생략하고 audit에서 검토 필요로 중단되는지 확인한다.

```powershell
conda activate ml310

python -m src.agents.snapshot_split_cli create `
  --name secom-split-check01 `
  --source data/splits/integrated/time_train.csv `
  --profile configs/agents/profiles/secom_train_split_check.json `
  --max-input-bytes 8388608 `
  --memory-mib 2048 --processes 2 --wall-seconds 300 `
  --allow-feature-quality-findings `
  --confirm-selection

# 플랫폼 사전 검사
python -m src.agents.snapshot_split_cli next --name secom-split-check01

# audit 승인 대기로 전환
python -m src.agents.snapshot_split_cli next --name secom-split-check01
python -m src.agents.snapshot_split_cli approve --name secom-split-check01
```

화면의 `SPLIT-...` 토큰을 직접 입력한 뒤 반환된 승인 ID로 현재 한 단계를 실행한다.

```powershell
python -m src.agents.snapshot_split_cli next --name secom-split-check01 --approval-id "화면에_반환된_승인_ID"
python -m src.agents.snapshot_split_cli status --name secom-split-check01
```

audit 통과 후 run_eda → review_eda → propose_split → commit_split 각각에서 `next`로 승인 대기 → `approve` → **새 승인 ID**로 `next`를 반복한다. run_eda 뒤에는 status의 local_reports 경로에서 EDA를 보고 review_eda를 승인한다. 제안 뒤에는 proposal.md·proposal.json을 로컬로 검토하고 commit_split을 승인한다. 한 승인 ID를 여러 단계에 재사용하지 않는다. 품질·분할 검토 상태는 설정을 검토한 뒤 새 이름·계획으로 진행하며 자동 재시도하지 않는다.

메모리·프로세스 수·실행 시간 제한은 각 worker 실행에 적용된다. 300초는 EDA의 590개 센서 상관 계산을 고려한 요청 예시이지 완료 보장이 아니다. 입력 상한은 파일 복사 상한이며 RAM 상한이 아니다. 출력과 통계는 로컬에만 보존하고 LLM에 보내지 않는다.

## 확정 후 내보내기

completed 이후 다음 명령으로 실제 DB·완료 파일과 연결된 commit_split 범위를 저장한다. 완료 전에 실행하면 거부한다. scope 파일은 승인 자체가 아니며 내보내기 CLI가 DB·예약·소비·파일을 다시 검증한다.

```powershell
python -m src.agents.snapshot_split_cli export-scope --name secom-split-check01

python -m src.agents.partition_export_cli create `
  --name secom-partition-export01 `
  --split-run-dir logs/agents/snapshot_split/runs/secom-split-check01 `
  --split-scope logs/agents/snapshot_split/runs/secom-split-check01/split_scope.json `
  --snapshot-dir logs/agents/snapshot_split/runs/secom-split-check01/snapshot `
  --profile logs/agents/snapshot_split/runs/secom-split-check01/profile.json `
  --max-partition-bytes 8388608 `
  --memory-mib 2048 --processes 2 --wall-seconds 300 `
  --confirm-selection
```

내보내기에는 **실행 폴더의 profile.json**을 사용한다. 명시한 센서 통지 정책이 반영되어 원래 템플릿과 서명이 다를 수 있기 때문이다. 이후 [내보내기 CLI](PARTITION_EXPORT.md)의 next 두 번 → approve → 승인 ID로 next → status를 수행한다.

## 통과 기준과 제한

### 2026-10-17 EDA 실패 보완

사용자 실행에서 audit은 완료됐지만 run_eda가 WORKER_FAILED로 종료됐다. 읽기 전용 확인 결과 종료 코드 2, 실행 시간 약 28.8초, 최대 Job 메모리 약 128.5 MiB였고 자원 초과 기록은 없었다. 종료 후 활성 프로세스 0·Job 정리·프로필 정리도 확인됐다. EDA 통계 CSV 7개는 생성됐지만 PNG·보고서 완료 표식은 없었다. 해당 부분 결과는 성공 산출물로 등록되지 않았다.

기존 EDA 전용 worker와 달리 새 분할 worker에 비GUI Agg 설정이 누락된 결함을 확인했다. snapshot_split_worker의 EDA 분기에서 pyplot 사용 전에 `matplotlib.use("Agg", force=True)`를 명시했다. 진단 단계도 CSV_LOAD에서 EDA로 전환해 이후 계산·렌더링 실패를 구분한다. 합성 테스트는 렌더러 지정 순서·실패 단계·완료 참조 없음과 전체 분할 연결을 검사한다.

기존 고정 진단에는 예외 원문이 없어 이 결함이 해당 네이티브 실패의 직접 원인이었다고 확정하지 않는다. 새 승인으로 실행해 확인해야 한다. 이전 실행·승인·부분 결과·상태는 변경하지 않는다. 코드 식별값이 바뀌므로 `secom-split-check02` 같은 새 이름으로 생성하고 단계별 새 승인을 사용한다. 실제 AppContainer를 비격리 실행으로 대체하거나 메모리 상한을 임의로 늘리지 않는다.

- 원본 보존·승인 전 미실행·한 승인 1회 소비·단계 생략 차단.
- 확정 제안/배정/출처/프로파일 정합성, 사용자 EDA 검토와 확정 승인 분리.
- Job 종료·활성 프로세스 0·한도 일치·프로필 정리 확인 후 커밋.
- 후속 내보내기에서 Train·Validation 구간 입력 계약 확인. Test 파일 출력 없음.

분할 제안은 입력 전체의 라벨·역할 정보를 사용한다. 내보내기는 입력 전체를 순회한다. 이는 모델의 학습·평가가 아니며 Test 미접근 모델 평가로 표현하지 않는다. 기존 ML Test를 입력으로 사용하지 않는 범위와 새 기능 검증용 test 배정은 구별한다.

이 실패 보완 당시에는 실제 Windows 실행 완료 근거를 확보하지 못했다. 이후의 실제 격리 실행 사례와 현재 검증 범위는 [운영 안내](../GUIDE.md#검증-상태와-남은-제한)를 따른다. 취소·실패·자원 초과 후에는 완료 표식을 만들지 않으며, 새 실행 이름과 승인이 필요하다. 로컬 관리자가 DB와 파일을 함께 변조하는 위협에 대한 인증 무결성 보장은 별도 검토 항목이다.

```powershell
python -m unittest tests.agents.test_snapshot_split_execution -v
python -m unittest discover -s tests/agents -q
```
