# Train 전용 전처리 기준 후보 산출

구현일: 2026-10-17. 승인·커밋된 Train 구간을 기존 복합 전처리 코어에 연결했다. 이 CLI의 범위는 **후보 기준 계산·저장·부모 검증·커밋**까지다. Train·Validation 변환은 별도 적용 CLI에서 수행한다. 모델 학습을 수행하지 않는다.

후속 갱신: [별도 적용 승인·저장 기준 적용](PREPARATION_APPLY.md)을 추가했다. 이 문서의 후보 CLI는 여전히 후보 산출까지만 수행한다. 적용 CLI에서 완료 후보를 선택하고 새 승인을 받아야 한다. 사용자 Windows 실행으로 후보 산출과 별도 적용 완료를 확인했다.

## 구현 구조

| 파일 | 역할 |
| --- | --- |
| `src/agents/preparation_candidate_contract.py` | Train 완료 증거·규칙·자원·후보 파일 상한 결합, 저장된 내보내기 승인 범위 복원 |
| `src/agents/preparation_candidate.py` | 사전 검사·별도 승인·작업 복사·격리 실행·검증·후보 커밋 |
| `src/agents/preparation_candidate_worker.py` | Train 특징만 읽고 기존 Pipeline.fit으로 전처리 통계 산출 |
| `src/agents/preparation_candidate_transport.py` | 고정 코드 복사·AppContainer·Job 제한·정리 |
| `src/agents/preparation_candidate_results.py` | 제한된 JSON의 상태·규칙·순서·출처·행 수·크기·해시 검사 |
| `src/agents/preparation_candidate_report.py` | 로컬 변경 예정 표와 안내 생성 |
| `src/agents/preparation_candidate_cli.py` | create/next/approve/status/cancel |
| `src/preprocessing/transform.py` | 기존 PreparationRules·build_preparation·fit·상태 검증 재사용, 저장 크기 상한 옵션 추가 |
| `configs/agents/preparation_candidate_example.json` | 데이터셋과 독립적인 후보 규칙 예시 |
| `tests/agents/test_preparation_candidate.py` | Train-only·변환 금지·승인·상한·변조·취소·복사 패키지 합성 검증 |

현재 `module_pipeline_cli`의 비파괴 모듈 목록에 변환을 몰아넣은 것이 아니다. 중간 적용 승인 경계를 보존하는 별도 후보 실행 경로이며, 모듈 등록부의 제안 ID 등록과 저장 상태 적용 연결은 후속 범위다.

## 입력과 승인

분할 확정 scope JSON, 분할 실행 폴더, 완료된 내보내기 실행 폴더가 필요하다. 내보내기의 저장 execution_plan.json에서 정확한 ingest scope를 복원한 뒤 DB 승인 소비·예약·커밋·완료 해시를 읽기 전용으로 다시 확인한다. 저장 범위 복원 자체가 승인 발급은 아니다. 과거 내보내기 worker나 CLI를 다시 실행하거나 승인 기록을 수정하지 않는다.

분할 프로파일의 서명을 기존 확정 scope와 대조하고, 내보내기 기록과 Train 위치·역할·snapshot을 결합한다. Validation 계약을 후보 산출 계약으로 전달하면 거부한다. 임의 CSV에 train이라는 이름만 붙인 입력은 받지 않는다.

고정 계획은 `preflight → derive_preparation`이며 새 작업 ID를 정책 허용 목록과 필수 승인 목록에 추가했다. 일반 prepare_data의 선행 분할 조건을 삭제하거나 기존 보안 검사를 우회하지 않는다. 새 후보 작업은 별도의 확정 분할·Train 증거를 입력 계약으로 재검증한다. 승인 scope에는 Train 내용·증거·규칙·코드·자원·출력 상한을 결합한다.

이전 완료 산출물은 저장 계획·DB·완료 파일 증거로 확인한다. 코드 변경 후 과거 실행의 승인으로 재실행하는 것과는 다르다. 과거 실행이 아직 진행 중이라면 변경된 코드로 재개하지 말고 새 실행·승인을 사용한다.

## worker 처리 범위

worker에는 Train의 input.bin·snapshot.json 및 고정 설정만 복사한다. Validation·Test·원본 전체 CSV·분할 DB·승인 DB·원본 폴더는 전달하지 않는다. 부모는 선행 완료 검증을 위해 다른 구간의 파일 해시를 읽을 수 있지만 해당 값을 기준 계산에 사용하거나 CSV로 재파싱하지 않는다.

Train CSV를 제한된 파서로 검사한 뒤 특징 컬럼만 로딩한다. 라벨·ID·그룹·시간을 fit 입력에 포함하지 않는다. 기존 `build_preparation(...).fit(...)`으로 제거 목록·대치값·선택한 변환 기준을 계산한다. 여기서 fit은 모델 학습이 아닌 **전처리 기준 산출**이다. Pipeline.transform·prepare_splits는 호출하지 않는다.

기존 코어가 지원하는 median/mean/none, scaling, IQR clip, 범주화, 범주 대치·인코딩 기준을 설정으로 선택할 수 있다. 알고리즘을 중복 구현하지 않는다. 예시는 결측률 50% 초과 제거·상수 제거·median·scaling none·outlier none·binning none을 선택한다. 예시가 자동 적용되거나 모든 모델에 최적인 기준이라고 주장하지 않는다.

제거 목록도 아직 후보 상태일 뿐이다. 원본·구간 CSV를 덮어쓰거나 새로운 변환 CSV를 만들지 않는다. 각 JSON/Markdown 파일은 요청한 파일별 상한을 검사하며 상태 JSON은 쓰기 전 조각별 UTF-8 바이트 크기를 확인한다. 초과 시 잘라서 성공 처리하지 않는다. Job 메모리·프로세스 수·시간 제한은 별도로 적용한다.

## 결과와 검증의 의미

```text
attempt/
├─ candidate/
│  ├─ preprocessor.json  # 제거 목록·대치값·변환 기준, pickle 아님
│  ├─ candidate.json     # Train 증거·설정·상태 파일 해시·변경 범위
│  └─ candidate.md       # 제거 후보·사유·설정·개수 표
└─ AGENT_COMPLETE.json  # 부모 검증 뒤 실행기가 기록
```

`completed`는 후보 산출 작업이 완료됐다는 뜻이다. `candidate_status: candidate_ready`, `transformed: false`로 표시한다. 전처리 적용 완료·모델 학습 완료·배포 승인이 아니다.

부모는 JSON 스키마·상태 유한성·역할·Train 행 수·규칙·제거 사유·출력 순서/폭·파일 상한·해시·요청 응답·Job 정리 증거를 검사한다. 원본 CSV 파싱·fit·통계 재계산·pickle/joblib 로드는 하지 않는다. 기존 `_validate_state`는 자료형과 범위를 검증하는 순수 상태 검사로 재사용한다.

이 검사는 일반 데이터의 중앙값 정답을 독립적으로 인증하지 않는다. 정확한 위치의 Train만 사용하고 예상 통계와 일치하는지는 정답을 아는 합성 테스트로 확인한다. 악성 로컬 관리자가 DB와 파일을 함께 변조하는 위협에 대한 인증 무결성 보장은 별도 검토다.

후보·설정·통계·제거 목록은 로컬 전용이며 LLM에 전달하지 않는다. Markdown에는 원문 컬럼명과 설정의 특수 문자를 이스케이프한다.

## SECOM 실행 방법

기존 사용자 실행에서 확정한 `secom-split-check02`와 `secom-partition-export01`을 사용한다. 해당 실행을 초기화하거나 재실행하지 않는다. 관리자 PowerShell의 프로젝트 루트에서 실행한다.

```powershell
python -m src.agents.preparation_candidate_cli create `
  --name secom-prep-candidate01 `
  --split-run-dir logs/agents/snapshot_split/runs/secom-split-check02 `
  --split-scope logs/agents/snapshot_split/runs/secom-split-check02/split_scope.json `
  --export-run-dir logs/agents/partition_export/runs/secom-partition-export01 `
  --rules configs/agents/preparation_candidate_example.json `
  --max-candidate-bytes 4194304 `
  --memory-mib 2048 --processes 2 --wall-seconds 300 `
  --confirm-candidate-only

python -m src.agents.preparation_candidate_cli next --name secom-prep-candidate01
python -m src.agents.preparation_candidate_cli next --name secom-prep-candidate01
python -m src.agents.preparation_candidate_cli approve --name secom-prep-candidate01
```

설정을 확인한 뒤 화면의 `CANDIDATE-...` 문자열을 정확히 입력한다. 반환된 새 승인 ID로 실행한다.

```powershell
python -m src.agents.preparation_candidate_cli next `
  --name secom-prep-candidate01 `
  --approval-id "새로_반환된_approval_id"

python -m src.agents.preparation_candidate_cli status --name secom-prep-candidate01
```

정상 완료 후 local_report 경로의 candidate.md를 로컬에서 검토한다. 적용은 별도 `preparation_apply_cli`에서 새 승인을 받아 수행한다. 취소·실패·상한 초과·변조 시 부분 결과를 커밋하지 않으며 자동 재실행하지 않는다.

## 검증과 다음 단계

```powershell
python -m unittest tests.agents.test_preparation_candidate -v
python -m unittest discover -s tests/agents -q
```

합성 테스트는 실제 worker·승인 소비·예약·커밋을 사용하고 OS 운송만 모의한다. 이후 사용자 실행에서 `secom-prep-candidate01`의 `candidate_ready`와 `secom-prep-apply01`의 `applied`를 확인했다. **정확한 후보 상태 해시 + 적용 대상 구간**에 별도 적용 승인을 결합하며, 후보 산출 승인과 재사용하지 않는다. 실제 적용 결과의 확인 범위와 남은 작업은 [저장 기준 적용 문서](PREPARATION_APPLY.md#실제-실행-확인-2026-10-17)에 기록했다.
