# 변환 모듈의 통합 실행 관리

2026-10-17. `src/agents/preparation_pipeline_cli.py`가 모듈 계획을 기존 후보 산출·적용 CLI에 연결한다. 새 전처리 알고리즘·격리 worker를 만들지 않는다. 실제 데이터 worker는 사용자 승인을 받은 기존 AppContainer 경로로만 실행한다.

## 실행 구조

```text
통합 실행 생성: 계획·분할/내보내기 선택 → 후보 작업 생성
    ↓
next: 후보 사전 검사 → 승인 대기
approve: CANDIDATE 확인 → next --approval-id: 후보 산출
    ↓
status: awaiting_candidate_review
review: 후보 보고서 확인 → 적용 작업 생성(실행·승인 아님)
    ↓
next: 적용 사전 검사 → 승인 대기
approve: APPLY 확인 → next --approval-id: 저장 기준 적용
    ↓
status: completed / applied / refit false / test_used false
```

한 번의 next는 한 하위 단계만 처리한다. 후보 완료의 next 출력에서 completed는 후보 작업의 완료를 뜻하며, 통합 status는 review_candidate 대기를 표시한다. 전체 완료는 적용 작업의 커밋·기존 파일 검증까지 통과했을 때다.

## 보존·재개·중단

- 통합 실행은 `logs/agents/preparation_pipeline/runs/<name>`에 계획·정규화 규칙·선택·검토 기록을 보존한다.
- 하위 작업 이름은 `<name>-candidate`, `<name>-apply`다. 이름은 소문자로 시작하는 영문·숫자·밑줄·하이픈 1~32자다.
- 원래 계획 파일의 이후 변경은 저장된 계획을 바꾸지 않는다. 저장 계획·통합 코드·하위 선택 기록·실행 ID가 달라지면 재개를 거부한다.
- 각 명령은 기존 하위 계약을 다시 검증한다. 하위 DB·승인·커밋·자원 검사를 통합 기록으로 대체하지 않는다.
- 완료된 후보는 다시 산출하지 않는다. review는 해당 후보의 정확한 식별값을 적용 작업에 결합한다. 적용 설정은 바꾸지 않는다.
- 승인 ID는 각 approve 명령에서 새로 받은 값을 사용한다. 후보 승인으로 적용하려 하면 기존 실행기가 차단하며 실패 상태를 자동 초기화하지 않는다.
- 통합 명령은 OS 파일 잠금으로 동시 진행을 막는다. 취소는 잠금 밖에서 신호를 전달하고 하위 실행의 미사용 승인을 폐기한다. 이미 완료된 결과는 삭제하지 않는다.
- 생성·검토 기록 저장 중 중단된 불완전한 초기화는 자동으로 덮어쓰거나 채택하지 않는다. 하위 기록을 보존하고 새 이름으로 확인한다.
- 기록·해시는 로컬 관리자의 동시 변조를 막는 암호학적 인증이 아니다. 원본·통계·라벨·후보 상태·결과를 LLM으로 보내지 않는다.

## 사용자 SECOM 기능 검증 명령

기존 `secom-prep-candidate01`, `secom-prep-apply01`을 덮어쓰거나 다시 시작하지 않는다. 아래 명령은 같은 완료 분할/내보내기에서 **새 통합 기능 검증 실행**을 만드는 선택 사항이다. 관리자 PowerShell의 프로젝트 루트에서 실행한다.

```powershell
python -m src.agents.preparation_pipeline_cli create `
  --name secom-module-prep01 `
  --pipeline configs/agents/pipelines/tabular_preparation_example.json `
  --split-run-dir logs/agents/snapshot_split/runs/secom-split-check02 `
  --split-scope logs/agents/snapshot_split/runs/secom-split-check02/split_scope.json `
  --export-run-dir logs/agents/partition_export/runs/secom-partition-export01 `
  --memory-mib 2048 --processes 2 --wall-seconds 300 `
  --max-candidate-bytes 4194304 --max-partition-bytes 33554432 `
  --confirm-selection

python -m src.agents.preparation_pipeline_cli next --name secom-module-prep01
python -m src.agents.preparation_pipeline_cli next --name secom-module-prep01
python -m src.agents.preparation_pipeline_cli approve --name secom-module-prep01
```

화면에서 CANDIDATE 확인 문구를 입력하고 새 승인 ID로 실행한다.

```powershell
python -m src.agents.preparation_pipeline_cli next `
  --name secom-module-prep01 --approval-id "후보_산출용_새_승인_ID"
python -m src.agents.preparation_pipeline_cli status --name secom-module-prep01
python -m src.agents.preparation_pipeline_cli review --name secom-module-prep01
```

후보 보고서를 로컬에서 확인하고 REVIEW 문구를 입력한다. 이는 적용 승인이 아니며 적용 작업 준비만 수행한다.

```powershell
python -m src.agents.preparation_pipeline_cli next --name secom-module-prep01
python -m src.agents.preparation_pipeline_cli next --name secom-module-prep01
python -m src.agents.preparation_pipeline_cli approve --name secom-module-prep01
python -m src.agents.preparation_pipeline_cli next `
  --name secom-module-prep01 --approval-id "적용용_새_승인_ID"
python -m src.agents.preparation_pipeline_cli status --name secom-module-prep01
```

필요하면 `cancel --name secom-module-prep01`로 취소한다. Test 처리·모델 학습·결과 게시·사용자 코드 자동 등록은 포함하지 않는다.

## 검증

`tests/agents/test_preparation_pipeline.py`는 합성 분할·내보내기·실제 기존 코어·승인 DB·커밋을 사용하고 OS 운송만 모의한다. 전체 연결·단계별 승인 분리·검토 거부·취소·계획 변경·덮어쓰기 거부를 확인한다. 실제 통합 CLI의 AppContainer 실행은 사용자 확인 전이다.

검증 결과: 새 통합 테스트 9개를 포함한 에이전트 전체 346개 중 345개 통과, 1개 건너뜀. 실제 SECOM worker는 실행하지 않았고 기존 데이터·실험·완료 실행은 변경하지 않았다.

후속 사용자 실행 확인: `secom-module-prep01`에서 후보 산출·검토·별도 적용 승인을 거쳐 `completed / applied / transformed: true / refit: false / test_used: false`가 보고됐다. 적용 attempt는 `71750b1e36c94d3ea94154aa9a293c4f`다. 위 사용자 확인 전 표기는 구현 시점의 이력이다. 이후 결과 전달은 [CSV 전달 묶음](CSV_HANDOFF.md)으로 진행하며 전처리를 다시 실행하지 않는다.

```powershell
python -m unittest tests.agents.test_preparation_pipeline -v
python -m unittest discover -s tests/agents -q
```
