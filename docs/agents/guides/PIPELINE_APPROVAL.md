# 범용 파이프라인 승인 및 실행

공통 승인 계층은 입력 형식별 바인더와 정적으로 등록된 실행 어댑터를 사용합니다. 사용자 입력으로 Python 모듈이나 import 경로를 선택하지 않습니다.

이 문서의 `tabular_csv_v1` 예제는 기존 모듈형 CSV 파이프라인 worker를 재사용합니다. 실행은 새 AppContainer와 Job 제한 안에서 수행되며, worker 산출물은 부모 프로세스가 요청 ID·파일 목록·결과 계약을 다시 확인한 뒤 기존 실행기에 커밋됩니다. 이 예제의 모듈은 worker 허용 목록의 기본 품질·EDA이며, 임의 사용자 Python 모듈이나 다른 형식을 실행하지 않습니다. 새 CSV v2의 canonical 진단 연결과 현재 지원 범위는 [DESIGN](../DESIGN.md#공통-표형-입력처리-확장)·[GUIDE](../GUIDE.md#표형-v2-지원-범위)를 따릅니다.

## 실행 흐름

1. `create`가 snapshot·프로파일·파이프라인·리소스 제한을 결합해 실행 바인딩을 저장합니다. 이때 worker는 시작하지 않습니다.
2. 첫 번째 `next`가 입력·계획 사전 검사를 합니다.
3. 두 번째 `next`가 `run_pipeline` 승인 대기 상태로 전환합니다.
4. `approve`는 화면에 제시한 작업 범위를 확인하고 1회성 승인만 기록합니다. 이 단계에서도 worker는 시작하지 않습니다.
5. `next --approval-id ...`가 현재 승인 범위와 코드를 다시 확인하고, 승인 토큰을 소비한 뒤 worker를 시작합니다.
6. 부모가 결과를 재검증하고 나서만 작업을 커밋합니다. 품질 결과가 검토 필요 또는 차단이면 해당 상태로 종료합니다.

## 예시 명령

```powershell
python -m src.agents.pipeline_approval_cli create `
  --name pipeline-review01 `
  --snapshot-dir <snapshot-directory> `
  --profile <profile.json> `
  --pipeline configs/agents/pipelines/tabular_eda_example.json `
  --memory-mib 2048 --processes 2 --wall-seconds 300 `
  --confirm-selection

python -m src.agents.pipeline_approval_cli next --name pipeline-review01
python -m src.agents.pipeline_approval_cli next --name pipeline-review01
python -m src.agents.pipeline_approval_cli approve --name pipeline-review01
python -m src.agents.pipeline_approval_cli next `
  --name pipeline-review01 --approval-id <approval-id>
python -m src.agents.pipeline_approval_cli status --name pipeline-review01
```

`<approval-id>`는 `approve`가 반환한 실제 ID로 바꿉니다. 승인 ID를 전달하지 않으면 실행되지 않습니다. 다른 입력 어댑터를 추가할 때는 입력 바인더와 실행 어댑터를 각각 고정 등록하고, 해당 worker의 요청·결과 계약과 OS 격리를 검증해야 합니다.

## 확인 범위

단위 테스트는 승인 전 미실행, 승인 ID 소비, worker 호출 순서, 바인딩 변경 거부를 확인합니다. 실제 Windows AppContainer에서 이 새 진입점을 통한 실행은 별도 합성 입력으로 확인해야 합니다. 성공 결과는 분석의 통계적 정확성이나 운영 배포 승인을 뜻하지 않습니다.
