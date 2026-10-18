# snapshot 기반 EDA 입력 계약

`src/agents/eda_input_contract.py`는 복사본 내용·크기·파일 상한·경로, 등록 어댑터 ID, 컬럼 역할, EDA 그래프 설정, worker 자원 요청을 하나의 로컬 계약으로 결합한다. 원본이나 복사본의 데이터 형식을 해석하지 않으며 승인 발급·worker 실행·LLM 호출을 하지 않는다.

## 연결 API

호출자는 `create_snapshot()`이 반환한 기대 `Snapshot`을 보존하고 검증한 `DatasetProfile` 및 사용자 지정 `WorkerLimits`와 함께 전달한다.

```python
from src.agents.eda_input_contract import bind_eda_input, EdaInputContract

# directory: input.bin·snapshot.json이 있는 복사본 폴더
# snapshot: 복사 시 반환받아 보존한 기대 계약
# profile: 검증된 DatasetProfile
# limits: 사용자 지정 WorkerLimits
contract = bind_eda_input(directory, snapshot, profile, limits)
payload = contract.to_json()  # 로컬 저장/IPC 전용
restored = EdaInputContract.from_json(payload)
restored.check_input()
plan = restored.execution_plan()
```

현재 EDA 연결 대상으로 등록한 어댑터는 `tabular_csv_v1` 하나다. snapshot 복사는 형식 독립적이지만 분석 요청은 등록된 형식만 허용한다. pickle 어댑터·임의 import·명령은 추가하지 않았다. `input.bin` 확장자로 로더를 추정하지 않는다.

계약 JSON은 64KiB로 제한하고 알 수 없는 필드·중복 키·비표준 숫자·잘못된 버전·어댑터를 거부한다. 기존 프로파일 파서를 재사용하여 역할과 설정을 검증한다. 실제 CSV 헤더와 컬럼 역할의 일치 여부는 후속 어댑터 실행에서 별도로 검사해야 한다.

## 계획·승인 결합

- `source_id`: 복사본의 SHA-256 기반 내용 식별값.
- `contract_id`: 복사본 크기·파일 상한·경로·전체 프로파일·자원 한도를 포함한 계약 식별값.
- `ExecutionPlan`: preflight → run_eda의 두 단계만 구성하는 연결용 계획. 프로파일의 전체 분할·준비 workflow를 실행하지 않는다.
- `config_id`에 contract_id를 사용하고 EDA 인자에 전체 계약을 넣어 기존 ApprovalScope의 설정·계획·인자 식별에 반영한다.

입력 변경 시 계획 생성 전에 복사본 재검증을 실패시킨다. 프로파일·자원 한도 변경 시 다른 계획·승인 범위가 된다. 계약 ID는 인증 서명이 아니며, 계약 생성만으로 승인이 발급되거나 기존 승인 소비가 발생하지 않는다.

manifest 내용만으로 기대 Snapshot을 자동 채택하지 않는다. 외부에 보존한 기대 계약과 비교한다. 계약 JSON에는 경로·컬럼명이 포함되므로 LLM DTO가 아니며 외부로 전달하지 않는다.

## 검증과 다음 단계

```powershell
python -m unittest tests.agents.test_eda_input_contract -v
```

합성 입력에서 JSON 왕복·계획 결합, 컬럼 역할·그래프 설정·시간 한도 변경 시 승인 범위 변경, 복사본·manifest 변조 거부, 미등록 어댑터와 잘못된 JSON 거부를 검증했다. 테스트는 worker를 실행하거나 실제 승인을 발급하지 않는다.

후속 구현으로 [snapshot EDA 승인 CLI·격리 worker](../guides/SNAPSHOT_EDA.md)를 연결했다. worker에 작업용 복사본·프로파일만 전달하고 부모에서 원본 재파싱 없는 결과 검증을 수행한다. 연결 테스트는 합성 입력과 모의 OS transport로 통과했으며, 별도로 snapshot-demo03의 실제 AppContainer 정상 실행과 로컬 기록 재검증을 확인했다. 실데이터 EDA는 별도 검증 대상이다. 기존 고정 합성 도구·모델·실데이터·완료 로그는 변경하지 않았다.
