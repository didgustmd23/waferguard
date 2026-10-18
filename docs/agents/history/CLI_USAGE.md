# 전처리 에이전트 합성 데모 CLI

> 초기 구현·실행 검증 이력입니다. 현재 상태와 권장 진입점은 [에이전트 문서 안내](../GUIDE.md)를 따릅니다. 당시 명령과 결과는 이력 보존용이며 현재 계약으로 자동 재개된다는 뜻이 아닙니다.

현재 실행 명령은 실제 데이터와 독립된 합성 도구만 실행합니다. 추가된 `plan --profile`은 범용 데이터 프로파일의 검토된 단계 목록을 조회합니다. 실제 WM-811K·SECOM 입력 경로, pickle·NPZ·모델은 읽지 않습니다. 합성 데모는 `preflight → ingest → audit → validate_contract` 순서로 고정 JSON을 생성합니다. 여기서 `ingest`는 실제 원본 변환이 아닌 합성 파일 생성 단계입니다.

## 범용 프로파일 계획 조회

~~~powershell
python -m src.agents.cli plan --profile configs/agents/profiles/tabular_example.json
python -m src.agents.cli plan --profile configs/agents/profiles/wm811k_metadata.json
~~~

`planning_only=true`, `source_bound=false`로 표시하며 입력 데이터는 읽지 않고 실행 폴더도 생성하지 않습니다. 실제 원본에 결합된 승인·실행 계획과 구분합니다. 프로파일 설정·공통 입력·지원 범위는 [범용 설계](../design/GENERIC_PREPROCESSING_DESIGN.md)를 따릅니다.

프로젝트 루트에서 Python 환경을 활성화한 뒤 실행합니다. `logs/agents/demo_v1`은 존재하지 않는 새 폴더여야 합니다. 이미 사용했다면 다른 이름을 지정하세요.

## 1. 계획 조회와 실행 생성

~~~powershell
python -m src.agents.cli plan
python -m src.agents.cli init-demo --run-dir logs/agents/demo_v1
python -m src.agents.cli status --run-dir logs/agents/demo_v1
~~~

`plan`은 파일을 만들지 않습니다. `init-demo`는 상태·승인 DB·데모 표식·생성 감사 기록을 저장하며 기존 폴더를 덮어쓰지 않습니다. 실행 주체는 OS 계정으로 정하고 CLI에서 임의 actor 문자열을 받지 않습니다.

## 2. 단계 실행과 승인 대기

~~~powershell
python -m src.agents.cli step --run-dir logs/agents/demo_v1
python -m src.agents.cli step --run-dir logs/agents/demo_v1
~~~

첫 명령은 preflight를 완료합니다. 두 번째 명령은 ingest에서 `APPROVAL_REQUIRED`를 반환하고 `awaiting_approval`로 대기합니다. 한 번의 명령은 현재 단계 하나만 처리합니다.

## 3. 승인과 재개

~~~powershell
python -m src.agents.cli approve --run-dir logs/agents/demo_v1
~~~

대화형 터미널에서 실행·단계·도구·계획·합성 입력·출력 루트·영향·유효기간을 확인합니다. 화면에 표시된 일회성 `DEMO-...` 문구를 직접 입력해야 발급됩니다. 다른 입력은 거절로 처리하고 파이프 입력은 받지 않습니다. 확인 중 상태가 바뀌면 발급하지 않습니다. 승인 화면 자체는 작업을 실행하지 않습니다.

출력의 `approval_id`를 복사하여 다음 명령의 `발급된_ID` 자리에 넣습니다.

~~~powershell
python -m src.agents.cli resume --run-dir logs/agents/demo_v1 --approval-id 발급된_ID
python -m src.agents.cli step --run-dir logs/agents/demo_v1
python -m src.agents.cli step --run-dir logs/agents/demo_v1
python -m src.agents.cli status --run-dir logs/agents/demo_v1
~~~

`resume`은 정확한 승인을 소비하고 ingest 한 단계를 처리합니다. 이어지는 두 `step`은 audit와 validate_contract를 처리하며 마지막 상태는 `completed`입니다. 30분이 지난 승인이나 다른 작업의 승인은 사용할 수 없습니다. 실패·보안 잠금·불명확한 예약을 자동 해제하지 않습니다.

## 4. 취소

완료 전 실행을 취소할 때 사용합니다.

~~~powershell
python -m src.agents.cli cancel --run-dir logs/agents/demo_v1
~~~

취소 상태를 저장하고 해당 실행의 미사용 승인을 폐기합니다. 부분 산출물을 삭제하지 않습니다. 현재 CLI는 같은 실행의 단계가 OS 잠금을 보유한 동안 취소 명령을 거부하며 단계 반환 후 취소해야 합니다. 장시간 도구 실행 중 취소 요청·worker 정지는 후속 구현 범위입니다.

## 5. 로컬 파일과 검증 범위

| 경로 | 역할 |
| --- | --- |
| `state.json` | 현재 상태·단계·revision |
| `agent_demo.json` | 고정 합성 데모 계약 |
| `approvals.sqlite3` | 승인·예약·작업 진행 이력 |
| `execution_plan.json` | 최초 실행 계획·주체 연결 |
| `events.jsonl` | 로컬 감사 기록 |
| `executor.lock` | 같은 실행의 동시 진행 차단 |
| `attempts/<작업 ID>/` | 합성 결과와 `AGENT_COMPLETE.json` |

OS 계정 확인과 대화형 승인은 데모에서 구현했습니다. 제한 계정·ACL·DB 인증 무결성·과거 DB 교체 방어·격리 worker·실데이터용 승인 체계가 완료된 것은 아닙니다. 자세한 보완 항목은 [상세 설계](../AGENT_DETAILED_DESIGN.md)의 후속 검토를 따릅니다. 로컬 상태·승인 화면·감사 기록을 LLM 또는 외부 추적 입력으로 사용하지 않습니다.

~~~powershell
python -m unittest tests.agents.test_agent_cli -v
python -m unittest discover -s tests/agents -v
~~~

범용 계약·EDA·프로파일 계획을 포함한 최신 테스트 결과는 [범용 설계](../design/GENERIC_PREPROCESSING_DESIGN.md)의 검증 기록을 따릅니다. OS 심볼릭 링크 생성 권한 부족으로 해당 검사가 건너뛰어질 수 있습니다. 합성 테스트가 실제 데이터 전처리·OS 격리 검증을 대신하지 않습니다.
