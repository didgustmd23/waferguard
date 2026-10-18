# 변환 모듈 등록과 승인 계획

2026-10-17. 기존 EDA 등록부·worker는 변경하지 않고 별도 변환 등록부와 정적 승인 계획을 추가했다. 기존 완료 실행의 코드 식별값을 바꾸지 않으며 과거 작업을 다시 실행하지 않는다.

## 소스

| 경로 | 역할 |
| --- | --- |
| `src/preprocessing/preparation_modules.py` | 후보 산출·저장 기준 적용의 고정 모듈 등록, 기존 코어 재사용 |
| `src/agents/preparation_module_plan.py` | 두 단계의 버전·설정·순서·새 승인 경계 검증, 기존 CLI 단계 매핑 |
| `configs/agents/pipelines/tabular_preparation_example.json` | 데이터 경로 없는 설정 예시 |
| `tests/agents/test_preparation_modules.py` | 재산출 방지·기준 재사용·승인 경계·예외 설정 검증 |

모듈은 `tabular_prepare_propose`와 `tabular_prepare_apply`, 버전은 각각 `1.0.0`이다. 전자는 Train에서 기준을 산출하고 후자는 검증한 후보 기준만 적용한다. 적용 설정은 비어 있어야 한다. 규칙을 바꾸려면 새 후보를 준비한다.

## 두 종류 계획의 구분

기존 `module_pipeline_v1`은 동일 CSV의 EDA·품질 검사를 한 작업으로 실행한다. 새 `preparation_module_pipeline_v1`은 후보 산출과 적용을 별도 승인 작업으로 선언한다. 새 설정을 기존 `module_pipeline_cli`에 전달하지 않는다.

```text
산출 승인 → derive_preparation → 커밋된 후보 확인·사용자 검토
                                         ↓
              새 적용 승인 → apply_preparation → 결과 검증·커밋
```

계획 생성만으로 데이터를 읽거나 worker를 실행하거나 승인하지 않는다. Train 표시를 신뢰하고 임의 파일을 실행하는 기능도 아니다. 실제 출처·분할·코드·자원·후보 해시 검증과 승인 소비는 기존 candidate/apply CLI가 담당한다. 정적 계획의 signature는 설정 변경 식별값이지 승인 증거가 아니다.

## 확인 방법

프로젝트 루트에서 아래 코드는 설정만 검증한다.

```powershell
python -c "import json; from pathlib import Path; from src.agents.preparation_module_plan import validate_preparation_modules; config = json.loads(Path('configs/agents/pipelines/tabular_preparation_example.json').read_text(encoding='utf-8')); plan = validate_preparation_modules(config); print(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2)); print(plan.signature)"

python -m unittest tests.agents.test_preparation_modules tests.agents.test_module_registry tests.agents.test_module_pipeline -v
```

관련 테스트 24개가 통과했다. 새 테스트는 합성 데이터만 사용하며 실제 AppContainer 실행은 하지 않았다.

## 구현 범위와 다음 연결

- 완료: 모듈 등록·기존 전처리 계산 재사용·설정 정규화·정적 단계 계획·CLI 매핑.
- 후속 구현: [통합 실행 CLI](../guides/PREPARATION_PIPELINE_EXECUTION.md)가 새 계획을 기존 candidate/apply 실행기에 연결한다. 후보 완료 후 review와 새 적용 승인은 별도로 유지한다. 실제 통합 AppContainer 실행은 사용자 확인 전이다.
- 유지: 후보 커밋 후 사용자 검토와 새 적용 승인. 한 번의 승인으로 두 단계를 실행하지 않는다.
- 제외: Test 적용·모델 학습·LLM 전송·사용자 Python 자동 import.

등록된 계산 함수를 개발 코드에서 직접 호출하는 것은 일반 로컬 코어 호출이다. 그 자체로 격리·승인·Train 출처 증거를 제공하지 않는다. 실제 에이전트는 기존 격리 worker 경로를 사용해야 한다. 메모리 결과 검증기는 구조만 확인하며 기존 부모 파일 검증기를 대체하지 않는다.
