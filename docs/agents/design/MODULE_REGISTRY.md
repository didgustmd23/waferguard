# 내장 전처리 모듈 등록부

구현일: 2026-10-16. 계약·등록부 이후 [정적 검증](MODULE_PIPELINE.md)과 [CSV 분석·검사의 승인·worker 연결](../guides/MODULE_PIPELINE_EXECUTION.md)을 추가했다. 등록부 자체는 실행 승인 기능이 아니며 새 실행 진입점의 실제 OS 검증은 사용자 확인 전이다.

## 소스 역할

| 파일 | 책임 |
| --- | --- |
| `src/preprocessing/module_contracts.py` | 입출력 포트와 불변 모듈 계약. ID·버전·선행 조건·원본 보존·전처리 기준·권한 요구 선언 |
| `src/preprocessing/module_registry.py` | 계산·설정 검증·결과 검증 함수 연결, 중복 거부·정확한 버전 조회·등록부 봉인 |
| `src/preprocessing/builtin_modules.py` | 검토된 내장 기능의 고정 연결 |
| `tests/agents/test_module_registry.py` | 등록·조회·설정·보안 선언·결과·원본 보존 합성 테스트 |

코어는 에이전트·SECOM·WM 코드에 의존하지 않는다. `register()`는 신뢰된 Python 초기화 코드용이며 사용자 JSON·파일·LLM 응답으로 임의 코드를 등록하는 API가 아니다.

## 초기 목록

모든 버전은 `1.0.0`이다. `latest` 같은 자동 선택은 거부한다.

| ID | 기능 | 허용 설정 | 메모리 출력 |
| --- | --- | --- | --- |
| `eda_missing` | 결측 수·비율·관측 고유값 | 없음 | DataFrame |
| `eda_numeric` | 수치 기술 통계 | 없음 | DataFrame |
| `eda_categories` | 범주 빈도 | 없음 | DataFrame |
| `eda_target` | 선택적 라벨 빈도 | 없음 | DataFrame |
| `eda_correlations` | Pearson·Spearman·관측 수 | 없음 | 세 DataFrame의 dict |
| `tabular_quality` | 비파괴 품질 검사 | 기존 `QualityRules` 필드 | QualityReport |

`eda_*` 결과 계약 ID는 `<module_id>_result` v1, 품질 결과 계약은 `quality_report` v1이다. 파일 저장 계약과는 구분한다. 품질 설정은 기존 `QualityRules`의 기본값·타입·범위를 재사용하며 미지원 키를 거부한다. `allowed_labels` 배열은 복사된 tuple로 정규화한다.

## 개발 코드에서 조회

```python
from src.preprocessing.builtin_modules import build_builtin_registry

registry = build_builtin_registry()

for contract in registry.contracts():
    print(contract.module_id, contract.version, contract.kind)

quality = registry.resolve("tabular_quality", "1.0.0")
settings = quality.validate_settings({"missing_ratio_threshold": 0.5})
```

위 예시는 파일 읽기·분석 없이 목록·설정만 조회한다. 등록부 생성 후 봉인하므로 같은 객체에 모듈을 추가할 수 없다. 새 기능은 코드 검토 후 초기화 목록에 추가해야 한다.

## 보안·실행 경계

- 등록은 실행 승인·실데이터 활성화·OS 격리 통과를 뜻하지 않는다. 계산 함수 참조를 직접 호출하는 것은 일반 로컬 코어 호출이지 에이전트 승인 실행이 아니다.
- 선행 조건·승인·자원 항목은 선언이다. 다음 파이프라인·실행 관리자 연결 단계에서 실제 증거로 강제한다.
- EDA의 필수 입력 검사는 코어가 수행하며 사용자 설정으로 생략할 수 없다. 품질 검사는 잘못된 내용도 보고할 수 있도록 같은 오류를 진단 결과로 반환한다.
- 메모리 결과 검증기는 타입·형태·일부 범위·품질 상태 일관성을 검사한다. 원본 대비 계산 정답·출처·권한·규칙 적용의 완전성을 인증하지 않는다.
- 기존 worker 결과의 부모 파일·해시·코드·자원 검증기를 대체하지 않는다. 품질 결과가 구조 검증을 통과해도 `review_required`·`blocked_input`이면 후속 실행을 중단해야 한다.
- 별도 모듈 CLI·worker·복사 허용 목록에 내장 등록부를 연결했다. 기존 고정 CLI는 유지하며 모델 학습·사용자 모듈 자동 로딩·LLM 결과 검증은 추가하지 않았다.
- 과거 보고서·노트북·승인 DB·데이터는 변경하지 않는다.

## 다음 단계와 테스트

사용자가 ID·버전·설정을 선택하는 파이프라인 설정, 입력 포트 연결·필수 검사·순서·의존성 검증을 준비한다. 이후 기존 실행 관리자에 연결한다. 변환·보고서 모듈은 각각의 결과 계약·검증기를 준비한 뒤 등록한다.

```powershell
python -m unittest tests.agents.test_module_registry -v
python -m unittest discover -s tests/agents -q
```
