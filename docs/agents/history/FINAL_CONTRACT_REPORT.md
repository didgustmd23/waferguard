# 최종 입력 계약 검사와 로컬 보고서

> 초기 구현·실행 검증 이력입니다. 현재 상태와 권장 진입점은 [에이전트 문서 안내](../GUIDE.md)를 따릅니다. 당시 명령과 결과는 이력 보존용이며 현재 계약으로 자동 재개된다는 뜻이 아닙니다.

## 구현 흐름

```text
품질 검사 → EDA → 사람 검토 → 분할 제안 → 분할 승인·확정
    ↓
준비 승인 → Train-only fit·Train/Validation 변환·저장 검증
    ↓
validate_contract: 독립 최종 검사
    ↓ 통과한 경우만
render_report: 고정 한국어 템플릿의 로컬 보고서
```

범용 코어는 `src/preprocessing/prepared_contract.py`와 `report.py`다. 등록 도구는 `src/agents/adapters/finalize.py`에서 연결한다. 기존 prepare_data의 저장 검증은 유지하고 독립 검사를 추가했다. 두 검사 모두 복원된 변환기의 transform만 수행하며 fit을 다시 호출하지 않는다.

최종 검사와 보고서를 포함하는 템플릿은 `tabular_preparation_v1`이다. 기존 `prepare_review_v1`는 준비 단계까지만 수행하므로 바꾸지 않았다. [프로파일 예시](../../../configs/agents/profiles/tabular_prepare_example.json)는 새 템플릿을 사용한다. 기존 실행 계획의 템플릿을 중간에 변경하지 말고 새 계획·승인으로 시작해야 한다.

## 검사 항목

| 검사 | 확인 내용 |
| --- | --- |
| PREPARED_FILES | 필수 파일 구성, 비어 있는 파일·예상 밖 파일 거부 |
| TRANSFORMER_CONTRACT | 저장 변환기의 컬럼 역할·전처리 설정과 현재 프로파일 일치 |
| SPLIT_BINDING | 복사한 확정 분할과 현재 원본·설정·분할 규칙 일치 |
| PREPARATION_BINDING | 원본·설정·분할 식별값, Train 학습 행 수, 제거 목록·진단 기록 일치 |
| ROW_ALIGNMENT | 원본 행 위치·정답·순서의 정확한 연결 |
| TRAIN/VALIDATION_OUTPUT_SCHEMA | 행 수, 동일한 출력 컬럼·순서, float64 타입 |
| TRAIN/VALIDATION_VALUE_POLICY | 무한대 금지, 설정상 허용된 수치 출력에서만 NaN 허용 |
| TRAIN/VALIDATION_TRANSFORM_MATCH | 저장 CSV와 복원 변환기의 현재 transform 결과 일치 |

수치 결측 대치를 none으로 설정하고 수치 범주화를 하지 않은 출력에는 NaN을 허용한다. 원핫 출력·대치 완료 출력의 NaN은 거부한다. 설정 자체를 자동 변경하거나 전체 결측 특징을 임의로 보정하지 않는다. 실패 시 완료 요약 없이 고정 검사 코드만 로컬에 기록하고 보고서 단계로 넘어가지 않는다.

원본과 분할 메타데이터를 대조하므로 Test 데이터에 전혀 접근하지 않는 경로라고 표현하지 않는다. Test 특징 변환·모델 학습·성능 평가는 하지 않는다. 이 검사는 저장 산출물의 정합성 검사이며, 모델의 학습 이력 전체나 모든 도메인 누수 부재를 증명하는 검사가 아니다.

## 산출물

준비 단계에는 `split_contract.json`을 추가했다. 분할 단계 산출물을 복사하여 독립 검사에서도 정확한 위치 목록·버전을 확인한다. 예전 준비 폴더에 이 파일이 없으면 새 최종 검사를 통과하지 못한다. 파일을 임의로 끼워 넣거나 기존 결과를 덮어쓰지 말고 새 실행·승인을 통해 준비해야 한다.

최종 검사 단계의 새 attempt 폴더:

- `contract_check.json`: 통과 여부·검사 코드와 성공한 경우의 로컬 상세 요약.
- `report_context.json`: 실행기가 연결한 준비 산출물의 위치·완료 표식 해시·원본/설정 식별값. 실패 시 생성하지 않는다.

보고서 단계의 새 attempt 폴더:

- `report.md`: 적용 설정, 처리 전후 결측 수, 제거 특징·이유, 새 범주 발생, 최종 검사 결과, 산출물 사용 범위.
- `report.json`: 기계 판독용 검증 요약.
- `report_context.json`: 같은 준비 산출물에 대한 로컬 출처 연결.

보고서 위치는 실행 상태의 `artifact_refs["render_report"]`로 조회한다. 준비 산출물 위치는 report_context.json의 prepared_artifact로 확인한다. 입력 경로를 LLM 인자로 받거나 임의 폴더에 게시하는 기능은 만들지 않았다. reports 폴더로 최종 게시하는 publish 기능은 별도 후속 작업이다.

결측 수는 원본 특징 전체, 제거 후 유지한 특징, 변환 출력의 세 기준을 구분한다. 원핫 인코딩은 열을 늘릴 수 있으므로 출력 열 증가를 잘못된 특징 제거로 해석하지 않는다. 새 범주 ignore의 전부 0 출력과 발생 건수도 표시한다. 컬럼명은 Markdown·HTML 이스케이프를 적용한다.

## 실행·보안 경계

`validate_contract`는 prepare_data, `render_report`는 validate_contract의 DB 커밋 이력·완료 파일에 의존한다. 실행기는 각 단계 전에 모든 선행 산출물의 해시를 다시 확인한다. 최종 검사를 통과한 뒤 준비 파일이 바뀌어도 보고서 생성을 차단한다.

보고서 생성은 LLM을 호출하지 않는다. 원본·통계·컬럼·경로·계약 식별값·상세 오류는 로컬에만 남긴다. 해시는 손상·불일치 확인이며 악의적 파일·DB 재서명을 방어하는 인증 수단이 아니다. OS worker 격리·ACL·DB 인증 무결성·실데이터 승인 화면 및 CLI는 아직 후속 범위다.

현재 적용은 표형 CSV 도구다. WM 이미지 전처리나 기존 SECOM 파이프라인을 새 도구로 자동 교체하지 않는다. 합성 연결 테스트에서는 실제 품질·EDA·분할·준비·최종 검사·보고서 도구를 사용하고 사전 점검·사람 검토는 합성 대체한다. 사용자 확인을 테스트 코드가 대신하는 구조를 실데이터 운영 승인으로 취급하지 않는다.

## 검증

```powershell
python -m unittest tests.agents.test_prepared_contract -v
python -m unittest discover -s tests/agents -q
```

검증 항목: 재학습 금지, 행·정답 연결 위반, 출력 컬럼/순서/NaN/무한대, 분할 결합, 추가 파일·손상 JSON, transform 결과 차이, 결정적 보고서·컬럼 이스케이프, 새 범주 기록, 실패 시 미완료, 최종 검사 이후 산출물 변경 차단, 합성 단계 연결.

2026-10-15 검증 결과: 신규 13개 통과. 전체 에이전트 합성 테스트 117개 중 116개 통과, Windows 링크 생성 권한에 따른 1개 건너뜀. 라이브러리의 Pyparsing 지원 중단 예정 경고는 있었으며 테스트 실패는 없었다.
