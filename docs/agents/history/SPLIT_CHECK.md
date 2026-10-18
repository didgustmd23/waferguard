# 범용 분할 제안과 승인 후 확정

> 초기 구현·실행 검증 이력입니다. 현재 상태와 권장 진입점은 [에이전트 문서 안내](../GUIDE.md)를 따릅니다. 당시 명령과 결과는 이력 보존용이며 현재 계약으로 자동 재개된다는 뜻이 아닙니다.

2026-10-16 갱신: 아래 코어·어댑터에 이어 [snapshot 격리 분할 CLI](../guides/SNAPSHOT_SPLIT.md)를 추가했다. 품질·EDA·사람 검토·제안·승인 확정과 내보내기용 승인 범위 저장을 합성 검증했다. 아래의 이전 CLI 미연결 설명은 초기 구현 시점의 범위이며 현재 실행 방법은 연결 문서를 따른다. 실제 Windows/SECOM 실행은 아직 별도 확인이 필요하다.

## 구현 범위

`src/preprocessing/split.py`는 에이전트 없이도 사용하는 범용 표형 분할 코어다. `src/agents/adapters/split.py`는 이를 `propose_split → commit_split`에 연결한다. 실제 데이터는 이번 구현에서 재분할하지 않았다.

| 방식 | 동작 | 제약 |
| --- | --- | --- |
| Random | 고정 시드로 행 위치를 분할. 명시한 경우 target 계층 분할 | group 역할이 있으면 거부. 클래스 표본이 부족해도 다른 방식으로 자동 변경하지 않음 |
| Group | 같은 그룹의 모든 행을 한 구간에 배치 | 그룹 수 기준 배치라 실제 행 비율은 목표와 다를 수 있음. 계층 분할 아님 |
| Time | 시각 순서대로 과거 Train → Validation → Test | 동일 시각은 경계에서 나누지 않음. stratify 금지. group 역할도 있으면 그룹 교차까지 거부 |

세 방식 모두 빈 구간·행 중복·누락을 검사한다. 기본적으로 target 결측, 복합 식별자 중복·결측을 거부하고, 한 구간에 클래스가 빠지면 `review_required`로 중단한다. `require_all_classes: false`를 명시하면 클래스 누락 경고를 남기면서 진행할 수 있으며 설정 변경은 새로운 프로파일 식별값에 반영된다.

분할 코어는 선언하지 않은 중복 개체나 도메인 연결 관계를 자동 추론하지 않는다. 그룹 정보가 필요한 데이터는 적절한 group 역할을 먼저 제공해야 한다. WM의 Lot·동일 맵 연결 그룹을 단순 Lot 분할로 대체하지 않도록, 현재 분할 어댑터는 `tabular_csv_v1`만 연결한다. 기존 `wafer_dl/split.py`와 SECOM 데이터는 그대로 유지한다.

## 프로파일 설정

분할까지만 준비하는 템플릿은 `split_review_v1`이다.

```text
preflight → audit → run_eda → review_eda → propose_split → commit_split
```

분할 템플릿에는 `split_rules`를 명시해야 한다. EDA 전용 프로파일에는 강제하지 않는다.

```json
"split_rules": {
  "strategy": "random",
  "train_ratio": 0.70,
  "validation_ratio": 0.15,
  "test_ratio": 0.15,
  "random_state": 42,
  "stratify": true,
  "require_all_classes": true,
  "timestamp_format": null,
  "algorithm_version": "tabular_split_v1"
}
```

Group은 `strategy: "group"`, `stratify: false`와 group 역할이 필요하다. Time은 `strategy: "time"`, `stratify: false`와 timestamp 역할이 필요하며, 문자열 시간에는 `%Y-%m-%d %H:%M:%S`처럼 정확한 형식을 명시한다. 시간 경계에 걸친 그룹이 있으면 자동 삭제하거나 Random으로 우회하지 않는다.

## 승인과 산출물

1. 등록 입력 어댑터가 원본 식별값·프로파일을 대조한다.
2. 제안 단계는 `proposal.json`과 `proposal.md`만 만든다. 위치 목록, 클래스 개수, 실제 비율, 해당하는 그룹 수·시간 범위는 로컬에 남긴다.
3. 검사를 통과해야 제안이 커밋된다. 클래스 누락 검토 상태는 확정 단계로 넘어가지 않는다.
4. 확정 단계의 승인 범위에는 원본·설정·계획·인자 외에 제안 완료 표식의 해시도 포함한다.
5. 실행기는 DB의 커밋 이력과 산출물 해시를 검사한 뒤 제안 폴더를 신뢰하는 도구에 별도 인자로 전달한다. JSON에서 임의 경로를 받지 않는다.
6. 승인 후 같은 규칙으로 제안을 재계산·대조한다. 일치하면 `assignments.csv`와 `split.json`을 새 작업 폴더에 저장하고 내용·파일 정합성을 검사한다.

`assignments.csv`는 원본 0부터 시작하는 `row_position`과 `split`만 담는다. 원본의 pandas index나 개별 특징·라벨을 복사하지 않는다. 반드시 `split.json`에 결합된 동일 원본 버전과 함께 사용해야 한다. 원본 변경·위치 목록 변경·설정 변경·불확실한 실패 후에는 기존 승인을 재사용하거나 자동 재실행하지 않는다.

상세 분포·파일 경로·식별값은 LLM에 전달하지 않는다. 해시는 정합성 확인이며 악의적인 파일·DB 재서명을 막는 인증 수단은 아니다. ACL·DB 인증 무결성·OS worker 격리는 기존 후속 보안 검토를 유지한다.

현재는 코어와 등록 도구 연결을 구현한 단계다. 실제 데이터용 승인 화면·CLI와 전 단계 도구의 실데이터 연결은 활성화하지 않았다. 테스트의 합성 승인·대체 도구는 실제 운영 승인 구현이 아니다.

## 검증

프로젝트 루트에서 실행한다.

```powershell
python -m unittest tests.agents.test_generic_split -v
python -m unittest discover -s tests/agents -q
```

2026-10-15: 신규 분할 테스트 14개 통과. 에이전트 전체 87개 중 86개 통과, Windows 링크 생성 권한에 따른 1개 건너뜀. Random 재현·계층 비율, Group 비중첩, Time 동일 시각·순서·그룹 교차, 잘못된 설정·부족한 입력·클래스 누락, 승인 대기, 승인 후 제안/원본 변경 차단, 완료 표식 변경 시 승인 무효화, 저장 위치 목록의 정합성을 합성 입력으로 확인했다.

분할 단계 자체는 결측 대치·스케일링·특징 선택을 학습하지 않는다. 후속 [Train-only 전처리](TRAIN_ONLY_PREPARATION.md)는 구현·합성 검증했으며, Train에서만 fit하고 Train·Validation을 변환한다. Test 변환은 별도 후속 단계이며 모델 학습은 하지 않는다.
