# 범용 전처리 에이전트 — 설계 정렬과 공통 계약

현재 기준 갱신일: 2026-10-20. 범용 전처리가 제품 목표이며 WM-811K는 적용 사례다. 이 문서는 공통 입력·EDA·프로파일별 계획의 기준이다. LLM 정보 제한·승인·격리·이력 보안은 [상세 설계](../AGENT_DETAILED_DESIGN.md)를 따른다.

현재 CSV는 품질·EDA·분할·Train 기준 산출·저장 기준 적용·로컬 전달과 EDA/CSV LLM 제안까지 구현했다. WM 정상 개발 6단계는 도메인 코어를 재사용해 공통 UI에 연결했다. 결과 해설만 별도 전달 정책으로 집계·그래프를 허용하며 제안에 원본 결과를 자동 첨부하지 않는다. 아래 2026-10-15~16의 차이 분석과 “후속 구현”은 초기 정렬 기록이고, 현재 완료·보류·검증 범위는 [GUIDE](../GUIDE.md#현재-구현-상태)를 우선한다. 모든 형식·사용자 모듈의 일반화가 완료된 것은 아니다.

## 1. 최초 목표와 구현의 차이

2026-10-16 추가: [모듈 계약 설계](MODULE_CONTRACT.md)와 [내장 등록부](MODULE_REGISTRY.md)를 정리했다. ID·버전·입출력·설정·원본 보존·전처리 기준 산출 구간·승인·결과 검증을 공통 항목으로 정의한다. 모델 학습은 제외하며 사용자 조합 실행은 후속 구현이다. 기존 고정 workflow·프로파일은 변경하지 않는다.

최초 요구는 다른 데이터에도 적용 가능한 전처리 코어와 작업 관리였다. 기초 문서는 최초 적용 대상을 WM으로 명시했지만 실제 예시의 맵·Lot·캐시 단계가 공통 처리 구조처럼 확대됐다. 아래 표는 이번 정비 이전의 차이와 이번 조치다.

| 영역 | 최초 목표 | 정비 전 구현 | 이번 조치 |
| --- | --- | --- | --- |
| 입력 | 데이터별 역할을 설정으로 정의 | WM canonical 필드·label 버전 고정 | `ColumnRoles`·`TabularInput` 공통 계약 |
| EDA | 공통 분석과 도메인 분석 분리 | 에이전트가 WM EDA 전체를 호출 | 공통 EDA를 `src/preprocessing/eda.py`로 분리 |
| 어댑터 | 읽기·공통 계약 변환 담당 | WM 도구 실행 wrapper 중심 | CSV·WM metadata가 같은 `TabularEdaAdapter` 사용 |
| 실행 계획 | 프로파일에 따라 필요한 처리 선택 | WM 단계·캐시 중심의 단일 순서 | 검토된 EDA/학습 준비 템플릿 선택 |
| 정책 | 데이터셋 독립 승인·필수 차단 | WM 정책 기본 이름·단계 중복 정의 | 공통 정책 이름·중앙 단계 정의·`prepare_data` |
| 검증 | 서로 다른 입력으로 재사용 확인 | 합성 파일과 WM fixture 중심 | 독립 CSV·다중 클래스·미라벨·범주형 입력 추가 |
| 품질 검사 | 프로파일 기준으로 문제 진단·진행 차단 | audit는 합성 도구 또는 WM 도메인 검사 | 공통 quality 코어와 실제 run_audit 연결 |
| 보안 | 원본·추측 가능한 상세 정보 LLM 전달 금지 | 같은 제한을 적용 | 제한 유지, 프로파일·EDA 표도 로컬 전용 |

승인 DB·산출물 검사·실행 잠금·재개 기능은 재사용한다. 기존 SECOM 실험·모델·설정과 `wafer_dl`의 WM 전처리·노트북은 변경하지 않는다.

## 2. 책임과 의존 방향

전체 프로젝트에서의 위치, 현재 격리 실행 경로, 향후 모듈형 구조는 [에이전트 구성도](../ARCHITECTURE.md)를 참고한다. 구현 완료와 예정 기능을 구분하고 실제 소스 위치를 함께 정리했다.

~~~text
로컬 프로파일 + 명시적 입력 위치
                ↓
검토된 어댑터 등록표 → 파일 읽기 / 표형 metadata 변환
                ↓
TabularInput + ColumnRoles
                ↓
범용 EDA 코어 → 로컬 표·그래프·보고서

프로파일 → 검토된 workflow 템플릿 → 공통 실행기
                                    ├─ 정책·승인
                                    ├─ 등록 도구 호출
                                    └─ 상태·산출물·안전한 재개

WM 도메인 확장: 맵 예시·다이/공간 통계·마스크·이미지 변환
표형 도메인 확장: 시간/그룹 진단·데이터별 품질 규칙
~~~

`preprocessing`은 `agents`, `sensor_ml`, `wafer_dl`을 import하지 않는다. 에이전트 없이 공통 입력 검사와 EDA를 호출할 수 있다. 공통 실행기는 WM 코어를 직접 import하지 않으며 어댑터 레지스트리에서 선택한 등록 도구를 받는다. 프로파일에서 임의 모듈·함수·셸·URL을 전달해 실행하지 않는다.

이미지·맵의 원본 배열을 억지로 DataFrame 특징으로 펼치지 않는다. 현재 공통 계약은 표형 데이터와 이미지에서 어댑터가 추출한 표형 metadata의 EDA 계약이다. 실제 맵·영상 텐서와 해당 변환은 도메인 코어에 남는다. 모든 데이터 형식의 완전한 처리를 지원한다는 의미가 아니다.

## 3. 범용 입력 계약

| 역할 | 형식 | 용도 |
| --- | --- | --- |
| numeric | 수치 특징 이름 목록 | 분포·기술 통계·상관분석 |
| categorical | 범주 특징 이름 목록 | 범주별 빈도·결측 |
| target | 문자열 컬럼명 또는 null | 라벨 분포, 라벨 없는 입력도 지원 |
| group | 문자열 컬럼명 또는 null | 독립 그룹 집계·후속 그룹 분할 기준 |
| timestamp | 문자열 컬럼명 또는 null | 시간 역할 보존. 날짜 파싱·시간 분할은 후속 처리 계약 |
| identifiers | 식별 컬럼 이름 목록 | 정합성·추적, 모델 특징·상관분석에서 제외 |

역할 중복·필수 컬럼 누락·중복 컬럼·비수치 numeric·무한대·빈 데이터는 거부한다. numeric의 NaN은 보존하고 EDA에서 결측을 집계한다. 클래스 수·양성 label·센서 접두사·공장명은 공통 코어에 고정하지 않는다. 추가 컬럼을 자동 특징으로 채택하지 않는다.

`TabularInput`은 로컬 DataFrame과 역할 계약을 전달한다. 전체 객체·컬럼명·라벨·원본 위치는 에이전트 상태나 LLM DTO로 직렬화하지 않는다. 프로파일은 데이터 구조를 선언하는 설정이며 데이터 자체를 포함하지 않는다.

## 4. 공통 EDA와 도메인 확장

| 공통 코어에서 구현 | 도메인 코어 또는 후속 구현 |
| --- | --- |
| 특징별 결측 수·비율·관측 고유값 | WM 라벨 충돌·동일 맵·Lot 연결 그룹 진단 |
| 수치 기술 통계·분포 | 맵 예시·다이 공간 분포·패턴 보존 |
| 범주 빈도·선택적 target 분포 | 원본 파일 형식별 안전성·전체 배열 검사 |
| Pearson·Spearman·변수 쌍별 관측 수 | 시간 drift·공정 구간 분석 |
| 동일 특징 행 집계·선택적 그룹 수 | 분할·대치·스케일링·특징 선택의 fit/transform |
| CSV·PNG·JSON·Markdown 보고서 | 원인 판단·제외/재라벨링·모델 성능 최적화 |

공통 품질 기준과 `run_audit` 연결은 [품질 검사](../history/QUALITY_CHECK.md)에 정리한다. 결측·상수 후보를 진단하고 label·ID·group·timestamp를 검사하며, 자동 삭제·대치와 학습용 fit은 수행하지 않는다.

상관은 지정한 numeric만 사용하며 target·group·timestamp·ID는 포함하지 않는다. 표는 전체 지정 특징을 기록하고 그림 수만 제한한다. 상수·관측 부족의 NaN을 0으로 바꾸지 않는다. 결측 대치·자동 특징 제거·라벨 변경은 EDA에 포함하지 않는다. 전체 데이터 EDA는 구조·품질 확인용이며 모델 선택과 학습용 변환은 Train/Validation 경계를 지켜야 한다.

현재 WM 어댑터는 기존 canonical 읽기 검증을 재사용하고 `quality_status='accepted'` 행을 공통 입력으로 전달한다. 이 적격 행 선택은 WM 어댑터 규칙이며 공통 코어의 규칙이 아니다. 기존 `wafer_dl.eda`의 도메인 보고서는 계속 단독 사용 가능하며 새 공통 보고서와 파일·버전을 혼용하지 않는다.

### 4.1 사용자 지정 EDA 추가 — 후속 검토 항목

2026-10-16 갱신. EDA 계산은 `eda.py`의 결측·기술 통계·범주·라벨·상관분석 함수로 분리하고 `collect_eda(analyses=...)`에서 선택 호출한다. 그림·파일 생성은 `eda_report.py`로 분리했다. 현재 격리 snapshot worker는 등록 CSV 로더와 전체 보고서용 `generate_eda_report()`에 연결돼 있다. 고정 내장 등록부는 구현했지만 CLI의 사용자 기능 선택·등록부 연결은 아직 구현하지 않았으며 사용자 `.py`의 자동 실행도 허용하지 않는다.

과거 보고서·실험 노트북은 당시 결과로 보존한다. 호환 목적으로 기존 `summarize_eda()`·`run_eda()`를 남기지 않고 현재 어댑터·worker·부모 검증기·테스트를 새 경로로 변경했다. 보고서 출력 형식은 이번 단계에서 유지한다. worker 복사 허용 목록·코드 식별값에 새 파일도 포함하므로 기존 실행의 승인을 재사용하지 않고 새 실행·승인을 준비해야 한다. 아래 항목은 후속 확장 방향이며 구현 완료를 뜻하지 않는다.

- [ ] 입력 어댑터 ID(파일을 읽는 방식)와 분석 ID(수행할 EDA)를 분리한다. 같은 CSV의 새 분석을 위해 로더를 복제하지 않는다.
- [ ] 사용자 지정 분석은 에이전트 없이 호출 가능한 독립 함수로 구현하고 검토한 고정 ID에 연결한다. 프로파일에서 임의 import·함수 경로·셸 명령을 받지 않는다.
- [ ] 분석 ID·버전·허용 인자·코드 식별값을 입력 계약과 승인 범위에 묶는다. 코드·설정 변경은 새 계획·새 승인을 요구한다.
- [ ] worker에 필요한 고정 소스·라이브러리만 명시적으로 전달한다. 파일·네트워크 권한과 자원 요청은 별도 검토하며 새 EDA라는 이유로 자동 확대하지 않는다.
- [ ] 분석별 출력 파일·버전·JSON/CSV 필드·크기 상한·부모 검증기를 등록한다. 부모는 원본 재파싱·pickle 역직렬화·자식 명령 실행 없이 제한된 결과를 검사한다.
- [ ] 새 분석의 계산 정확성은 정답이 있는 합성 입력으로 검증하고, AppContainer 호환성·승인·변조·실패·취소·완료 등록까지 별도로 확인한다.
- [ ] 도메인 EDA(WM 맵·시간 drift 등)는 기존 도메인 코어를 재사용한다. 일반 입력 모두에 해당 분석을 강제하지 않는다.
- [ ] 자동 도구 선택(향후 LLM 포함)과 사용자 분석 코드 등록을 구분한다. 선택기가 임의 코드를 생성·등록·실행하거나 데이터·통계·경로를 외부로 전달하지 못하게 한다.

등록 API·분석 프로파일 스키마·의존성 검토/배포 방식·정확성 교차 검증 수준은 후속 구현 전에 결정한다. 현재 일반 어댑터 등록표가 있다는 사실만으로 새 분석이 격리 worker에 연결된 것은 아니다. [snapshot EDA의 실제 실행 확인](../guides/SNAPSHOT_EDA.md)을 선행 기준으로 사용한다.

## 5. 프로파일 설정과 실행 계획

선택 모듈의 [승인·격리 실행 연결](../guides/MODULE_PIPELINE_EXECUTION.md)을 새 CLI로 추가했다. 내장 CSV EDA·품질 검사만 지원하고 출력 재입력·변환은 거부한다. 기존 고정 CLI를 유지하며 새 경로의 실제 AppContainer 실행은 사용자 확인 전이다.

사용자 모듈 조합용 [파이프라인 JSON과 정적 검증](MODULE_PIPELINE.md)을 추가했다. 아래 기존 데이터 프로파일과는 별도 설정이며 입력 경로·권한·승인을 포함하지 않는다. 실제 실행 연결은 아직 하지 않았다.

2026-10-16 추가: [계약 타입·내장 모듈 등록부](MODULE_REGISTRY.md)를 구현했다. EDA 5개 계산 기능과 품질 검사 1개를 정확한 ID·버전으로 조회하고 설정을 검증한다. 아래 기존 프로파일·고정 workflow와 사용자 조합형 파이프라인의 연결은 아직 구현하지 않았다.

모듈형 파이프라인 개발 이후의 선택형 보안 모드·LLM 해설 제안은 [후속 검토 안건](LLM_COMMENTARY_REVIEW.md)에 별도 보존한다. 이는 현재 집계·그림·경로의 LLM 전달 금지 정책 변경이나 전송 승인이 아니며, 통합 모듈 계약·등록부·조합 검증을 먼저 정비한다.

프로파일은 `configs/agents/profiles/`에 둔다. 기존 `configs/datasets/secom.json`은 보존한 ML 계약이므로 자동 변환·교체하지 않는다. 공장·데이터마다 JSON을 복사하고 실제 컬럼 역할을 지정하며 입력 위치는 로컬 실행 호출에서 명시한다.

~~~json
{
  "schema_version": "dataset_profile_v1",
  "profile_id": "tabular_example",
  "adapter_id": "tabular_csv_v1",
  "workflow_id": "eda_review_v1",
  "roles": {
    "numeric": ["measurement_a", "measurement_b"],
    "categorical": ["process_type"],
    "target": "quality_label",
    "group": "batch_id",
    "timestamp": null,
    "identifiers": ["sample_id"]
  },
  "max_plot_features": 12
}
~~~

지원 어댑터는 `tabular_csv_v1`, `wm811k_metadata_v1`다. JSON 중복 키·알 수 없는 필드·미등록 workflow는 거부한다. 미등록 adapter ID는 등록표에서 거부한다. 원본·프로파일·인자를 자동 식별하고 프로파일 변경은 `config_id`와 `plan_id`에 반영된다.

선택 항목 `quality_rules`에서 결측 기준·허용 라벨·중복 ID 등의 통지 수준과 시간 형식을 지정한다. 생략하면 기본값을 사용하고 기존 프로파일 식별값을 보존한다. 기본값과 다른 기준은 계획·승인 식별에 포함한다. 상세 설정 예시는 [품질 기준 설정](../history/QUALITY_CHECK.md#프로파일-설정)을 따른다.

| 템플릿 | 고정 단계 | 현재 의미 |
| --- | --- | --- |
| eda_review_v1 | preflight → audit → run_eda → review_eda → validate_contract → render_report | EDA와 사람 검토. 분할·캐시 생성 없음 |
| split_review_v1 | preflight → audit → run_eda → review_eda → propose_split → commit_split | 분할 제안 검증·승인 후 확정. Train-only 변환은 별도 |
| prepare_review_v1 | preflight → audit → run_eda → review_eda → propose_split → commit_split → prepare_data | 확정 분할 기반 Train-only fit·Train/Validation 변환·저장 검사 |
| tabular_preparation_v1 | preflight → audit → run_eda → review_eda → propose_split → commit_split → prepare_data → validate_contract → render_report | 표형 준비·독립 최종 검사·로컬 보고서 |
| training_preparation_v1 | preflight → ingest → audit → run_eda → review_eda → propose_split → commit_split → prepare_data → validate_contract → publish → render_report | 학습 준비 구조를 정의한 템플릿. 실제 도구 일괄 구현 완료는 아님 |

프로파일 실행에서는 템플릿 단계의 생략·순서 변경을 거부한다. `prepare_data`는 표형 Pipeline 입력·이미지 캐시 등 도메인에 맞는 준비 도구를 연결할 공통 단계다. 기존 `prepare_cache`는 이전 데모·계획 호환 경로로 유지한다. 등록 도구가 부족하면 실행기 생성 시 거부하며 임의 함수를 자동 생성하지 않는다.

일반 CSV와 WM metadata가 같은 실행기·공통 EDA를 사용하는 것까지 합성 검증했다. 새 표형 연결 테스트는 실제 audit·EDA·분할·준비·최종 계약·보고서 도구를 사용하며 preflight·사람 검토만 합성 대체한다. 이전 EDA 전용 테스트의 최종 단계 대체 도구는 유지한다. 실데이터용 진입점·운영 승인·보안 격리까지 완성한 것은 아니다.

## 6. 파일 구성과 호환

~~~text
src/preprocessing/
  contracts.py       # ColumnRoles·TabularInput
  profiles.py        # 프로파일 JSON 검증·식별
  workflows.py       # 공통 단계·검토된 템플릿
  eda.py             # 독립 계산·분석 선택
  eda_report.py      # 그림·파일·전체 보고서 생성
  quality.py         # 비파괴 품질 검사·로컬 근거 보고서
  split.py           # Random·Group·Time 분할 제안·경계 검증
  transform.py       # 설정형 Train-only Pipeline·JSON 저장/복원
  prepared_contract.py # 독립 최종 입력 계약 검사
  report.py          # 결정적 한국어 로컬 보고서
src/agents/
  workflow.py        # 프로파일 → ExecutionPlan
  executor.py        # 데이터셋 독립 실행 관리
  adapters/
    registry.py      # 등록 ID만 해석
    tabular.py       # 공통 EDA 연결·CSV 로더
    quality.py       # 공통 품질 검사 → run_audit
    split.py         # 분할 제안 → 승인 후 동일 분할 확정
    prepare.py       # 확정 분할 → 승인 후 Train/Validation 입력 준비
    finalize.py      # 독립 최종 검사 → 로컬 보고서
    wm811k_metadata.py # WM 읽기·적격 행·공통 입력 변환
    wm_eda.py        # 기존 import 호환용, 분석 구현 없음
configs/agents/profiles/
  tabular_example.json
  wm811k_metadata.json
~~~

기존 WM 어댑터의 import 경로는 호환 wrapper로 유지하지만 프로파일·출력 계약이 바뀌었으므로 이전 EDA 산출물·승인을 새 계획에 사용하지 않는다. 비프로파일 합성 데모의 계획 저장 형식·식별값은 보존하여 기존 데모가 의미 없이 무효화되지 않게 한다. 과거 원본·데이터 파일을 이동하거나 삭제하지 않는다.

## 7. 사용·검증·남은 범위

~~~powershell
python -m src.agents.cli plan --profile configs/agents/profiles/tabular_example.json
python -m src.agents.cli plan --profile configs/agents/profiles/wm811k_metadata.json
python -m unittest tests.agents.test_generic_preprocessing -v
python -m unittest discover -s tests/agents -v
~~~

CLI의 프로파일 계획 조회는 입력을 읽거나 실행 폴더를 만들지 않는다. `planning_only=true`, `source_bound=false` 상태이며 원본을 결합해 승인한 실행 계획과 구분한다. 실제 실행용 Python API는 `create_eda_adapter(profile, input_path)` → `build_profile_plan(profile, adapter.source_id)` → 등록 도구를 갖춘 `RulesExecutor`다. 사용자 데이터 실행은 이번 작업에서 수행하지 않는다.

2026-10-15 검증 기록: 품질 검사를 포함한 에이전트·공통 전처리 합성 테스트 73개 중 72개 통과, 심볼릭 링크 생성 권한 부족으로 1개 건너뜀. 다중 클래스·미라벨·범주형 입력, 역할 중복·누락·무한대·CSV 중복 헤더 거부, target/group/timestamp/ID의 상관분석 제외, 프로파일 변경과 템플릿 생략 차단, CSV와 WM metadata의 공통 EDA 연결, 품질 기준 경계와 품질 중단 후 EDA 차단을 포함한다.

기존 소스 경로 호환 3개와 WM 단독 EDA의 합성 검증 2개도 통과했다. 기존 실제 데이터·모델·노트북 출력은 재생성하지 않았다.

공통 계약·EDA·품질·분할·Train-only 표형 Pipeline·저장 복원·독립 최종 검사·로컬 보고서 연결까지 구현했다. [분할 검사](../history/SPLIT_CHECK.md), [Train-only 준비](../history/TRAIN_ONLY_PREPARATION.md), [최종 검사·보고서](../history/FINAL_CONTRACT_REPORT.md)에 설정·산출물·제약 및 최신 합성 검증을 기록한다. 전체 실데이터 CLI·게시·OS 격리·LLM 최소 게이트는 후속 구현이다. 승인 DB 변조 방어 검토도 유지한다. 새 프로파일이나 어댑터를 추가해도 원본·정확한 통계·경로·설정의 LLM 전달 금지는 완화하지 않는다.
