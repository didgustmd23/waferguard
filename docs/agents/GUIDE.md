# 에이전트 문서 안내

갱신일: 2026-10-18. 문서를 처음 읽을 때와 현재 구현 상태를 확인할 때의 진입점입니다. 상세 계약·명령·과거 검증 이력을 분리하며, 소스나 실행 동작은 이 문서 정리로 변경하지 않습니다.

## 먼저 읽을 문서

1. [구성도](ARCHITECTURE.md): 전처리 코어·에이전트·ML·DL의 역할과 신뢰 경계.
2. [기초 설계](AGENT_BASIC_DESIGN.md): 목표·승인·상태·LLM 정보 제한.
3. [상세 설계](AGENT_DETAILED_DESIGN.md): 실행 계약·보안 통제·결과 검증 기준.
4. 아래 작업별 사용법: 실제 실행할 기능만 선택해서 확인.

기초·상세 설계의 날짜별 구현 메모와 `history/`의 “다음 단계”는 당시 기록입니다. 현재 지원 여부는 이 안내와 해당 기능의 최신 사용법을 먼저 확인합니다.

## 현재 구현 상태

### 로컬 Qwen 연결 — 실행안 계약 단계

2026-10-18 사용자 환경에서 Ollama `qwen3.5:9b` 설치와 Python `ollama.chat`의 합성 한국어 응답을 확인했습니다. LangChain·LangGraph 연결이나 실제 데이터 해설 검증은 아직 아닙니다.

`src/agents/llm_proposal_contract.py`는 모델 응답을 `llm_proposal_v1` JSON으로 검사합니다. 필드는 `schema_version`, `status`, `pipeline`, `questions`만 허용합니다. `proposed`는 기존 표형 또는 WM 계획 형식을 사용하며, `needs_clarification`은 `pipeline: null`과 확인 질문만 반환합니다.

- 로컬 호출자가 정한 논리 입력 계약과 해당 요청의 모듈 ID·정확한 버전 허용 목록을 따릅니다. 모델이 입력·권한 범위를 정하지 않습니다.
- 기존 등록·설정·포트·순서 검증을 재사용하며, 표형 분석·변환·보고는 선행 품질 검사에 의존해야 합니다.
- 중복 JSON 키·비표준 숫자·64 KiB 초과 응답·추가 승인/명령 필드를 거부합니다. 설명문에서 JSON을 임의 추출하지 않습니다.
- 정적 계약 통과는 실제 입력의 품질·사용자 의도 충족·실행 안전성·사용자 승인을 보증하지 않습니다. 질문과 설명도 신뢰할 수 없는 모델 출력입니다.
- 이 단계는 모델 호출·원본 읽기·승인 발급·worker 실행을 하지 않습니다. 새 계약 테스트 9개가 통과했습니다. 이후 모델 호출부와 제안 전용 CLI를 별도로 연결합니다.

사용자와 논의한 기능별 정보 공개 설정은 후속 설계 범위입니다. 사용자가 전달 항목과 실제 내용을 확인·허용하고 시스템은 승인 범위를 강제하는 방향이며, 아직 공개 제어 계층을 구현한 상태는 아닙니다. 현재 연결 시험은 합성 요청과 데이터가 없는 등록 명세만 대상으로 합니다. LLM 전달 허용과 외부 추적 저장 허용은 구분합니다.

검증 명령: `python -m unittest tests.agents.test_llm_proposal_contract -v`

| 영역 | 현재 범위 | 남은 범위 |
| --- | --- | --- |
| 공통 실행 관리 | 실행 계획·상태 저장·SQLite 승인 1회 소비·취소·완료 검증 | LLM 기반 제안·선택 연결 |
| 격리 실행 | AppContainer·Job, 파일 권한·자원 제한, 종료·정리 확인 | 새 어댑터마다 실제 격리 실행 검증 필요 |
| CSV | 품질·EDA·분할, Train 기준 후보·저장 기준 적용·전달 묶음 | 형식별 확장과 전체 실행 동선 통합 |
| pickle 입력 | 격리된 구조 판별·등록 변환기, 일반 표·숫자 배열·WM canonical 변환 | 일반 표·배열 canonical의 후속 EDA 연결 |
| WM canonical | 승인·품질·EDA, 충돌 제외 후보·적용, 그룹 제안·확정, 변환 검사·Train·Validation 캐시·CPU Dataset 첫 배치 연결, 정상 6단계의 공통 생성·배정·조회·사용자 승인/한 단계 실행과 새 합성 흐름 실제 격리 검증 완료 | 충돌 제외 분기 통합. 전달 범용화는 후순위 |
| 사용자 모듈 | 명세·코드 해시·설정 검사, 참조 모듈의 고정 합성 격리 실행 | 임의 사용자 모듈·실데이터 실행 일반화 |
| 공통 등록부 | 코어→에이전트 역참조 제거, 계약 주입, 표형 8개·WM 8개 통합 조회와 WM 증거 포트 계획, 정상 6단계 완료 이력 연결·분할 제안 준비 생성 | 다른 모듈 생성과 승인·실행 동선 통합 |

최근 전체 에이전트 테스트는 427개 중 420개 통과·선택 실행 7개 생략이며 실패는 없습니다. 새 충돌 제외 후보의 실제 AppContainer 합성 검사 1개와 기존 가상 제외 알고리즘 검사 3개도 별도로 통과했습니다. WM canonical의 실제 합성 정상·검토 중단 2개, 기존 검증·EDA·분할 증거 회귀 74개, pickle 변환의 합성 표·숫자 배열·WM 맵 실제 격리 3개는 앞선 실행 기록입니다. 기존 실데이터 품질 결과의 연결 가능성은 읽기 전용으로 확인했으며, 실데이터 후보 진단이나 실제 제외를 자동 실행하지 않았습니다.

모델 학습은 전처리 에이전트의 범위가 아닙니다. 현재 구현에서는 데이터·집계·그림·구조 보고서·상세 경로를 LLM에 전달하지 않습니다. 후속 [기능별 정보 공개 검토](design/LLM_COMMENTARY_REVIEW.md)는 사용자가 전송 내용을 선택·확인하는 별도 기능이며 아직 구현하지 않았습니다. 실행 승인과 정보 전달 승인은 구분합니다.

WM 캐시 연결은 관련 회귀 17개와 실제 AppContainer 합성 검사 1개가 통과했습니다. 준비 진단 보완 후 관련 8개를 추가 검증했고, 사용자가 실행한 `wm-development-cache02`는 `cache_complete`로 완료했습니다. 실패한 `wm-development-cache01` 기록과 소비 승인은 보존했습니다. [캐시 생성 절차](guides/PICKLE_IMPORT.md#확인한-변환-설정으로-trainvalidation-캐시-생성)를 참고하세요.

WM Dataset·첫 배치 연결은 합성·기존 캐시/Dataset 회귀 17개와 실제 AppContainer의 PyTorch 합성 검사 1개가 통과했습니다. 실제 검사 worker에는 Train·Validation 캐시만 전달하며, 모델 학습·Test 처리는 하지 않습니다. 실데이터의 성공 여부는 별도 승인 실행으로 확인합니다. [Dataset 검사 절차](guides/PICKLE_IMPORT.md#완료-캐시의-dataset첫-배치-검사)를 참고하세요. 전체 에이전트 테스트 전체집합은 이번 변경 후 재실행하지 않았습니다.

사용자 실행 출력 기준 `wm-dataset-check01`은 `dataset_batch_passed`, `development_inputs_ready: true`로 완료했습니다. 위 별도 실행 확인 단계까지 진행한 기록이며, Test 맵 처리나 모델 학습 결과는 아닙니다.

WM 공통 모듈 연결 1단계는 [통합 등록·계획](design/METADATA_REGISTRY.md#wm-작업의-공통-등록계획-연결)을 참고하세요. 새 검사 10개를 포함한 관련 회귀 38개와 WM Dataset·캐시 회귀 25개가 통과했습니다(중복 제외 53개). 현재 CLI는 목록·정적 검증만 제공하며, 기존 작업 연결 API도 자동 실행·승인 소비는 하지 않습니다. 전체 흐름을 하나의 CLI로 실행하는 단계와 구분합니다.

후속 [실제 작업 연결 조회](design/METADATA_REGISTRY.md#실제-wm-작업-배정과-읽기-전용-연결-조회)를 추가했습니다. `wm_run_binding_cli`는 배정된 정상 WM 6단계의 완료·승인 소비·부모 결과 연결을 읽기 전용으로 대조합니다. 현재 실제 완료 작업 6개도 `bindings_verified`로 확인했습니다. 새 작업 생성·승인 발급·worker 실행 기능은 아니며, 충돌 제외 분기의 통합 조회는 아직 지원하지 않습니다.

[다음 작업 안내](design/METADATA_REGISTRY.md#다음-작업-안내)는 `wm_guidance_cli`로 조회합니다. 준비·승인 요청·승인 대기·유효 승인 보유·예약·실패·취소·보안 잠금을 구분하고 행동만 안내합니다. 새 테스트 7개를 포함한 관련 회귀 22개가 통과했습니다. 현재 단계는 안내까지이며 생성·승인·실행 명령을 자동 호출하지 않습니다.

별도 [생성·배정 CLI](design/METADATA_REGISTRY.md#첫-eda-작업-준비-통합)는 첫 EDA부터 분할·변환·캐시·Dataset까지 정상 6단계를 기존 생성 함수로 준비하고 새 배정 파일을 기록합니다. 첫 EDA도 canonical 바이트 복사·선택 고정만 수행하며 생성 명령 자체는 worker를 실행하지 않습니다. [공통 승인·실행 CLI](design/METADATA_REGISTRY.md#공통-승인한-단계-실행)는 배정된 현재 작업을 명시해 기존 사용자 승인 화면 또는 한 단계 실행을 호출합니다. 자동 승인·전체 자동 진행·실패 재시도 기능은 아닙니다. 실제 AppContainer의 단일 합성 EDA 정상·검토 중단 경로와 [새 합성 6단계 전체 연결](design/METADATA_REGISTRY.md#새-합성-6단계-전체-완료-검증)을 확인했습니다. 실데이터 성능·모델 학습·자동 승인 성공으로 확대 해석하지 않습니다.

충돌 제외 적용 후속 검증은 관련 합성 회귀 57개와 실제 AppContainer 합성 적용 1개가 통과했습니다. 새 canonical 전체 재검증까지 포함하며 실데이터 적용은 자동 실행하지 않았습니다. 이 후속 변경 후 전체 테스트 전체집합은 재실행하지 않았으므로 위 427개 기록과 구분합니다.

## 작업별 사용법 — guides

| 작업 | 문서 |
| --- | --- |
| 입력 snapshot으로 일반 CSV EDA | [SNAPSHOT_EDA](guides/SNAPSHOT_EDA.md) |
| 비파괴 품질 검사 | [SNAPSHOT_AUDIT](guides/SNAPSHOT_AUDIT.md) |
| 모듈을 조합한 CSV 품질·EDA 실행 | [MODULE_PIPELINE_EXECUTION](guides/MODULE_PIPELINE_EXECUTION.md) |
| 형식 중립 승인 CLI와 현재 CSV 실행 | [PIPELINE_APPROVAL](guides/PIPELINE_APPROVAL.md) |
| 검토된 참조 모듈의 합성 격리 실행 | [REVIEWED_MODULE_EXECUTION](guides/REVIEWED_MODULE_EXECUTION.md) |
| 품질·EDA 검토 후 분할 승인 | [SNAPSHOT_SPLIT](guides/SNAPSHOT_SPLIT.md) |
| 확정 분할의 Train·Validation 추출 | [PARTITION_EXPORT](guides/PARTITION_EXPORT.md) |
| 전처리 후보→검토→적용 통합 실행 | [PREPARATION_PIPELINE_EXECUTION](guides/PREPARATION_PIPELINE_EXECUTION.md) |
| 후보 산출 단계의 상세 계약·명령 | [PREPARATION_CANDIDATE](guides/PREPARATION_CANDIDATE.md) |
| 저장 기준 적용 단계의 상세 계약·명령 | [PREPARATION_APPLY](guides/PREPARATION_APPLY.md) |
| 완료된 CSV 결과의 로컬 전달·검증 | [CSV_HANDOFF](guides/CSV_HANDOFF.md) |
| pickle 구조 판별·표/배열/WM 변환 | [PICKLE_IMPORT](guides/PICKLE_IMPORT.md) |
| 이미 변환된 WM canonical의 승인·품질·EDA | [WM canonical 실행](guides/PICKLE_IMPORT.md#wm-canonical의-승인격리-품질-검사와-eda) |
| 검토 중단된 WM 맵 충돌의 가상 제외 후보 | [WM 충돌 후보 진단](guides/PICKLE_IMPORT.md#wm-맵-라벨-충돌의-가상-제외-후보) |
| 완료 후보에 별도 승인 후 새 canonical 생성 | [WM 충돌 제외 적용](guides/PICKLE_IMPORT.md#승인한-충돌-후보로-새-canonical-생성) |
| 완료 WM EDA에서 그룹 분할안만 생성 | [WM 그룹 분할 제안](guides/PICKLE_IMPORT.md#완료-eda에-연결한-wm-그룹-분할-제안) |
| 검토한 WM 고정 그룹 배정 확정 | [WM 그룹 분할 확정](guides/PICKLE_IMPORT.md#검토한-wm-그룹-분할안-확정) |
| 확정 WM 개발 구간의 크기·채널·패턴 보존 검사 | [WM 변환 검사](guides/PICKLE_IMPORT.md#확정-분할의-trainvalidation-변환패턴-보존-검사) |
| 그림 확인·분할 재확인 후 Train·Validation 숫자 캐시 | [WM 캐시 생성](guides/PICKLE_IMPORT.md#확인한-변환-설정으로-trainvalidation-캐시-생성) |
| 완료 캐시의 CPU Dataset·첫 배치 검사 | [WM Dataset 검사](guides/PICKLE_IMPORT.md#완료-캐시의-dataset첫-배치-검사) |

일반적으로 CSV는 `입력 선택 → 품질·EDA → 사람 검토 → 분할 → 구간 추출 → 후보 → 검토·적용 → 전달` 순서입니다. 각 CLI의 연결 조건을 확인하며, CLI별 승인 ID를 다른 실행이나 단계에 재사용하지 않습니다. 위 기능이 하나의 명령으로 전부 자동 실행된다는 뜻은 아닙니다.

## 구현·확장 계약 — design

| 목적 | 문서 |
| --- | --- |
| 범용 코어·프로파일·도메인 어댑터 경계 | [GENERIC_PREPROCESSING_DESIGN](design/GENERIC_PREPROCESSING_DESIGN.md) |
| 형식 독립 복사본·파일 크기 상한 | [INPUT_SNAPSHOT](design/INPUT_SNAPSHOT.md) |
| EDA 입력·프로파일·자원 결합 | [EDA_INPUT_CONTRACT](design/EDA_INPUT_CONTRACT.md) |
| 분할 증거와 구간 입력 계약 | [PARTITION_INPUT_CONTRACT](design/PARTITION_INPUT_CONTRACT.md) |
| 모듈 ID·버전·입출력·설정·검증 | [MODULE_CONTRACT](design/MODULE_CONTRACT.md) |
| 내장 모듈 목록·등록 API | [MODULE_REGISTRY](design/MODULE_REGISTRY.md) |
| 모듈 파이프라인 JSON·순서·의존성 검사 | [MODULE_PIPELINE](design/MODULE_PIPELINE.md) |
| 명세·설정 스키마와 의존성 방향 | [METADATA_REGISTRY](design/METADATA_REGISTRY.md) |
| 사용자 모듈의 검토·등록 절차 | [USER_MODULE_REGISTRATION](design/USER_MODULE_REGISTRATION.md) |
| 검토 모듈 연결의 설계 기준 | [REVIEWED_CSV_PIPELINE_DESIGN](design/REVIEWED_CSV_PIPELINE_DESIGN.md) |
| 첫 CSV 바인더의 범위 | [REVIEWED_CSV_PIPELINE_BINDING](design/REVIEWED_CSV_PIPELINE_BINDING.md) |
| 분할 후 후보·적용의 연결 원칙 | [MODULE_TRANSFORM_PLAN](design/MODULE_TRANSFORM_PLAN.md) |
| 전처리 모듈 조합 계획 | [PREPARATION_MODULE_PLAN](design/PREPARATION_MODULE_PLAN.md) |
| 후속 검토만 가능: 선택형 보안·LLM 해설 | [LLM_COMMENTARY_REVIEW](design/LLM_COMMENTARY_REVIEW.md) |

계획 문서가 있다는 것만으로 기능이 실행 가능하거나 보안 정책이 변경된 것은 아닙니다. 특히 사용자 모듈과 LLM 확장은 현재 실행 허용 범위와 구분합니다.

후순위 검토: [형식 중립 전달 코어·CSV/WM 어댑터 분리](guides/CSV_HANDOFF.md#후순위-검토-형식-중립-전달-코어와-어댑터)는 사용자 결정으로 구현을 보류했습니다. 변경 범위·승인 방식·용량 상한·기존 실행 해시 연결 영향만 기록했으며, WM 전달 묶음 기능을 새로 구현한 것은 아닙니다.

후순위 개선 검토: [조회 응답 시간](design/METADATA_REGISTRY.md#추후-개선-검토-조회-응답-시간)에 빠른 참고 조회·정밀 검증 분리, 중복 읽기 계측, 캐시 무효화·실행 직전 필수 검증 유지 기준을 기록했습니다. 현재 검증 동작이나 CLI 기본 모드는 변경하지 않았습니다.

## 격리·보안 검증 — security

- [WORKER_EXECUTION](security/WORKER_EXECUTION.md): worker 요청·응답·경로 계약.
- [APPCONTAINER_PROBE](security/APPCONTAINER_PROBE.md): 합성 호환성 검사와 실제 검사 이력.
- [RESOURCE_LIMITS](security/RESOURCE_LIMITS.md): 메모리·프로세스 수·시간 상한과 요청량.
- [WFP_DIAGNOSTIC](security/WFP_DIAGNOSTIC.md): Windows 로컬 통신 차단 증거 확인.

## 과거 구현·검증 기록 — history

현재 기능의 기본 사용법이 아니라 초기 시연·진행 과정 확인용입니다. 내용과 당시 결과는 삭제하지 않았습니다.

- [CLI_USAGE](history/CLI_USAGE.md): 최초 합성 데모 CLI.
- [ISOLATED_EDA](history/ISOLATED_EDA.md): 고정 합성 EDA의 첫 격리 실행.
- [SECOM_TRAIN_EDA](history/SECOM_TRAIN_EDA.md): SECOM Train EDA 연결 확인 이력.
- [QUALITY_CHECK](history/QUALITY_CHECK.md): 공통 품질 검사 초기 연결.
- [SPLIT_CHECK](history/SPLIT_CHECK.md): 공통 분할 초기 검사.
- [TRAIN_ONLY_PREPARATION](history/TRAIN_ONLY_PREPARATION.md): 학습 구간 기준 전처리 초기 연결.
- [FINAL_CONTRACT_REPORT](history/FINAL_CONTRACT_REPORT.md): 초기 최종 입력 계약·로컬 보고서 검사.
- [WM_PICKLE_IMPORT](history/WM_PICKLE_IMPORT.md): 범용 변환 전 WM 전용 최초 연결.

## 문서 관리 원칙

- 새 기능마다 최상위 문서를 늘리지 않고 기존 역할별 문서를 먼저 갱신합니다.
- 현재 지원 상태와 읽는 순서는 이 안내에서, 실제 명령은 `guides/`에서 관리합니다.
- 상세 보안 기준은 상세 설계·`security/`, 데이터·모듈 계약은 `design/`을 따릅니다.
- 과거 실행·검증 기록은 `history/`로 구분합니다. 오래된 명령이 재개 가능하다고 가정하지 않습니다.
- 문서 이동 시 상대 링크를 함께 갱신합니다. 하위 폴더마다 README를 추가하지 않습니다.
