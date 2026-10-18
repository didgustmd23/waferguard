# SECOM Train CSV — 실제 EDA 실행 준비

> 초기 구현·실행 검증 이력입니다. 현재 상태와 권장 진입점은 [에이전트 문서 안내](../GUIDE.md)를 따릅니다. 당시 명령과 결과는 이력 보존용이며 현재 계약으로 자동 재개된다는 뜻이 아닙니다.

2026-10-16. 합성 snapshot-demo03 실제 실행 확인 이후 첫 CSV EDA 대상으로 기존 `data/splits/integrated/time_train.csv`를 선택했다. 병합 CSV가 현재 없으므로 준비된 Train을 사용한다. Test·Validation 데이터는 읽지 않았으며 기존 ML 모델·split·전처리 설정·실험 수치는 변경하지 않았다.

## 준비한 계약

| 항목 | 내용 |
| --- | --- |
| 프로파일 | configs/agents/profiles/secom_train_eda.json |
| 어댑터 | tabular_csv_v1 |
| 특징 | sensor_0부터 sensor_589까지 수치형 역할 |
| 라벨·시간·행 ID | label / timestamp / __source_row_id, 특징에서 제외 |
| 기타 split metadata | __split_role·__split_protocol_id는 특징으로 채택하지 않음 |
| 그래프 특징 상한 | 12개, 통계 CSV는 전체 지정 특징 |
| 입력 파일 상한 | 8MiB, snapshot 복사본 약 3.90MiB |
| worker 요청 | 메모리 2048MiB·동시 프로세스 2개·실행 60초 |
| snapshot | logs/agents/input_snapshots/secom-train-eda01 |
| 실행 기록 | logs/agents/snapshot_eda/runs/secom-train-eda01 |

준비 단계에서는 프로파일과 파일 헤더만 대조해 중복 헤더 없음·필수 컬럼 존재·metadata의 특징 제외를 확인했다. 행 데이터·통계는 출력하거나 외부로 전송하지 않았다. 크기 제한 복사·해시 검증·프로파일 결합·preflight 이후 `awaiting_approval / APPROVAL_REQUIRED`에서 멈췄으며 해당 시점에는 worker 실행·승인 발급/소비가 없었다. 이후 사용자의 승인·실행 결과는 아래에 기록한다.

기존 완료 합성 실행과 현재 실행 생성 주체가 같은 OS 계정인지 확인했다. 실제 실행은 사용자의 관리자 터미널에서 표시된 설정·접근 범위를 검토한 뒤 승인한다. 60초 요청은 초기 실행값이지 대규모 EDA에 충분하다는 보장이 아니다. 한도 초과 시 실패 기록을 검토하고 상한 변경·새 승인은 사용자가 결정한다.

## 사용자 승인과 실행

프로젝트 루트의 관리자 PowerShell, ml310 환경에서 진행한다. 이미 snapshot·실행 생성·preflight·승인 대기까지 준비했으므로 해당 단계를 반복하지 않는다.

```powershell
python -m src.agents.snapshot_eda_cli approve --name secom-train-eda01

# 출력한 approval_id로 교체
python -m src.agents.snapshot_eda_cli next --name secom-train-eda01 --approval-id <approval_id>

python -m src.agents.snapshot_eda_cli status --name secom-train-eda01
```

실패하면 자동 재실행하지 않는다. 오류 상태와 최신 고정 진단만 확인한다.

```powershell
Get-ChildItem logs/agents/snapshot_eda/workers/*/scratch/task/diagnostic.json |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1 |
  Get-Content
```

실제 데이터의 상세 결과·컬럼별 통계·그림·원본·복사본을 LLM으로 보내지 않는다. 승인 화면의 컬럼 역할은 로컬 사용자 검토용이다.

## 해석 범위

이 실행은 지정한 Train의 결측·분포·상관 등 EDA만 한다. 결측 대치·특징 제거·스케일링·재분할·모델 학습·문턱 조정·Test 평가를 하지 않는다. 역할 JSON의 `eda_review_v1` 전체 workflow가 아니라 snapshot EDA 전용 두 단계 계획을 사용하며 품질 audit와 사람의 분석 결과 검토도 자동 완료하지 않는다.

상세 출력 검증의 한도와 보안 범위는 [snapshot EDA 안내](../guides/SNAPSHOT_EDA.md)를 따른다. 아래 정상 실행 확인은 해당 Train CSV의 EDA 한 건에 한정한다.

## 실제 실행·기록 재검증 — 완료

2026-10-16 사용자 관리자 터미널에서 명시 승인 후 실행을 완료했다. 이후 부모 상태·승인 DB·Job 실행 기록·완료 표식·선택 및 작업용 snapshot을 읽기 전용으로 대조했다. 재실행·재승인·원본 수정·모델 학습·Test 접근은 하지 않았다. 상세 EDA 통계·값·그림은 출력하지 않았다.

| 확인 항목 | 결과 |
| --- | --- |
| 상태 | completed |
| 승인·작업 DB | consumed, preflight·run_eda 모두 committed |
| worker 종료 | exited, 종료 코드 0 |
| 고정 단계 진단 | COMPLETE / PASSED |
| Job 실행 시간 | 24.406초, 부모 준비 시간 제외 |
| 최고 Job 메모리 | 209485824 bytes ≈ 199.78MiB, 상한 2048MiB |
| 프로세스 | 동시 상한 2개, 누적 생성 3개, 정리 후 활성 0개 |
| 자원 초과 알림 | 없음 |
| Job·AppContainer 프로필 정리 | 모두 확인 |
| 선택·staging snapshot | 기대 해시·크기·파일 상한과 일치 |
| 요청/응답·계획·현재 코드 | 일치 |
| 완료 산출물 | 계획 기준 ARTIFACT_REUSABLE, 부모 결과 계약 재검증 통과 |

작업 ID: `0980eed5cd3141fb81419fbe2fc87cd8`.

로컬 EDA 보고서는 `logs/agents/snapshot_eda/runs/secom-train-eda01/attempts/0980eed5cd3141fb81419fbe2fc87cd8/eda/eda.md`다. 같은 폴더의 CSV·PNG·JSON은 로컬에서 확인하며 외부 전송하지 않는다. OS 실행 기록은 `logs/agents/snapshot_eda/workers/0980eed5cd3141fb81419fbe2fc87cd8/transport.json`에 있다.

결론은 **등록 CSV 프로파일을 사용하는 실제 Train 데이터의 승인·격리 EDA·부모 검증·완료 등록 정상 경로 확인**이다. 원본 대비 통계의 독립 재계산·추가 형식 어댑터·전체 전처리/학습 workflow·LLM 선택기·모든 보안 위협의 검증 완료를 뜻하지 않는다. 후속 단계에서는 로컬 EDA 결과를 검토하고 품질 audit·분할·Train-only 준비 도구의 격리 연결 범위를 별도로 결정한다.
