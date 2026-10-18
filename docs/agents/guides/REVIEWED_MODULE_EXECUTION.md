# 검토된 참조 모듈의 합성 격리 실행

2026-10-17. 사용자 모듈 등록의 첫 연결 사례로 `user_missing_row_histogram` v1.0.0을 별도 승인·AppContainer·Job·부모 결과 검증에 연결했다. 고정 합성 CSV만 허용하며 임의 사용자 코드·실데이터를 실행하는 일반 플러그인 기능은 아니다.

## 소스 구성

| 소스 | 역할 |
| --- | --- |
| `src/agents/reviewed_module_cli.py` | 합성 생성·새 승인·진행·취소·완료 확인 |
| `src/agents/reviewed_module_contract.py` | 고정 합성 입력·프로파일·모듈 버전·설정 결합 |
| `src/agents/reviewed_module.py` | 기존 실행기·예약·승인 소비·커밋 재사용 |
| `src/agents/reviewed_module_transport.py` | 고정 소스/명세 복사·새 AppContainer·Job·정리 증거 |
| `src/agents/reviewed_module_worker.py` | 품질 검사 후 격리 안에서 참조 모듈 import·계산 |
| `src/agents/reviewed_module_results.py` | 표준 파서로 결과 구조·정수 범위·합성 정답 검사 |
| `tests/agents/test_reviewed_module.py` | 승인·변경·중단·취소·부팅·부모 검증 합성 검사 |

기존 SECOM·WM·CSV 전달 결과는 변경하지 않는다. 기존 모듈 실행기의 허용 목록도 확장하지 않았다. 별도 실행 루트는 `logs/agents/reviewed_module`이다.

## 실행 경계

```text
고정 합성 CSV·프로파일 생성
    ↓
preflight → 사용자 새 승인
    ↓
새 AppContainer+Job: 품질 검사
    ├─ 검토 필요/차단 → 참조 모듈을 import하지 않고 중단
    └─ 통과 → 명세·소스 검사 → 참조 모듈 계산
    ↓
부모: 요청·정리 증거·파일·정수 빈도·합성 정답 검사
    ↓
완료 커밋 또는 실패/취소 보존
```

승인 범위는 입력 snapshot·프로파일·모듈 ID/버전·명세 해시·include_complete 설정·원시 코드 파일 해시·자원 한도를 포함한다. 변경된 설정이나 코드로 기존 승인을 사용할 수 없다. 승인 전에 worker를 시작하지 않으며 완료된 작업은 재실행하지 않는다.

부모 CLI·결과 검증기는 확장 모듈을 import하거나 통계를 다시 계산하지 않는다. 모듈 import는 품질 통과 후 worker 안에서 수행한다. 입력은 알려진 합성 fixture와 정확히 같아야 하므로 부모는 고정된 합성 정답을 검사할 수 있다. 이를 일반 실데이터 계산 정답의 독립 인증으로 확대하지 않는다.

`module_catalogue`에 staged 검사를 추가했다. 부모는 기존 전체 경로 검사를 유지하고, worker는 고정 `tools` 하위의 명세·소스만 검사한다. 허용 밖의 조상 폴더를 조회하거나 원본·승인 DB에 권한을 추가하지 않는다. 명세 검사 CLI만으로 네이티브 실행이 활성화되는 것은 아니며, 이 별도 CLI에서 새 승인한 합성 실행만 연결한다.

## 사용자 Windows 실행 명령

관리자 PowerShell에서 프로젝트 루트·Python 3.10 환경으로 실행한다.

```powershell
python -m src.agents.reviewed_module_cli create-demo `
  --name reviewed-module01 `
  --memory-mib 2048 --processes 2 --wall-seconds 60

python -m src.agents.reviewed_module_cli next --name reviewed-module01
python -m src.agents.reviewed_module_cli next --name reviewed-module01
python -m src.agents.reviewed_module_cli approve --name reviewed-module01
```

표시된 MODULE-... 문구를 직접 입력하고 새 승인 ID를 사용한다.

```powershell
python -m src.agents.reviewed_module_cli next `
  --name reviewed-module01 `
  --approval-id "새로_반환된_approval_id"

python -m src.agents.reviewed_module_cli status --name reviewed-module01
```

정상 완료는 completed / module_status passed / synthetic_only true다. 결과 폴더 pipeline에는 quality 보고서·row_missing/missing_row_histogram.csv·pipeline_result.json이 있다. 합성 입력 8행에서 결측 특징이 0개인 행은 7개, 1개인 행은 1개다. `--exclude-complete`로 새 실행을 생성하면 0개 항목을 제외한다. 실행 중 설정을 바꾸지 않는다.

필요하면 `cancel --name reviewed-module01`로 취소한다. 실패한 이름·승인을 자동 재사용하지 않으며 비격리 대체 실행은 없다.

## 검증 범위와 다음 단계

신규 실행 연결 테스트 11개와 등록 테스트 8개, 합계 19개가 통과했다. 승인 미지급·설정 변경·다른 CSV·변조된 결과·품질 중단·자원 증거 누락·자원 초과·실행 중 취소를 검사했다. 고정 복사 목록만으로 새 Python 프로세스를 부팅하는 검사도 통과했다. OS 운송은 모의했으며 실제 AppContainer 실행은 사용자 확인 전이다.

```powershell
python -m unittest tests.agents.test_reviewed_module tests.agents.test_module_catalogue -v
```

## 사용자 Windows 실행 확인

2026-10-17 사용자 제공 결과에서 `reviewed-module01`의 `completed / module_status: passed / synthetic_only: true / cancel_requested: false`를 확인했다. 적용된 작업 ID는 `c0f6841b7f7d4b40a76a514d4dbaeaa0`이며 결과는 `logs/agents/reviewed_module/runs/reviewed-module01/attempts/c0f6841b7f7d4b40a76a514d4dbaeaa0/pipeline`이다. 위 사용자 확인 전 표기는 구현 당시 이력이다. 이 갱신에서 worker를 재실행하거나 승인 기록을 변경하지 않았다.

다음 연결 범위는 [검토된 CSV 모듈의 범용 파이프라인 설계](../design/REVIEWED_CSV_PIPELINE_DESIGN.md)에 정리했다. 임의 업로드·동적 import·새 의존성 자동 설치·LLM 등록 승인·외부 전송은 추가하지 않는다. 합성 성공이 실데이터 실행 승인을 대신하지 않는다.
