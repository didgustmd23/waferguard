# 에이전트 문서 안내

갱신일: 2026-10-18. 문서를 처음 읽을 때와 현재 구현 상태를 확인할 때의 진입점입니다. 상세 계약·명령·과거 검증 이력을 분리하며, 소스나 실행 동작은 이 문서 정리로 변경하지 않습니다.

## 먼저 읽을 문서

1. [구성도](ARCHITECTURE.md): 전처리 코어·에이전트·ML·DL의 역할과 신뢰 경계.
2. [기초 설계](AGENT_BASIC_DESIGN.md): 목표·승인·상태·LLM 정보 제한.
3. [상세 설계](AGENT_DETAILED_DESIGN.md): 실행 계약·보안 통제·결과 검증 기준.
4. 아래 작업별 사용법: 실제 실행할 기능만 선택해서 확인.

기초·상세 설계의 날짜별 구현 메모와 `history/`의 “다음 단계”는 당시 기록입니다. 현재 지원 여부는 이 안내와 해당 기능의 최신 사용법을 먼저 확인합니다.

## 현재 구현 상태

### 로컬 Qwen 연결 — 합성 제안 호출과 질문 지침 보완

2026-10-18 사용자 환경에서 Ollama `qwen3.5:9b` 설치와 Python `ollama.chat`의 합성 한국어 응답을 확인했습니다. LangChain·LangGraph 연결이나 실제 데이터 해설 검증은 아직 아닙니다.

`src/agents/llm_proposal_contract.py`는 모델 응답을 `llm_proposal_v1` JSON으로 검사합니다. 필드는 `schema_version`, `status`, `pipeline`, `questions`만 허용합니다. `proposed`는 기존 표형 또는 WM 계획 형식을 사용하며, `needs_clarification`은 `pipeline: null`과 확인 질문만 반환합니다.

- 로컬 호출자가 정한 논리 입력 계약과 해당 요청의 모듈 ID·정확한 버전 허용 목록을 따릅니다. 모델이 입력·권한 범위를 정하지 않습니다.
- 기존 등록·설정·포트·순서 검증을 재사용하며, 표형 분석·변환·보고는 선행 품질 검사에 의존해야 합니다.
- 중복 JSON 키·비표준 숫자·64 KiB 초과 응답·추가 승인/명령 필드를 거부합니다. 설명문에서 JSON을 임의 추출하지 않습니다.
- 정적 계약 통과는 실제 입력의 품질·사용자 의도 충족·실행 안전성·사용자 승인을 보증하지 않습니다. 질문과 설명도 신뢰할 수 없는 모델 출력입니다.
- 계약 검사 함수 자체는 모델 호출·원본 읽기·승인 발급·worker 실행을 하지 않습니다. 별도 합성 제안 호출부를 연결했으며 실제 모델 제안 품질은 사용자 터미널에서 확인합니다.

`llm_config.py`는 `configs/agents/llm/local_ollama.json`을 읽습니다. `local_ollama.py`는 검증된 loopback 주소의 Ollama HTTP API에 한 번만 요청하고, `llm_proposal.py`는 공개 명세와 요청 문장을 구성한 뒤 응답을 기존 계약으로 검증합니다. `llm_proposal_cli.py`는 미리보기·제안과 별도 확인을 받는 로컬 보관·복원을 제공합니다. 데이터 파일 입력·분석 결과 저장·실행 기능은 없습니다. 표형 품질·결측·상관 모듈 3개에 대해 합성 요청 3종 또는 사용자가 직접 작성한 문장을 지원합니다. WM 제안·기능별 데이터 공개·LangChain·LangGraph 연결은 후속 범위입니다.

모델 설정은 아래 JSON에서 변경합니다. CLI의 `--config`로 다른 설정 파일을 선택할 수도 있습니다. 설정 파일 경로는 모델에 전달하지 않습니다.

| 설정 | 의미 |
| --- | --- |
| `model` | Ollama에 설치된 로컬 모델의 정확한 이름·태그 |
| `host`, `port` | 로컬 서버 주소·포트. `127.0.0.1` 또는 `::1`만 허용 |
| `timeout_seconds` | 소켓 대기 한도. 전체 처리 시간을 강제 종료하는 OS 제한은 아님 |
| `think` | 해당 모델이 지원할 경우 추론 출력 사용 여부 |
| `temperature` | 생성의 무작위성 정도. 낮춰도 의미적 정답이나 완전한 재현성을 보장하지 않음 |
| `num_ctx` | 모델 문맥 길이. 늘리면 메모리 부담이 커질 수 있음 |
| `num_predict` | 응답 생성 토큰 상한. 부족해서 응답이 잘리면 성공 처리하지 않음 |

외부 주소 금지·설정과 응답 크기 상한·허용 범위는 코드의 보호 규칙으로 유지합니다. 설정만 바꾸어 임의 외부 서버·추적 SDK·자동 fallback을 활성화하지 않습니다. 모델명에 cloud를 포함하면 거부하지만 이것만으로 서버 자체의 외부 통신을 차단하거나 인증하지는 않습니다. 신뢰된 로컬 Ollama에서 설치된 로컬 모델을 사용하고, 서버의 클라우드 기능·네트워크 정책은 별도로 확인해야 합니다. 상세 예외·프롬프트·모델 원문 오류를 파일이나 tracing으로 자동 기록하지 않습니다. 설정을 공유할 때도 키·개인 경로를 추가하지 않습니다.

프로젝트 루트의 Python 환경에서 먼저 전송 내용을 미리 봅니다. `--preview`는 모델을 호출하지 않습니다.

```powershell
python -m src.agents.llm_proposal_cli --example missing --preview
```

내용 확인 후 사용자가 합성 제안만 호출합니다. 이 확인 옵션은 전처리 실행 승인이 아닙니다.

```powershell
python -m src.agents.llm_proposal_cli `
  --config configs/agents/llm/local_ollama.json `
  --example missing --confirm-local-model-call
```

다른 합성 요청은 `--example correlations`와 `--example clarify`입니다. 정상 계약은 `proposal_validated`, 잘못된 응답은 `PROPOSAL_CONTRACT_REJECTED`로 종료하며 재시도·자동 수정을 하지 않습니다. 계약 통과만으로 사용자 의도를 충족했다고 평가하지 않으며, 모호한 요청에서 질문을 반환했는지도 사람이 확인합니다. 모델이 직접 도구 호출을 반환하면 실행하지 않고 거부합니다.

#### 직접 작성한 요청 문장

셸 명령 이력에 요청을 넣지 않도록 `--request "문장"` 인자 대신 터미널에서 한 줄을 입력합니다. 원본·집계·그림·보고서를 자동 첨부하지 않지만, 사용자가 문장에 직접 적은 정보는 그대로 전송 대상입니다. 자동 익명화·비밀정보 탐지·추측 불가능성을 보장하지 않습니다. 원본·통계·인증정보를 문장에 넣지 말고 표시 내용을 확인하세요. 화면·터미널 녹화에도 문장이 남을 수 있습니다. 이 기능은 임의 데이터 공개 옵션을 구현한 것이 아닙니다.

```powershell
python -m src.agents.llm_proposal_cli --interactive-request --preview
```

위 명령은 입력한 문장·공개 명세·옵션과 수신 주소를 표시하고 종료합니다. 모델에 전달하려면 아래 명령을 실행하고, 문장 입력 후 표시된 실제 내용을 확인한 뒤 `SEND`를 직접 입력합니다. 다른 값을 입력하면 `not_sent`로 종료하며 모델을 호출하지 않습니다.

```powershell
python -m src.agents.llm_proposal_cli `
  --interactive-request --confirm-local-model-call
```

`--example`과 `--interactive-request`는 함께 사용할 수 없습니다. 직접 입력은 대화형 터미널에서만 지원하고 파이프·리다이렉트 입력은 거부합니다. 문장은 4000자·UTF-8 8 KiB 이하이며 빈 값·터미널 제어 문자는 거부합니다. 호출 확인 옵션이 없으면 직접 입력도 미리보기만 합니다. 전송 확인은 정보 전달에 대한 이번 CLI 확인이지 전처리 실행 승인이 아닙니다. 결과는 `synthetic_only: false`, `proposal_only: true`, `execution_approved: false`, `worker_started: false`입니다. 자유 문장이 실제 또는 합성 데이터인지 자동 판정하지 않으므로 합성 전용이라고 표시하지 않습니다. 전송 도중 중단되면 `interrupted`, `delivery_unknown: true`로 알리고 자동 재시도하지 않습니다.

예시 문장: “품질 검사를 먼저 하고 결측률과 상관관계를 확인해 주세요. 데이터를 변경하지 마세요.” 요청 내용이 기존 허용 범위보다 넓어도 모듈 권한·입력 계약·실행 권한을 늘리지 않습니다. 확인 질문의 후속 답변을 기억하는 다중 턴 기능은 아직 없으므로 다음 요청에는 목적과 조건을 함께 적습니다.

#### 제안의 로컬 저장·복원

`llm_proposal_store.py`는 제안 내용을 현재 계약으로 다시 검사한 뒤 `logs/agents/llm_proposals/<이름>.json`에 새 파일로 보관합니다. 이 폴더는 Git에서 제외됩니다. 기본적으로 자동 저장하지 않으며 사용자가 `--save-name`을 지정한 경우에만 생성한 실행안을 표시하고 별도 `SAVE` 확인을 받습니다. 이름은 소문자로 시작하는 64자 이하 영문·숫자·밑줄 ID이며 모델이 만든 경로나 파일명을 사용하지 않습니다. 같은 이름은 덮어쓰지 않습니다.

```powershell
python -m src.agents.llm_proposal_cli `
  --interactive-request --confirm-local-model-call `
  --save-name user_eda01
```

문장 입력 → 전송 미리보기 → `SEND` → 모델 제안 → 제안 검토 → `SAVE` 순서입니다. 합성 예제도 `--example missing --confirm-local-model-call --save-name example_eda01`로 같은 저장 확인을 받습니다. 질문만 반환한 `needs_clarification`, 취소·미리보기·실패 응답은 실행안으로 저장하지 않습니다. 보관 파일은 제안·내용 체크섬·계획 지문·권한 없음 선언만 포함하며 요청 원문·서버 원문·모델 설정·승인 토큰을 별도 필드로 저장하지 않습니다. 모델이 제안에 요청의 표현을 반영할 수 있으므로 보관 파일 전체가 무조건 비민감하다는 뜻은 아닙니다. 사용자 전용 파일 권한을 새로 설정하는 기능도 아니므로 로컬 계정·디스크 접근 통제는 별도로 유지하세요.

복원은 모델 설정 로드·LLM 재호출 없이 수행합니다.

```powershell
python -m src.agents.llm_proposal_cli --load-name user_eda01
```

지원 버전·정확한 필드·크기·경로 링크·내용 체크섬·현재 모듈 ID/버전·설정·입력·의존성·계획 지문을 다시 확인합니다. 통과하면 `proposal_restored`, `model_called: false`, `execution_approved: false`, `worker_started: false`입니다. 파일을 바꾸거나 불완전하게 저장했으면 실패하며 자동 수리·재실행하지 않습니다. 저장 중 중단돼 생긴 불완전 파일도 덮어쓰지 않고 새 이름을 사용합니다.

체크섬·계획 지문은 서명·작성자 인증·익명화가 아니며 파일과 지문을 함께 재작성할 수 있는 로컬 사용자에 대한 위변조 방어를 보장하지 않습니다. 동일 버전으로 구현 소스만 변경된 모든 경우를 이 계획 지문이 탐지하는 것도 아닙니다. 실제 입력·코드·자원에 결합한 기존 실행 승인 검증은 후속 연결에서 따로 수행해야 합니다. 현재 복원은 승인 발급·worker 실행·계산 정확성 검증을 하지 않습니다.

#### 제안과 입력을 결합한 기존 승인 작업 준비

사용자가 `user_eda01`의 실제 저장·복원을 확인했습니다. 후속 `llm_execution_candidate.py`는 복원한 제안과 사용자가 지정한 snapshot·프로파일·자원 한도를 기존 입력 어댑터 바인더로 결합합니다. 새 분석기나 별도 승인 DB 규격을 만들지 않으며 `pipeline_approval_cli.create`를 재사용합니다. 현재 바인더 등록은 `tabular_csv_v1`뿐입니다. WM 입력을 이 경로로 임의 처리하지 않습니다.

먼저 기존 합성 입력으로 로컬 미리보기만 확인합니다. 아래 snapshot·profile은 앞선 `module-demo01`에서 준비한 합성 파일이며 현재 PC에 없으면 다른 검증된 snapshot·프로파일을 명시해야 합니다. 원본 CSV 경로 대신 snapshot 폴더를 지정합니다.

```powershell
python -m src.agents.llm_execution_candidate_cli preview `
  --proposal-name user_eda01 `
  --snapshot-dir logs/agents/module_pipeline/examples/module-demo01/snapshot `
  --profile logs/agents/module_pipeline/examples/module-demo01/profile.json `
  --memory-mib 2048 --processes 2 --wall-seconds 300
```

`candidate_preview`는 snapshot 바이트 정합성·프로파일·등록 모듈·코드·자원 결합을 확인한 상태입니다. CSV 값을 분석하거나 승인·worker 실행을 한 것이 아닙니다. 표시되는 컬럼·품질 설정·상세 경로와 지문은 로컬 전용이며 LLM에 다시 보내지 않습니다. 품질 검사 설정 `{}`는 선택한 프로파일 규칙으로 구체화되므로 제안의 `proposal_plan_signature`와 유효 실행 계획의 지문이 다를 수 있습니다. 바인더가 제안 설정을 적용했을 때의 정확한 `effective_execution`을 사람이 확인하며, 프로파일을 약화시키는 설정을 별도 검토 없이 허용하지 않습니다.

준비하려면 같은 입력으로 `create`를 호출하고 새 작업 이름을 지정합니다. CLI가 현재 내용을 다시 표시한 뒤 사용자가 `CREATE`를 입력해야 합니다. 실행 승인이 아닌 로컬 작업 생성 확인입니다.

```powershell
python -m src.agents.llm_execution_candidate_cli create `
  --proposal-name user_eda01 --name llm_eda_run01 `
  --snapshot-dir logs/agents/module_pipeline/examples/module-demo01/snapshot `
  --profile logs/agents/module_pipeline/examples/module-demo01/profile.json `
  --memory-mib 2048 --processes 2 --wall-seconds 300
```

확인 후 제안·snapshot·프로파일·코드·자원이 바뀌면 중단합니다. 모델은 경로·작업 이름·자원을 선택하지 않습니다. 새 `logs/agents/llm_execution_candidates/<이름>/`에 후보·pipeline·profile 복사 설정을 배타 저장하고 기존 생성기로 `logs/agents/pipeline_approval/runs/<이름>/`의 초기 작업 상태와 빈 승인 DB를 준비합니다. 생성기의 최종 입력 결합이 확인 내용과 같을 때만 `CREATED.json` 영수증을 남깁니다. 후보 생성은 원본 데이터 복사·분석·ACL 확장·실행 승인 발급·사전 검사 실행·worker 시작을 하지 않습니다. 모든 기록은 Git 제외 로그 폴더에만 저장합니다.

정상 결과는 `execution_candidate_created`, `execution_approval_issued: false`, `model_called: false`, `worker_started: false`입니다. 생성 중 실패하면 부분 후보·미승인 작업이 남을 수 있으며 자동 삭제·재시도·덮어쓰기를 하지 않습니다. 기존 작업 생성 완료 후의 다음 단계는 같은 이름으로 기존 `pipeline_approval_cli next`를 호출하는 것이며, 사전 검사·승인 요청·사용자 승인·격리 실행은 각각 기존 경계를 그대로 따릅니다. 현재 단계에서 `approve`·승인 소비·실행을 자동 호출하지 않습니다.

이 연결은 새 합성 테스트 9개로 읽기 전용 미리보기·확인 내용 변경 중단·기존 작업 생성 재사용·빈 승인 DB·중복 거부·사용자 취소를 확인했습니다. 실제 입력에 대한 후보 준비·실행은 사용자 터미널에서 별도로 검증합니다.

2026-10-18 사용자 실행에서 `llm_eda_run01` 생성·preflight·승인 요청은 정상 완료했지만, 승인 소비 뒤 `TOOL_RUNTIME_FAILED`, `worker_started: false`로 중단했습니다. 로컬 기록은 입력 staging 완료, runtime·tools·출력·scratch 파일 없음, `transport.json`에 `profile_cleanup: true`와 launch 기록 없음입니다. 따라서 코드 목록·런타임 목록·AppContainer 생성·초기 ACL 부근의 실패로 좁혔지만 원래 예외가 저장되지 않아 정확한 원인은 확정하지 못했습니다. 현재 환경의 런타임 목록 읽기 전용 검사는 통과했으나 과거 실패 시점의 성공을 증명하지는 않습니다.

`module_pipeline_transport.py`에 고정 준비 단계와 오류 분류·숫자 errno/winerror만 남기는 `diagnostic`을 추가했습니다. 경로·원문·명령 출력·데이터는 넣지 않으며 profile 정리 예외도 원래 진단과 분리하고 정리 성공으로 표시하지 않습니다. 이 보완은 원인 수집 개선이지 해당 환경 오류의 해결 확인이 아닙니다. 운송 모의 테스트 7개와 기존 승인·후보 연결 회귀를 합한 21개가 통과했습니다. 실제 AppContainer 재실행은 하지 않았습니다.

운송 소스는 승인 코드 지문에 포함되므로 이전 작업·승인은 재사용하지 않습니다. 실패한 `llm_eda_run01`과 소비된 승인은 유지하고 같은 입력으로 새 이름 `llm_eda_run02`를 준비한 뒤 기존 사전 검사·새 사용자 승인 절차를 수행합니다. 다시 실패하면 해당 run의 `attempts/<작업 ID>`와 같은 ID의 `logs/agents/pipeline_approval/workers/<작업 ID>/transport.json`에서 `diagnostic`만 확인합니다. 다른 최신 worker의 진단과 혼동하지 않습니다. 승인 ID·DB·입력·전체 transport 원문을 외부로 전달하지 않습니다. 자동 재시도·보안 제한 해제·기존 상태 초기화는 하지 않습니다.

#### 일반 PowerShell의 출력 폴더 권한 보완

후속 `llm_eda_run02`는 worker 시작 전 `ACL_OUTPUT`에서 실패했고, 사용자가 관리자 PowerShell에서 실행한 새 작업은 `completed`, `worker_started: true`로 완료했습니다. 이는 권한 설정 단계 문제를 뒷받침하지만, 관리자 실행만으로 세부 명령의 원인을 확정하지는 않습니다.

`writable_directory_acl.py`를 추가하고 모듈 파이프라인 운송 계층에 연결했습니다. 새 작업의 `output/result`·`scratch/task`가 링크 없는 빈 폴더인지 확인하고, 소유자 SID와 현재 프로세스의 사용자 SID가 같을 때만 호출자 자신에게 비상속 `WO`를 추가합니다. 기존 명시 권한은 보존하며 폴더 소유자를 바꾸지 않습니다. AppContainer에는 기존처럼 `M`과 Low 레이블만 적용합니다. 프로젝트·원본·입력·승인 DB의 권한은 변경하지 않으며, 일반 사용자에게 실행에 필요한 모든 권한이 항상 확보된다고 보장하지 않습니다.

Low 레이블 변경에는 `WRITE_OWNER`가 필요합니다([Microsoft SECURITY_INFORMATION](https://learn.microsoft.com/en-us/windows/win32/secauthz/security-information)). `icacls /grant`는 기존 명시 권한에 추가하고 `WO`는 소유자 쓰기 권한입니다([Microsoft icacls](https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/icacls)). 호출자 WO는 해당 빈 폴더에 남으며 하위 파일로 상속하지 않습니다. worker에 WO·FullControl을 추가하거나 자동 관리자 승격·보안 우회·실패 재시도를 하지 않습니다. 다른 전용 운송 경로의 ACL 동작은 이번 변경 대상이 아닙니다.

실패 시 `diagnostic.acl_operation`으로 `TARGET_CHECK`, `OWNER_CHECK`, `OWNER_MISMATCH`, `CALLER_WRITE_OWNER`, `WORKER_GRANT`, `LOW_INTEGRITY`, `WORKER_ACL`을 구분합니다. 경로와 명령 출력은 넣지 않습니다. 실패하면 worker를 실행하지 않고 기존 승인 소비·실패 기록은 유지합니다.

검증: `python -m unittest tests.agents.test_writable_directory_acl tests.agents.test_module_pipeline_transport tests.agents.test_pipeline_approval tests.agents.test_llm_execution_candidate -q` — 31개 통과. 제한 실행 환경의 임시 폴더 접근 거부로 같은 테스트를 허용된 환경에서 재실행했습니다. 자동 테스트의 ACL 명령·AppContainer 실행은 모킹했으며, 실제 실행 검증은 아래 사용자 실행 기록과 구분합니다.

기존 `tests.agents.test_module_pipeline`·`tests.agents.test_module_pipeline_execution`까지 확장한 관련 회귀는 총 57개 통과했습니다. 전체 에이전트 테스트 전체집합의 재실행 기록은 아닙니다.

2026-10-18 사용자가 일반 PowerShell에서 실행한 `llm_eda_normal01`은 `completed`, `worker_started: true`로 완료했습니다. 이후 로컬 기록을 읽기 전용으로 확인하여 완료 상태·`run_pipeline` 단계, 승인 `consumed`와 저장된 승인 범위 일치, worker 종료 코드 0, AppContainer 프로필 정리 성공, Job 정리 확인·잔여 프로세스 0을 확인했습니다. 품질·결측·상관분석 3단계의 결과는 기존 `read_pipeline_result`로 다시 검사하여 `passed`였고, 완료 표식과 worker 영수증도 존재했습니다. worker 재실행·승인 재발급·DB 변경은 하지 않았습니다.

따라서 저장된 LLM 제안 → 입력 선택·실행 후보 → 사전 검사 → 사용자 승인 → 일반 권한 AppContainer 실행 → 결과 검증의 합성 CSV 연결을 확인했습니다. 일반 권한 여부는 사용자 실행 설명에 근거하며 토큰 상승 여부를 별도로 수집한 검사는 아닙니다. 모든 사용자·드라이브·입력 형식에서의 성공이나 새로운 WFP 차단 검증을 의미하지 않습니다. 승인 ID·상세 데이터·경로·전체 원문은 이 검증 기록에 옮기지 않습니다.

일반 PowerShell에서 아래 새 합성 작업을 준비한 뒤 `CREATE`를 직접 확인합니다. 이름이 이미 있으면 다른 새 이름을 사용하고, 관리자 실행으로 만든 기존 작업이나 소비된 승인을 재사용하지 않습니다.

```powershell
python -m src.agents.llm_execution_candidate_cli create `
  --proposal-name user_eda01 --name llm_eda_normal01 `
  --snapshot-dir logs/agents/module_pipeline/examples/module-demo01/snapshot `
  --profile logs/agents/module_pipeline/examples/module-demo01/profile.json `
  --memory-mib 2048 --processes 2 --wall-seconds 300

python -m src.agents.pipeline_approval_cli next --name llm_eda_normal01
python -m src.agents.pipeline_approval_cli next --name llm_eda_normal01
python -m src.agents.pipeline_approval_cli approve --name llm_eda_normal01

python -m src.agents.pipeline_approval_cli next `
  --name llm_eda_normal01 `
  --approval-id (Read-Host "이번 실행의 approval_id").Trim()

python -m src.agents.pipeline_approval_cli status --name llm_eda_normal01
```

통신은 [Ollama Chat API](https://docs.ollama.com/api/chat)의 `stream: false`, `think`와 [구조화 출력](https://docs.ollama.com/capabilities/structured-outputs)의 `format`을 사용합니다. 초기 연결 검증은 표준 라이브러리 HTTP로 분리하며 LangChain·LangGraph는 아직 설치·호출하지 않습니다.

사용자와 논의한 기능별 정보 공개 설정은 후속 설계 범위입니다. 사용자가 전달 항목과 실제 내용을 확인·허용하고 시스템은 승인 범위를 강제하는 방향이며, 아직 공개 제어 계층을 구현한 상태는 아닙니다. 현재 연결 시험은 합성 요청과 데이터가 없는 등록 명세만 대상으로 합니다. LLM 전달 허용과 외부 추적 저장 허용은 구분합니다.

사용자 터미널의 실제 Qwen 합성 요청 결과에서 결측·상관분석 제안은 요청한 모듈과 품질 의존성을 충족했습니다. 모호한 요청은 실행안 없이 질문했지만, 요청하지 않은 상관분석을 가정하고 지원하지 않는 계수 선택을 물었습니다. 따라서 JSON 계약 통과와 질문의 적절성은 구분합니다. 작업 목적·검사 종류부터 확인하고, 미선택 작업을 가정하거나 고정 동작을 변경 가능한 옵션으로 질문하지 않도록 공통 지침과 상관 모듈 설명을 보완했습니다. 합성 예제 이름에 따라 정답 질문을 강제 반환하지는 않습니다.

사용자가 프롬프트 보완 후 `clarify`를 재실행하여 실행안 없이 검사 종류부터 묻는 응답을 확인했습니다. 실제 데이터나 전처리 실행 결과는 아닙니다.

사용자가 직접 입력한 품질→결측·상관 요청도 제안 계약을 통과했고 두 분석의 품질 단계 의존성과 승인·실행 없음 상태를 확인했습니다. `synthetic_only: false`는 사용자 문장을 합성 전용으로 보증하지 않는다는 뜻이지 데이터 파일을 읽었다는 뜻은 아닙니다.

검증 명령: `python -m unittest tests.agents.test_llm_proposal_store tests.agents.test_llm_proposal tests.agents.test_llm_proposal_contract -v` — 설정·운송·CLI·계약·직접 입력·저장/복원 합성 테스트 39개 통과. 자동 테스트는 실제 모델·데이터·worker를 호출하지 않았으며 사용자 환경의 실제 제안 저장·복원은 별도로 확인합니다.

| 영역 | 현재 범위 | 남은 범위 |
| --- | --- | --- |
| 공통 실행 관리 | 실행 계획·상태 저장·SQLite 승인 1회 소비·취소·완료 검증, 로컬 LLM 제안 저장·복원·선택·CSV 실행 후보 연결과 합성 격리 실행 확인 | 확인 질문의 다중 턴 연결, 전체 사용자 동선 통합 |
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
