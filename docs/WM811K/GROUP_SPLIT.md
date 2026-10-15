# WM-811K 충돌 제외·그룹 분할 실행

## 적용 정책

- 같은 맵에 서로 다른 명시적 라벨이 있으면 해당 맵의 **모든 행**을 파생 manifest에서 격리합니다. 미라벨도 포함하며 원본 파일은 삭제·수정하지 않습니다.
- 남은 행에서 Lot·동일 맵 연결 그룹을 다시 구성합니다. 미라벨과 기존 격리 행도 연결 보호에 포함합니다.
- accepted 라벨 표본의 목표 비율은 Train/Validation/Test 약 70/15/15입니다. 그룹 보호와 클래스 지원이 정확한 비율보다 우선합니다.
- seed 42, 최대 32개 고정 후보에서 표본 비율 제곱편차 합과 클래스별 비율 제곱편차 평균의 합이 가장 작은 후보를 선택합니다. 동점은 후보 번호가 빠른 것을 선택합니다. 모델 성능·예측·Test 이미지는 선택에 사용하지 않습니다.
- 클래스별 독립 그룹 3개 이상, 최대 그룹 비율 20% 이하, 충돌 해소를 먼저 검사합니다. 실제 후보는 각 split에 9개 클래스가 모두 있어야 합니다. 필요조건만으로 후보 존재를 보장하지 않습니다.
- Validation/Test의 클래스 표본 10건 미만은 소표본 경고입니다. 10건 이상도 통계적 안정성을 보장하지 않습니다.
- 실패 시 그룹을 쪼개거나 후보 수·seed를 자동 변경하지 않습니다. 후보 탐색 실패가 분할의 수학적 불가능성을 증명하는 것은 아닙니다.

설정: [split.json](../../configs/wm811k/split.json) / 구현: [split.py](../../src/wafer_dl/split.py)

### 후보 균형 점수

제약조건을 통과한 후보는 아래 목적함수 **J**가 가장 작은 것을 선택합니다. GitHub Markdown의 기본 렌더링에서도 동일하게 보이도록 LaTeX 블록 대신 텍스트 수식으로 기록합니다.

~~~text
J = Σs (n_s / N - r_s)^2
  + (1 / 9) Σc Σs (n_cs / N_c - r_s)^2
~~~

| 기호 | 의미 |
| --- | --- |
| s | 분할 구간: Train·Validation·Test |
| c | 패턴 클래스: 9개 |
| N, n_s | 전체 라벨 표본 수, 구간 s의 전체 표본 수 |
| r_s | 목표 구간 비율: 0.70·0.15·0.15 |
| N_c, n_cs | 클래스 c 전체 표본 수, 그중 구간 s에 배정된 표본 수 |

첫 항은 **전체 표본의 구간 비율**이 목표와 얼마나 다른지의 제곱편차 합입니다. 예를 들어 Train 비율이 정확히 0.70이면 해당 오차는 0이고, 0.72이면 (0.72 - 0.70)^2 = 0.0004입니다.

두 번째 항은 **클래스별 구간 비율**의 제곱편차를 9개 클래스에 대해 평균 낸 값입니다. 전체 비율만 맞추고 희소 클래스가 특정 구간에 몰리는 후보를 피하기 위한 항입니다. 예를 들어 Donut 555건 중 Train에 439건이면 Train 비율은 약 0.791이고, 이 클래스·구간의 오차는 (0.791 - 0.70)^2입니다.

클래스 수로 나누는 1 / 9는 클래스가 많아질수록 두 번째 항이 단순히 커지는 것을 막습니다. J는 0에 가까울수록 목표 비율에 가깝지만, 성능 점수·누수 점수·확률이 아닙니다.

후보는 J를 계산하기 전에 다음 조건을 먼저 만족해야 합니다.

- 동일 Lot·동일 맵 연결 그룹이 여러 구간에 섞이지 않음
- 충돌 맵 그룹이 격리됨
- 각 클래스가 Train·Validation·Test에 모두 존재함
- 클래스별 독립 그룹 수와 최대 연결 그룹 비율 제한을 만족함

즉, **누수 방지와 클래스 지원이 먼저이고 J는 통과 후보 중 가장 균형 잡힌 분할을 고르는 기준**입니다. 전체 비율과 클래스별 비율을 동시에 완전히 맞출 수 없는 것은 그룹 단위 분할의 자연스러운 제약입니다.

## 실행 전

`wm_label_v2` canonical 묶음과 EDA 결과를 확인한 후 실행합니다. 실행 시 manifest와 NPZ 정합성을 다시 검사합니다. 원본 pickle은 열지 않습니다. 전체 배열 검사·색인·후보 배정·manifest 저장이므로 시간이 걸릴 수 있습니다.

출력은 원본 canonical 바깥의 새 폴더여야 합니다. 기존 결과는 덮어쓰지 않습니다. 설정 파일의 상대 경로는 현재 작업 디렉터리 기준이며, 아래 명령은 프로젝트 루트에서 실행합니다.

## PowerShell 명령

첫 경로를 **analysis.ipynb에서 확인한 canonical_dir과 동일한 실제 경로**로 바꾸세요.

```powershell
$canonicalDir = "data/processed/wm811k/canonical/canonical_실제폴더명"
python -m src.wafer_dl.split `
  --input-dir "$canonicalDir" `
  --config configs/wm811k/split.json `
  --output-dir data/splits/wm811k/group_v1 `
  --approve-conflict-exclusion
```

`--approve-conflict-exclusion`은 충돌 맵 전체 제외·그룹 분할 정책에 대한 명시적 승인입니다. 실행하지 않으면 데이터는 분할되지 않습니다.

노트북에서 이미 선택한 경로를 그대로 사용하려면 다음 셀을 **직접 실행**해도 됩니다. 노트북 커널의 Python 환경을 사용합니다.

```python
import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "src.wafer_dl.split",
     "--input-dir", str(canonical_dir),
     "--config", str(project_root / "configs/wm811k/split.json"),
     "--output-dir", str(project_root / "data/splits/wm811k/group_v1"),
     "--approve-conflict-exclusion"],
    cwd=project_root,
    check=True,
)
```

## 결과 확인

| 출력 파일 | 역할 |
| --- | --- |
| split_manifest.jsonl | 원본 전체 행에 split·그룹·제외 사유를 추가한 파생 색인 |
| split.json | 설정, 후보 점수, 선택 후보, 실제 클래스 수, 경고, canonical 연결 정보 |
| split_summary.csv | 클래스별 구간 표본 수·실제 전체 표본 비율 |
| split_summary.md | 사람이 읽는 로컬 결과 요약 |
| SUCCESS.json | 검증·저장 완료 후 마지막에 생성하는 완료 표식 |

`split_complete`이어도 `training_ready=false`입니다. 입력 변환·캐시·Dataset 검증이 남아 있습니다. 완료 표식이 없는 부분 출력 폴더는 사용하지 말고, 실패 원인을 확인한 후 새 폴더로 실행합니다.

### manifest 필드 주의사항

- `split`: train / validation / test / unlabeled / quarantined
- `split_group_id`: 연결 그룹의 결정적 순번. 개인 Lot 이름 대신 사용하지만 민감한 로컬 연결 정보입니다.
- `group_split`: 지도학습 배정이 있는 그룹의 구간. 미라벨도 이 값을 유지합니다. 라벨 표본이 전혀 없는 그룹은 null입니다.
- `exclusion_reasons`: 충돌 맵이면 `MAP_LABEL_CONFLICT`. 원본 quality_status·라벨·기존 reason_codes는 보존합니다. 따라서 accepted 원본 행도 split은 quarantined가 될 수 있습니다.
- `split_protocol_id`: manifest 내용과 설정을 바탕으로 정한 분할 계약 ID. 입력 원본을 인증하거나 소스 전체의 동등성을 보증하는 값은 아닙니다.
- `map_ref`: **원본 canonical 루트 기준** NPZ 참조입니다. split 폴더 기준으로 열면 안 됩니다. NPZ는 복사하지 않습니다. 이동 시 split.json의 canonical_root 연결을 별도로 관리해야 합니다.

미라벨은 학습 승인 대상이 아닙니다. 특히 `group_split=test`인 미라벨을 학습에 사용하면 평가 Lot 보호가 깨질 수 있습니다. 추후 Dataset은 `split=train` 등 명시적으로 허용한 행만 선택해야 합니다.

상세 결과는 로컬에서 검토합니다. 향후 에이전트의 외부 LLM에는 경로·집계·식별자·맵을 보내지 않으며 공통 함수의 최소 상태 반환도 승인된 전송 DTO와 구분합니다.

## 검증 테스트

```powershell
python -m unittest tests.test_wafer_split -v
```

합성 데이터만 사용해 설정 오류, 최소 클래스 지원, 불가능한 지원 조합, 거대 그룹 차단, 미라벨 연결, 원본 보존, 충돌 전체 격리, 비중첩, 재현성과 완료 표식을 검사합니다.
