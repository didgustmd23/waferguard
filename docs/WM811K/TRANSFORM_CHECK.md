# WM-811K 입력 변환·패턴 보존 검사

## 현재 구현 범위

[변환 함수](../../src/wafer_dl/transform.py), [개발 구간 검사](../../src/wafer_dl/check_transform.py), [설정](../../configs/wm811k/transform.json)을 사용합니다. 실제 Train/Validation 맵 검사, 캐시·Dataset, CNN 학습과 동결 모델의 Test 평가는 완료했습니다. 결과와 Test 사용 범위는 [캐시·학습·평가 기록](CACHE_DATASET.md)을 확인하세요.

이번 단계는 PyTorch 없이 NumPy로 실행합니다. PyTorch를 설치하지 못하면 다른 보간으로 대체하는 방식이 아니라, `numpy_half_pixel_nearest_v1`을 명시한 독립 변환 구현입니다. Dataset은 저장된 categorical 맵에서 같은 2채널을 생성합니다.

## 변환 계약

1. 원본 uint8 맵의 `0=웨이퍼 밖`, `1=정상 다이`, `2=불량 다이`를 유지합니다.
2. 기본 목표는 128×128입니다. 원본을 crop하거나 방향을 정렬하지 않습니다.
3. 배율은 `min(목표 높이/원본 높이, 목표 너비/원본 너비)`입니다.
4. 새 길이는 `floor(원본 길이×배율+0.5)`로 계산하고 1 이상·목표 길이 이하로 제한합니다.
5. 최근접 좌표는 `floor((출력 인덱스+0.5)×원본 길이/새 길이)`입니다. 정수 연산으로 좌표를 고정하며 선형 보간을 하지 않습니다.
6. 여백은 0으로 중앙 padding하고, 홀수 여백의 추가 픽셀은 아래·오른쪽에 둡니다.
7. 변환 결과에서는 채널 0=`맵==2`, 채널 1=`맵>0`입니다. NumPy 변환기는 contiguous float32 `[2,H,W]` 배열을 제공하고 Dataset은 이를 Tensor로 연결합니다.

상태별 채널값은 외부 `[0,0]`, 정상 `[0,1]`, 불량 `[1,1]`입니다. `/255`, ImageNet 평균·표준편차, 중앙값 대치, 증강은 적용하지 않습니다.

설계에서 제안한 PyTorch `nearest-exact`와 일반 `nearest`는 구분해야 합니다. [PyTorch 공식 문서](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html)는 `nearest-exact`를 PIL/Scikit-Image 최근접 방식과 대응시키며 일반 `nearest`와 차이를 설명합니다. 현재 구현은 NumPy 좌표 규칙을 자체 버전으로 고정한 것이며, 설치되지 않은 PyTorch와의 런타임 정합성을 검증했다고 주장하지 않습니다.

## 실행

프로젝트 루트의 PowerShell에서 실행합니다.

```powershell
python -m src.wafer_dl.check_transform `
  --split-dir data/splits/wm811k/group_v2 `
  --config configs/wm811k/transform.json `
  --output-dir reports/wm811k/transform_v2
```

분할 산출물의 canonical_root를 사용합니다. canonical 폴더를 옮겼다면 `--canonical-dir "현재 실제 경로"`를 추가하세요. 원본 manifest 내용이 일치해야 하며 잘못 연결한 묶음은 거부합니다.

출력은 입력 바깥의 존재하지 않는 새 폴더여야 합니다. 기존 결과는 덮어쓰지 않습니다. 최초 metadata 검증·임시 DB 색인에는 진행 출력 없이 시간이 걸릴 수 있으며 이후 개발 구간 맵을 1,000건 단위로 안내합니다.

## 읽는 범위와 출력

- 분할·원본 전체 metadata를 대조하여 라벨·참조·충돌 제외·클래스 집계·구간 연결을 검사합니다.
- 숫자 맵은 **Train/Validation만** 복원하고 내용 해시·크기·다이 수를 확인합니다. Test metadata는 분할 계약 검사에 포함하지만 **Test 배열은 읽지 않습니다.**
- shard별로 열고 맵 하나씩 변환합니다. 전체 float32 입력을 RAM에 적재하지 않습니다.
- 원본 파일·분할 manifest·NPZ는 수정하지 않습니다. pickle을 열거나 학습하지 않습니다.

| 출력 | 역할 |
| --- | --- |
| preservation.csv | 행별 resize·padding 위치와 원본/변환 비율·소실 여부 |
| transform_check.json | 설정·분할 계약 ID·구간/클래스별 집계·상태 |
| transform_check.md | 결과 해석과 비교 그림 |
| examples.png | 클래스별 첫 사례, 소형·대형·극단 종횡비, 있으면 첫 불량 소실 사례 |
| SUCCESS.json | 검사 산출물 저장 완료 표식. 통과·학습 준비 완료와는 다름 |

그림은 대표성을 보장하지 않습니다. 보고서 PNG를 학습 입력으로 재사용하지 않습니다.

## 판정·해석

- `transform_check_passed`: 검사한 개발 맵에 불량 영역 전체 소실·유효 영역 전체 소실이 없음. **형태 보존 시각 검토는 여전히 필요**합니다.
- `transform_review_required`: 두 소실 조건 중 하나라도 발생. 정상적인 진단 완료이며 CLI 종료 코드는 2입니다. 자동 해상도 변경·예외 표본 삭제는 하지 않습니다.
- 실패 코드: 파일·연결·참조·구조 문제 등. 완료 표식이 없는 부분 결과는 사용하지 않습니다.

불량 비율의 절대 변화 `0.01`은 **1%p**입니다. resize 전후 픽셀 수가 달라지므로 불량 다이 개수의 동일성은 요구하지 않습니다. 평균뿐 아니라 행별 변화와 극단 사례도 살펴보세요. 비율 변화에 대한 승인 임계값은 아직 추가하지 않았습니다.

검토 필요라면 Train/Validation 사례로 원인을 확인한 후 공통 입력 해상도 변경 등 별도 변환 버전을 결정합니다. 표본마다 유리한 변환을 골라 적용하거나 Test 그림을 보고 설정을 선택하지 않습니다.

## 노트북 확인

[analysis.ipynb](../../reports/wm811k/analysis.ipynb)의 **8. 입력 변환 검증 결과**에서 저장된 결과 표와 비교 그림을 읽습니다. 이 셀은 변환 검사를 자동 실행하지 않습니다. 다른 출력 경로를 썼다면 `TRANSFORM_CHECK_DIR`을 바꾸세요.

상세 집계·그림·행별 ID는 로컬 산출물입니다. 에이전트가 이 함수를 호출하더라도 외부 LLM·LangSmith에 전송하지 않으며 최소 반환 상태도 승인된 Provider DTO와 별도로 취급합니다.

## 합성 테스트

```powershell
python -m unittest tests.test_wafer_transform -v
```

좌표 선택·크기 반올림·홀수 padding·2채널 값·소실 검출·잘못된 설정 차단·분할 변경 거부·Test NPY 미접근·원본 보존을 검사합니다.
