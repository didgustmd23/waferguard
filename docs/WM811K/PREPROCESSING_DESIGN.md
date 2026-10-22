# WM-811K 딥러닝 전처리 상세 설계

작성일: 2026-10-09. 상태: **구현 전 설계안**. 실제 원본 구조·분포·메모리는 아직 확인하지 않았으며 아래 설정과 모듈은 구현 완료가 아니다.

기초 설계는 [WM811K_BASIC_DESIGN.md](WM811K_BASIC_DESIGN.md)를 따른다. 전처리·작업 관리 담당은 양현승, 딥러닝 모델·학습 담당은 이종수다. 완료한 SECOM 코드·모델·설정은 변경하지 않는다. 이 문서는 먼저 에이전트 없이 실행되는 전처리 코어를 정의한다.

## 1. 이번에 만들 범위와 초기 기준

| 항목 | 초기 설계 기준 | 이유·확정 시점 |
| --- | --- | --- |
| 과제 | 웨이퍼맵 1건의 9종 패턴 분류 | 정상 출하 승인이나 검사 이전 예측이 아님 |
| 구현 프레임워크 | PyTorch 제안 | Dataset·CNN 전달을 한 체계로 구성. 팀 환경 확인 후 버전 확정 |
| 지도학습 대상 | 유효한 명시적 라벨이 있는 맵 | 미라벨을 `None` 정답으로 만들지 않음 |
| 분할 | Train/Validation/Test 약 70/15/15 | Lot 비중첩이 정확한 비율보다 우선 |
| 중복 보호 | Lot과 동일 맵 연결 그룹의 비중첩을 우선 제안 | 실제 중복 구조 확인 후 실행 가능성 승인 |
| CNN 입력 | 불량 마스크·유효 다이 마스크 순서의 2채널 | 웨이퍼 밖과 정상 다이를 구분 |
| 크기 | 우선 128×128 | 임시 기준. Train/Validation의 패턴 보존·비용 확인 후 동결 |
| 기본 변환 | 종횡비 유지 resize + 중앙 padding | 찌그러짐·영역 절단을 최소화 |
| 기본 증강 | 비활성 | 증강 없는 baseline부터 확보 |
| 다음 증강 후보 | 90도 회전·반전의 D4 변환 | 보간 없는 변환만 비교. 방향 의미 확인 후 Train에서만 활성화 |
| 불균형 | 원래 분포 유지, Train 기반 loss 가중치 별도 전달 | Validation/Test까지 균형화하지 않음 |
| 에이전트 | 초기 필수 아님 | 같은 코어를 CLI와 향후 에이전트가 호출 |

이는 설계 기본안이지 성능이 검증된 최적값이 아니다. 맵 크기·희소 클래스 Lot 수·하드웨어를 확인하지 않고 최종 처리 시간을 약속하지 않는다.

## 2. 처리 흐름과 단계별 책임

```text
출처 확인·신뢰 승인
  → 원본 읽기·canonical 변환
  → 맵/라벨/ID/Lot 검사와 격리
  → 전체 데이터의 구조적 EDA·중복 연결 검사
  → Lot·중복 보호 split 확정
  → split manifest 저장·Test 사용 제한
  → 결정적 입력 변환·캐시
  → Train 전용 증강·가중치 + Dataset
  → 전달 계약 검사
  → CNN·ResNet18 학습 담당에게 전달
```

전처리는 표준 함수·설정 파일로 동작한다. 에이전트는 함수 호출·실행 상태·실패 안내만 담당하며 라벨·분할·제외 기준을 스스로 바꾸지 않는다. 클래스 매핑과 결정적 변환은 학습·추론에서 동일하게 재사용한다.

## 3. 원본 확보와 안전한 읽기

### 3.1 원본 보존

- 사용자가 승인한 배포본을 `data/raw/wm811k/`에 보존한다. 경로는 설정에서 받으며 파일명을 코어에 고정하지 않는다.
- 출처 URL, 다운로드 시점, 표시 라이선스, 파일명·크기, 원본 파일 SHA-256, 실제 Python·pandas·NumPy 버전을 기록한다. 라이선스는 기초 설계의 과거 확인값을 무조건 승계하지 않고 실제 확보 시점에 재확인한다.
- 원본은 수정하거나 덮어쓰지 않는다. 검사를 위해 컬럼명을 바꾼 경우 원본 이름과 canonical 이름의 매핑을 남긴다.
- `pickle`은 읽을 때 코드를 실행할 수 있다. 명시적 로컬 신뢰 승인 없이 역직렬화하지 않는다. 해시는 파일 식별 수단이지 안전성 보증이 아니다. 외부 업로드 파일을 서버에서 바로 pickle로 읽지 않는다. [Python 공식 보안 안내](https://docs.python.org/3/library/pickle.html).

### 3.2 메모리와 호환성

일반적인 단일 pandas pickle은 CSV처럼 `chunksize`로 분할 읽을 수 있다고 가정하지 않는다. 먼저 사용 가능한 RAM을 확인하고, 승인된 원본을 별도 변환 프로세스에서 읽어 캐시로 내보낸다. 파일 크기만으로 RAM 사용량을 추정하지 않는다.

RAM이 부족하거나 구버전 객체가 로드되지 않으면 중단하고, 출처가 확인된 변환 배포본 또는 격리된 호환 환경을 검토한다. `pd.read_pickle`의 실패를 임의 모듈 주입·수상한 외부 스크립트 실행으로 해결하지 않는다. 초기 검사는 전체 pickle 로드 후 일부 행만 선택할 수 있으므로, 일부 행 실험도 원본 로드 비용을 없애지 못한다.

원본 로드 후에만 행 단위 변환을 chunk로 나누며, 변환된 모든 맵을 float32 배열로 동시에 만들지 않는다.

## 4. Canonical 데이터 계약

### 4.1 라벨과 클래스 매핑

고정 매핑 제안은 다음과 같다. 클래스 순서는 빈도나 원본 행 순서로 재계산하지 않는다.

| class_index | pattern_label |
| ---: | --- |
| 0 | None |
| 1 | Center |
| 2 | Donut |
| 3 | Edge-Loc |
| 4 | Edge-Ring |
| 5 | Loc |
| 6 | Near-Full |
| 7 | Random |
| 8 | Scratch |

`None`은 문자열 클래스명이다. Python `None`, JSON `null`, 빈 배열, 빈 문자열, 결측값과 구분한다. 원본의 단일 원소 중첩 배열은 구조를 검증한 뒤 스칼라로 풀고, 명시한 별칭만 정규화한다. 별칭은 `none`→`None`, `Near-full`→`Near-Full`이다. 알 수 없는 이름과 복수 라벨은 조용히 첫 원소만 채택하지 않고 격리한다. 미라벨에는 `class_index=null`을 유지한다.

### 4.2 Manifest 필드

| 필드 | 타입·의미 |
| --- | --- |
| sample_id | 문자열. `dataset_version:source_row_position`으로 배포본 내부 유일성 확보 |
| source_row_position | 정수. 원본 행 위치. 원본 DataFrame index와 구분 |
| source_index | 원본 index의 추적용 표현. 유일하다고 가정하지 않음 |
| lot_id, wafer_index | 원본 Lot·웨이퍼 번호. 누락은 별도 상태로 기록 |
| map_height, map_width | 양의 정수, 원본 배열 크기 |
| map_hash | shape와 canonical uint8 값의 SHA-256. 파일 해시와 별개 |
| valid_die_count, failed_die_count | 원본의 유효·불량 다이 수 |
| failed_die_ratio | failed_die_count / valid_die_count |
| pattern_label, class_index | 고정 매핑의 클래스. 미라벨은 null |
| label_status | labeled / unlabeled / invalid |
| quality_status, reason_codes | accepted / quarantined 및 복수 사유 목록 |
| duplicate_group_id, split_group_id | 동일 맵 그룹과 Lot 보호 분할 그룹 |
| source_partition | 원본 Training/Test 표식, 사용 이력 보존용 |
| split | train / validation / test / unlabeled / quarantined |
| map_ref | 저장된 원본 canonical 맵의 shard·offset |
| dataset_version, split_protocol_id | 원본 정규화 버전·확정 분할 계약 |

`lot_id`, `wafer_index`, ID, split, 해시와 원본 partition은 모델 feature가 아니다. 모든 행은 최종 manifest에 남겨 accepted·미라벨·격리의 합계가 원본 행 수와 맞아야 한다. 삭제로 제외 이력을 숨기지 않는다.

## 5. 품질 검사와 실패 정책

| 검사 | 정상 조건 | 실패 처리 |
| --- | --- | --- |
| 원본 필수 컬럼 | 맵·라벨·Lot 필드 존재, 승인된 별칭만 사용 | 파일 단계 중단 |
| 맵 차원·값 | 비어 있지 않은 2D, 유한한 정수값 0/1/2 | 행 격리. 임의 clipping·NaN 채우기 금지 |
| 유효 다이 | `map > 0`인 위치가 하나 이상 | 행 격리 |
| 해상도 | 양의 shape, 처리 가능한 byte 크기 | 비정상 과대 배열 격리, 한도는 원본 검사 후 명시 |
| Lot | 비어 있지 않은 식별자 | Lot 분할용 지도학습에서 격리. 하나의 `unknown` Lot으로 합치지 않음 |
| ID | sample_id 유일, Lot+waferIndex 충돌 확인 | 서로 다른 맵·라벨의 동일 ID는 충돌 그룹 격리 |
| 라벨 | 알려진 단일 클래스 또는 명시적 미라벨 | 미지·복수 라벨 격리 |
| 동일 맵의 라벨 | 명시적 라벨 간 일관성 확인 | 서로 다른 정답이면 그룹 격리·검토 |
| 원본 partition | 값·빈도·표기 기록 | 빈 값이 있어도 새 Lot split의 자동 정답으로 보완하지 않음 |

`None` 클래스에도 불량 다이가 존재할 수 있다. 이를 자동으로 오라벨로 판정하지 않는다. 반대로 패턴 라벨이 있는데 불량 다이가 0개면 검토 사유로 기록하되, 임의 라벨 수정은 하지 않는다.

품질 검사는 무효 행을 기록하며, 희소 클래스라는 이유로 예외 허용하지 않는다. 기준 변경은 새 설정·버전으로 남긴다.

## 6. 중복·Lot 분할과 누수 방지

### 6.1 중복을 먼저 구분한다

동일 맵 내용은 같은 입력이라는 뜻이지 반드시 같은 실물 웨이퍼를 복제했다는 뜻은 아니다. 특히 불량 다이가 없는 맵은 서로 다른 Lot에서도 동일할 수 있다. 다음을 따로 집계한다.

1. 동일 원본 ID·동일 내용: 중복 기록 후보. 원본은 유지하고 대표행·별칭 관계를 기록한다.
2. 서로 다른 ID·동일 맵·동일 라벨: 반복 입력 후보. 임의 삭제하지 않는다.
3. 동일 맵·서로 다른 명시적 라벨: 라벨 충돌 후보. 자동 다수결 금지.
4. 미라벨과 라벨 맵의 일치: 라벨을 전파하지 않는다. 추후 미라벨 학습에서도 평가 중복 보호에 사용한다.

원본 맵 해시는 크기와 정규화된 값으로 계산한다. 회전·반전·resize된 맵을 같은 원본으로 자동 병합하지 않는다. 변환 후 동일 입력이 되는 충돌도 별도 집계한다.

### 6.2 초기 분할안

강한 중복 보호 기준안은 **같은 Lot 또는 같은 원본 맵 해시로 연결된 행을 하나의 연결 그룹**으로 묶는 것이다. 연결 그룹 전체를 Train/Validation/Test 중 하나에 배정한다. 라벨·미라벨을 함께 검사해 같은 Lot의 미라벨 데이터가 향후 평가 Lot 학습에 들어가지 않도록 한다.

이 방식은 동일 정상 맵 때문에 여러 Lot이 큰 그룹으로 연결되는 문제가 있다. 따라서 구현 전에 최대 그룹 비율·연결 Lot 수·클래스별 그룹 수를 검사한다. 최대 그룹이 라벨 유효 데이터의 20%를 넘으면 자동 진행하지 않고 분할 가능성 검토 대상으로 표시한다. 20%는 운영·성능 기준이 아니라 **설계 검토를 위한 임시 경고값**이다.

분할이 불가능하면 그룹을 강제로 쪼개거나 seed를 계속 바꾸지 않는다. 다음 중 하나를 명시적으로 결정하고 별도 프로토콜로 기록한다.

- 동일 내용 대표행만 사용하는 중복 제거 벤치마크: 클래스·Lot 분포가 달라지므로 제외량을 공개.
- Lot 비중첩을 유지하고 자연 발생 동일 입력을 허용하는 벤치마크: 맵 해시 비중첩이라고 주장하지 않고 중복 건수를 보고.
- 문제 그룹을 격리하는 벤치마크: 클래스별 제외 편향과 평가 범위 축소를 보고.

기본 기초 설계의 Lot·중복 비중첩 목표를 조용히 완화하지 않는다. 실제 구조를 확인하기 전 어느 대안이 최선인지 확정하지 않는다.

### 6.3 그룹 배정 절차

- 라벨 데이터의 목표 비율은 70/15/15, seed는 42를 초기값으로 제안한다.
- 클래스별 샘플 수와 그룹 크기를 함께 고려하는 결정적 그룹 배정을 사용한다. 단순 행 단위 `train_test_split(stratify=y)`는 사용하지 않는다.
- 초기 구현안은 고정된 최대 32개 그룹 배정 후보를 만들고, 학습 전에 정한 샘플 비율·클래스 비율 편차 점수로 선택한다. 모델 성능·예측·Test 이미지는 후보 선택에 사용하지 않는다. 후보 수·목적식·tie-break도 설정에 기록한다.
- 목적식 제안은 `J = Σ_s(n_s/N-r_s)^2 + (1/9)Σ_cΣ_s(n_cs/N_c-r_s)^2`다. `r_s`는 목표 split 비율, `N_c`는 전체 유효 라벨 중 클래스 c 건수, `n_cs`는 해당 split의 클래스 건수다. 클래스별 정규화로 다수 클래스만 비율 점수를 지배하지 않게 한다. 필수 클래스 지원·비중첩 조건을 먼저 통과한 후보 중 J가 가장 작은 것을 선택하며, 동점은 고정 후보 번호 순서로 결정한다. 전역 최적 분할을 보장하는 알고리즘은 아니다.
- 그룹 층화는 정확한 클래스 비율을 보장하지 않는다. `StratifiedGroupKFold`도 그룹 구성이 제한적이면 층화가 어려울 수 있다. 라이브러리 이름만으로 70/15/15가 달성됐다고 표시하지 않는다. [scikit-learn 공식 설명](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).
- 각 클래스가 3개 이상의 독립 그룹에 있는지 먼저 검사한다. 부족하면 모든 split에 그 클래스를 넣을 수 없으므로 중단·목표 재검토한다.
- 초기 인수 조건은 9개 클래스 모두 각 split에 1건 이상이다. Validation/Test의 특정 클래스가 10건 미만이면 소표본 경고를 붙인다. 10건 이상도 충분한 통계적 안정성을 보장하지 않는다.
- 목표 샘플 비율에서 5%p 이상 벗어나면 경고한다. Lot·중복 보호를 깨서 비율을 맞추지 않는다.
- split 확정 후 manifest를 보존하고 모델 결과를 이유로 재분할하지 않는다. 수정이 불가피하면 새 프로토콜과 사용 이력을 남긴다.

미라벨은 지도학습에서 제외하지만 연결된 split_group_id와 평가 Lot 여부는 보존한다. 미라벨에 `split=unlabeled`라고 썼다는 이유로 Test Lot 학습이 허용되는 것은 아니다.

## 7. 맵 변환의 정확한 규칙

### 7.1 마스크의 의미

원본 `M`은 uint8의 2D 배열이다. 기대 값의 의미는 `0=웨이퍼 밖`, `1=정상 다이`, `2=불량 다이`이며 실제 원본 검사에서 대조한다. 원본 필드의 활용은 [AWS WM-811K 예제](https://github.com/aws-samples/amazon-ec2-nice-dcv-semiconductor-wafer-data/blob/master/README.adoc)를 참고하되 배포본 검사로 확정한다.

```text
채널 0: defect_mask = (M == 2)
채널 1: valid_mask  = (M > 0)

웨이퍼 밖 → [0, 0]
정상 다이 → [0, 1]
불량 다이 → [1, 1]
```

모든 위치에서 `defect_mask <= valid_mask`여야 한다. 색상으로 렌더링한 보고서 PNG를 CNN 원본으로 다시 읽지 않는다. 컬러맵·축·제목·라벨 텍스트가 모델 입력에 섞이지 않도록 한다.

### 7.2 Resize와 padding

구현 보완: 현재 [transform.py](../../src/wafer_dl/transform.py)는 PyTorch 없이 실행 가능한 `numpy_half_pixel_nearest_v1`을 명시적으로 사용합니다. 아래 `nearest-exact`는 원래의 PyTorch 제안이며 현재 실행 backend가 아닙니다. 좌표·반올림·padding 규칙과 검증 범위는 [입력 변환 검사](TRANSFORM_CHECK.md)에 기록합니다. 미설치 PyTorch와의 런타임 정합성을 검증했다고 간주하지 않습니다. 입력 캐시·Tensor Dataset은 후속 단계입니다.

1. 최초 기준에서는 원본 배열을 crop하지 않는다. 웨이퍼 중심·경계·notch 정보를 임의 정렬하지 않는다.
2. 목표 크기 `(T_h,T_w)`에 대해 `s=min(T_h/H,T_w/W)`를 구한다.
3. 새 높이·너비는 `floor(원래 길이*s+0.5)`를 적용하고 1 이상·목표 길이 이하로 제한한다. 라이브러리의 암묵적 반올림에 맡기지 않는다.
4. categorical map 자체를 같은 좌표로 최근접 resize한 뒤 두 마스크를 생성한다. PyTorch 구현은 `interpolate(mode='nearest-exact', size=...)`를 제안한다. 지원 버전과 출력을 테스트하고, 지원하지 않는 환경에서 `nearest`로 조용히 대체하지 않는다. 두 방식의 차이는 [PyTorch 공식 문서](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.interpolate.html)에 설명돼 있다.
5. 중앙 padding 값은 categorical map의 0이다. 홀수 여백의 남는 1픽셀은 아래·오른쪽에 둔다. 두 채널의 위치가 같아야 한다.
6. resize 크기·배율·여백을 기록한다. 변환 후 값은 0/1/2, 마스크는 0/1이어야 한다.
7. 모델 직전에 `[2,T_h,T_w]`의 contiguous float32 tensor로 바꾼다. batch는 `[B,2,T_h,T_w]`, 라벨은 int64 `[B]`다.

마스크는 이미 0/1이므로 `/255`, StandardScaler, ImageNet RGB 평균·표준편차를 적용하지 않는다. 원본 값 0/1/2를 연속 센서 수치처럼 중앙값 대치하지 않는다.

### 7.3 얇은 패턴 보존 검사

최근접 보간은 허용 값은 유지하지만 작은 결함이나 선을 없앨 수 있다. 다음 항목을 Train/Validation에서 확인한다.

- 원본 불량 다이가 있었는데 변환 후 하나도 남지 않은 맵 수·비율.
- 원본/변환의 불량 비율과 그 차이. 픽셀 수가 달라지므로 원시 개수 동일성을 요구하지 않음.
- Scratch·Donut·Edge-Loc·Near-Full의 원본/변환 대표 그림, 소형·대형·비정방형 맵.
- 불량 마스크가 유효 다이 마스크 밖에 생기지 않는지, 웨이퍼 유효 영역 소실 여부.

불량 마스크 완전 소실은 baseline 전달 전에 중단·검토하는 기준으로 제안한다. 특정 샘플만 높은 해상도로 바꾸거나 Test 이미지를 보고 해상도를 고르지 않는다. 문제가 있으면 모든 샘플에 같은 192/256 입력을 쓰는 새 변환 버전 또는 영역 집계 기반 변환을 별도 비교한다. max pooling식 불량 보존도 패턴 두께를 바꿀 수 있어 무조건 정답 대안으로 간주하지 않는다.

Test 변환의 구조 검사에서는 shape·범주·불량 마스크 소실 같은 기계적 검사만 수행한다. 실패하면 기존 버전의 무효 사유와 Test 검사 이력을 남긴 뒤 수정한다. 이미 Test를 확인한 이력을 숨기고 독립 미사용 상태로 되돌리지 않는다.

## 8. Train 전용 증강과 불균형 처리

### 8.1 증강

첫 baseline은 증강하지 않는다. 다음 비교에서 정방형 입력의 D4 변환 8가지(90도 회전과 반전 조합)를 균등하게 선택하는 옵션을 제안한다. 변환 하나를 두 채널에 함께 적용하고 라벨·sample_id는 유지한다. 회전은 배열 위치 변경으로 구현해 추가 보간을 피한다.

Scratch 등 클래스가 방향에 무관하다는 가정과 원본 notch·장비 방향의 의미를 담당자가 확인해야 한다. 8가지가 8개 독립 웨이퍼를 만든다는 뜻은 아니다. 원본 샘플 수·지원 건수는 증강 전 기준으로 보고한다.

임의 각도 회전, blur, crop, cutout, CutMix, MixUp, 다이 상태 반전과 가짜 결함 생성은 초기 기본값에서 제외한다. Validation/Test/추론은 결정적 변환만 적용하고 초기 TTA도 사용하지 않는다.

### 8.2 클래스 불균형

- 분할 후 accepted Train 라벨 빈도만 사용한다. 전체 데이터나 Validation/Test 빈도로 loss 가중치를 계산하지 않는다.
- 기본 전달물은 클래스별 Train count다. 비교용 class weight는 `N_train/(9*n_c)`를 초기식으로 제안하되 최종 적용은 학습 담당자가 loss 설정에 기록한다. Train에 클래스가 없으면 계산을 중단한다.
- baseline은 일반 Cross Entropy, 다음 실험은 weighted Cross Entropy다. 처음부터 가중 loss와 과도한 oversampling을 동시에 쓰지 않는다.
- `None`을 대량 삭제하거나 희소 클래스를 복제한 뒤 전체 데이터를 split하지 않는다. Validation/Test는 확정된 원래 분포를 유지한다.
- SMOTE는 공간 맵의 기본 전처리로 사용하지 않는다. 센서용 전처리·정책 문턱을 재사용하지 않는다.

## 9. 캐시·Dataset·재현성

### 9.1 저장 형식과 용량

초기 캐시는 두 층으로 나눈다.

- canonical 원본: 가변 크기 uint8 맵을 shard별 1D `.npy` 배열에 이어 저장하고 manifest의 offset·height·width로 복원. NumPy object 배열·추가 pickle은 만들지 않음.
- 입력 캐시: 결정적 resize+padding을 끝낸 categorical uint8 `[N,128,128]` shard. `mmap_mode='r'`, `allow_pickle=False`로 읽고 Dataset에서 2채널 생성.

입력 캐시에는 증강 결과를 저장하지 않는다. 원본 캐시를 유지해 해상도를 바꿀 때 최초 pickle을 다시 읽지 않아도 되게 한다. Train/Validation/Test shard는 별도로 만들고 Dataset은 지정 split만 연다. 초기 shard 크기는 최대 4,096건을 제안하고 실제 읽기 성능·RAM에 맞춰 조절한다.

128×128 기준 172,950개 입력의 uint8 저장량은 약 2.83 GB, 2채널 float32를 모두 펼치면 약 22.67 GB다(십진 단위, metadata·원본 캐시 제외). 따라서 전체 float 적재를 금지하고 미라벨 약 64만 건의 확대 입력 캐시는 초기 단계에서 만들지 않는다. 실제 accepted 수와 해상도에 따른 용량을 실행 전에 다시 계산한다.

### 9.2 Dataset 반환 계약

```python
{
    "image": Tensor,      # float32 [2, H, W], 값 0/1
    "target": Tensor,     # int64 스칼라, 0~8
    "sample_id": str,     # 추적용, 모델 forward에는 전달하지 않음
}
```

미라벨은 별도 Dataset이 `target=None`을 반환하며 초기 supervised DataLoader에 섞지 않는다. `lot_id` 등 분석 metadata는 sample_id로 manifest에서 조회한다. Test Dataset은 최종 평가 확인 옵션이 있는 경로에서만 열도록 한다. 이것은 실수 방지 장치이지 악의적 접근을 막는 보안 경계는 아니다.

### 9.3 병렬 로딩과 seed

- Windows에서는 `num_workers=0`으로 먼저 계약 검사를 통과시키고 2/4 worker를 별도 측정한다. 실행 진입점에 `if __name__ == '__main__'` 가드를 둔다.
- worker에서 최초 pickle 전체를 다시 읽지 않는다. shard mmap 핸들은 프로세스별로 지연 생성하며 Dataset 직렬화 때 살아 있는 파일 핸들을 공유하지 않는다.
- Train만 shuffle한다. seed·epoch·sample_id에 따른 증강 난수 규칙을 명시하고 실행마다 변하는 Python 기본 `hash()`에 의존하지 않는다. worker 개수 변경에 따른 재현 범위도 검사한다.
- Validation/Test는 순서·변환이 결정적이어야 한다. sample_id를 항상 결과에 보존한다.
- 전처리 재현성과 GPU 학습의 완전한 bit-level 재현은 구분한다. 원본·split·입력 변환·학습 seed를 함께 기록한다.

## 10. 설정 파일과 실행 계약

설정은 SECOM의 `config.json`·`configs/datasets/secom.json`과 분리한다. 처음부터 공장별 공통 추상화를 과도하게 만들지 않고, 아래 범용 맵 계약과 WM 원본 adapter를 분리한다.

```text
configs/wm811k/
  dataset.json       # 출처·원본 필드·상태 값·라벨 별칭·매핑
  preprocessing.json # 분할·품질·해상도·캐시·증강 기본안
  training.json      # 모델·optimizer·loss·학습 예산 [이종수]
```

전처리 설정 예시(문서용이며 실제 파일·스키마는 아직 만들지 않음):

```json
{
  "preprocess_version": "wm_preprocess_v1",
  "split": {
    "ratios": [0.7, 0.15, 0.15],
    "seed": 42,
    "group_policy": "lot_and_exact_map_components",
    "max_candidates": 32,
    "large_component_warning_ratio": 0.2,
    "rare_class_warning_count": 10
  },
  "transform": {
    "target_size": [128, 128],
    "preserve_aspect_ratio": true,
    "interpolation": "nearest-exact",
    "padding_value": 0,
    "channel_order": ["defect", "valid"],
    "augmentation": "none"
  },
  "cache": {"format": "npy_uint8", "shard_max_rows": 4096},
  "loader": {"num_workers": 0}
}
```

비율 합계·클래스 매핑·허용 값·seed·해상도·보간 이름은 실행 전에 검증한다. 알 수 없는 설정 키는 경고로 무시하지 않고 거부한다. 설정의 경로는 설정 파일 또는 프로젝트 루트 기준을 명시하고, 개인 PC의 절대 경로를 공유용 설정에 쓰지 않는다.

### 10.1 버전과 재실행 범위

| 변경 | 다시 실행할 범위 |
| --- | --- |
| 원본·adapter·라벨 정규화·품질 기준 | ingest 이후 전부. 기존 split과 동일성 재확인 |
| Lot·중복·split 설정 | split 이후. 과거 평가 사용 이력은 유지 |
| 입력 해상도·보간·채널 | 결정적 캐시 이후, 모델 재학습 |
| Train 증강 | Train 변환·모델 재학습. 원래 split 유지 |
| loss 가중치·모델 구조 | 학습 이후. 결정적 캐시는 재사용 가능 |
| 그림·설명문 | 보고서만. 데이터 재생성·재학습 불필요 |

재실행 식별은 원본 파일·설정·manifest의 자동 계산값으로 관리하고, 소스 각 파일에 수작업으로 해시를 쓰지 않는다. 변환 코드의 의미가 바뀌면 담당자가 preprocess_version을 갱신한다. 다른 버전을 같은 완료 산출물에 덮어쓰지 않는다.

## 11. 모듈과 에이전트 연결

| 예정 모듈 | 책임 | 담당 |
| --- | --- | --- |
| src/wafer_dl/config.py | 타입·필수 키·경로·버전 검증 | 양현승 |
| src/wafer_dl/ingestion.py | 신뢰 확인이 필요한 WM pickle adapter·원본 캐시 | 양현승 |
| src/wafer_dl/audit.py | 맵·라벨·ID·Lot·중복·제외 보고서 | 양현승 |
| src/wafer_dl/split.py | 그룹 구성·분할·manifest·보호 검사 | 양현승 |
| src/wafer_dl/transforms.py | 결정적 resize·마스크·Train 전용 증강 | 양현승 |
| src/wafer_dl/cache.py | shard 생성·읽기·원본 대응 | 양현승 |
| src/wafer_dl/dataset.py | split 제한·배치 계약·worker 재현성 | 양현승 |
| src/wafer_dl/prepare.py | 단계 CLI, 기존 코어 호출 | 양현승 |
| src/agents/wm_preprocess_agent.py | 승인된 단계 실행·상태 관리 | 양현승, 코어 완료 후 |
| src/wafer_dl/models.py·train.py 등 | 모델·학습·평가·추론 | 이종수 |

WM 고유의 원본 필드명·라벨 중첩 구조 처리는 adapter에 두고, 마스크·resize·캐시는 canonical 맵 계약에 의존한다. SECOM 코어의 이진 분류 설정을 억지로 이미지용으로 바꾸지 않는다.

향후 LangGraph로 `ingest → audit → split → cache → validate_contract`를 연결해도, 각 단계는 단독 CLI로 같은 결과를 내게 한다. 사용자 합의에 따라 원본·상세 집계·설정·실제 오류·식별자의 LLM 입력 금지는 유지하되, 승인된 최소 정형 검사 상태와 허용 동작 ID만 제공하는 제한적 자율 선택을 적용한다. LLM은 추가 진단·기존 계약 검증·사용자 확인 요청 중 다음 작업을 선택하고, 실행기는 필수 차단·권한·승인을 강제한다. 분할·라벨·품질 기준의 자동 변경은 금지한다. 최소 상태·호출 시점에도 추측 위험이 남으므로 무정보 노출로 설명하지 않는다. 실제 결과 설명은 로컬 템플릿으로 생성하고 LangSmith 등 실제 외부 추적은 계속 금지한다. 권한 분리·전송 게이트·차단 검증은 [에이전트 보안 설계](../agents/DESIGN.md#목표와-범위)를 따른다.

상태는 `blocked_input`(구조·품질·분할 불가), `failed_runtime`(I/O·환경), `completed`로 구분한다. 품질 실패를 재시도해서 통과시키거나, 임의 라벨 변경·seed 재탐색·Test 제한 해제를 하지 않는다. 중간 폴더는 완료 표식과 검증이 갖춰질 때까지 학습 입력으로 공개하지 않는다.

## 12. 산출물과 인수 검사

```text
data/processed/wm811k/<dataset_version>/
  canonical/              # 가변 길이 uint8 shard·offset
  manifest.parquet        # 전체 행·라벨 상태·품질·추적
  <split_protocol_id>/
    split_manifest.parquet
    class_mapping.json
    preprocessing.json
    train_class_counts.json
    input_cache/<preprocess_version>/<split>/*.npy
logs/wm811k/<run_id>/
  execution.json          # 원본·설정·환경·단계 상태
  quality_summary.json
  excluded_samples.parquet
  split_summary.csv
  transform_audit.csv
  contract_check.json
reports/wm811k/
  figures/<run_id>/       # 원본/변환·클래스·Lot·해상도
  preprocessing_report.md
```

원본 맵·대용량 manifest·캐시는 공개 저장소에 넣지 않는다. 집계 보고서·대표 그림·실제 설정도 자동 공개하지 않고 우선 로컬에 보존한다. 공개할 문서·재현 예시·그림은 추측 가능한 정보와 Lot 식별자를 별도로 검토한 뒤 결정한다. 저장소 공개 승인이 LLM 입력 허용을 뜻하지 않는다. 그림은 logs가 아니라 reports 밑에 둔다. 위 파일 형식·parquet 의존성은 환경 확인 후 구현한다.

### 필수 검사

- [ ] 전체 원본 행이 accepted·unlabeled·quarantined와 중복 대표/별칭 기록으로 설명된다.
- [ ] 미라벨·문자열 None·미지 라벨·중첩 배열·잘못된 값을 구분한다.
- [ ] split 사이 sample_id·Lot·승인된 중복 보호 그룹의 교집합이 0이다.
- [ ] 클래스별 샘플 수·Lot 수·독립 그룹 수·원본 partition 교차를 기록한다.
- [ ] 2채널 순서·shape·dtype·0/1 값·defect<=valid를 검증한다.
- [ ] 1픽셀 불량·얇은 선·가장자리·비정방형 합성 변환 검사와 실제 패턴 보존 검사를 수행한다.
- [ ] Validation/Test에 무작위 증강·weighted sampling을 적용하지 않는다.
- [ ] class weight는 Train에만 의존하고 Validation 라벨 변경에 영향받지 않는다.
- [ ] shard 경계·파손·알 수 없는 map_ref·manifest 불일치는 명확히 실패한다.
- [ ] 같은 입력의 배치 내 위치·별도 프로세스 복원에 관계없이 결정적 변환이 일치한다.
- [ ] 소규모 배치로 RAM·읽기 속도·CPU/GPU 전송을 측정하고 전체 실행 예상치를 남긴다.

의미 있는 계약 검사 중심으로 묶고 기능마다 불필요한 테스트를 대량 추가하지 않는다. 실제 데이터 확인과 합성 테스트를 구분한다.

## 13. 구현 순서와 먼저 확인할 사항

1. **원본과 실행 환경 확인**: 확보한 파일 위치, RAM, GPU, 디스크 여유, 학습 담당자의 PyTorch 환경.
2. **설정 로더·ingest·audit**: 먼저 원본 구조·라벨·Lot·동일 내용 그룹을 확인한다. 아직 split은 확정하지 않는다.
3. **EDA와 분할 가능성 검토**: 대표 패턴·크기·희소 클래스·대형 중복 그룹을 확인하고 분할 프로토콜을 승인한다.
4. **split·변환·캐시·Dataset**: Test 사용을 제한하고 Train/Validation에서 128 입력과 패턴 보존을 확인한다.
5. **담당 간 전달**: 클래스 매핑·첫 배치·설정·품질/분할 보고서를 전달한 뒤 작은 CNN을 시작한다.
6. **baseline 이후**: 입력 해상도·증강을 하나씩 비교한다. 처리 경로가 안정된 후 에이전트를 연결한다.

PyTorch 사용·입력 128·강한 중복 보호·증강 없는 baseline은 이번 설계안이며 원본 검사 전의 최적값 확정이 아니다. 다음 구체 작업은 **데이터 확보·환경 확인과 ingest/audit 설계의 구현**이다. 학습·Test·외부 API 연결은 이번에 실행하지 않는다.
