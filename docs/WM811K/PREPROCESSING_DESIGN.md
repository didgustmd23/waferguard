# WM-811K 전처리·학습 설계

이 문서는 웨이퍼 도메인의 설계 기준이다. 전처리·학습의 구현 범위와 후속 비교 계획을 구분하며, 에이전트 공통 설계는 [DESIGN](../agents/DESIGN.md)을 따른다. 원본 재실행·모델 학습·정책 변경을 지시하는 문서가 아니다.

## 현재 범위와 담당

숫자 전용 canonical·검증/EDA·Lot/동일 맵 보호 분할·입력 변환·캐시·Dataset, 작은 CNN 학습·평가를 구현했다. 사용자 기록 기준 Train·Validation 첫 배치, GPU 학습·checkpoint 동결, Test 캐시 준비와 1회 평가를 확인했다. 이 실제 성능은 2026-10-24~25 코드의 실행 기록이며, 이후 checkpoint v2·분할 해시·수치 검증 보완 코드로 다시 평가한 결과는 아니다. ResNet18 비교·seed 변동·학습 담당자 인수 확인은 별도 미확인 사항이다. 과거 EDA·전처리는 [결과 보고서](../../reports/wm811k/report.md), 현재 실행 계약과 학습·평가 기록은 [CACHE_DATASET](CACHE_DATASET.md)을 따른다. 이 설계의 모든 검증 항목이 모든 환경에서 재확인됐다는 뜻은 아니다.

전처리·에이전트는 양현승, WM 모델·학습·평가는 이종수 담당이다. 클래스 매핑·split·변환/평가 기준은 공동 계약이며, 변경 시 새 버전과 사용 이력을 보존한다. 기존 SECOM 실험·모델·설정은 변경하지 않는다.

WM-811K는 다이별 검사 결과 맵의 9종 패턴 분류이며 검사 이전의 불량 예측·출하 승인 데이터가 아니다. SECOM과 동일 웨이퍼 대응 관계가 없어 두 모델 성능을 합산·곱셈해 통합 검출률로 주장하지 않는다. `None` 패턴을 이유로 센서 경고를 자동 해제하지 않는다. 가상 연결은 데이터 전달/표시 데모로만 구분한다.

## 1. 전처리 계약과 학습 후보

| 항목 | 초기 설계 기준 | 이유·확정 시점 |
| --- | --- | --- |
| 과제 | 웨이퍼맵 1건의 9종 패턴 분류 | 정상 출하 승인이나 검사 이전 예측이 아님 |
| 구현 프레임워크 | NumPy 변환·PyTorch Dataset | 모델 학습 환경/버전은 별도 동결 |
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

### 출처와 원본 필드

사용자가 지정한 배포 출처는 [Kaggle QINGYI의 WM-811K](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map?resource=download)다. 기존 문서에는 2026-10-08 당시 [메타데이터](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map/metadata)의 CC0 표기를 기록했다. 이번 문서 정리에서 라이선스를 다시 확인한 것은 아니며, 실제 확보한 배포본의 조건·버전·다운로드 시점은 별도로 확인한다. 원 연구는 Wu·Jang·Chen의 [Wafer Map Failure Pattern Recognition and Similarity Ranking for Large-Scale Data Sets](https://doi.org/10.1109/TSM.2014.2364237)다.

원본의 `waferMap`은 맵, `failureType`은 라벨, `lotName`은 그룹, `waferIndex`는 Lot 내부 식별자다. `trianTestLabel`/`trainTestLabel`은 원본 분할 이력이며 새 split이나 모델 특징으로 자동 채택하지 않는다. `dieSize`의 의미·단위는 원본 정의와 대조하고 이미지 해상도·공정 시간으로 추정하지 않는다. 중첩 라벨·필드 별칭은 변환기로 명시 정규화한다. 단일 9종 분류의 결과로 혼합/미지 패턴이나 공정 원인을 확정하지 않는다.

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

고정 매핑은 다음과 같다. 클래스 순서는 빈도나 원본 행 순서로 재계산하지 않는다.

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
| map_ref | 저장된 원본 canonical 맵의 NPZ shard·배열 key |
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

기본 Lot·중복 비중첩 목표를 조용히 완화하지 않는다. 실제 구조를 확인하기 전 어느 대안이 최선인지 확정하지 않는다.

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

현재 변환은 `numpy_half_pixel_nearest_v1`이다. 정확한 좌표·반올림·padding과 판정은 [TRANSFORM_CHECK](TRANSFORM_CHECK.md)를 따른다. PyTorch의 다른 보간으로 자동 대체하지 않으며 캐시/Dataset에서도 같은 채널 의미를 사용한다.

1. 최초 기준에서는 원본 배열을 crop하지 않는다. 웨이퍼 중심·경계·notch 정보를 임의 정렬하지 않는다.
2. 목표 크기 `(T_h,T_w)`에 대해 `s=min(T_h/H,T_w/W)`를 구한다.
3. 새 높이·너비는 `floor(원래 길이*s+0.5)`를 적용하고 1 이상·목표 길이 이하로 제한한다. 라이브러리의 암묵적 반올림에 맡기지 않는다.
4. categorical map을 `floor((출력 인덱스+0.5)×원본 길이/새 길이)`의 고정 좌표로 최근접 resize한 뒤 두 마스크를 생성한다.
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

## 8. Train 전용 증강과 불균형 처리 계획

현재 결정적 변환은 augmentation=none만 허용한다. 증강은 후속 계획이며 미구현이다. Train 빈도 기반 가중 Cross Entropy는 학습기에 구현돼 기본 설정으로 적용한다. 일반 Cross Entropy와의 비교 및 증강의 효과 검증은 후속 범위다.

### 8.1 증강

첫 baseline은 증강하지 않는다. 다음 비교에서 정방형 입력의 D4 변환 8가지(90도 회전과 반전 조합)를 균등하게 선택하는 옵션을 제안한다. 변환 하나를 두 채널에 함께 적용하고 라벨·sample_id는 유지한다. 회전은 배열 위치 변경으로 구현해 추가 보간을 피한다.

Scratch 등 클래스가 방향에 무관하다는 가정과 원본 notch·장비 방향의 의미를 담당자가 확인해야 한다. 8가지가 8개 독립 웨이퍼를 만든다는 뜻은 아니다. 원본 샘플 수·지원 건수는 증강 전 기준으로 보고한다.

임의 각도 회전, blur, crop, cutout, CutMix, MixUp, 다이 상태 반전과 가짜 결함 생성은 초기 기본값에서 제외한다. Validation/Test/추론은 결정적 변환만 적용하고 초기 TTA도 사용하지 않는다.

### 8.2 클래스 불균형

- 분할 후 accepted Train 라벨 빈도만 사용한다. 전체 데이터나 Validation/Test 빈도로 loss 가중치를 계산하지 않는다.
- 기본 전달물은 클래스별 Train count다. 캐시의 후보 가중치는 `N_train/(9*n_c)`다. 현재 학습기는 역빈도 가중치에 공통 배율과 상한을 적용해 클래스별 가중치 평균 1을 맞춘다. 실제 적용 여부·상한은 학습 설정에 기록하며 Train에 클래스가 없으면 계산을 중단한다.
- 현재 기본 설정은 class-weighted Cross Entropy다. 일반 Cross Entropy와의 비교는 아직 완료하지 않았다. 가중 loss와 과도한 oversampling을 동시에 강화하지 않는다.
- `None`을 대량 삭제하거나 희소 클래스를 복제한 뒤 전체 데이터를 split하지 않는다. Validation/Test는 확정된 원래 분포를 유지한다.
- SMOTE는 공간 맵의 기본 전처리로 사용하지 않는다. 센서용 전처리·정책 문턱을 재사용하지 않는다.

## 9. 캐시·Dataset·재현성

### 9.1 저장 형식과 용량

초기 캐시는 두 층으로 나눈다.

- canonical 원본: 가변 크기 uint8 맵을 `maps/shard_*.npz`에 숫자 배열로 저장하고 manifest의 shard·key로 복원. NumPy object 배열·추가 pickle은 만들지 않음.
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

현재 `WaferDataset`은 지도학습용 확정 구간을 읽는다. 미라벨 Dataset의 `target=None` 반환은 후속 확장 계약이며 구현 완료가 아니다. 미라벨을 supervised DataLoader에 섞지 않는다. `lot_id` 등 분석 metadata는 sample_id로 manifest에서 조회한다. Test Dataset은 최종 평가 확인 옵션이 있는 경로에서만 열도록 한다. 이것은 실수 방지 장치이지 악의적 접근을 막는 보안 경계는 아니다.

### 9.3 병렬 로딩과 seed

- Windows에서는 `num_workers=0`으로 먼저 계약 검사를 통과시키고 2/4 worker를 별도 측정한다. 실행 진입점에 `if __name__ == '__main__'` 가드를 둔다.
- worker에서 최초 pickle 전체를 다시 읽지 않는다. shard mmap 핸들은 프로세스별로 지연 생성하며 Dataset 직렬화 때 살아 있는 파일 핸들을 공유하지 않는다.
- Train만 shuffle한다. seed·epoch·sample_id에 따른 증강 난수 규칙을 명시하고 실행마다 변하는 Python 기본 `hash()`에 의존하지 않는다. worker 개수 변경에 따른 재현 범위도 검사한다.
- Validation/Test는 순서·변환이 결정적이어야 한다. sample_id를 항상 결과에 보존한다.
- 전처리 재현성과 GPU 학습의 완전한 bit-level 재현은 구분한다. 원본·split·입력 변환·학습 seed를 함께 기록한다.

## 10. 구현·설정과 실행 안내

| 책임 | 현재 구현/설정 | 실행·확인 |
| --- | --- | --- |
| 원본 구조·canonical | `scripts/wm_sandbox/inspect_pickle.py`, `export_canonical.py` | [SANDBOX_SETUP](SANDBOX_SETUP.md) |
| 검증·EDA·충돌 영향 | `src/wafer_dl/validate_canonical.py`, `eda.py`, `diagnose_split.py` | [결과 보고서](../../reports/wm811k/report.md) |
| 충돌 제외·그룹 분할 | `exclude_conflicts.py`, `split.py`, `configs/wm811k/split.json` | [GROUP_SPLIT](GROUP_SPLIT.md) |
| 변환·패턴 보존 | `transform.py`, `check_transform.py`, `configs/wm811k/transform.json` | [TRANSFORM_CHECK](TRANSFORM_CHECK.md) |
| 캐시·Dataset | `cache.py`, `dataset.py`, `configs/wm811k/cache.json` | [CACHE_DATASET](CACHE_DATASET.md) |
| 기존 승인형 WM 실행 | `src/agents/wm_*`와 고급 등록 모듈 | [에이전트 GUIDE](../agents/GUIDE.md) |

정확한 허용 설정·상한은 실제 설정/검증 코드가 기준이다. 존재하지 않는 dataset.json·preprocessing.json·단일 wm_preprocess_agent.py를 실행 구성으로 가정하지 않는다. 모델·학습·평가 코드의 계획과 현재 구현을 구분한다.

### 변경과 재실행 범위

- 원본/라벨/품질 변경은 canonical 이후, 분할 변경은 split 이후의 새 산출물과 검증이 필요하다.
- 해상도·보간·채널 변경은 입력 캐시 이후와 모델 학습, 증강·loss·모델 변경은 학습 단계에 영향을 준다. 기존 완료 산출물에 덮어쓰지 않는다.
- 그림·설명문 정리는 데이터 생성·학습 재실행 사유가 아니다.
- 파일·설정·manifest의 자동 식별값과 버전을 사용한다. 과거 Test 사용 이력은 재분할이나 경로 변경으로 초기화되지 않는다.

## 11. 에이전트와의 경계

기존 WM 개별 서비스와 통합 작업의 웨이퍼 프로파일 연결은 별개다. 공통 조정기·동의·정보 전달은 [에이전트 단일 설계](../agents/DESIGN.md)가 기준이며, 옛 최소 상태 선택기 제안을 현재 활성 정책으로 사용하지 않는다. 웨이퍼 고유 필드·라벨·Lot·마스크 의미는 로더/도메인 프로파일 내부에 둔다. 에이전트가 분할·라벨·제외·변환 규칙을 임의로 변경하지 않는다.

## 12. 산출물과 인수 검사

전달은 각 단계의 검증된 묶음으로 한다. 파일을 수동으로 합쳐 새로 만들어 완료 표식을 붙이지 않는다.

| 묶음 | 핵심 파일 |
| --- | --- |
| canonical | manifest JSONL, maps/shard_*.npz, conversion.json, provenance.json, SUCCESS.json. 정확한 경로/필드는 exporter·검증기 기준 |
| 분할 | split_manifest.jsonl, split.json, split_summary.csv, SUCCESS.json |
| 입력 캐시 | cache.json, cache.md, SUCCESS.json, 구간별 index.jsonl·shard_*.npy |
| 결과 설명 | reports/wm811k/report.md, analysis.ipynb와 로컬 검사 보고서/그림 |

원본·대용량 manifest·캐시를 자동 공개하지 않는다. 집계·대표 그림·설정도 공개 전에 식별 정보와 사용 범위를 검토한다. 저장소 공개가 LLM 전송 허용을 뜻하지 않는다. 실제 파일 이름·매핑·해시는 각 완료 묶음과 검증기를 기준으로 확인하며 옛 parquet/offset 저장 제안을 현재 구현으로 취급하지 않는다.

### 필수 검사

다음은 인수 때 확인할 계약이다. 모든 항목의 실제 환경 검증이나 학습 담당자 인수가 끝났다는 체크리스트가 아니다.

- 전체 원본 행이 accepted·unlabeled·quarantined와 중복 대표/별칭 기록으로 설명된다.
- 미라벨·문자열 None·미지 라벨·중첩 배열·잘못된 값을 구분한다.
- split 사이 sample_id·Lot·승인된 중복 보호 그룹의 교집합이 0이다.
- 클래스별 샘플 수·Lot 수·독립 그룹 수·원본 partition 교차를 기록한다.
- 2채널 순서·shape·dtype·0/1 값·defect<=valid를 검증한다.
- 1픽셀 불량·얇은 선·가장자리·비정방형 합성 변환 검사와 실제 패턴 보존 검사를 수행한다.
- Validation/Test에 무작위 증강·weighted sampling을 적용하지 않는다.
- class weight는 Train에만 의존하고 Validation 라벨 변경에 영향받지 않는다.
- shard 경계·파손·알 수 없는 map_ref·manifest 불일치는 명확히 실패한다.
- 같은 입력의 배치 내 위치·별도 프로세스 복원에 관계없이 결정적 변환이 일치한다.
- 소규모 배치로 RAM·읽기 속도·CPU/GPU 전송을 측정하고 전체 실행 예상치를 남긴다.

의미 있는 계약 검사 중심으로 묶고 기능마다 불필요한 테스트를 대량 추가하지 않는다. 실제 데이터 확인과 합성 테스트를 구분한다.


## 학습·평가 계획

작은 CNN과 Train 최빈 클래스 기준선은 구현됐고 실제 학습·Test 평가 기록은 [CACHE_DATASET](CACHE_DATASET.md#실제-실행-기록--2026-10-24)에 있다. 아래는 그 구현 범위와 후속 비교 조건을 구분한 설계이며 새로운 실험 지시가 아니다. ResNet18·여러 seed·검토 기준 비교의 완료를 뜻하지 않는다.

### 모델별 구현 범위와 선정 이유

| 모델 | 우선순위·목적 | 선정 이유와 초기 방안 |
| --- | --- | --- |
| 다수 클래스 예측 기준선 | 구현·평가 기록 있음 | 불균형 데이터에서 Accuracy의 효과를 확인. Train의 최빈 클래스 고정 예측 |
| 작은 CNN | 구현·학습·평가 기록 있음 | 고리·선·국소 불량 같은 공간 패턴을 단순 구조로 학습. Conv 블록 → Global Average Pooling → 9종 출력 |
| ResNet18 | 후속 비교·미구현 | 작은 CNN보다 깊은 특징 추출의 효과를 비교. 첫 convolution을 2채널에 맞추고 출력층을 9종으로 구성 |
| MobileNetV3-Small | 후속 선택 | 모델 크기·처리 시간이 병목일 때 정확도와 비용의 균형을 비교 |

현재 작은 CNN은 무작위 초기화로 학습한다. 후속 ResNet18 비교도 무작위 초기화를 기본으로 계획한다. 사전학습 가중치를 추가하면 2채널 입력 변환 방법과 학습 조건을 별도 실험으로 기록한다. Transformer·준지도학습·생성 모델은 baseline 이후 확장 후보로 남긴다.

CNN 입력은 제안 기준 `2 × 128 × 128`, 출력은 9종 logits다. 최종 모델은 Validation Macro-F1만 단독으로 보지 않고 희소 클래스 Recall, seed별 변동과 추론 비용을 함께 검토해 선정한다. Test 성능을 보고 모델을 선택하지 않는다.

### 학습과 평가

- ResNet18을 추가할 때 동일한 데이터·split·입력 변환·학습 예산으로 CNN과 비교한다. 현재 Test 결과를 보고 후보를 조정하지 않는다.
- 기본 Cross Entropy와 Train 빈도 기반 class-weighted Cross Entropy의 비교는 후속이다. 현재 학습 설정은 가중 loss를 사용하며, 가중치·샘플링을 동시에 강화하는 실험은 초기 범위에서 제외한다.
- 현재 작은 CNN은 AdamW를 사용하고 learning rate·batch size·epoch 상한·early stopping을 설정으로 관리한다. 다른 모델의 설정·실제 GPU 메모리·처리 시간은 별도로 확인한다.
- checkpoint 선택과 early stopping은 Validation Macro-F1을 사용한다. 클래스별 Recall도 함께 확인한다.
- 지표는 Macro-F1, 클래스별 Precision·Recall·F1·지원 건수, 혼동행렬, Accuracy, 학습 시간, 추론 시간과 최대 메모리다.
- 작은 CNN·ResNet18 baseline을 확보한 뒤 채택 후보는 여러 seed로 변동을 확인한다. 반복 실험은 최종 Test를 사용하지 않는다.
- 클래스별 점수와 최대 점수를 출력한다. 낮은 신뢰도의 검토 기준은 Validation에서 선택하고 Test 전에 동결한다. softmax 점수를 보정된 불량 확률이라고 해석하지 않는다.
- 검토 기능은 검토 비율과 자동 분류 대상의 오류를 함께 평가한다. 검토 대상으로 빠진 샘플을 정답 처리하지 않는다.
- `1 - P(None)`은 패턴 존재 점수로만 다룬다. SECOM 불량 확률과 같게 해석하거나 SECOM 문턱을 재사용하지 않는다.


## 남은 확인과 전달

- Test 캐시 준비·작은 CNN 학습·동결·저장·1회 평가는 실행 기록이 있다. 현행 checkpoint v2의 실제 재학습·평가 재현, 별도 복원·추론 인계 확인, ResNet18·seed 변동·운영 검증은 이 기록으로 완료 처리하지 않는다.
- Train·Validation 첫 배치 완료와 학습 담당자의 인수 확인을 구분한다. 모델 담당자는 채널 순서·dtype·라벨·manifest/캐시 대응·분할 비중첩을 확인한다.
- 실제 성능·독립 평가·현장 적용은 구현/합성 검증으로 대체하지 않는다. 출처·라이선스·pickle 신뢰와 환경 요구는 새 입력을 사용할 때 확인한다.
- 실행 중 상태나 보고서만 고쳐 통과시키지 않는다. 새 작업은 현재 계약과 승인 범위를 확인하고 준비한다.
