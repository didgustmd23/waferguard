# WM-811K 입력 캐시와 PyTorch Dataset

## 현재 범위

`src/wafer_dl/cache.py`는 확정 분할과 변환 검사를 확인한 뒤 숫자 전용 uint8 NPY 캐시를 생성합니다. `src/wafer_dl/dataset.py`는 선택 구간의 캐시를 읽어 PyTorch 입력을 구성합니다. 실제 Train·Validation 캐시·배치 확인과 GPU CNN 학습을 마쳤으며, 동결한 checkpoint를 별도 Test에서 1회 평가했습니다.

## 처리 흐름과 분할 파일

```text
원본 pickle → 숫자 전용 canonical → 검증·EDA → 그룹 분할
    → Train/Validation 맵 변환·형태 확인 → uint8 캐시
    → Dataset 2채널 Tensor → Train/Validation 첫 배치 검사
    → 모델 학습·Validation 선택 → 동결 → Test 캐시·단일 평가 (완료)
```

Train 약 70%·Validation 약 15%·Test 약 15%의 분할은 이미 확정했습니다. Test가 없는 것이 아니라 Test용 입력 캐시 준비를 미룬 상태입니다. 원본 맵을 세 폴더에 복사해서 나누는 대신, manifest에 각 행의 소속을 기록합니다.

| 파일·컬럼 | 역할 |
| --- | --- |
| 원본 canonical NPZ | 실제 웨이퍼 맵 보관 |
| `split_summary.csv` | 구간별 클래스 표본 수·비율 요약 |
| `split_manifest.jsonl` | 각 행의 그룹과 최종 분할 소속 기록 |
| `split_group_id` | Lot·동일 맵 관계로 함께 묶인 그룹의 식별자 |
| `group_split` | 연결 그룹이 배정된 train·validation·test. 배정되지 않은 그룹은 null |
| `split` | 행의 최종 용도: train·validation·test·unlabeled·quarantined |

같은 그룹의 미라벨 행은 `group_split`이 있어도 `split=unlabeled`이므로 지도학습 캐시에 넣지 않습니다. 충돌로 격리한 행도 캐시에서 제외합니다. Dataset이나 DataLoader에서 데이터를 다시 무작위 분할하지 않습니다.

## 실행 확인 기록 — 2026-10-10

아래는 사용자가 공유한 실행 출력 기준이며 이 기록을 위해 실제 데이터를 재실행하지 않았습니다.

| 구간 | 상태 | 이미지 형태 | 이미지 자료형 | 라벨 자료형 | Test 맵 접근 |
| --- | --- | --- | --- | --- | --- |
| Train | `dataset_batch_passed` | `[32,2,128,128]` | `torch.float32` | `torch.int64` | 없음 |
| Validation | `dataset_batch_passed` | `[32,2,128,128]` | `torch.float32` | `torch.int64` | 없음 |

두 실행의 분할 프로토콜은 동일합니다.

```text
wm_split_d02d993fd73c2cc4bf496d54774401c60a94b0d1737c7c135e7ae4837a6b85d6
```

이 결과는 선택 구간의 캐시 정합성과 첫 배치 입력 계약 확인입니다. 전체 Epoch 실행·GPU 학습·성능 평가 완료를 뜻하지 않습니다. 이후 명령은 재현·전달용이며 이미 완료한 캐시를 다시 생성할 필요는 없습니다.

## GPU 통합 환경 설치 — 2026-10-19

Windows x64/Python 3.12의 `wm-gpu312` 하나로 UI·에이전트·전처리·GPU 학습을 실행합니다. 루트 `requirements.txt`에 ML·DL·UI·에이전트·노트북 의존성을 통합했고 기존 분리 파일 6개는 제거했습니다. 이전 문서·명령에 분리 requirements가 나오면 이 절차를 사용하세요.

```powershell
# 이미 환경이 있으면 생성은 생략합니다.
conda create -n wm-gpu312 python=3.12
conda activate wm-gpu312
python -m pip install -r requirements.txt
python -m pip check
python -m streamlit run src/ui/app.py --server.address 127.0.0.1
```

PyTorch는 공식 [torch CUDA 목록](https://download.pytorch.org/whl/cu130/torch/)과 [torchvision CUDA 목록](https://download.pytorch.org/whl/cu130/torchvision/)의 Windows/Python 3.12 wheel 주소로 고정했습니다. 일반 패키지는 PyPI를 사용하고 GPU 파일에만 공식 직접 주소를 지정하므로 별도 index 설정·CPU/GPU 파일 선택·분리 설치가 필요 없습니다. 고정 빌드는 `torch 2.14.1+cu130`, `torchvision 0.29.1+cu130`입니다. 다른 OS/Python은 이 설치 파일의 지원 범위가 아닙니다.

설치 명령은 사용자가 실행합니다. 기존 CPU 빌드가 있으면 GPU 빌드로 교체되며 의존성 해결에 따라 다른 패키지도 변경될 수 있습니다. UI·worker를 종료하고 설치하며 충돌을 `--no-deps`로 숨기지 않습니다.

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); x=torch.arange(4,device='cuda'); torch.cuda.synchronize(); assert (x*2).cpu().tolist()==[0,2,4,6]; print(torch.cuda.get_device_name(0))"
```

CUDA 설치와 전처리 연산 장치는 별개입니다. WM 정상 개발 6단계는 이 GPU 설치 환경도 허용하지만 격리 전처리·Dataset 검사는 CPU 연산을 유지합니다. GPU 학습은 학습 코드에서 모델·Tensor를 CUDA로 옮겨 수행합니다. 실제 GPU 빌드의 AppContainer 호환성은 별도 합성 검사로 확인합니다. 런타임 또는 코드가 바뀌면 해시에 묶인 기존 작업·승인은 재사용하지 않습니다.

원본 pickle 검사 도구인 `scripts/wm_sandbox/requirements.txt`는 프로젝트 주 환경과 용도가 달라 유지합니다.

## 1. 개발 구간 캐시 생성

변환 검사 결과가 `transform_check_passed`이고 `examples.png`에서 패턴 보존을 직접 확인한 경우에만 `--confirm-visual-review`를 지정합니다. 이 옵션은 자동 시각 검사를 뜻하지 않습니다.

```powershell
python -m src.wafer_dl.cache `
  --split-dir data/splits/wm811k/group_v2 `
  --check-dir reports/wm811k/transform_v2 `
  --output-dir data/processed/wm811k/input_cache/wm_resize_v2 `
  --confirm-visual-review
```

실제 변환 검사 결과 폴더가 다르면 `--check-dir`을 변경하세요. canonical 경로는 분할 보고서에서 읽습니다. 데이터 묶음을 이동했다면 같은 원본 묶음의 현재 위치를 `--canonical-dir`로 지정할 수 있습니다.

완료 기준은 `cache_complete`와 출력 폴더의 `SUCCESS.json`입니다. 기존 출력 폴더는 덮어쓰지 않습니다. 중단된 폴더는 완료 결과로 사용하지 말고, 원인을 확인한 후 다른 새 출력 경로로 실행하세요.

- 기본 대상은 Train·Validation입니다. Test 배열은 읽지 않습니다.
- 미라벨·격리 행은 학습 캐시에 포함하지 않습니다.
- 원본 맵·라벨·분할을 변경하거나 다시 나누지 않습니다.
- 선택한 배열의 무결성을 확인하고 기존 결정적 resize·padding을 재사용합니다.
- 유효 영역 또는 불량 영역이 완전히 사라지면 중단합니다. 일부 형태 변화까지 없다는 보장은 아닙니다.

## 2. 첫 배치 확인

캐시 생성 성공 후 각각 실행합니다.

```powershell
python -m src.wafer_dl.dataset `
  --cache-dir data/processed/wm811k/input_cache/wm_resize_v2 `
  --split train `
  --batch-size 32

python -m src.wafer_dl.dataset `
  --cache-dir data/processed/wm811k/input_cache/wm_resize_v2 `
  --split validation `
  --batch-size 32
```

기대 상태는 `dataset_batch_passed`입니다. 기본 크기에서 이미지 형태는 `[32,2,128,128]`, 이미지 자료형은 `torch.float32`, 라벨은 `torch.int64`입니다. 마지막 배치 등 표본 수가 적으면 배치 수는 줄어들 수 있습니다.

초기화 시 선택 구간의 전체 shard 해시·배열 헤더·색인을 검사합니다. 따라서 첫 출력까지 파일 크기에 비례한 시간이 걸릴 수 있습니다. 전체 데이터를 RAM에 적재하지는 않습니다. 채널값 검사는 첫 배치 및 이후 실제 접근한 표본에 적용합니다. 다른 분할 파일은 열지 않습니다.

이 검사는 CPU·`num_workers=0`으로 실행합니다. GPU 연산·다중 worker 처리량·전체 Epoch·모델 성능 검사가 아닙니다. 캐시의 `training_ready=false`를 자동 변경하지 않습니다.

## 저장 계약과 학습 연결

캐시 한 표본은 `[128,128]` uint8 범주형 맵입니다. 상태는 `0=외부`, `1=정상 다이`, `2=불량 다이`입니다. 웨이퍼 전체의 9개 패턴 라벨과 픽셀 상태는 별개입니다.

`index.jsonl`은 표본 ID·클래스·shard 파일·파일 내부 위치를 연결합니다. `cache.json`은 설정·분할 프로토콜·개수·파일 해시를 기록하고 `cache.md`는 사람이 읽는 요약입니다. 해시는 자동 기록하며 안전성 인증이나 사용자 수동 해시 관리 절차가 아닙니다.

캐시는 실제 `split_manifest.jsonl`의 SHA-256도 기록합니다. 새 checkpoint는 이
해시를 포함하며 Test 캐시의 값과 다르면 평가를 거부합니다. 이 필드가 없는
이전 checkpoint는 새 계약으로 평가할 수 없으므로 새 checkpoint를 생성해야 합니다.

```python
from torch.utils.data import DataLoader
from src.wafer_dl.dataset import WaferDataset

dataset = WaferDataset("data/processed/wm811k/input_cache/wm_resize_v2", split="train")
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
try:
    batch = next(iter(loader))
    images = batch["image"]   # [B,2,H,W]: defect, valid 순서
    targets = batch["target"] # [B]: 정수 0~8
    # sample_id는 추적용이며 모델 입력에 포함하지 않습니다.
finally:
    dataset.close()
```

Dataset은 읽기 전용 mmap을 최대 두 파일까지 유지하고 필요한 표본만 두 채널로 변환합니다. 캐시 맵은 한 장 16KiB, 모델용 두 float32 채널은 한 장 128KiB입니다. 현재 개발 구간 147,088장 기준 맵 본체는 약 2.41GB이며 색인·헤더 공간이 추가됩니다.

`configs/wm811k/cache.json`의 `shard_max_rows` 기본값은 4,096, `buffer_max_bytes`는 64MiB입니다. 두 상한 중 작은 기준으로 나눕니다. Train·Validation writer 버퍼는 각각 할당되므로 버퍼 합계는 기본 최대 약 128MiB이고, 검사·변환의 추가 메모리는 별도입니다.

`suggested_train_class_weights`는 Train 빈도만 사용한 `N_train/(9×클래스 표본 수)` 후보입니다. 학습 코드는 Train 빈도에서 같은 형태의 가중치를 계산하고 설정의 상한으로 제한해 class-weighted loss를 적용합니다. class-weighted 설정과 상한은 비교 실험으로 검증해야 합니다. 증강·추가 정규화·통계 재학습은 하지 않습니다.

## CNN 학습과 Validation

`src/wafer_dl/models.py`의 작은 CNN은 2채널 입력과 9종 logits를 사용합니다. `src/wafer_dl/train.py`는 확인된 캐시의 Train으로 학습하고 Validation으로 매 epoch 분류 지표를 계산합니다. 기본 checkpoint 선택 기준은 Validation Macro-F1이며 클래스별 Precision·Recall·F1·지원 수와 혼동행렬을 함께 기록합니다. Recall을 checkpoint 선택의 우선 기준으로 삼을 실험은 설정의 `selection_metric`을 `macro_recall`로 바꿀 수 있습니다. 이 경우에도 Precision과 클래스별 혼동을 같이 살펴야 합니다. Train 기반 class-weighted loss는 희소 패턴을 고려하는 실험 출발점이지 Recall 개선을 보장하지 않습니다. 코드 구현과 실제 모델 성능 확인은 구분합니다.

학습 중 클래스 가중치·모델 파라미터·logits·loss의 NaN/무한대를 검사해 발견 즉시
중단합니다. epoch별 weighted loss는 배치 평균을 표본 수로 단순 평균하지 않고
정답 클래스 가중치 합을 사용해 집계합니다.

먼저 Train·Validation 입력 캐시가 존재하고 GPU/PyTorch 환경이 준비되어 있어야 합니다. 프로젝트 루트에서 실행합니다.

```powershell
python -m src.wafer_dl.train `
  --cache-dir data/processed/wm811k/input_cache/wm_resize_v2 `
  --output-dir models/wm811k/cnn_v1 `
  --config configs/wm811k/train.json `
  --device auto
```

`--device auto`는 CUDA를 사용할 수 있으면 GPU를, 아니면 CPU를 선택합니다. 실행은 Test를 열지 않습니다. 완료 시 `best_model.pt`와 `training_report.json`이 출력 폴더에 저장되고, 기존 출력 폴더가 있으면 덮어쓰지 않고 실패합니다. 보고서에는 split protocol, 선택 epoch, epoch별 Macro-F1·Macro-Recall·클래스별 Recall과 혼동행렬이 들어갑니다. 같은 경로에 재실행하려면 기존 결과를 보존한 채 새 `--output-dir`을 지정하세요.

### 실제 실행 기록 — 2026-10-24

검증한 canonical 데이터와 새 Lot 비중첩 분할로 Train 121,261건·Validation 25,827건의 캐시를 만들었습니다. RTX 4060 Ti의 CUDA PyTorch에서 최대 30 epoch, patience 6으로 실행했고 Validation Macro-F1이 6 epoch 연속 개선되지 않아 20 epoch에서 조기 종료했습니다. 최고 Macro-F1 checkpoint는 14 epoch입니다.

| Validation 지표 | 5 epoch 탐색 실행 | 최대 30 epoch 실행, 14 epoch checkpoint |
| --- | ---: | ---: |
| Accuracy | 92.26% | 96.07% |
| Macro-Recall | 79.66% | 86.38% |
| Macro-F1 | 69.95% | 77.69% |

같은 split에서 Macro-Recall은 6.72%p, Macro-F1은 7.74%p 높아졌습니다. 이는 두 번의 개발 실행에 대한 Validation 비교이지 독립 Test 성능이나 현장 오류 검출률을 보장하지 않습니다. 선택 checkpoint에서 클래스별 Recall은 `None` 97.87%, `Center` 88.13%, `Donut` 79.31%, `Edge-Loc` 84.03%, `Edge-Ring` 86.46%, `Loc` 53.85%, `Near-Full` 100% (18건 중 18건), `Random` 95.51%, `Scratch` 92.31%였습니다. `Scratch` Precision은 30.08%로 낮아 해당 클래스를 많이 놓치지는 않았지만 오탐이 많습니다. `Near-Full`은 Validation 지원 수가 18건뿐이므로 점수가 불안정할 수 있습니다.

선택 모델은 `models/wm811k/cnn_v2_gpu_30epoch/best_model.pt`, epoch별 결과·혼동행렬은 `models/wm811k/cnn_v2_gpu_30epoch/training_report.json`에 있습니다. Validation 결과만으로 학습·checkpoint 선택을 완료한 뒤 아래에서 별도 Test 평가를 수행했습니다.

## 동결 CNN Test 단일 평가 — 2026-10-25

Validation Macro-F1 기준으로 선택한 epoch 14 checkpoint와 전처리·클래스·분할 계약을 검증한 뒤 `src/wafer_dl/evaluate.py`로 Test를 한 번 평가했습니다. `weights_only=True`로 checkpoint를 읽고 학습된 모델과 Train 최빈 클래스 기준선을 같은 Test 표본에 적용했습니다. Test는 모델 선택이나 조정에 사용하지 않았습니다.

| 지표 | Train 최빈 클래스 기준선 | 동결 작은 CNN | 차이 |
| --- | ---: | ---: | ---: |
| Accuracy | 88.38% | 96.13% | +7.75%p |
| Macro-Recall | 11.11% | 85.61% | +74.50%p |
| Macro-F1 | 10.43% | 77.17% | +66.74%p |

Test 25,834건 중 22,831건은 `None`입니다. CNN의 클래스별 Recall은 `None` 97.99%, `Center` 88.68%, `Donut` 67.24%, `Edge-Loc` 81.39%, `Edge-Ring` 85.54%, `Loc` 58.10%, `Near-Full` 100% (18건), `Random` 93.26%, `Scratch` 98.28%였습니다. `Scratch` Precision은 30.56%로 낮아 이 패턴의 오탐이 많습니다. 희소 클래스, 특히 `Near-Full`의 작은 지원 건수로 점수 변동이 클 수 있습니다.

Accuracy만으로는 불균형 영향을 가릴 수 있어 Macro 지표와 클래스별 결과를 함께 봐야 합니다. 이 결과는 현재 WM-811K 그룹 분할 안의 패턴 분류 일반화 평가이며, 실제 공정의 불량 검출률·보정된 확률·SECOM과 통합된 성능을 뜻하지 않습니다. Test 결과를 확인했으므로 이 Test로 추가 튜닝하지 않습니다. 모델 개선을 다시 시작한다면 새 독립 평가 데이터를 확보해야 합니다. ResNet18 비교, seed 변동, 운영 기준 검증은 아직 완료되지 않았습니다.

전체 클래스 지표·혼동행렬·실행 시간·장치 정보는 로컬의 `reports/wm811k/test_evaluation_group_v2_macro_f1/test_evaluation.json`, 동결 checkpoint는 `models/wm811k/cnn_v2_gpu_30epoch/best_model.pt`에 있습니다. 이 실행별 보고서와 checkpoint는 저장소에 올리지 않습니다. 저장소에는 소스·설정·테스트와 요약 지표만 포함합니다.

다음은 같은 절차를 재현하기 위한 명령입니다. 이미 생성된 출력 폴더는 덮어쓰지 않습니다. Test 배열을 읽는 단계임을 확인한 경우에만 승인 플래그를 사용합니다.

```powershell
python -m src.wafer_dl.cache `
  --split-dir data/splits/wm811k/group_v2 `
  --check-dir reports/wm811k/transform_v2 `
  --output-dir data/processed/wm811k/input_cache/wm_resize_v2_test `
  --canonical-dir data/processed/wm811k/canonical/canonical_1b2ac8be615a462786dd4d5dbcfb8b1e `
  --splits test `
  --confirm-visual-review `
  --confirm-test-preparation
```

첫 배치 확인과 동결 평가 예시는 다음과 같습니다.

```powershell
python -m src.wafer_dl.dataset `
  --cache-dir data/processed/wm811k/input_cache/wm_resize_v2_test `
  --split test `
  --confirm-test-preparation

python -m src.wafer_dl.evaluate `
  --cache-dir data/processed/wm811k/input_cache/wm_resize_v2_test `
  --checkpoint models/wm811k/cnn_v2_gpu_30epoch/best_model.pt `
  --output-dir reports/wm811k/test_evaluation_reproduction `
  --device auto `
  --confirm-test-evaluation
```

평가기와 Dataset 모두 명시적 Test 승인 없이는 열리지 않습니다. 승인 옵션은 실수 방지 장치이지 운영체제 보안 경계가 아닙니다. Test 준비 중 변환 소실이 발생하면 자동 보정하지 않고 중단하며 별도 검토합니다. 한 번 공개한 Test 성능을 반복적인 모델 선택·개선에 사용하지 않습니다.

## 합성 테스트

```powershell
python -m unittest tests.test_wafer_cache_dataset tests.test_wafer_transform tests.test_wafer_split -v
```

실제 데이터·외부 pickle·모델 학습 없이 저장·복원 정합성, Test 읽기 제한, 승인·덮어쓰기 거부, 손상·색인 중복 거부, Tensor 독립성과 worker 전달 상태를 검사합니다. 다중 worker 실제 실행은 아직 검증 범위가 아닙니다.

상세 캐시·색인·표본·집계·경로는 로컬 산출물입니다. 향후 전처리 에이전트의 외부 LLM·LangSmith 입력으로 그대로 전달하면 안 됩니다.
