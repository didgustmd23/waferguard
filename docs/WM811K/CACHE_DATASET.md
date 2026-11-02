# WM-811K 입력 캐시와 PyTorch Dataset

## 현재 범위

`src/wafer_dl/cache.py`는 확정 분할과 변환 검사를 확인한 뒤 숫자 전용 uint8 NPY 캐시를 생성합니다. `src/wafer_dl/dataset.py`는 선택 구간의 캐시를 읽어 PyTorch 입력을 구성합니다. 실제 Train·Validation 캐시·배치 확인을 마쳤고, 이 캐시로 학습 39회와 Validation 재추론을 사용자가 로컬 GPU에서 실행했습니다. 사용자는 Validation 결과를 보고 ResNet18+D4 고정 학습률을 종합후보로, SmallWaferCNN+D4 ReduceLROnPlateau를 비교후보로 정해 각 3 seed를 `reports/wm811k/final_freeze_v1`에 동결했습니다. 튜닝과 동결에는 Test를 사용하지 않았고, 동결 뒤 별도 Test 캐시 `wm_resize_final_test_v1`로 최종 평가를 마쳤습니다. 결과는 [보고서 22절](../../reports/wm811k/report.md#22-최종-test-결과)에 있습니다. 실험 집계 도구의 `선정 보류`는 자동 판정을 하지 않는다는 표시이며 이 사용자 결정과 별개입니다.

## 처리 흐름과 분할 파일

```text
원본 pickle → 숫자 전용 canonical → 검증·EDA → 그룹 분할
    → Train/Validation 맵 변환·형태 확인 → uint8 캐시
    → Dataset 2채널 Tensor → Train/Validation 첫 배치 검사
    → 모델 학습·Validation 비교 → 사용자 결정으로 두 후보 동결
    → 사전점검 → 별도 Test 캐시 → 최종 평가·집계 (완료)
```

Train 약 70%·Validation 약 15%·Test 약 15%의 분할을 확정했습니다. 실험 캐시 `wm_resize_v1`에는 Train·Validation만 있고, Test는 `wm_resize_final_test_v1`에 따로 있습니다. 원본 맵을 세 폴더에 복사해서 나누는 대신, manifest에 각 행의 소속을 기록합니다.

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

이 결과는 선택 구간의 캐시 정합성과 첫 배치 입력 계약 확인이며 성능 평가가 아닙니다. 이후 명령은 재현·전달용이며 이미 완료한 캐시를 다시 생성할 필요는 없습니다.

## 현재 결과 경로와 새 경로

현재 분할·변환 검사·캐시는 이미 있으므로 **다시 만들 필요가 없습니다.** 아래 현재 경로는 조회와 입력으로만 쓰고, 출력으로 지정하지 않습니다. 다른 조건으로 새로 만들 때만 아직 없는 새 경로를 정합니다. 명령의 `<...>`는 실제 이름으로 바꾸는 자리이며 그대로 실행하면 안 됩니다.

| 항목 | 현재 경로 (입력·조회) | 새로 만들 때 |
| --- | --- | --- |
| 분할 | `data/splits/wm811k/group_v1` | `data/splits/wm811k/<새_분할_이름>` |
| 변환 검사 | `reports/wm811k/transform_v1` | `reports/wm811k/<새_검사_이름>` |
| Train·Validation 캐시 | `data/processed/wm811k/input_cache/wm_resize_v1` | `data/processed/wm811k/input_cache/<새_캐시_이름>` |

폴더 이름의 `v1`은 저장 위치 이름입니다. 과거 문서의 `group_v2`, `wm_resize_v2`, `transform_v2`는 현재 저장소 환경에 없는 경로이므로 필수 입력이 아닙니다.

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

## 1. 새 경로에 개발 구간 캐시 생성

현재 캐시 `wm_resize_v1`을 쓸 때는 이 단계를 건너뛰고 2단계로 갑니다. 분할이나 변환이 달라 새 캐시가 필요할 때만 실행합니다.

변환 검사 결과가 `transform_check_passed`이고 `examples.png`에서 패턴 보존을 직접 확인한 경우에만 `--confirm-visual-review`를 지정합니다. 이 옵션은 자동 시각 검사를 뜻하지 않습니다.

```powershell
$splitDir = "data/splits/wm811k/<사용할_분할_이름>"
$checkDir = "reports/wm811k/<그_분할의_변환_검사_이름>"
$newCacheDir = "data/processed/wm811k/input_cache/<새_캐시_이름>"

python -m src.wafer_dl.cache `
  --split-dir $splitDir `
  --check-dir $checkDir `
  --output-dir $newCacheDir `
  --confirm-visual-review
```

`$checkDir`은 `$splitDir`로 실행한 변환 검사 결과여야 합니다. canonical 경로는 분할 보고서에서 읽습니다. 데이터 묶음을 이동했다면 같은 원본 묶음의 현재 위치를 `--canonical-dir`로 지정할 수 있습니다.

완료 기준은 `cache_complete`와 출력 폴더의 `SUCCESS.json`입니다. 기존 출력 폴더는 덮어쓰지 않습니다. 중단된 폴더는 완료 결과로 사용하지 말고, 원인을 확인한 후 다른 새 출력 경로로 실행하세요.

- 기본 대상은 Train·Validation입니다. Test 배열은 읽지 않습니다.
- 미라벨·격리 행은 학습 캐시에 포함하지 않습니다.
- 원본 맵·라벨·분할을 변경하거나 다시 나누지 않습니다.
- 선택한 배열의 무결성을 확인하고 기존 결정적 resize·padding을 재사용합니다.
- 유효 영역 또는 불량 영역이 완전히 사라지면 중단합니다. 일부 형태 변화까지 없다는 보장은 아닙니다.

## 2. 첫 배치 확인

각각 실행합니다. 아래는 현재 캐시를 읽기만 하는 확인입니다. 새 캐시를 만들었다면 `$cacheDir`을 그 경로로 바꿉니다.

```powershell
$cacheDir = "data/processed/wm811k/input_cache/wm_resize_v1"

python -m src.wafer_dl.dataset `
  --cache-dir $cacheDir `
  --split train `
  --batch-size 32

python -m src.wafer_dl.dataset `
  --cache-dir $cacheDir `
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
이전 checkpoint는 새 계약으로 평가할 수 없습니다. 기존 파일에 해시를 임의로
추가하지 않으며, 현행 계약으로 새 평가를 하려면 그 계약으로 생성한 checkpoint가
필요합니다.

```python
from torch.utils.data import DataLoader
from src.wafer_dl.dataset import WaferDataset

dataset = WaferDataset("data/processed/wm811k/input_cache/wm_resize_v1", split="train")
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

`suggested_train_class_weights`는 Train 빈도만 사용한 `N_train/(9×클래스 표본 수)` 후보입니다. 학습 코드는 이 역빈도 값에 공통 배율과 상한을 적용해 클래스별 가중치 평균이 1이 되도록 맞춥니다. 따라서 캐시의 원시 후보에 상한만 적용한 값과는 다를 수 있습니다. 가중치 설정의 비교 결과는 결과 보고서 18절에 있습니다. 기본 학습은 증강·추가 정규화 없이 진행하고, D4 증강은 아래 재실험 절의 학습 옵션으로만 켭니다. 저장된 입력 변환을 다시 학습하지 않습니다.

## CNN 학습과 Validation

`src/wafer_dl/models.py`의 작은 CNN은 2채널 입력과 9종 logits를 사용합니다. `src/wafer_dl/train.py`는 확인된 캐시의 Train으로 학습하고 Validation으로 매 epoch 분류 지표를 계산합니다. 기본 checkpoint 선택 기준은 Validation Macro-F1이며 클래스별 Precision·Recall·F1·지원 수와 혼동행렬을 함께 기록합니다. Recall을 checkpoint 선택의 우선 기준으로 삼을 실험은 설정의 `selection_metric`을 `macro_recall`로 바꿀 수 있습니다. 이 경우에도 Precision과 클래스별 혼동을 같이 살펴야 합니다. Train 기반 class-weighted loss는 희소 패턴을 고려하는 실험 출발점이지 Recall 개선을 보장하지 않습니다. 코드 구현과 실제 모델 성능 확인은 구분합니다.

학습 중 클래스 가중치·모델 파라미터·logits·loss의 NaN/무한대를 검사해 발견 즉시
중단합니다. epoch별 weighted loss는 배치 평균을 표본 수로 단순 평균하지 않고
정답 클래스 가중치 합을 사용해 집계합니다.

먼저 Train·Validation 입력 캐시가 존재하고 GPU/PyTorch 환경이 준비되어 있어야 합니다. 프로젝트 루트에서 실행합니다. 입력은 현재 캐시이고, 출력은 아직 없는 새 폴더로 정합니다.

```powershell
python -m src.wafer_dl.train `
  --cache-dir data/processed/wm811k/input_cache/wm_resize_v1 `
  --output-dir models/wm811k/<새_실행_이름> `
  --config configs/wm811k/train.json `
  --device auto
```

`--device auto`는 CUDA를 사용할 수 있으면 GPU를, 아니면 CPU를 선택합니다. 실행은 Test를 열지 않습니다. 완료 시 `best_model.pt`와 `training_report.json`이 출력 폴더에 저장되고, 기존 출력 폴더가 있으면 덮어쓰지 않고 실패합니다. 보고서에는 split protocol, 선택 epoch, epoch별 Macro-F1·Macro-Recall·클래스별 Recall과 혼동행렬이 들어갑니다. 같은 경로에 재실행하려면 기존 결과를 보존한 채 새 `--output-dir`을 지정하세요.

### 현재 실험 상태

사용자가 로컬 GPU에서 실행한 현재 실험은 다음과 같습니다. 결과 수치와 해석은 [결과 보고서 15~23절](../../reports/wm811k/report.md#15-현재-모델-실험의-범위와-근거)에 한 곳으로 모읍니다.

| 실험 | 규모 | 입력 | 결과 |
| --- | ---: | --- | --- |
| 초기 모델 비교 | 18회 | `models/wm811k/compare_v2/` | `reports/wm811k/results/compare_v2/` |
| ResNet18+D4 Validation 재추론 | 3 seed, 학습 아님 | `compare_v2`의 ResNet18+D4 | `reports/wm811k/validation_errors/resnet18_d4_v1/` |
| ResNet18+D4 가중치 후보 | 9회 | `models/wm811k/weight_v1/<후보>/` | `reports/wm811k/results/weight_v1/` |
| SmallWaferCNN+D4 가중치 후보 | 6회, `scratch0_7` 3회 미실행 | `models/wm811k/weight_cnn_v1/<후보>/` | `reports/wm811k/results/weight_cnn_v1/` |
| 두 모델 학습률 스케줄 | 6회 | `models/wm811k/lr_v1/<모델>/plateau_alpha0_75/` | `reports/wm811k/results/lr_v1/` |
| 동결 후보 최종 평가 | checkpoint 6개, 학습 아님 | `reports/wm811k/final_freeze_v1/` | `reports/wm811k/final_test_v1/` |

위 표의 학습은 합계 39회이며 재추론과 Test 평가는 합산하지 않습니다. 학습 결과는 모두 Train/Validation까지이고 집계 도구의 표시는 `선정 보류`입니다. Validation 비교는 일반화 성능이 아닙니다. 후보 동결은 이 표시와 별도의 사용자 결정입니다.

## Test 사용 이력과 평가 예시

이 데이터에는 과거 Test 평가 이력이 있습니다. 과거 분할과 현재 분할의 Test 표본이 얼마나 겹치는지는 미확인이므로, 이력만으로 현재 Test가 이미 사용됐거나 비독립이라고 단정하지 않고 완전한 미사용을 보장하지도 않습니다. 현재 실험의 학습·튜닝·후보 동결에는 현재 Test를 사용하지 않았습니다. 상세 과거 이력은 Git에서 확인합니다.

**아래 예시는 단일 checkpoint 형식입니다.** 동결 후보 6개는 이어지는 [최종 평가 순서](#동결-후보-6개의-최종-평가-순서)의 명령을 사용합니다. 새 평가에는 현재 계약에 맞는 checkpoint와 별도 실행 결정이 필요합니다. 동결 후보용 Test 캐시는 이미 생성했습니다. 이미 생성된 출력 폴더는 덮어쓰지 않으며, 지정한 경로가 이미 있으면 재생성하지 않습니다. Test 배열을 읽는 단계임을 확인한 경우에만 승인 플래그를 사용합니다.

```powershell
$testCacheDir = "data/processed/wm811k/input_cache/<새_Test_캐시_이름>"

python -m src.wafer_dl.cache `
  --split-dir data/splits/wm811k/group_v1 `
  --check-dir reports/wm811k/transform_v1 `
  --output-dir $testCacheDir `
  --splits test `
  --confirm-visual-review `
  --confirm-test-preparation
```

canonical 폴더를 옮겼다면 `--canonical-dir "현재 실제 경로"`를 추가합니다.

첫 배치 확인과 동결 평가 형식은 다음과 같습니다. `<최종_후보_checkpoint_폴더>`는 확정한 후보의 현행 계약 checkpoint 폴더로 바꿔야 합니다.

```powershell
python -m src.wafer_dl.dataset `
  --cache-dir $testCacheDir `
  --split test `
  --confirm-test-preparation

python -m src.wafer_dl.evaluate `
  --cache-dir $testCacheDir `
  --checkpoint "models/wm811k/<최종_후보_checkpoint_폴더>/best_model.pt" `
  --output-dir reports/wm811k/test_evaluation_<새_평가_이름> `
  --device auto `
  --confirm-test-evaluation
```

평가기와 Dataset 모두 명시적 Test 승인 없이는 열리지 않습니다. 승인 옵션은 실수 방지 장치이지 운영체제 보안 경계가 아닙니다. Test 준비 중 변환 소실이 발생하면 자동 보정하지 않고 중단하며 별도 검토합니다. 한 번 공개한 Test 성능을 반복적인 모델 선택·개선에 사용하지 않습니다.

### 동결 후보 6개의 최종 평가 순서

**이 절차는 완료했습니다.** 세션 시도는 실행별 1회이고 실패는 없습니다. 아래 명령은 재현·검토 기록이며 다시 실행하지 않습니다. 같은 Test를 다시 평가하려면 별도 결정이 필요하고, 기존 `final_test_v1`은 덮어쓰지 않습니다.

동결 명세 `reports/wm811k/final_freeze_v1`의 ResNet18·SmallWaferCNN 각 3 seed를 `src/wafer_dl/final_test.py`로 평가합니다. 순서는 동결 명세 → 사전점검 → 별도 캐시 생성 → 재점검·평가 → 집계이고, 각 명령을 사람이 따로 실행합니다. 캐시 생성과 평가는 자동으로 이어지지 않습니다.

```powershell
$freezeSha = '437f19874de4ce67e4d23baba946e47853f2f8c28fe49b585cafd91dc01d6f1c'
$testCacheDir = 'data/processed/wm811k/input_cache/wm_resize_final_test_v1'

# 1. 사전점검: Test 배열·추론 없음, 승인 아님, 파일 출력 없음
python -m src.wafer_dl.final_test preflight --expected-freeze-sha256 $freezeSha

# 2. Test 캐시 생성: 개발 구간 examples.png를 직접 확인한 경우에만 실행
python -m src.wafer_dl.cache `
  --split-dir data/splits/wm811k/group_v1 `
  --check-dir reports/wm811k/transform_v1 `
  --output-dir $testCacheDir `
  --splits test `
  --confirm-visual-review `
  --confirm-test-preparation

# 3. 캐시 보고서·색인 대조: Test 배열은 아직 읽지 않음
python -m src.wafer_dl.final_test preflight --expected-freeze-sha256 $freezeSha --test-cache-dir $testCacheDir

# 4. 평가: 예정된 한 세션에서 동결 모델 상태 6개를 같은 Test로 평가
python -m src.wafer_dl.final_test evaluate `
  --expected-freeze-sha256 $freezeSha `
  --test-cache-dir $testCacheDir `
  --device cuda `
  --confirm-final-test-evaluation

# 5. 집계: 저장 결과만 검증해 비교표 생성
python -m src.wafer_dl.final_test summarize
```

`canonical_root`를 옮겼다면 사전점검·평가·캐시 명령에 `--canonical-dir "현재 실제 경로"`를 붙입니다. pickle은 다시 읽지 않습니다. 결과는 `reports/wm811k/final_test_v1/<모델>/seed<n>/`과 `comparison/`에 생깁니다. 평가 중에는 `attempts.jsonl`에 시작·읽은 행 수·종료만 남고 지표는 출력하지 않습니다. 평가와 집계에는 Test로 모델·seed를 다시 고르는 동작이 없습니다.

| 상황 | 동작과 다음 행동 |
| --- | --- |
| 사전점검 실패 (`FREEZE_*`, `SPLIT_*`, `TRAIN_CACHE_*`, `TRANSFORM_CONFIG_MISMATCH`) | 평가하지 않습니다. 동결 파일을 고치거나 다른 checkpoint로 바꾸지 말고 변경 경위를 확인합니다. |
| `CACHE_PATTERN_LOSS_REVIEW_REQUIRED`, 변환 계약 위반 | 멈추고 원인을 검토합니다. Test를 보고 해상도·전처리를 바꾸거나 문제 표본을 지우지 않습니다. 완료 표식 없는 폴더는 사용하지 않습니다. |
| `TEST_CACHE_*` | 평가하지 않습니다. 캐시 폴더를 고치지 말고 경로와 생성 입력을 확인합니다. Train·Validation 캐시에 Test를 추가하지 않습니다. |
| 평가 중 한 건 실패 (`final_test_stopped`) | 그 자리에서 멈추고 남은 건을 평가하지 않습니다. 성공 건은 보존되고 실패 건의 부분 폴더는 `seed<n>.incomplete_N`이 됩니다. |
| 프로세스 강제 종료 | 마지막 기록이 `started`나 진행 행 수로 끝나 `interrupted`로 표시됩니다. |
| 재개 | `python -m src.wafer_dl.final_test status`로 완료·실패·남은 건과 읽은 행 수를 봅니다. 사람이 판단한 뒤 4단계 명령에 `--resume`을 붙이고, 다시 평가할 실패·중단 건만 `--retry 모델:seed`로 지정합니다. 완료 건은 해시를 대조하고 다시 평가하지 않습니다. 학습 재개와 무관합니다. |
| `FINAL_TEST_OUTPUT_EXISTS`, `FINAL_TEST_REPORT_ALREADY_EXISTS` | 기존 결과를 덮어쓰지 않습니다. 세션을 이어 갈 때만 `--resume`을 씁니다. |
| `FINAL_TEST_SESSION_MISMATCH` | 동결·캐시·추론 코드·장치가 세션 시작 때와 다릅니다. 변경을 되돌리기 전에는 이어서 평가하지 않습니다. |
| `FINAL_TEST_RUN_MISSING`, `FINAL_TEST_EXTRA_RUN`, `FINAL_TEST_MIXED_TEST`, `FINAL_TEST_RUN_*`, `FINAL_TEST_METRIC_MISMATCH` | 집계를 공개하지 않습니다. 결과를 고치지 말고 빠진 건은 재개 절차로, 혼입·변조는 경위 확인 후 처리합니다. |

`comparison.md`의 수치는 튜닝 구성 비교입니다. 같은 Test 표본을 seed별로 반복 평가한 값이라 독립 표본이 늘어난 것이 아니며, 평균 차이를 통계적 우월성으로 해석하지 않습니다. 수치와 해석은 [보고서 22~23절](../../reports/wm811k/report.md#22-최종-test-결과)을 따르며 이 문서에 복제하지 않습니다.

## 처음부터 다시 하는 모델 비교

이 절은 사용자 실행 안내입니다. 기본 18회는 실행을 마쳤고 결과는 [결과 보고서](../../reports/wm811k/report.md#16-초기-비교-18회)에 있습니다. 다시 실행할 때는 프로젝트 루트에서 `wm-gpu312` 환경을 활성화한 PowerShell로 순서대로 실행합니다. 아래 새 경로가 이미 있으면 덮어쓰지 말고 다른 이름을 쓰세요.

### 범위와 고정 조건

- 기본 조합은 공간 패턴 분류용 CNN 계열 세 후보 `SmallWaferCNN`, `ResNet18`, `MobileNetV3Small` × `none`, `d4` × seed 42·43·44로 18회입니다. `--models`, `--augmentations`, `--seeds`로 일부만 고를 수 있습니다.
- MLP 계열(`WaferMLP`, `ResNet18MLP`, `MobileNetV3SmallMLP`)은 이번 비교에 포함하지 않습니다. 성능을 비교해 제외한 것이 아니며, 호환성을 위해 코드와 명시 선택 옵션만 유지합니다.
- `d4`는 90도 회전·반전을 Train 읽기 시점에만 두 채널에 같게 적용합니다. 캐시와 Validation·Test·추론에는 적용하지 않습니다. 패턴 라벨이 방향과 무관하다는 가정이므로 notch·장비 방향의 의미를 먼저 확인하세요.
- 이 비교는 Train/Validation까지이며 현재 Test를 사용하지 않습니다. 과거 Test 평가 이력과 현재 Test의 표본 겹침은 미확인이므로 어느 쪽으로도 단정하지 않습니다. 최종 Test는 동결 후 [최종 평가 순서](#동결-후보-6개의-최종-평가-순서)로 수행합니다.
- 집계 도구는 최종 모델을 자동 선정하지 않고 `선정 보류`로 표시해 지표를 나란히 보여 줍니다. 후보 결정은 사람이 합니다.

### 1. 원본 식별과 canonical 검사

원본 파일을 기록합니다. 기대 파일은 `data/raw/wm811k/LSWMD.pkl`입니다.

```powershell
Get-FileHash data/raw/wm811k/LSWMD.pkl -Algorithm SHA256
```

새 canonical은 기본 경로인 [Sandbox 절차](SANDBOX_SETUP.md)로 만듭니다. Sandbox를 쓸 수 없으면 pickle 신뢰 여부를 사용자가 명시적으로 결정한 경우에만 같은 문서의 로컬 경로를 사용합니다. 그 뒤 `$canonicalDir`에 실제 폴더를 지정합니다.

```powershell
$canonicalDir = "data/processed/wm811k/canonical/canonical_<생성ID>"
python -m src.wafer_dl.validate_canonical `
  --input-dir $canonicalDir `
  --output-dir logs/wm811k/rerun_canonical_validation
```

확인값은 `integrity_passed`입니다. 실패하면 이 묶음으로 진행하지 말고 새 canonical을 만드세요. 분할 사전 진단이 `review_required`면 다음 단계에서 사유를 확인합니다.

### 2. 중복·Lot 분할 검증

```powershell
python -m src.wafer_dl.diagnose_split `
  --input-dir $canonicalDir `
  --output-dir logs/wm811k/rerun_split_diagnosis

python -m src.wafer_dl.split `
  --input-dir $canonicalDir `
  --config configs/wm811k/split.json `
  --output-dir data/splits/wm811k/group_rerun_v1 `
  --approve-conflict-exclusion
```

`--approve-conflict-exclusion`은 진단 결과를 확인하고 라벨 충돌 그룹 제외를 승인할 때만 붙입니다. 확인값은 `split_complete`와 출력 폴더의 `SUCCESS.json`, 보고서의 `overlap_check_passed=true`입니다. 실패하면 임계값을 완화하지 말고 [분할 문서](GROUP_SPLIT.md)의 사유를 확인하세요.

### 3. 변환 검사와 캐시

```powershell
python -m src.wafer_dl.check_transform `
  --split-dir data/splits/wm811k/group_rerun_v1 `
  --config configs/wm811k/transform.json `
  --output-dir reports/wm811k/transform_rerun_v1
```

확인값은 `transform_check_passed`입니다. `examples.png`를 직접 본 뒤에만 다음 명령의 `--confirm-visual-review`를 붙입니다. 기본 대상은 Train·Validation이며 Test 배열은 읽지 않습니다.

```powershell
python -m src.wafer_dl.cache `
  --split-dir data/splits/wm811k/group_rerun_v1 `
  --check-dir reports/wm811k/transform_rerun_v1 `
  --output-dir data/processed/wm811k/input_cache/wm_resize_rerun_v1 `
  --confirm-visual-review

python -m src.wafer_dl.dataset `
  --cache-dir data/processed/wm811k/input_cache/wm_resize_rerun_v1 `
  --split train
```

확인값은 `cache_complete`와 `dataset_batch_passed`입니다. 중단된 캐시 폴더는 완료 결과로 쓰지 말고 새 경로로 다시 실행합니다.

### 4. 후보 학습

먼저 실행 수를 확인합니다. 학습은 하지 않으며 `pending_max_epochs`는 남은 epoch의 상한입니다. 조기 종료되면 더 짧아집니다.

```powershell
$cacheDir = "data/processed/wm811k/input_cache/wm_resize_rerun_v1"
python -m src.wafer_dl.compare plan --cache-dir $cacheDir --device cuda
```

CNN 기준선부터 실행해 정상 동작을 확인한 뒤 나머지를 이어서 실행합니다. 같은 `--output-root`에서는 캐시·설정·조합과 실행 코드가 같은 완료 실행을 건너뜁니다. 실행 코드 식별값은 학습 경로의 `src/wafer_dl` 파일 9개(`augment`, `cache`, `compare`, `dataset`, `metrics`, `models`, `train`, `transform`, `validate_canonical`) 내용으로 계산합니다. 혼용 검사용이며 인증이나 파일 잠금이 아닙니다. 실험을 시작한 뒤에는 이 파일들을 수정하지 마세요.

```powershell
python -m src.wafer_dl.compare run --cache-dir $cacheDir --device cuda --models SmallWaferCNN --augmentations none --seeds 42

python -m src.wafer_dl.compare run --cache-dir $cacheDir --device cuda
```

실행별 완료 표식은 `models/wm811k/compare_v2/<모델>_<증강>_seed<n>/SUCCESS.json`입니다. 마지막 출력의 `failures`가 빈 목록이어야 합니다.

| 상황 | 동작과 다음 행동 |
| --- | --- |
| 중간에 끊김 | 같은 명령을 다시 실행합니다. `.tmp` 폴더는 `.incomplete_<n>`으로 보존되고 그 실행만 새로 학습합니다. |
| `failures`에 실패 표시 | 나머지 실행은 계속됩니다. 코드 변경 없이 해결되는 원인(메모리 부족, 디스크 공간 등)이면 같은 명령을 다시 실행합니다. 공개되지 않은 `.tmp` 폴더는 `.incomplete_<n>`으로 보존됩니다. |
| 해결에 위 9개 파일 수정이 필요함 | 기존 완료 결과와 코드가 달라져 재사용이 거부됩니다. 기존 결과를 두고 새 `--output-root`에서 전체 조합을 다시 실행합니다. |
| `COMPARE_RUN_CODE_MISMATCH` | 실행 코드가 다른 결과입니다. 기존 결과를 두고 새 `--output-root`에서 같은 코드로 실행합니다. 학습 중 코드를 바꾸면 그 실행은 완료로 공개되지 않습니다. |
| `COMPARE_RUN_CONTRACT_OUTDATED` | 코드 식별값이 없는 이전 계약의 결과입니다. 자동으로 갱신하지 않으므로 새 `--output-root`를 지정합니다. |
| `COMPARE_RUN_CONTRACT_MISMATCH` | 캐시·설정·증강·장치가 다른 결과입니다. 기존 결과를 두고 새 `--output-root`를 지정합니다. |
| `COMPARE_RUN_INCOMPLETE`, `COMPARE_RUN_INTEGRITY_FAILED` | 해당 폴더를 재사용할 수 없습니다. 지우지 말고 새 `--output-root`를 지정합니다. |

### 5. 조건별·seed별 비교

```powershell
python -m src.wafer_dl.compare summarize
```

일부 조합만 실행했다면 같은 `--models`, `--augmentations`, `--seeds`를 붙입니다. 결과는 `reports/wm811k/results/compare_v2/`의 `comparison.md`, `summary.csv`, `runs.csv`입니다. `COMPARE_REPORT_ALREADY_EXISTS`가 나오면 새 `--report-dir`을 지정하세요. 실행이 빠지면 `COMPARE_RUN_INCOMPLETE`로 중단되므로 4단계를 먼저 끝냅니다. 서로 다른 코드의 실행이 섞이면 `COMPARE_RUN_CODE_MISMATCH`로 중단하고 집계를 공개하지 않습니다. 한 코드로 완결된 과거 묶음은 집계할 수 있으며, 현재 구조로 가중치를 다시 로딩하지 않고 `comparison.md`에 현재 코드와 다르다고 표시합니다.

이전 코드의 `compare_pilot`과 계약 버전이 다른 결과는 재사용하지 않습니다. 증강 키가 없는 기존 설정은 `none`으로 읽고, 기존 `SmallWaferCNN` checkpoint의 Test 평가 계약은 유지합니다.

## ResNet18+D4 Validation 오분류 분석

`compare_v2`의 ResNet18+D4 세 seed checkpoint로 Validation만 재추론합니다. 학습·재학습·앙상블·Test 평가는 없고 재추론에는 증강을 적용하지 않습니다. 입력은 읽기만 하며 예측을 근거로 라벨을 수정하지 않습니다. 결과는 모델 선택에 쓰인 Validation 분석이므로 일반화 성능이 아닙니다.

먼저 seed 하나로 출력을 확인한 뒤 전체를 처리합니다. 두 명령의 출력 경로는 서로 다릅니다.

```powershell
conda activate wm-gpu312
python -m src.wafer_dl.validation_errors --device cuda --seeds 42 --output-dir reports/wm811k/validation_errors/resnet18_d4_check_seed42
python -m src.wafer_dl.validation_errors --device cuda
```

기본 입력은 `data/processed/wm811k/input_cache/wm_resize_v1`, `models/wm811k/compare_v2`이고 기본 출력은 `reports/wm811k/validation_errors/resnet18_d4_v1/`입니다. `--cache-dir`, `--input-root`, `--output-dir`, `--batch-size`, `--max-images-per-group`(기본 24, 최대 200), `--images-per-page`(기본 12, 최대 12)로 바꿀 수 있습니다.

완료 폴더에는 `SUCCESS.json`이 있습니다. `report.md`부터 읽고, seed별 전체 예측은 `predictions_seed*.csv`, 표본별 결합은 `joined_predictions.csv`, 그림에서 생략된 표본까지 포함한 우선 검토 목록은 `priority_errors.csv`, 그림은 `images/`에서 확인합니다. 그림은 캐시의 128×128 모델 입력을 3색으로 합성한 것이며 원본 사진이 아닙니다. 표본은 그룹별로 오류 seed 수 내림차순, 동률은 `sample_id` 순으로 선정합니다.

| 실패 코드·상태 | 다음 행동 |
| --- | --- |
| `VALIDATION_ERRORS_OUTPUT_EXISTS`, `VALIDATION_ERRORS_PARTIAL_EXISTS` | 기존 폴더를 지우지 말고 새 `--output-dir`을 지정합니다. |
| `VALIDATION_ERRORS_CODE_MISMATCH` | 학습 당시와 코드가 달라 예측하지 않았습니다. 학습 당시 코드로 되돌립니다. 분석만을 위해 재학습하거나 식별값을 고치지 않습니다. |
| `VALIDATION_ERRORS_CACHE_MISMATCH` | 해당 실행이 학습에 쓴 캐시를 `--cache-dir`로 지정합니다. |
| `VALIDATION_ERRORS_CONFUSION_MISMATCH` | 저장된 혼동행렬이 재현되지 않아 완료로 공개하지 않습니다. `<출력 이름>.mismatch_N/MISMATCH.json`의 차이와 장치를 확인합니다. CPU 실행은 학습 GPU와 미세하게 다를 수 있습니다. |
| `VALIDATION_ERRORS_INPUT_CHANGED` | 분석 중 입력이나 코드가 바뀌었습니다. 다른 작업이 끝난 뒤 새 경로로 다시 실행합니다. |
| `COMPARE_RUN_INCOMPLETE`, `COMPARE_RUN_INTEGRITY_FAILED` | 다른 checkpoint로 대체하지 않습니다. `--input-root`와 해당 실행 폴더를 확인합니다. |
| `<출력 이름>.incomplete_N` 폴더 | 중단된 부분 결과이며 완료가 아닙니다. 원인을 해결한 뒤 실행합니다. GPU 메모리 부족이면 `--batch-size`를 줄입니다. |

학습 경로 코드가 바뀌면 기존 실행 재추론은 `VALIDATION_ERRORS_CODE_MISMATCH`로 거부됩니다. 가중치 설정 추가로 `compare_v2`도 이 상태입니다. 과거 분석은 이미 공개된 `report.md`로 조회합니다.

## ResNet18+D4 순차 가중치 실험

기준은 `compare_v2`의 `ResNet18_d4_seed42·43·44`입니다. 각 후보는 기준에서 독립적으로 바꾼 조건이며 앞 후보를 누적하지 않습니다. 후보 3개 × seed 3개 = 추가 9회입니다. 나머지 조건(ResNet18, D4, 같은 캐시·분할, batch 64, AdamW lr 0.001, weight_decay 0.0001, 최대 30 epoch, patience 6, Validation `macro_f1` 선택)은 같습니다.

| 순서 | 후보 | 설정 파일 | 상한 | alpha | Scratch 계수 | 바뀌는 범위 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 1 | `cap1_5` | `configs/wm811k/weight_v1/cap1_5.json` | 1.5 | 1 | 1 | 희소 클래스 여러 개가 함께 상한에 걸림 |
| 2 | `alpha0_75` | `configs/wm811k/weight_v1/alpha0_75.json` | 10 | 0.75 | 1 | 모든 클래스 비율 완화 |
| 3 | `scratch0_7` | `configs/wm811k/weight_v1/scratch0_7.json` | 10 | 1 | 0.7 | Scratch만 조정 |

계산 순서는 Train 역빈도 → `alpha` 지수 → Scratch 계수 → 평균 1·상한을 맞추는 공통 배율입니다. 공통 배율은 가중 평균 손실에서 상쇄되므로 클래스 사이 비율로 해석합니다. `class_weight_alpha`, `scratch_weight_factor`가 없는 설정은 1로 읽으며 허용 범위는 0 초과 1 이하입니다. 기존 상한 10은 Near-Full 5.41에도 걸리지 않아 기준선은 순수 역빈도입니다. 적용한 9클래스 가중치는 `training_report.json`과 checkpoint의 `class_weights`에 남습니다.

세 후보의 9회는 실행을 마쳤고 결과는 [결과 보고서](../../reports/wm811k/report.md#18-resnet18d4-가중치-후보-9회)에 있습니다. 다시 실행할 때는 새 `$runs`·보고서 경로로 조건을 하나 실행하고 결과를 확인한 뒤 다음 조건으로 넘어갑니다. 아래는 `cap1_5`이며 `$name`만 `alpha0_75`, `scratch0_7`로 바꿉니다.

```powershell
$name = 'cap1_5'
$cacheDir = 'data/processed/wm811k/input_cache/wm_resize_v1'
$config = "configs/wm811k/weight_v1/$name.json"
$runs = "models/wm811k/weight_v1/$name"
$matrix = @('--models', 'ResNet18', '--augmentations', 'd4')

# 1. 학습 없이 3회인지 확인: total_runs 3, pending_max_epochs 90
python -m src.wafer_dl.compare plan --cache-dir $cacheDir --device cuda --config $config --output-root $runs @matrix

# 2. 학습 3회
python -m src.wafer_dl.compare run --cache-dir $cacheDir --device cuda --config $config --output-root $runs @matrix

# 3. 조건별 집계
python -m src.wafer_dl.compare summarize --output-root $runs --report-dir "reports/wm811k/results/weight_v1/$name" @matrix

# 4. 기준과 읽기 전용 대조
python -m src.wafer_dl.weight_compare --candidate-root $runs --report-dir "reports/wm811k/results/weight_v1/${name}_vs_baseline"
```

`weight_compare`는 학습·추론 없이 저장된 selected_epoch 혼동행렬만 읽습니다. 가중치 설정 외의 조건이 같은지 확인하고, 신구 코드 차이는 숨기지 않고 `comparison.md`에 표시합니다. 기준 실행은 현재 코드와 달라 재로딩하지 않습니다. 결과는 `comparison.md`, `summary.csv`, `seed_differences.csv`, `comparison.json`이며 항상 `선정 보류`입니다. 추가 튜닝에 쓴 Validation 결과이므로 일반화 성능이나 통계적 우월성을 뜻하지 않습니다.

판정 참고안은 `configs/wm811k/weight_v1/decision_reference.json`입니다. 사용자가 수치를 승인한 기준이나 유의성 기준이 아니며, 결과를 본 뒤 바꾸면 사후 변경으로 기록합니다. FN 증가 합계는 (후보 3seed FN 합 − 기준 3seed FN 합)이고, 같은 Validation의 반복 예측이라 독립 표본 수가 아닙니다.

| 실패 코드·상태 | 다음 행동 |
| --- | --- |
| 학습이 끊겨 `*.tmp`가 남음 | 코드가 같으면 같은 `run` 명령을 다시 실행합니다. `.tmp`는 `*.incomplete_N`으로 보존되고 남은 실행만 학습합니다. |
| `COMPARE_RUN_INCOMPLETE` | 완료 표식 없는 실행 폴더입니다. 지우지 말고 새 `$runs` 경로로 3회를 실행합니다. |
| 3회 중간에 코드가 바뀜 (`COMPARE_RUN_CODE_MISMATCH`) | 이어서 학습할 수 없습니다. 코드를 되돌리거나, 기존 폴더를 보존한 채 새 `$runs` 경로로 3회를 전부 실행합니다. |
| `WEIGHT_COMPARE_CODE_MIXED` | 한 조건에 다른 코드가 섞였습니다. 새 경로에 같은 코드로 3회를 실행합니다. |
| `WEIGHT_COMPARE_CONDITION_MISMATCH` | 가중치 외 조건이 다릅니다. 후보 설정·캐시를 확인하고 새 경로에 다시 학습합니다. |
| `WEIGHT_COMPARE_RECORD_MISMATCH` | 기록된 가중치가 설정과 다릅니다. 해당 실행을 대조에 쓰지 않습니다. |
| `WEIGHT_COMPARE_REPORT_ALREADY_EXISTS`, `WEIGHT_COMPARE_PARTIAL_EXISTS`, `COMPARE_REPORT_ALREADY_EXISTS` | 기존 폴더를 지우지 말고 새 `--report-dir`을 지정합니다. |

조건마다 다른 `$runs`, 보고서 경로를 씁니다. 설정 파일과 `configs/wm811k/train.json`을 덮어쓰지 않습니다.

## SmallWaferCNN+D4 순차 가중치 실험

ResNet18과 같은 세 후보를 `SmallWaferCNN`+D4에 적용합니다. 기준은 `compare_v2`의 `SmallWaferCNN_d4_seed42·43·44`이며 증강 없는 CNN이나 ResNet18과 대조하지 않습니다. 설정 파일은 모델 이름을 갖지 않으므로 `configs/wm811k/weight_v1/`을 복제 없이 재사용합니다. 후보 3개 × seed 3개 = 추가 9회이고, 완료하면 초기 18회 + ResNet18 9회 + CNN 9회 = 학습 36회입니다. 각 후보는 기준에서 독립적으로 바꾼 조건이며 누적하지 않습니다.

조건을 하나 실행하고 결과를 확인한 뒤 다음 조건으로 넘어갑니다. 첫 조건은 `cap1_5`이고 다음은 `$name`만 `alpha0_75`, `scratch0_7`로 바꿉니다. 출력은 ResNet18 결과와 섞이지 않도록 `weight_cnn_v1` 아래에 둡니다.

```powershell
$name = 'cap1_5'
$cacheDir = 'data/processed/wm811k/input_cache/wm_resize_v1'
$config = "configs/wm811k/weight_v1/$name.json"
$runs = "models/wm811k/weight_cnn_v1/$name"
$reports = "reports/wm811k/results/weight_cnn_v1"
$matrix = @('--models', 'SmallWaferCNN', '--augmentations', 'd4')

# 1. 학습 없이 3회인지 확인: total_runs 3, pending_max_epochs 90
python -m src.wafer_dl.compare plan --cache-dir $cacheDir --device cuda --config $config --output-root $runs @matrix

# 2. 학습 3회
python -m src.wafer_dl.compare run --cache-dir $cacheDir --device cuda --config $config --output-root $runs @matrix

# 3. 조건별 집계
python -m src.wafer_dl.compare summarize --output-root $runs --report-dir "$reports/$name" @matrix

# 4. 같은 모델 기준과 읽기 전용 대조
python -m src.wafer_dl.weight_compare --model SmallWaferCNN --candidate-root $runs --report-dir "$reports/${name}_vs_baseline"
```

기존 CNN+D4 평균 52.6분을 그대로 적용하면 9회에 약 7.9시간입니다. 조기 종료가 달라질 수 있는 거친 참고값이며 보장값이 아닙니다.

`weight_compare`의 `--model` 기본값은 `ResNet18`이고 지원 모델은 `ResNet18`, `SmallWaferCNN`입니다. 다른 모델·다른 증강·누락 seed·다른 조건의 실행은 거부합니다. 판정 참고안은 같은 `decision_reference.json`을 상대 차이와 FN 순증으로 재사용하며, 사용자 승인 운영 기준이나 유의성 기준이 아닙니다. ResNet18의 절대 수치를 CNN 기준으로 쓰지 않습니다. 결과는 항상 `선정 보류`이고, 두 모델의 설정을 Validation으로 정한 뒤 별도 동결·Test 계획을 세웁니다.

중단·충돌 대응은 [ResNet18 절의 실패 표](#resnet18d4-순차-가중치-실험)와 같습니다. 경로만 `weight_cnn_v1`로 읽습니다.

- **출력 충돌:** 기존 `$runs`·보고서 폴더를 지우거나 덮어쓰지 말고 새 경로를 지정합니다.
- **부분 결과:** `*.tmp`는 코드가 같으면 같은 `run`으로 이어 가고 `*.incomplete_N`으로 보존됩니다. 완료 표식 없는 실행 폴더는 쓰지 말고 새 `$runs`로 3회를 실행합니다.
- **코드 변경 후 재개:** 학습 경로 파일이 바뀌면 이어서 학습할 수 없습니다. 코드를 되돌리거나 새 `$runs`로 3회를 전부 실행합니다. 실험 중에는 이 파일들을 수정하지 마세요.

## CNN·ResNet18 학습률 감소 스케줄 실험

고정 학습률 0.001 대신 Validation 성능이 정체될 때 학습률을 줄이면 후반 성능과 안정성이 나아지는지 확인합니다. 개선을 보장하지 않으며, seed 변동의 원인이 학습률이라고 단정하지 않습니다. 생성 모델·LLM·RAG와 무관합니다.

기준은 이미 완료된 두 alpha0.75 조건의 seed 42·43·44입니다. CNN 기준은 `models/wm811k/weight_cnn_v1/alpha0_75`, ResNet18 기준은 `models/wm811k/weight_v1/alpha0_75`입니다. 새 후보는 모델별 3회, 합계 6회입니다. CNN `scratch0_7` 3회는 아직 미실행이며 이 실험의 선행 조건이 아니어서 그대로 미실행으로 둡니다. 나중에 실행하면 `train.py`가 바뀐 코드로 학습되고, `weight_compare`가 코드 차이를 표시합니다.

설정은 `configs/wm811k/lr_v1/plateau_alpha0_75.json` 하나를 두 모델이 함께 씁니다. `configs/wm811k/weight_v1/alpha0_75.json`에 `lr_scheduler`만 더한 파일이며 캐시 `wm_resize_v1`, 같은 분할, batch 64, AdamW, 초기 lr 0.001, weight_decay 0.0001, 최대 30 epoch, patience 6, Validation `macro_f1` 선택은 같습니다.

| 스케줄 항목 | 값 |
| --- | ---: |
| 방식 | `ReduceLROnPlateau`, `mode=max` |
| 지표 | Validation `macro_f1` |
| `factor` | 0.5 |
| `patience` | 2 |
| `threshold`, `threshold_mode` | 0.0, `abs` |
| `cooldown` | 0 |
| `min_lr` | 0.00001 |
| `eps` | 1e-8 |

**실제 동작:**

- PyTorch의 `patience=2`는 불개선 2 epoch까지 허용한다는 뜻입니다. 연속 3번째 불개선 epoch의 평가 뒤에 학습률이 절반이 되고, 그 값은 다음 epoch부터 적용됩니다.
- `threshold=0`, `abs`이므로 개선 판정은 checkpoint와 같은 `score > best_score`입니다.
- 조기 종료는 그대로 연속 불개선 6 epoch이고, 학습률이 줄어도 카운터는 초기화되지 않습니다. 최고 점수가 갱신되지 않으면 3번째 뒤에 첫 감소, 6번째 뒤에 두 번째 감소와 종료가 함께 일어나 두 번째 감소는 학습에 쓰이지 않습니다. 첫 감소 뒤 학습 기회는 이 경우 3 epoch입니다.
- 최대 30 epoch를 늘리지 않습니다. 감소가 늦거나 없으면 효과를 볼 기회가 없을 수 있으므로, 그런 실행을 "효과 없음"으로 단정하지 않습니다.

`history`에는 해당 epoch에 쓴 `learning_rate`와 다음 epoch 값 `next_learning_rate`가 구분되어 남습니다. `training_report.json`에는 `lr_scheduler`, `lr_reductions`가, 최고 checkpoint에는 `lr_schedule`(설정, 학습률, 감소 이력, 스케줄러 상태)이 남습니다. 중단된 학습을 checkpoint에서 이어 가는 기능은 없습니다. `lr_scheduler` 키가 없는 기존 설정은 고정 학습률이며 스케줄러 객체를 만들지 않습니다.

실행 전 `plan`에서 `total_runs` 3, `pending_runs` 3, `pending_max_epochs` 90을 확인합니다. 첫 실험은 `SmallWaferCNN`이며, 결과를 확인한 뒤 ResNet18을 실행합니다.

```powershell
# 첫 실험: SmallWaferCNN. 두 번째 실험은 이 줄만 'ResNet18'로 바꿉니다.
$model = 'SmallWaferCNN'
$cacheDir = 'data/processed/wm811k/input_cache/wm_resize_v1'
$config = 'configs/wm811k/lr_v1/plateau_alpha0_75.json'
$runs = "models/wm811k/lr_v1/$model/plateau_alpha0_75"
$reports = "reports/wm811k/results/lr_v1/$model"
$matrix = @('--models', $model, '--augmentations', 'd4')

# 1. 학습 없이 3회인지 확인: total_runs 3, pending_max_epochs 90
python -m src.wafer_dl.compare plan --cache-dir $cacheDir --device cuda --config $config --output-root $runs @matrix

# 2. 학습 3회
python -m src.wafer_dl.compare run --cache-dir $cacheDir --device cuda --config $config --output-root $runs @matrix

# 3. 후보 집계
python -m src.wafer_dl.compare summarize --output-root $runs --report-dir "$reports/plateau_alpha0_75" @matrix

# 4. 같은 모델 고정 학습률 기준과 읽기 전용 대조
python -m src.wafer_dl.lr_compare --model $model --candidate-root $runs --report-dir "$reports/plateau_alpha0_75_vs_baseline"
```

기존 alpha0.75 평균 시간을 그대로 적용하면 CNN 42.2분 × 3 ≈ 2.1시간, ResNet18 37.4분 × 3 ≈ 1.9시간, 합계 약 4시간입니다. 조기 종료가 달라질 수 있는 거친 참고값입니다.

`lr_compare`는 학습·추론 없이 저장된 selected_epoch 혼동행렬과 보고서만 읽습니다. 같은 모델·D4·가중치·seed·캐시·분할·변환·매핑·학습 예산·optimizer 설정인지 확인하고, 스케줄과 명시적인 코드 차이만 비교 축으로 허용합니다. 후보의 학습률 기록은 설정과 저장 점수로 다시 계산해 대조합니다. 기준은 현재 코드로 다시 추론하지 않고 기존 `SUCCESS.json`을 바꾸지 않습니다.

결과 `comparison.md`, `summary.csv`, `seed_differences.csv`, `comparison.json`에는 seed별 값과 평균·표준편차·동일 seed 차이(Macro-F1, 전체 클래스 P/R/F1·FP/FN, 학습 시간, epoch 수, 학습률 감소 횟수), seed별 학습률 동작, 코드·설정·입력 식별값이 들어갑니다. 합격 기준이 없고 항상 `선정 보류`입니다. 가중치 실험의 판정 참고안(`Scratch Precision +5%p` 등)을 적용하지 않습니다. 추가 튜닝에 쓴 Validation 결과이므로 일반화 성능이 아니며, 이 실험은 Test에 접근하지 않습니다.

| 실패 코드·상태 | 다음 행동 |
| --- | --- |
| 학습이 끊겨 `*.tmp`가 남음 | 코드가 같으면 같은 `run` 명령을 다시 실행합니다. `.tmp`는 `*.incomplete_N`으로 보존되고 남은 실행만 학습합니다. |
| `COMPARE_RUN_INCOMPLETE` | 완료 표식 없는 실행 폴더이거나 대조할 실행이 없습니다. 지우지 말고 새 `$runs` 경로로 3회를 실행합니다. |
| `COMPARE_RUN_CONTRACT_MISMATCH` | 설정이 바뀐 뒤 같은 폴더로 재개했습니다. 설정을 되돌리거나 새 `$runs` 경로로 3회를 전부 실행합니다. |
| `COMPARE_RUN_CODE_MISMATCH` | 코드가 바뀐 뒤 재개했습니다. 코드를 되돌리거나 새 `$runs` 경로로 3회를 전부 실행합니다. 실험 중에는 학습 경로 파일을 수정하지 마세요. |
| `COMPARE_REPORT_ALREADY_EXISTS`, `LR_COMPARE_REPORT_ALREADY_EXISTS`, `LR_COMPARE_PARTIAL_EXISTS` | 기존 폴더를 지우지 말고 새 `--report-dir`을 지정합니다. |
| `LR_COMPARE_CONDITION_MISMATCH`, `LR_COMPARE_WEIGHT_MISMATCH` | 다른 조건이 섞였습니다. `$config`, `--candidate-root`, `--baseline-root`를 확인하고 새 경로에 다시 학습합니다. |
| `LR_COMPARE_BASELINE_HAS_SCHEDULER`, `LR_COMPARE_CANDIDATE_HAS_NO_SCHEDULER` | 기준과 후보 폴더가 바뀌었습니다. 경로를 확인합니다. |
| `LR_COMPARE_RECORD_MISMATCH` | 저장 기록이 다시 계산한 값과 다릅니다. 해당 실행을 대조에 쓰지 않습니다. |

**수정 전 상태로 돌아가기:** 기존 모델·캐시·보고서는 바뀌지 않았고, 새 결과는 `lr_v1` 경로에만 생깁니다. 수정 전 코드 사본은 `_workspace/lr_v1_restore_point/files/`에, 해시 목록은 같은 폴더의 `MANIFEST.json`에 있습니다. 자동 복원은 없습니다. 코드가 미커밋 상태이므로 Git HEAD로 되돌리면 이 기준으로 복원되지 않습니다. 학습 경로에서 바뀐 파일은 `src/wafer_dl/train.py` 하나입니다. 복원이 필요하면 실행 중인 학습이 없는지 확인한 뒤 해당 파일만 사본으로 교체합니다. 복원 후에는 `lr_scheduler` 설정이 `INVALID_TRAIN_CONFIG`로 거부되어 새 스케줄 학습을 할 수 없습니다. 이미 완료된 `lr_v1` 결과는 그대로 남습니다.

```powershell
Copy-Item -LiteralPath '_workspace/lr_v1_restore_point/files/src/wafer_dl/train.py' -Destination 'src/wafer_dl/train.py'
```

## 합성 테스트

```powershell
python -m unittest tests.test_wafer_cache_dataset tests.test_wafer_transform tests.test_wafer_split tests.test_wafer_model_compare tests.test_wafer_validation_errors tests.test_wafer_weight_compare tests.test_wafer_lr_compare tests.test_wafer_final_test -v
```

실제 데이터·외부 pickle·모델 학습 없이 저장·복원 정합성, Test 읽기 제한, 승인·덮어쓰기 거부, 손상·색인 중복 거부, Tensor 독립성과 worker 전달 상태를 검사합니다. 다중 worker 실제 실행은 아직 검증 범위가 아닙니다.

상세 캐시·색인·표본·집계·경로는 로컬 산출물입니다. 향후 전처리 에이전트의 외부 LLM·LangSmith 입력으로 그대로 전달하면 안 됩니다.
