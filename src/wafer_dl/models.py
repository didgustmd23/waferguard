"""WM-811K 웨이퍼맵 패턴 분류 모델과 비교 실험용 모델 생성 함수."""

import torch
from torch import nn
from torchvision.models import mobilenet_v3_small, resnet18

from .validate_canonical import require


# 생성 가능한 모델 이름. checkpoint·보고서의 model_name과 CLI 선택지에 그대로 쓰인다.
# MLP 계열은 호환성을 위해 유지하며 --models로 지정할 때만 실행한다.
MODEL_NAMES = (
    'SmallWaferCNN',
    'WaferMLP',
    'ResNet18',
    'ResNet18MLP',
    'MobileNetV3Small',
    'MobileNetV3SmallMLP',
)
# 별도로 지정하지 않을 때 실행하는 기본 비교 실험군. 공간 패턴 분류용 CNN 계열 세 후보다.
DEFAULT_COMPARE_MODELS = (
    'SmallWaferCNN',
    'ResNet18',
    'MobileNetV3Small',
)
# 128×128 전체를 펼치면 32,768차원이라 먼저 32×32 평균 풀링으로 줄인다.
WAFER_MLP_POOL_SIZE = 32
WAFER_MLP_HIDDEN_SIZES = (512, 256)
# 튜닝으로 찾은 값이 아니라 두 백본의 헤드 조건을 맞추려고 고정한 값이다.
# Validation 결과를 본 뒤 바꾸면 튜닝이 되므로 별도 실험으로 구분한다.
MLP_HIDDEN_SIZE = 256
# SmallWaferCNN 헤드와 MobileNetV3-Small 기본 헤드의 Dropout과 같은 값.
MLP_DROPOUT = 0.2


class SmallWaferCNN(nn.Module):
    """불량·유효 다이 2채널 웨이퍼맵을 분류하는 작은 CNN."""

    def __init__(self, num_classes=9):
        super().__init__()
        if type(num_classes) is not int or num_classes < 2:
            raise ValueError("num_classes must be an integer greater than one")
        self.features = nn.Sequential(
            self._block(2, 32),
            self._block(32, 64),
            self._block(64, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def _block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, images):
        return self.classifier(self.pool(self.features(images)))


class WaferMLP(nn.Module):
    """합성곱 없이 공간 배치를 펼쳐 분류하는 독립 MLP 기준 모델."""

    def __init__(self, num_classes=9):
        super().__init__()
        first, second = WAFER_MLP_HIDDEN_SIZES
        self.pool = nn.AdaptiveAvgPool2d(WAFER_MLP_POOL_SIZE)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * WAFER_MLP_POOL_SIZE ** 2, first),
            nn.ReLU(inplace=True),
            nn.Dropout(p=MLP_DROPOUT),
            nn.Linear(first, second),
            nn.ReLU(inplace=True),
            nn.Dropout(p=MLP_DROPOUT),
            nn.Linear(second, num_classes),
        )

    def forward(self, images):
        return self.classifier(self.pool(images))


def _mlp_head(in_features, num_classes):
    """백본 비교에서 공통으로 쓰는 한 은닉층 분류 헤드.

    ResNet18에서는 기본 FC보다 커지고 MobileNetV3-Small에서는 기본 헤드보다
    작아지므로, 두 백본의 결과를 같은 "MLP 추가 효과"로 해석하지 않는다.
    """
    return nn.Sequential(
        nn.Linear(in_features, MLP_HIDDEN_SIZE),
        nn.ReLU(inplace=True),
        nn.Dropout(p=MLP_DROPOUT),
        nn.Linear(MLP_HIDDEN_SIZE, num_classes),
    )


def _build_resnet18(num_classes, use_mlp):
    """사전학습 없이 2채널 입력의 ResNet18을 만든다."""
    model = resnet18(weights=None, num_classes=num_classes)

    # 입력 채널만 RGB 3채널에서 2채널로 바꾸고 나머지는 표준 구조를 유지한다.
    # bias는 바로 뒤 BatchNorm이 대신하므로 쓰지 않는다.
    # 이 층과 뒤의 maxpool로 입력이 4배 줄어 128×128의 가는 선이 약해질 수 있다.
    model.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)

    if use_mlp:
        model.fc = _mlp_head(model.fc.in_features, num_classes)

    return model


def _build_mobilenet_v3_small(num_classes, use_mlp):
    """사전학습 없이 2채널 입력의 MobileNetV3-Small을 만든다."""
    model = mobilenet_v3_small(weights=None, num_classes=num_classes)

    # 첫 합성곱의 출력 크기와 stride는 유지하고 입력 채널만 2채널로 바꾼다.
    first = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        2,
        first.out_channels,
        kernel_size=3,
        stride=2,
        padding=1,
        bias=False,
    )

    if use_mlp:
        # 기본 헤드도 2층이므로 덧붙이지 않고 공통 헤드로 통째로 교체한다.
        model.classifier = _mlp_head(model.classifier[0].in_features, num_classes)

    return model


def describe_model(model_name):
    """비교 산출물에서 모델 구조를 식별할 설명을 돌려준다."""
    require(model_name in MODEL_NAMES, 'INVALID_MODEL_NAME')

    first, second = WAFER_MLP_HIDDEN_SIZES
    size = WAFER_MLP_POOL_SIZE
    return {
        'SmallWaferCNN': '합성곱 3블록 + 평균 풀링 + 선형 헤드',
        'WaferMLP': (
            f'{size}×{size} 평균 풀링 뒤 MLP {first}·{second} '
            '(원본 해상도 입력 MLP 아님)'
        ),
        'ResNet18': '2채널 ResNet18, 기본 FC 헤드',
        'ResNet18MLP': f'2채널 ResNet18, MLP {MLP_HIDDEN_SIZE} 헤드',
        'MobileNetV3Small': '2채널 MobileNetV3-Small, 기본 헤드',
        'MobileNetV3SmallMLP': f'2채널 MobileNetV3-Small, MLP {MLP_HIDDEN_SIZE} 헤드',
    }[model_name]


def build_model(model_name, num_classes=9):
    """비교 대상 이름에 맞는 2채널 입력 분류 모델을 만든다.

    사전학습 가중치는 쓰지 않으므로 외부 다운로드가 일어나지 않는다.
    """
    require(model_name in MODEL_NAMES, 'INVALID_MODEL_NAME')

    if model_name == 'SmallWaferCNN':
        return SmallWaferCNN(num_classes=num_classes)

    if model_name == 'WaferMLP':
        return WaferMLP(num_classes=num_classes)

    use_mlp =model_name.endswith('MLP')
    if model_name.startswith('ResNet18'):
        return _build_resnet18(num_classes, use_mlp)

    return _build_mobilenet_v3_small(num_classes, use_mlp)
