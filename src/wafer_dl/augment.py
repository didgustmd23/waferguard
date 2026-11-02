"""Train 전용 WM-811K D4 증강.

캐시와 저장된 맵은 바꾸지 않고, 학습 Dataset에서 읽을 때만 변환을 적용한다.
D4는 90도 회전과 반전의 조합 8가지다. 보간이 없어 채널 값은 0/1로 유지되고,
두 채널에 같은 변환을 적용하므로 defect<=valid 관계도 유지된다.

이 증강은 패턴 라벨이 웨이퍼 방향과 무관하다고 가정한다. notch나 장비 방향에
의미가 있는 라벨이라면 이 가정이 맞지 않으므로 사용 전에 데이터 담당자가 확인한다.
"""

import hashlib

import torch
from torch.utils.data import Dataset

from .validate_canonical import require


AUGMENTATIONS = ('none', 'd4')
D4_TRANSFORM_COUNT = 8


def apply_d4(image, transform_index):
    """[채널, 높이, 너비] 이미지에 D4 변환 하나를 적용한다."""
    require(
        type(transform_index) is int and 0 <= transform_index < D4_TRANSFORM_COUNT,
        'INVALID_D4_TRANSFORM',
    )
    require(
        image.ndim == 3 and image.shape[1] == image.shape[2],
        'D4_REQUIRES_SQUARE_IMAGE',
    )

    result = torch.rot90(image, transform_index % 4, dims=(1, 2))
    if transform_index >= 4:
        result = torch.flip(result, dims=(2,))

    return result


def d4_transform_index(seed, epoch, sample_id):
    """seed·epoch·표본 ID로 변환을 정해 worker 수와 무관하게 재현한다."""
    digest = hashlib.sha256(f'{seed}:{epoch}:{sample_id}'.encode('utf-8')).digest()
    return digest[0] % D4_TRANSFORM_COUNT


class D4TrainDataset(Dataset):
    """Train Dataset을 감싸 표본마다 D4 변환 하나를 적용한다."""

    def __init__(self, dataset, seed):
        require(getattr(dataset, 'split', None) == 'train', 'AUGMENTATION_TRAIN_ONLY')
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        """epoch마다 다른 변환이 선택되도록 현재 epoch를 기록한다."""
        self.epoch = epoch

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        transform_index = d4_transform_index(self.seed, self.epoch, sample['sample_id'])
        return {**sample, 'image': apply_d4(sample['image'], transform_index)}
