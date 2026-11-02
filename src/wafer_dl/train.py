"""캐시된 Train·Validation 분할만 사용해 WM-811K 분류 모델을 학습한다."""

import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .augment import AUGMENTATIONS, D4TrainDataset
from .dataset import WaferDataset
from .metrics import classification_metrics
from .models import MODEL_NAMES, build_model, describe_model
from .validate_canonical import CLASS_NAMES, read_json, require, validation_stage


# 키가 없는 기존 설정에 적용할 값. 기본값의 계산은 기존 가중치와 정확히 같다.
OPTIONAL_TRAIN_CONFIG = {
    'augmentation': 'none',
    'class_weight_alpha': 1.0,
    'scratch_weight_factor': 1.0,
}
CLASS_WEIGHT_METHOD = 'train_inverse_frequency_power_scratch_factor_mean_one_cap_v1'

# lr_scheduler 키가 없으면 고정 학습률이다. 기존 설정의 실행 계약이 그대로 유지되도록
# 기본값을 채워 넣지 않는다.
LR_SCHEDULER_KEY = 'lr_scheduler'
LR_SCHEDULER_TYPE = 'ReduceLROnPlateau'
LR_SCHEDULER_KEYS = {
    'type', 'metric', 'mode', 'factor', 'patience', 'threshold',
    'threshold_mode', 'cooldown', 'min_lr', 'eps',
}


def _valid_weight_option(value):
    """역빈도 지수·Scratch 계수가 허용 범위 (0, 1]의 유한한 수인지 판정한다."""
    return type(value) in (int, float) and np.isfinite(value) and 0 < value <= 1


def _finite_number(value, low, high):
    """bool을 제외한 유한한 수가 닫힌 구간 안에 있는지 판정한다."""
    return type(value) in (int, float) and np.isfinite(value) and low <= value <= high


def _validate_lr_scheduler(config):
    """Validation 지표가 정체될 때 줄이는 고정 형태의 스케줄 설정만 허용한다."""
    settings = config[LR_SCHEDULER_KEY]
    require(type(settings) is dict and set(settings) == LR_SCHEDULER_KEYS,
            'INVALID_LR_SCHEDULER')
    require(settings['type'] == LR_SCHEDULER_TYPE and
            settings['metric'] == config['selection_metric'] and
            settings['mode'] == 'max' and
            settings['threshold_mode'] == 'abs',
            'INVALID_LR_SCHEDULER')
    require(_finite_number(settings['factor'], 0.0, 1.0) and
            0 < settings['factor'] < 1 and
            _finite_number(settings['threshold'], 0.0, 1.0) and
            _finite_number(settings['min_lr'], 0.0, config['learning_rate']) and
            _finite_number(settings['eps'], 0.0, 1.0),
            'INVALID_LR_SCHEDULER')
    for key in ('patience', 'cooldown'):
        require(type(settings[key]) is int and 0 <= settings[key] <= 100,
                'INVALID_LR_SCHEDULER')


def load_train_config(path):
    config = read_json(Path(path))
    keys = {
        'training_version', 'batch_size', 'epochs', 'learning_rate',
        'weight_decay', 'patience', 'seed', 'class_weighted',
        'max_class_weight', 'selection_metric',
    }
    # 선택 키가 없는 기존 설정은 기본값으로 읽어 하위 호환을 유지한다.
    require(set(config) - set(OPTIONAL_TRAIN_CONFIG) - {LR_SCHEDULER_KEY} == keys and
            config['training_version'] == 'wm_cnn_train_v1',
            'INVALID_TRAIN_CONFIG')
    config = {**OPTIONAL_TRAIN_CONFIG, **config}
    require(config['augmentation'] in AUGMENTATIONS, 'INVALID_TRAIN_CONFIG')
    require(_valid_weight_option(config['class_weight_alpha']) and
            _valid_weight_option(config['scratch_weight_factor']),
            'INVALID_TRAIN_CONFIG')
    for key, low, high in (
        ('batch_size', 1, 1024), ('epochs', 1, 500),
        ('patience', 1, 100), ('seed', 0, 2**32 - 1),
    ):
        require(type(config[key]) is int and low <= config[key] <= high,
                'INVALID_TRAIN_CONFIG')
    for key, low, high in (
        ('learning_rate', 1e-7, 1.0), ('weight_decay', 0.0, 1.0),
        ('max_class_weight', 1.0, 100.0),
    ):
        require(type(config[key]) in (int, float) and
                np.isfinite(config[key]) and low <= config[key] <= high,
                'INVALID_TRAIN_CONFIG')
    require(type(config['class_weighted']) is bool and
            config['selection_metric'] in ('macro_f1', 'macro_recall'),
            'INVALID_TRAIN_CONFIG')
    if LR_SCHEDULER_KEY in config:
        _validate_lr_scheduler(config)
    return config


def build_lr_scheduler(optimizer, config):
    """설정이 있을 때만 스케줄러를 만든다. 없으면 객체도 난수 소비도 없다."""
    settings = config.get(LR_SCHEDULER_KEY)
    if settings is None:
        return None
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=settings['mode'],
        factor=settings['factor'],
        patience=settings['patience'],
        threshold=settings['threshold'],
        threshold_mode=settings['threshold_mode'],
        cooldown=settings['cooldown'],
        min_lr=settings['min_lr'],
        eps=settings['eps'],
    )


def _require_finite_tensor(tensor, code):
    """Tensor가 NaN·무한대를 포함하지 않는지 지정한 실패 코드로 검사한다."""
    require(bool(torch.isfinite(tensor).all().item()), code)


def class_weights(class_counts, max_weight, alpha=1.0, scratch_factor=1.0):
    """Train 클래스 빈도에서 평균 1·상한 이하의 유한 가중치를 계산한다.

    역빈도에 지수를 적용하고 Scratch에만 계수를 곱한 뒤 공통 배율을 찾는다.
    공통 배율은 가중 평균 손실에서 상쇄되므로 클래스 사이 비율만 의미가 있다.
    """
    require(type(class_counts) in (list, tuple) and len(class_counts) == len(CLASS_NAMES) and
            all(type(count) is int and count > 0 for count in class_counts),
            'INVALID_TRAIN_CLASS_COUNTS')
    require(type(max_weight) in (int, float) and np.isfinite(max_weight) and
            1 <= max_weight <= 100, 'INVALID_MAX_CLASS_WEIGHT')
    require(_valid_weight_option(alpha), 'INVALID_CLASS_WEIGHT_ALPHA')
    require(_valid_weight_option(scratch_factor), 'INVALID_SCRATCH_WEIGHT_FACTOR')
    counts = torch.tensor(class_counts, dtype=torch.float64)
    raw_weights = (counts.sum() / (len(class_counts) * counts)).pow(float(alpha))
    raw_weights[CLASS_NAMES.index('Scratch')] *= float(scratch_factor)
    low, high = 0.0, float(max_weight) / float(raw_weights.min())
    for _ in range(64):
        scale = (low + high) / 2
        if float(raw_weights.mul(scale).clamp(max=float(max_weight)).mean()) < 1:
            low = scale
        else:
            high = scale
    weights = raw_weights.mul(high).clamp(max=float(max_weight))
    weights = weights.to(dtype=torch.float32)
    _require_finite_tensor(weights, 'NONFINITE_CLASS_WEIGHT')
    return weights


def class_weight_record(weights, config):
    """실제 적용한 클래스 가중치와 계산 설정을 기록용으로 정리한다."""
    return {
        'method': CLASS_WEIGHT_METHOD,
        'class_weighted': config['class_weighted'],
        'max_class_weight': config['max_class_weight'],
        'class_weight_alpha': config['class_weight_alpha'],
        'scratch_weight_factor': config['scratch_weight_factor'],
        'weights': (
            None if not config['class_weighted']
            else dict(zip(CLASS_NAMES, weights.tolist()))
        ),
    }


def _accumulate_epoch_loss(loss_total, normalizer_total, loss, targets, weights):
    """배치 loss를 표본 수 또는 정답 클래스 가중치 합으로 환산해 누적한다."""
    denominator = (
        float(targets.numel())
        if weights is None else float(weights[targets].sum().item())
    )
    return (
        loss_total + float(loss.detach().item()) * denominator,
        normalizer_total + denominator,
    )


def _require_finite_model_weights(model):
    """모델 파라미터·버퍼에 NaN 또는 무한대가 없는지 확인한다."""
    for value in model.state_dict().values():
        _require_finite_tensor(value, 'NONFINITE_MODEL_WEIGHT')


def _validate_train_validation_contract(train_dataset, validation_dataset):
    """Train·Validation 캐시가 동일 프로토콜과 실제 분할 목록인지 확인한다."""
    require(
        train_dataset.protocol == validation_dataset.protocol,
        'TRAIN_VALIDATION_PROTOCOL_MISMATCH',
    )
    require(
        train_dataset.source_split_manifest_sha256 ==
        validation_dataset.source_split_manifest_sha256,
        'TRAIN_VALIDATION_SPLIT_MANIFEST_MISMATCH',
    )


def _evaluate(model, loader, device):
    """Validation logits를 검사하고 분류 지표를 계산한다."""
    matrix = torch.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.int64)
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            images = batch['image'].to(device, non_blocking=device.type == 'cuda')
            targets = batch['target'].to(device, non_blocking=device.type == 'cuda')
            logits = model(images)
            _require_finite_tensor(logits, 'NONFINITE_LOGITS')
            predictions = logits.argmax(dim=1)
            encoded = targets * len(CLASS_NAMES) + predictions
            matrix += torch.bincount(
                encoded.to(device='cpu'), minlength=len(CLASS_NAMES) ** 2
            ).reshape(len(CLASS_NAMES), len(CLASS_NAMES))
    return classification_metrics(matrix)


def _save_checkpoint(
    path,
    model,
    config,
    train_dataset,
    metrics,
    epoch,
    model_name='SmallWaferCNN',
    weight_record=None,
    lr_schedule=None,
):
    """실제 분할 manifest 지문을 포함한 동결 checkpoint를 저장한다."""
    _require_finite_model_weights(model)
    temporary = path.with_name(path.name + '.tmp')
    model_state = {
        name: value.detach().to(device='cpu')
        for name, value in model.state_dict().items()
    }
    torch.save({
        'checkpoint_version': 'wm_cnn_checkpoint_v2',
        'model_name': model_name,
        'model_config': {'num_classes': len(CLASS_NAMES), 'input_channels': 2},
        'model_state_dict': model_state,
        'class_mapping': {str(i): name for i, name in enumerate(CLASS_NAMES)},
        'transform_config': train_dataset.transform_config,
        'split_protocol_id': train_dataset.protocol,
        'source_split_manifest_sha256': train_dataset.source_split_manifest_sha256,
        'training_config': config,
        'augmentation': config.get('augmentation', 'none'),
        'class_weights': weight_record,
        'lr_schedule': lr_schedule,
        'selected_epoch': epoch,
        'selection_metric': config['selection_metric'],
        'validation_metrics': metrics,
    }, temporary)
    os.replace(temporary, path)


def train(
    cache_dir,
    output_dir,
    config_path,
    device='auto',
    max_epochs=None,
    model_name='SmallWaferCNN',
    seed=None,
    augmentation=None,
):
    """Train으로 모델을 학습하고 Validation 선택 결과와 checkpoint를 저장한다."""
    config = load_train_config(config_path)

    if max_epochs is not None:
        require(type(max_epochs) is int and 1 <= max_epochs <= 500, 'INVALID_EPOCH_OVERRIDE')
        config = {**config, 'epochs': max_epochs}

    # 설정 파일은 그대로 두고 이번 실행에만 seed를 바꾼다. 실제 값은 checkpoint에 남는다.
    if seed is not None:
        require(type(seed) is int and 0 <= seed <= 2**32 - 1, 'INVALID_SEED_OVERRIDE')
        config = {**config, 'seed': seed}

    if augmentation is not None:
        require(augmentation in AUGMENTATIONS, 'INVALID_AUGMENTATION')
        config = {**config, 'augmentation': augmentation}

    require(model_name in MODEL_NAMES, 'INVALID_MODEL_NAME')

    output = Path(output_dir).resolve()
    require(not output.exists(), 'TRAIN_OUTPUT_ALREADY_EXISTS')
    require(device in ('auto', 'cpu', 'cuda'), 'INVALID_TRAIN_DEVICE')
    if device == 'cuda' and not torch.cuda.is_available():
        require(False, 'CUDA_DEVICE_UNAVAILABLE')
    selected_device = torch.device(
        'cuda' if device == 'cuda' or (device == 'auto' and torch.cuda.is_available()) else 'cpu'
    )

    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if selected_device.type == 'cuda':
        torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    generator = torch.Generator().manual_seed(config['seed'])

    train_dataset = WaferDataset(cache_dir, split='train')
    validation_dataset = None
    try:
        validation_dataset = WaferDataset(cache_dir, split='validation')
        _validate_train_validation_contract(train_dataset, validation_dataset)
        # 증강은 Train 로더에만 적용하고 Validation은 원본 그대로 평가한다.
        augmented_dataset = (
            D4TrainDataset(train_dataset, config['seed'])
            if config['augmentation'] == 'd4' else None
        )
        train_loader = DataLoader(
            train_dataset if augmented_dataset is None else augmented_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=selected_device.type == 'cuda',
            generator=generator,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=selected_device.type == 'cuda',
        )
        weights = class_weights(
            train_dataset.class_counts,
            config['max_class_weight'],
            config['class_weight_alpha'],
            config['scratch_weight_factor'],
        )
        weight_record = class_weight_record(weights, config)
        if not config['class_weighted']:
            weights = None
        elif weights is not None:
            weights = weights.to(selected_device)
            _require_finite_tensor(weights, 'NONFINITE_CLASS_WEIGHT')
        model = build_model(model_name, num_classes=len(CLASS_NAMES)).to(selected_device)
        _require_finite_model_weights(model)
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config['learning_rate']),
            weight_decay=float(config['weight_decay']),
        )
        scheduler = build_lr_scheduler(optimizer, config)
        output.mkdir(parents=True, exist_ok=False)
        # 비교 지표용으로 학습 구간의 시간과 peak CUDA 메모리만 측정한다.
        if selected_device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(selected_device)
        started_at = time.perf_counter()

        best_score = -1.0
        best_epoch = 0
        epochs_without_improvement = 0
        history = []
        lr_reductions = []
        for epoch in range(1, config['epochs'] + 1):
            if augmented_dataset is not None:
                augmented_dataset.set_epoch(epoch)
            learning_rate = optimizer.param_groups[0]['lr']
            model.train()
            loss_total = 0.0
            loss_normalizer = 0.0
            for batch in train_loader:
                images = batch['image'].to(selected_device, non_blocking=selected_device.type == 'cuda')
                targets = batch['target'].to(selected_device, non_blocking=selected_device.type == 'cuda')
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                _require_finite_tensor(logits, 'NONFINITE_LOGITS')
                loss = criterion(logits, targets)
                _require_finite_tensor(loss, 'NONFINITE_LOSS')
                loss.backward()
                optimizer.step()
                _require_finite_model_weights(model)
                loss_total, loss_normalizer = _accumulate_epoch_loss(
                    loss_total, loss_normalizer, loss, targets, weights
                )

            metrics = _evaluate(model, validation_loader, selected_device)
            score = metrics[config['selection_metric']]

            # 평가가 끝난 뒤 호출하므로 감소한 학습률은 다음 epoch부터 적용된다.
            if scheduler is not None:
                scheduler.step(score)
            next_learning_rate = optimizer.param_groups[0]['lr']
            if next_learning_rate < learning_rate:
                lr_reductions.append({
                    'epoch': epoch,
                    'learning_rate': learning_rate,
                    'next_learning_rate': next_learning_rate,
                })

            epoch_result = {
                'epoch': epoch,
                'train_loss': loss_total / loss_normalizer,
                'learning_rate': learning_rate,
                'next_learning_rate': next_learning_rate,
                'validation': metrics,
            }
            history.append(epoch_result)
            print(json.dumps(epoch_result, ensure_ascii=False))
            if score > best_score:
                best_score = score
                best_epoch = epoch
                epochs_without_improvement = 0
                _save_checkpoint(
                    output / 'best_model.pt',
                    model,
                    config,
                    train_dataset,
                    metrics,
                    epoch,
                    model_name,
                    weight_record,
                    None if scheduler is None else {
                        'settings': config[LR_SCHEDULER_KEY],
                        'learning_rate': learning_rate,
                        'next_learning_rate': next_learning_rate,
                        'reductions': list(lr_reductions),
                        'scheduler_state': scheduler.state_dict(),
                    },
                )
            else:
                # 학습률이 줄어도 조기 종료 카운터는 최고 점수 갱신 때만 초기화한다.
                epochs_without_improvement += 1
                if epochs_without_improvement >= config['patience']:
                    break

        report = {
            'status': 'training_complete',
            'model_name': model_name,
            'model_description': describe_model(model_name),
            'parameter_count': sum(value.numel() for value in model.parameters()),
            'seed': config['seed'],
            'augmentation': config['augmentation'],
            'device': str(selected_device),
            'training_seconds': time.perf_counter() - started_at,
            'peak_cuda_memory_bytes': (
                torch.cuda.max_memory_allocated(selected_device)
                if selected_device.type == 'cuda' else None
            ),
            'max_epochs': config['epochs'],
            'split_protocol_id': train_dataset.protocol,
            'train_rows': len(train_dataset),
            'validation_rows': len(validation_dataset),
            'class_weighted': config['class_weighted'],
            'class_weights': weight_record,
            'lr_scheduler': config.get(LR_SCHEDULER_KEY),
            'lr_reductions': lr_reductions,
            'selection_metric': config['selection_metric'],
            'selected_epoch': best_epoch,
            'best_validation_score': best_score,
            'test_split_used': False,
            'history': history,
        }
        (output / 'training_report.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
            encoding='utf-8',
        )
        return report
    finally:
        train_dataset.close()
        if validation_dataset is not None:
            validation_dataset.close()


def main():
    parser = argparse.ArgumentParser(description='WM-811K 9종 패턴 모델 학습 (Train/Validation only)')
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--output-dir', default='models/wm811k/cnn_v1')
    parser.add_argument('--config', default='configs/wm811k/train.json')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--epochs', type=int, help='설정 파일의 epoch 상한을 이번 실행에만 덮어씀')
    parser.add_argument('--model', choices=MODEL_NAMES, default='SmallWaferCNN')
    parser.add_argument('--seed', type=int, help='설정 파일의 seed를 이번 실행에만 덮어씀')
    parser.add_argument('--augmentation', choices=AUGMENTATIONS,
                        help='설정 파일의 Train 증강을 이번 실행에만 덮어씀')
    args = parser.parse_args()
    try:
        with validation_stage('training'):
            result = train(
                args.cache_dir,
                args.output_dir,
                args.config,
                args.device,
                args.epochs,
                args.model,
                args.seed,
                args.augmentation,
            )
    except ValueError as error:
        print('WM 학습 실패 코드: ' + str(error))
        return 1
    print(json.dumps({
        'status': result['status'],
        'model_name': result['model_name'],
        'device': result['device'],
        'selected_epoch': result['selected_epoch'],
        'best_validation_score': result['best_validation_score'],
        'selection_metric': result['selection_metric'],
        'test_split_used': result['test_split_used'],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
