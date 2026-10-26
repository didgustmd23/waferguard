"""Train the WM-811K CNN using only the cached Train and Validation splits."""

import argparse
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import WaferDataset
from .metrics import classification_metrics
from .models import SmallWaferCNN
from .validate_canonical import CLASS_NAMES, read_json, require, validation_stage


def load_train_config(path):
    config = read_json(Path(path))
    keys = {
        'training_version', 'batch_size', 'epochs', 'learning_rate',
        'weight_decay', 'patience', 'seed', 'class_weighted',
        'max_class_weight', 'selection_metric',
    }
    require(set(config) == keys and config['training_version'] == 'wm_cnn_train_v1',
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
    return config


def class_weights(class_counts, max_weight):
    require(type(class_counts) in (list, tuple) and len(class_counts) == len(CLASS_NAMES) and
            all(type(count) is int and count > 0 for count in class_counts),
            'INVALID_TRAIN_CLASS_COUNTS')
    require(type(max_weight) in (int, float) and np.isfinite(max_weight) and
            1 <= max_weight <= 100, 'INVALID_MAX_CLASS_WEIGHT')
    counts = torch.tensor(class_counts, dtype=torch.float64)
    raw_weights = counts.sum() / (len(class_counts) * counts)
    low, high = 0.0, float(max_weight) / float(raw_weights.min())
    for _ in range(64):
        scale = (low + high) / 2
        if float(raw_weights.mul(scale).clamp(max=float(max_weight)).mean()) < 1:
            low = scale
        else:
            high = scale
    weights = raw_weights.mul(high).clamp(max=float(max_weight))
    return weights.to(dtype=torch.float32)


def _evaluate(model, loader, device):
    matrix = torch.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.int64)
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            images = batch['image'].to(device, non_blocking=device.type == 'cuda')
            targets = batch['target'].to(device, non_blocking=device.type == 'cuda')
            predictions = model(images).argmax(dim=1)
            encoded = targets * len(CLASS_NAMES) + predictions
            matrix += torch.bincount(
                encoded.to(device='cpu'), minlength=len(CLASS_NAMES) ** 2
            ).reshape(len(CLASS_NAMES), len(CLASS_NAMES))
    return classification_metrics(matrix)


def _save_checkpoint(path, model, config, train_dataset, metrics, epoch):
    temporary = path.with_name(path.name + '.tmp')
    model_state = {
        name: value.detach().to(device='cpu')
        for name, value in model.state_dict().items()
    }
    torch.save({
        'checkpoint_version': 'wm_cnn_checkpoint_v1',
        'model_name': 'SmallWaferCNN',
        'model_config': {'num_classes': len(CLASS_NAMES), 'input_channels': 2},
        'model_state_dict': model_state,
        'class_mapping': {str(i): name for i, name in enumerate(CLASS_NAMES)},
        'transform_config': train_dataset.transform_config,
        'split_protocol_id': train_dataset.protocol,
        'training_config': config,
        'selected_epoch': epoch,
        'selection_metric': config['selection_metric'],
        'validation_metrics': metrics,
    }, temporary)
    os.replace(temporary, path)


def train(cache_dir, output_dir, config_path, device='auto', max_epochs=None):
    config = load_train_config(config_path)
    if max_epochs is not None:
        require(type(max_epochs) is int and 1 <= max_epochs <= 500, 'INVALID_EPOCH_OVERRIDE')
        config = {**config, 'epochs': max_epochs}
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
        require(train_dataset.protocol == validation_dataset.protocol,
                'TRAIN_VALIDATION_PROTOCOL_MISMATCH')
        train_loader = DataLoader(
            train_dataset,
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
        weights = class_weights(train_dataset.class_counts, config['max_class_weight'])
        if not config['class_weighted']:
            weights = None
        model = SmallWaferCNN(num_classes=len(CLASS_NAMES)).to(selected_device)
        criterion = nn.CrossEntropyLoss(
            weight=weights.to(selected_device) if weights is not None else None
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config['learning_rate']),
            weight_decay=float(config['weight_decay']),
        )
        output.mkdir(parents=True, exist_ok=False)

        best_score = -1.0
        best_epoch = 0
        epochs_without_improvement = 0
        history = []
        for epoch in range(1, config['epochs'] + 1):
            model.train()
            loss_total = 0.0
            samples = 0
            for batch in train_loader:
                images = batch['image'].to(selected_device, non_blocking=selected_device.type == 'cuda')
                targets = batch['target'].to(selected_device, non_blocking=selected_device.type == 'cuda')
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(images), targets)
                loss.backward()
                optimizer.step()
                batch_size = targets.size(0)
                loss_total += float(loss.detach().item()) * batch_size
                samples += batch_size

            metrics = _evaluate(model, validation_loader, selected_device)
            score = metrics[config['selection_metric']]
            epoch_result = {
                'epoch': epoch,
                'train_loss': loss_total / samples,
                'validation': metrics,
            }
            history.append(epoch_result)
            print(json.dumps(epoch_result, ensure_ascii=False))
            if score > best_score:
                best_score = score
                best_epoch = epoch
                epochs_without_improvement = 0
                _save_checkpoint(
                    output / 'best_model.pt', model, config, train_dataset, metrics, epoch
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= config['patience']:
                    break

        report = {
            'status': 'training_complete',
            'model_name': 'SmallWaferCNN',
            'device': str(selected_device),
            'split_protocol_id': train_dataset.protocol,
            'train_rows': len(train_dataset),
            'validation_rows': len(validation_dataset),
            'class_weighted': config['class_weighted'],
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
    parser = argparse.ArgumentParser(description='WM-811K 9종 패턴 CNN 학습 (Train/Validation only)')
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--output-dir', default='models/wm811k/cnn_v1')
    parser.add_argument('--config', default='configs/wm811k/train.json')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--epochs', type=int, help='설정 파일의 epoch 상한을 이번 실행에만 덮어씀')
    args = parser.parse_args()
    try:
        with validation_stage('training'):
            result = train(args.cache_dir, args.output_dir, args.config, args.device, args.epochs)
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
