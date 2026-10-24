"""One-time evaluation of a frozen WM-811K model on the approved Test split."""

import argparse
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from .dataset import WaferDataset
from .metrics import classification_metrics
from .models import SmallWaferCNN
from .validate_canonical import CLASS_NAMES, require, validation_stage


def validate_checkpoint(checkpoint, dataset, train_class_counts):
    require(type(checkpoint) is dict and
            checkpoint.get('checkpoint_version') == 'wm_cnn_checkpoint_v1' and
            checkpoint.get('model_name') == 'SmallWaferCNN' and
            checkpoint.get('model_config') == {
                'num_classes': len(CLASS_NAMES), 'input_channels': 2,
            } and checkpoint.get('class_mapping') ==
            {str(index): name for index, name in enumerate(CLASS_NAMES)} and
            checkpoint.get('split_protocol_id') == dataset.protocol and
            checkpoint.get('transform_config') == dataset.transform_config and
            checkpoint.get('selection_metric') in ('macro_f1', 'macro_recall') and
            type(checkpoint.get('selected_epoch')) is int and
            checkpoint['selected_epoch'] >= 1 and
            type(checkpoint.get('model_state_dict')) is dict,
            'CHECKPOINT_CACHE_CONTRACT_MISMATCH')
    require(type(train_class_counts) is list and len(train_class_counts) == len(CLASS_NAMES) and
            all(type(count) is int and count > 0 for count in train_class_counts),
            'INVALID_TRAIN_CLASS_COUNTS')
    return int(max(range(len(train_class_counts)), key=train_class_counts.__getitem__))


def evaluate(cache_dir, checkpoint_path, output_dir, device='auto', approved=False):
    require(approved is True, 'TEST_EVALUATION_APPROVAL_REQUIRED')
    require(device in ('auto', 'cpu', 'cuda'), 'INVALID_EVALUATION_DEVICE')
    if device == 'cuda' and not torch.cuda.is_available():
        require(False, 'CUDA_DEVICE_UNAVAILABLE')
    selected_device = torch.device(
        'cuda' if device == 'cuda' or (device == 'auto' and torch.cuda.is_available()) else 'cpu'
    )
    checkpoint_file = Path(checkpoint_path).resolve(strict=True)
    output = Path(output_dir).resolve()
    require(checkpoint_file.is_file() and not output.exists(), 'INVALID_EVALUATION_OUTPUT')
    working_root = Path.cwd().resolve()
    checkpoint_reference = (
        checkpoint_file.relative_to(working_root).as_posix()
        if checkpoint_file.is_relative_to(working_root) else checkpoint_file.name
    )

    checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=True)
    cache_root = Path(cache_dir).resolve(strict=True)
    cache_report = json.loads((cache_root / 'cache.json').read_text(encoding='utf-8'))
    require(cache_report.get('status') == 'cache_complete' and
            'test' in cache_report.get('splits', {}) and
            cache_report.get('test_maps_read') is True,
            'TEST_CACHE_NOT_PREPARED')
    train_counts = cache_report.get('train_class_counts')

    dataset = WaferDataset(cache_root, split='test', allow_test=True)
    try:
        majority_class = validate_checkpoint(checkpoint, dataset, train_counts)
        model = SmallWaferCNN(num_classes=len(CLASS_NAMES)).to(selected_device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        model.eval()
        loader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            num_workers=0,
            pin_memory=selected_device.type == 'cuda',
        )
        model_confusion = torch.zeros(
            (len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.int64
        )
        baseline_confusion = torch.zeros_like(model_confusion)
        if selected_device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(selected_device)
            torch.cuda.synchronize(selected_device)
        started = time.perf_counter()
        with torch.inference_mode():
            for batch in loader:
                images = batch['image'].to(
                    selected_device, non_blocking=selected_device.type == 'cuda'
                )
                targets = batch['target'].to(
                    selected_device, non_blocking=selected_device.type == 'cuda'
                )
                predictions = model(images).argmax(dim=1)
                model_encoded = (targets * len(CLASS_NAMES) + predictions).cpu()
                baseline_encoded = (
                    targets * len(CLASS_NAMES) + majority_class
                ).cpu()
                model_confusion += torch.bincount(
                    model_encoded, minlength=len(CLASS_NAMES) ** 2
                ).reshape(len(CLASS_NAMES), len(CLASS_NAMES))
                baseline_confusion += torch.bincount(
                    baseline_encoded, minlength=len(CLASS_NAMES) ** 2
                ).reshape(len(CLASS_NAMES), len(CLASS_NAMES))
        if selected_device.type == 'cuda':
            torch.cuda.synchronize(selected_device)
        elapsed = time.perf_counter() - started
        model_metrics = classification_metrics(model_confusion)
        baseline_metrics = classification_metrics(baseline_confusion)
        peak_memory = (
            int(torch.cuda.max_memory_allocated(selected_device))
            if selected_device.type == 'cuda' else None
        )
        report = {
            'status': 'test_evaluation_complete',
            'evaluation_version': 'wm_frozen_test_v1',
            'checkpoint_version': checkpoint['checkpoint_version'],
            'model_name': checkpoint['model_name'],
            'checkpoint_path': checkpoint_reference,
            'selected_epoch': checkpoint['selected_epoch'],
            'selection_metric': checkpoint['selection_metric'],
            'split_protocol_id': dataset.protocol,
            'test_rows': len(dataset),
            'test_split_used_once_for_this_report': True,
            'test_used_for_model_selection': False,
            'device': str(selected_device),
            'inference_seconds': elapsed,
            'inference_ms_per_sample': elapsed * 1000 / len(dataset),
            'peak_cuda_memory_bytes': peak_memory,
            'majority_baseline_class_index': majority_class,
            'majority_baseline_class_name': CLASS_NAMES[majority_class],
            'majority_baseline': baseline_metrics,
            'frozen_model': model_metrics,
            'model_minus_baseline': {
                metric: model_metrics[metric] - baseline_metrics[metric]
                for metric in ('accuracy', 'macro_recall', 'macro_f1')
            },
        }
        output.mkdir(parents=True, exist_ok=False)
        (output / 'test_evaluation.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
            encoding='utf-8',
        )
        (output / 'SUCCESS.json').write_text(json.dumps({
            'status': report['status'],
            'split_protocol_id': dataset.protocol,
            'selected_epoch': checkpoint['selected_epoch'],
            'test_used_for_model_selection': False,
        }), encoding='utf-8')
        return report
    finally:
        dataset.close()


def main():
    parser = argparse.ArgumentParser(
        description='동결된 WM-811K CNN의 명시적 단일 Test 평가'
    )
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--confirm-test-evaluation', action='store_true')
    args = parser.parse_args()
    try:
        with validation_stage('test_evaluation'):
            report = evaluate(
                args.cache_dir,
                args.checkpoint,
                args.output_dir,
                args.device,
                args.confirm_test_evaluation,
            )
    except (ValueError, OSError, RuntimeError) as error:
        print('Test 평가 실패 코드: ' + str(error))
        return 1
    print(json.dumps({
        'status': report['status'],
        'test_rows': report['test_rows'],
        'device': report['device'],
        'majority_baseline': {
            metric: report['majority_baseline'][metric]
            for metric in ('accuracy', 'macro_recall', 'macro_f1')
        },
        'frozen_model': {
            metric: report['frozen_model'][metric]
            for metric in ('accuracy', 'macro_recall', 'macro_f1')
        },
        'model_minus_baseline': report['model_minus_baseline'],
        'test_used_for_model_selection': report['test_used_for_model_selection'],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
