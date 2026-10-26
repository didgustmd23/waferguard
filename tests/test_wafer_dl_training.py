"""Synthetic tests for the WM-811K CNN and its evaluation contracts."""

from pathlib import Path
import tempfile
import unittest

import torch

from src.wafer_dl.metrics import classification_metrics
from src.wafer_dl.models import SmallWaferCNN
from src.wafer_dl.evaluate import evaluate, validate_checkpoint
from src.wafer_dl.train import (
    _accumulate_epoch_loss,
    _require_finite_model_weights,
    _require_finite_tensor,
    _save_checkpoint,
    _validate_train_validation_contract,
    class_weights,
)
from src.wafer_dl.validate_canonical import CLASS_NAMES, ValidationError


class FakeDataset:
    protocol = 'wm_split_test'
    source_split_manifest_sha256 = 'a' * 64
    transform_config = {'transform_version': 'wm_resize_v1'}


class WaferDlTrainingTest(unittest.TestCase):
    def test_test_evaluation_requires_explicit_approval(self):
        with self.assertRaisesRegex(ValidationError, 'TEST_EVALUATION_APPROVAL_REQUIRED'):
            evaluate('missing-cache', 'missing-checkpoint', 'missing-output')

    def test_cnn_returns_nine_pattern_logits(self):
        model = SmallWaferCNN()
        logits = model(torch.zeros((3, 2, 128, 128), dtype=torch.float32))
        self.assertEqual(tuple(logits.shape), (3, len(CLASS_NAMES)))

    def test_metrics_include_rare_class_recall_and_confusion_matrix(self):
        confusion = torch.zeros((9, 9), dtype=torch.int64)
        confusion[0, 0] = 2
        confusion[1, 1] = 1
        confusion[2, 0] = 1
        result = classification_metrics(confusion)
        self.assertEqual(result['per_class'][2]['recall'], 0.0)
        self.assertEqual(result['per_class'][2]['support'], 1)
        self.assertEqual(result['confusion_matrix'][2][0], 1)
        self.assertEqual(result['accuracy'], 0.75)

    def test_class_weights_are_finite_bounded_and_mean_one(self):
        weights = class_weights([100, 10, 10, 10, 10, 10, 10, 10, 1], 5.0)
        self.assertTrue(torch.isfinite(weights).all())
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        self.assertLessEqual(float(weights.max()), 5.0)
        self.assertGreater(float(weights[-1]), float(weights[0]))

    def test_class_weights_reject_missing_training_class(self):
        with self.assertRaisesRegex(ValidationError, 'INVALID_TRAIN_CLASS_COUNTS'):
            class_weights([100, 10, 10, 10, 10, 10, 10, 10, 0], 5.0)

    def test_weighted_epoch_loss_uses_class_weight_sum_and_batch_size(self):
        """서로 다른 배치 크기에서도 weighted mean의 분모를 정확히 합산한다."""
        weights = torch.tensor([1.0, 3.0])
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
        first_targets = torch.tensor([0, 1])
        second_targets = torch.tensor([1])
        first_loss = criterion(torch.tensor([[2.0, 0.0], [0.0, 1.0]]), first_targets)
        second_loss = criterion(torch.tensor([[0.0, 2.0]]), second_targets)

        total, normalizer = _accumulate_epoch_loss(
            0.0, 0.0, first_loss, first_targets, weights
        )
        total, normalizer = _accumulate_epoch_loss(
            total, normalizer, second_loss, second_targets, weights
        )
        expected = torch.nn.functional.cross_entropy(
            torch.tensor([[2.0, 0.0], [0.0, 1.0], [0.0, 2.0]]),
            torch.tensor([0, 1, 1]),
            weight=weights,
        )
        self.assertAlmostEqual(total / normalizer, float(expected), places=6)

    def test_nonfinite_tensor_and_model_weights_are_rejected(self):
        """NaN·무한대가 loss·logits·모델 가중치 검사를 통과하지 못한다."""
        for value in (float('nan'), float('inf'), float('-inf')):
            with self.subTest(value=value):
                for code in ('NONFINITE_LOGITS', 'NONFINITE_LOSS', 'NONFINITE_CLASS_WEIGHT'):
                    with self.subTest(code=code):
                        with self.assertRaisesRegex(ValidationError, code):
                            _require_finite_tensor(torch.tensor([value]), code)

        model = torch.nn.Linear(2, 2)
        with torch.no_grad():
            model.weight[0, 0] = float('inf')
        with self.assertRaisesRegex(ValidationError, 'NONFINITE_MODEL_WEIGHT'):
            _require_finite_model_weights(model)

    def test_checkpoint_persists_actual_split_manifest_hash(self):
        """checkpoint v2에 캐시가 기록한 실제 분할 manifest 해시를 보존한다."""
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_path = Path(temporary) / 'model.pt'
            config = {'selection_metric': 'macro_f1'}
            _save_checkpoint(
                checkpoint_path,
                SmallWaferCNN(),
                config,
                FakeDataset(),
                {'macro_f1': 0.5},
                1,
            )

            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

        self.assertEqual(checkpoint['checkpoint_version'], 'wm_cnn_checkpoint_v2')
        self.assertEqual(
            checkpoint['source_split_manifest_sha256'],
            FakeDataset.source_split_manifest_sha256,
        )

    def test_frozen_checkpoint_contract_selects_train_majority_baseline(self):
        """모델·분할 계약이 맞을 때 Train 최빈 클래스를 기준선으로 선택한다."""
        checkpoint = {
            'checkpoint_version': 'wm_cnn_checkpoint_v2',
            'model_name': 'SmallWaferCNN',
            'model_config': {'num_classes': 9, 'input_channels': 2},
            'class_mapping': {str(i): name for i, name in enumerate(CLASS_NAMES)},
            'split_protocol_id': FakeDataset.protocol,
            'source_split_manifest_sha256': FakeDataset.source_split_manifest_sha256,
            'transform_config': FakeDataset.transform_config,
            'selection_metric': 'macro_f1',
            'selected_epoch': 14,
            'model_state_dict': {},
        }
        majority = validate_checkpoint(
            checkpoint, FakeDataset(), [100, 10, 10, 10, 10, 10, 10, 10, 1]
        )
        self.assertEqual(majority, 0)
        checkpoint['split_protocol_id'] = 'different_split'
        with self.assertRaisesRegex(ValidationError, 'CHECKPOINT_CACHE_CONTRACT_MISMATCH'):
            validate_checkpoint(
                checkpoint, FakeDataset(), [100, 10, 10, 10, 10, 10, 10, 10, 1]
            )

    def test_frozen_checkpoint_rejects_different_split_manifest_hash(self):
        """실제 분할 manifest 해시가 다른 Test 캐시는 거부한다."""
        checkpoint = {
            'checkpoint_version': 'wm_cnn_checkpoint_v2',
            'model_name': 'SmallWaferCNN',
            'model_config': {'num_classes': 9, 'input_channels': 2},
            'class_mapping': {str(i): name for i, name in enumerate(CLASS_NAMES)},
            'split_protocol_id': FakeDataset.protocol,
            'source_split_manifest_sha256': 'b' * 64,
            'transform_config': FakeDataset.transform_config,
            'selection_metric': 'macro_f1',
            'selected_epoch': 14,
            'model_state_dict': {},
        }
        with self.assertRaisesRegex(ValidationError, 'CHECKPOINT_CACHE_CONTRACT_MISMATCH'):
            validate_checkpoint(
                checkpoint, FakeDataset(), [100, 10, 10, 10, 10, 10, 10, 10, 1]
            )

    def test_training_rejects_mixed_split_manifests(self):
        """Train과 Validation의 원본 분할 목록이 다르면 학습 계약을 거부한다."""
        validation_dataset = FakeDataset()
        validation_dataset.source_split_manifest_sha256 = 'b' * 64
        with self.assertRaisesRegex(
            ValidationError, 'TRAIN_VALIDATION_SPLIT_MANIFEST_MISMATCH'
        ):
            _validate_train_validation_contract(FakeDataset(), validation_dataset)


if __name__ == '__main__':
    unittest.main()
