"""Synthetic tests for the WM-811K CNN and its evaluation contracts."""

import unittest

import torch

from src.wafer_dl.metrics import classification_metrics
from src.wafer_dl.models import SmallWaferCNN
from src.wafer_dl.evaluate import evaluate, validate_checkpoint
from src.wafer_dl.train import class_weights
from src.wafer_dl.validate_canonical import CLASS_NAMES, ValidationError


class FakeDataset:
    protocol = 'wm_split_test'
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

    def test_frozen_checkpoint_contract_selects_train_majority_baseline(self):
        checkpoint = {
            'checkpoint_version': 'wm_cnn_checkpoint_v1',
            'model_name': 'SmallWaferCNN',
            'model_config': {'num_classes': 9, 'input_channels': 2},
            'class_mapping': {str(i): name for i, name in enumerate(CLASS_NAMES)},
            'split_protocol_id': FakeDataset.protocol,
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


if __name__ == '__main__':
    unittest.main()
