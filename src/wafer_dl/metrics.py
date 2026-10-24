"""Metrics for the fixed nine-class WM-811K label mapping."""

import torch

from .validate_canonical import CLASS_NAMES, require


def classification_metrics(confusion):
    """Return accuracy, macro scores, per-class scores, and the confusion matrix."""
    require(isinstance(confusion, torch.Tensor) and
            confusion.shape == (len(CLASS_NAMES), len(CLASS_NAMES)) and
            not confusion.is_floating_point() and not confusion.is_complex() and
            bool((confusion >= 0).all()), 'INVALID_CONFUSION_MATRIX')
    matrix = confusion.to(dtype=torch.float64, device='cpu')
    support = matrix.sum(dim=1)
    predicted = matrix.sum(dim=0)
    true_positive = matrix.diagonal()
    precision = torch.where(predicted > 0, true_positive / predicted, 0.0)
    recall = torch.where(support > 0, true_positive / support, 0.0)
    f1_denominator = precision + recall
    f1 = torch.where(f1_denominator > 0, 2 * precision * recall / f1_denominator, 0.0)
    total = int(matrix.sum().item())
    require(total > 0, 'EMPTY_CONFUSION_MATRIX')
    return {
        'accuracy': float(true_positive.sum().item() / total),
        'macro_precision': float(precision.mean().item()),
        'macro_recall': float(recall.mean().item()),
        'macro_f1': float(f1.mean().item()),
        'per_class': [
            {
                'class_index': index,
                'pattern_label': name,
                'support': int(support[index].item()),
                'precision': float(precision[index].item()),
                'recall': float(recall[index].item()),
                'f1': float(f1[index].item()),
            }
            for index, name in enumerate(CLASS_NAMES)
        ],
        'confusion_matrix': confusion.to(dtype=torch.int64, device='cpu').tolist(),
    }
