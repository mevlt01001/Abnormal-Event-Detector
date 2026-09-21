"""Training modules and evaluators for video anomaly detection."""

from training.trainer import (
    ClassAwareMILTrainer,
    compute_average_precision,
    compute_roc_auc,
)

__all__ = [
    "ClassAwareMILTrainer",
    "compute_average_precision",
    "compute_roc_auc",
]
