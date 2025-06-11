"""High level orchestrator for model training

!!! warning
    This module is incomplete. they only contain annotations for future use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, TypeAlias

if TYPE_CHECKING:
    from annotated_types import Gt

Epoch: TypeAlias = int
"""The number of times the model has seen the entire training dataset."""

TrainingBatchSize: TypeAlias = Annotated[int, Gt(0)]
"""Number of training examples (audio chunks) processed before a weight update.

Larger batches may offer more stable gradients but require more memory.
"""

GradientAccumulationSteps: TypeAlias = Annotated[int, Gt(0)]
"""Number of batches to process before performing a weight update.

This simulates a larger batch size without increasing memory, e.g., a
[batch size][splifft.training.TrainingBatchSize] of 4 with 8 accumulation steps has an effective
batch size of 32.
"""

OptimizerName: TypeAlias = str
"""Algorithm used to update the model weights to minimize the loss function."""

LearningRateSchedulerName: TypeAlias = str
"""Algorithm used to adjust the learning rate during training.

e.g. `ReduceLROnPlateau` can reduce the learning rate when a metric stops improving.
"""

# TODO: patience

# TODO: loss function
# TODO: gradient clipping

UseAutomaticMixedPrecision: TypeAlias = bool
"""Whether to use [automatic mixed precision (AMP)][torch.amp] during training."""

UseLoRA: TypeAlias = bool
"""Whether to use [Low-Rank Adaptation](https://arxiv.org/abs/2106.09685) for efficient fine-tuning.

This freezes pre-trained weights and injects smaller, trainable low-rank matrices, dramatically
reducing the number of trainable parameters.
"""

#
# data augmentation (move to new module?)
#

# TODO: mixup
# TODO: loudness jitter
# TODO: effects augmentation
