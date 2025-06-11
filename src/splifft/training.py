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
"""Number of training examples (audio chunks) processed in one iteration before the weight update.

- Larger batch sizes may lead to more stable gradients and faster convergence, but require more
  memory.
- Smaller batch sizes may result in a noisier gradient estimate
"""

GradientAccumulationSteps: TypeAlias = Annotated[int, Gt(0)]
"""Number of batches to process before performing a weight update.

A technique to simulate larger batch sizes without increasing memory consumption.
For example, a [batch size][splifft.training.TrainingBatchSize] of 4 with gradient accumulation of 8 
results in an effective batch size of 32. Gradients are computed for each of the 8 mini-batches,
and summed up before the optimizer updates the model weights. This may be useful for training
large models on consumer hardware with limited memory.
"""

OptimizerName: TypeAlias = str
"""Algorithm used to update the model weights to minimize the loss function."""

LearningRateSchedulerName: TypeAlias = str
"""Algorithm used to adjust the learning rate during training.

For example, `ReduceLROnPlateau` reduces the learning rate when a monitored metric (e.g. validation
SDR) stops improving for a certain number of [epochs][splifft.training.Epoch] (patience).
"""

# TODO: patience

# TODO: loss function
# TODO: gradient clipping

UseAutomaticMixedPrecision: TypeAlias = bool
"""Whether to use [automatic mixed precision (AMP)][torch.amp] during training."""

UseLoRA: TypeAlias = bool
"""Low-Rank Adaptation for efficient fine-tuning.

Instead of retraining all parameters of a large pre-trained model, LoRA freezes the original weights
and injects small, trainable low-rank matrices into the layers (typically attention layer's query
and value projections). This dramatically reduces the number of trainable parameters, making it
fast and efficient to adapt a model to a new dataset or task.
"""

#
# data augmentation (move to new module?)
#

# TODO: mixup
# TODO: loudness jitter
# TODO: effects augmentation
